from pathlib import Path
from typing import List, Callable

import pandas as pd
import gc

from freqtrade.ml.lightning import TradingLightningModule


class TrainingConfiguration:
    """
    Where the data flows:
    1. get_dataset_paths() -> List[Path]
    2. for every path to data json, load data to dataframe. one dataframe is one pair.
    3. for every pair dataframe:
        3.1 add_features()
        3.2 add_labels()
    4. the full dataframe of all pairs is returned. call post_processing()
    5. if you are on notebook, you can EDA that returned dataframe.
    6. training_pipeline()
    7. every training step: training_step()
    """
    
    @staticmethod
    def get_dataset_paths(path_data: Path, exchange: Path, timeframe: str) -> List[Path]:
        paths = []
        for p in (path_data / exchange).glob(f"*-{timeframe}.json"):
            if p.name.split("-")[0].replace("_", "/") not in PAIRS_HIGH_CAP:
                continue
            paths.append(p)
        print(f"Pairs: {[it.name for it in paths]}")
        return paths

    """ 
    Configuration
    """
    NAME = "catboost-5n20"
    TIMEFRAME = "5m"
    EXCHANGE = "binance"
    DATASET = get_dataset_paths(Path.cwd(), EXCHANGE, TIMEFRAME)
    COLUMNS_NOT_USE = ["date", "open", "high", "low", "close", "volume", "pair", "ml_target", "ml_target_real"]
    
    NUM_FUTURE_CANDLES = 4
    
    TASK_TYPE = "classification"
    TASK_CLASSIFICATION_NUM_CLASSES = 3

    """
    Date <= Last 2 months: Train and Val (Shuffled val data)
    Date <= Last month   : Strategy optimization
    Last Month to Present: Unbiased backtesting
    """
    TRAINVAL_START, TRAINVAL_END = "2021-03-01", "2021-05-26"
    OPT_START, OPT_END           = "2021-05-26", "2021-06-26"
    TEST_START, TEST_END         = "2021-06-26", "2021-07-26"
    
    def add_features(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Define the features of dataset. Called after load data of this pair. 
            NOTE: Import libraries that features are dependent on. E.G: import talib.abstract as ta
        """
        import talib.abstract as ta

        # Memory improvements
        import gc
        df_copy = df_per_pair.copy()
        del df_per_pair
        gc.collect()

        # Start add features:
        spaces = [3, 5, 9, 15, 25, 50, 100, 200, 500]
        for i in spaces:
            df_copy[f"ml_smadiff_{i}"] = (df_copy['close'].rolling(i).mean() - df_copy['close'])
            df_copy[f"ml_maxdiff_{i}"] = (df_copy['close'].rolling(i).max() - df_copy['close'])
            df_copy[f"ml_mindiff_{i}"] = (df_copy['close'].rolling(i).min() - df_copy['close'])
            df_copy[f"ml_std_{i}"] = df_copy['close'].rolling(i).std()
            df_copy[f"ml_ma_{i}"] = df_copy['close'].pct_change(i).rolling(i).mean()
            df_copy[f"ml_rsi_{i}"] = ta.RSI(df_copy["close"], timeperiod=i)

        df_copy['ml_bop'] = ta.BOP(df_copy['open'], df_copy['high'], df_copy['low'], df_copy['close'])
        df_copy["ml_volume_pctchange"] = df_copy['volume'].pct_change()
        df_copy['ml_z_score_120'] = ((df_copy["ml_ma_15"] - df_copy["ml_ma_15"].rolling(21).mean() + 1e-9) 
                             / (df_copy["ml_ma_15"].rolling(21).std() + 1e-9))

        # Performance improvements
        df_copy = df_copy.astype({col: "float32" for col in df_copy.columns if "float" in str(df_copy[col].dtype)})
        return df_copy

    def add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Define the label (target prediction) of the data. Called after add_features() """

        # Memory improvements
        df_copy = df_per_pair.copy()
        del df_per_pair
        gc.collect()

        future_price = df_copy['close'].shift(-self.NUM_FUTURE_CANDLES)

        # ML Target Regularized
        df_copy["ml_target"] = (future_price - df_copy['close']) / df_copy['close']

        # ML Target for EDA before Regularized
        df_copy["ml_target_real"] = (future_price - df_copy['close']) / df_copy['close']

        # Split the targets into several classes (Classificatioi)
        df_copy['ml_target'] = pd.qcut(df_copy["ml_target"], self.TASK_CLASSIFICATION_NUM_CLASSES, labels=False)
        
        return df_copy

    def post_processing(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        """ Extra processing of the combined data (all pairs). Called after add_labels() """
        # Memory improvements
        df_copy = df_combined.copy()
        del df_combined
        gc.collect()
        
        # Balance num of datas in every class
        lengths_every_class = list(df_copy.groupby(by=["ml_target"]).count()["date"])
        df_copy_2 = pd.DataFrame()

        for classname in df_copy["ml_target"].unique():
            minimum_of_all = min(lengths_every_class)
            df_copy_2 = df_copy_2.append(df_copy.loc[df_copy["ml_target"] == classname, :].iloc[:minimum_of_all])

        # Performance improvements
        df_copy_2 = df_copy_2.astype({col: "float32" for col in df_copy.columns if "float" in str(df_copy[col].dtype)})
        del df_copy
        gc.collect()
        
        return df_copy_2


def load_dataset(module: TradingLightningModule) -> pd.DataFrame:
    df_list = []

    for filepath in module.DATA_PATHS:
        load_one(module, df_list, filepath)

    df = pd.concat(df_list)
    df = df.dropna()
    
    free_mem(df_list)
    print("-LOAD DATASET FINISHED-")
    
    return df


# pyright: reportGeneralTypeIssues=false
def load_one(module: TradingLightningModule, df_list: list, path: Path) -> pd.DataFrame:
    print(f"Loading: {path.name}")
    headers = ["date", "open", "high", "low", "close", "volume"]

    d1 = pd.read_json(path)
    d1.columns = headers
    d1["pair"] = path.name.split("-")[0].replace("_", "/")
    d1["date"] = pd.to_datetime(d1["date"], unit='ms')
    d1 = d1.reset_index(drop=True)
    d1 = d1[((d1.date >= module.TRAINVAL_START) &
            (d1.date <= module.TRAINVAL_END))]
    
    # Convert to FP32
    d1 = d1.astype({col: "float32" for col in ["open", "high", "low", "close", "volume"]})
    d1 = module.add_features(d1)
    d1 = module.add_labels(d1)
    
    df_list.append(d1)
    free_mem(d1)
    

def free_mem(var):
    del var
    gc.collect()