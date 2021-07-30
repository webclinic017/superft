from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import gc
import attr

from freqtrade.ml.lightning import LightningModule
from freqtrade.nbtools.helper import free_mem
from freqtrade.nbtools.pairs import PAIRS_HIGHCAP_NONSTABLE


@attr.s(repr=False)  # Repr=False to support pickle
class RandomForest(LightningModule):
    """ Example RandomForest Module """
    
    def __attrs_pre_init__(self):
        """ Define your custom attributes here """
        self.task_type = "classification"
        self.num_future_candles = 2
        self.num_classification_classes = 3
    
    def on_get_data_paths(self, timeframe: str, exchange: str) -> List[Path]:
        """ List of Path object to your per pair JSON data consisting of [date, open, high, low, close, volume] columns"""
        path_data_exchange = Path.cwd().parent / "mount" / "data" / exchange

        return [
            datapath
            for datapath in list(path_data_exchange.glob(f"*-{timeframe}.json"))
            if datapath.name.split("-")[0].replace("_", "/")
            in PAIRS_HIGHCAP_NONSTABLE[:5]
        ]
    
    def on_add_features(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        import talib.abstract as ta

        # Start add features
        spaces = [3, 5, 9, 15, 25, 50, 100, 200]
        for i in spaces:
            df_per_pair[f"ml_smadiff_{i}"] = (df_per_pair['close'].rolling(i).mean() - df_per_pair['close'])
            df_per_pair[f"ml_maxdiff_{i}"] = (df_per_pair['close'].rolling(i).max() - df_per_pair['close'])
            df_per_pair[f"ml_mindiff_{i}"] = (df_per_pair['close'].rolling(i).min() - df_per_pair['close'])
            df_per_pair[f"ml_std_{i}"] = df_per_pair['close'].rolling(i).std()
            df_per_pair[f"ml_ma_{i}"] = df_per_pair['close'].pct_change(i).rolling(i).mean()
            df_per_pair[f"ml_rsi_{i}"] = ta.RSI(df_per_pair["close"], timeperiod=i)

        df_per_pair['ml_bop'] = ta.BOP(df_per_pair['open'], df_per_pair['high'], df_per_pair['low'], df_per_pair['close'])
        df_per_pair["ml_volume_pctchange"] = df_per_pair['volume'].pct_change()
        df_per_pair['ml_z_score_120'] = ((df_per_pair["ml_ma_15"] - df_per_pair["ml_ma_15"].rolling(21).mean() + 1e-9) 
                             / (df_per_pair["ml_ma_15"].rolling(21).std() + 1e-9))

        # Performance improvements
        df_per_pair = df_per_pair.astype({col: "float32" for col in df_per_pair.columns if "float" in str(df_per_pair[col].dtype)})
        return df_per_pair

    def on_add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        # Create labels for classification task
        future_price = df_per_pair['close'].shift(-self.num_future_candles)
        ml_label = (future_price - df_per_pair['close']) / df_per_pair['close']
        df_per_pair[self.column_y] = pd.qcut(ml_label, self.num_classification_classes, labels=False)
        return df_per_pair

    def on_final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        df_allpairs = self._balance_class_dataset(df_allpairs)
        X = df_allpairs[self.columns_x]
        y = df_allpairs[self.column_y]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        return X_train, X_val, y_train, y_val

    def _balance_class_dataset(self, df_allpairs: pd.DataFrame) -> pd.DataFrame:
        """Balance num of datas in every class"""
        lengths_every_class = list(df_allpairs.groupby(by=["ml_label"]).count()["date"])
        df_allpairs_copy = pd.DataFrame()

        for classname in df_allpairs["ml_label"].unique():
            minimum_of_all = min(lengths_every_class)
            df_allpairs_copy = df_allpairs_copy.append(df_allpairs.loc[df_allpairs["ml_label"] == classname, :].iloc[:minimum_of_all])

        # Performance improvements
        df_allpairs_copy = df_allpairs_copy.astype(
            {col: "float32" for col in df_allpairs_copy.columns if "float" in str(df_allpairs_copy[col].dtype)}
        )
        free_mem(df_allpairs)
        
        return df_allpairs_copy

    def on_define_model(self, run: Run, X_train, X_val, y_train, y_val) -> Any:
        return RandomForestClassifier(max_depth=2, random_state=0)
    
    def on_start_training(self, run: Run, X_train, X_val, y_train, y_val):
        self.model: RandomForestClassifier
        self.model.fit(X_train, y_train)

    def on_predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """ Returns the Series of prediction. This inference will be
            used in strategy.py file loaded from wandb.
        
        1. Convert to FP32 if it isn't.
        2. Call add_features() if no existing features.
        3. 
           
        NOTE: This method must NOT change the existing input dataframe.
        NOTE: If predict need to import external libraries, import it here.
        """
        raise NotImplementedError()

    def on_training_step(self, run: Run, data: dict):
        """ Called every training step to log process. """
        raise NotImplementedError()
    
    
if __name__ == "__main__":
    """ Example Training Usage """
    
    import wandb
    from freqtrade.ml.trainer import TradingTrainer
    from freqtrade.ml import dataloader
    
    name = "15n30-randomforest"
    
    with wandb.init(project=name) as run:
        rf_module = RandomForest(
            name = name,
            timeframe = "15m",
            exchange = "binance",
        )
        trainer = TradingTrainer()
        container = trainer.fit(rf_module, run, log_wandb=False)
    
    df = dataloader.load_df(Path.cwd().parent / "mount" / "data" / "binance" / "BTC_USDT-15m.json")
    # df = container.module.model.predict(df)
    print(df.head())
    print(df.info())
    