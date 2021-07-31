from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import gc
import attr

from freqtrade.ml.lightning import LightningModule, LightningConfig
from freqtrade.nbtools.helper import free_mem
from freqtrade.nbtools.pairs import PAIRS_HIGHCAP_NONSTABLE


attr.s(repr=False)
class RandomForest(LightningModule):
    """ Template for LightningModule """
        
    def on_configure(self) -> LightningConfig:
        # This datetime can be replaced with datetime.now()
        now = datetime(2021, 7, 26)
        
        # Lighting Configuration
        config = LightningConfig(
            
            # Basic info
            name        = "5n20-randomforest",
            timeframe   = "5m",
            exchange    = "binance",
            
            # Train and validation datetime
            trainval_start  = now - timedelta(days=120),
            trainval_end    = now - timedelta(days=60),
            
            # Backtest Optimization datetime
            opt_start = now - timedelta(days=59),
            opt_end   = now - timedelta(days=30),
            
            # Unbiased Backtest datetime
            test_start = now - timedelta(days=29),
            test_end   = now,
        )
        
        # Optional custom config attributes
        config.add_custom("num_epochs", 1000)
        config.add_custom("num_future_candles", 4)
        config.add_custom("num_classification_classes", 3)
        
        return config
        
    def on_get_data_paths(self, cwd: Path, timeframe: str, exchange: str) -> List[Path]:
        path_data_exchange = cwd.parent / "mount" / "data" / exchange

        return [
            datapath
            for datapath in list(path_data_exchange.glob(f"*-{timeframe}.json"))
            if datapath.name.split("-")[0].replace("_", "/")
            in PAIRS_HIGHCAP_NONSTABLE[:5]
        ]
    
    def on_add_features(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        import talib.abstract as ta

        # Start add features
        spaces = [3, 5, 9, 15, 25, 50, 100, 200]
        for i in spaces:
            df_onepair[f"ml_smadiff_{i}"] = (df_onepair['close'].rolling(i).mean() - df_onepair['close'])
            df_onepair[f"ml_maxdiff_{i}"] = (df_onepair['close'].rolling(i).max() - df_onepair['close'])
            df_onepair[f"ml_mindiff_{i}"] = (df_onepair['close'].rolling(i).min() - df_onepair['close'])
            df_onepair[f"ml_std_{i}"] = df_onepair['close'].rolling(i).std()
            df_onepair[f"ml_ma_{i}"] = df_onepair['close'].pct_change(i).rolling(i).mean()
            df_onepair[f"ml_rsi_{i}"] = ta.RSI(df_onepair["close"], timeperiod=i)

        df_onepair['ml_bop'] = ta.BOP(df_onepair['open'], df_onepair['high'], df_onepair['low'], df_onepair['close'])
        df_onepair["ml_volume_pctchange"] = df_onepair['volume'].pct_change()
        df_onepair['ml_z_score_120'] = ((df_onepair["ml_ma_15"] - df_onepair["ml_ma_15"].rolling(21).mean() + 1e-9) 
                             / (df_onepair["ml_ma_15"].rolling(21).std() + 1e-9))

        return df_onepair
    
    def on_add_labels(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        # Create labels for classification task
        future_price = df_onepair['close'].shift(-self.config.num_future_candles)
        ml_label = (future_price - df_onepair['close']) / df_onepair['close']
        df_onepair[self.config.column_y] = pd.qcut(ml_label, self.config.num_classification_classes, labels=False)
        return df_onepair
    
    def on_final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        df_allpairs = self._balance_class_dataset(df_allpairs)
        X = df_allpairs[self.config.columns_x]
        y = df_allpairs[self.config.column_y]
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
        print("Start Training...")
        self.model: RandomForestClassifier
        self.model.fit(X_train, y_train)
    
    def on_predict(self, df_input_onepair: pd.DataFrame) -> pd.DataFrame:
        preds = self.model.predict_proba(df_input_onepair)
        return pd.DataFrame(preds)
    
    def on_training_step(self, run: Run, data: dict):
        raise NotImplementedError()
    
    
if __name__ == "__main__":
    """ Example Training Usage """
    
    import wandb
    from freqtrade.ml.trainer import TradingTrainer
    from freqtrade.ml import loader
    
    name = "15n30-randomforest"
    
    with wandb.init(project=name) as run:
        rf_module = RandomForest()
        trainer = TradingTrainer()
        container = trainer.fit(rf_module, run, log_wandb=False)
    
    df = loader.load_df(Path.cwd().parent / "mount" / "data" / "binance" / "BTC_USDT-15m.json")
    # df = container.module.model.predict(df)
    print(df.head())
    print(df.info())
    