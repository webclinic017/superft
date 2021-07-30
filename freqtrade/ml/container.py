from typing import List, Tuple, Any
from pathlib import Path
from wandb.wandb_run import Run

import attr
import pandas as pd

from freqtrade.ml.lightning import LightningModule
from freqtrade.nbtools.helper import free_mem


@attr.s(repr=False)
class LightningContainer:
    """ Interface to interact with LightningModule. Used when training, ft-backtesting, and ft-liverun.
        Why? This abstraction level will let us modify the interaction process without
        re-implementing in the core TradingLightningModule implementation.
        In other words, this improves reusability.
    """
    module: LightningModule = attr.ib()
    
    def get_data_paths(self, timeframe: str, exchange: str) -> List[Path]:
        """ List of Path to your per pair JSON data consisting of [timestamp, open, high, low, close, volume] columns"""
        return self.module.on_get_data_paths(timeframe, exchange)
    
    def get_dataset(self) -> pd.DataFrame:
        """ Loads the whole DataFrame according to get_data_paths()
        """
        df_list = [
            self._load_one(filepath)
            for filepath in self.get_data_paths(
                self.module.timeframe, self.module.exchange
            )
        ]

        df_allpairs = pd.concat(df_list)
        df_allpairs = df_allpairs.dropna()
        df_allpairs = to_fp32(df_allpairs)

        free_mem(df_list)
        
        print(f"Loaded Pairs: {[it for it in df_allpairs['pair'].unique()]}")
        
        self.module.columns_x = [col for col in df_allpairs.columns if col not in self.module.columns_unused]
        
        return df_allpairs
    
    def _load_one(self, path: Path) -> pd.DataFrame:
        """ Load helper for one DataFrame by path.
        """
        headers = ["date", "open", "high", "low", "close", "volume"]
        df_onepair = pd.read_json(path)
        df_onepair.columns = headers
        df_onepair["pair"] = path.name.split("-")[0].replace("_", "/")
        df_onepair["date"] = pd.to_datetime(df_onepair["date"], unit='ms')
        df_onepair = df_onepair.reset_index(drop=True)
        df_onepair = df_onepair[((df_onepair.date >= self.module.trainval_start) &
                                 (df_onepair.date <= self.module.trainval_end))]
        
        df_onepair = self.add_features(df_onepair)
        df_onepair = self.add_labels(df_onepair)
        
        if "ml_label" not in df_onepair.columns:
            raise Exception("Please add your training labels as 'ml_label' column.")
        
        return df_onepair
    
    def add_features(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Container wrapper for module.add_features()
        """
        df_per_pair = to_fp32(df_per_pair)
        return self.module.on_add_features(df_per_pair)

    def add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        """ Container wrapper for module.add_labels()
        """
        df_per_pair = to_fp32(df_per_pair)
        return self.module.on_add_labels(df_per_pair)
    
    def final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        """ Container wrapper for module.final_processing()
        """
        df_allpairs = to_fp32(df_allpairs)
        X_train, X_val, y_train, y_val = self.module.on_final_processing(df_allpairs)
        free_mem(df_allpairs) 
        return X_train, X_val, y_train, y_val
    
    def define_model(self, run: Run, X_train, X_val, y_train, y_val):
        """ Container wrapper for module.define_model()
        """
        self.module.model = self.module.on_define_model(run, X_train, X_val, y_train, y_val)
    
    def start_training(self, run: Run, X_train, X_val, y_train, y_val):
        """ Container wrapper for module.start_training()
        """
        return self.module.on_start_training(run, X_train, X_val, y_train, y_val)
    
    def predict(self, df_perpair: pd.DataFrame) -> pd.DataFrame:
        """ This is a container wrapper for module.predict().
        Returns the original of DataFrame with prediction columns. 
        This inference will be used in strategy.py file loaded from wandb.
        """
        df_perpair = self.add_features(df_perpair)
        df_with_prediction = self.module.on_predict(df_perpair)
        df_with_prediction = df_with_prediction.drop(columns=self.module.columns_x)
        return df_with_prediction
    
    def training_step(self, run: Run, data: dict):
        """ Container wrapper for module.training_step()
        """
        return self.module.on_training_step(run, data)
 
    
def to_fp32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            col: "float32"
            for col in df.columns
            if str(df[col].dtype) in ["float64", "float16"]
        }
    )