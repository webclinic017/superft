from typing import List, Tuple, Any
from pathlib import Path
from wandb.wandb_run import Run

import attr
import pandas as pd
import logging
import numpy as np
    
from freqtrade.ml.loader import clean_ohlcv_dataframe
from freqtrade.ml.lightning import LightningModule
from freqtrade.nbtools.helper import free_mem

logger = logging.getLogger(__name__)


@attr.s(repr=False)
class LightningContainer:
    """ Interface to interact with LightningModule. Used when training, ft-backtesting, and ft-liverun.
        Why? This abstraction level will let us modify the interaction process without
        re-implementing in the core TradingLightningModule implementation.
        In other words, this improves reusability.
    """
    module: LightningModule = attr.ib()
    
    def configure(self):
        self.module.config = self.module.on_configure()
    
    def get_data_paths(self, cwd: Path, timeframe: str, exchange: str) -> List[Path]:
        """ List of Path to your per pair JSON data consisting of [timestamp, open, high, low, close, volume] columns"""
        return self.module.on_get_data_paths(cwd, timeframe, exchange)
    
    def get_dataset(self) -> Tuple[Any, Any, Any, Any]:
        """ Loads the whole DataFrame according to get_data_paths()
        """
        df_allpairs = self._load_df_allpairs()
        return self.final_processing(df_allpairs)
    
    def _load_df_allpairs(self) -> pd.DataFrame:
        """ Load helper for all pairs of DataFrame """
        df_list = [
            self._load_one(filepath)
            for filepath in self.get_data_paths(Path.cwd(), self.module.config.timeframe, self.module.config.exchange)
        ]

        df_allpairs = pd.concat(df_list)
        df_allpairs = df_allpairs.dropna()
        df_allpairs = to_fp32(df_allpairs)
        
        # Validate columns
        self.module.config.columns_x = [col for col in df_allpairs.columns if col not in self.module.config.columns_unused]
        
        if self.module.config.column_y not in df_allpairs.columns:
            raise Exception(f"Please add your training labels as '{self.module.config.column_y}' column.")

        free_mem(df_list)
        return df_allpairs
    
    def _load_one(self, path: Path) -> pd.DataFrame:
        """ Load helper for one DataFrame by path.
        """
        headers = ["date", "open", "high", "low", "close", "volume"]
        df_onepair: pd.DataFrame = pd.read_json(path)
        df_onepair.columns = headers
        
        df_onepair["date"] = pd.to_datetime(df_onepair["date"], unit='ms', utc=True, infer_datetime_format=True)
        df_onepair = df_onepair.astype(
            dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32','volume': 'float32'}
        )
        df_onepair = clean_ohlcv_dataframe(df_onepair, self.module.config.timeframe, fill_missing=True, drop_incomplete=True)
        
        df_onepair["pair"] = path.name.split("-")[0].replace("_", "/")
        df_onepair = df_onepair.reset_index(drop=True)
        
        trainval_start = pd.to_datetime(self.module.config.trainval_start, utc=True, infer_datetime_format=True)
        trainval_end = pd.to_datetime(self.module.config.trainval_end, utc=True, infer_datetime_format=True)
        
        df_onepair = df_onepair[((df_onepair.date >= trainval_start) &
                                 (df_onepair.date <= trainval_end))]
        
        df_onepair = self.add_features(df_onepair)
        df_onepair = self.add_labels(df_onepair)
        
        return df_onepair
    
    def add_features(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        """ Container wrapper for module.add_features()"""
        df_onepair = to_fp32(df_onepair)
        df_onepair = self.module.on_add_features(df_onepair)
        
        df_onepair = df_onepair.loc[df_onepair.first_valid_index():]
        df_onepair = df_onepair.replace([np.inf, -np.inf], np.nan)
        df_onepair = df_onepair.dropna()
        
        return df_onepair

    def add_labels(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        """ Container wrapper for module.add_labels() """
        df_onepair = to_fp32(df_onepair)
        df_onepair = self.module.on_add_labels(df_onepair)
        
        df_onepair = df_onepair.loc[df_onepair.first_valid_index():]
        df_onepair = df_onepair.replace([np.inf, -np.inf], np.nan)
        df_onepair = df_onepair.dropna()
        
        return df_onepair
    
    def final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        """ Container wrapper for module.final_processing() """
        df_allpairs = to_fp32(df_allpairs)
        X_train, X_val, y_train, y_val = self.module.on_final_processing(df_allpairs)
        free_mem(df_allpairs) 
        return X_train, X_val, y_train, y_val
    
    def define_model(self, run: Run, X_train, X_val, y_train, y_val):
        """ Container wrapper for module.define_model() """
        self.module.model = self.module.on_define_model(run, X_train, X_val, y_train, y_val)
    
    def start_training(self, run: Run, X_train, X_val, y_train, y_val):
        """ Container wrapper for module.start_training() """
        return self.module.on_start_training(run, X_train, X_val, y_train, y_val)
    
    def predict(self, df_onepair_original: pd.DataFrame) -> pd.DataFrame:
        """ Container wrapper for module.predict().
        Returns the original of DataFrame with prediction columns. 
        This inference will be used in strategy.py file that loaded from wandb.
        """
        df_preds = df_onepair_original.copy()
        df_preds = self.add_features(df_preds)
        
        # Only use X columns.
        df_preds = df_preds[self.module.config.columns_x]
        df_preds = self.module.on_predict(df_preds)
        
        try:
            # Drop X columns because freqtrade doesn't need it.
            df_preds = df_preds.drop(columns=self.module.config.columns_x)
        except KeyError:
            logger.info("Not dropping X columns in predict because it doesn't exist in predict columns")
        
        df_preds.columns = [str(f"ml_{it}") for it in df_preds.columns]
        logger.info(f"Returned new columns from df_preds: {list(df_preds.columns)}")
        
        # Return original freqtrade dataframe with prediction columns.
        df_onepair_original = concat_columns_last_index(df_onepair_original, df_preds)
        return df_onepair_original
    
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
    
    
def list_difference(a: list, b: list) -> Tuple[list, list]:
    """Returns (Cols in A only, Cols in B only)"""
    a_only = list(set(a) - set(b))
    b_only = list(set(b) - set(a))
    return a_only, b_only


def concat_columns_last_index(df_main: pd.DataFrame, df_to_merge: pd.DataFrame) -> pd.DataFrame:
    len_main = len(df_main)
    len_to_merge = len(df_to_merge)
    df_to_merge_copy = df_to_merge.copy()
    df_to_merge_copy.index = df_main.index[len_main - len_to_merge : len_main]
    return pd.concat([df_main, df_to_merge_copy], axis=1)