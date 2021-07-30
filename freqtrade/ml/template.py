from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run

import attr
import pandas as pd
import gc

from freqtrade.ml.lightning import LightningModule
from freqtrade.nbtools.helper import free_mem
from freqtrade.nbtools.pairs import PAIRS_HIGHCAP_NONSTABLE


@attr.s(repr=False)
class LightningModuleTemplate(LightningModule):
    
    def __attrs_pre_init__(self):
        """ Define your custom attributes here """
        self.task_type = "classification"
        self.num_future_candles = 2
        self.num_classification_classes = 3
        
    def on_get_data_paths(self, timeframe: str, exchange: str) -> List[Path]:
        raise NotImplementedError()
    
    def on_add_features(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def on_add_labels(self, df_per_pair: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def on_final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        raise NotImplementedError()
    
    def on_define_model(self, run: Run, X_train, X_val, y_train, y_val) -> Any:
        raise NotImplementedError()
    
    def on_start_training(self, run: Run, X_train, X_val, y_train, y_val):
        raise NotImplementedError()
    
    def on_predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def on_training_step(self, run: Run, data: dict):
        raise NotImplementedError()