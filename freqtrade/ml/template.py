from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run
from datetime import datetime, timedelta

import attr
import pandas as pd
import gc


from freqtrade.ml.lightning import LightningModule, LightningConfig
from freqtrade.ml.trainer import TradingTrainer

from freqtrade.nbtools.helper import free_mem
from freqtrade.nbtools.pairs import PAIRS_HIGHCAP_NONSTABLE


@attr.s(repr=False)
class LightningModuleTemplate(LightningModule):
    """ Template for LightningModule """
        
    def on_configure(self) -> LightningConfig:
        # This datetime can be replaced with datetime.now()
        now = datetime(2021, 7, 26)
        # Lighting Configuration
        config = LightningConfig(
            name        = "YOUR_MODEL_NAME",
            timeframe   = "15m",
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
        config.add_custom("num_future_candles", 4)
        config.add_custom("classification_classes", 3)
        
        return config
        
    def on_get_data_paths(self, cwd: Path, timeframe: str, exchange: str) -> List[Path]:
        raise NotImplementedError()
    
    def on_add_features(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def on_add_labels(self, df_onepair: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def on_final_processing(self, df_allpairs: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        raise NotImplementedError()
    
    def on_define_model(self, run: Run, X_train, X_val, y_train, y_val) -> Any:
        raise NotImplementedError()
    
    def on_start_training(self, run: Run, X_train, X_val, y_train, y_val):
        raise NotImplementedError()
    
    def on_predict(self, df_input_onepair: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def on_training_step(self, run: Run, data: dict):
        raise NotImplementedError()