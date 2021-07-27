from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from wandb.wandb_run import Run

import attr
import pandas as pd

from freqtrade.ml.lightning import TradingLightningModule
from freqtrade.ml import dataloader



@attr.s
class TradingTrainer:
    """
    Trading ML "Lightning" Trainer:
    1. get_dataset_paths() -> List[Path]
    2. for every path to data json, load data to dataframe. one dataframe is one pair.
    3. for every pair dataframe:
        3.1 add_features()
        3.2 add_labels()
    4. the full dataframe of all pairs is returned. call post_processing()
    5. if you are on notebook, you can EDA that returned dataframe.
    6. start_training()
    7. every training step: training_step()
    """
    
    def fit(self, module: TradingLightningModule, wandb_run: Run):
        data_paths = module.get_data_paths()
        
        df_full = dataloader.load_dataset(module)
        df_train, df_val = module.post_processing(df_full)
        dataloader.free_mem(df_full)
        
        module_attrs_before = attr.asdict(module)
        # TODO: Save module_attrs_before to pickle
        
        module.MODEL = module.define_model(wandb_run, df_train, df_val)
        module.start_training(wandb_run, df_train, df_val)
        
        module_attrs_after = attr.asdict(module)
        # TODO: Save module_attrs_before to pickle
        
        self.module = module