from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
from wandb.sdk.wandb_run import Run

import attr
import cloudpickle
import pandas as pd
import wandb
import json

from freqtrade.ml.lightning import TradingLightningModule
from freqtrade.ml import dataloader
from freqtrade.nbtools.helper import free_mem, get_readable_date


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
    6. define_model()
    7. start_training()
    8. every training step: training_step()
    9. inference: predict()
    """
    
    def fit(self, module: TradingLightningModule, wandb_run: Run) -> Any:
        """ Start Training Model. Returns Trained Model """
        
        df_full = dataloader.load_dataset(module)
        print(f"Loaded Pairs: {[it for it in df_full['pair'].unique()]}")
        
        if "ml_label" not in df_full.columns:
            raise Exception("Please add your training labels as 'ml_label' column.")
        
        module.columns_x = [col for col in df_full.columns if col not in module.columns_unused]
        
        X_train, X_val, y_train, y_val = module.final_processing(df_full)
        free_mem(df_full)
        
        module.model = module.define_model(wandb_run, X_train, X_val, y_train, y_val)
        module.start_training(wandb_run, X_train, X_val, y_train, y_val)
        
        for clean in [X_train, X_val, y_train, y_val]:
            free_mem(clean)
        
        self._log_module(module, wandb_run)
        
        return module.model
    
    # pyright: reportGeneralTypeIssues=false
    def _log_module(self, module: TradingLightningModule, run: Run):
        """ - Log string and non object attrs as JSON
            - Log whole module object
        """
        foldername = f"lightning_{module.name}_{get_readable_date()}"
        
        path_to_folder = Path.cwd() / ".temp" / foldername
        path_to_folder.mkdir()
        
        with (path_to_folder / "module.json").open("w") as fs:
            json.dump(attr.asdict(module), fs, default=str)
        with (path_to_folder / "module.pkl").open("wb") as fs:
            cloudpickle.dump(module, fs)
            
        artifact = wandb.Artifact(module.name, type="model_files")
        artifact.add_dir(str(path_to_folder))
        run.log_artifact(artifact)
        