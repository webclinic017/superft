from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
from wandb.sdk.wandb_run import Run

import attr
import cloudpickle
import pandas as pd
import wandb
import json

from freqtrade.ml.lightning import LightningModule
from freqtrade.ml.container import LightningContainer
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
    
    def fit(self, module: LightningModule, wandb_run: Run, log_wandb: bool = True) -> LightningContainer:
        """ Start Training Model. Returns LightningContainer with Trained Model. 
        """
        cont = LightningContainer(module)
        
        X_train, X_val, y_train, y_val = cont.get_dataset()
        
        cont.define_model(wandb_run, X_train, X_val, y_train, y_val)
        cont.start_training(wandb_run, X_train, X_val, y_train, y_val)
        
        for clean in [X_train, X_val, y_train, y_val]:
            free_mem(clean)
        
        if log_wandb:
            self._wandb_log(cont, wandb_run)
        
        return cont
    
    def _wandb_log(self, cont: LightningContainer, run: Run):
        """ - Log string and non object attrs as JSON
            - Log whole module object
        """
        foldername = f"lightning_{cont.module.config.name}_{get_readable_date()}"
        
        path_to_folder = Path.cwd() / ".temp" / foldername
        path_to_folder.mkdir()
        
        with (path_to_folder / "configuration.json").open("w") as fs:
            json.dump(attr.asdict(cont.module.config), fs, default=str, indent=4)
        
        with (path_to_folder / "container.pkl").open("wb") as fs:
            cloudpickle.dump(cont, fs)
            
        artifact = wandb.Artifact(cont.module.config.name, type="lightning_files")
        artifact.add_dir(str(path_to_folder))
        run.log_artifact(artifact) # pyright: reportGeneralTypeIssues=false
        