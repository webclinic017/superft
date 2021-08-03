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
from freqtrade.ml.container import LightningContainer, list_difference
from freqtrade.ml.loader import load_df
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
        
        self.validate_predict(cont)
        
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
        
    def validate_predict(self, cont: LightningContainer):
        """ Validate model container predict function:
        - Type of df_predict must DataFrame
        - There must be a new columns in the df_predict DataFrame
        - df_predict length must same as the df_original
        - df_predict index must not changed
        """
        df_original: pd.DataFrame = load_df("BTC_USDT-5m.json", "5m").iloc[:5000]
        df_predict = df_original.copy()
        df_predict = cont.predict(df_predict)
        
        # Type of df_predict must DataFrame
        if not isinstance(df_predict, pd.DataFrame):
            raise TypeError(f"'df_predict' type is '{type(df_predict)}'. needed: DataFrame")
        
        # df_predict length must same as the df_original
        if len(df_predict) != len(df_original):
            raise ValueError(
                "Length of df_predict must same as the length of df_original."
                f"len(df_predict) = {len(df_predict)}, len(df_original) = {len(df_original)}"
            )
            
        # df_predict index must not changed
        if not df_original.index.equals(df_predict.index):
            raise ValueError(f"df_predict index must same as df_original index. ")
        
        # There must be a new columns in the df_predict DataFrame
        df_predict_cols = df_predict.columns.tolist()
        if df_original.columns.tolist() == df_predict_cols:
            raise ValueError(f"'df_predict' columns is same as df_original.")
        
        new_columns = list_difference(df_predict_cols, df_original.columns.tolist())
        print("PASSED! New columns from prediction")
        print(df_predict[new_columns].info())
        print(df_predict[new_columns].head())