from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
from copy import deepcopy
from wandb.sdk.wandb_run import Run

import attr
import cloudpickle
import pandas as pd
import wandb
import json
import logging
import numpy as np

from freqtrade.ml.lightning import LightningModule
from freqtrade.ml.container import LightningContainer, list_difference
from freqtrade.ml.loader import load_df
from freqtrade.nbtools.helper import free_mem, get_readable_date

logger = logging.getLogger(__name__)


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
    
    def fit(self, module: LightningModule, wandb_run: Run, log_wandb: bool = True, skip: bool = False) -> LightningContainer:
        """ 
        Start Training Model. Returns LightningContainer with Trained Model. 
        """
        cont = LightningContainer(module)
        
        if not skip:
            self._mini_training(cont, wandb_run)
        
        X_train, X_val, y_train, y_val = cont.get_dataset()
        
        cont.define_model(wandb_run, X_train, X_val, y_train, y_val)
        
        try:
            self.validate_predict(cont, do_print=False)
        except Exception as e:
            logging.warning(f"ERROR on validating predict function: `{e}`")
            logging.warning(f"Please run `trainer.validate_predict()` after training finished!")
            logging.warning(f"Otherwise, future runtime errors may occur.")
        
        cont.start_training(wandb_run, X_train, X_val, y_train, y_val)
        
        for clean in [X_train, X_val, y_train, y_val]:
            free_mem(clean)
        
        if log_wandb:
            self._wandb_log(cont, wandb_run)
            
        return cont
    
    def _mini_training(self, container: LightningContainer, wandb_run: Run):
        """ 
        Train using smaller dataset to validate if the algorithm behaves correctly
        """
        logger.info("Validating model using mini training...")
        cont = deepcopy(container)
        
        def on_get_data_paths(cwd: Path, timeframe: str, exchange: str):
            path_data_exchange = cwd.parent / "mount" / "data" / exchange

            return [
                datapath
                for datapath in list(path_data_exchange.glob(f"*-{timeframe}.json"))
                if datapath.name.split("-")[0].replace("_", "/")
                in ["BTC/USDT"]
            ]
        
        cont.module.on_get_data_paths = on_get_data_paths

        if cont.module.config.num_training_epochs is not None:
            cont.module.config.num_training_epochs = 10
        
        X_train, X_val, y_train, y_val = cont.get_dataset()
        cont.define_model(wandb_run, X_train, X_val, y_train, y_val)
        cont.start_training(wandb_run, X_train, X_val, y_train, y_val)
        
        for clean in [X_train, X_val, y_train, y_val]:
            free_mem(clean)
        
        self.validate_predict(cont, do_print=False)
        logger.info("Validate model OK!")
    
    def _wandb_log(self, cont: LightningContainer, run: Run):
        """ 
        - Log string and non object attrs as JSON
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
        
    def validate_predict(self, cont: LightningContainer, do_print: bool = True):
        """ 
        Validate model container predict function:
        - Type of df_predict must DataFrame
        - There must be a new columns in the df_predict DataFrame
        - df_predict length must same as the df_original
        - df_predict index must not changed
        """
        tf = cont.config.timeframe
        
        df_original: pd.DataFrame = load_df(f"user_data/dataset/BTC_USDT-{tf}.json", tf).iloc[-10000:]
        df_predict = df_original.copy()
        df_predict = cont.predict(df_predict)
        df_vanilla_preds = self.vanilla_predict(cont, df_original)
        
        # Type of df_predict must DataFrame
        if not isinstance(df_predict, pd.DataFrame):
            raise TypeError(f"'df_predict' type is '{type(df_predict)}'. needed: pandas.DataFrame")
        
        if do_print:
            print(f"\nDataset: Binance BTC/USDT {tf} iloc[-10000:] (Freqtrade Regularized)")
            print("\n\nDF WITH PREDICTIONS INFO\n----------")
            print(df_predict.info())
            print("\n\nDF Original\n----------")
            print(df_original)
            print("\n\nDF + Preds\n----------")
            print(df_predict)
            print("\n\nVanilla Prediction [MAKE SURE SAME WITH PREDICTION DF!]\n----------")
            print(df_vanilla_preds)
            print("\nLEN VANILLA PREDS: %s" % len(df_vanilla_preds))
            print("LEN DF + PREDS NON NAN: %s" % len(df_predict.dropna()))
        
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
        
        cols_new_predict, cols_new_original = list_difference(df_predict_cols, df_original.columns.tolist())
        if len(cols_new_predict) <= 0:
            raise ValueError(f"No new columns in 'df_predict'")
        
        # The original data must not changed
        original_columns = ["date", "open", "high", "low", "close", "volume"]
        changed_cols = [col for col in original_columns if not df_predict[col].equals(df_original[col])]
        if len(changed_cols) > 0:
            raise ValueError(f"Changed original columns: {changed_cols}")
        
        # There are no columns full of NaN or InF
        invalids = [it for it in list(df_predict.columns) if df_predict[it].isnull().all()]
        if len(invalids) > 0:
            raise ValueError(f"Invalid (Full of NaN / InF Columns): {invalids}")
        
        # Length of vanilla predictions must same
        if len(df_vanilla_preds) != len(df_predict.dropna()):
            raise ValueError("Length of Vanilla Predict and Non-NAN aren't same.")
        
        # It passed!
        print("\nPASSED: The model passed the validation test!")        
        return df_predict
        
    def vanilla_predict(self, container: LightningContainer, df_original_: pd.DataFrame) -> pd.DataFrame:
        """ 
        Predict without going through the container's predict processing
        """
        df_original = container.add_features(df_original_.copy())
        df_original = df_original[container.module.config.columns_x]
        return container.module.on_predict(df_original)