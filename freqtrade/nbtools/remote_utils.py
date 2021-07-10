from typing import *
from pandas import DataFrame
from pathlib import Path
import wandb
import pandas as pd
import os
import dill

from freqtrade.nbtools import constants

"""
The final version of remote_utils.py should uploaded to independent github repo.
it is:
- constants.py
- remote_utils.py
"""


def preset_log(preset_directory: str, project: str, preset_name: str):
    with wandb.init(project=project, job_type="load-data") as run:
        artifact = wandb.Artifact(preset_name, type="preset")
        artifact.add_dir(f"./{preset_directory}", name=preset_name)
        run.log_artifact(artifact)


def table_retrieve(project: str, artifact_name: str, table_key: str) -> pd.DataFrame:
    with wandb.init(project=project) as run:
        my_table = run.use_artifact(f"{artifact_name}:latest").get(f"{table_key}")
        return pd.DataFrame(my_table.data, columns=my_table.columns)
    
    
def table_add_row(row_dict: dict, project: str, artifact_name: str, table_key: str):
    
    with wandb.init(project=project) as run:
        my_table = run.use_artifact(f"{artifact_name}:latest").get(f"{table_key}")
        
        my_table_cols = list(my_table.columns)
        input_cols = list(row_dict.keys())
        my_table_cols.sort()
        input_cols.sort()

        assert my_table_cols == input_cols

        my_table.add_data(*[row_dict[col] for col in my_table.columns])
        my_table = pd.DataFrame(my_table.data, columns=my_table.columns)
        my_table = wandb.Table(dataframe=my_table)
    
        table_artifact = wandb.Artifact(artifact_name, type="table")
        table_artifact.add(my_table, table_key)
        run.log_artifact(table_artifact)


def table_update(new_df: pd.DataFrame, project: str, artifact_name: str, table_key: str):
    
    with wandb.init(project=project) as run:
        try:
            my_table = run.use_artifact(f"{artifact_name}:latest").get(f"{table_key}")
            my_table = pd.DataFrame(my_table.data, columns=my_table.columns)
            assert list(my_table.columns) == list(new_df.columns)
        except Exception as e:
            print(e)
            print("Creating new table...")
        
        my_table = wandb.Table(dataframe=new_df)
        table_artifact = wandb.Artifact(artifact_name, type="table")
        table_artifact.add(my_table, table_key)
        run.log_artifact(table_artifact)


def cloud_load_preset(preset_name: str) -> Any:
    """ Returns {Preset} by downloading preset folder from the cloud according to preset name, then load it.
        The goal of Saving presets are for:
        - Reproducibility
        - Easy integration to live / dry run (Download preset artifact -> insert big file -> Reroute big file in strategy.py)
    
    Why load presets are hard to implement in Backtesting:
    1. By loading the preset, you loaded previously "Backtested" preset that you know the results by just looking
       at the metadata. What are you looking for? Plot profits?
       - Well, you may want to retest the same preset with different settings (maybe you want to backtest with present data).
         > Backtest Settings: Timerange, Pairlists, Timeframe, Max Open Trades, Stake Amount, Starting Balance
         > What to Backtest: Strategy code
         > When they all combined: Preset
    2. TODO: Resolve loading big files. It should only `model = load_big_file("my_big_file.pkl")` (vendor.wandb_utils)
             So we will need:
             - Code to save big file after training
             - Code to download and cache big file
    """
    pass


def cloud_get_presets_df(from_run_history: bool = False) -> DataFrame:
    """ Returns {DataFrame} list of presets table along with metadata.
        Goals:
        - Obvious, saves time by looking the numbers you want when deciding which strategy to dry / live run.
    """
    df = table_retrieve(constants.PROJECT_NAME, constants.ARTIFACT_TABLE_METADATA, constants.TABLEKEY_METADATA)
    if from_run_history:
        df = df.loc[df["preset_name"].str.contains("__run-")]
    return df


def add_single_asset(path, project, asset_name):
    with wandb.init(project=project) as run:
        artifact = wandb.Artifact(asset_name, type="single")
        artifact.add_file(path, name=asset_name)
        run.log_artifact(artifact)


def load_pickle_asset(project, asset_name):
    with wandb.init(project=project) as run:
        artifact = run.use_artifact(f'{asset_name}:latest')
        path = Path.cwd() / artifact.download()
        filepath = path / os.listdir(path)[0]
        with filepath.open("rb") as f:
            return dill.load(f)