# mypy: ignore-errors
import os
import time
from pathlib import Path
from typing import Any, Union

import cloudpickle
import dill
from numpy import isin
import pandas as pd
from pandas import DataFrame
from functools import cache

import wandb
import logging

from freqtrade.nbtools import constants
from freqtrade.nbtools.helper import log_execute_time
from freqtrade.ml.container import LightningContainer


os.environ["WANDB_SILENT"] = "true"
logger = logging.getLogger(__name__)


"""
The final version of remote_utils.py should uploaded to independent github repo.
it is:
- constants.py
- remote_utils.py
"""


class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args, **kwargs):
        if args not in self.memo:
            self.memo[args] = self.fn(*args, **kwargs)
        return self.memo[args]


def preset_log(run, preset_directory: str, preset_name: str):
    artifact = wandb.Artifact(preset_name, type="preset")
    artifact.add_dir(f"./{preset_directory}", name=preset_name)
    run.log_artifact(artifact)


def table_retrieve(project: str, artifact_name: str, table_key: str) -> pd.DataFrame:
    with wandb.init(project=project, name=f"retrieve_table_{artifact_name}_{table_key}") as run:
        my_table = run.use_artifact(f"{artifact_name}:latest").get(f"{table_key}")
        return pd.DataFrame(my_table.data, columns=my_table.columns)


def table_add_row(run, row_dict: dict, project: str, artifact_name: str, table_key: str):
    logger.debug("Adding new row to table...")
    
    cloud_table = run.use_artifact(f"{artifact_name}:latest").get(table_key)
    cloud_df = pd.DataFrame(cloud_table.data, columns=cloud_table.columns)

    real_cols = list(cloud_df.columns)
    input_cols = list(row_dict.keys())
    real_cols.sort()
    input_cols.sort()
    
    if real_cols != input_cols:
        added_cols = list(set(input_cols) - set(real_cols))
        removed_cols = list(set(real_cols) - set(input_cols))

        logger.debug("Columns are not identical.")
        logger.debug("New columns    : %s" % added_cols)
        logger.debug("Removed columns: %s" % removed_cols)

        if len(added_cols) > 0:
            logger.debug("Create newly added columns, leaving older data values to its column: 'None'")

            for col in added_cols:
                cloud_df[col] = None

            table_update(run, cloud_df, artifact_name, table_key)
            return table_add_row(run, row_dict, project, artifact_name, table_key)
            # cloud_table = run.use_artifact(f"{artifact_name}:latest").get(table_key)

        if len(removed_cols) > 0:
            logger.debug("Inserting 'None' to removed columns")

            for col in removed_cols:
                row_dict[col] = None

    cloud_table.add_data(*[row_dict[col] for col in cloud_table.columns])
    cloud_table = pd.DataFrame(cloud_table.data, columns=cloud_table.columns)
    cloud_table = wandb.Table(dataframe=cloud_table)

    table_artifact = wandb.Artifact(artifact_name, type="table")
    table_artifact.add(cloud_table, table_key)
    run.log_artifact(table_artifact)


def table_update(run, new_df: pd.DataFrame, artifact_name: str, table_key: str):
    my_table = wandb.Table(dataframe=new_df)
    table_artifact = wandb.Artifact(artifact_name, type="table")
    table_artifact.add(my_table, table_key)
    run.log_artifact(table_artifact)


@log_execute_time("Download Cloud Preset")
def cloud_retrieve_preset(preset_name: str) -> Any:
    """Returns {Preset} by downloading preset folder from the cloud according to preset name,
    then load it.
    The goal of Saving presets are for:
    - Reproducibility
    - Easy integration to live / dry run
    (Download preset artifact -> insert big file -> Reroute big file in strategy.py)
    """
    with wandb.init(project=constants.PROJECT_NAME_PRESETS, job_type="load-data", name=f"retrieve_preset_{preset_name}") as run:
        dataset = run.use_artifact(f"{preset_name}:latest")
        directory = dataset.download()
        return os.path.join(directory, os.listdir(directory)[0])


def cloud_get_presets_df(from_run_history: bool = False) -> DataFrame:
    """Returns {DataFrame} list of presets table along with metadata.
    Used in: Notebook Backtester, ftrunner
    """
    df = table_retrieve(
        constants.PROJECT_NAME_PRESETS,
        constants.PRESETS_ARTIFACT_METADATA,
        constants.PRESETS_TABLEKEY_METADATA,
    )
    if from_run_history:
        df = df.loc[df["preset_name"].str.contains("__run-")]
    return df


def add_single_asset(path, project, asset_name):
    """Used in: Training code"""
    with wandb.init(project=project, name=f"add_artifact_{asset_name}") as run:
        artifact = wandb.Artifact(asset_name, type="single")
        artifact.add_file(path, name=asset_name)
        run.log_artifact(artifact)


def load_pickle_asset(project, asset_name, version: Union[int, str] = "latest"):
    """Used in: Strategy and ftrunner"""
    msg = f"Load pickle asset version '{version}' of project: '{project}' - asset_name: '{asset_name}'."
    logger.warning(msg)
    
    if version == "latest":
        logger.warning("WARNING: You are using the LATEST version of pickle asset!")
    
    with wandb.init(project=project, name=f"retrieve_pickle_{asset_name}:{version}") as run:
        artifact = run.use_artifact(f"{asset_name}:{version}")
        path = Path.cwd() / artifact.download()
        
        for fpath in path.glob("*.pkl"):
            with fpath.open("rb") as f:
                return dill.load(f)
        
        raise FileNotFoundError(f"No '.pkl' file in '{path}''. Files: '{list(path.glob('*'))}'")


@cache
def load_lightning_container(project, asset_name, version: Union[int, str]) -> LightningContainer:
    """Used in: Strategy and ftrunner"""
    
    msg = f"Load LightningContainer version '{version}' of project: '{project}' - asset_name: '{asset_name}'."
    logger.warning(msg)

    if version == "latest":
        raise Exception("You can't load the 'latest' version of LightningContainer asset because will break "
                        "the reproducibility when backtesting!")
    
    if isinstance(version, int):
        version = "v{version}"
    
    with wandb.init(project=project, name=f"retrieve_lightning_{asset_name}:{version}") as run:
        artifact = run.use_artifact(f"{asset_name}:{version}")
        path = Path.cwd() / artifact.download()
        
        for fpath in path.glob("*.pkl"):
            with fpath.open("rb") as f:
                container = cloudpickle.load(f)
                if not isinstance(container, LightningContainer):
                    raise Exception("Not a LightningContainer.")
                return container
        
        raise FileNotFoundError(f"No '.pkl' file in '{path}''.")


def get_lightning_artifact_ver(project, asset_name) -> str:
    with wandb.init(project=project, name=f"retrieve_lightning_{asset_name}:latest") as run:
        artifact = run.use_artifact(f"{asset_name}:latest")
        path = Path.cwd() / artifact.download()
        
        for fpath in path.glob("*.pkl"):
            with fpath.open("rb") as f:
                container = cloudpickle.load(f)
                if not isinstance(container, LightningContainer):
                    raise Exception("Not a LightningContainer.")
                return str(artifact.version)
        
        raise FileNotFoundError(f"No '.pkl' file in '{path}''.")


if __name__ == "__main__":
    ver = get_lightning_artifact_ver("15n30-catboost_l1", "15n30-catboost_l1")
    print(ver)
