# mypy: ignore-errors
import os
from pathlib import Path
from typing import Any

import dill
import pandas as pd
from pandas import DataFrame

import wandb
from freqtrade.nbtools import constants


os.environ["WANDB_SILENT"] = "true"


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

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


def preset_log(preset_directory: str, preset_name: str):
    with wandb.init(project=constants.PROJECT_NAME_PRESETS, job_type="load-data") as run:
        artifact = wandb.Artifact(preset_name, type="preset")
        artifact.add_dir(f"./{preset_directory}", name=preset_name)
        run.log_artifact(artifact)


def table_retrieve(project: str, artifact_name: str, table_key: str) -> pd.DataFrame:
    with wandb.init(project=project) as run:
        my_table = run.use_artifact(f"{artifact_name}:latest").get(f"{table_key}")
        return pd.DataFrame(my_table.data, columns=my_table.columns)


def table_add_row(row_dict: dict, project: str, artifact_name: str, table_key: str):

    with wandb.init(project=project) as run:
        cloud_table = run.use_artifact(f"{artifact_name}:latest").get(f"{table_key}")
        cloud_df = pd.DataFrame(cloud_table.data, columns=cloud_table.columns)

        real_cols = list(cloud_df.columns)
        input_cols = list(row_dict.keys())
        real_cols.sort()
        input_cols.sort()

        if real_cols != input_cols:
            added_cols = list(set(input_cols) - set(real_cols))
            removed_cols = list(set(real_cols) - set(input_cols))

            print("Columns are not identical.")
            print("Added columns  :", added_cols)
            print("Removed columns:", removed_cols)

            if len(added_cols) > 0:
                print("Create newly added columns, leaving older data values to its column: 'old'.")

                for col in added_cols:
                    cloud_df[col] = "old"

                table_update(cloud_df, project, artifact_name, table_key)
                return table_add_row(row_dict, project, artifact_name, table_key)

            elif len(removed_cols) > 0:
                print("Insert value to removed column: 'rem'.")

                for col in removed_cols:
                    row_dict[col] = "rem"

        cloud_table.add_data(*[row_dict[col] for col in cloud_table.columns])
        cloud_table = pd.DataFrame(cloud_table.data, columns=cloud_table.columns)
        cloud_table = wandb.Table(dataframe=cloud_table)

        table_artifact = wandb.Artifact(artifact_name, type="table")
        table_artifact.add(cloud_table, table_key)
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


def cloud_retrieve_preset(preset_name: str) -> Any:
    """Returns {Preset} by downloading preset folder from the cloud according to preset name,
    then load it.
    The goal of Saving presets are for:
    - Reproducibility
    - Easy integration to live / dry runis
    (Download preset artifact -> insert big file -> Reroute big file in strategy.py)
    """
    with wandb.init(project=constants.PROJECT_NAME_PRESETS, job_type="load-data") as run:
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
    with wandb.init(project=project) as run:
        artifact = wandb.Artifact(asset_name, type="single")
        artifact.add_file(path, name=asset_name)
        run.log_artifact(artifact)


@Memoize
def load_pickle_asset(project, asset_name):
    """Used in: Strategy and ftrunner"""
    with wandb.init(project=project) as run:
        artifact = run.use_artifact(f"{asset_name}:latest")
        path = Path.cwd() / artifact.download()
        filepath = path / os.listdir(path)[0]
        with filepath.open("rb") as f:
            return dill.load(f)


if __name__ == "__main__":
    print("LOAD 1st")
    print(str(load_pickle_asset("legacy-models", "15m-next30m-10_06_new.pkl"))[:50])
    print("LOAD 2nd")
    print(str(load_pickle_asset("legacy-models", "15m-next30m-10_06_new.pkl"))[:50])
