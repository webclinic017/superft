from os import PathLike
import wandb
import pandas as pd


def preset_retrieve(project: str, preset_name: str):
    pass


def preset_log(preset_directory: str, project: str, preset_name: str):
    with wandb.init(project=project, job_type="load-data") as run:
        artifact = wandb.Artifact(preset_name, type="preset")
        artifact.add_dir(f"./{preset_directory}", name=preset_name)
        run.log_artifact(artifact)


def table_retrieve(project: str, artifact_name: str, table_key: str):
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