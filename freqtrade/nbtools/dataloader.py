from pathlib import Path
from typing import List, Callable

import pandas as pd
import gc


def load_dataset(paths: List[Path], feature_function: Callable[[pd.DataFrame], pd.DataFrame], start_date: str, stop_date: str):
    df_list = []

    for filepath in paths:
        load_one(df_list, filepath, feature_function, start_date, stop_date)

    df = pd.concat(df_list)
    df = df.dropna()
    
    free_mem(df_list)
    print("LOAD DATASET FINISHED.")
    
    return df


# pyright: reportGeneralTypeIssues=false
def load_one(df_list: list, path: Path, feature_function: Callable[[pd.DataFrame], pd.DataFrame], start_date: str, stop_date: str):
    print(f"Loading: {path.name}")
    headers = ["date", "open", "high", "low", "close", "volume"]

    d1 = pd.read_json(path)
    d1.columns = headers
    d1["pair"] = path.name.split("-")[0].replace("_", "/")
    d1["date"] = pd.to_datetime(d1["date"], unit='ms')
    d1 = d1.reset_index(drop=True)
    d1 = d1[((d1.date >= start_date) &
            (d1.date <= stop_date))]
    
    d1 = feature_function(d1)
    
    df_list.append(d1)
    free_mem(d1)
    

def free_mem(var):
    del var
    gc.collect()