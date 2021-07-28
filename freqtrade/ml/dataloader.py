from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run

import pandas as pd
import gc

from freqtrade.ml.lightning import TradingLightningModule
from freqtrade.nbtools.helper import free_mem


def load_dataset(module: TradingLightningModule) -> pd.DataFrame:
    df_list = []

    for filepath in module.data_paths:
        load_one(module, df_list, filepath)

    df = pd.concat(df_list)
    df = df.dropna()
    
    free_mem(df_list)
    print("-LOAD DATASET FINISHED-")
    
    return df


# pyright: reportGeneralTypeIssues=false
def load_one(module: TradingLightningModule, df_list: list, path: Path) -> pd.DataFrame:
    headers = ["date", "open", "high", "low", "close", "volume"]

    d1 = pd.read_json(path)
    d1.columns = headers
    d1["pair"] = path.name.split("-")[0].replace("_", "/")
    d1["date"] = pd.to_datetime(d1["date"], unit='ms')
    d1 = d1.reset_index(drop=True)
    d1 = d1[((d1.date >= module.trainval_start) &
            (d1.date <= module.trainval_end))]
    
    # Convert to FP32
    d1 = d1.astype({col: "float32" for col in ["open", "high", "low", "close", "volume"]})
    d1 = module.add_features(d1)
    d1 = module.add_labels(d1)
    
    df_list.append(d1)
    free_mem(d1)