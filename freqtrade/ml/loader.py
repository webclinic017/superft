from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run

import pandas as pd
import gc

from freqtrade.ml.lightning import LightningModule
from freqtrade.nbtools.helper import free_mem
    
    
def load_df(path_json: Path):
    headers = ["date", "open", "high", "low", "close", "volume"]
    d1 = pd.read_json(path_json)
    d1.columns = headers
    d1["date"] = pd.to_datetime(d1["date"], unit='ms')
    d1 = d1.reset_index(drop=True)
    d1 = to_fp32(d1)
    return d1


def to_fp32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            col: "float32"
            for col in df.columns
            if str(df[col].dtype) in ["float64", "float16"]
        }
    )