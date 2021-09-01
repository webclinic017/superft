# Standard Imports
from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run
from datetime import datetime, timedelta
from copy import deepcopy

import gc
import os
import logging
import sys
import pandas as pd
import numpy as np
import stackprinter
import click

pd.set_option('display.max_rows', 200)
stackprinter.set_excepthook(style='darkbg2')  # for jupyter notebooks try style='lightbg'


# Resolve CWD
gc.collect()

while "freqtrade" not in os.listdir():
    os.chdir("..")
    # if "freqtrade" in os.listdir():
    #     logger = logging.getLogger("freqtrade")
    #     handler = logging.StreamHandler(stream=sys.stdout)
    #     handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    #     logger.addHandler(handler)
    #     logger.setLevel(logging.INFO)
    #     logging.getLogger("distributed.utils_perf").setLevel(logging.ERROR)

# Freqtrade Imports
from freqtrade.nbtools.preset import LocalPreset, ConfigPreset, FilePreset, CloudPreset
from freqtrade.nbtools.hyperopt import start_hyperopt
from freqtrade.nbtools import plotting, configs

# Constants
PATH_MOUNT = Path.cwd().parent / "mount"
PATH_DATA = PATH_MOUNT / "data"
PATH_PRESETS = PATH_MOUNT / "presets"
PATH_STRATEGIES = PATH_PRESETS / ".strategy_files"

# Define custom functions
def foo() -> str:
    return "bar"


@click.command()
@click.option("--strategy", help="Strategy filename (in .strategy_files). Must same as class name", required=True)
@click.option("--timerange", help="Freqtrade timerange", required=True)
@click.option("--pairs", help="Separate with space!", required=True)
@click.option("--spaces", help="all, buy, sell, roi, stoploss, trailing, default (all exc. trailing)", required=True)
@click.option("--epochs", help="Number of epochs", required=True, type=int)
@click.option("--hyperopt_loss", 
    help="SharpeHyperOptLoss, SortinoHyperOptLoss, OnlyProfitHyperOptLoss, ShortTradeDurHyperOptLoss, Sharpe/Sortino + Daily",
    required=True)
@click.option("--hyperopt_min_trades", help="Minimum trades", required=True, type=int)
@click.option("--hyperopt_random_state", help="Random state to be reproducible", required=True, default=2, type=int)
@click.option("--hyperopt_jobs", help="Number of cores", required=True, default=12, type=int)
def main(
    strategy: str,
    timerange: str,
    pairs,
    spaces: str,
    epochs: int,
    hyperopt_loss: str,
    hyperopt_min_trades: int,
    hyperopt_random_state: int,
    hyperopt_jobs: int
):
    pairs = pairs.split(",")
    print(f"\n{locals()}\n")
    
    # Hyperopt Arguments
    hyperopt_args = {
        "spaces": spaces,
        "epochs": epochs,
        "hyperopt_loss": hyperopt_loss,
        "hyperopt_min_trades": hyperopt_min_trades,
        "hyperopt_random_state": hyperopt_random_state,
        "hyperopt_jobs": hyperopt_jobs,
        "verbosity": 0,
    }

    preset = FilePreset(
        timerange = timerange,
        config_dict = configs.DEFAULT,
        path_to_file = PATH_STRATEGIES / f"{strategy}.py",
        path_data = PATH_DATA,
    )

    # Optional override
    preset.overwrite_config(
        strategy_search_path = PATH_STRATEGIES,
        pairs                = pairs,
    )

    start_hyperopt(preset, hyperopt_args=hyperopt_args, clsname=strategy)
    

if __name__ == "__main__":
    main()