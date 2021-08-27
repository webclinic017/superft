# Standard Imports
from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run
from datetime import datetime, timedelta
from copy import deepcopy

import attr
import gc
import os
import wandb
import nest_asyncio
import logging
import sys
import pandas as pd
import numpy as np
import qgrid
import stackprinter
pd.set_option('display.max_rows', 200)
stackprinter.set_excepthook(style='darkbg2')  # for jupyter notebooks try style='lightbg'


# Resolve CWD
gc.collect()
nest_asyncio.apply()

while "freqtrade" not in os.listdir():
    os.chdir("..")
    if "freqtrade" in os.listdir():
        nest_asyncio.apply()
        logger = logging.getLogger("freqtrade")
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logging.getLogger("distributed.utils_perf").setLevel(logging.ERROR)

# Freqtrade Imports
from freqtrade.optimize.optimize_reports import text_table_add_metrics
from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.ml.lightning import LightningModule, LightningConfig
from freqtrade.ml.trainer import TradingTrainer
from freqtrade.ml.container import LightningContainer
from freqtrade.ml import loader, lightning_utils
from freqtrade.nbtools.preset import LocalPreset, ConfigPreset, FilePreset, CloudPreset
from freqtrade.nbtools.hyperopt import start_hyperopt
from freqtrade.nbtools.helper import free_mem
from freqtrade.nbtools.pairs import PAIRS_HIGHCAP_NONSTABLE
from freqtrade.nbtools import plotting, configs

# Constants
PATH_MOUNT = Path.cwd().parent / "mount"
PATH_DATA = PATH_MOUNT / "data"
PATH_PRESETS = PATH_MOUNT / "presets"
PATH_STRATEGIES = PATH_PRESETS / ".strategy_files"

# Define custom functions
def foo() -> str:
    return "bar"


strategy_classname = "DIY_MACDLongTermHS"  # Strategy Filename and Classname must same! 
timerange          = "20210101-"
pairs              = PAIRS_HIGHCAP_NONSTABLE

# pairs              = [
#     "DOGE/USDT",
#     "BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "BCH/USDT", "EOS/USDT", "NEO/USDT", "NANO/USDT", "XMR/USDT", "ZEC/USDT",
# ]

# Hyperopt Arguments
hyperopt_args = {
    # all, buy, sell, roi, stoploss, trailing, default (all exc. trailing)
    "spaces": "buy",
    "epochs": 3,
    # SharpeHyperOptLoss, SortinoHyperOptLoss, OnlyProfitHyperOptLoss, ShortTradeDurHyperOptLoss, or Sharpe/Sortino + Daily
    "hyperopt_loss": "SharpeHyperOptLoss",
    "hyperopt_min_trades": 10,
    "hyperopt_random_state": 2,
    "hyperopt_jobs": 12,
}

preset = FilePreset(
    timerange = timerange,
    config_dict = configs.DEFAULT,
    path_to_file = PATH_STRATEGIES / f"{strategy_classname}.py",
    path_data = PATH_DATA,
)

# Optional override
preset.overwrite_config(
    strategy_search_path = PATH_STRATEGIES,
    pairs                = pairs,
#     max_open_trades      = 3,
#     starting_balance     = 100,
#     stake_amount         = "unlimited",
#     timeframe            = "2h", # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
)

# preset._config = {
#     **preset._config,
#     **hyperopt_args,
# }

start_hyperopt(preset, hyperopt_args=hyperopt_args, clsname=strategy_classname)