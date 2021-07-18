# Freqtrade Notebook Backtesting

## Motivation

This project is intended to be used in cloud notebooks such as **Kaggle**, where editing Python file is not possible. So we want to edit our freqtrade strategies in the notebook, then backtest it using code. This project makes doing that possible.

As time passes, we noticed that our iteration is ineffective enough to produce better strategies. We decided to introduce `wandb` to this project. This was intended to log and sync every backtesting data (from code, to results) into cloud storage (`wandb` is free for the first 100 GB). 

In the upcoming release, this project will store the freqtrade data (obtained from `freqtrade download-data`) to that cloud service as well. This was proposed because `freqtrade download-data` is slow when we start new notebook from **Kaggle**. Through GitHub Actions, data downloading will be scheduled and synced every day.

## Installation

This script was tested and working in Kaggle.

```bash
import os

# Install TA-Lib
!wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz 2>&1 > /dev/null
!tar xvzf ta-lib-0.4.0-src.tar.gz 2>&1 > /dev/null
os.chdir('ta-lib') # Can't use !cd in co-lab
!./configure --prefix=/usr 2>&1 > /dev/null
!make 2>&1 > /dev/null
!make install 2>&1 > /dev/null
os.chdir('../')
!pip install TA-Lib 2>&1 > /dev/null

# Install Freqtrade
!git clone https://github.com/gyo-dor/superft
os.chdir("/kaggle/working/superft")
!pip install -r requirements-nb.txt
!pip uninstall sqlalchemy -y
!pip install sqlalchemy==1.4.20

# Important: if you skip this, the code will crash.
!wandb login <WANDB_API_KEY>
```

## Usage

Inside our notebook, we can define our strategy like this:

```python
def strategy_func():
    """ Start Strategy Code """
    from freqtrade.nbtools.strategy import INbStrategy
    from numpy.lib.npyio import save
    from numpy.lib.utils import info
    from pandas import DataFrame
    import numpy as np  # noqa
    import pandas as pd  # noqa
    import talib.abstract as ta

    class NotebookStrategy(INbStrategy):
	# These attributes are REQUIRED!
        timeframe = "15m"
        minimal_roi = {"0": 0.02, "30": 0.01}
        stoploss = -0.01 
        startup_candle_count: int = 100

        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe["ema_10"] = ta.EMA(dataframe["close"], timeperiod=10)
            dataframe["ema_20"] = ta.EMA(dataframe["close"], timeperiod=20)
            return dataframe

        def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (
                    (dataframe['ema_10'] > dataframe['ema_20']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'buy'] = 1
            return dataframe

        def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe.loc[
                (
                    (dataframe['ema_10'] < dataframe['ema_20']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'sell'] = 1
            return dataframe
    """ End Strategy Code """
```

Then we need to feed that strategy function into Backtester, which needs Preset, which we will introduce below.

**NOTE : Always name your strategy class "NotebookStrategy" otherwise the backtester can't detect it!**

## Config Preset Usage

This introduces config templating. You now can craft your configs and strategies everywhere. To learn more about configs, see the example config provided.

`from freqtrade.nbtools import configs` is just like any freqtrade config, but in Python importable version. You can check the official freqtrade repo then go to config_full.json.example

```python
from pathlib import Path
from freqtrade.nbtools.preset import ConfigPreset
from freqtrade.nbtools.backtest import backtest
from freqtrade.nbtools import configs

# Instantiate your preset
preset = ConfigPreset(
    config_dict=configs.DEFAULT,
    name="ma_cross",
    path_data=Path.cwd() / "data",
    timerange="20210101-20210201",
)

# Optional: Overwrite configs (fee, stake_amount, max_open_trades, etc.)
preset.overwrite_config(stake_amount=15)

# Start backtesting
stats, summary = backtest(preset, strategy_func)
```

## Local Preset Usage

This will use the config in that preset, and it's built in strategy.
To create a new Local Preset:

1. Create new folder with name of the preset. Example `ma_cross`
2. Inside that folder, create `config-backtesting.json` based on freqtrade's default full config (see in freqtrade repo)
3. Create new folder called `strategies`, `logs`, and `exports`
4. Inside `strategies`, create your own freqtrade strategy, filename must be `strategy.py`. **NOTE**: Class name must **`NotebookStrategy`**!

```python
from freqtrade.nbtools.preset import LocalPreset

preset = LocalPreset(
    path_local_preset=Path.cwd() / "presets" / "ma_cross",
    path_data=Path.cwd() / "data",
    timerange="20210101-20210201"
)

# Optional: You can still overwrite configs
preset.overwrite_config(stake_amount=15)

# To backtest it's own local strategy, you need to refer to preset.default_strategy_code
stats, summary = backtest(preset, preset.default_strategy_code)

```

## Cloud Preset Usage

This usage was intended to reproduce past strategy results (but if you want to modify configs, it's possible) since all presets that backtested through `*Preset` class will be synced using WandB (Free 100 GB storage!). This enabled us to log our progress to creating profitable freqtrade strategies.

```python
from freqtrade.nbtools.preset import CloudPreset

preset = CloudPreset(
    name="your_cloud_preset_name",
    path_data=Path.cwd() / "data",
    timerange="20210101-20210201"
)

# Optional: You can still overwrite configs
preset.overwrite_config(stake_amount=15)

# To backtest it's own strategy, you need to refer to preset.default_strategy_code
stats, summary = backtest(preset, preset.default_strategy_code)
```
