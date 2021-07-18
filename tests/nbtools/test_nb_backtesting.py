from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import logging

from freqtrade.nbtools.preset import ConfigPreset, LocalPreset, CloudPreset
from freqtrade.nbtools.backtest import backtest, log_preset
from freqtrade.nbtools.configs import TESTING_BTC_USDT


"""
All backtest tests, DISABLE log_preset! 
"""

logger = logging.getLogger(__name__)

TIMERANGE = "20210101-20210201"
PROFIT = ""


def path_data():
    return Path.cwd() / "tests" / "testdata"


def strategy_func():
    from freqtrade.nbtools.strategy import INbStrategy
    from numpy.lib.npyio import save
    from numpy.lib.utils import info
    from pandas import DataFrame
    import numpy as np  # noqa
    import pandas as pd  # noqa
    import talib.abstract as ta

    class NotebookStrategy(INbStrategy):
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


def test_backtest_configpreset(path_data = path_data()):
    with patch('freqtrade.nbtools.backtest.log_preset', MagicMock(return_value=None)):
        preset = ConfigPreset(
            name="test_backtest_configpreset",
            path_data=path_data,
            timerange=TIMERANGE,
            config_dict=TESTING_BTC_USDT
        )
        stats, summary = backtest(preset, strategy_func)
        assert stats is not None


def test_backtest_overwritten(path_data = path_data()):
    STAKE_AMOUNT = 69
    
    with patch('freqtrade.nbtools.backtest.log_preset', MagicMock(return_value=None)):
        preset = ConfigPreset(
            name="test_backtest_configpreset",
            path_data=path_data,
            timerange=TIMERANGE,
            config_dict=TESTING_BTC_USDT
        )
        preset.overwrite_config(stake_amount=STAKE_AMOUNT)
        stats, summary = backtest(preset, strategy_func)
        assert stats is not None
        assert stats["strategy"]["NotebookStrategy"]["stake_amount"] == STAKE_AMOUNT


def test_backtest_localpreset(path_data = path_data()):
    with patch('freqtrade.nbtools.backtest.log_preset', MagicMock(return_value=None)):
        preset = LocalPreset(
            path_data=path_data,
            timerange=TIMERANGE,
            path_local_preset=path_data / "preset-test"
        )
        stats, summary = backtest(preset, preset.default_strategy_code)
        assert stats is not None


def test_backtest_cloudpreset(path_data = path_data()):
    with patch('freqtrade.nbtools.backtest.log_preset', MagicMock(return_value=None)):
        preset = CloudPreset(
            # By the time this test run, name `test_backtest_e2e` has existed in the cloud.
            name="test_backtest_e2e__backtest-2021-07-17_21-33-35", 
            timerange=TIMERANGE,
            path_data=path_data,
        )
        stats, summary = backtest(preset, preset.default_strategy_code)
        assert stats is not None


def test_backtest_e2e(path_data = path_data()):
    # TODO: Assert preset with name exists in the cloud (table and folder).
    preset = ConfigPreset(
        name="test_backtest_e2e",
        path_data=path_data,
        timerange=TIMERANGE,
        config_dict=TESTING_BTC_USDT
    )
    stats, summary = backtest(preset, strategy_func)
    assert stats is not None