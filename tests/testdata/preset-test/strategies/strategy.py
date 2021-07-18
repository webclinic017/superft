# For the pkl file, it is at: https://drive.google.com/file/d/1-L3ZQAOsBIpYa5ibStH84Edle8XOCGn_/view?usp=sharing
from numpy.lib.npyio import save
from numpy.lib.utils import info
from pandas import DataFrame
import numpy as np  # noqa
import pandas as pd  # noqa
from freqtrade.strategy import IStrategy, merge_informative_pair
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import sklearn
import pickle
import marshal
import types


class NotebookStrategy(IStrategy):
    INTERFACE_VERSION = 2
    timeframe = '15m'

    # NOTE: Current Max Open Trades is 1000! # NOTE NOTE NOTE
    minimal_roi = {"0": 0.02, "30": 0.01}
    stoploss = -0.01
    startup_candle_count: int = 100

    # Trailing stoploss (NOTE: DON'T USE!)
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

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
