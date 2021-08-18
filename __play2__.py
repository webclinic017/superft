import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import pandas_ta


def ma_crossover(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ 
    Indicators needed:
    - Moving Average with length of 15
    - Moving Average with length of 30
    
    Buy when:
    - MA 15 crosses above MA 30
    - Volume > 0
    """
    dataframe['MA15'] = dataframe.ta.sma(length=15)
    dataframe['MA30'] = dataframe.ta.sma(length=30)
    dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['MA15'], dataframe['MA30']) &
            dataframe['volume'] > 0
        ),
    "buy" ] = 1
    return dataframe


def ssl_strategy(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ 
    Indicators needed:
    - EMA 200
    - SSL 21
    - WAE (Sensitivity: 150, FastEMA: 20, SlowEMA: 40, BB Channel: 20, BB Stdev Mult: 2)
    
    Buy when:
    - MACD crosses up
    - Price is above EMA200
    """
    dataframe['MACD'] = dataframe.ta.macd(length_short=12, length_long=26)
    