from pathlib import Path
from typing import List, Callable, Tuple, Any
from wandb.wandb_run import Run

import pandas as pd
import gc
import ccxt
    
    
def load_df(path_json: Path, timeframe: str):
    headers = ["date", "open", "high", "low", "close", "volume"]
    d1 = pd.read_json(str(path_json))
    d1.columns = headers
    d1["date"] = pd.to_datetime(d1["date"], unit='ms', utc=True, infer_datetime_format=True)
    d1 = d1.astype(
        dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
    )
    d1 = clean_ohlcv_dataframe(d1, timeframe, fill_missing=True, drop_incomplete=True)
    return d1


def to_fp32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            col: "float32"
            for col in df.columns
            if str(df[col].dtype) in ["float64", "float16"]
        }
    )
    

def clean_ohlcv_dataframe(data: pd.DataFrame, timeframe: str, *,
                          fill_missing: bool = True,
                          drop_incomplete: bool = True) -> pd.DataFrame:
    """
    Cleanse a OHLCV dataframe by
      * Grouping it by date (removes duplicate tics)
      * dropping last candles if requested
      * Filling up missing data (if requested)
    :param data: DataFrame containing candle (OHLCV) data.
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :return: DataFrame
    """
    # group by index and aggregate results to eliminate duplicate ticks
    data = data.groupby(by='date', as_index=False, sort=True).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'max',
    })
    # eliminate partial candle
    if drop_incomplete:
        data.drop(data.tail(1).index, inplace=True)

    if fill_missing:
        return ohlcv_fill_up_missing_data(data, timeframe)
    else:
        return data


def ohlcv_fill_up_missing_data(dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Fills up missing data with 0 volume rows,
    using the previous close as price for "open", "high" "low" and "close", volume is set to 0

    """
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    timeframe_minutes = timeframe_to_minutes(timeframe)
    # Resample to create "NAN" values
    df = dataframe.resample(f'{timeframe_minutes}min', on='date').agg(ohlcv_dict)

    # Forwardfill close for missing columns
    df['close'] = df['close'].fillna(method='ffill')
    # Use close for "open, high, low"
    df.loc[:, ['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(
        value={'open': df['close'],
               'high': df['close'],
               'low': df['close'],
               })
    df.reset_index(inplace=True)
    len_before = len(dataframe)
    len_after = len(df)
    pct_missing = (len_after - len_before) / len_before if len_before > 0 else 0
    return df


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60