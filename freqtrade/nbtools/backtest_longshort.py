from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from pandas import DataFrame

import talib.abstract as ta
import pandas_ta as pta
import pandas as pd


""" 
Trading DataFrame Columns: date, open, high, low, close, volume
"""


class ILongShortStrategy(ABC):
    timeframe: str
    startup_candle_count: int = 0
    minimal_roi: Dict
    stoploss: float
    use_sell_signal: bool = True


    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        raise NotImplementedError()
    
    def populate_long_enter(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """ `dataframe["long_enter"] = 1` for Long entry """
        return dataframe

    def populate_long_exit(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """ `dataframe["long_exit"] = 1` for Long exit """
        return dataframe
    
    def populate_short_enter(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """ `dataframe["short_enter"] = 1` for Short entry """
        return dataframe
    
    def populate_short_exit(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """ `dataframe["short_exit"] = 1` for Short exit """
        return dataframe


@dataclass
class TradingStrategy:
    """Basic trading strategy"""
    timeframe: str = "5m"

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["ma_200"] = ta.SMA(dataframe["close"], timeperiod=7)
        macd = ta.MACD(dataframe["close"])
        dataframe["macd"] = macd[0]
        dataframe["macdsignal"] = macd[1]
        return dataframe

    def populate_long_enter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[dataframe["ma_7"] > dataframe["ma_21"], "long_enter"] = 1
        return dataframe
        
    def populate_long_exit(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[dataframe["ma_7"] < dataframe["ma_21"], "long_exit"] = 1
        return dataframe
        
    def populate_short_enter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[dataframe["macd"] < dataframe["macdsignal"], "short_enter"] = 1
        return dataframe
        
    def populate_short_exit(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe.loc[dataframe["macd"] > dataframe["macdsignal"], "short_exit"] = 1
        return dataframe


@dataclass
class Position:
    """Generic trading position"""
    open_date: datetime
    open_rate: float
    direction: int  # 0 for short, 1 for long
    close_date: datetime = None  # type: ignore
    close_rate: float = None  # type: ignore
    close_reason: str = None  # type: ignore
   
    
@dataclass
class BacktestConfiguration:
    """Basic backtest config"""
    starting_balance: float = 1000
    stake_amount: float = 20
    max_open_trades: int = 5
    stoploss: float = 0.05


class Backtesting:
    """Class to backtest a trading strategy"""

    def __init__(self, strategy: ILongShortStrategy, data: pd.DataFrame, config: BacktestConfiguration):
        self.strategy: ILongShortStrategy = strategy
        self.data: pd.DataFrame = data
        self.config: BacktestConfiguration = config
        self.long_position: Optional[Position] = None
        self.short_position: Optional[Position] = None
        self.closed_positions: List[Position] = []
    
    def start_backtesting(self) -> pd.DataFrame:
        self._populate_signals()
        self._populate_positions()
        return self._calculate_profits()
    
    def _populate_signals(self):
        """Populate data in a strategy"""
        self.data["long_enter"] = 0
        self.data["long_exit"] = 0
        self.data["short_enter"] = 0
        self.data["short_exit"] = 0
        metadata = {}
        self.data = self.strategy.populate_indicators(self.data, metadata)
        self.data = self.strategy.populate_long_enter(self.data, metadata)
        self.data = self.strategy.populate_long_exit(self.data, metadata)
        self.data = self.strategy.populate_short_enter(self.data, metadata)
        self.data = self.strategy.populate_short_exit(self.data, metadata)
        
    def _enter(self, direction: int, open_date: datetime, open_rate: float):
        """Enter a position"""
        if direction == 0:
            self.short_position = Position(open_date, open_rate, direction)
        elif direction == 1:
            self.long_position = Position(open_date, open_rate, direction)
            
    def _exit(self, direction: int, close_date: datetime, close_rate: float, close_reason: str):
        """Exit a position"""
        if direction == 0:
            self.short_position.close_date = close_date
            self.short_position.close_rate = close_rate
            self.short_position.close_reason = close_reason
            self.closed_positions.append(self.short_position) # type: ignore
            self.short_position = None
        elif direction == 1:
            self.long_position.close_date = close_date
            self.long_position.close_rate = close_rate
            self.long_position.close_reason = close_reason
            self.closed_positions.append(self.long_position) # type: ignore
            self.long_position = None
            
    def _check_long_exit(self, should_exit: bool, date: datetime, open: float, high: float, low: float, close: float):
        """Close a Long position if exists and condition meets"""
        if low < self.long_position.open_rate * (1 - self.config.stoploss):
            return self._exit(1, date, low, "stoploss")
        if should_exit:
            return self._exit(1, date, close, "signal")
            
    def _check_short_exit(self, should_exit: bool, date: datetime, open: float, high: float, low: float, close: float):
        """Close a Short position if exists and condition meets"""
        if high > self.short_position.open_rate * (1 + self.config.stoploss):
            return self._exit(0, date, high, "stoploss")
        if should_exit:
            return self._exit(0, date, close, "signal")
            
    def _populate_positions(self):
        """Convert DataFrame to list then loop through it to populate positions"""
        IDX_DATE = 0
        IDX_OPEN = 1
        IDX_HIGH = 2
        IDX_LOW = 3
        IDX_CLOSE = 4
        IDX_VOLUME = 5
        IDX_long_enter = 6
        IDX_long_exit = 7
        IDX_short_enter = 8
        IDX_short_exit = 9
        
        data = self._filter_columns().values.tolist()
        
        for i in range(len(data)):
            if self.long_position is not None:
                self._check_long_exit(
                    data[i][IDX_long_exit] == 1, 
                    data[i][IDX_DATE],
                    data[i][IDX_OPEN],
                    data[i][IDX_HIGH],
                    data[i][IDX_LOW],
                    data[i][IDX_CLOSE],
                )
            if self.short_position is not None:
                self._check_short_exit(
                    data[i][IDX_short_exit] == 1, 
                    data[i][IDX_DATE],
                    data[i][IDX_OPEN],
                    data[i][IDX_HIGH],
                    data[i][IDX_LOW],
                    data[i][IDX_CLOSE],
                )
            
            if data[i][IDX_long_enter] == 1 and self.long_position is None:
                self._enter(1, data[i][IDX_DATE], data[i][IDX_CLOSE])
            if data[i][IDX_short_enter] == 1 and self.short_position is None:
                self._enter(0, data[i][IDX_DATE], data[i][IDX_CLOSE])
                
    def _get_positions_dataframe(self) -> pd.DataFrame:
        """Convert closed positions into pandas DataFrame"""
        if len(self.closed_positions) == 0:
            return pd.DataFrame()
        return pd.DataFrame([[
            position.open_date,
            position.open_rate,
            position.close_date,
            position.close_rate,
            position.direction
        ] for position in self.closed_positions], columns=[
            "open_date",
            "open_rate",
            "close_date",
            "close_rate",
            "direction"
        ])
    
    def _calculate_profits(self) -> pd.DataFrame:
        """Calculate profits of our positions and other things"""
        df_positions = self._get_positions_dataframe()
        df_positions.loc[df_positions["direction"] == 0, "profit_ratio"] = (df_positions["open_rate"] - df_positions["close_rate"]) / df_positions["open_rate"]
        df_positions.loc[df_positions["direction"] == 1, "profit_ratio"] = (df_positions["close_rate"] - df_positions["open_rate"]) / df_positions["open_rate"]
        df_positions["profit_abs"] = df_positions["profit_ratio"] * self.config.stake_amount
        df_positions["profit_abs_cum"] = df_positions["profit_abs"].cumsum()
        return df_positions
        
    def _filter_columns(self) -> pd.DataFrame:
        """Filter columns from self.data"""
        df_data = self.data.copy()
        df_data = df_data[["date", "open", "high", "low", "close", "volume", "long_enter", "long_exit", "short_enter", "short_exit"]]
        return df_data