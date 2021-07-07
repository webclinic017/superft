from typing import *
from pandas import DataFrame
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from os import PathLike
import inspect
import attr

from freqtrade.configuration.check_exchange import remove_credentials
from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.enums.runmode import RunMode
from .config import base_config
from .configuration import setup_optimize_configuration
from .backtest import NbBacktesting
from .helper import get_function_body, get_class_from_string


@attr.s
class Preset:
    name: str = attr.ib()
    datadir: str = attr.ib()
    exchange: str = attr.ib()
    timeframe: str = attr.ib()
    timerange: str = attr.ib()
    stake_amount: int = attr.ib()
    pairs: List[str] = attr.ib()
    initial_balance: float = attr.ib(default=0)
    max_open_trades: int = attr.ib(default=1000)
    fee: float = attr.ib(default=0.001)
        
    @classmethod
    def from_preset(preset_path: PathLike):
        pass
        
    def _get_config(self):
        """Using pipeline of freqtrade's config and args system"""
        config = deepcopy(base_config)
        config["max_open_trades"] = self.max_open_trades
        config["stake_amount"] = self.stake_amount
        config["exchange"]["name"] = self.exchange
        config["exchange"]["pair_whitelist"] = self.pairs
        config["bot_name"] = self.name
        config["fee"] = self.fee
        args = {
            "datadir": self.datadir, 
            "timerange": self.timerange, 
            "timeframe": self.timeframe
        }
        config = setup_optimize_configuration(config, args, RunMode.BACKTEST)
        return config
        
    def backtest(self, strategy_func: Callable[[Any], Any]) -> Tuple[dict, str]:
        """Given config, pairs, do a freqtrade backtesting"""
        strategy_code = get_function_body(strategy_func)
        config = self._get_config()
        backtester = NbBacktesting(config)
        stats, summary = backtester.start_nb_backtesting(strategy_code)
        self._post_backtest()
        return stats, summary
    
    def _post_backtest(self):
        """
        Don't upload anything if anything errors.
        Upload:
        - metadata.json (contains backtesting parameters defined in notebook)
        - config_base.json
        - config_backtesting.json
        - config_dryrun.json
        - config_liverun.json (without keys)
        - strategies/strategy.py (Strategy code extracted from function by the help of inspect)
        - exports/trades.json
        - exports/profits.png (The three profit plotting)
        - exports/summary.txt (The ones in terminal you see after backtesting)
        """
        pass
    
    def _create_preset_files_as_dict(self):
        pass
    
    def _upload(self, filename: str, content_str: str):
        pass
    