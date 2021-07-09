from typing import *
from pandas import DataFrame
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from os import PathLike
from copy import deepcopy
import attr
import os
import wandb
import rapidjson

from freqtrade.enums.runmode import RunMode
from freqtrade.nbtools import constants

from .config import base_config
from .configuration import setup_optimize_configuration
from .backtest import NbBacktesting
from .helper import get_function_body, get_readable_date
from .remote_utils import preset_log, table_add_row


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
    config_raw: Dict[str, Any] = attr.ib(default={}, init=False)
    config_optimize: Dict[str, Any] = attr.ib(default={}, init=False)
        
    @classmethod
    def from_preset(preset_path: PathLike):
        pass
    
    def __attrs_post_init__(self):
        wandb.login()
        self.config_raw = self._get_config_raw()
        self.config_optimize = self._get_config_optimize(self.config_raw)
        
    def _get_config_raw(self) -> dict:
        """Editable config.json"""
        config = deepcopy(base_config)
        config["max_open_trades"] = self.max_open_trades
        config["stake_amount"] = self.stake_amount
        config["exchange"]["name"] = self.exchange
        config["exchange"]["pair_whitelist"] = self.pairs
        config["bot_name"] = self.name
        config["fee"] = self.fee
        return config
        
    def _get_config_optimize(self, config_raw) -> dict:
        """Using pipeline of freqtrade's config and args system"""
        args = {
            "datadir": self.datadir, 
            "timerange": self.timerange, 
            "timeframe": self.timeframe
        }
        config = setup_optimize_configuration(config_raw, args, RunMode.BACKTEST)
        return config
        
    def backtest_by_strategy_func(self, strategy_func: Callable[[Any], Any]) -> Tuple[dict, str]:
        """ Given config, pairs, do a freqtrade backtesting
            TODO: Backtest multiple strategies
        """
        strategy_code = get_function_body(strategy_func)
        backtester = NbBacktesting(self.config_optimize)
        stats, summary = backtester.start_nb_backtesting(strategy_code)
        self._log_preset(strategy_code, stats, summary)
        return stats, summary
    
    def _log_preset(self, strategy_code: str, stats: dict, summary: str):
        """
        Don't upload anything if anything error.
        Upload:
        - metadata.json (contains backtesting parameters defined in notebook)
        - config_backtesting.json
        - strategies/strategy.py (Strategy code extracted from function by the help of inspect)
        - exports/stats.json
        - exports/summary.txt (The ones in terminal you see after backtesting)
        """
        current_date = get_readable_date()
        preset_name = f"{self.name}__backtest-{current_date}"
        filename_and_content = {
            "metadata.json": self._generate_metadata(stats, preset_name, current_date),
            "config-backtesting.json": self.config_raw,
            "config-optimize.json": self.config_optimize,
            "exports/stats.json": stats,
            "exports/summary.txt": summary,
            "strategies/strategy.py": strategy_code,
        }
        
        # Generate folder and files
        os.mkdir(f"./.temp/{preset_name}")
        os.mkdir(f"./.temp/{preset_name}/exports")
        os.mkdir(f"./.temp/{preset_name}/strategies")
        
        for filename, content in filename_and_content.items():
            with open(f"./.temp/{preset_name}/{filename}", mode="w") as f:
                if filename.endswith(".json"):
                    rapidjson.dump(content, f, default=str, number_mode=rapidjson.NM_NATIVE)
                    continue
                f.write(content)
        
        # wandb log artifact
        preset_log( f"./.temp/{preset_name}", constants.PROJECT_NAME, preset_name)
        # wandb add row
        table_add_row(filename_and_content["metadata.json"], 
                      constants.PROJECT_NAME, 
                      constants.ARTIFACT_TABLE_METADATA, 
                      constants.TABLEKEY_METADATA)
        
    def _generate_metadata(self, stats: dict, folder_name: str, current_date: str) -> dict:
        """ Generate backtesting summary in dict / json format
        """
        trades = DataFrame(deepcopy(stats["strategy"]["NotebookStrategy"]["trades"]))
        trades_summary = deepcopy(stats["strategy"]["NotebookStrategy"])
        current_date = get_readable_date()

        del trades_summary["trades"]
        del trades_summary["locks"]
        del trades_summary["best_pair"]
        del trades_summary["worst_pair"]
        del trades_summary["results_per_pair"]
        del trades_summary["sell_reason_summary"]
        del trades_summary["left_open_trades"]

        metadata = {}
        metadata["preset_name"] = folder_name
        metadata["backtest_date"] = current_date
        metadata["leverage"] = 1
        metadata["direction"] = "long"
        metadata["is_hedging"] = False
        metadata["num_pairs"] = len(trades_summary["pairlist"])
        metadata["data_source"] = "self.config['exchange']"
        metadata["win_rate"] = trades_summary["wins"] / trades_summary["total_trades"]
        metadata["avg_profit_winners_abs"] = trades.loc[trades["profit_abs"] >= 0, "profit_abs"].dropna().mean()
        metadata["avg_profit_losers_abs"] = trades.loc[trades["profit_abs"] < 0, "profit_abs"].dropna().mean()
        metadata["sum_profit_winners_abs"] = trades.loc[trades["profit_abs"] >= 0, "profit_abs"].dropna().sum()
        metadata["sum_profit_losers_abs"] = trades.loc[trades["profit_abs"] < 0, "profit_abs"].dropna().sum()
        metadata["profit_factor"] = metadata["sum_profit_winners_abs"] / abs(metadata["sum_profit_losers_abs"])
        metadata["profit_per_drawdown"] = trades_summary["profit_total_abs"] / abs(trades_summary["max_drawdown_abs"])
        metadata["expectancy_abs"] = (
            (metadata["win_rate"] * metadata["avg_profit_winners_abs"]) + 
            ((1 - metadata["win_rate"]) * metadata["avg_profit_losers_abs"])
        )

        for key, value in trades_summary.items():
            is_valid = any(
                [isinstance(value, it) for it in (str, int, float, bool)] + [value is None],
            )
            if not is_valid:
                trades_summary[key] = str(value)

        return {**metadata, **trades_summary}
    