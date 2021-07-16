import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import attr
import rapidjson
from pandas import DataFrame

import wandb
from freqtrade.enums.runmode import RunMode
from freqtrade.nbtools import configs, constants

from .backtest import NbBacktesting
from .configuration import setup_optimize_configuration
from .helper import get_function_body, get_readable_date
from .remote_utils import cloud_retrieve_preset, preset_log, table_add_row


wandb.login()


@attr.s
class Preset:
    name: str = attr.ib()
    exchange: str = attr.ib()
    stake_amount: int = attr.ib()
    pairs: List[str] = attr.ib()
    starting_balance: float = attr.ib(default=1000)
    max_open_trades: int = attr.ib(default=1000)
    fee: float = attr.ib(default=0.001)
    strategy_code: Optional[str] = attr.ib(default=None, init=False)
    path_local_preset: Optional[Path] = attr.ib(default=None, init=False)
    datadir: Optional[str] = attr.ib(default=None, kw_only=True)
    timeframe: Optional[str] = attr.ib(default=None, kw_only=True)
    timerange: Optional[str] = attr.ib(default=None, kw_only=True)

    @staticmethod
    def from_cloud(cloud_preset_name: str) -> Tuple["Preset", str]:
        preset_path = cloud_retrieve_preset(cloud_preset_name)
        preset_path = Path.cwd() / preset_path
        return Preset.from_local(preset_path, is_from_cloud=True)

    @staticmethod
    def from_local(path_local_preset: Path, is_from_cloud: bool = False) -> Tuple["Preset", str]:
        """Loads preset from local folder then returns Preset and strategy code"""
        if not isinstance(path_local_preset, Path):
            raise RuntimeError("Expected Path object, got '{}'".format(type(path_local_preset)))

        with (path_local_preset / "config-backtesting.json").open("r") as fs:
            config_dict = rapidjson.load(fs)

        try:
            with (path_local_preset / "metadata.json").open("r") as fs:
                metadata_dict = rapidjson.load(fs)
        except FileNotFoundError:
            metadata_dict = {"preset_name": path_local_preset.name}

        with (path_local_preset / "strategies" / "strategy.py").open("r") as fs:
            strategy_code = fs.read()

        preset = Preset(
            name=metadata_dict["preset_name"].split("__")[0],
            exchange=config_dict["exchange"]["name"],
            timeframe=config_dict.get("timeframe") or metadata_dict.get("timeframe") or None,
            timerange=config_dict.get("timerange")
            or metadata_dict.get("timerange")
            or "[ PLEASE ENTER TIMERANGE ]",
            stake_amount=config_dict["stake_amount"],
            pairs=config_dict["exchange"]["pair_whitelist"],
            starting_balance=config_dict.get("dry_run_wallet") or 1000,
            max_open_trades=config_dict["max_open_trades"],
            fee=config_dict.get("fee") or 0.001,
        )
        preset.strategy_code = strategy_code
        if not is_from_cloud:
            preset.path_local_preset = path_local_preset
        return preset, strategy_code

    def backtest_by_strategy_func(self, strategy_func: Callable[[Any], Any]) -> Tuple[dict, str]:
        """
        Given config, pairs, do a freqtrade backtesting
        """
        assert self.datadir is not None, "Please fix your datadir!"

        config_editable, config_optimize = self._get_configs()
        strategy_code = get_function_body(strategy_func)
        backtester = NbBacktesting(config_optimize)
        stats, summary = backtester.start_nb_backtesting(strategy_code)
        self._log_preset(strategy_code, stats, summary, config_editable, config_optimize)
        return stats, summary

    def backtest_by_default_strategy_code(self) -> Tuple[dict, str]:
        """
        Presets retrieved from the cloud can use default strategy code to backtest.
        """
        assert self.datadir is not None, "Please fix your datadir!"
        assert self.strategy_code is not None, "No default strategy code"

        config_editable, config_optimize = self._get_configs()
        backtester = NbBacktesting(config_optimize)
        stats, summary = backtester.start_nb_backtesting(self.strategy_code)
        self._log_preset(self.strategy_code, stats, summary, config_editable, config_optimize)
        return stats, summary

    def _get_configs(self) -> Tuple[dict, dict]:
        """
        Process config through freqtrade's config parser
        """
        # TODO: If exists, Load from "config-backtesting.json" as base config.
        config = deepcopy(configs.DEFAULT)
        update = {
            "max_open_trades": self.max_open_trades,
            "stake_amount": self.stake_amount,
            "dry_run_wallet": self.starting_balance,
            "fee": self.fee,
            "bot_name": self.name,
            "exchange": {
                "name": self.exchange,
                "pair_whitelist": self.pairs,
            },
        }
        config.update(update)
        args = {
            "datadir": self.datadir,
            "timerange": self.timerange,
        }
        if self.timeframe is not None:
            args.update({"timeframe": self.timeframe})

        config_optimize = setup_optimize_configuration(config, args, RunMode.BACKTEST)
        return config, config_optimize

    def _log_preset(
        self,
        strategy_code: str,
        stats: dict,
        summary: str,
        config_editable: dict,
        config_optimize: dict,
    ):
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
        metadata = self._generate_metadata(stats, preset_name, current_date)
        filename_and_content = {
            "metadata.json": metadata,
            "config-backtesting.json": config_editable,
            "config-optimize.json": config_optimize,
            "exports/stats.json": stats,
            "exports/summary.txt": summary,
            "strategies/strategy.py": strategy_code,
        }

        # Generate folder and files to be uploaded
        os.mkdir(f"./.temp/{preset_name}")
        os.mkdir(f"./.temp/{preset_name}/exports")
        os.mkdir(f"./.temp/{preset_name}/strategies")

        for filename, content in filename_and_content.items():
            with open(f"./.temp/{preset_name}/{filename}", mode="w") as f:
                if "config" in filename or filename == "metadata.json":
                    json.dump(content, f, default=str, indent=4)
                    continue
                if filename.endswith(".json"):
                    rapidjson.dump(content, f, default=str, number_mode=rapidjson.NM_NATIVE)
                    continue
                if isinstance(content, str):
                    f.write(content)

        # wandb log artifact
        preset_log(f"./.temp/{preset_name}", preset_name)
        # wandb add row
        table_add_row(
            metadata,
            constants.PROJECT_NAME_PRESETS,
            constants.PRESETS_ARTIFACT_METADATA,
            constants.PRESETS_TABLEKEY_METADATA,
        )

        # (if use local preset) update local results
        if self.path_local_preset is not None:
            print(
                f"You are backtesting a local preset `{self.path_local_preset}`"
            )
            print(
                "This will update backtest results (such as metadata.json, exports)"
            )
            print(
                "Updating strategy via function will not update the strategy file"
            )
            # local metadata
            with (self.path_local_preset / "metadata.json").open("w") as fs:
                json.dump(metadata, fs, default=str, indent=4)
            # local exports/stats.json
            with (self.path_local_preset / "exports" / "stats.json").open("w") as fs:
                json.dump(stats, fs, default=str, indent=4)
            # local exports/summary.txt
            with (self.path_local_preset / "exports" / "summary.txt").open("w") as fs:
                fs.write(summary)

        print(f"[BACKTEST FINISHED] Synced preset with name: {preset_name}")

    def _generate_metadata(self, stats: dict, name: str, current_date: str) -> dict:
        """Generate backtesting summary in dict / json format"""
        trades = DataFrame(deepcopy(stats["strategy"]["NotebookStrategy"]["trades"]))
        trades_summary = deepcopy(stats["strategy"]["NotebookStrategy"])
        current_date = get_readable_date()

        del_keys = [
            "trades",
            "locks",
            "best_pair",
            "worst_pair",
            "results_per_pair",
            "sell_reason_summary",
            "left_open_trades",
        ]
        for key in del_keys:
            del trades_summary[key]

        metadata = {
            "preset_name": name,
            "backtest_date": current_date.split("_")[0]
            + " "
            + current_date.split("_")[1].replace("-", ":"),
            "leverage": 1,
            "direction": "long",
            "is_hedging": False,
            "num_pairs": len(trades_summary["pairlist"]),
            "data_source": self.exchange,
            "win_rate": trades_summary["wins"] / trades_summary["total_trades"],
            "avg_profit_winners_abs": trades.loc[trades["profit_abs"] >= 0, "profit_abs"]
            .dropna()
            .mean(),
            "avg_profit_losers_abs": trades.loc[trades["profit_abs"] < 0, "profit_abs"]
            .dropna()
            .mean(),
            "sum_profit_winners_abs": trades.loc[trades["profit_abs"] >= 0, "profit_abs"]
            .dropna()
            .sum(),
            "sum_profit_losers_abs": trades.loc[trades["profit_abs"] < 0, "profit_abs"]
            .dropna()
            .sum(),
            "profit_mean_abs": trades_summary["profit_total_abs"] / trades_summary["total_trades"],
            "profit_per_drawdown": trades_summary["profit_total_abs"]
            / abs(trades_summary["max_drawdown_abs"]),
        }
        metadata.update(
            {
                "profit_factor": metadata["sum_profit_winners_abs"]
                / abs(metadata["sum_profit_losers_abs"]),
                "expectancy_abs": (
                    (metadata["win_rate"] * metadata["avg_profit_winners_abs"])
                    + ((1 - metadata["win_rate"]) * metadata["avg_profit_losers_abs"])
                ),
            }
        )

        for key, value in trades_summary.items():
            is_valid = any(
                [isinstance(value, it) for it in (str, int, float, bool)] + [value is None],
            )
            if not is_valid:
                trades_summary[key] = str(value)

        return {**metadata, **trades_summary}
