import logging
import os
import json
import rapidjson
import pandas as pd
import random
import wandb
import attr
import gc

from functools import wraps, cache
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Deque
from pandas import DataFrame
from pathlib import Path
from collections import deque

from freqtrade.configuration import remove_credentials, validate_config_consistency, TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data import history
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.optimize_reports import generate_backtest_stats, show_backtest_results
from freqtrade.persistence import PairLocks, Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.strategy.interface import IStrategy
from freqtrade.wallets import Wallets
from freqtrade.nbtools import constants
from freqtrade.constants import DATETIME_PRINT_FORMAT


from .helper import Capturing, get_class_from_string, parse_function_body, get_readable_date, log_execute_time, run_in_thread
from .preset import BasePreset, ConfigPreset, LocalPreset, CloudPreset
from .configuration import setup_optimize_configuration
from .remote_utils import cloud_retrieve_preset, preset_log, table_add_row


logger = logging.getLogger(__name__)


@attr.s
class DataLoader:
    """ Instantiate this on the top of notebook. This will save time freqtrade from
        loading the same 'n' datasets. To remove cached datasets, call `instance.clear()`
    """
    max_n_datasets: int = attr.ib(default=5)
    
    def __attrs_post_init__(self):
        self._loaded_datasets: Deque[Dict[int, Any]] = deque(maxlen=self.max_n_datasets)
        logger.info(f"Initialized DataLoader with {self.max_n_datasets} max datas.")

    @log_execute_time("Load BT Data")
    def load_data(self, 
                     datadir: Path,
                     timeframe: str,
                     pairs: List[str], *,
                     timerange: Optional[TimeRange] = None,
                     fill_up_missing: bool = True,
                     startup_candles: int = 0,
                     fail_without_data: bool = False,
                     data_format: str = 'json'
                     ) -> Any:
        
        _locals = deepcopy(locals())
        
        if timerange is not None:
            _locals["timerange"] = timerange.__dict__
        
        hashed = hash(str(_locals))
        old_hashes = [list(it.keys())[0] for it in self._loaded_datasets]
        
        if hashed in old_hashes:
            logger.info(f"DATALOADER: Dataset with hash `{hashed}` exists in cache!")
            return self._loaded_datasets[old_hashes.index(hashed)][hashed]
        
        logger.info(f"DATALOADER: Dataset with hash `{hashed}` doesn't exist. Loading from disk...")
        data = history.load_data(
            datadir=datadir,
            pairs=pairs,
            timeframe=timeframe,
            timerange=timerange,
            fill_up_missing=fill_up_missing,
            startup_candles=startup_candles,
            fail_without_data=fail_without_data,
            data_format=data_format
        )
        self._loaded_datasets.append({hashed: data})
        return data

    def clear(self):
        self._loaded_datasets.clear()
        gc.collect()


class NbBacktesting(Backtesting):
    
    def __init__(self, config: Dict[str, Any]) -> None:

        LoggingMixin.show_output = False
        self.config = config

        # Reset keys for backtesting
        remove_credentials(self.config)
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}

    
    @log_execute_time("Backtest")
    def start_nb_backtesting(self, parsed_strategy_code: str, clsname: str, dataloader: DataLoader) -> Tuple[dict, str]:
        """
        Run backtesting end-to-end
        :return: None
        """
        logger.info("Backtesting...")

        strategy = get_class_from_string(parsed_strategy_code, clsname)(self.config)
        strategy = process_strategy(strategy, self.config)
        self.strategylist.append(strategy)
        self.exchange = ExchangeResolver.load_exchange(self.config["exchange"]["name"], self.config)
        self.dataprovider = DataProvider(self.config, None)

        # No strategy list specified, only one strategy
        validate_config_consistency(self.config)

        if "timeframe" not in self.config:
            raise OperationalException(
                "Timeframe (ticker interval) needs to be set in either "
                "configuration or as cli argument `--timeframe 5m`"
            )
        self.timeframe = str(self.config.get("timeframe"))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)

        self.pairlists = PairListManager(self.exchange, self.config)
        if "VolumePairList" in self.pairlists.name_list:
            raise OperationalException("VolumePairList not allowed for backtesting.")
        if "PerformanceFilter" in self.pairlists.name_list:
            raise OperationalException("PerformanceFilter not allowed for backtesting.")

        if len(self.strategylist) > 1 and "PrecisionFilter" in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if self.config.get("fee", None) is not None:
            self.fee = self.config["fee"]
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])

        Trade.use_db = False
        Trade.reset_trades()
        PairLocks.timeframe = self.config["timeframe"]
        PairLocks.use_db = False
        PairLocks.reset_locks()

        self.wallets = Wallets(self.config, self.exchange, log=False)

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])

        data: Dict[str, Any] = {}

        data, timerange = self.dataloader_load_bt_data(dataloader)
        logger.info("Dataload complete. Calculating indicators")

        for strat in self.strategylist:
            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)
        if len(self.strategylist) > 0:
            stats = generate_backtest_stats(
                data, self.all_results, min_date=min_date, max_date=max_date
            )
            # Backtest results
            with Capturing() as print_text:
                show_backtest_results(self.config, stats)

        return (stats, "\n".join(print_text))

    def dataloader_load_bt_data(self, dataloader: DataLoader) -> Tuple[Dict[str, DataFrame], TimeRange]:
        """
        Loads backtest data and returns the data combined with the timerange
        as tuple.
        Modified: Cache loaded data
        """
        timerange = TimeRange.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))

        data = dataloader.load_data(
            datadir=self.config['datadir'],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
        )

        min_date, max_date = history.get_timerange(data)

        logger.info(f'Loading data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days).')

        # Adjust startts forward if not enough data is available
        timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe),
                                            self.required_startup, min_date)

        return data, timerange


def process_strategy(strategy: IStrategy, config: Dict[str, Any] = None) -> IStrategy:
    """
    Load the custom class from config parameter
    :param config: configuration dictionary or None
    """
    config = config or {}

    # make sure ask_strategy dict is available
    if "ask_strategy" not in config:
        config["ask_strategy"] = {}

    if hasattr(strategy, "ticker_interval") and not hasattr(strategy, "timeframe"):
        # Assign ticker_interval to timeframe to keep compatibility
        if "timeframe" not in config:
            logger.warning(
                "DEPRECATED: Please migrate to using 'timeframe' instead of 'ticker_interval'."
            )
            strategy.timeframe = strategy.ticker_interval

    # Set attributes
    # Check if we need to override configuration
    #             (Attribute name,                    default,     subkey)
    attributes = [
        ("minimal_roi", {"0": 10.0}, None),
        ("timeframe", None, None),
        ("stoploss", None, None),
        ("trailing_stop", None, None),
        ("trailing_stop_positive", None, None),
        ("trailing_stop_positive_offset", 0.0, None),
        ("trailing_only_offset_is_reached", None, None),
        ("use_custom_stoploss", None, None),
        ("process_only_new_candles", None, None),
        ("order_types", None, None),
        ("order_time_in_force", None, None),
        ("stake_currency", None, None),
        ("stake_amount", None, None),
        ("protections", None, None),
        ("startup_candle_count", None, None),
        ("unfilledtimeout", None, None),
        ("use_sell_signal", True, "ask_strategy"),
        ("sell_profit_only", False, "ask_strategy"),
        ("ignore_roi_if_buy_signal", False, "ask_strategy"),
        ("sell_profit_offset", 0.0, "ask_strategy"),
        ("disable_dataframe_checks", False, None),
        ("ignore_buying_expired_candle_after", 0, "ask_strategy"),
    ]
    for attribute, default, subkey in attributes:
        if subkey:
            StrategyResolver._override_attribute_helper(
                strategy, config.get(subkey, {}), attribute, default
            )
        else:
            StrategyResolver._override_attribute_helper(strategy, config, attribute, default)

    # Loop this list again to have output combined
    for attribute, _, subkey in attributes:
        if subkey and attribute in config[subkey]:
            logger.info("Strategy using %s: %s", attribute, config[subkey][attribute])
        elif attribute in config:
            logger.info("Strategy using %s: %s", attribute, config[attribute])

    StrategyResolver._normalize_attributes(strategy)

    StrategyResolver._strategy_sanity_validations(strategy)
    return strategy


@log_execute_time("Whole Backtesting Process (Backtest + Log)")
def backtest(preset: BasePreset, 
             strategy: Union[str, Callable[[], None]], 
             clsname: str = "NotebookStrategy",
             dataloader: Optional[DataLoader] = None,
             ):
    """ Start freqtrade backtesting. Callable in notebook.
        preset: Any kind of Preset (ConfigPreset, LocalPreset, CloudPreset)
        strategy: str or function that has strategy code
    """
    config_backtesting, config_optimize = preset.get_configs()
    
    if callable(strategy):
        parsed_strategy_code = parse_function_body(strategy)
    elif isinstance(strategy, str):
        parsed_strategy_code = strategy
    else:
        raise ValueError("Strategy must instance of Callable or plain String (from strategy code)")
    
    if dataloader is None:
        logger.warning(("WARNING: You are not using DataLoader." \
                       "Expect slow process when freqtrade loads the same BT data."))
        dataloader = DataLoader(max_n_datasets=0)
    
    backtester = NbBacktesting(config_optimize)
    stats, summary = backtester.start_nb_backtesting(parsed_strategy_code, clsname, dataloader)
    
    
    log_preset(preset, parsed_strategy_code, stats, config_backtesting, config_optimize)
    
    return stats, summary


@log_execute_time("Log Preset")
def log_preset(preset: BasePreset, strategy_code: str, stats: dict, config_backtesting: dict, config_optimize: dict):
    """ Upload preset to cloud WandB. """
    logger.info("Logging preset...")

    current_date = get_readable_date()
    preset_name = f"{preset.name}__backtest-{current_date}"
    metadata = generate_metadata(preset, stats, config_backtesting, config_optimize, current_date)
    stats["metadata"] = metadata
    
    filename_and_content = {
        "metadata.json": metadata,
        "config-backtesting.json": config_backtesting,
        "config-optimize.json": config_optimize,
        "exports/stats.json": stats,
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

    wandb_log(preset_name, metadata)

    if isinstance(preset, LocalPreset):
        logger.info(f"You are backtesting a local preset `{preset.path_local_preset}`")
        logger.info("This will update backtest results (such as metadata.json, exports)")
        logger.info("Updating strategy via function will not update the strategy file")
        # local metadata
        with (preset.path_local_preset / "metadata.json").open("w") as fs:
            json.dump(metadata, fs, default=str, indent=4)
        # local exports/stats.json
        with (preset.path_local_preset / "exports" / "stats.json").open("w") as fs:
            json.dump(stats, fs, default=str, indent=4)

    logger.info("[LOG PRESET OFFLINE SUCCESS]")
    logger.info(f"Sync preset with name: {preset_name}")
    logger.info(f"with random name: {metadata['random_name']}")
    logger.info("[WANDB LOG PRESET CONTINUES IN BACKGROUND]")


@run_in_thread
def wandb_log(preset_name: str, metadata: dict):
    with wandb.init(project=constants.PROJECT_NAME_PRESETS) as run:
        preset_log(run, f"./.temp/{preset_name}", preset_name)
        table_add_row(
            run,
            metadata,
            constants.PROJECT_NAME_PRESETS,
            constants.PRESETS_ARTIFACT_METADATA,
            constants.PRESETS_TABLEKEY_METADATA,
        )
    logger.info("===============================")
    logger.info("|                             |")
    logger.info("|  WANDB LOG PRESET FINISHED  |")
    logger.info("|                             |")
    logger.info("===============================")
    


def generate_metadata(
    preset: BasePreset, 
    stats: Dict[str, Any], 
    config_backtesting: dict, 
    config_optimize: dict, 
    current_date: str) -> Dict[str, Any]:
    """Generate backtest summary in dict to be exported in json format"""

    strats = list(stats["strategy"].keys())
    if len(strats) > 1:
        raise ValueError(f"Got strats `{strats}`")
    clsname = strats[0]
        
    trades = pd.DataFrame(deepcopy(stats["strategy"][clsname]["trades"]))
    trades_summary = deepcopy(stats["strategy"][clsname])
    current_date_fmt = current_date.split("_")[0] + " " + current_date.split("_")[1].replace("-", ":")
    
    # You can add or remove any columns you want here
    metadata = {
        "random_name": get_random_name(),
        "preset_name": f"{preset.name}__backtest-{current_date}",
        "preset_type": preset.__class__.__name__,
        "backtest_date": current_date_fmt,
        "leverage": 1,  # TODO
        "direction": "long",  # TODO
        "is_hedging": False,  # TODO
        "fee": config_optimize["fee"],
        "num_pairs": len(trades_summary["pairlist"]),
        "data_source": config_backtesting["exchange"]["name"],
        "win_rate": trades_summary["wins"] / trades_summary["total_trades"],
        "avg_profit_winners_abs": trades.loc[trades["profit_abs"] >= 0, "profit_abs"].dropna().mean(),
        "avg_profit_losers_abs": trades.loc[trades["profit_abs"] < 0, "profit_abs"].dropna().mean(),
        "sum_profit_winners_abs": trades.loc[trades["profit_abs"] >= 0, "profit_abs"].dropna().sum(),
        "sum_profit_losers_abs": trades.loc[trades["profit_abs"] < 0, "profit_abs"].dropna().sum(),
        "profit_mean_abs": trades_summary["profit_total_abs"] / trades_summary["total_trades"],
        "profit_per_drawdown": trades_summary["profit_total_abs"] / abs(trades_summary["max_drawdown_abs"]),
    }
    # Needs extra calculation involves previous metadata
    metadata.update(
        {
            "profit_factor": metadata["sum_profit_winners_abs"] / abs(metadata["sum_profit_losers_abs"]),
            "expectancy_abs": (
                (metadata["win_rate"] * metadata["avg_profit_winners_abs"])
                + ((1 - metadata["win_rate"]) * metadata["avg_profit_losers_abs"])
            ),
        }
    )

    # Filter out "too long for table" data
    for key in list(trades_summary):
        value = trades_summary[key]
        
        is_valid_type = any(
            [isinstance(value, it) for it in (str, int, float, bool)] + [value is None],
        )
        
        if not is_valid_type:
            trades_summary[key] = str(value)
            
            if len(trades_summary[key]) > 30:
                del trades_summary[key]

    return {**metadata, **trades_summary}


def get_random_name() -> str:
    """Generate memorizable name for backtest results"""
    list_of_possible_names_1 = [
        "rage", "happy", "sad", "sick", "angry", "depressed", "broken", "boring", "satisfied",
        "disgusted", "fearful", "hardworking", "loving", "destroyed", "bipolar", "dissatisfied",
        "insane", "furious", "mad", "sane", "healthy", "friendly", "unfriendly", "introvert",
        "extrovert", "silly", "smart", "dumb", "sadistic", "sociable", "unsociable", "lazy",
    ]
    list_of_possible_names_2 = [
        "kirito", "asuna", "alice", "vector", "yoshiko", "kazuya", "rei", "phantom",
        "gabriel", "vegeta", "frieza", "nobita", "vivy", "thor", "ironman", "captain",
        "spiderman", "superman", "piccolo", "goku", "naruto", "sasuke", "midoriya",
        "crypto", "todoroki", "batman", "thanos", "doraemon", "shinchan", "covid",
    ]
    list_of_possible_names_3 = [
        "table", "speaker", "remote", "monitor", "keyboard", "camera", "smartphone",
        "phone", "virus", "mask", "computer", "cpu", "ram", "memory", "eyeglasses",
        "sanitizer", "charger", "cable", "sticker", "sword", "armor", "necklace", "shield",
        "adapter", "electricity", "bulb", "laptop", "desktop", "mouse", "drug",
    ]
    random_name = (
        random.choice(list_of_possible_names_1) +  "-" +
        random.choice(list_of_possible_names_2) +  "-" +
        random.choice(list_of_possible_names_3)
    )
    return random_name