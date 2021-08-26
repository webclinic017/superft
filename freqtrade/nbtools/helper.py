from importlib import util as importlib_util
from io import StringIO
from itertools import dropwhile
from pathlib import Path
from typing import Any, Dict, Union, Callable
from functools import wraps

import inspect
import json
import re
import sys
import arrow
import logging
import time
import gc
import threading

from freqtrade.strategy.interface import IStrategy
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.enums.runmode import RunMode


logger = logging.getLogger(__name__)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout



def get_preset_strategy(preset, config_optimize: dict, clsname: str) -> IStrategy:
    from freqtrade.nbtools.preset import FilePreset
    
    if isinstance(preset, FilePreset):
        return process_strategy(
            StrategyResolver._load_strategy(clsname, config_optimize, extra_dir=str(preset.path_to_file.parent)),
            config=config_optimize
        )
    raise Exception("Must use FilePreset!")


def get_class_from_string(code: str, clsname: str) -> Any:
    my_name = "stratmodule"
    my_spec = importlib_util.spec_from_loader(my_name, loader=None)

    if my_spec is None:
        raise ImportError(f"Module {my_name} not found")

    stratmodule = importlib_util.module_from_spec(my_spec)
    exec(code, stratmodule.__dict__)
    sys.modules["stratmodule"] = stratmodule
    return getattr(stratmodule, clsname)


def parse_function_body(func) -> str:
    source_lines = inspect.getsource(func)
    source_lines = source_lines.splitlines(keepends=True)
    source_lines = dropwhile(lambda x: x.startswith("@"), source_lines)
    source = "".join(source_lines)
    pattern = re.compile(r"(async\s+)?def\s+\w+\s*\(.*?\)\s*:\s*(.*)", flags=re.S)
    lines = pattern.search(source).group(2).splitlines()
    if len(lines) == 1:
        return lines[0]
    indentation = len(lines[1]) - len(lines[1].lstrip())
    return "\n".join([lines[0]] + [line[indentation:] for line in lines[1:]])


def parse_strategy_code(strategy: Union[str, Callable[[], None]]) -> str:
    if callable(strategy):
        parsed_strategy_code = parse_function_body(strategy)
    elif isinstance(strategy, str):
        parsed_strategy_code = strategy
    else:
        raise ValueError("Strategy must instance of Callable or plain String (from strategy code)")
    return parsed_strategy_code


def get_strategy_object(parsed_strategy_code, config_optimize, clsname) -> IStrategy:
    strategy_object = get_class_from_string(parsed_strategy_code, clsname)(config_optimize)
    strategy_object = process_strategy(strategy_object, config_optimize)
    return strategy_object


def process_strategy(strategy: IStrategy, config: Dict[str, Any]) -> IStrategy:
    """
    Load the custom class from config parameter
    :param config: configuration dictionary or None
    """
    strategy._populate_fun_len = len(inspect.getfullargspec(strategy.populate_indicators).args)
    strategy._buy_fun_len = len(inspect.getfullargspec(strategy.populate_buy_trend).args)
    strategy._sell_fun_len = len(inspect.getfullargspec(strategy.populate_sell_trend).args)
    if any(x == 2 for x in [strategy._populate_fun_len,
                            strategy._buy_fun_len,
                            strategy._sell_fun_len]):
        strategy.INTERFACE_VERSION = 1

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

    if strategy._ft_params_from_file:
        # Set parameters from Hyperopt results file
        params = strategy._ft_params_from_file
        strategy.minimal_roi = params.get('roi', strategy.minimal_roi)

        strategy.stoploss = params.get('stoploss', {}).get('stoploss', strategy.stoploss)
        trailing = params.get('trailing', {})
        strategy.trailing_stop = trailing.get('trailing_stop', strategy.trailing_stop)
        strategy.trailing_stop_positive = trailing.get('trailing_stop_positive',
                                                        strategy.trailing_stop_positive)
        strategy.trailing_stop_positive_offset = trailing.get(
            'trailing_stop_positive_offset', strategy.trailing_stop_positive_offset)
        strategy.trailing_only_offset_is_reached = trailing.get(
            'trailing_only_offset_is_reached', strategy.trailing_only_offset_is_reached)

    # Set attributes
    # Check if we need to override configuration
    #             (Attribute name,                    default,     subkey)
    attributes = [("minimal_roi",                     {"0": 10.0}),
                    ("timeframe",                       None),
                    ("stoploss",                        None),
                    ("trailing_stop",                   None),
                    ("trailing_stop_positive",          None),
                    ("trailing_stop_positive_offset",   0.0),
                    ("trailing_only_offset_is_reached", None),
                    ("use_custom_stoploss",             None),
                    ("process_only_new_candles",        None),
                    ("order_types",                     None),
                    ("order_time_in_force",             None),
                    ("stake_currency",                  None),
                    ("stake_amount",                    None),
                    ("protections",                     None),
                    ("startup_candle_count",            None),
                    ("unfilledtimeout",                 None),
                    ("use_sell_signal",                 True),
                    ("sell_profit_only",                False),
                    ("ignore_roi_if_buy_signal",        False),
                    ("sell_profit_offset",              0.0),
                    ("disable_dataframe_checks",        False),
                    ("ignore_buying_expired_candle_after",  0)
                    ]
    for attribute, default in attributes:
        StrategyResolver._override_attribute_helper(strategy, config,
                                                    attribute, default)

    # Loop this list again to have output combined
    for attribute, _ in attributes:
        if attribute in config:
            logger.info("Strategy using %s: %s", attribute, config[attribute])

    StrategyResolver._normalize_attributes(strategy)

    StrategyResolver._strategy_sanity_validations(strategy)
    return strategy


def write_str(path: Path, content: str, temp=False):
    with open(path, "w") as stream:
        stream.write(content)


def write_json(path: Path, content: dict, temp=False):
    with open(path, "w") as stream:
        json.dump(content, stream)


def get_readable_date() -> str:
    return arrow.utcnow().shift(hours=7).strftime("%Y-%m-%d_%H-%M-%S")


def free_mem(var):
    del var
    gc.collect()
    

def log_execute_time(name: str = None):
    def somedec_outer(fn):
        @wraps(fn)
        def somedec_inner(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                end = time.time() - start
                logger.info('"{}" executed in {:.2f}s'.format(
                    name if name is not None else function.__name__, end
                ))
        return somedec_inner
    return somedec_outer

    
def run_in_thread(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper