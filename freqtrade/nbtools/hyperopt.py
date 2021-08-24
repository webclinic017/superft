from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Deque

from freqtrade.nbtools.helper import (
    Capturing, get_class_from_string, parse_function_body, get_readable_date, log_execute_time, run_in_thread
)
from freqtrade.nbtools.preset import BasePreset, ConfigPreset, LocalPreset, CloudPreset
from freqtrade.nbtools.configuration import setup_optimize_configuration
from freqtrade.nbtools.remote_utils import cloud_retrieve_preset, preset_log, table_add_row
from freqtrade.nbtools import constants


def hyperopt(preset: BasePreset, 
             strategy: Union[str, Callable[[], None]], 
             clsname: str = "NotebookStrategy"):
    pass