from typing import List
from datetime import datetime

from freqtrade.ml.container import LightningContainer


def _get_timerange(container: LightningContainer, attrib: str) -> str:
    start: datetime = getattr(container.module.config, f"{attrib}_start")
    end: datetime = getattr(container.module.config, f"{attrib}_end")
    start_str = datetime.strftime(start, '%Y%m%d')
    end_str = datetime.strftime(end, '%Y%m%d')
    return f'{start_str}-{end_str}'


def get_timerange_trainval(container: LightningContainer) -> str:
    return _get_timerange(container, "trainval")


def get_timerange_opt(container: LightningContainer) -> str:
    return _get_timerange(container, "opt")


def get_timerange_test(container: LightningContainer) -> str:
    return _get_timerange(container, "test")