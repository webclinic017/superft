from typing import List, Union
from datetime import datetime

import pandas as pd

from freqtrade.ml.container import LightningContainer
from freqtrade.ml.lightning import LightningModule


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


def get_dataset_df(container: Union[LightningContainer, LightningModule]) -> pd.DataFrame:
    if not isinstance(container, LightningContainer):
        return LightningContainer(container)._load_df_allpairs()
    return container._load_df_allpairs()