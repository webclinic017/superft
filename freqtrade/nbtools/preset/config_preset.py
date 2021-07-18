from typing import Tuple
from copy import deepcopy
import attr
import logging

from .base_preset import BasePreset


logger = logging.getLogger(__name__)


@attr.s
class ConfigPreset(BasePreset):
    config_dict: dict = attr.ib()
   
    def get_configs(self) -> Tuple[dict, dict]:
        """ Returns (config_backtesting, config_optimize)
        """
        logger.debug(f"Preparing ConfigPreset for `{self.name}`")
        config_backtesting = deepcopy(self.config_dict)
        config_optimize = self.get_config_optimize(config_backtesting)
        return (config_backtesting, config_optimize)