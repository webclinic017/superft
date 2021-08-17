from pathlib import Path
from typing import Tuple
from copy import deepcopy
import attr
import rapidjson
import logging

from .base_preset import BasePreset


logger = logging.getLogger(__name__)


@attr.s
class FilePreset(BasePreset):
    config_dict: dict = attr.ib()
    path_to_file: Path = attr.ib()
    name: str = attr.ib(init=False)
    default_strategy_code: str = attr.ib(init=False)
    
    def __attrs_post_init__(self):
        logger.debug(f"Preparing FilePreset in `{self.path_to_file}`")
        
        with self.path_to_file.open("r") as fs:
            self.default_strategy_code = fs.read()
            logger.debug(f"Detected strategy with {len(self.default_strategy_code.splitlines())} lines")
        
        self.name = self.path_to_file.name.replace(".py", "")
        print(f"Preset name: {self.name}")
    
    def get_configs(self) -> Tuple[dict, dict]:
        """ Returns (config_backtesting, config_optimize)
        """
        
        config_backtesting = deepcopy(self.config_dict)
        config_optimize = self.get_config_optimize(config_backtesting)
        return (config_backtesting, config_optimize)