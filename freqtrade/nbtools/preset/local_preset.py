from pathlib import Path
from typing import Tuple
import attr
import rapidjson
import logging


from .base_preset import BasePreset


logger = logging.getLogger(__name__)


@attr.s
class LocalPreset(BasePreset):
    path_local_preset: Path = attr.ib()
    name: str = attr.ib(init=False)
    default_strategy_code: str = attr.ib(init=False)

    def __attrs_post_init__(self):
        logger.debug(f"Preparing LocalPreset for `{self.path_local_preset}`")
        logger.debug(f"Using path local preset: {self.path_local_preset}")

        with (self.path_local_preset / "strategies" / "strategy.py").open("r") as fs:
            self.default_strategy_code = fs.read()
            logger.debug(f"Detected default strategy with {len(self.default_strategy_code.splitlines())} lines")
        
        self.name = self.path_local_preset.name

    def get_configs(self) -> Tuple[dict, dict]:
        """ Returns (config_backtesting, config_optimize)
        """
        with (self.path_local_preset / "config-backtesting.json").open("r") as fs:
            config_backtesting = rapidjson.load(fs)
        
        config_optimize = self.get_config_optimize(config_backtesting)
        return config_backtesting, config_optimize