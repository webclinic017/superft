from typing import Tuple
from pathlib import Path
import attr
import rapidjson
import logging

from freqtrade.nbtools.remote_utils import cloud_retrieve_preset, preset_log, table_add_row
from .base_preset import BasePreset


logger = logging.getLogger(__name__)


@attr.s
class CloudPreset(BasePreset):
    default_strategy_code: str = attr.ib(type=str, init=False)
    path_to_preset: Path = attr.ib(type=Path, init=False)

    def __attrs_post_init__(self):
        logger.debug(f"Preparing CloudPreset for `{self.name}`")
        
        preset_path = cloud_retrieve_preset(self.name)
        preset_path = Path.cwd() / preset_path

        with (preset_path / "strategies" / "strategy.py").open("r") as fs:
            self.default_strategy_code = fs.read()
            logger.debug(f"Detected default strategy with {len(self.default_strategy_code.splitlines())} lines")

        self.name = self.name.split("__")[0]

    def get_configs(self) -> Tuple[dict, dict]:
        """ Returns (config_backtesting, config_optimize)
        """
        with (self.path_to_preset / "config-backtesting.json").open("r") as fs:
            config_backtesting = rapidjson.load(fs)

        config_optimize = self.get_config_optimize(config_backtesting)
        return config_backtesting, config_optimize