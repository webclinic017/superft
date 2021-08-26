from typing import Tuple
from copy import deepcopy
import attr
import logging

from .base_preset import BasePreset


logger = logging.getLogger(__name__)


@attr.s
class ConfigPreset(BasePreset):
    config_dict: dict = attr.ib()
    
    def __attrs_post_init__(self):
        self._config = deepcopy(self.config_dict)