from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path
import attr
import logging

from freqtrade.enums.runmode import RunMode
from freqtrade.nbtools.configuration import setup_optimize_configuration


logger = logging.getLogger(__name__)


@attr.s
class BasePreset(ABC):
    """ Preset to be synced every backtest.
        Optional parameters:
        - pairs: List[str]
        - exchange: str
        - starting_balance: int
        - stake_amount: int
        - max_open_trades: int
        - fee: float
        You can overwrite them using preset.overwrite_config()
    """
    # Basic Parameters
    name: str = attr.ib()
    path_data: Path = attr.ib()
    # Must Input Backtesting Parameters
    timerange: str = attr.ib()
    # Optional but Customizable Parameters. Overwrites the cloud config but not local.
    pairs: Optional[List[str]] = attr.ib(init=False)
    exchange: Optional[str] = attr.ib(init=False)
    starting_balance: Optional[float] = attr.ib(init=False)
    stake_amount: Optional[float] = attr.ib(init=False)
    max_open_trades: Optional[int] = attr.ib(init=False)
    fee: Optional[float] = attr.ib(init=False)
    strategy_search_path: Optional[Path] = attr.ib(init=False)

    def __attrs_pre_init__(self):
        setattr(self, "pairs", None)
        setattr(self, "exchange", None)
        setattr(self, "starting_balance", None)
        setattr(self, "stake_amount", None)
        setattr(self, "max_open_trades", None)
        setattr(self, "fee", None)
        setattr(self, "strategy_search_path", None)

    def get_config_optimize(self, config_backtesting: dict) -> dict:
        """ Overwrite config_backtesting (if any overwrites) then get the configuration 
            for the optimize module
        """
        logger.info(f"Setting config for {self.name} ...")
        
        if self.pairs is not None:
            logger.info(
                f"Overwriting pairs (from {len(config_backtesting['exchange']['pair_whitelist'])} to {len(self.pairs)} pairs)"
            )
            config_backtesting["exchange"]["pair_whitelist"] = self.pairs
        
        if self.exchange is not None:
            logger.info(f"Overwriting exchange from {config_backtesting['exchange']['name']} to {self.exchange}")
            config_backtesting["exchange"]["name"] = self.exchange
        
        if self.starting_balance is not None:
            logger.info(f"Overwriting starting balance from {config_backtesting['dry_run_wallet']} to {self.starting_balance}")
            config_backtesting.update({"dry_run_wallet": self.starting_balance})
        
        if self.stake_amount is not None:
            logger.info(f"Overwriting stake amount from {config_backtesting['stake_amount']} to {self.stake_amount}")
            config_backtesting.update({"stake_amount": self.stake_amount})
        
        if self.max_open_trades is not None:
            logger.info(f"Overwriting max open trades from {config_backtesting['max_open_trades']} to {self.max_open_trades}")
            config_backtesting.update({"max_open_trades": self.max_open_trades})
        
        if self.fee is not None:
            logger.info(f"Overwriting max open trades from {config_backtesting['fee']} to {self.fee}")
            config_backtesting.update({"fee": self.fee})

        if self.strategy_search_path is not None:
            logger.info(f"Add strategy search path {self.strategy_search_path}")
            config_backtesting.update({"strategy_path": self.strategy_search_path})

        args = {
            "datadir": self.path_data / config_backtesting["exchange"]["name"],
            "timerange": self.timerange,
        }
        
        logger.info(f"Setting arg `datadir` to {args['datadir']}")
        logger.info(f"Setting arg `timerange` to {args['timerange']}")

        return setup_optimize_configuration(config_backtesting, args, RunMode.BACKTEST)

    def overwrite_config(self, 
                         pairs: Optional[List[str]] = None,
                         exchange: Optional[str] = None,
                         starting_balance: Optional[float] = None,
                         stake_amount: Optional[float] = None,
                         max_open_trades: Optional[int] = None,
                         fee: Optional[float] = None,
                         strategy_search_path: Optional[Path] = None
                         ):
        # Loop through this function args, set key if not None
        for key, value in locals().items():
            if value is not None:
                setattr(self, key, value)

    @abstractmethod
    def get_configs(self) -> Tuple[dict, dict]:
        """ Returns (config_backtesting, config_optimize)
        """
        raise NotImplementedError()
