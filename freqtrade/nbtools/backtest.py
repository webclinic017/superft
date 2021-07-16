from copy import deepcopy
from typing import *
from pandas import DataFrame
import logging

from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.optimize_reports import (generate_backtest_stats, show_backtest_results, store_backtest_stats)
from freqtrade.mixins import LoggingMixin
from freqtrade.configuration import remove_credentials, validate_config_consistency
from freqtrade.strategy.interface import IStrategy
from freqtrade.wallets import Wallets
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.exchange.exchange import timeframe_to_minutes
from freqtrade.exceptions import OperationalException
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.persistence import PairLocks, Trade
from .helper import Capturing, get_class_from_string

logger = logging.getLogger(__name__)


class NbBacktesting(Backtesting):
    
    def __init__(self, config: Dict[str, Any]) -> None:
        
        LoggingMixin.show_output = False
        self.config = config

        # Reset keys for backtesting
        remove_credentials(self.config)
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}
    
    def start_nb_backtesting(self, strategy_code: str) -> Tuple[dict, str]:
        """
        Run backtesting end-to-end
        :return: None
        """
        strategy = get_class_from_string(strategy_code, "NotebookStrategy")(self.config)
        strategy = process_strategy(strategy, self.config)
        self.strategylist.append(strategy)
        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
        self.dataprovider = DataProvider(self.config, None)
        
        # No strategy list specified, only one strategy
        validate_config_consistency(self.config)

        if "timeframe" not in self.config:
            raise OperationalException("Timeframe (ticker interval) needs to be set in either "
                                       "configuration or as cli argument `--timeframe 5m`")
        self.timeframe = str(self.config.get('timeframe'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)

        self.pairlists = PairListManager(self.exchange, self.config)
        if 'VolumePairList' in self.pairlists.name_list:
            raise OperationalException("VolumePairList not allowed for backtesting.")
        if 'PerformanceFilter' in self.pairlists.name_list:
            raise OperationalException("PerformanceFilter not allowed for backtesting.")

        if len(self.strategylist) > 1 and 'PrecisionFilter' in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if self.config.get('fee', None) is not None:
            self.fee = self.config['fee']
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])

        Trade.use_db = False
        Trade.reset_trades()
        PairLocks.timeframe = self.config['timeframe']
        PairLocks.use_db = False
        PairLocks.reset_locks()

        self.wallets = Wallets(self.config, self.exchange, log=False)

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        
        data: Dict[str, Any] = {}

        data, timerange = self.load_bt_data()
        logger.info("Dataload complete. Calculating indicators")

        for strat in self.strategylist:
            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)
        if len(self.strategylist) > 0:
            stats = generate_backtest_stats(data, self.all_results,
                                            min_date=min_date, max_date=max_date)
            # Backtest results
            with Capturing() as print_text:
                show_backtest_results(self.config, stats)
            
        return (stats, "\n".join(print_text))
    
    
def process_strategy(strategy: IStrategy, config: Dict[str, Any] = None) -> IStrategy:
    """
    Load the custom class from config parameter
    :param config: configuration dictionary or None
    """
    config = config or {}

    # make sure ask_strategy dict is available
    if 'ask_strategy' not in config:
        config['ask_strategy'] = {}

    if hasattr(strategy, 'ticker_interval') and not hasattr(strategy, 'timeframe'):
        # Assign ticker_interval to timeframe to keep compatibility
        if 'timeframe' not in config:
            logger.warning(
                "DEPRECATED: Please migrate to using 'timeframe' instead of 'ticker_interval'."
                )
            strategy.timeframe = strategy.ticker_interval

    # Set attributes
    # Check if we need to override configuration
    #             (Attribute name,                    default,     subkey)
    attributes = [("minimal_roi",                     {"0": 10.0}, None),
                    ("timeframe",                       None,        None),
                    ("stoploss",                        None,        None),
                    ("trailing_stop",                   None,        None),
                    ("trailing_stop_positive",          None,        None),
                    ("trailing_stop_positive_offset",   0.0,         None),
                    ("trailing_only_offset_is_reached", None,        None),
                    ("use_custom_stoploss",             None,        None),
                    ("process_only_new_candles",        None,        None),
                    ("order_types",                     None,        None),
                    ("order_time_in_force",             None,        None),
                    ("stake_currency",                  None,        None),
                    ("stake_amount",                    None,        None),
                    ("protections",                     None,        None),
                    ("startup_candle_count",            None,        None),
                    ("unfilledtimeout",                 None,        None),
                    ("use_sell_signal",                 True,        'ask_strategy'),
                    ("sell_profit_only",                False,       'ask_strategy'),
                    ("ignore_roi_if_buy_signal",        False,       'ask_strategy'),
                    ("sell_profit_offset",              0.0,         'ask_strategy'),
                    ("disable_dataframe_checks",        False,       None),
                    ("ignore_buying_expired_candle_after",  0,       'ask_strategy')
                    ]
    for attribute, default, subkey in attributes:
        if subkey:
            StrategyResolver._override_attribute_helper(strategy, config.get(subkey, {}),
                                                        attribute, default)
        else:
            StrategyResolver._override_attribute_helper(strategy, config,
                                                        attribute, default)

    # Loop this list again to have output combined
    for attribute, _, subkey in attributes:
        if subkey and attribute in config[subkey]:
            logger.info("Strategy using %s: %s", attribute, config[subkey][attribute])
        elif attribute in config:
            logger.info("Strategy using %s: %s", attribute, config[attribute])

    StrategyResolver._normalize_attributes(strategy)

    StrategyResolver._strategy_sanity_validations(strategy)
    return strategy