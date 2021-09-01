"""
This module contains the notebook hyperopt logic
"""

from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Deque

import logging
import random
import warnings
import progressbar
import rapidjson
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional


from colorama import Fore, Style
from colorama import init as colorama_init
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from pandas import DataFrame

from freqtrade.constants import DATETIME_PRINT_FORMAT, FTHYPT_FILEVERSION, LAST_BT_RESULT_FN
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.misc import deep_merge_dicts, file_dump_json, plural
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.optimize.hyperopt_tools import HyperoptTools, hyperopt_serializer
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver, HyperOptResolver
from freqtrade.commands.optimize_commands import start_hyperopt
from freqtrade.enums.runmode import RunMode
from freqtrade.strategy.interface import IStrategy


from freqtrade.nbtools.helper import (
    Capturing, get_class_from_string, parse_function_body, get_readable_date, log_execute_time, run_in_thread,
    parse_strategy_code, get_strategy_object, get_preset_strategy
)
from freqtrade.nbtools.preset import BasePreset, FilePreset
from freqtrade.nbtools.backtest import NbBacktesting
from freqtrade.nbtools import constants


# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension

progressbar.streams.wrap_stderr()
progressbar.streams.wrap_stdout()
logger = logging.getLogger(__name__)


INITIAL_POINTS = 30

# Keep no more than SKOPT_MODEL_QUEUE_SIZE models
# in the skopt model queue, to optimize memory consumption
SKOPT_MODEL_QUEUE_SIZE = 10

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization


class NbHyperopt(Hyperopt):
    """Hyperopt class override to support freqtrade presets via start_hyperopt()"""

    def __init__(self, config: Dict[str, Any], strategy_object: IStrategy) -> None:
        self.buy_space: List[Dimension] = []
        self.sell_space: List[Dimension] = []
        self.roi_space: List[Dimension] = []
        self.stoploss_space: List[Dimension] = []
        self.trailing_space: List[Dimension] = []
        self.dimensions: List[Dimension] = []

        self.config = config

        self.backtesting = NbBacktesting(self.config, strategies=[strategy_object])

        if not self.config.get('hyperopt'):
            self.custom_hyperopt = HyperOptAuto(self.config)
            self.auto_hyperopt = True
        else:
            self.custom_hyperopt = HyperOptResolver.load_hyperopt(self.config)
            self.auto_hyperopt = False

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        strategy = str(self.config['strategy'])
        self.results_file: Path = (self.config['user_data_dir'] / 'hyperopt_results' /
                                   f'strategy_{strategy}_{time_now}.fthypt')
        self.data_pickle_file = (self.config['user_data_dir'] /
                                 'hyperopt_results' / 'hyperopt_tickerdata.pkl')
        self.total_epochs = config.get('epochs', 0)

        self.current_best_loss = 100

        self.clean_hyperopt()

        self.num_epochs_saved = 0
        self.current_best_epoch: Optional[Dict[str, Any]] = None

        # Populate functions here (hasattr is slow so should not be run during "regular" operations)
        if hasattr(self.custom_hyperopt, 'populate_indicators'):
            self.backtesting.strategy.advise_indicators = (  # type: ignore
                self.custom_hyperopt.populate_indicators)  # type: ignore
        if hasattr(self.custom_hyperopt, 'populate_buy_trend'):
            self.backtesting.strategy.advise_buy = (  # type: ignore
                self.custom_hyperopt.populate_buy_trend)  # type: ignore
        if hasattr(self.custom_hyperopt, 'populate_sell_trend'):
            self.backtesting.strategy.advise_sell = (  # type: ignore
                self.custom_hyperopt.populate_sell_trend)  # type: ignore

        # Use max_open_trades for hyperopt as well, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            self.max_open_trades = self.config['max_open_trades']
        else:
            logger.debug('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            self.max_open_trades = 0
        self.position_stacking = self.config.get('position_stacking', False)

        if HyperoptTools.has_space(self.config, 'sell'):
            # Make sure use_sell_signal is enabled
            if 'ask_strategy' not in self.config:
                self.config['ask_strategy'] = {}
            self.config['ask_strategy']['use_sell_signal'] = True

        self.print_all = self.config.get('print_all', False)
        self.hyperopt_table_header = 0
        self.print_colorized = self.config.get('print_colorized', False)
        self.print_json = self.config.get('print_json', False)

    def start_nb_hyperopt(self) -> None:
        self.random_state = self._set_random_state(self.config.get('hyperopt_random_state', None))
        logger.info(f"Using optimizer random state: {self.random_state}")
        self.hyperopt_table_header = -1
        # Initialize spaces ...
        self.init_spaces()

        self.prepare_hyperopt_data()

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None  # type: ignore
        self.backtesting.exchange._api_async = None  # type: ignore
        # self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        config_jobs = self.config.get('hyperopt_jobs', -1)
        logger.info(f'Number of parallel jobs set as: {config_jobs}')

        self.opt = self.get_optimizer(self.dimensions, config_jobs)

        if self.print_colorized:
            colorama_init(autoreset=True)

        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f'Effective number of parallel workers used: {jobs}')

                # Define progressbar
                if self.print_colorized:
                    widgets = [
                        ' [Epoch ', progressbar.Counter(), ' of ', str(self.total_epochs),
                        ' (', progressbar.Percentage(), ')] ',
                        progressbar.Bar(marker=progressbar.AnimatedMarker(
                            fill='\N{FULL BLOCK}',
                            fill_wrap=Fore.GREEN + '{}' + Fore.RESET,
                            marker_wrap=Style.BRIGHT + '{}' + Style.RESET_ALL,
                        )),
                        ' [', progressbar.ETA(), ', ', progressbar.Timer(), ']',
                    ]
                else:
                    widgets = [
                        ' [Epoch ', progressbar.Counter(), ' of ', str(self.total_epochs),
                        ' (', progressbar.Percentage(), ')] ',
                        progressbar.Bar(marker=progressbar.AnimatedMarker(
                            fill='\N{FULL BLOCK}',
                        )),
                        ' [', progressbar.ETA(), ', ', progressbar.Timer(), ']',
                    ]
                with progressbar.ProgressBar(
                         max_value=self.total_epochs, redirect_stdout=False, redirect_stderr=False,
                         widgets=widgets
                     ) as pbar:
                    EVALS = ceil(self.total_epochs / jobs)
                    for i in range(EVALS):
                        # Correct the number of epochs to be processed for the last
                        # iteration (should not exceed self.total_epochs in total)
                        n_rest = (i + 1) * jobs - self.total_epochs
                        current_jobs = jobs - n_rest if n_rest > 0 else jobs

                        asked = self.opt.ask(n_points=current_jobs)
                        f_val = self.run_optimizer_parallel(parallel, asked, i)
                        self.opt.tell(asked, [v['loss'] for v in f_val])

                        # Calculate progressbar outputs
                        for j, val in enumerate(f_val):
                            # Use human-friendly indexes here (starting from 1)
                            current = i * jobs + j + 1
                            val['current_epoch'] = current
                            val['is_initial_point'] = current <= INITIAL_POINTS

                            logger.debug(f"Optimizer epoch evaluated: {val}")

                            is_best = HyperoptTools.is_best_loss(val, self.current_best_loss)
                            # This value is assigned here and not in the optimization method
                            # to keep proper order in the list of results. That's because
                            # evaluations can take different time. Here they are aligned in the
                            # order they will be shown to the user.
                            val['is_best'] = is_best
                            self.print_results(val)

                            if is_best:
                                self.current_best_loss = val['loss']
                                self.current_best_epoch = val

                            self._save_result(val)

                            pbar.update(current)

        except KeyboardInterrupt:
            print('User interrupted..')

        logger.info(f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
                    f"saved to '{self.results_file}'.")

        if self.current_best_epoch:
            if self.auto_hyperopt:
                HyperoptTools.try_export_params(
                    self.config,
                    self.backtesting.strategy.get_strategy_name(),
                    self.current_best_epoch)

            HyperoptTools.show_epoch_details(self.current_best_epoch, self.total_epochs,
                                             self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")


def start_hyperopt(preset: BasePreset, 
                   hyperopt_args: dict,
                   clsname: str = "NotebookStrategy"):
    
    if not isinstance(preset, FilePreset):
        raise Exception("Only FilePreset support hyperopt.")
    
    from filelock import FileLock, Timeout
    
    print("Starting HyperOpt")
    
    config_optimize_hyperopt = preset.get_config_optimize(RunMode.HYPEROPT, extra_args=hyperopt_args)
    strategy_object = get_preset_strategy(preset, config_optimize_hyperopt, clsname)    
    lock = FileLock(Hyperopt.get_lock_filename(config_optimize_hyperopt))

    try:
        with lock.acquire(timeout=1):
            # Remove noisy log messages
            logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)
            logging.getLogger('filelock').setLevel(logging.WARNING)

            # Initialize backtesting object
            hyperopter = NbHyperopt(config_optimize_hyperopt, strategy_object)
            hyperopter.start_nb_hyperopt()

    except Timeout:
        logger.info("Another running instance of freqtrade Hyperopt detected.")
        logger.info("Simultaneous execution of multiple Hyperopt commands is not supported. "
                    "Hyperopt module is resource hungry. Please run your Hyperopt sequentially "
                    "or on separate machines.")
        logger.info("Quitting now.")
        # TODO: return False here in order to help freqtrade to exit
        # with non-zero exit code...
        # Same in Edge and Backtesting start() functions.