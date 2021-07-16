from copy import deepcopy
from typing import Any, Dict, Optional

from freqtrade import constants
from freqtrade.configuration.check_exchange import check_exchange, remove_credentials
from freqtrade.configuration.config_validation import validate_config_consistency
from freqtrade.configuration.configuration import Configuration
from freqtrade.configuration.deprecated_settings import process_temporary_deprecated_settings
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.misc import round_coin_value


class NbConfiguration(Configuration):
    def __init__(
        self, base_config: Dict[str, Any], args: Dict[str, Any], runmode: RunMode = None
    ) -> None:
        self.base_config: Dict[str, Any] = base_config
        self.args = args
        self.config: Optional[Dict[str, Any]] = None
        self.runmode = runmode

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # Load config from base
        config: Dict[str, Any] = deepcopy(self.base_config)

        # Normalize config
        if "internals" not in config:
            config["internals"] = {}
        if "ask_strategy" not in config:
            config["ask_strategy"] = {}
        if "pairlists" not in config:
            config["pairlists"] = []

        # Keep a copy of the original configuration file
        config["original_config"] = deepcopy(config)

        self._process_config(config)
        return config

    def _process_config(self, config: Dict[str, Any]) -> None:
        self._process_logging_options(config)
        self._process_runmode(config)
        self._process_common_options(config)
        self._process_trading_options(config)
        self._process_optimize_options(config)
        self._process_plot_options(config)
        self._process_data_options(config)
        # Check if the exchange set by the user is supported
        check_exchange(config, config.get("experimental", {}).get("block_bad_exchanges", True))
        self._resolve_pairs_list(config)
        process_temporary_deprecated_settings(config)


def setup_utils_configuration(
    base_config: Dict[str, Any], args: Dict[str, Any], method: RunMode
) -> Dict[str, Any]:
    """
    Prepare the configuration for utils subcommands
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    configuration = NbConfiguration(base_config, args, method)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    remove_credentials(config)
    validate_config_consistency(config)

    return config


def setup_optimize_configuration(
    base_config: Dict[str, Any], args: Dict[str, Any], method: RunMode
) -> Dict[str, Any]:
    """
    Prepare the configuration for the Hyperopt module
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    config = setup_utils_configuration(base_config, args, method)

    no_unlimited_runmodes = {
        RunMode.BACKTEST: "backtesting",
        RunMode.HYPEROPT: "hyperoptimization",
    }
    if method in no_unlimited_runmodes.keys():
        if (
            config["stake_amount"] != constants.UNLIMITED_STAKE_AMOUNT
            and config["stake_amount"] > config["dry_run_wallet"]
        ):
            wallet = round_coin_value(config["dry_run_wallet"], config["stake_currency"])
            stake = round_coin_value(config["stake_amount"], config["stake_currency"])
            raise OperationalException(
                f"Starting balance ({wallet}) " f"is smaller than stake_amount {stake}."
            )

    return config
