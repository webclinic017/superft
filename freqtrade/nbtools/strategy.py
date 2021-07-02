from freqtrade.strategy.interface import IStrategy


class INbStrategy(IStrategy):
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 9999
    }
    stoploss = -0.99
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False
    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    # Number of candles the strategy requires before producing valid signals
    # NOTE: Depends on your training feature timeframe!
    startup_candle_count: int = 800
    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }