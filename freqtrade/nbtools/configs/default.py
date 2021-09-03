from copy import deepcopy

DEFAULT = {
    "stake_currency": "USDT",
    "fiat_display_currency": "USD",
    "stake_amount": 15,
    "max_open_trades": 6,
    "tradable_balance_ratio": 0.99,
    "fee": 0.001,
    "dry_run": True,
    "use_sell_signal": True,
    "sell_profit_only": False,
    "sell_profit_offset": 0.0,
    "cancel_open_orders_on_exit": True,
    "ignore_roi_if_buy_signal": True,
    "unfilledtimeout": {"buy": 300, "sell": 60, "unit": "seconds"},
    "order_types": {
        "buy": "limit",
        "sell": "limit",
        "forcesell": "market",
        "emergencysell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
    },
    "bid_strategy": {
        "price_side": "bid",
        "ask_last_balance": 0.0,
        "use_order_book": True,
        "order_book_top": 1,
        "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
    },
    "ask_strategy": {"price_side": "ask", "use_order_book": True, "order_book_top": 1},
    "exchange": {
        "name": "binance",
        "key": "your_exchange_key",
        "secret": "your_exchange_secret",
        "ccxt_config": {"enableRateLimit": True},
        "ccxt_async_config": {"enableRateLimit": True, "rateLimit": 200},
        "pair_whitelist": [],
        "pair_blacklist": [
            # Exchange
            ".*(BNB|KCS)/.*",
            # Leverage
            ".*(_PREMIUM|3L|3S|BEAR|BULL|UP|DOWN|HALF|HEDGE|[1235][SL])/.*",
            # Fiat
            ".*(AUD|BRZ|CAD|CHF|EUR|GBP|HKD|IDRT|JPY|NGN|RUB|SGD|TRY|UAH|USD|ZAR)/.*",
            # Stablecoins
            ".*(BUSD|CUSDT|DAI|PAX|PAXG|SUSD|TUSD|USDC|USDT|VAI)/.*",
            # Fan Tokens
            ".*(ACM|AFA|ALA|ALL|APL|ASR|ATM|BAR|CAI|CITY|FOR|GAL|GOZ|IBFK|JUV|LEG|LOCK-1|NAVI|NMR|NOV|PFL|PSG|ROUSH|STV|TRA|UCH|UFC|YBO)/.*",
            # Others
            ".*(SHIB|SLP)/.*"
        ]
    },
    "pairlists": [{"method": "StaticPairList"}],
    "edge": {
        "enabled": False,
        "process_throttle_secs": 3600,
        "calculate_since_number_of_days": 7,
        "allowed_risk": 0.01,
        "stoploss_range_min": -0.01,
        "stoploss_range_max": -0.1,
        "stoploss_range_step": -0.01,
        "minimum_winrate": 0.60,
        "minimum_expectancy": 0.20,
        "min_trade_number": 10,
        "max_trade_duration_minute": 1440,
        "remove_pumps": False,
    },
    "telegram": {
        "enabled": False,
        "token": "your_telegram_token",
        "chat_id": "your_telegram_chat_id",
        "keyboard": [   
            ["/status table", "/trades", "/profit"],
            ["/daily", "/stats", "/performance"],
            ["/show_config"],
            ["/start", "/stop", "/logs", "/help"],
        ],
    },
    "api_server": {
        "enabled": False,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "jwt_secret_key": "somethingrandom",
        "CORS_origins": [],
        "username": "freqtrader",
        "password": "SuperSecurePassword",
    },
    "bot_name": "freqtrade",
    "initial_state": "running",
    "forcebuy_enable": False,
    "internals": { "process_throttle_secs": 10 },
}