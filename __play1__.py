from pathlib import Path
import pandas as pd

from freqtrade.ml.loader import load_df
from __play2__ import *


path_data = Path.cwd().parent / "mount" / "data" / "binance"
btc_usdt = load_df(path_data / "BTC_USDT-5m.json", "5m").loc[-5000:]


btc_usdt = macd_strategy(btc_usdt)


print(btc_usdt)