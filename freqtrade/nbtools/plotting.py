from pathlib import Path
from matplotlib import dates
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def parse_dtstring(string: str):
    """ From: 20201231
        To: 2020-12-31
    """
    return datetime.strptime(string, "%Y%m%d").strftime("%Y-%m-%d")


# pyright: reportGeneralTypeIssues=false
def plot_profits(trades_data: pd.DataFrame, start: str, end: str, path_mount: Path):
    trades = trades_data.copy()

    start_ts = pd.Timestamp(start, tz="UTC")
    if start_ts < trades["open_date"].min():
        start_ts = trades["open_date"].min()

    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts > trades["close_date"].max():
        end_ts = trades["close_date"].max()

    # Plot style
    grid_color = "black"
    grid_alpha = 0.1

    # Section 1.1: BTC/USDT price over time
    btc_usdt_df = pd.read_json(path_mount / "data" / "binance" / "BTC_USDT-15m.json")
    btc_usdt_df.columns = ["date", "open", "high", "low", "close", "volume"]
    btc_usdt_df["date"] = pd.to_datetime(btc_usdt_df["date"], unit="ms", utc=True)
    btc_usdt = btc_usdt_df.loc[(btc_usdt_df["date"] >= start) & (btc_usdt_df["date"] <= end)]
    btc_usdt = btc_usdt.set_index("date").resample("1h").mean()
    fig, ax1 = plt.subplots(figsize=(18, 5))
    ax1.plot(btc_usdt["close"], color="orange", label="BTC/USDT")
    ax1.tick_params(axis="y", labelcolor="orange")

    # Section 1.2: Cumulative profit $ over time
    trades = trades.loc[(trades.open_date >= start) & (trades.close_date <= end)]
    trades = trades.set_index("close_date")
    profits_usd = trades.profit_abs.cumsum()
    ax2 = ax1.twinx()
    ax2.plot(profits_usd, color="green", label="Returns")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.xaxis.set_major_formatter(dates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(dates.MonthLocator(interval=1))
    ax2.xaxis.set_minor_formatter(dates.DateFormatter("%d"))
    ax2.xaxis.set_minor_locator(dates.AutoDateLocator())

    # Stylize and plot our section 1
    plt.title("BTC/USDT (orange), Returns in $ (green)")
    ax1.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)
    ax2.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)
    ax1.grid(b=False, which="both", axis="y")
    plt.show()

    # Section 2: Create [left and right] plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))

    # Section 2.1 Left: Profit $ with trades cumulative
    ax1.plot(list(trades.profit_abs.cumsum()), color="g")
    ax1.set_title("Trade $ returns")
    ax1.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)

    # Section 2.2 Right: Profit $ distribution histogram
    mean_profits = trades.profit_abs.mean()
    std_profits = trades.profit_abs.std()
    ax2.hist(
        trades.profit_abs.clip(mean_profits - 4 * std_profits, mean_profits + 4 * std_profits),
        bins=100,
    )
    ax2.set_title("Returns $ distribution")
    ax2.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)
    plt.show()

    portfolio_summary = {
        "Min Balance": round(min(profits_usd), 2),
        "Max Balance": round(max(profits_usd), 2),
        "End Balance": round(profits_usd[-1], 2),
        "Trades": len(profits_usd),
        "Avg. Profit %": round(trades["profit_ratio"].mean() * 100, 2),
        "Avg. Profit $": round(profits_usd[-1] / len(profits_usd), 2),
        "Biggest Profit $": round(trades.profit_abs.max(), 2),
        "Biggest Loss $": round(trades.profit_abs.min(), 2),
    }
    df = pd.DataFrame({k: [v] for k, v in portfolio_summary.items()}).T
    df.columns = ["Portfolio Summary"]
    return df   


def plot_profits_timerange(trades_data: pd.DataFrame, timerange: str, path_mount: Path):
    start_stop = timerange.split("-")
    
    if len(start_stop) == 1:
        start = parse_dtstring(start_stop[0])
        stop = "2022-12-30"
    elif len(start_stop) == 2:
        start, stop = parse_dtstring(start_stop[0]), parse_dtstring(start_stop[1])
    else:
        raise ValueError(f"Unknown timerange: '{timerange}' ")
    
    return plot_profits(trades_data, start, stop, path_mount)