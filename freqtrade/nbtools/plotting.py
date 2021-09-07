from pathlib import Path
from matplotlib import dates
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import time


def parse_dtstring(string: str):
    """ From: 20201231
        To: 2020-12-31
    """
    return datetime.strptime(string, "%Y%m%d").strftime("%Y-%m-%d")


def load_last_plot(n: int, path_mount: Path) -> pd.Series:
    pass


def plot_profits(trades_data: pd.DataFrame, start: str, end: str, path_mount: Path, name: str = "plot", load_last: int = 0):
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
    btc_usdt_df = pd.read_json(path_mount / "data" / "binance" / "BTC_USDT-1h.json")
    btc_usdt_df.columns = ["date", "open", "high", "low", "close", "volume"]
    btc_usdt_df["date"] = pd.to_datetime(btc_usdt_df["date"], unit="ms", utc=True)
    btc_usdt = btc_usdt_df.loc[(btc_usdt_df["date"] >= start) & (btc_usdt_df["date"] <= end)]
    btc_usdt = btc_usdt.set_index("date").resample("1h").mean()
    fig, ax1 = plt.subplots(figsize=(18, 5))
    ax1.plot(btc_usdt["close"], color="orange", label="BTC/USDT", alpha=1)
    ax1.tick_params(axis="y", labelcolor="orange")
    ax1.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)
    ax1.grid(b=False, which="both", axis="y")

    # Section 1.2: Cumulative profit $ over time
    trades = trades.loc[(trades.open_date >= start) & (trades.close_date <= end)]
    trades = trades.set_index("close_date", drop=False)
    cum_profit_abs = trades["profit_abs"].cumsum()
    ax2 = ax1.twinx()
    
    ax2.plot(cum_profit_abs, color="green", label="Returns")
    # TODO: Plot last n profits
    
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.xaxis.set_major_formatter(dates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(dates.MonthLocator(interval=1))
    ax2.xaxis.set_minor_formatter(dates.DateFormatter("%d"))
    ax2.xaxis.set_minor_locator(dates.AutoDateLocator())
    
    if (end_ts - start_ts) > timedelta(days=365):
        ax2.xaxis.set_major_formatter(dates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(dates.YearLocator())
        ax2.xaxis.set_minor_formatter(dates.DateFormatter("%b"))
        ax2.xaxis.set_minor_locator(dates.AutoDateLocator())
        
    ax2.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)

    plt.title(name)
    plt.show()

    # Section 2: Create [left and right] plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
    # Section 2.1 Left: Profit $ with trades cumulative
    ax1.plot(list(trades.profit_abs.cumsum()), color="g")
    ax1.set_title("Trade $ returns")
    ax1.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)
    # Section 2.2 Right: Profit $ distribution histogram
    trades["profit_percentage"] = trades.profit_ratio * 100
    mean_profits = trades.profit_percentage.mean()
    std_profits = trades.profit_percentage.std()
    ax2.hist(
        trades.profit_percentage.clip(mean_profits - 4 * std_profits, mean_profits + 4 * std_profits),
        bins=100,
    )
    ax2.set_title("Returns %% distribution")
    ax2.grid(b=True, which="both", color=grid_color, linestyle="-", axis="both", alpha=grid_alpha)
    plt.show()

    # Start generate summary
    trades_win = trades.loc[trades["profit_ratio"] > 0]
    trades_lost = trades.loc[trades["profit_ratio"] <= 0]
    win_rate = len(trades_win) / len(trades)
    lose_rate = len(trades_lost) / len(trades)
    expectancy = (win_rate * trades_win["profit_ratio"].mean()) + (lose_rate * trades_lost["profit_ratio"].mean())
    
    portfolio_summary = {
        "Trades": len(trades),
        "Avg. Stake Amount": trades["stake_amount"].mean(),
        "Number of Pairs": len(trades["pair"].unique()),
        "Min Balance": min(cum_profit_abs),
        "Max Balance": max(cum_profit_abs),
        "Final Balance": cum_profit_abs[-1],
        "-": "-",
        "Wins": len(trades_win),
        "Loses": len(trades_lost),
        "Win Rate": str(round(win_rate * 100, 2)) + "%",
        " - ": " - ",
        "Profit Factor": trades_win["profit_ratio"].sum() / -trades_lost["profit_ratio"].sum(),
        "Expectancy (% Per Trade)": expectancy * 100,
        "  -  ": "  -  ",
        "Avg. Profit (%)": trades["profit_ratio"].mean() * 100,
        "Avg. Profit (%) Winners": trades_win["profit_ratio"].mean() * 100,
        "Avg. Profit (%) Losers": trades_lost["profit_ratio"].mean() * 100,
        "Net Profit (Rate)": trades_win["profit_ratio"].sum() + trades_lost["profit_ratio"].sum(),
        "Sum Profit Winners (Rate)": trades_win["profit_ratio"].sum(),
        "Sum Profit Losers (Rate)": trades_lost["profit_ratio"].sum(),
        "Avg. Duration": str((trades["close_date"] - trades["open_date"]).mean()).split(".")[0],
        "Avg. Duration Winners": str((trades_win["close_date"] - trades_win["open_date"]).mean()).split(".")[0],
        "Avg. Duration Losers": str((trades_lost["close_date"] - trades_lost["open_date"]).mean()).split(".")[0],
    }
    
    for k in portfolio_summary.keys():
        if isinstance(portfolio_summary[k], float):
            portfolio_summary[k] = round(portfolio_summary[k], 2)
    
    df = pd.DataFrame({k: [v] for k, v in portfolio_summary.items()}).T
    df.columns = ["Portfolio Summary"]
    
    filename = f"{name}___{start}_to_{end}___{int(time.time())}___{int(cum_profit_abs[-1])}.csv"
    cum_profit_abs.to_csv(path_mount / "plot_history" / filename)
    return df


def plot_profits_timerange(trades_data: pd.DataFrame, timerange: str, path_mount: Path, name: str = "plot"):
    if "-" not in timerange:
        raise ValueError("Please follow freqtrade's timerange format. Example: `20210101-20220201` or `20210101-`")
    
    start_stop = timerange.split("-")
    
    if len(start_stop[1]) == 0:
        start = parse_dtstring(start_stop[0])
        stop = "2022-12-30"
    elif len(start_stop[1]) > 0:
        start, stop = parse_dtstring(start_stop[0]), parse_dtstring(start_stop[1])
    else:
        raise ValueError(f"Unknown timerange: '{timerange}' ")
    
    return plot_profits(trades_data, start, stop, path_mount, name=name)


def save_plot():
    boom = "hi"