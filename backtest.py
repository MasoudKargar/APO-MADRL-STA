from config import config
import pandas as pd 
from finrl.marketdata.yahoodownloader import YahooDownloader
from pyfolio import timeseries
import pyfolio
import matplotlib.pyplot as plt
from copy import deepcopy

def baseline_strat(ticker, start, end):
    dji = YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    dji["daily_return"] = dji["close"].pct_change(1)
    dow_strat = backtest_strat(dji)
    return dji, dow_strat


def backtest_strat(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    ts = pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)
    return ts

# Modify the baseline_strat function to ensure it returns both DataFrames
def baseline_strat(ticker, start, end):
    dji = YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    dji["daily_return"] = dji["close"].pct_change(1)
    dow_strat = backtest_strat(dji)
    return dji, dow_strat


# Use the modified baseline_strat in the BaselineStats function
def BaselineStats(baseline_ticker="^DJI", baseline_start=config.START_TRADE_DATE, baseline_end=config.END_DATE):
    dji, dow_strat = baseline_strat(ticker=baseline_ticker, start=baseline_start, end=baseline_end)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=dow_strat, factor_returns=dow_strat, positions=None, transactions=None,
                               turnover_denom="AGB")
    return perf_stats_all
