import pandas as pd
import numpy as np

from pyfolio import timeseries
import pyfolio
import matplotlib.pyplot as plt
from copy import deepcopy

from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.config import config


def get_daily_return(df, value_col_name="account_value"):
    '''
    This function takes the return of env.get_save_asset_memory dataframe
    It then computes daily returns based on the column 'value_col_name'
    '''
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace = True, drop = True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df['daily_return'], index = df.index)



def backtest_stats(account_value, value_col_name="account_value"):
    '''
    This function takes in an account value dataframe and creates backtesting statistics about the value specified. 
    '''
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all

def BackTestStats(account_value, value_col_name="account_value"):
    '''
    This function takes in an account value dataframe and creates backtesting statistics about the value specified. 
    '''
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all



def backtest_plot(
    account_value,
    baseline_start=config.START_TRADE_DATE,
    baseline_end=config.END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    '''
    This function takes in the output of env.save_asset_memory function dataframe, and the specified value column and returns 
        a backtesting plot for display. 
        It also takes in a comparison ticker to see how the account value compares to the baseline. 
    '''
    df = deepcopy(account_value)
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )
    
    baseline_returns = get_daily_return(baseline_df, value_col_name='close')
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )

def BackTestPlot(
    account_value,
    baseline_start=config.START_TRADE_DATE,
    baseline_end=config.END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    '''
    This function takes in the output of env.save_asset_memory function dataframe, and the specified value column and returns 
        a backtesting plot for display. 
        It also takes in a comparison ticker to see how the account value compares to the baseline. 
    '''
    df = deepcopy(account_value)
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )
    
    baseline_returns = get_daily_return(baseline_df, value_col_name='close')
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )



def get_baseline(ticker, start, end):
    '''
    This function downloads a ticker from the yahoo downloader. 
    '''
    dji = YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()
    return dji

'''
def baseline_strat(ticker, start, end):
    dji = YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    dji["daily_return"] = dji["close"].pct_change(1)
    dow_strat = backtest_strat(dji)
    return dji, dow_strat
'''
def baseline_strat(ticker, start, end):
    dji = YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()

    print("=== Data from YahooDownloader ===")
    print(dji.head())  # Print the first few rows of the DataFrame
    print("=== Columns ===")
    print(dji.columns)  # Check all the columns of the DataFrame

    # Manually set the 'date' column based on the index
    dji["date"] = dji.index

    # Identify the correct column name for daily returns
    close_column_name = "close"  # Replace with the actual column name
    print(dji)
    input("--dij column---")
    dji["daily_return"] = dji[close_column_name].pct_change(1)
    print("=== DataFrame after adding daily_return ===")
    print(dji.head())  # Check the first few rows again

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
'''
# Modify the baseline_strat function to ensure it returns both DataFrames
def baseline_strat(ticker, start, end):
    dji = YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    dji["daily_return"] = dji["close"].pct_change(1)
    dow_strat = backtest_strat(dji)
    return dji, dow_strat
'''

# Use the modified baseline_strat in the BaselineStats function
def BaselineStats(baseline_ticker="^DJI", baseline_start=config.START_TRADE_DATE, baseline_end=config.END_DATE):
    dji, dow_strat = baseline_strat(ticker=baseline_ticker, start=baseline_start, end=baseline_end)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=dow_strat, factor_returns=dow_strat, positions=None, transactions=None,
                               turnover_denom="AGB")
    return perf_stats_all

