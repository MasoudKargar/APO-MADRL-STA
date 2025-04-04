import numpy as np
import pandas as pd
import pyfolio as pf
from pyfolio import timeseries
import matplotlib.pylab as plt
#from backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
from backtest import backtest_strat, baseline_strat

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""### 6.2 Load the Train and Test Data for Portfolios"""

# Commented out IPython magic to ensure Python compatibility.
# %store -r max_sharpe_portfolio
# %store -r uniform_weights_port

# %store -r prices_train_df
# %store -r prices_test_df


# %store -r a2c_train_daily_return
# %store -r ppo_train_daily_return
# %store -r ddpg_train_daily_return
# %store -r td3_train_daily_return
# %store -r sac_train_daily_return


# %store -r a2c_test_returns
# %store -r ppo_test_returns
# %store -r ddpg_test_returns
# %store -r td3_test_returns
# %store -r sac_test_returns

# %store -r all_agents_normalized_test_daily_return
# %store -r dr_kargar_second_algorithm

max_sharpe_portfolio

returns_train = prices_train_df.pct_change()  # get the assets daily returns
returns_test = prices_test_df.pct_change()

# get the culmulative returns for each portfolio
uw_weights = uniform_weights_port.values.flatten()
uw_returns = returns_train.dot(uw_weights)
uw_cum_returns = (1 + uw_returns).cumprod()
uw_cum_returns.name = "portfolio 1: uniform weights"

max_sharpe_weights = max_sharpe_portfolio.values.flatten()
max_sharpe_returns = returns_train.dot(max_sharpe_weights)
max_sharpe_cum_returns = (1 + max_sharpe_returns).cumprod()
max_sharpe_cum_returns.name = "portfolio 2: max sharpe"


a2c_train_cum_returns = (
    1 + a2c_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
a2c_train_cum_returns = a2c_train_cum_returns['daily_return']
a2c_train_cum_returns.name = 'Portfolio 1: A2C Model'

ppo_train_cum_returns = (
    1 + ppo_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
ppo_train_cum_returns = ppo_train_cum_returns['daily_return']
ppo_train_cum_returns.name = 'Portfolio 2: PPO Model'

ddpg_train_cum_returns = (
    1 + ddpg_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
ddpg_train_cum_returns = ddpg_train_cum_returns['daily_return']
ddpg_train_cum_returns.name = 'Portfolio 3: DDPG Model'

td3_train_cum_returns = (
    1 + td3_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
td3_train_cum_returns = td3_train_cum_returns['daily_return']
td3_train_cum_returns.name = 'Portfolio 4: TD3 Model'

sac_train_cum_returns = (
    1 + sac_train_daily_return.reset_index(drop=True).set_index(['date'])).cumprod()
sac_train_cum_returns = sac_train_cum_returns['daily_return']
sac_train_cum_returns.name = 'Portfolio 5: SAC Model'


date_list = list(ddpg_train_cum_returns.index)

max_sharpe_cum_returns = max_sharpe_cum_returns[(
    max_sharpe_cum_returns.index).isin(date_list)]
uw_cum_returns = uw_cum_returns[(uw_cum_returns.index).isin(date_list)]

a2c_train_cum_returns.to_csv('./confrenceresult/a2c_train_cum_returns.csv')
ppo_train_cum_returns.to_csv('./confrenceresult/ppo_train_cum_returns.csv')
ddpg_train_cum_returns.to_csv('./confrenceresult/ddpg_train_cum_returns.csv')
td3_train_cum_returns.to_csv('./confrenceresult/td3_train_cum_returns.csv')
sac_train_cum_returns.to_csv('./confrenceresult/sac_train_cum_returns.csv')

max_sharpe_returns.sum()

# Plot the culmulative returns of the portfolios
#fig, ax = plt.subplots(figsize=(8, 4))
fig, ax = plt.subplots(figsize=(25, 10))

#uw_cum_returns.plot(ax=ax, color="black", alpha=0.4)
#max_sharpe_cum_returns.plot(ax=ax, color="darkorange", alpha=0.4)

a2c_train_cum_returns.plot(ax=ax, color='gray', alpha=0.4)
ppo_train_cum_returns.plot(ax=ax, color='green', alpha=0.4)
ddpg_train_cum_returns.plot(ax=ax, color='purple', alpha=0.4)
td3_train_cum_returns.plot(ax=ax, color='red', alpha=0.4)
sac_train_cum_returns.plot(ax=ax, color='blue', alpha=0.4)
plt.legend(loc="best", fontsize=19, handlelength=4,
           handleheight=2, handletextpad=2)
plt.grid(True)
ax.set_ylabel("cummulative return", fontsize=30)
ax.set_xlabel("Date", fontsize=28)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_title(
    "Backtest based on the data from 2009-03-20 to 2021-03-26", fontsize=30)
fig.savefig('results/back_test_on_train_data.pdf')

"""### 6.4 Evaluating the Portfolios on Test Data"""

import pandas as pd

shahin_new_return_alghortihm_eachstock_return = pd.read_csv(
    'hosini_daily_return.csv')
shahin_new_return_alghortihm_eachstock_return['daily_return'] = shahin_new_return_alghortihm_eachstock_return.sum(axis=1)

shahin_new_return_alghortihm_eachstock_return['daily_return']

returns_train

# get the culmulative returns for each portfolio
uw_weights = uniform_weights_port.values.flatten()
uw_test_returns = returns_test.dot(uw_weights)
uw_test_cum_returns = (1 + uw_test_returns).cumprod()
uw_test_cum_returns.name = "portfolio: uniform weights"

max_sharpe_weights = max_sharpe_portfolio.values.flatten()
max_sharpe_test_returns = returns_test.dot(max_sharpe_weights)
max_sharpe_test_cum_returns = (1 + max_sharpe_test_returns).cumprod()
max_sharpe_test_cum_returns.name = "portfolio: max sharpe"

a2c_test_cum_returns = (1 + a2c_test_returns['daily_return']).cumprod()
a2c_test_cum_returns.name = 'Portfolio 1: A2C Model'

ppo_test_cum_returns = (1 + ppo_test_returns['daily_return']).cumprod()
ppo_test_cum_returns.name = 'Portfolio 2: PPO Model'

ddpg_test_cum_returns = (1 + ddpg_test_returns['daily_return']).cumprod()
ddpg_test_cum_returns.name = 'Portfolio 3: DDPG Model'

td3_test_cum_returns = (1 + td3_test_returns['daily_return']).cumprod()
td3_test_cum_returns.name = 'Portfolio 4: TD3 Model'

sac_test_cum_returns = (1 + sac_test_returns['daily_return']).cumprod()
sac_test_cum_returns.name = 'Portfolio 5: SAC Model'

all_agents_normalized_test_cum_return = (
    1 + all_agents_normalized_test_daily_return['daily_return']).cumprod()
all_agents_normalized_test_cum_return.name = 'Portfolio 6: Proposed method 1'

shahin_cum_new_return_alghortihm_eachstock_return = (
    1 + shahin_new_return_alghortihm_eachstock_return['daily_return']).cumprod()
shahin_cum_new_return_alghortihm_eachstock_return.name = 'Portfolio 7: Main proposed method'

a2c_test_cum_returns.to_csv('./confrenceresult/a2c_test_cum_returns.csv')
ppo_test_cum_returns.to_csv('./confrenceresult/ppo_test_cum_returns.csv')
ddpg_test_cum_returns.to_csv('./confrenceresult/ddpg_test_cum_returns.csv')
td3_test_cum_returns.to_csv('./confrenceresult/td3_test_cum_returns.csv')
sac_test_cum_returns.to_csv('./confrenceresult/sac_test_cum_returns.csv')
all_agents_normalized_test_cum_return.to_csv(
    './confrenceresult/all_agents_normalized_test_cum_return.csv')
shahin_cum_new_return_alghortihm_eachstock_return.to_csv(
    './confrenceresult/shahin_cum_new_return_alghortihm_eachstock_return.csv')

a2c_test_cum_returns

shahin_cum_new_return_alghortihm_eachstock_return

# Plot the culmulative returns of the portfolios
fig, ax = plt.subplots(figsize=(25,10))
#fig, ax = plt.subplots(figsize=(8, 4))
#uw_test_cum_returns.plot(ax=ax, color="black", alpha=.4);
#max_sharpe_test_cum_returns.plot(ax=ax, color="darkorange");
a2c_test_cum_returns.plot(ax=ax, color='blue', alpha=.4);
ppo_test_cum_returns.plot(ax=ax, color='green', alpha=.4);
ddpg_test_cum_returns.plot(ax=ax, color='purple', alpha=.4);
td3_test_cum_returns.plot(ax=ax, color=(0.5, 0.4, 0.0), alpha=0.4);
sac_test_cum_returns.plot(ax=ax, color='darkred', alpha=0.4);
#dr_kargar_second_cum_returns.plot(ax=ax, color='darkred', alpha=0.4)
all_agents_normalized_test_cum_return.plot(ax=ax, color=(0.95, 0.0, 0.0), alpha=0.6)
shahin_cum_new_return_alghortihm_eachstock_return.plot(ax=ax, color='black', alpha=0.8)
plt.legend(loc="best", fontsize=17, handlelength=4, handleheight=1,handletextpad=2);
plt.grid(True);
ax.set_ylabel("cummulative return", fontsize = 30);
ax.set_xlabel("Date", fontsize=28);
ax.set_title("Backtest based on the data from 2021-03-29 to 2024-03-28", fontsize=30);
ax.tick_params(axis='x', labelsize=25);
ax.tick_params(axis='y', labelsize=25);
fig.savefig('results/back_test_on_test_data.pdf');

"""### 6.5 Get the Portfolio Statistics"""

# Define a Function for Getting the Portfolio Statistics

def portfolio_stats(portfolio_returns):
    # Pass the returns into a dataframe
    port_rets_df = pd.DataFrame(portfolio_returns)
    port_rets_df = port_rets_df.reset_index()
    port_rets_df.columns = ['date', 'daily_return']

    # Use the FinRL Library to get the Portfolio Returns
    # This makes use of the Pyfolio Library

    DRL_strat = backtest_strat(port_rets_df)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(returns=DRL_strat,
                               factor_returns=DRL_strat,
                               positions=None, transactions=None, turnover_denom="AGB")
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.columns = ['Statistic']
    return perf_stats_all

shahin_new_return_alghortihm_eachstock_return

# Get the Portfolio Statistics for all the portfolios
portfolios_returns_dict = {'uniform_weights': uw_test_returns, 'maximum_sharpe': max_sharpe_test_returns,
                           'a2c Model': a2c_test_returns['daily_return'],
                           'ppo Model': ppo_test_returns['daily_return'],
                           'ddpg Model': ddpg_test_returns['daily_return'],
                           'td3 Model': td3_test_returns['daily_return'],
                           'sac Model': sac_test_returns['daily_return'],
                           'all_agents_normalized Model': all_agents_normalized_test_daily_return['daily_return'],
                           'second_alg': shahin_new_return_alghortihm_eachstock_return['daily_return']
                           }

portfolios_stats = pd.DataFrame()
for i, j in portfolios_returns_dict.items():
    port_stats = portfolio_stats(j)
    portfolios_stats[i] = port_stats['Statistic']

portfolios_stats.to_csv('portfolios_stats.csv')
portfolios_stats

"""### 6.6 Benchmarking the Best Portfolio against the Benchmark Index"""

#a2c_test_returns = a2c_test_returns.set_index('date')
ppo_test_returns = ppo_test_returns.set_index('date')
ddpg_test_returns = ddpg_test_returns.set_index('date')

a2c_test_returns.head()

# Getting the best performing portfolio

ps_cum = [a2c_test_cum_returns, ppo_test_cum_returns,
          ddpg_test_cum_returns, td3_test_cum_returns, sac_test_cum_returns, shahin_cum_new_return_alghortihm_eachstock_return]
ps = [a2c_test_returns['daily_return'], ppo_test_returns['daily_return'],
      ddpg_test_returns['daily_return'], td3_test_returns['daily_return'], sac_test_returns['daily_return'], shahin_new_return_alghortihm_eachstock_return['daily_return']]

final_return = []
for p in ps_cum:
    final_return.append(p.iloc[-1])

id_ = np.argmax(final_return)
best_p = ps[id_]
best_p.name = (ps_cum[id_]).name

print("Best portfolio: ",  best_p.name)
print("Final cumulative return: {:.2f} ".format(final_return[id_]))

# convert the best portfolio into a Dataframe

best_p = pd.DataFrame(best_p)
best_p = best_p.reset_index()
best_p.columns=['date','daily_return']
best_p['date'] = a2c_test_returns['date']

best_p.head()

# Best portfolio stats
best_port_strat = backtest_strat(best_p)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=best_port_strat, factor_returns=best_port_strat,  positions=None, transactions=None, turnover_denom="AGB")

from backtest import BaselineStats
prices_test_dates = list(prices_test_df.index)
print("==============Get Index Stats===========")

baesline_perf_stats=BaselineStats('^DJI', baseline_start = prices_test_dates[0], baseline_end = prices_test_dates[-1])

dji, dow_strat = baseline_strat('^DJI',prices_test_dates[1], prices_test_dates[-1])

dow_strat

dow_strat_cum.name = 'Portfolio: dow_strat'

dow_strat_cum = (1 + dow_strat['daily_return']).cumprod()

# Plot the culmulative returns of the portfolios
fig, ax = plt.subplots(figsize=(28, 14))
# uw_test_cum_returns.plot(ax=ax, color="black", alpha=.4);
# max_sharpe_test_cum_returns.plot(ax=ax, color="darkorange");
#a2c_test_cum_returns.plot(ax=ax, color='blue', alpha=.4)
ppo_test_cum_returns.plot(ax=ax, color='green', alpha=.4)
ddpg_test_cum_returns.plot(ax=ax, color='purple', alpha=.4)
dow_strat_cum.plot(ax=ax, color='darkgray', alpha=0.8)
#sac_test_cum_returns.plot(ax=ax, color='darkred', alpha=0.4)
# dr_kargar_second_cum_returns.plot(ax=ax, color='darkred', alpha=0.4)
shahin_cum_new_return_alghortihm_eachstock_return.plot(
    ax=ax, color='black', alpha=0.8)
#all_agents_normalized_test_cum_return.plot(ax=ax, color='darkred', alpha=0.4)
plt.legend(loc="best")
plt.grid(True)
ax.set_ylabel("cummulative return")
ax.set_title(
    "Backtest based on the data from 2021-03-29 to 2024-03-28", fontsize=14)
fig.savefig('results/back_test_on_test_data.pdf')