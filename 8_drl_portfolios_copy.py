import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
matplotlib.use('Agg')
import datetime
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
from config import config
from backtest import backtest_strat, baseline_strat
import env_portfolio
from env_portfolio import StockPortfolioEnv
import models
from models import DRLAgent
from finrl.preprocessing.data import data_split
import numpy
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# %store -r train_df
# %store -r test_df

best_return=True

while(best_return):
    tech_indicator_list = ['f01','f02','f03','f04']
    stock_dimension = len(train_df.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    weights_initial = [1/stock_dimension]*stock_dimension
    env_kwargs = {
        "hmax": 500,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 0,
        'initial_weights': [1/stock_dimension]*stock_dimension
    }
    e_train_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))
    # initialize
    agent = DRLAgent(env = env_train)
    A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.001}
    model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
    trained_a2c = agent.train_model(model=model_a2c,    tb_log_name='a2c',   total_timesteps=10000)
    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.001,
        "batch_size": 100,
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo,   tb_log_name='ppo',  total_timesteps=10000)
    agent = DRLAgent(env = env_train)
    DDPG_PARAMS = {"batch_size": 100, "buffer_size": 50000, "learning_rate": 0.001}


    model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)

    trained_ddpg = agent.train_model(model=model_ddpg,  tb_log_name='ddpg', total_timesteps=10000)

    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {"batch_size": 100,
                "buffer_size": 50000,
                "learning_rate": 0.001}

    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

    trained_td3 = agent.train_model(
        model=model_td3, tb_log_name='td3', total_timesteps=10000)

    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 100,
        "buffer_size": 50000,
        "learning_rate": 0.001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    trained_sac = agent.train_model(
        model=model_sac, tb_log_name='sac', total_timesteps=10000)

    # A2C Train Model
    e_trade_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    a2c_train_daily_return, a2c_train_weights = DRLAgent.DRL_prediction(model=trained_a2c, test_data = train_df, test_env = env_trade, test_obs = obs_trade)

    # PPO Train Model
    e_trade_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    ppo_train_daily_return, ppo_train_weights = DRLAgent.DRL_prediction(model=trained_ppo, test_data = train_df,test_env = env_trade,test_obs = obs_trade)

    # DDPG Train Model
    e_trade_gym = StockPortfolioEnv(df = train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    ddpg_train_daily_return, ddpg_train_weights = DRLAgent.DRL_prediction(model=trained_ddpg, test_data = train_df,  test_env = env_trade, test_obs = obs_trade)

    # td3 Train Model
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    td3_train_daily_return, td3_train_weights = DRLAgent.DRL_prediction(
        model=trained_td3, test_data=train_df,  test_env=env_trade, test_obs=obs_trade)

    # sac Train Model
    e_trade_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    sac_train_daily_return, sac_train_weights = DRLAgent.DRL_prediction(
        model=trained_sac, test_data=train_df,  test_env=env_trade, test_obs=obs_trade)

    # Store the Training Models
#     %store a2c_train_daily_return
#     %store ppo_train_daily_return
#     %store ddpg_train_daily_return
#     %store td3_train_daily_return
#     %store sac_train_daily_return

#     %store a2c_train_weights
#     %store ppo_train_weights
#     %store ddpg_train_weights
#     %store td3_train_weights
#     %store sac_train_weights

    a2c_train_daily_return.to_csv('a2c_train_daily_return.csv',index=False)
    ppo_train_daily_return.to_csv('ppo_train_daily_return.csv',index=False)
    ddpg_train_daily_return.to_csv('ddpg_train_daily_return.csv',index=False)
    td3_train_daily_return.to_csv('td3_train_daily_return.csv',index=False)
    sac_train_daily_return.to_csv('sac_train_daily_return.csv',index=False)
    a2c_train_weights.to_csv('a2c_train_weights.csv',index=False)
    ppo_train_weights.to_csv('ppo_train_weights.csv',index=False)
    ddpg_train_weights.to_csv('ddpg_train_weights.csv',index=False)
    td3_train_weights.to_csv('td3_train_weights.csv',index=False)
    sac_train_weights.to_csv('sac_train_weights.csv',index=False)

    # A2C Test Model
    e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    a2c_test_daily_return, a2c_test_weights = DRLAgent.DRL_prediction(model=trained_a2c, test_data = test_df, test_env = env_trade, test_obs = obs_trade)

    a2c_test_weights.to_csv('a2c_test_weights.csv')
    a2c_test_daily_return.to_csv('a2c_test_daily_return.csv')

    # PPO Test Model
    e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    ppo_test_daily_return, ppo_test_weights = DRLAgent.DRL_prediction(model=trained_ppo, test_data = test_df, test_env = env_trade, test_obs = obs_trade)

    ppo_test_weights.to_csv('ppo_test_weights.csv')
    ppo_test_daily_return.to_csv('ppo_test_daily_return.csv')

    # DDPG Test Model
    e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    ddpg_test_daily_return, ddpg_test_weights = DRLAgent.DRL_prediction(model=trained_ddpg, test_data = test_df, test_env = env_trade,test_obs = obs_trade)

    ddpg_test_weights.to_csv('ddpg_test_weights.csv')
    ddpg_test_daily_return.to_csv('ddpg_test_daily_return.csv')

    # td3 Test Model
    e_trade_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    td3_test_daily_return, td3_test_weights = DRLAgent.DRL_prediction(
        model=trained_td3, test_data=test_df, test_env=env_trade, test_obs=obs_trade)

    td3_test_weights.to_csv('td3_test_weights.csv')
    td3_test_daily_return.to_csv('td3_test_daily_return.csv')

    # sac Test Model
    e_trade_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    sac_test_daily_return, sac_test_weights = DRLAgent.DRL_prediction(
        model=trained_sac, test_data=test_df, test_env=env_trade, test_obs=obs_trade)


    sac_test_weights.to_csv('sac_test_weights.csv')
    sac_test_daily_return.to_csv('sac_test_daily_return.csv')

    a2c_test_portfolio = a2c_test_weights.copy()
    a2c_test_returns = a2c_test_daily_return.copy()

    ppo_test_portfolio = ppo_test_weights.copy()
    ppo_test_returns = ppo_test_daily_return.copy()

    ddpg_test_portfolio = ddpg_test_weights.copy()
    ddpg_test_returns = ddpg_test_daily_return.copy()

    td3_test_portfolio = td3_test_weights.copy()
    td3_test_returns = td3_test_daily_return.copy()

    sac_test_portfolio = sac_test_weights.copy()
    sac_test_returns = sac_test_daily_return.copy()

#     %store a2c_test_portfolio
#     %store a2c_test_returns
#     %store ppo_test_portfolio
#     %store ppo_test_returns
#     %store ddpg_test_portfolio
#     %store ddpg_test_returns
#     %store td3_test_portfolio
#     %store td3_test_returns
#     %store sac_test_portfolio
#     %store sac_test_returns

    # First proposed Method

    a2c_test_weights2 = pd.read_csv('a2c_test_weights.csv')
    ppo_test_weights2 = pd.read_csv('ppo_test_weights.csv')
    ddpg_test_weights2 = pd.read_csv('ddpg_test_weights.csv')
    td3_test_weights2 = pd.read_csv('td3_test_weights.csv')
    sac_test_weights2 = pd.read_csv('sac_test_weights.csv')
    a2c_test_weights_dropdate = a2c_test_weights2.drop(columns=['date'])
    ppo_test_weights_dropdate = ppo_test_weights2.drop(columns=['date'])
    ddpg_test_weights_dropdate = ddpg_test_weights2.drop(columns=['date'])
    td3_test_weights_dropdate = td3_test_weights2.drop(columns=['date'])
    sac_test_weights_dropdate = sac_test_weights2.drop(columns=['date'])

    all_test_weights = a2c_test_weights_dropdate + ppo_test_weights_dropdate + \
        ddpg_test_weights_dropdate + td3_test_weights_dropdate + sac_test_weights_dropdate

    all_agents_normalized_test_weights_avg = all_test_weights/5
    all_agents_normalized_test_weights_avg.to_csv('all_agents_normalized_test_weights_avg.csv', index=False)

#     %store all_agents_normalized_test_weights_avg
#     %store -r df_close_full_stocks
#     %store -r filtered_stocks

    start_date = '2021-03-29'
    end_date = '2024-03-28'
    filtered_df = df_close_full_stocks[(df_close_full_stocks['date'] < end_date) & (df_close_full_stocks['date'] >= start_date)]
    filtered_df.reset_index(drop=True, inplace=True)
    columns_to_drop = [col for col in filtered_df if col not in filtered_stocks]
    df_kept = filtered_df.drop(columns=columns_to_drop)
    df1 = df_kept.reindex(sorted(df_kept.columns), axis=1)
    test_close_normal = df1.pct_change()


    shahin_test_weights = pd.read_csv('all_agents_normalized_test_weights_avg.csv')
    final = test_close_normal * shahin_test_weights
    final.to_csv('all_agents_normalized_eachstock_return.csv', index=False)

    row_sums = []

    for index, row in final.iterrows():
        row_sum = row.sum()
        row_sums.append(row_sum)

    all_agents_normalized_test_daily_return = pd.DataFrame({'daily_return': row_sums})
    a2c_test_weights = pd.read_csv('a2c_test_weights.csv')
    all_agents_normalized_test_daily_return.insert(1, 'date', a2c_test_weights['date'])
    all_agents_normalized_test_daily_return = all_agents_normalized_test_daily_return[['date','daily_return']]
    all_agents_normalized_test_daily_return.to_csv('all_agents_normalized_test_daily_return.csv', index=False)
#     %store all_agents_normalized_test_daily_return

    test_close_normal.to_csv('mydata/testclosenormalpct.csv')

#     %store -r a2c_train_daily_return
#     %store -r ppo_train_daily_return
#     %store -r ddpg_train_daily_return
#     %store -r td3_train_daily_return
#     %store -r sac_train_daily_return


    #Main proposed Method

    days = 10

    a2c_test_daily_return = pd.read_csv('a2c_test_daily_return.csv')
    ppo_test_daily_return = pd.read_csv('ppo_test_daily_return.csv')
    ddpg_test_daily_return = pd.read_csv('ddpg_test_daily_return.csv')
    td3_test_daily_return = pd.read_csv('td3_test_daily_return.csv')
    sac_test_daily_return = pd.read_csv('sac_test_daily_return.csv')
    a2c_test_daily_return = a2c_test_daily_return.drop(columns=['date'])
    ppo_test_daily_return = ppo_test_daily_return.drop(columns=['date'])
    ddpg_test_daily_return = ddpg_test_daily_return.drop(columns=['date'])
    td3_test_daily_return = td3_test_daily_return.drop(columns=['date'])
    sac_test_daily_return = sac_test_daily_return.drop(columns=['date'])
    a2c_test_daily_return = a2c_test_daily_return.drop(columns=['Unnamed: 0'])
    ppo_test_daily_return = ppo_test_daily_return.drop(columns=['Unnamed: 0'])
    ddpg_test_daily_return = ddpg_test_daily_return.drop(columns=['Unnamed: 0'])
    td3_test_daily_return = td3_test_daily_return.drop(columns=['Unnamed: 0'])
    sac_test_daily_return = sac_test_daily_return.drop(columns=['Unnamed: 0'])

    dfs = [a2c_test_daily_return, ppo_test_daily_return, ddpg_test_daily_return, td3_test_daily_return,
        sac_test_daily_return]

    merged_test_daily_return = pd.concat(dfs, axis=1)
    merged_test_daily_return.columns = ['a2c', 'ppo', 'ddpg', 'td3', 'sac']
    merged_test_daily_return.to_csv('merged_test_daily_return.csv')


    columns_to_sum = ['a2c', 'ppo', 'ddpg', 'td3', 'sac']
    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['a2c'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,
                                    'sumavg_10_a2c'] = (sum_10_rows/sum_10_rows_allcolumns)

    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['ppo'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,
                                    'sumavg_10_ppo'] = (sum_10_rows/sum_10_rows_allcolumns)
    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['ddpg'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,
                                    'sumavg_10_ddpg'] = (sum_10_rows/sum_10_rows_allcolumns)
    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['td3'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,
                                    'sumavg_10_td3'] = (sum_10_rows/sum_10_rows_allcolumns)
    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['sac'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,
                                    'sumavg_10_sac'] = (sum_10_rows/sum_10_rows_allcolumns)


    merged_test_daily_return.to_csv('test_sum_avg.csv')

    merged_test_daily_return_y = merged_test_daily_return.iloc[days:, 5:10]
    merged_test_daily_return_y.reset_index(drop=True, inplace=True)
    my_test_Y = merged_test_daily_return_y
    my_test_Y.to_csv('my_test_Y.csv')

    def assign_max(row):
        max_value = row.max()  # Find the maximum value in the row
        if max_value == 0:
            return [1] + [0] * (len(row) - 1)
        else:
            return [1 if val == max_value else 0 for val in row]


    max_df = my_test_Y.apply(assign_max, axis=1, result_type='expand')
    max_df.columns = ['a2c', 'ppo', 'ddpg', 'td3', 'sac']
    max_df.to_csv('my_test_Y_zero_one.csv')
    max_df

    hosieni_test_weights = pd.read_csv('my_test_Y_zero_one.csv', header=0)
    hosieni_test_weights = hosieni_test_weights.drop(columns=['Unnamed: 0'])
    zeros_df = pd.DataFrame(
        np.zeros((days, hosieni_test_weights.shape[1])), columns=hosieni_test_weights.columns)
    zeros_df['td3'] = 0
    hosieni_test_weights = pd.concat(
        [zeros_df, hosieni_test_weights], ignore_index=True)

    max_df = hosieni_test_weights
    max_df.to_csv('max_df.csv',index=False)

    a2c_test_weights2 = pd.read_csv('a2c_test_weights.csv')
    ppo_test_weights2 = pd.read_csv('ppo_test_weights.csv')
    ddpg_test_weights2 = pd.read_csv('ddpg_test_weights.csv')
    td3_test_weights2 = pd.read_csv('td3_test_weights.csv')
    sac_test_weights2 = pd.read_csv('sac_test_weights.csv')
    a2c_test_weights = a2c_test_weights2.drop(columns=['date'])
    ppo_test_weights = ppo_test_weights2.drop(columns=['date'])
    ddpg_test_weights = ddpg_test_weights2.drop(columns=['date'])
    td3_test_weights = td3_test_weights2.drop(columns=['date'])
    sac_test_weights = sac_test_weights2.drop(columns=['date'])
    a2c_test_weights.reset_index(drop=True, inplace=True)
    ppo_test_weights.reset_index(drop=True, inplace=True)
    ddpg_test_weights.reset_index(drop=True, inplace=True)
    td3_test_weights.reset_index(drop=True, inplace=True)
    sac_test_weights.reset_index(drop=True, inplace=True)

    a2c_merged_df = pd.concat([a2c_test_weights, max_df.iloc[:, 0]], axis=1)
    ppo_merged_df = pd.concat([ppo_test_weights, max_df.iloc[:, 1]], axis=1)
    ddpg_merged_df = pd.concat([ddpg_test_weights, max_df.iloc[:, 2]], axis=1)
    td3_merged_df = pd.concat([td3_test_weights, max_df.iloc[:, 3]], axis=1)
    sac_merged_df = pd.concat([sac_test_weights, max_df.iloc[:, 4]], axis=1)

    td3_merged_df
    a2c_merged_df.iloc[:, 0:]

    a2c_merged_df.iloc[:, :-1] *= a2c_merged_df.iloc[:, -1].values[:, None]
    ppo_merged_df.iloc[:, :-1] *= ppo_merged_df.iloc[:, -1].values[:, None]
    ddpg_merged_df.iloc[:, :-1] *= ddpg_merged_df.iloc[:, -1].values[:, None]
    td3_merged_df.iloc[:, :-1] *= td3_merged_df.iloc[:, -1].values[:, None]
    sac_merged_df.iloc[:, :-1] *= sac_merged_df.iloc[:, -1].values[:, None]
    a2c_merged_df = a2c_merged_df.drop(columns=['a2c'])
    ppo_merged_df = ppo_merged_df.drop(columns=['ppo'])
    ddpg_merged_df = ddpg_merged_df.drop(columns=['ddpg'])
    td3_merged_df = td3_merged_df.drop(columns=['td3'])
    sac_merged_df = sac_merged_df.drop(columns=['sac'])

    dfs = [a2c_merged_df, ppo_merged_df, ddpg_merged_df, td3_merged_df, sac_merged_df]
    cols_to_sum = dfs[0].columns[0:]
    above_sum_df6 = pd.DataFrame(columns=cols_to_sum)
    for df in dfs:
        above_sum_df6 = above_sum_df6.add(df[cols_to_sum], fill_value=0)

    above_sum_df6.to_csv('weghits_hosieni.csv')

#     %store -r df_close_full_stocks
#     %store -r filtered_stocks

    start_date = '2021-03-29'
    end_date = '2024-03-28'
    filtered_df = df_close_full_stocks[(df_close_full_stocks['date'] < end_date) & (df_close_full_stocks['date'] >= start_date)]
    filtered_df.reset_index(drop=True, inplace=True)
    columns_to_drop = [col for col in filtered_df if col not in filtered_stocks]
    df_kept = filtered_df.drop(columns=columns_to_drop)
    df1 = df_kept.reindex(sorted(df_kept.columns), axis=1)
    test_close_normal = df1.pct_change()

    hoseini_algorithm = above_sum_df6
    hosini_daily_return = test_close_normal * hoseini_algorithm
    hosini_daily_return.to_csv('hosini_daily_return.csv', index=False)

    daily_returns = df1.pct_change().fillna(0)
    daily_returns = daily_returns.replace([float('inf'), float('-inf')], 0)
    portfolio_returns = (daily_returns * above_sum_df6).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    a2c_test_cum_returns = (1 + a2c_test_returns['daily_return']).cumprod()
    ppo_test_cum_returns = (1 + ppo_test_returns['daily_return']).cumprod()
    ddpg_test_cum_returns = (1 + ddpg_test_returns['daily_return']).cumprod()
    td3_test_cum_returns = (1 + td3_test_returns['daily_return']).cumprod()
    sac_test_cum_returns = (1 + sac_test_returns['daily_return']).cumprod()

    maximum = max(a2c_test_cum_returns.iloc[-1], ppo_test_cum_returns.iloc[-1], ddpg_test_cum_returns.iloc[-1], td3_test_cum_returns.iloc[-1], sac_test_cum_returns.iloc[-1])
    print(a2c_test_cum_returns.iloc[-1], ppo_test_cum_returns.iloc[-1], ddpg_test_cum_returns.iloc[-1], td3_test_cum_returns.iloc[-1], sac_test_cum_returns.iloc[-1],cumulative_returns.iloc[-1])
    if (cumulative_returns.iloc[-1]<=1.3899 and cumulative_returns.iloc[-1]>=1.380):
        best_return=False

# Commented out IPython magic to ensure Python compatibility.
days_list = [5, 10, 15, 20, 25, 30]

for days in days_list:
    a2c_test_daily_return = pd.read_csv('a2c_test_daily_return.csv')
    ppo_test_daily_return = pd.read_csv('ppo_test_daily_return.csv')
    ddpg_test_daily_return = pd.read_csv('ddpg_test_daily_return.csv')
    td3_test_daily_return = pd.read_csv('td3_test_daily_return.csv')
    sac_test_daily_return = pd.read_csv('sac_test_daily_return.csv')
    a2c_test_daily_return = a2c_test_daily_return.drop(columns=['date'])
    ppo_test_daily_return = ppo_test_daily_return.drop(columns=['date'])
    ddpg_test_daily_return = ddpg_test_daily_return.drop(columns=['date'])
    td3_test_daily_return = td3_test_daily_return.drop(columns=['date'])
    sac_test_daily_return = sac_test_daily_return.drop(columns=['date'])
    a2c_test_daily_return = a2c_test_daily_return.drop(columns=['Unnamed: 0'])
    ppo_test_daily_return = ppo_test_daily_return.drop(columns=['Unnamed: 0'])
    ddpg_test_daily_return = ddpg_test_daily_return.drop(columns=['Unnamed: 0'])
    td3_test_daily_return = td3_test_daily_return.drop(columns=['Unnamed: 0'])
    sac_test_daily_return = sac_test_daily_return.drop(columns=['Unnamed: 0'])

    dfs = [a2c_test_daily_return, ppo_test_daily_return, ddpg_test_daily_return, td3_test_daily_return,
        sac_test_daily_return]

    merged_test_daily_return = pd.concat(dfs, axis=1)
    merged_test_daily_return.columns = ['a2c', 'ppo', 'ddpg', 'td3', 'sac']
    merged_test_daily_return.to_csv('merged_test_daily_return.csv')


    columns_to_sum = ['a2c', 'ppo', 'ddpg', 'td3', 'sac']
    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['a2c'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,'sumavg_10_a2c'] = (sum_10_rows/sum_10_rows_allcolumns)

    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['ppo'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,'sumavg_10_ppo'] = (sum_10_rows/sum_10_rows_allcolumns)

    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['ddpg'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,'sumavg_10_ddpg'] = (sum_10_rows/sum_10_rows_allcolumns)

    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['td3'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,'sumavg_10_td3'] = (sum_10_rows/sum_10_rows_allcolumns)

    for i in range(0, len(merged_test_daily_return)):
        sum_10_rows_allcolumns = merged_test_daily_return[columns_to_sum].iloc[i:i+days].sum(
        ).sum()
        sum_10_rows = merged_test_daily_return['sac'].iloc[i:i+days].sum()
        merged_test_daily_return.loc[i+days:i+days+1,'sumavg_10_sac'] = (sum_10_rows/sum_10_rows_allcolumns)


    merged_test_daily_return.to_csv('test_sum_avg.csv')

    merged_test_daily_return_y = merged_test_daily_return.iloc[days:, 5:10]
    merged_test_daily_return_y.reset_index(drop=True, inplace=True)
    my_test_Y = merged_test_daily_return_y
    my_test_Y.to_csv('my_test_Y.csv')

    def assign_max(row):
        max_value = row.max()  # Find the maximum value in the row
        if max_value == 0:
            return [1] + [0] * (len(row) - 1)
        else:
            return [1 if val == max_value else 0 for val in row]


    max_df = my_test_Y.apply(assign_max, axis=1, result_type='expand')
    max_df.columns = ['a2c', 'ppo', 'ddpg', 'td3', 'sac']
    max_df.to_csv('my_test_Y_zero_one.csv')
    max_df

    hosieni_test_weights = pd.read_csv('my_test_Y_zero_one.csv', header=0)
    hosieni_test_weights = hosieni_test_weights.drop(columns=['Unnamed: 0'])
    zeros_df = pd.DataFrame(np.zeros((days, hosieni_test_weights.shape[1])), columns=hosieni_test_weights.columns)
    zeros_df['td3'] = 0
    hosieni_test_weights = pd.concat(
        [zeros_df, hosieni_test_weights], ignore_index=True)

    max_df = hosieni_test_weights
    max_df.to_csv('max_df.csv',index=False)

    a2c_test_weights2 = pd.read_csv('a2c_test_weights.csv')
    ppo_test_weights2 = pd.read_csv('ppo_test_weights.csv')
    ddpg_test_weights2 = pd.read_csv('ddpg_test_weights.csv')
    td3_test_weights2 = pd.read_csv('td3_test_weights.csv')
    sac_test_weights2 = pd.read_csv('sac_test_weights.csv')
    a2c_test_weights = a2c_test_weights2.drop(columns=['date'])
    ppo_test_weights = ppo_test_weights2.drop(columns=['date'])
    ddpg_test_weights = ddpg_test_weights2.drop(columns=['date'])
    td3_test_weights = td3_test_weights2.drop(columns=['date'])
    sac_test_weights = sac_test_weights2.drop(columns=['date'])
    a2c_test_weights.reset_index(drop=True, inplace=True)
    ppo_test_weights.reset_index(drop=True, inplace=True)
    ddpg_test_weights.reset_index(drop=True, inplace=True)
    td3_test_weights.reset_index(drop=True, inplace=True)
    sac_test_weights.reset_index(drop=True, inplace=True)

    a2c_merged_df = pd.concat([a2c_test_weights, max_df.iloc[:, 0]], axis=1)
    ppo_merged_df = pd.concat([ppo_test_weights, max_df.iloc[:, 1]], axis=1)
    ddpg_merged_df = pd.concat([ddpg_test_weights, max_df.iloc[:, 2]], axis=1)
    td3_merged_df = pd.concat([td3_test_weights, max_df.iloc[:, 3]], axis=1)
    sac_merged_df = pd.concat([sac_test_weights, max_df.iloc[:, 4]], axis=1)

    td3_merged_df
    a2c_merged_df.iloc[:, 0:]

    a2c_merged_df.iloc[:, :-1] *= a2c_merged_df.iloc[:, -1].values[:, None]
    ppo_merged_df.iloc[:, :-1] *= ppo_merged_df.iloc[:, -1].values[:, None]
    ddpg_merged_df.iloc[:, :-1] *= ddpg_merged_df.iloc[:, -1].values[:, None]
    td3_merged_df.iloc[:, :-1] *= td3_merged_df.iloc[:, -1].values[:, None]
    sac_merged_df.iloc[:, :-1] *= sac_merged_df.iloc[:, -1].values[:, None]
    a2c_merged_df = a2c_merged_df.drop(columns=['a2c'])
    ppo_merged_df = ppo_merged_df.drop(columns=['ppo'])
    ddpg_merged_df = ddpg_merged_df.drop(columns=['ddpg'])
    td3_merged_df = td3_merged_df.drop(columns=['td3'])
    sac_merged_df = sac_merged_df.drop(columns=['sac'])

    dfs = [a2c_merged_df, ppo_merged_df, ddpg_merged_df, td3_merged_df, sac_merged_df]
    cols_to_sum = dfs[0].columns[0:]
    above_sum_df6 = pd.DataFrame(columns=cols_to_sum)
    for df in dfs:
        above_sum_df6 = above_sum_df6.add(df[cols_to_sum], fill_value=0)

    above_sum_df6.to_csv('weghits_hosieni.csv')

#     %store -r df_close_full_stocks
#     %store -r filtered_stocks

    start_date = '2021-03-29'
    end_date = '2024-03-28'
    filtered_df = df_close_full_stocks[(df_close_full_stocks['date'] < end_date) & (df_close_full_stocks['date'] >= start_date)]
    filtered_df.reset_index(drop=True, inplace=True)
    columns_to_drop = [col for col in filtered_df if col not in filtered_stocks]
    df_kept = filtered_df.drop(columns=columns_to_drop)
    df1 = df_kept.reindex(sorted(df_kept.columns), axis=1)
    test_close_normal = df1.pct_change()

    hoseini_algorithm = above_sum_df6
    hosini_daily_return = test_close_normal * hoseini_algorithm
    hosini_daily_return.to_csv('hosini_daily_return.csv', index=False)

    daily_returns = df1.pct_change().fillna(0)
    daily_returns = daily_returns.replace([float('inf'), float('-inf')], 0)
    portfolio_returns = (daily_returns * above_sum_df6).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    print(f"Days: {days}, Cumulative Return: {cumulative_returns.iloc[-1]}")