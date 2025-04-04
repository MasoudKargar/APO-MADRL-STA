import pandas as pd
import numpy as np
from config import config
import matplotlib.pylab as plt

import yfinance as yf
from pandas_datareader import data as pdr

# %matplotlib inline

"""### 2.2 Create Folders"""

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

"""### 2.3 Download the Dow Jines Industrial Average 30 Stocks"""

ticker_list = config.DOW_30_TICKER

#Download the data

import yfinance as yf

df = yf.download(ticker_list, start='2008-01-01', end='2024-03-29')


#df = pdr.get_data_yahoo([ticker_list][0],   start='2008-01-01', end="2021-01-01")

data = df.copy()
#data = pd.read_csv('datasets\data.csv')

data

data = data.stack().reset_index()
data.columns.names = [None]
data = data.drop(['Close'], axis=1)

data.head()

data.columns = ['date','tic','close','high','low','open','volume']

data.columns

"""### 2.4 Clean the Data"""

# check for missing values
data.isna().sum()

# check if all tickers have the same number of data points

no_datasets = []
for i in ticker_list:
    no_data_points = data[data['tic']==i].shape[0]
    no_datasets.append((i,no_data_points))
    data_points_df = pd.DataFrame(no_datasets)

data_points_df.head()

# Plot a bar chart to check that all tickers have same number of data points
plt.subplots(figsize = (8, 4))
plt.bar(data_points_df[0], data_points_df[1],alpha=0.3)
plt.xticks(rotation=90)
plt.show()

# create a list for dates with all ticker data points
date_list = list(data[data['tic']=='V'].date)
# filter with date list
data_filtered = data[data['date'].isin(date_list)]

# check if all tickers have the same number of data points

no_datasets = []
for i in ticker_list:
    no_data_points = data_filtered[data_filtered['tic']==i].shape[0]
    no_datasets.append((i,no_data_points))
    data_points_df = pd.DataFrame(no_datasets)

data_points_df.head()

# Plot a bar chart to check that all tickers have same number of data points
plt.subplots(figsize = (8, 4))
plt.bar(data_points_df[0], data_points_df[1],alpha=0.3)
plt.xticks(rotation=90)
plt.show()

data_filtered.head()

"""### 2.5 Save the Data to csv"""

data_filtered.to_csv('datasets/data.csv', index=False)

"""### 2.6 Create a Dataset for the Close Prices"""

# read the data from the saved csv file
df_prices = pd.read_csv('./datasets/data.csv')

# Reset the Index to tic and date
df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()

# Get the list of all the tickers
tic_list = list(set([i for i,j in df_prices.index]))

# Create an empty data frame for the close prices
df_close = pd.DataFrame()

len(tic_list)

# Reset the Index to tic and date
df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()

# Get all the Close Prices
df_close = pd.DataFrame()

for ticker in tic_list:
    series = df_prices.xs(ticker).close
    df_close[ticker] = series

df_close = df_close.reset_index()

df_close.head()

# Get Discriptive statistics
df_close.describe().T

# Save the Close Price datase

df_close.to_csv('datasets/close_prices.csv', index=False)

# Close prices for all the stocks
df_close_full_stocks = df_close

df_close_full_stocks.head()

# Commented out IPython magic to ensure Python compatibility.
# %store df_close_full_stocks

ticker_list = df_close_full_stocks.columns

print(ticker_list)