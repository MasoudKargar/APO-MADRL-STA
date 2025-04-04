import pandas as pdp
import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna

from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer
from pickleshare import PickleShareDB

"""### 4.2 Load the Data from the csv Files"""

# Load the whole data set
data = pdp.read_csv('./datasets/data.csv')

# Load the close prices dataset
prices_data = pdp.read_csv('./datasets/close_prices.csv')

# Commented out IPython magic to ensure Python compatibility.
filtered_stocks = pd.read_csv('filtered_stocks.csv')
filtered_stocks = filtered_stocks.drop(columns=['Unnamed: 0'])
filtered_stocks = filtered_stocks['stock_name'].tolist()
# %store filtered_stocks

list_of_stocks = filtered_stocks
print(list_of_stocks)

data.head()

data = data[data['tic'].isin(list_of_stocks)]

data.tic.unique()

"""### 4.3 Add Technical Indicators
---
We define a function to add technical indicators to the dataset by making use of the ta library

The folloing indicators are considered:
* Volatility Average True Range (ATR)
* Volatility Bollinger Band Width (BBW)
* Volume On-balance Volume (OBV
* Volume Chaikin Money Flow (CMF)
* Trend Moving Average Convergence Divergence (MACD)
* Trend Average Directional Index (ADX)
* Trend Fast Simple Moving Average (SMA)
* Trend Fast Exponential Moving Average (EMA)
* Trend Commodity Channel Index (CCI)
* Momentum Relative Strength Index (RSI)
"""

# Define a Function for adding technical indicators

def add_features(data, feature_list, short_names):
    """
    Function to add technical indicators for features
    -Takes in a dataset with Open, High, Low, Close and Volume
    -Also takes in a list of the technical indicators to be added
     as well as a list of the shortened indicator names
    """

    # list of column names to filter the features
    data_col_names = list(data.columns)
    filter_names = data_col_names + feature_list
    col_rename = data_col_names +  short_names

    # Add technical indicators using the ta Library
    data = add_all_ta_features(data, open="open", high="high",
    low="low", close="close", volume="volume")

    # Filter the Indicators with the required features
    data = data[filter_names]
    data.columns = col_rename # rename the columns to use shortened indicator names
    data = data.dropna()

    return data

# List of Features to add
feature_list= ['volatility_atr','volatility_bbw','volume_obv','volume_cmf',
               'trend_macd', 'trend_adx', 'trend_sma_fast',
               'trend_ema_fast', 'trend_cci', 'momentum_rsi']

# Short names of the features
short_names = ['atr', 'bbw','obv','cmf','macd', 'adx', 'sma', 'ema', 'cci', 'rsi']

#feature_list= ['volatility_atr','volatility_bbw','volume_obv','volume_cmf','trend_macd']

# Short names of the features
#short_names = ['atr', 'bbw','obv','cmf','macd']

# Add Indicators to our dataset
data_with_features = data.copy()

data_with_features = add_features(data_with_features, feature_list, short_names)

data_with_features.head()

feature_list = list(data_with_features.columns)[7:]

print(feature_list)

"""### 4.4 Add Covariance Matrix
---
We define a function that will add Covarance Matrices to our dataset
"""

def add_cov_matrix(df):
    """
    Function to add Coveriance Matrices as part of the defined states
    """
    # Sort the data and index by date and tic
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = [] # create empty list for storing coveriance matrices at each time step

    # look back for constructing the coveriance matrix is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values
        covs = covs#/covs.max()
        cov_list.append(covs)

    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)

    return df

# Add Covariance Matrices to our dataset
data_with_features_covs = data_with_features.copy()
data_with_features_covs = add_cov_matrix(data_with_features_covs)

data_with_features_covs.head()

"""### 4.6 Store the Dataframe"""

df = data_with_features_covs

# Commented out IPython magic to ensure Python compatibility.
df.to_csv('df.csv', index=False)
# %store df