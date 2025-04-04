import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from finrl.preprocessing.data import data_split
from finrl.preprocessing.preprocessors import FeatureEngineer

df_close_full_stocks = pd.read_csv('datasets/close_prices.csv')

"""### 6.2 Load the data"""

# Commented out IPython magic to ensure Python compatibility.
filtered_stocks = pd.read_csv('filtered_stocks.csv')
# df_close_full_stocks = pd.read_csv('datasets/close_prices.csv')
data_df = pd.read_csv('data_df.csv')
filtered_stocks = filtered_stocks.drop(columns=['Unnamed: 0'])
filtered_stocks = filtered_stocks['stock_name'].tolist()
data_df = data_df.drop(columns=['Unnamed: 0'])

#%store filtered_stocks
#%store df_close_full_stocks
#%store data_df

# %store -r filtered_stocks
# %store -r df_close_full_stocks
# %store -r data_df

data_df.head()

df_close_full_stocks.head()

# Close Prices data frame

# Reset the Index to tic and date
df_prices = data_df.reset_index().set_index(['tic', 'date']).sort_index()

# Get all the Close Prices
df_close = pd.DataFrame()

for ticker in filtered_stocks:
    series = df_prices.xs(ticker).close
    df_close[ticker] = series

df_close.head()

df_close = df_close.reset_index()

"""### 6.3 Split the Data"""

# Define the start and end dates for the train and test data

train_pct = 0.8 # percentage of train data

date_list = list(data_df.date.unique()) # List of dates in the data

date_list_len = len(date_list) # len of the date list
train_data_len = int(train_pct * date_list_len) # length of the train data

train_start_date = date_list[0]
train_end_date = date_list[train_data_len]

test_start_date = date_list[train_data_len+1]
test_end_date = date_list[-1]

print('Training Data: ', 'from ', train_start_date, ' to ', train_end_date)

print('Testing Data: ', 'from ', test_start_date, ' to ', test_end_date)

df_close_full_stocks

# Split the whole dataset
train_data = data_split(data_df, train_start_date, train_end_date)
test_data = data_split(data_df, test_start_date, test_end_date)

# Split the Close Prices dataset
prices_train_data = df_close[df_close['date']<=train_end_date]
prices_test_data = df_close[df_close['date']>=test_start_date]

# split the Close Prices of all stocks
prices_full_train = df_close_full_stocks[df_close_full_stocks['date']<=train_end_date]
prices_full_test = df_close_full_stocks[df_close_full_stocks['date']>=test_start_date]

"""### 6.4 Store the Dataframes"""

prices_train = prices_train_data.copy()
prices_test = prices_test_data.copy()

train_df = train_data.copy()
test_df = test_data.copy()

prices_full_train_df = prices_full_train.copy()
prices_full_test_df = prices_full_test.copy()

# Commented out IPython magic to ensure Python compatibility.
prices_train.to_csv('./prices_train.csv')
prices_test.to_csv('./prices_test.csv')

train_df.to_csv('./train_df.csv')
test_df.to_csv('./test_df.csv')

prices_full_train_df.to_csv('./prices_full_train_df.csv')
prices_full_train_df.to_csv('./prices_full_test_df.csv')

# %store prices_train
# %store prices_test

# %store train_df
# %store test_df

# %store prices_full_train_df
# %store prices_full_test_df

# Commented out IPython magic to ensure Python compatibility.
# %store df_close_full_stocks