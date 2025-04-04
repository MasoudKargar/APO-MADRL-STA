import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras import regularizers, optimizers

from sklearn import preprocessing

"""### 3.2 Load the Data"""

# Load the close prices dataset
prices_data = pd.read_csv('./datasets/close_prices.csv')

df = prices_data.copy()

df = df.reset_index(drop=True).set_index(['date'])

df.head()

"""### 3.3 Define Functions"""

def defineAutoencoder(num_stock, encoding_dim = 5, verbose=0):

    """
    Function for fitting an Autoencoder
    """

    # connect all layers
    input = Input(shape=(num_stock,))

    encoded = Dense(encoding_dim, kernel_regularizer=regularizers.l2(0.00001),name ='Encoder_Input')(input)

    decoded = Dense(num_stock, kernel_regularizer=regularizers.l2(0.00001), name ='Decoder_Input')(encoded)
    decoded = Activation("linear", name='Decoder_Activation_function')(decoded)

    # construct and compile AE model
    autoencoder = Model(inputs=input, outputs=decoded)
    adam = optimizers.Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=adam, loss='mean_squared_error')
    if verbose!= 0:
        autoencoder.summary()

    return autoencoder

def getReconstructionErrorsDF(df_pct_change, reconstructed_data):

    """
    Function for calculating the reconstruction Errors
    """
    array = []
    stocks_ranked = []
    num_columns = reconstructed_data.shape[1]
    for i in range(0, num_columns):
        diff = np.linalg.norm((df_pct_change.iloc[:, i] - reconstructed_data[:, i]))  # 2 norm difference
        array.append(float(diff))

    ranking = np.array(array).argsort()
    r = 1
    for stock_index in ranking:
        stocks_ranked.append([ r
                              ,stock_index
                              ,df_pct_change.iloc[:, stock_index].name
                              ,array[stock_index]
                              ])
        r = r + 1

    columns = ['ranking','stock_index', 'stock_name' ,'recreation_error']
    df = pd.DataFrame(stocks_ranked, columns=columns)
    df = df.set_index('stock_name')
    return df

"""### 3.4 Get the Percentage Change of the Close Prices"""

col_names = df.columns.to_list()

print(col_names)

df_pct_change = df.pct_change(1).astype(float)
df_pct_change = df_pct_change.replace([np.inf, -np.inf], np.nan)
df_pct_change = df_pct_change.fillna(method='ffill')

# the percentage change function will make the first two rows equal to nan
df_pct_change = df_pct_change.tail(len(df_pct_change) - 2)

df_pct_change.shape

# remove columns where there is no change over a longer time period
df_pct_change = df_pct_change[df_pct_change.columns[((df_pct_change == 0).mean() <= 0.05)]]

df_pct_change.head()

"""### 3.5 Construct the Autoencoder"""

# define the input parameters
hidden_layers = 5
batch_size = 300
epochs = 500
stock_selection_number = 20
num_stock = df_pct_change.shape[1]
verbose = 1

print('-' * 20 + 'Step 1 : Returns vs. recreation error (recreation_error)')
print('-' * 25 + 'Transform dataset with MinMax Scaler')

# Normalize the data
df_scaler = preprocessing.MinMaxScaler()
df_pct_change_normalised = df_scaler.fit_transform(df_pct_change)

# define autoencoder
print('-' * 25 + 'Define autoencoder model')
num_stock = len(df_pct_change.columns)
autoencoder = defineAutoencoder(num_stock=num_stock, encoding_dim=hidden_layers, verbose=verbose)
#plot_model(autoencoder, to_file='img/model_autoencoder_1.png', show_shapes=True,
#           show_layer_names=True)

# train autoencoder
print('-' * 25 + 'Train autoencoder model')
autoencoder.fit(df_pct_change_normalised, df_pct_change_normalised, shuffle=True, epochs=epochs,batch_size=batch_size,verbose=verbose)

# predict autoencoder
print('-' * 25 + 'Predict autoencoder model')
reconstruct = autoencoder.predict(df_pct_change_normalised)

# Inverse transform dataset with MinMax Scaler
print('-' * 25 + 'Inverse transform dataset with MinMax Scaler')
reconstruct_real = df_scaler.inverse_transform(reconstruct)
df_reconstruct_real = pd.DataFrame(data=reconstruct_real, columns=df_pct_change.columns)

print('-' * 25 + 'Calculate L2 norm as reconstruction loss metric')
df_recreation_error = getReconstructionErrorsDF(df_pct_change=df_pct_change,
                                                reconstructed_data=reconstruct_real)

df_recreation_error

filtered_stocks = df_recreation_error.head(stock_selection_number).index

filtered_stocks
my_filtered_stocks = pd.DataFrame(filtered_stocks)
my_filtered_stocks.to_csv('filtered_stocks.csv')

# Commented out IPython magic to ensure Python compatibility.
# store the list of selected stocks
# %store filtered_stocks