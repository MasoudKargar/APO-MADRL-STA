import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from numpy import array
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from sklearn import preprocessing

"""### 5.2 Load the Data"""

# Commented out IPython magic to ensure Python compatibility.
df = pd.read_csv('./df.csv')
# %store -r df

df.head()

# Get the list of all the features
features_list = list(df.columns)[7:-1]

print(features_list)

"""### 5.2 Construct a Data Frame of Features"""

# construct a data frame of features
features_df = df[features_list]
features_df.index = df['date']

features_df.head()

features_df.shape

df.shape

"""### 5.3 Normalize the Features and Construct an Autoencoder"""

features_array = np.array(features_df)

features_scaler = preprocessing.MinMaxScaler()
features_normalised = features_scaler.fit_transform(features_array)

features_normalised.shape

# rescale the features array
features_normalised = features_normalised.reshape(-1,20,10)

features_normalised.shape

# define model
model = Sequential()
model.add(LSTM(4, activation='relu', input_shape=(20,10)))
model.add(RepeatVector(20))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(10)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(features_normalised, features_normalised, epochs=200, verbose=1)
#plot_model(model, show_shapes=True, to_file='./results/reconstruct_lstm_autoencoder.png')

model.summary()

# connect the encoder LSTM as the output layer
model_feature = Model(inputs=model.inputs, outputs=model.layers[1].output)
#plot_model(model_feature, show_shapes=True, show_layer_names=True, to_file='./results/lstm_encoder.png')

model_feature.summary()

"""### 5.4 Get the Reconstructed Features"""

# get the feature vector for the input sequence
yhat = model_feature.predict(features_normalised)
print(yhat.shape)

# reshape the vector
features_reduced = yhat.reshape(-1,4)

df.shape

features_reduced.shape

# Copy original data frame and drop the original features
df_reduced = df.copy()
df_reduced = df_reduced .drop(features_list, axis=1)

df_reduced.head()

# convert the reduced features to a data frame and merge with the original data frame
features_reduced_df = pd.DataFrame(features_reduced, columns=['f01','f02','f03','f04'])

features_reduced_df.head()

df_reduced[['f01','f02','f03','f04']] = features_reduced_df[['f01','f02','f03','f04']]

df_reduced.head()

data_df = df_reduced.copy()

# Commented out IPython magic to ensure Python compatibility.
# %store data_df
data_df.to_csv('./data_df.csv')