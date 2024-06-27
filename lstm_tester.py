# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import copy
from graphing import Stocks

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

def calculate_indicators(df):
    df.bbCalc()
    df.rsiCalc()
    #df.maCalc(13)
    #df.stochRSICalc()
    df.macdCalc()
    return

# grab data
file_path = "./prices.txt"
prcAll = loadPrices(file_path)
data = Stocks(prcAll)

# Adding indicators
calculate_indicators(data)
future_df = copy.deepcopy(data)
stock_no = 0
data = data.data[data.data['Stock'] == stock_no]
future_df.data = future_df.data[future_df.data['Stock'] == stock_no]
#print(data)

data.set_index('Day', inplace=True)
future_df.data.set_index('Day', inplace=True)

# calculate targnextclose to use as test
data = data.copy()
data['TargetNextClose'] = data['Price'].shift(-1)
future_df.data['TargetNextClose'] = future_df.data['Price'].shift(-1)

# drop NaN rows
data.dropna(inplace=True)
future_df.data.dropna(inplace=True)

# reset index starting from 0 again to assist in calculations
data.reset_index(inplace = True)
future_df.data.reset_index(inplace=True)

print(future_df.whatToGraph) # need to fix whatToGraph in graphing.py

#data_set = data.iloc[:, :]#.values
features = ['Price','21MA', 'Upper Band', 'Lower Band', 'RSI 14','MACD','MACD Signal','TargetNextClose']
data_set = data[features]
future_df.data = future_df.data[features]
#pd.set_option('display.max_columns', None)

future_df.data
#prcAll
#data_set.head(20)
#print(data_set.shape)
#print(data.shape)
#print(type(data_set))

#Target column Categories
#y =[1 if data.Open[i]>data.Close[i] else 0 for i in range(0, len(data))]
#yi = [data.Open[i]-data.Close[i] for i in range(0, len(data))]
#print(yi)
#print(len(yi))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)
print(data_set_scaled.shape)
print(data_set_scaled)

# multiple feature from data provided to the model
X = []
#print(data_set_scaled[0].size)
#data_set_scaled=data_set.values
backcandles = 60
print(data_set_scaled.shape[0])
for j in range(len(features)-1):#data_set_scaled[0].size):#2 columns are target not X
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
        X[j].append(data_set_scaled[i-backcandles:i, j])

#move axis from 0 to position 2
X=np.moveaxis(X, [0], [2])

#Erase first elements of y because of backcandles to match X length
#del(yi[0:backcandles])
#X, yi = np.array(X), np.array(yi)
# Choose -1 for last column, classification else -2...
X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
y=np.reshape(yi,(len(yi),1))
#y=sc.fit_transform(yi)
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#print(X)
print(X.shape)
#print(y)
print(y.shape)

#also comprehensions for X
#X = np.array([data_set_scaled[i-backcandles:i,:4].copy() for i in range(backcandles,len(data_set_scaled))])
#print(X)
#print(X.shape)

# split data into train test sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#print(y_train)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
from kerastuner import RandomSearch

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.optimizers import Adam
import numpy as np
#tf.random.set_seed(20)
np.random.seed(10)

"""
original lstm model

lstm_input = Input(shape=(backcandles, len(features)-1), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)
"""

def create_model(backcandles):
    dropout_level = 0.3
    lstm_input = Input(shape=(backcandles, len(features)-1), name='lstm_input')

    # first lstm layer with dropout
    x = LSTM(128, return_sequences=True, name='first_layer')(lstm_input)
    x = Dropout(dropout_level, name='first_dropout')(x)

    # second lstm layer with dropout
    x = LSTM(64, return_sequences=True, name='second_layer')(x)
    x = Dropout(dropout_level, name='second_dropout')(x)

    # third lstm layer with dropout
    x = LSTM(32, name='third_layer')(x)
    x = Dropout(dropout_level, name='third_dropout')(x)

    # dense layer and output
    x = Dense(1, name='dense_layer')(x)
    output=Activation('linear', name='output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    return model

model = create_model(backcandles)

# compile the model with the Adam optimiser
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='mse')

model.fit(x=X_train, y=y_train, batch_size=16, epochs=30, shuffle=True, validation_split=0.1)


"""
lstm tuner. attempts to find the best model


from kerastuner import RandomSearch

def build_model(hp):
    lstm_input = Input(shape=(backcandles, 5), name='lstm_input')

    x = LSTM(units=hp.Int('units1', min_value=32, max_value=256, step=32), return_sequences=True, name='first_layer')(lstm_input)
    x = Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1))(x)

    x = LSTM(units=hp.Int('units2', min_value=32, max_value=256, step=32), name='second_layer')(x)
    x = Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1))(x)

    x = Dense(1, name='dense_layer')(x)
    output = Activation('linear', name='output')(x)

    model = Model(inputs=lstm_input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')), loss='mse')

    return model

tuner = RandomSearch(build_model, objective='val_loss', max_trials=5, executions_per_trial=2)
tuner.search(X_train, y_train, epochs=30, validation_split=0.1)

model = tuner.get_best_models(num_models=1)[0]
"""

y_pred = model.predict(X_test)
#y_pred=np.where(y_pred > 0.43, 1,0)
for i in range(10):
    print(y_pred[i], y_test[i])

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(y_test, color = 'black', label = 'Test')
plt.plot(y_pred, color = 'green', label = 'Pred')
plt.legend()
#plt.show()


# predict future prices
print(future_df.data)
future_predictions = []
num_predictions = 250

for _ in range(num_predictions):
    latest_data = future_df.data.tail(backcandles)
    latest_data_scaled = sc.transform(latest_data)

    # create input sequence
    x_input = []
    for i in range(len(features)-1):
        x_input.append(latest_data_scaled[:, i])

    x_input = np.array(x_input).T
    x_input = np.expand_dims(x_input, axis=0)

    # make predictions
    predicted_scaled_price = model.predict(x_input)
    predicted_price = sc.inverse_transform(
        np.concatenate([np.zeros((predicted_scaled_price.shape[0], data_set.shape[1]-1)), predicted_scaled_price], axis=1)
    )[:, -1]

    future_predictions.append(predicted_price[0])

    # update latest data with the new predicted price
    new_row = latest_data.iloc[-1].copy()
    new_row['Price'] = predicted_price[0]

    future_df.data = pd.concat([future_df.data, pd.DataFrame([new_row])], ignore_index=True)

    filler = np.zeros(len(future_df.data),dtype=np.int8)
    future_df.data['Stock'] = filler.tolist()

    calculate_indicators(future_df)
    future_df.data['TargetNextClose'] = future_df.data['Price'].shift(-1)

    # only keep features
    future_df.data = future_df.data[features]



#print(f"Future predictions: {future_predictions}")

# Combine original data and future predictions for plotting
original_data = data_set['Price']
prediction_index = range(len(original_data), len(original_data) + num_predictions)
predictions_df = pd.DataFrame({'Price': future_predictions}, index=prediction_index)

#y_price = sc.inverse_transform(np.concatenate([np.zeros((y_pred.shape[0], data_set.shape[1]-1)), y_pred], axis=1))
# Plotting the data
#plt.figure(figsize=(12, 6))
plt.subplot(2,1,2)
plt.plot(original_data, label='Original Data')
plt.plot(predictions_df, label='Predictions', color='red')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title(f'Stock {stock_no} Price Predictions')
plt.legend()
plt.show()