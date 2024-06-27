# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import copy
from graphing import Stocks

import tensorflow as tf
import keras
from keras import optimizers
from kerastuner import RandomSearch
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random

""" Parameters """

SEED = 10

# all models
BACK_CANDLES = 60 # how many candles to use as input
SPLIT_RATIO = 0.8 # splitting between training and test
VALIDATION_SPLIT = 0.1

# Hyperparameters for multi layer model

BATCH_SIZE = 16
EPOCHS = 30
DROPOUT_LEVEL = 0.3
UNITS_LSTM1 = 128
UNITS_LSTM2 = 64
UNITS_LSTM3 = 32
LEARNING_RATE = 0.001

NUM_PREDICTIONS = 250 # future predictions

STOCK_NO = 0

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

# Set seeds for reproducibility
def set_seeds(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds()

# fetch data
file_path = "./prices.txt"
prcAll = loadPrices(file_path)
data = Stocks(prcAll)

# Adding indicators
calculate_indicators(data)
future_df = copy.deepcopy(data)
data = data.data[data.data['Stock'] == STOCK_NO]
future_df.data = future_df.data[future_df.data['Stock'] == STOCK_NO]
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
print(future_df.whatToGraph) # TODO need to fix whatToGraph in graphing.py need to match features below

features = ['Price','21MA', 'Upper Band', 'Lower Band', 'RSI 14','MACD','MACD Signal','TargetNextClose']
data_set = data[features]
future_df.data = future_df.data[features]

# normalise the data
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)

# Preparing the feature set for the model
X = []

# Generate the feature set for each feature except the last one (which is the target)
for j in range(len(features) - 1):
    X.append([data_set_scaled[i - BACK_CANDLES:i, j] for i in range(BACK_CANDLES, data_set_scaled.shape[0])])

# Convert X to a numpy array and move the axis
X = np.moveaxis(X, [0], [2])

# Preparing the target variable
y = data_set_scaled[BACK_CANDLES:, -1].reshape(-1, 1)

X = np.array(X)
y = np.array(y)

# split data into train test sets
splitlimit = int(len(X) * SPLIT_RATIO)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

#original lstm model
def og_model():
    original_lstm_input = Input(shape=(BACK_CANDLES, len(features)-1), name='lstm_input')
    original_inputs = LSTM(150, name='first_layer')(original_lstm_input)
    original_inputs = Dense(1, name='dense_layer')(original_inputs)
    original_output = Activation('linear', name='output')(original_inputs)
    original_model = Model(inputs=original_lstm_input, outputs=original_output)
    original_adam = optimizers.Adam()
    original_model.compile(optimizer=original_adam, loss='mse')
    original_model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split = VALIDATION_SPLIT)
    return original_model

#original_model = og_model()

def create_model(backcandles=BACK_CANDLES):
    lstm_input = Input(shape=(backcandles, len(features)-1), name='lstm_input')

    # first lstm layer with dropout
    x = LSTM(UNITS_LSTM1, return_sequences=True, name='first_layer')(lstm_input)
    x = Dropout(DROPOUT_LEVEL, name='first_dropout')(x)

    # second lstm layer with dropout
    x = LSTM(UNITS_LSTM2, return_sequences=True, name='second_layer')(x)
    x = Dropout(DROPOUT_LEVEL, name='second_dropout')(x)

    # third lstm layer with dropout
    x = LSTM(UNITS_LSTM3, name='third_layer')(x)
    x = Dropout(DROPOUT_LEVEL, name='third_dropout')(x)

    # dense layer and output
    x = Dense(1, name='dense_layer')(x)
    output=Activation('linear', name='output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    return model

model = create_model()
# compile the model with the Adam optimiser
adam = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=VALIDATION_SPLIT)


""" LSTM Tuner. Attempts to find the best model """
def build_model(hp):
    lstm_input = Input(shape=(BACK_CANDLES, 5), name='lstm_input')

    x = LSTM(units=hp.Int('units1', min_value=32, max_value=256, step=32), return_sequences=True, name='first_layer')(lstm_input)
    x = Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1))(x)

    x = LSTM(units=hp.Int('units2', min_value=32, max_value=256, step=32), name='second_layer')(x)
    x = Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1))(x)

    x = Dense(1, name='dense_layer')(x)
    output = Activation('linear', name='output')(x)

    model = Model(inputs=lstm_input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')), loss='mse')

    return model

def tuned_model():
    tuner = RandomSearch(tuned_model, objective='val_loss', max_trials=5, executions_per_trial=2)
    tuner.search(X_train, y_train, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

    model = tuner.get_best_models(num_models=1)[0]
    
    return model

# model = turned_model()

""" Test Data """
y_pred = model.predict(X_test)
for i in range(10):
    print(y_pred[i], y_test[i])

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(y_test, color = 'black', label = 'Test')
plt.plot(y_pred, color = 'green', label = 'Pred')
plt.legend()

""" Predict Future Prices """
future_predictions = []

for _ in range(NUM_PREDICTIONS):
    latest_data = future_df.data.tail(BACK_CANDLES)
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

# Combine original data and future predictions for plotting
original_data = data_set['Price']
prediction_index = range(len(original_data), len(original_data) + NUM_PREDICTIONS)
predictions_df = pd.DataFrame({'Price': future_predictions}, index=prediction_index)

plt.subplot(2,1,2)
plt.plot(original_data, label='Original Data')
plt.plot(predictions_df, label='Predictions', color='red')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title(f'Stock {STOCK_NO} Price Predictions')
plt.legend()
plt.show()