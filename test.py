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
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import random
from pathlib import Path

import psutil
import gc
from datetime import datetime

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

STOCK_NO = 10
    
def lstm(prcAll, STOCK_NO):
    def calculate_indicators(df):
        df.bbCalc()
        df.rsiCalc()
        #df.maCalc(13)
        #df.stochRSICalc()
        df.macdCalc()
        return

    data = Stocks(prcAll)

    # Adding indicators
    calculate_indicators(data)
    future_df = copy.deepcopy(data)
    data = data.data[data.data['Stock'] == STOCK_NO]
    future_df.data = future_df.data[future_df.data['Stock'] == STOCK_NO]
    #print(data)

    # calculate volatility of stock
    data['Returns'] = data['Price'].pct_change()

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
    
    # after we dropna, we can calculate volatility
    volatility = data['Returns'].std()
    #print(volatility)

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
    # To inverse transform y_pred to the original scale
    y_pred_original_scale = sc.inverse_transform(np.concatenate((np.zeros((len(y_pred), data_set.shape[1] - 1)), y_pred), axis=1))[:, -1]
    y_test_original_scale = sc.inverse_transform(np.concatenate((np.zeros((len(y_test), data_set.shape[1] - 1)), y_test), axis=1))[:, -1]

    #print(type(y_pred_original_scale))
    #for i in range(10):
    #    print(y_pred[i], y_test[i])

    plt.figure(figsize=(16,8))
    plt.subplot(2,1,1)
    plt.plot(y_test, color = 'black', label = 'Test')
    plt.plot(y_pred, color = 'green', label = 'Pred')
    plt.legend()

    """ Predict Future Prices """
    future_predictions = []

    for _ in range(NUM_PREDICTIONS+len(data_set)-BACK_CANDLES):
        latest_data = future_df.data.head(BACK_CANDLES)
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

        future_df.data.loc[i] = pd.DataFrame([new_row])
        #future_df.data = pd.concat([future_df.data, pd.DataFrame([new_row])], ignore_index=True)

        filler = np.zeros(len(future_df.data),dtype=np.int8)
        future_df.data['Stock'] = STOCK_NO

        calculate_indicators(future_df)
        future_df.data['TargetNextClose'] = future_df.data['Price'].shift(-1)

        # only keep features
        future_df.data = future_df.data[features]

    # Combine original data and future predictions for plotting
    original_data = data_set['Price']
    prediction_index = range(BACK_CANDLES, len(original_data) + NUM_PREDICTIONS)
    predictions_df = pd.DataFrame({'Price': future_predictions}, index=prediction_index)

    # TODO return these to calculate r2 of model
    y_pred_original_scale = y_pred_original_scale.reshape(-1, 1)
    #print(y_pred_original_scale)
    #print(np.array(future_df.data.tail(250).Price).reshape(-1, 1).shape)
    all_predicted_values = np.array(future_df.data)
    np.concatenate((y_pred_original_scale, np.array(future_df.data.tail(250).Price).reshape(-1, 1)), axis=0)
    noise = np.random.uniform(-volatility, volatility, all_predicted_values.shape)
    all_predicted_values += noise
    #print(all_predicted_values)
    # prediction_index = range(len(original_data)-len(y_test), len(original_data) + NUM_PREDICTIONS)
    plt.subplot(2,1,2)
    plt.plot(original_data, label='Original Data')
    plt.plot(prediction_index, all_predicted_values, label='Predictions', color='red')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.title(f'Stock {STOCK_NO} Price Predictions')
    plt.legend()
    plt.close()
    #plt.show()

    # Clear session to free memory
    tf.keras.backend.clear_session()
    # Call garbage collector
    gc.collect()

    return all_predicted_values, y_pred_original_scale, y_test_original_scale, original_data, predictions_df
    
# Set seeds for reproducibility
def set_seeds(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

if __name__ == "__main__":
    set_seeds()

    # fetch data
    file_path = "./prices.txt"
    prcAll = loadPrices(file_path)

    """ run one specific configuration only 
    BACK_CANDLES = 60 # 30, 40, 50, 60, 70, 80, 90, 100
    BATCH_SIZE = 8 # 16, 32, 64, 128
    EPOCHS = 30 # 20, 25, 30, 35, 40, 45, 50
    DROPOUT_LEVEL = 0.3 # 0.1, 0.2, 0.3, 0.4, 0.5
    UNITS_LSTM1 = 128 # 256, 128, 64, 32, 16, 8, 4
    UNITS_LSTM2 = 64 # 128, 64, 32, 16, 8, 4, 2
    UNITS_LSTM3 = 32 # 64, 32, 16, 8, 4, 2, 1
    LEARNING_RATE = 0.001 # 1e-5, 1e-4, 1e-3, 1e-2, 1e-1

    r2_list = []
    rmse_list = []
    hyperparameter_dict = {}
    output_dir = Path("LSTM Output")
    output_dir.mkdir(parents=True, exist_ok=True)


    # Create directory if it doesn't exist
    file_name = f"{BACK_CANDLES}_backcandles"
    output_file = output_dir / f"{file_name}.txt"
    file = open(output_file, "a")
    counter = 0
    file.write("\n")
    file.write(f"--- Independent Variable---\nBack Candle : {BACK_CANDLES}\n\n")
    file.write(f"--- Control Variables ---\nBatch Size: {BATCH_SIZE}\nEpochs: {EPOCHS}\nDropout Level: {DROPOUT_LEVEL}\nLearning Rate: {LEARNING_RATE}\nLSTM Layer 1: {UNITS_LSTM1}\nLSTM Layer 2: {UNITS_LSTM2}\nLSTM Layer 3: {UNITS_LSTM3}\n\n")
        
    for i in range(50):
        print(f"Stock No {i}")
        pred_values, y_pred, y_test, og_data, pred_df = lstm(prcAll, i)
        
        # use y_test and x_test to calculate r2
        r2_res = r2_score(y_test, y_pred)
        r2_list.append(r2_res)

        scaler = MinMaxScaler()
        y_test_normalised = scaler.fit_transform(y_test.reshape(-1,1))
        y_pred_normalised = scaler.transform(y_pred.reshape(-1,1))
        mse = mean_squared_error(y_test_normalised, y_pred_normalised)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        if counter % 3 == 0 and counter != 0:
            file.write("\n")
        file.write(f"Stock {i} Normalised RMSE score: {rmse}\n")
    """

    """ finding best configuration for data """
    back_candles_list = [80,90,100]
    batch_size_list = [8, 16, 32, 64, 128]
    epochs_list = [20, 25, 30, 35, 40, 45, 50]
    dropout_level_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    units_lstm_list = [(256,128,64),(128,64,32),(64,32,16),(32,16,8),(16,8,4),(8,4,2),(4,2,1)]
    learning_rate_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    """
    for backcandle in back_candles_list:
        for batchsize in batch_size_list:
            for epochs in epochs_list:
                for dropoutlvl in dropout_level_list:
                    for learningrate in learning_rate_list:
                        for unitslstm in units_lstm_list:
                            "work"
                            BACK_CANDLES = backcandle
                            BATCH_SIZE = batchsize
                            EPOCHS = epochs
                            DROPOUT_LEVEL = dropoutlvl
                            LEARNING_RATE = learningrate
                            UNITS_LSTM1 = 128 # 256, 128, 64, 32, 16, 8, 4
                            UNITS_LSTM2 = 64 # 128, 64, 32, 16, 8, 4, 2
                            UNITS_LSTM3 = 32 # 64, 32, 16, 8, 4, 2, 1
    
    TOO INEFFICIENT. WILL RUN FOR MORE THAN 17 DAYS
    """
    output_dir = Path("LSTM Output")
    output_dir.mkdir(parents=True, exist_ok=True)

    """ FINDING WHICH VALUE WORKS BEST FOR EACH HYPERPARAMETER """
    avg_rmse_list = []
    for backcandle in back_candles_list:
        # continue
        BACK_CANDLES = 16
        BATCH_SIZE = 16
        EPOCHS = 30
        DROPOUT_LEVEL = 0.3
        UNITS_LSTM1 = 128
        UNITS_LSTM2 = 64
        UNITS_LSTM3 = 32
        LEARNING_RATE = 0.001
        # Create directory if it doesn't exist
        file_name = f"{backcandle}_backcandles"
        output_file = output_dir / f"{file_name}.txt"
        # delete file
        # file = open(output_file, "w")
        file = open(output_file, "a")
        r2_list = []
        rmse_list = []
        counter = 0
        file.write("\n")
        file.write(f"{datetime.now()}")
        file.write(f"--- Independent Variable---\nBack Candle : {BACK_CANDLES}\n\n")
        file.write(f"--- Control Variables ---\nBatch Size: {BATCH_SIZE}\nEpochs: {EPOCHS}\nDropout Level: {DROPOUT_LEVEL}\nLearning Rate: {LEARNING_RATE}\nLSTM Layer 1: {UNITS_LSTM1}\nLSTM Layer 2: {UNITS_LSTM2}\nLSTM Layer 3: {UNITS_LSTM3}\n\n")
            
        for i in range(50):
            print(f"Stock No {i}")
            pred_values, y_pred, y_test, og_data, pred_df = lstm(prcAll, i)
            
            # use y_test and x_test to calculate r2
            r2_res = r2_score(y_test, y_pred)
            r2_list.append(r2_res)

            scaler = MinMaxScaler()
            y_test_normalised = scaler.fit_transform(y_test.reshape(-1,1))
            y_pred_normalised = scaler.transform(y_pred.reshape(-1,1))
            mse = mean_squared_error(y_test_normalised, y_pred_normalised)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)

            if counter % 3 == 0 and counter != 0:
                file.write("\n")
            file.write(f"Stock {i} Normalised RMSE score: {rmse}\t")
            counter += 1

            # Memory usage diagnostics
            process = psutil.Process()
            print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

            #print(f'Stock {i} r2 score: {r2_res}')
            #print(f'Stock {i} Normalised rmse score: {rmse}')
        file.write("\n\n")
        plt.figure(figsize=(16,8))
        plt.plot(rmse_list, label='RMSE')
        plt.xlabel('Stock No')
        plt.ylabel('RMSE')
        plt.title(f'Model with {backcandle} Back Candles')
        plt.legend()
        plot_path = output_dir / f'{file_name}.png'
        plt.savefig(plot_path)
        plt.close()
        
        avg_rmse = np.mean(rmse_list)
        file.write(f"Average RMSE: {avg_rmse}")
    
    file.write(f"Lowest RMSE: {min(avg_rmse_list)}")
    file.close()


