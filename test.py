import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Function to calculate moving averages
def calculate_moving_averages(df, short_window, long_window):
    df['SMA50'] = df['Close'].rolling(window=short_window).mean()
    df['SMA200'] = df['Close'].rolling(window=long_window).mean()

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window, num_std_dev):
    df['MiddleBand'] = df['Close'].rolling(window=window).mean()
    df['StdDev'] = df['Close'].rolling(window=window).std()
    df['UpperBand'] = df['MiddleBand'] + (df['StdDev'] * num_std_dev)
    df['LowerBand'] = df['MiddleBand'] - (df['StdDev'] * num_std_dev)

# Function to calculate MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['SignalLine'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

# Fetch stock data
ticker = 'AAPL'  # Example ticker, replace with your stock symbol
df = yf.download(ticker, start='2020-01-01', end='2024-01-01')

# Calculate moving averages
calculate_moving_averages(df, short_window=50, long_window=200)

# Calculate Bollinger Bands
calculate_bollinger_bands(df, window=20, num_std_dev=2)

# Calculate MACD
calculate_macd(df)

# Define signals
df['Long'] = np.where((df['SMA50'] > df['SMA200']) & (df['Close'] > df['LowerBand']) & (df['MACD'] > df['SignalLine']), 1, 0)
df['Short'] = np.where((df['SMA50'] < df['SMA200']) & (df['Close'] < df['UpperBand']) & (df['MACD'] < df['SignalLine']), -1, 0)

# Visualize the strategy
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['SMA50'], label='50-day SMA', alpha=0.5)
plt.plot(df['SMA200'], label='200-day SMA', alpha=0.5)
plt.plot(df['UpperBand'], label='Upper Bollinger Band', alpha=0.3)
plt.plot(df['LowerBand'], label='Lower Bollinger Band', alpha=0.3)

# Mark Buy and Sell signals
plt.scatter(df.loc[df['Long'] == 1].index, df['Close'][df['Long'] == 1], label='Buy Signal', marker='^', color='g')
plt.scatter(df.loc[df['Short'] == -1].index, df['Close'][df['Short'] == -1], label='Sell Signal', marker='v', color='r')

plt.title('Trading Strategy')
plt.legend()
plt.show()

# Plot MACD
plt.figure(figsize=(14, 7))
plt.plot(df['MACD'], label='MACD', color='b', alpha=0.75)
plt.plot(df['SignalLine'], label='Signal Line', color='r', alpha=0.75)
plt.bar(df.index, df['MACD'] - df['SignalLine'], label='MACD Histogram', color='grey', alpha=0.3)
plt.title('MACD')
plt.legend()
plt.show()
