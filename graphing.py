import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
from pathlib import Path

# read the data from the text file
file_path = 'prices.txt'
data = pd.read_csv(file_path, delim_whitespace=True, header=None)
ma_period = 9
ema_period = 9
no_sd = 2

# average % price change of all stock on each day
price_changes = data.diff().dropna()
price_changes_percent = price_changes.div(data.shift(1)).dropna() * 100
# calculate the average price movement across all stocks
mean_price_changes_percent = price_changes_percent.mean(axis=1)

# Moving Average
ma = data.rolling(window=ma_period).mean().dropna()

# Exponential Moving Average
ema = data.ewm(span=ema_period, adjust=False).mean()

# bollinger bands
stocks_dict = {}
# populate dictionary with individual stocks
for col in data.columns:
    stocks_dict[col] = data[col]

# Create directory if it doesn't exist
output_dir = Path("bollinger_band")
output_dir.mkdir(parents=True, exist_ok=True)

for id, stock in stocks_dict.items():
    ma = stock.rolling(window=ma_period).mean().dropna()
    sd = stock.rolling(window=ma_period).std().dropna()
    upper_band = ma + (no_sd*sd)
    lower_band = ma - (no_sd*sd)

    bollinger_bands = pd.DataFrame({
    'Price': stock,
    'Moving Average': ma,
    'Upper Band': upper_band,
    'Lower Band': lower_band
    })

    # Plotting the Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(bollinger_bands['Price'], label='Price')
    plt.plot(bollinger_bands['Moving Average'], label='Moving Average', linestyle='--')
    plt.plot(bollinger_bands['Upper Band'], label='Upper Band', linestyle='--')
    plt.plot(bollinger_bands['Lower Band'], label='Lower Band', linestyle='--')
    plt.fill_between(bollinger_bands.index, bollinger_bands['Upper Band'], bollinger_bands['Lower Band'], color='gray', alpha=0.3)

    plt.title(f'Stock {id} Bollinger Bands')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot to the directory
    plot_path = output_dir / f'stock_{id}_bollinger_bands.png'
    plt.savefig(plot_path)
    plt.close()

# to graph
chart_data = ema


plt.figure()
for stock in chart_data.columns:
    plt.plot(chart_data[stock], label=f'Stock {stock}', linewidth=0.5)
plt.title('Exponential Moving Average of Stocks (9 days)')
plt.xlabel('Day')
plt.ylabel('Price')
# plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.close()