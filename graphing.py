import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
from pathlib import Path

class Stocks:
    def __init__(self, raw_data, ma_period=21, ema_period=9, no_sd=2):
        self.data = raw_data
        self.stocks_dict = {}
        # populate dictionary with individual stocks
        for col in self.data:
            self.stocks_dict[col] = self.data[col]
        self.ma_period = ma_period
        self.ema_period = ema_period
        self.no_sd = no_sd

    def raw(self):
        # Create directory if it doesn't exist
        output_dir = Path("Raw Data")
        output_dir.mkdir(parents=True, exist_ok=True)

        for id, stock in self.stocks_dict.items():
            plt.figure(figsize=(12, 6))
            plt.plot(stock, label=f'Stock {id}', linewidth=0.5)
            plt.title(f'Stock {id} Price')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.legend(loc='upper left', fontsize='small')
            plt.grid(True)

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}.png'
            plt.savefig(plot_path)
            plt.close()

    def bbCalc(self) -> dict:
        dictionary = {}
        for id, stock in self.stocks_dict.items():
            ma = stock.rolling(window=self.ma_period).mean().dropna()
            sd = stock.rolling(window=self.ma_period).std().dropna()
            upper_band = ma + (self.no_sd*sd)
            lower_band = ma - (self.no_sd*sd)

            bollinger_bands = pd.DataFrame({
            'Price': stock,
            'Moving Average': ma,
            'Upper Band': upper_band,
            'Lower Band': lower_band
            })

            dictionary[id] = bollinger_bands
        
        return dictionary

    def bbGraph(self):

        # Create directory if it doesn't exist
        output_dir = Path("Bollinger_Bands")
        output_dir.mkdir(parents=True, exist_ok=True)
        bollinger_bands_dict = self.bbCalc()

        for id, bollinger_bands in bollinger_bands_dict.items():
            # Plotting the Bollinger Bands
            plt.figure(figsize=(12, 6))
            plt.plot(bollinger_bands['Price'], label='Price')
            plt.plot(bollinger_bands['Moving Average'], label='Moving Average', linestyle='--')
            plt.plot(bollinger_bands['Upper Band'], label='Upper Band', linestyle='--')
            plt.plot(bollinger_bands['Lower Band'], label='Lower Band', linestyle='--')
            plt.fill_between(bollinger_bands.index, bollinger_bands['Upper Band'], bollinger_bands['Lower Band'], color='gray', alpha=0.3)

            plt.title(f'Stock {id} - {self.ma_period}MA Bollinger Bands')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_{self.ma_period}MA_BB.png'
            plt.savefig(plot_path)
            plt.close()

if __name__ == '__main__':
    # read the data from the text file
    file_path = './prices.txt'   
    raw_data = pd.read_csv(file_path, sep='\s+', header=None) 
    ma_period = 21
    ema_period = 9
    no_sd = 2

    df = Stocks(raw_data, ma_period, ema_period, no_sd)

    # df.bbCalc()
    # df.bbGraph()
    #df.raw()