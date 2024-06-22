import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
from pathlib import Path

class Stocks:
    def __init__(self, raw_data, ma_period=21, ema_period=9, no_sd=2):
        self.data = pd.DataFrame(raw_data.T)
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

        for id, stock_price in self.stocks_dict.items():
            plt.figure(figsize=(12, 6))
            plt.plot(stock_price, label=f'Stock {id}', linewidth=0.5)
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
        for id, stock_price in self.stocks_dict.items():
            ma = stock_price.rolling(window=self.ma_period).mean().dropna()
            sd = stock_price.rolling(window=self.ma_period).std().dropna()
            upper_band = ma + (self.no_sd*sd)
            lower_band = ma - (self.no_sd*sd)

            bollinger_bands = pd.DataFrame({
            'Price': stock_price,
            'Moving Average': ma,
            'Upper Band': upper_band,
            'Lower Band': lower_band
            })

            dictionary[id] = bollinger_bands
        
        return dictionary

    def bbGraph(self):

        # Create directory if it doesn't exist
        output_dir = Path("Bollinger Bands")
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

            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - {self.ma_period}MA Bollinger Bands')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_{self.ma_period}MA_BB.png'
            plt.savefig(plot_path)
            plt.close()

    def goldenCrossCalc(self) -> dict:
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            ma50 = stock_price.rolling(window=50).mean().dropna()
            ma200 = stock_price.rolling(window=200).mean().dropna()

            goldenCross = pd.DataFrame({
                'Price': stock_price,
                '50 Day MA': ma50,
                '200 Day MA': ma200 
            })

            dictionary[id] = goldenCross
        
        return dictionary
    
    def goldenCrossGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("Golden Cross")
        output_dir.mkdir(parents=True, exist_ok=True)
        golden_cross_dict = self.goldenCrossCalc()

        for id, golden_cross in golden_cross_dict.items():
            # Plotting the Bollinger Bands
            plt.figure(figsize=(12, 6))
            plt.plot(golden_cross['Price'], label='Price')
            plt.plot(golden_cross['50 Day MA'], label='50 Day MA')
            plt.plot(golden_cross['200 Day MA'], label='200 Day MA')

            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - Golden Crossover')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_GC.png'
            plt.savefig(plot_path)
            plt.close()

    def rsiCalc(self, window=14) -> dict:
        dictionary = {}
        for id, stock_price in self.stocks_dict.items():
            delta = stock_price.diff()

            # seperate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()

            rs = avg_gain/avg_loss

            rsi = 100 - (100 / (1+rs))

            rsi_df = pd.DataFrame({
            'Price': stock_price,
            'RSI': rsi
            })

            dictionary[id] = rsi_df
        
        return dictionary
    
    def rsiGraph(self):
        # Create directory if it doesn't exist
        output_dir = Path("Relative Strength Index")
        output_dir.mkdir(parents=True, exist_ok=True)
        rsi_dict = self.rsiCalc()

        for id, rsi in rsi_dict.items():
            plt.figure(figsize=(12, 6))
            
            # Plotting the price
            plt.subplot(2,1,1)
            plt.plot(rsi['Price'], label='Price')
            # Add a vertical line at id=250
            plt.axvline(x=250, color='red', linestyle='--', linewidth=1)

            plt.title(f'Stock {id} - RSI')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()

            # plotting the rsi
            plt.subplot(2,1,2)
            plt.plot(rsi['RSI'], label='RSI', color='blue')
            plt.axhline(y=30, color='red', linestyle='--', label='Oversold (30)')
            plt.axhline(y=70, color='green', linestyle='--', label='Overbought (70)')
            plt.ylabel('RSI')
            plt.grid()
            plt.tight_layout()

            # Save the plot to the directory
            plot_path = output_dir / f'stock{id}_RSI.png'
            plt.savefig(plot_path)
            plt.close()

    def stochRsiCalc(self, rsi, window=14) -> dict:
        dictionary = {}
        rsi_dict = rsi
        for id, stock_price in rsi_dict.items():
            stoch_rsi = (rsi - rsi.rolling(window=window, min_periods=1).min()) / (rsi.rolling(window=window, min_periods=1).max() - rsi.rolling(window=window, min_periods=1).min())*100
            
            stoch_rsi_df = pd.DataFrame({
            'Price': stock_price,
            'Stochastic RSI': stoch_rsi
            })

            dictionary[id] = stoch_rsi_df
        
        return dictionary

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

if __name__ == '__main__':
    # read the data from the text file
    file_path = './prices.txt'   
    prcAll = loadPrices(file_path)
    ma_period = 21
    ema_period = 9
    no_sd = 2

    df = Stocks(prcAll, ma_period, ema_period, no_sd)

    dict = df.bbCalc()
    #df.bbGraph()
    #df.raw()
    #df.goldenCrossGraph()
    df.rsiGraph()