import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch as pt
from pathlib import Path

class Stocks:
    def __init__(self, raw_data):
        self.data = pd.DataFrame(raw_data.T).stack().reset_index()
        self.data.columns = ['Day', 'Stock', 'Price']
        self.whatToGraph = []

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

    def bbCalc(self, ma_period=21):
        # Calculate the moving average, standard deviation, and Bollinger Bands
        self.data[f'{ma_period}MA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.rolling(window=ma_period).mean())
        sd = self.data.groupby('Stock')['Price'].transform(lambda x: x.rolling(window=ma_period).std())
        self.data['Upper Band'] = self.data[f'{ma_period}MA'] + (sd * 2)
        self.data['Lower Band'] = self.data[f'{ma_period}MA'] - (sd * 2)

        self.whatToGraph.append('Bollinger Bands')

        return
    
    def maCalc(self, ma_period=21):
        self.data[f'{ma_period}MA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.rolling(window=ma_period).mean())
        
        self.whatToGraph.append(f'{ma_period}_MA')

        return

    def rsiCalc(self, window=14):
        # calculate the price differences
        price_diff = self.data.groupby('Stock')['Price'].diff()

        # seperate gains and losses
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        avg_gain = gain.groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        avg_loss = loss.groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        rs = avg_gain/avg_loss

        rsi = 100 - (100 / (1+rs))

        self.data[f'RSI {window}'] = rsi

        self.whatToGraph.append('RSI')

        return

    def stochRSICalc(self, window=14):
        rsi_min = self.data[f'RSI {window}'].groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window,min_periods=1).min())
        rsi_max = self.data[f'RSI {window}'].groupby(self.data['Stock']).transform(lambda x: x.rolling(window=window, min_periods=1).max())
        stoch_rsi = (self.data[f'RSI {window}'] - rsi_min) / (rsi_max - rsi_min) * 100

        self.data[f'StochRSI {window}'] = stoch_rsi

        self.whatToGraph.append('StochRSI')
        return
    
    def macdCalc(self, slow_ema=26, fast_ema=12, signal=9):
        # long term ema
        self.data[f'{slow_ema}EMA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.ewm(span=slow_ema, adjust=False).mean())
        # short term ema
        self.data[f'{fast_ema}EMA'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.ewm(span=fast_ema, adjust=False).mean())

        # calculate macd line
        self.data[f'MACD'] = self.data[f'{fast_ema}EMA'] - self.data[f'{slow_ema}EMA']

        self.data['MACD Signal'] = self.data.groupby('Stock')['Price'].transform(lambda x: x.ewm(span=signal, adjust=False).mean())

        self.whatToGraph.append('MACD')

        return

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

if __name__ == '__main__':
    nInst = 50
    # read the data from the text file
    file_path = './prices.txt'   
    prcAll = loadPrices(file_path)
    ma_period = 21
    ema_period = 9
    no_sd = 2

    df = Stocks(prcAll)
    df = df.data[df.data['Stock'] == 0]
    # Set 'Day' as the index
    df.set_index('Day', inplace=True)

    # Extract only the 'Price' column and create a new DataFrame
    df = df[['Price']]
    print(df)

    #df.bbCalc()
    #df.rsiCalc()
    #df.stochRSICalc()
    #df.macdCalc()
    #output_file_path = './output.txt'
    #df.data.to_csv(output_file_path, sep='\t',index=False)
    #print(df.data)


    #dict = df.bbCalc()
    #df.bbGraph()
    #df.raw()
    #df.goldenCrossGraph()
    # df.rsiGraph()