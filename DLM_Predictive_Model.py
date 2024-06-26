import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch as nn
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from graphing import Stocks

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    for i in range(1, n_steps+1):
        df[f'Price (t-{i})'] = df.groupby('Stock')['Price'].shift(i)
    
    df.dropna(inplace=True)

    return df


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

    df = Stocks(prcAll)
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(df.data,lookback)
    #print(shifted_df)
    
    scalar = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scalar.fit_transform(shifted_df)

    # print(shifted_df_as_np)
    
    x = shifted_df_as_np[:, 1:]
    x = dc(np.flip(x, axis=1))
    y = shifted_df_as_np[:, 0]

    split_index = int
