import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt # Visualization 
import matplotlib.dates as mdates # Formatting dates
import seaborn as sns # Visualization
from sklearn.preprocessing import MinMaxScaler
import torch # Library for implementing Deep Neural Network 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Loading the Apple.Inc Stock Data

import yfinance as yf
from datetime import date, timedelta, datetime 

end_date = date.today().strftime("%Y-%m-%d") #end date for our data retrieval will be current date 
start_date = '1990-01-01' # Beginning date for our historical data retrieval 

df = yf.download('AAPL', start=start_date, end=end_date)# Function used to fetch the data 

def data_plot(df):
	df_plot = df.copy()

	ncols = 2
	nrows = int(round(df_plot.shape[1] / ncols, 0))

	fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
						sharex=True, figsize=(14, 7))
	for i, ax in enumerate(fig.axes):
		sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
		ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
		ax.xaxis.set_major_locator(mdates.AutoDateLocator())
	fig.tight_layout()
	plt.show()

#data_plot(df)
df.reset_index(inplace=True)
df['Day'] = (df['Date'] - df['Date'].min()).dt.days
df.set_index('Day', inplace=True)
# Drop the original Date column if no longer needed
df.drop(columns=['Date'], inplace=True)

data_plot(df)
print(df)
