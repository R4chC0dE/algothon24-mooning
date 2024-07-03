import pandas as pd
import numpy as np
from graphing import Stocks

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
entryInfo = {}

def getMyPosition(prcSoFar):
    global currentPos, entryInfo

    # bollinger bands variables
    ma_period = 21
    # rsi variables
    rsi_window = 14
    overbought_rsi = 70
    oversold_rsi = 30
    overbought_s_rsi = 80
    oversold_s_rsi = 20
    # macd variables
    slow_ema = 26
    fast_ema = 12
    macd_signal_length = 9

    df = Stocks(prcSoFar)
    df.bbCalc(ma_period)
    df.rsiCalc(rsi_window)
    df.stochRSICalc(rsi_window)
    df.macdCalc(slow_ema, fast_ema, macd_signal_length)

    today_data = df.data.groupby('Stock').tail(1).reset_index(drop=True)
    today_price = today_data['Price'] # will return as dataframe rather than series
    #print(today_price)
    posLimits = np.array([int(x) for x in 10000 / today_price])
    #print(posLimits[5])
    #print(today_price)
    
    for stock_id, stock_data in today_data.groupby('Stock'):
        longOrShort_dict = {}
        price = float(stock_data['Price'].iloc[0])
        posLimit = int(10000 / price)
        currPos = currentPos[stock_id]

        # Bollinger Bands
        ub = float(stock_data['Upper Band'].iloc[0])
        lb = float(stock_data['Lower Band'].iloc[0])

        if price <= lb:
            longOrShort_dict['BB'] = 'Long'
        elif price >= ub:
            longOrShort_dict['BB'] = 'Short'

        # RSI
        rsi = float(stock_data[f'RSI {rsi_window}'].iloc[0])

        if rsi < oversold_rsi:
            longOrShort_dict['RSI'] = 'Long'
        elif rsi > overbought_rsi:
            longOrShort_dict['RSI'] = 'Short'

        # Stochastic RSI
        s_rsi = float(stock_data[f'StochRSI {rsi_window}'].iloc[0])

        if s_rsi < oversold_s_rsi:
            longOrShort_dict['Stoch RSI'] = 'Long'
        elif s_rsi > overbought_s_rsi:
            longOrShort_dict['Stoch RSI'] = 'Short'

        # MACD
        macd = float(stock_data['MACD'].iloc[0])
        macd_signal = float(stock_data['MACD Signal'].iloc[0])

        macd_diff = macd - macd_signal
        if macd_diff >= 0:
            longOrShort_dict['MACD'] = 'Long'
        else:
            longOrShort_dict['MACD'] = 'Short'

        # position size calculation
        max_confluence = 4
        long_strat = []
        short_strat = []
        for strat, condition in longOrShort_dict.items():
            if condition == 'Long':
                long_strat.append(strat)
            elif condition == 'Short':
                short_strat.append(strat)
        
        long_counter = len(long_strat)
        short_counter = len(short_strat)

        if currPos == 0: # if not in a position
            if long_counter >= short_counter and long_counter > 0:
                position_size = posLimit * (long_counter/max_confluence)
                currentPos[stock_id] = position_size
                entryInfo[stock_id] = {'Position': 'Long', 'Strategies': long_strat, 'Entry Price': price}
            elif short_counter >= long_counter and short_counter > 0:
                position_size = -posLimit * (short_counter/max_confluence)
                currentPos[stock_id] = position_size
                entryInfo[stock_id] = {'Position': 'Short', 'Strategies': short_strat, 'Entry Price': price}
        else: # if in a position, check if we need to close it
            entry = entryInfo[stock_id]
            position = entry['Position']
            strategies = entry['Strategies']
            entry_price = entry['Entry Price']
            close_position = False

            if position == 'Long':
                # Breakeven stop-loss: exit if current price falls below entry price
                # if price < entry_price:
                #    close_position = True

                for strat in strategies:
                    if strat == 'BB':
                        if price >= ub:
                            close_position = True
                    elif strat == 'RSI':
                        if rsi >= overbought_rsi:
                            close_position = True
                    elif strat == 'Stoch RSI':
                        if price > overbought_s_rsi:
                            close_position = True
                    elif strat == 'MACD':
                        if macd_diff < 0:
                            close_position = True

            elif position == 'Short':
                # Breakeven stop-loss: exit if current price rises above entry price
                # if price > entry_price:
                #    close_position = True

                for strat in strategies:
                    if strat == 'BB':
                        if price <= lb:
                            close_position = True
                    elif strat == 'RSI':
                        if rsi <= oversold_rsi:
                            close_position = True
                    elif strat == 'Stoch RSI':
                        if price <= oversold_s_rsi:
                            close_position = True
                    elif strat == 'MACD':
                        if macd_diff > 0:
                            close_position = True
            
            if close_position:
                # close the position
                currentPos[stock_id] = 0
                entryInfo[stock_id] = None

                # look for new position
                longOrShort_dict = {}
                if price <= lb:
                    longOrShort_dict['BB'] = 'Long'
                elif price >= ub:
                    longOrShort_dict['BB'] = 'Short'

                if rsi <= oversold_rsi:
                    longOrShort_dict['RSI'] = 'Long'
                elif rsi >= overbought_rsi:
                    longOrShort_dict['RSI'] = 'Short'

                if s_rsi <= oversold_s_rsi:
                    longOrShort_dict['Stoch RSI'] = 'Long'
                elif s_rsi >= overbought_s_rsi:
                    longOrShort_dict['Stoch RSI'] = 'Short'

                if macd_diff >= 0:
                    longOrShort_dict['MACD'] = 'Long'
                else:
                    longOrShort_dict['MACD'] = 'Short'
                
                long_strat = []
                short_strat = []
                for strat, condition in longOrShort_dict.items():
                    if condition == 'Long':
                        long_strat.append(strat)
                    elif condition == 'Short':
                        short_strat.append(strat)
                
                long_counter = len(long_strat)
                short_counter = len(short_strat)

                if long_counter >= short_counter and long_counter > 0:
                    position_size = posLimit * (long_counter/max_confluence)
                    currentPos[stock_id] = position_size
                    entryInfo[stock_id] = {'Position': 'Long', 'Strategies': long_strat, 'Entry Price': price}
                elif short_counter >= long_counter and short_counter > 0:
                    position_size = -posLimit * (short_counter/max_confluence)
                    currentPos[stock_id] = position_size
                    entryInfo[stock_id] = {'Position': 'Short', 'Strategies': short_strat, 'Entry Price': price}

    return currentPos

def newStrat(prcSoFar):
    global currentPos, entryInfo

    # bollinger bands variables
    ma_period = 21
    # rsi variables
    rsi_window = 14
    atr_window = 14
    overbought_rsi = 70
    oversold_rsi = 30
    overbought_s_rsi = 80
    oversold_s_rsi = 20
    # macd variables
    slow_ema = 26
    fast_ema = 12
    macd_signal_length = 9

    df = Stocks(prcSoFar)
    df.bbCalc(ma_period)
    df.rsiCalc(rsi_window)
    df.stochRSICalc(rsi_window)
    df.macdCalc(slow_ema, fast_ema, macd_signal_length)
    df.maCalc(200)
    df.maCalc(50)
    df.atrCalc()
    df.dailyReturnyCalc()

    #volatility = df.data.groupby('Stock')[f'ATR {atr_window}'].mean()
    #volatility = df.data.groupby('Stock')['Daily Return'].std()

    last_week_of_data = df.data.groupby('Stock').tail(7).reset_index(drop=True)
    today_data = df.data.groupby('Stock').tail(1).reset_index(drop=True)
    print(today_data)
    today_price = today_data['Price'] # will return as dataframe rather than series

    for stock_id, stock_data in today_data.groupby('Stock'):
        price = float(stock_data['Price'].iloc[0])
        stock_historical_volatility = volatility[stock_id]
        #curr_vol = stock_data[f'ATR {atr_window}'].iloc[0]
        curr_vol = stock_data[f'STD {atr_window}'].iloc[0]
        volatile = curr_vol > stock_historical_volatility
        posLimit = int(10000 / price)
        currPos = currentPos[stock_id]

        # Bollinger Bands
        ub = float(stock_data['Upper Band'].iloc[0])
        lb = float(stock_data['Lower Band'].iloc[0])
        # Relative Strength Index
        rsi = float(stock_data[f'RSI {rsi_window}'].iloc[0])
        # Stochastic RSI
        s_rsi = float(stock_data[f'StochRSI {rsi_window}'].iloc[0])
        # MACD
        macd = float(stock_data['MACD'].iloc[0])
        macd_signal = float(stock_data['MACD Signal'].iloc[0])
        macd_diff = macd - macd_signal

        print(stock_id)
        print(stock_data)
        print(f"historical volatility: {stock_historical_volatility}")
        print(f"today's volatility: {curr_vol}")
        print("\n")

def _getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

if __name__ == '__main__':
    nInst = 0
    nt = 0
    commRate = 0.0010
    dlrPosLimit = 10000

    def loadPrices(fn):
        global nt, nInst
        df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
        (nt, nInst) = df.shape
        return (df.values).T


    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)

    (_, nt) = prcAll.shape
    prcHistSoFar = prcAll[:, :250]

    newStrat(prcHistSoFar)
