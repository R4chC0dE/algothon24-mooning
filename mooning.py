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

        if rsi <= oversold_rsi:
            longOrShort_dict['RSI'] = 'Long'
        elif rsi >= overbought_rsi:
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

    volatile_atr_range = 4
    less_volatile_atr_range = 7
    stop_loss = 0.05

    df = Stocks(prcSoFar)
    df.bbCalc(ma_period)
    df.rsiCalc(rsi_window)
    df.stochRSICalc(rsi_window)
    df.macdCalc(slow_ema, fast_ema, macd_signal_length)
    df.maCalc(200)
    df.maCalc(50)
    df.atrCalc()
    df.sdCalc(365)
    df.atrCalc(volatile_atr_range)
    df.atrCalc(less_volatile_atr_range)


    last_week_of_data = df.data.groupby('Stock').tail(7)
    #today_price = today_data['Price'] # will return as dataframe rather than series
    average_volatility = df.data.groupby('Stock')['SD 365'].tail(1).mean()
    #volatility = df.data.groupby('Stock')[f'ATR {atr_window}']
    #volatility = df.data.groupby('Stock')['Daily Return'].std()
    #print(volatility)
    print(df.data.groupby('Stock').tail(1))
    print(f"average: {average_volatility}")

    for stock_id, last_week_stock_data in last_week_of_data.groupby('Stock'):
        longOrShort_dict = {}
        today_data = last_week_stock_data.tail(1)
        price = float(today_data['Price'].iloc[0])
        stock_historical_volatility = float(today_data['SD 365'].iloc[0])
        volatile = stock_historical_volatility > average_volatility
        posLimit = int(10000 / price)
        currPos = currentPos[stock_id]

        # Bollinger Bands
        ub = float(today_data['Upper Band'].iloc[0])
        mub = float(today_data['Upper Mid Band'].iloc[0])
        lb = float(today_data['Lower Band'].iloc[0])
        mlb = float(today_data['Lower Mid Band'].iloc[0])
        # Relative Strength Index
        rsi = float(today_data[f'RSI {rsi_window}'].iloc[0])
        # Stochastic RSI
        s_rsi = float(today_data[f'StochRSI {rsi_window}'].iloc[0])
        # MACD
        macd = float(today_data['MACD'].iloc[0])
        macd_signal = float(today_data['MACD Signal'].iloc[0])
        macd_diff = macd - macd_signal
        # moving averages
        ma50 = float(today_data['50MA'].iloc[0])
        ma200 = float(today_data['200MA'].iloc[0])
        # ATRs
        volatile_atr = float(today_data[f'ATR {volatile_atr_range}'])
        less_volatile_atr = float(today_data[f'ATR {less_volatile_atr_range}'])

        if price <= lb:
            longOrShort_dict['BB'] = 'Long'
        elif price >= ub:
            longOrShort_dict['BB'] = 'Short'

        if rsi <= oversold_rsi:
            longOrShort_dict['RSI'] = 'Long'
        elif rsi >= overbought_rsi:
            longOrShort_dict['RSI'] = 'Short'
        elif rsi <= overbought_rsi and rsi >= oversold_rsi:
            longOrShort_dict['RSI'] = 'Any'

        if s_rsi <= oversold_s_rsi: 
            longOrShort_dict['Stoch RSI'] = 'Long'
        elif s_rsi >= overbought_s_rsi:
            longOrShort_dict['Stoch RSI'] = 'Short'

        if macd_diff >= 0:
            longOrShort_dict['MACD'] = 'Long'
        else:
            longOrShort_dict['MACD'] = 'Short'

        if ma50 >= ma200:
            longOrShort_dict['GC'] = 'Long'
        else:
            longOrShort_dict['GC'] = 'Short'

        # calculate position size
        max_confluence = 5
        long_strat = []
        short_strat = []
        for strat, condition in longOrShort_dict.items():
            if condition == 'Long':
                long_strat.append(strat)
            elif condition == 'Short':
                short_strat.append(strat)
            if strat == 'RSI':
                if condition == 'Any':
                    long_strat.append(strat)
                    short_strat.append(strat)
        
        long_counter = len(long_strat)
        short_counter = len(short_strat)

        if long_counter >= short_counter and long_counter > 0:
            position_size = posLimit * (long_counter/max_confluence)
        elif short_counter >= long_counter and short_counter > 0:
            position_size = -posLimit * (short_counter/max_confluence)

        if volatile:
            if currPos == 0:
                if longOrShort_dict['BB'] == 'Long':
                    if longOrShort_dict['RSI'] == 'Any' or longOrShort_dict['RSI'] == 'Long':
                        currentPos[stock_id] = position_size
                        entryInfo[stock_id] = {'Position': 'Long', 'Entry Price': price, 'Stop Loss': price}
                        # stop loss at entry for now
                elif longOrShort_dict['BB'] == 'Short':
                    if longOrShort_dict['RSI'] == 'Any' or longOrShort_dict['RSI'] == 'Short':
                        currentPos[stock_id] = position_size
                        entryInfo[stock_id] = {'Position': 'Short', 'Entry Price': price, 'Stop Loss': price}
            
            else:
                position = entryInfo[stock_id]['Position'] 
                entry_price = entryInfo[stock_id]['Entry Price']
                sl_price = entryInfo[stock_id]['Stop Loss']
                if position == 'Long':
                    if (price >= ub and volatile_atr < 0) or rsi >= overbought_rsi:
                        currentPos[stock_id] = 0
                    elif price > mub and volatile_atr < 0:
                        entryInfo[stock_id]['Stop Loss'] = mub
                    elif price < sl_price:
                        currentPos[stock_id] = 0
                elif position == 'Short':
                    if (price <= lb and volatile_atr > 0):
                        return # TODO short exit conditions


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


    pricesFile = "./prices 750 days.txt"
    prcAll = loadPrices(pricesFile)

    (_, nt) = prcAll.shape
    prcHistSoFar = prcAll[:, :750]

    newStrat(prcHistSoFar)
