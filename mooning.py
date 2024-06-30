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

def bbStrat_getMyPosition(prcSoFar):
    global currentPos, entryPos
    df = Stocks(prcSoFar)
    bb_res = df.bbCalc()
    
    for i in range(len(currentPos)):
        pos_pnl = 0
        currPos = currentPos[i]
        entry = entryPrice[i] 
        currPrice = bb_res[i]['Price'].iloc[-1]
        lastPrice = bb_res[i]['Price'].iloc[-2]
        currMA = bb_res[i]['Moving Average'].iloc[-1]
        currLB = bb_res[i]['Lower Band'].iloc[-1]
        currUB = bb_res[i]['Upper Band'].iloc[-1]
        if entry != 0:
            pos_pnl = (currPrice-entry)/entry

        # if we don't have a position then open a position if at lower bound
        if currPos == 0:
            if currPrice <= currLB:
                numOfStock = int(10000/currPrice)
                currentPos[i] = numOfStock
                entryPrice[i] = currPrice
                #print(f"Bought {numOfStock} of stock {i} @ ${currPrice}")
                continue

        else:
            # if position is at -1%, close position
            if pos_pnl <= -0.01:
                currentPos[i] = 0.0
            # if price falls below ma, close 50% of position
            elif lastPrice >= currMA and currPrice <= currMA:
                rPos = -0.5*currPos
                currentPos[i] += rPos
            elif currPrice >= currUB:
                rPos = -0.5*currPos
                currentPos[i] += rPos

    return currentPos

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

    getMyPosition(prcHistSoFar)