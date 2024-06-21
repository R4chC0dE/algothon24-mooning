
import numpy as np
from graphing import Stocks

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
entryPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos, entryPos
    df = Stocks(prcSoFar)
    bb_res = df.bbCalc()
    price = bb_res['Price'][:,-1]
    prev_price = bb_res['Price'][:,-2]
    ma = bb_res['Moving Average'][:,-1]
    ub = bb_res['Upper Band'][:,-1]
    lb = bb_res['Lower Band'][:,-1]
    
    for i in range(len(currentPos)):
        currPos = currentPos[i]
        entry = entryPos[i] 
        currPrice = price[i]
        lastPrice = prev_price[i]
        currMA = ma[i]
        currLB = lb[i]
        currUB = ub[i]
        pos_pnl = (entry-currPrice)/entry

        # if we don't have a position then open a position if at lower bound
        if currPos == 0:
            if currPrice <= currLB:
                currentPos[i] = 10000
                continue

        else:
            # if position is at -1%, close position
            if pos_pnl <= -0.01:
                currentPos[i] = 0.0
            # if price falls below ma, close 50% of position
            elif lastPrice >= currMA and currPrice <= currMA:
                newPos = 0.5*currPos
                currentPos[i] = newPos
            elif currPrice >= currUB:
                newPos = 0.5*currPos
                currentPos[i] = newPos

    return currentPos

if __name__ == '__main__':
    print("hello")