
import numpy as np
from graphing import Stocks

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
entryPrice = np.zeros(nInst)


def getMyPosition(prcSoFar):
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
    print("hello")