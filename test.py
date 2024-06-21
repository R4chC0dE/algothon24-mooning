import numpy as np
from graphing import Stocks

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

for i in range(len(currentPos)):
    currPos = currentPos[i]
    if currPos == 0:
        currentPos[i] = 10000

print(currentPos)