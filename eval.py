#!/usr/bin/env python

import numpy as np
import pandas as pd
from mooning import getMyPosition as getPosition
from pathlib import Path
import os

nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

def get_unique_filename(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(unique_filename):
        unique_filename = f"{base} ({counter}){ext}"
        counter += 1

    return unique_filename

def get_unique_directory(dirname):
    base = dirname
    counter = 1
    unique_dirname = Path(dirname)

    while unique_dirname.exists():
        unique_dirname = Path(f"{base} ({counter})")
        counter += 1

    return unique_dirname

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(250, 501):
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        print("Day %d value: %.2lf today PL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
        
        # Open the file in write mode
        with open(unique_output_file, "a") as file:
            file.write("Day %d value: %.2lf today PL: $%.2lf $-traded: %.0lf return: %.5lf \n" %
                        (t, value, todayPL, totDVolume, ret))

    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


# own output code
output_dir = Path("Evaluation Output")
output_dir.mkdir(parents=True, exist_ok=True)
# Create directory if it doesn't exist
output_file = output_dir / "results.txt"
unique_output_file = get_unique_filename(output_file)

(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)

with open(unique_output_file, "a") as file:
    file.write("=====\n")
    file.write("mean(PL): %.1lf \n" % meanpl)
    file.write("return: %.5lf \n" % ret)
    file.write("StdDev(PL): %.2lf \n" % plstd)
    file.write("annSharpe(PL): %.2lf \n" % sharpe)
    file.write("totDvolume: %.0lf \n" % dvol)
    file.write("Score: %.2lf \n" % score)

