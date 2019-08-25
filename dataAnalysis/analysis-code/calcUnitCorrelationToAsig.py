"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --trialIdx=trialIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --plotting                                plot out the correlation matrix? [default: True]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --inputBlockName=inputBlockName           filename for inputs [default: fr]
    --secondaryBlockName=secondaryBlockName   filename for secondary inputs [default: RC]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                   query what the units will be aligned to? [default: midPeak]
    --selector=selector                       filename if using a unit selector
"""

import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import numpy as np
#  import pandas as pd
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    analysisSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
secondaryPath = os.path.join(
    analysisSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['secondaryBlockName'], arguments['window']))
resultPath = os.path.join(
    analysisSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1,
    transposeToColumns='bin', concatOn='index',
    getMetaData=False,
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)

from sklearn.preprocessing import scale, robust_scale

def corFun(
        x, y,
        xBounds=None, yBounds=None,
        plotting=False):
    maskX = (x.columns > xBounds[0]) & (x.columns < xBounds[1])
    maskY = (y.columns > yBounds[0]) & (y.columns < yBounds[1])
    scaledX = x.stack()
    scaledX.iloc[:] = robust_scale(scaledX.to_numpy())
    scaledX = scaledX.unstack(level='bin')
    scaledY = y.stack()
    scaledY.iloc[:] = robust_scale(scaledY.to_numpy())
    scaledY = scaledY.unstack(level='bin')
    dt = x.columns[1] - x.columns[0]
    allMaxCorr = np.zeros(x.index.shape[0], dtype=np.float)
    allMaxLag = np.zeros(x.index.shape[0], dtype=np.int)
    for idx in range(scaledX.index.shape[0]):
        cor = np.correlate(
            (scaledX.iloc[idx, :].loc[maskX]),
            (scaledY.iloc[idx, :].loc[maskY]))
        lag = np.argmax(cor)
        allMaxLag[idx] = np.atleast_1d(lag)[0]
        allMaxCorr[idx] = cor[allMaxLag[idx]]
    if plotting:
        import matplotlib.pyplot as plt
        import seaborn as sns
        #
        fig, ax = plt.subplots()
        ax.plot(x.columns, scaledX.mean(axis=0), label='FR')
        ax.plot(y.columns, scaledY.mean(axis=0), label='EMG')
        ax.legend()
        plt.show()
        ax = sns.distplot(allMaxLag, bins=100)
        ax.set_xlabel('maximum lag')
        plt.show()
        ax = sns.distplot(allMaxCorr, bins=100)
        ax.set_xlabel('maximum correlation')
        plt.show()
        #
    return np.mean(allMaxCorr), np.mean(allMaxLag) * dt


resultNames = ['emgMaxCrossCorr', 'emgMaxCrossCorrLag']

resDFList = ash.applyFun(
    triggeredPath=triggeredPath, resultPath=resultPath,
    resultName=resultNames,
    fun=corFun, applyType='func', loadType='pairwise',
    funKWargs={
        'xBounds': [0, 0.3],
        'yBounds': [0, 0.15],
        'plotting': arguments['plotting']},
    lazy=arguments['lazy'],
    verbose=arguments['verbose'],
    secondaryPath=secondaryPath,
    secondaryUnitQuery=namedQueries['unit']['oechorins'],
    loadArgs=alignedAsigsKWargs)
