"""
Creates a list of units to include in future analyses

Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --window=window                        process with short window? [default: long]
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --selectorName=selectorName            filename for resulting selector [default: minfrmaxcorr]
"""
import os
#  import numpy as np
#  import pandas as pd
import pdb
import re
from datetime import datetime as dt
import numpy as np
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  import neo
import dill as pickle
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
selectorPath = os.path.join(
    alignSubFolder,
    'unitSelector_{}.pickle'.format(
        arguments['selectorName']))
#
meanFRDF = pd.read_hdf(resultPath, 'meanFR')
corrDF = pd.read_hdf(resultPath, 'corr')
for n in corrDF.index:
    corrDF.loc[n, n] = 0
impedanceDF = pd.Series(np.nan, index=meanFRDF.index)
for (feature, lag) in impedanceDF.index:
    chName = feature.split('#')[0]
    impedanceDF[(feature, lag)] = impedances.loc[chName, 'impedance']


def selFun(
        meanDF, corrDF, impedanceDF, 
        meanThresh=5, corrThresh=0.95, impedanceBounds=(1, 500)):
    # meanDF=meanFRDF; meanThresh=5; corrThresh=0.95; impedanceBounds=(1, 500)
    meanMask = (meanDF > meanThresh)
    impedanceMask = (
        (impedanceDF > impedanceBounds[0]) &
        (impedanceDF < impedanceBounds[1])
        )
    unitMask = meanMask & impedanceMask
    corrMask = (corrDF.loc[unitMask, unitMask].abs().max() < corrThresh)
    for unitIdx in unitMask.index:
        if unitIdx in corrMask.index:
            unitMask.loc[unitIdx] = unitMask.loc[unitIdx] & corrMask.loc[unitIdx]
    return unitMask[unitMask].index.to_list()


# meanDF=meanFRDF; meanThresh=5; corrThresh=0.95
# import matplotlib.pyplot as plt
# plt.hist(impedanceDF, bins='sqrt'); plt.show()
thisCorrThresh = .85
thisMeanThresh = 5
# import pdb; pdb.set_trace()
outputFeatures = selFun(
    meanFRDF, corrDF, impedanceDF, meanThresh=thisMeanThresh,
    corrThresh=thisCorrThresh)
#
def trimSuffix(featureName, suffix):
    return featureName.replace('_{}#0'.format(suffix), '')
#
selectorMetadata = {
    'trainingDataPath': os.path.basename(resultPath),
    'path': os.path.basename(selectorPath),
    'name': arguments['selectorName'],
    'inputBlockName': arguments['inputBlockName'],
    'inputFeatures': [
        trimSuffix(i[0], arguments['inputBlockName'])
        for i in corrDF.columns.to_list()],
    'outputFeatures': [
        trimSuffix(i[0], arguments['inputBlockName'])
        for i in outputFeatures],
    'selFun': selFun,
    'selFunInputs': {'meanThresh': 5, 'corrThresh': thisCorrThresh}
    }
#

with open(selectorPath, 'wb') as f:
    pickle.dump(
        selectorMetadata, f)
print('Selected {} units'.format(len(outputFeatures)))