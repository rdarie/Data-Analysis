"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --window=window                        process with short window? [default: long]
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --selectorName=selectorName            filename for resulting selector [default: minfrmaxcorr]
"""
import os
#  import numpy as np
#  import pandas as pd
import pdb
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  import neo
import dill as pickle
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import pandas as pd
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
resultPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
selectorPath = os.path.join(
    scratchFolder,
    prefix + '_{}.pickle'.format(
        arguments['selectorName']))

meanFRDF = pd.read_hdf(resultPath, 'meanFR')
corrDF = pd.read_hdf(resultPath, 'corr')
for n in corrDF.index:
    corrDF.loc[n, n] = 0

def selFun(
        meanDF, corrDF, meanThresh=5,
        corrThresh=0.85):
    unitMask = ((meanDF > meanThresh) & (corrDF.max() < corrThresh))
    return unitMask[unitMask].index.to_list()

thisCorrThresh = 0.85
outputFeatures = selFun(meanFRDF, corrDF, corrThresh=thisCorrThresh)
selectorMetadata = {
    'trainingDataPath': os.path.basename(resultPath),
    'path': os.path.basename(selectorPath),
    'name': arguments['selectorName'],
    'inputBlockName': arguments['inputBlockName'],
    'inputFeatures': corrDF.columns.to_list(),
    'outputFeatures': [
        i.split('_{}'.format(arguments['inputBlockName']))[0]
        for i in outputFeatures],
    'selFun': selFun,
    'selFunInputs': {'meanThresh': 5, 'corrThresh': thisCorrThresh}
    }

with open(selectorPath, 'wb') as f:
    pickle.dump(
        selectorMetadata, f)
