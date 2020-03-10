"""
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
    --selector=selector                    filename for resulting selector [default: minfrmaxcorr]
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
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
selectorPath = os.path.join(
    alignSubFolder,
    prefix + '_{}.pickle'.format(
        arguments['selector']))
#
if arguments['verbose']:
    print('Saving selector {}'.format(selectorPath))
dataPath = os.path.join(
    analysisSubFolder,
    prefix + '_analyze.nix')
dataReader, dataBlock = ns5.blockFromPath(
    dataPath, lazy=arguments['lazy'])
if arguments['verbose']:
    print('Loading {}'.format(dataPath))

allSpikeTrains = dataBlock.filter(objects=ns5.SpikeTrain)
# pdb.set_trace()
allSpikeWaveforms = {}
for spt in allSpikeTrains:
    if 'elec' in spt.name:
        baseName = ns5.childBaseName(spt.name, 'seg')
        if baseName in allSpikeWaveforms:
            allSpikeWaveforms[baseName] = np.concatenate(
                [
                    allSpikeWaveforms[baseName],
                    spt.waveforms.magnitude])
        else:
            allSpikeWaveforms[baseName] = spt.waveforms.magnitude
allSpikeMaxAmp = pd.Series(
    {
        (k + '_fr#0', 0): np.max(np.mean(np.abs(v), axis=2))
        for k, v in allSpikeWaveforms.items()}
    )
# allSpikeMaxAmp = pd.Series(
#     {
#         (i.name + '_fr#0', 0): np.abs(i.annotations['max_peak_amplitude'])
#         for i in allSpikeTrains
#         if 'elec' in i.name}
#     )
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
        meanDF, corrDF, impedanceDF, maxAmpDF,
        meanThresh=5, corrThresh=0.95, impedanceBounds=(1, 500), ampThresh=5):
    # meanDF=meanFRDF; meanThresh=5; corrThresh=0.95; impedanceBounds=(1, 500)
    meanMask = (meanDF > meanThresh)
    ampMask = (maxAmpDF > ampThresh)
    impedanceMask = (
        (impedanceDF > impedanceBounds[0]) &
        (impedanceDF < impedanceBounds[1])
        )
    unitMask = meanMask & impedanceMask & ampMask
    corrMask = (corrDF.loc[unitMask, unitMask].abs().max() < corrThresh)
    for unitIdx in unitMask.index:
        if unitIdx in corrMask.index:
            unitMask.loc[unitIdx] = unitMask.loc[unitIdx] & corrMask.loc[unitIdx]
    return unitMask[unitMask].index.to_list()


# meanDF=meanFRDF; meanThresh=5; corrThresh=0.95
# import matplotlib.pyplot as plt
# plt.hist(impedanceDF, bins='sqrt'); plt.show()
thisCorrThresh = .85
thisMeanThresh = 3
thisAmpThresh = 5
# import pdb; pdb.set_trace()
outputFeatures = selFun(
    meanFRDF, corrDF, impedanceDF, allSpikeMaxAmp,
    meanThresh=thisMeanThresh, corrThresh=thisCorrThresh,
    ampThresh=thisAmpThresh)
# outputFeatures = [i for i in outputFeatures if i[0].replace('_fr#0', '') in ampUnits]
print('Selecting features: \n {}'.format(outputFeatures))
def trimSuffix(featureName, suffix):
    return featureName.replace('_{}#0'.format(suffix), '')

selectorMetadata = {
    'trainingDataPath': os.path.basename(resultPath),
    'path': os.path.basename(selectorPath),
    'name': arguments['selector'],
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
