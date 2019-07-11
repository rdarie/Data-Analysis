"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --inputBlockName=inputBlockName        filename for inputs [default: RC]
    --window=window                        process with short window? [default: RC]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: oech]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: stimOn]
    --selector=selector                    filename if using a unit selector
    --resultName=resultName                filename for result [default: emg]
"""

import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
import dill as pickle
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultFolder = os.path.join(scratchFolder, 'npy')
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder, exist_ok=True)
resultPath = os.path.join(
    resultFolder, prefix + '_{}_{}.pickle'.format(
        arguments['inputBlockName'], arguments['window']))

#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1,
    transposeToColumns='bin', concatOn='index',
    getMetaData=True,
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
dataReader, dataBlock = ns5.blockFromPath(triggeredPath, lazy=arguments['lazy'])

alignedAsigsKWargs.update({'decimate': 15})
alignedAsigsKWargs.update({'windowSize': (-13e-3, 45e-3)})
masterSpikeMat = ns5.alignedAsigsToDF(
    dataBlock,
    **alignedAsigsKWargs)

allMats = []
allElectrodeNames = (
    ['SpCB{}'.format(i) for i in range(1, 5)] +
    ['SpRB{}'.format(i) for i in range(1, 5)] +
    ['C'])
for idx, (name, group) in enumerate(masterSpikeMat.groupby('feature')):
    if idx == 0:
        metaData = group.index.to_frame().reset_index(drop=True)
        pwMask = (group.columns >= 0) & (group.columns <= 250e-6)
        pwNSamp = pwMask.sum()
        for electrodeName in allElectrodeNames:
            electrodeAmps = np.zeros(group.shape)
            isCathode = metaData['electrode'].str.contains('-' + electrodeName, regex=False)
            fillVal = metaData['amplitude'].to_numpy() * isCathode.to_numpy() * (-1)
            if pwNSamp > 1:
                fillVal = np.hstack((fillVal for i in range(pwNSamp)))
            if fillVal.ndim == 1:
                fillVal = fillVal[:, np.newaxis]
            electrodeAmps[:, pwMask] = fillVal
            isAnode = metaData['electrode'].str.contains('+' + electrodeName, regex=False)
            fillVal = metaData['amplitude'].to_numpy() * isAnode.to_numpy()
            if pwNSamp > 1:
                fillVal = np.hstack((fillVal for i in range(pwNSamp)))
            if fillVal.ndim == 1:
                fillVal = fillVal[:, np.newaxis]
            electrodeAmps[:, pwMask] = fillVal
            allMats.append(electrodeAmps[:, np.newaxis, :])
            allNames = allElectrodeNames.copy()
    allMats.append(group.to_numpy()[:, np.newaxis, :])
    allNames.append(name)
#
data = np.concatenate(allMats, axis=1)
results = {
    'data': data,
    'ax1': allNames,
    'ax2': group.columns.to_numpy()
}
with open(resultPath, 'wb') as f:
    pickle.dump(results, f)
prf.print_memory_usage('just loaded firing rates')
if arguments['lazy']:
    dataReader.file.close()