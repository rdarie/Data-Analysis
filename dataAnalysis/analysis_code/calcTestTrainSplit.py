"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                            which experimental day to analyze
    --maskOutlierBlocks                                  delete outlier trials? [default: False]
    --blockIdx=blockIdx                                  which trial to analyze [default: 1]
    --processAll                                         process entire experimental day? [default: False]
    --analysisName=analysisName                          append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName                    append a name to the resulting blocks? [default: motion]
    --window=window                                      process with short window? [default: long]
    --winStart=winStart                                  start of absolute window (when loading)
    --winStop=winStop                                    end of absolute window (when loading)
    --ROIWinStart=ROIWinStart                            start of relative window (masked per trial) [default: 200]
    --ROIWinStop=ROIWinStop                              end of relative window (masked per trial) [default: 400]
    --lazy                                               load from raw, or regular? [default: False]
    --verbose                                            print diagnostics? [default: False]
    --unitQuery=unitQuery                                how to restrict channels? [default: fr_sqrt]
    --inputBlockSuffix=inputBlockSuffix                  which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix                  which trig_ block to pull [default: Block]
    --alignQuery=alignQuery                              what will the plot be aligned to? [default: midPeak]
    --iteratorSuffix=iteratorSuffix                      filename for resulting iterator
    --selector=selector                                  filename if using a unit selector
    --calcTimeROI                                        if trial length isn't constant, use this to remove extraneous data [default: False]
    --timeROIAlignQuery=timeROIAlignQuery                what will the plot be aligned to? [default: stimOff]
    --eventBlockPrefix=eventBlockPrefix                  name of event block
    --eventBlockSuffix=eventBlockSuffix                  name of events object to align to [default: analyze]
    --eventName=eventName                                name of events object to align to [default: motionStimAlignTimes]
    --eventSubfolder=eventSubfolder                      name of folder where the event block is [default: None]
    --loadFromFrames                                     load data from pre-saved dataframes?
    --saveDataFrame                                      save corresponding dataframe?
    --selectionName=selectionName                        name in h5 for the saved data
"""

import os, sys
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
import numpy as np
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
# from sklearn.model_selection import cross_val_score, GridSearchCV
# import joblib as jb
import dill as pickle
# import gc
from docopt import docopt
import pandas as pd

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
calcSubFolder = os.path.join(
    alignSubFolder, 'testTrainSplits')
if not(os.path.exists(calcSubFolder)):
    os.makedirs(calcSubFolder)
dataFramesFolder = os.path.join(
    alignSubFolder, 'dataframes'
    )
if not(os.path.exists(dataFramesFolder)):
    os.makedirs(dataFramesFolder)

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchFolder, blockBaseName, **arguments)

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, metaDataToCategories=False,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=essentialMetadataFields, decimate=1))
alignedAsigsKWargs['verbose'] = arguments['verbose']

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))

if 'windowSize' not in alignedAsigsKWargs:
    alignedAsigsKWargs['windowSize'] = [ws for ws in rasterOpts['windowSizes'][arguments['window']]]
if 'winStart' in arguments:
    if arguments['winStart'] is not None:
        alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
if 'winStop' in arguments:
    if arguments['winStop'] is not None:
        alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)

if arguments['calcTimeROI']:
    if arguments['eventBlockSuffix'] is not None:
        eventBlockSuffix = '_{}'.format(arguments['eventBlockSuffix'])
    else:
        eventBlockSuffix = ''
    if arguments['eventBlockPrefix'] is not None:
        eventPrefix = '{}{:0>3}'.format(arguments['eventBlockPrefix'], blockIdx)
    else:
        eventPrefix = ns5FileName
    if arguments['eventSubfolder'] != 'None':
        eventPath = os.path.join(
            scratchFolder, arguments['eventSubfolder'],
            eventPrefix + '{}.nix'.format(eventBlockSuffix))
    else:
        eventPath = os.path.join(
            scratchFolder,
            eventPrefix + '{}.nix'.format(eventBlockSuffix))
    if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC'):
        # has stim but no motion
        if arguments['eventName'] == 'motion':
            print('Block does not have motion!')
            sys.exit()
        if arguments['eventName'] == 'stim':
            eventName = 'stimAlignTimes'
    elif blockExperimentType == 'proprio-motionOnly':
        # has motion but no stim
        if arguments['eventName'] == 'motion':
            eventName = 'motionAlignTimes'
        if arguments['eventName'] == 'stim':
            print('Block does not have stim!')
            sys.exit()
    elif blockExperimentType == 'proprio':
        if arguments['eventName'] == 'stim':
            eventName = 'stimPerimotionAlignTimes'
        elif arguments['eventName'] == 'motion':
            eventName = 'motionStimAlignTimes'
    elif blockExperimentType == 'isi':
        if arguments['eventName'] == 'stim':
            eventName = 'stimAlignTimes'
    eventReader, eventBlock = ns5.blockFromPath(
        eventPath, lazy=arguments['lazy'],
        loadList={'events': ['seg0_{}'.format(eventName)]},
        purgeNixNames=True)
    if 'ROIWinStart' in arguments:
        ROIWinStart = float(arguments['ROIWinStart']) * (1e-3)
    else:
        ROIWinStart = 0
    if 'ROIWinStop' in arguments:
        ROIWinStop = float(arguments['ROIWinStop']) * (1e-3)
    else:
        ROIWinStop = 0
    listOfROIMasks = []
    listOfExampleIndexes = []
#
if arguments['iteratorSuffix'] is not None:
    iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
else:
    iteratorSuffix = ''
iteratorPath = os.path.join(
    calcSubFolder,
    '{}_{}_{}{}_cvIterators.pickle'.format(
        blockBaseName,
        arguments['window'],
        arguments['alignQuery'],
        iteratorSuffix))

if arguments['verbose']:
    prf.print_memory_usage('before load data')

binOpts = rasterOpts['binOpts'][arguments['analysisName']]
if glmOptsLookup['nCovariateBasisTerms'] > 1:
    lags = np.linspace(
        -1 * glmOptsLookup['covariateHistoryLen'],
        glmOptsLookup['covariateHistoryLen'],
        glmOptsLookup['nCovariateBasisTerms']) / binOpts['binInterval']
    alignedAsigsKWargs['addLags'] = {'all': lags.astype(int).tolist()}
alignedAsigsKWargs['decimate'] = int(glmOptsLookup['regressionBinInterval'] / binOpts['binInterval'])
nSplits = 7
cv_kwargs = dict(
    shuffle=True,
    stratifyFactors=stimulusConditionNames,
    continuousFactors=['segment', 'originalIndex', 't'])
listOfIterators = []
listOfDataFrames = []
if not arguments['loadFromFrames']:
    print('loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    nSeg = len(dataBlock.segments)
    for segIdx in range(nSeg):
        if arguments['verbose']:
            prf.print_memory_usage(
                'fitting on segment {}'.format(segIdx))
        aakwa = alignedAsigsKWargs.copy()
        # pdb.set_trace()
        dataDF = ns5.alignedAsigsToDF(
            dataBlock,
            whichSegments=[segIdx],
            **aakwa)
        # trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        if arguments['calcTimeROI']:
            endMaskQuery = ash.processAlignQueryArgs(
                namedQueries, alignQuery=arguments['timeROIAlignQuery'])
            evList = eventBlock.filter(
                objects=ns5.Event,
                name='seg{}_{}'.format(segIdx, eventName))
            assert len(evList) == 1
            targetTrialAnnDF = ns5.unitSpikeTrainArrayAnnToDF(evList).query(endMaskQuery)
            targetMask = pd.Series(True, index=dataDF.index)
            for (_, _, t), group in dataDF.groupby(['segment', 'originalIndex', 't']):
                timeDifference = (targetTrialAnnDF['t'] - t)
                deltaT = timeDifference[timeDifference >= 0].min()
                groupBins = group.index.get_level_values('bin')
                print('Looking for bins >= {:.3f} and < {:.3f}'.format(ROIWinStart, deltaT + ROIWinStop))
                targetMask.loc[group.index] = (groupBins >= ROIWinStart) & (groupBins < deltaT + ROIWinStop)
            listOfROIMasks.append(targetMask)
            print('targetMask has dimension {}'.format(targetMask.shape))
            dataDF = dataDF.loc[targetMask, :]
            exampleIndex = dataDF.index
            listOfExampleIndexes.append(exampleIndex)
        listOfDataFrames.append(dataDF)
    if arguments['lazy']:
        dataReader.file.close()
else:    # loading frames
    if not arguments['processAll']:
        experimentsToAssemble = {
            experimentName: [blockIdx]}
    currBlockNum = 0
    for expName, lOfBlocks in experimentsToAssemble.items():
        thisScratchFolder = os.path.join(scratchPath, expName)
        analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
            arguments, thisScratchFolder)
        thisDFFolder = os.path.join(alignSubFolder, 'dataframes')
        for bIdx in lOfBlocks:
            theseArgs = arguments.copy()
            theseArgs['blockIdx'] = '{}'.format(bIdx)
            theseArgs['processAll'] = False
            thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
            dFPath = os.path.join(
                thisDFFolder,
                '{}_{}_{}_df{}.h5'.format(
                    thisBlockBaseName,
                    arguments['window'],
                    arguments['alignQuery'],
                    iteratorSuffix))
            thisDF = pd.read_hdf(dFPath, arguments['selectionName'])
            # newSegLevel = [currBlockNum for i in range(thisDF.shape[0])]
            thisDF.index = thisDF.index.set_levels([currBlockNum], level='segment')
            listOfDataFrames.append(thisDF)
            currBlockNum += 1

'''
trialInfo = dataDF.index.to_frame().reset_index(drop=True)
for cN in trialInfo.columns:
    print('{}'.format(cN))
    print(trialInfo[cN].unique())
    print('   ')
    '''
if not arguments['processAll']:
    for dataDF in listOfDataFrames:
        cvIterator = tdr.trainTestValidationSplitter(
            dataDF, tdr.trialAwareStratifiedKFold,
            n_splits=nSplits, splitterKWArgs=cv_kwargs
            )
        listOfIterators.append(cvIterator)
else:
    exportDF = pd.concat(listOfDataFrames)
    cvIterator = tdr.trainTestValidationSplitter(
        exportDF, tdr.trialAwareStratifiedKFold,
        n_splits=nSplits, splitterKWArgs=cv_kwargs
        )
    listOfIterators.append(cvIterator)
###
# cvIterator.plot_schema()
exportAAKWA = alignedAsigsKWargs.copy()
exportAAKWA.pop('unitNames', None)
exportAAKWA.pop('unitQuery', None)
iteratorMetadata = {
    'alignedAsigsKWargs': exportAAKWA,
    'iteratorsBySegment': listOfIterators,
    'cv_kwargs': cv_kwargs,
    'experimentsToAssemble': experimentsToAssemble
}
if arguments['calcTimeROI']:
    iteratorMetadata.update({
        'listOfROIMasks': listOfROIMasks,
        'listOfExampleIndexes': listOfExampleIndexes
    })
print('saving\n{}\n'.format(iteratorPath))
if os.path.exists(iteratorPath):
    os.remove(iteratorPath)
with open(iteratorPath, 'wb') as f:
    pickle.dump(
        iteratorMetadata, f)
if arguments['saveDataFrame']:
    outputDFPath = os.path.join(
        dataFramesFolder,
        '{}_{}_{}_df{}.h5'.format(
            blockBaseName,
            arguments['window'],
            arguments['alignQuery'],
            iteratorSuffix))
    exportDF.to_hdf(outputDFPath, arguments['selectionName'], mode='w')
