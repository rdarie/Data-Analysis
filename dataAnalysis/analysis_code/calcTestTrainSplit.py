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
    --lazy                                               load from raw, or regular? [default: False]
    --verbose                                            print diagnostics? [default: False]
    --unitQuery=unitQuery                                how to restrict channels? [default: fr_sqrt]
    --inputBlockSuffix=inputBlockSuffix                  which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix                  which trig_ block to pull [default: Block]
    --alignQuery=alignQuery                              what will the plot be aligned to? [default: midPeak]
    --iteratorSuffix=iteratorSuffix                      filename for resulting iterator
    --selector=selector                                  filename if using a unit selector
    --eventBlockPrefix=eventBlockPrefix                  name of event block
    --eventBlockSuffix=eventBlockSuffix                  name of events object to align to [default: analyze]
    --eventName=eventName                                name of events object to align to [default: motionStimAlignTimes]
    --eventSubfolder=eventSubfolder                      name of folder where the event block is [default: None]
    --loadFromFrames                                     load data from pre-saved dataframes?
    --saveDataFrame                                      save corresponding dataframe?
    --selectionName=selectionName                        name in h5 for the saved data
    --controlSet                                         regular data, or control?
"""

import os, sys
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb, traceback
import numpy as np
from numpy.random import default_rng
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
# from sklearn.model_selection import cross_val_score, GridSearchCVc
# import joblib as jb
import dill as pickle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# import gc
from docopt import docopt
import pandas as pd
from copy import copy, deepcopy
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
cvIteratorSubfolder = os.path.join(
    alignSubFolder, 'testTrainSplits')
if not(os.path.exists(cvIteratorSubfolder)):
    os.makedirs(cvIteratorSubfolder)
dataFramesFolder = os.path.join(
    analysisSubFolder, 'dataframes'
    )
if not(os.path.exists(dataFramesFolder)):
    os.makedirs(dataFramesFolder)

theseIteratorOpts = iteratorOpts[arguments['iteratorSuffix']]

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
alignedAsigsKWargs['getFeatureMetaData'] = ['xCoords', 'yCoords', 'freqBandName', 'parentFeature']
#
triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))
#
if 'windowSize' not in alignedAsigsKWargs:
    alignedAsigsKWargs['windowSize'] = [ws for ws in rasterOpts['windowSizes'][arguments['window']]]
if 'winStart' in arguments:
    if arguments['winStart'] is not None:
        alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
if 'winStop' in arguments:
    if arguments['winStop'] is not None:
        alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)

binOpts = rasterOpts['binOpts'][arguments['analysisName']]

if theseIteratorOpts['calcTimeROI'] and not arguments['loadFromFrames']:
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
    listOfROIMasks = []
    listOfExampleIndexes = []
#
if arguments['iteratorSuffix'] is not None:
    iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
else:
    iteratorSuffix = ''
if arguments['controlSet']:
    controlSuffix = '_ctrl'
else:
    controlSuffix = ''
iteratorPath = os.path.join(
    cvIteratorSubfolder,
    '{}_{}_{}{}{}_cvIterators.pickle'.format(
        blockBaseName,
        arguments['window'],
        arguments['alignQuery'],
        iteratorSuffix, controlSuffix))

if arguments['verbose']:
    prf.print_memory_usage('before load data')

if theseIteratorOpts['nCovariateBasisTerms'] > 1:
    lags = np.linspace(
        -1 * theseIteratorOpts['covariateHistoryLen'],
        theseIteratorOpts['covariateHistoryLen'],
        theseIteratorOpts['nCovariateBasisTerms']) / binOpts['binInterval']
    alignedAsigsKWargs['addLags'] = {'all': lags.astype(int).tolist()}
if theseIteratorOpts['forceBinInterval'] is not None:
    alignedAsigsKWargs['decimate'] = int(theseIteratorOpts['forceBinInterval'] / binOpts['binInterval'])
    alignedAsigsKWargs['rollingWindow'] = alignedAsigsKWargs['decimate']
#
# nSplits = theseIteratorOpts['nSplits']
listOfIterators = []
listOfDataFrames = []
if (not arguments['loadFromFrames']):
    print('loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    nSeg = len(dataBlock.segments)
    for segIdx in range(nSeg):
        if arguments['verbose']:
            prf.print_memory_usage(
                'fitting on segment {}'.format(segIdx))
        aakwa = alignedAsigsKWargs.copy()
        try:
            dataDF = ns5.alignedAsigsToDF(
                dataBlock,
                whichSegments=[segIdx],
                **aakwa)
        except Exception:
            traceback.print_exc()
            continue
        # trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        if theseIteratorOpts['calcTimeROI']:
            if arguments['controlSet']:
                aQ = theseIteratorOpts['timeROIOpts_control']['alignQuery']
                if aQ is None:
                    aQ = arguments['alignQuery']
                endMaskQuery = ash.processAlignQueryArgs(
                    namedQueries, alignQuery=aQ)
                ROIWinStart = theseIteratorOpts['timeROIOpts_control']['winStart']
                ROIWinStop = theseIteratorOpts['timeROIOpts_control']['winStop']
            else:
                aQ = theseIteratorOpts['timeROIOpts']['alignQuery']
                if aQ is None:
                    aQ = arguments['alignQuery']
                endMaskQuery = ash.processAlignQueryArgs(
                    namedQueries, alignQuery=aQ)
                ROIWinStart = theseIteratorOpts['timeROIOpts']['winStart']
                ROIWinStop = theseIteratorOpts['timeROIOpts']['winStop']
            evList = eventBlock.filter(
                objects=ns5.Event,
                name='seg{}_{}'.format(segIdx, eventName))
            assert len(evList) == 1
            targetTrialAnnDF = ns5.unitSpikeTrainArrayAnnToDF(evList, columnNames=aakwa['getMetaData'])
            try:
                targetTrialAnnDF = targetTrialAnnDF.query(endMaskQuery)
            except Exception:
                traceback.print_exc()
                pass
            targetMask = pd.Series(True, index=dataDF.index)
            # select custom time ranges 
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
            #
            colRenamer = {fN: fN.replace('#0', '') for fN in dataDF.columns.get_level_values('feature')}
            dataDF.rename(columns=colRenamer, level='feature', inplace=True)
        listOfDataFrames.append(dataDF)
    if arguments['lazy']:
        dataReader.file.close()
else:    # loading frames
    if not arguments['processAll']:
        experimentsToAssemble = {
            experimentName: [blockIdx]}
    else:
        experimentsToAssemble = theseIteratorOpts['experimentsToAssemble']
    currBlockNum = 0
    for expName, lOfBlocks in experimentsToAssemble.items():
        thisScratchFolder = os.path.join(scratchPath, expName)
        analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
            arguments, thisScratchFolder)
        thisDFFolder = os.path.join(analysisSubFolder, 'dataframes')
        for bIdx in lOfBlocks:
            theseArgs = arguments.copy()
            theseArgs['blockIdx'] = '{}'.format(bIdx)
            theseArgs['processAll'] = False
            thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
            dFPath = os.path.join(
                thisDFFolder,
                '{}_{}_df{}.h5'.format(
                    thisBlockBaseName,
                    arguments['window'],
                    iteratorSuffix))
            try:
                with pd.HDFStore(dFPath,  mode='r') as store:
                    theseDF = {}
                    dataKey = '/{}/data'.format(arguments['selectionName'])
                    if dataKey in store:
                        theseDF['main'] = pd.read_hdf(store, dataKey)
                        print('Loaded {} from {}'.format(dataKey, dFPath))
                    controlKey = '/{}/control'.format(arguments['selectionName'])
                    if controlKey in store:
                        theseDF['control'] = pd.read_hdf(store, controlKey)
                        print('Loaded {} from {}'.format(controlKey, dFPath))
                    assert len(theseDF.keys()) > 0
                    thisDF = pd.concat(theseDF, names=['controlFlag'])
                    colRenamer = {fN: fN.replace('#0', '') for fN in thisDF.columns.get_level_values('feature')}
                    thisDF.rename(columns=colRenamer, level='feature', inplace=True)
            except Exception:
                traceback.print_exc()
                print('Skipping...')
                continue
            print('Loaded {} from {}'.format(arguments['selectionName'], dFPath))
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
            dataDF=dataDF, **theseIteratorOpts['cvKWArgs'])
        listOfIterators.append(cvIterator)
else:
    exportDF = pd.concat(listOfDataFrames)
    print('exportDF.shape[0] =  {}'.format(exportDF.shape[0]))
    if theseIteratorOpts['controlProportion'] is not None:
        trialInfo = exportDF.index.to_frame().reset_index(drop=True)
        infoPerTrial = trialInfo.drop_duplicates(subset=['controlFlag', 'segment', 'originalIndex', 't'])
        valueCounts = infoPerTrial.groupby('controlFlag').count().iloc[:, 0]
        if theseIteratorOpts['controlProportion'] == 'majority':
            targetNControls = (
                infoPerTrial
                    .loc[infoPerTrial['controlFlag'] == 'main', :]
                    .groupby(stimulusConditionNames)
                    .count().iloc[:, 0].max())
        elif theseIteratorOpts['controlProportion'] == 'minority':
            targetNControls = (
                infoPerTrial
                    .loc[infoPerTrial['controlFlag'] == 'main', :]
                    .groupby(stimulusConditionNames)
                    .count().iloc[:, 0].min())
        else:
            targetNControls = int(theseIteratorOpts['controlProportion'] * valueCounts['main'])
        #
        controlIndices = infoPerTrial.index[infoPerTrial['controlFlag'] == 'control']
        dropIndices = default_rng().choice(controlIndices, size=(controlIndices.size - targetNControls), replace=False)
        keepMI = infoPerTrial.drop(index=dropIndices).set_index(['controlFlag', 'segment', 'originalIndex', 't']).index
        controlProportionMask = trialInfo.set_index(['controlFlag', 'segment', 'originalIndex', 't']).index.isin(keepMI)
        exportDF = exportDF.loc[controlProportionMask, :]
        print('After controlProportion deleter exportDF.shape[0] =  {}'.format(exportDF.shape[0]))
    else:
        controlProportionMask = None
    # reject bins where there aren't enough observations to average
    minBinMask = pd.Series(True, index=exportDF.index)
    if theseIteratorOpts['minBinCount']:
        for stimCnd, stimGrp in exportDF.groupby(stimulusConditionNames):
            binCount = stimGrp.groupby('bin').count().iloc[:, 0]
            # which bins need to be rejected?
            binsTooFew = binCount.index[binCount < theseIteratorOpts['minBinCount']]
            binsTooFewMask = stimGrp.index.get_level_values('bin').isin(binsTooFew)
            binsTooFewIndices = stimGrp.index[binsTooFewMask]
            # which indices don't have enough bin observations?
            minBinMask.loc[binsTooFewIndices] = False
            print('minBinMask.sum() = {}'.format(minBinMask.sum()))
        exportDF = exportDF.loc[minBinMask, :]
        print('After minBinMask exportDF.shape[0] = {}'.format(exportDF.shape[0]))
    else:
        minBinMask = None
    cvIterator = tdr.trainTestValidationSplitter(
        dataDF=exportDF, **theseIteratorOpts['cvKWArgs'])
    listOfIterators.append(cvIterator)
###
# cvIterator.plot_schema()
exportAAKWA = alignedAsigsKWargs.copy()
exportAAKWA.pop('unitNames', None)
exportAAKWA.pop('unitQuery', None)
iteratorMetadata = {
    'alignedAsigsKWargs': exportAAKWA,
    'iteratorsBySegment': listOfIterators,
    'iteratorOpts': theseIteratorOpts,
    'experimentsToAssemble': experimentsToAssemble
}
if theseIteratorOpts['calcTimeROI'] and (not arguments['loadFromFrames']):
    iteratorMetadata.update({
        'listOfROIMasks': listOfROIMasks,
        'listOfExampleIndexes': listOfExampleIndexes
    })
if arguments['processAll']:
    if controlProportionMask is not None:
        iteratorMetadata.update({
            'controlProportionMask': controlProportionMask
        })
    if minBinMask is not None:
        iteratorMetadata.update({
            'minBinMask': minBinMask
        })
print('saving\n{}\n'.format(iteratorPath))
if os.path.exists(iteratorPath):
    os.remove(iteratorPath)
with open(iteratorPath, 'wb') as f:
    pickle.dump(
        iteratorMetadata, f)

'''if arguments['saveDataFrame']:
    outputDFPath = os.path.join(
        dataFramesFolder,
        '{}_{}_{}_df{}.h5'.format(
            blockBaseName,
            arguments['window'],
            arguments['alignQuery'],
            iteratorSuffix))
    exportDF.to_hdf(outputDFPath, arguments['selectionName'], mode='w')'''
