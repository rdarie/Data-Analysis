"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: hiRes]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --selectionName=selectionName          name in h5 for the saved data
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --iteratorSuffix=iteratorSuffix        filename for cross_val iterator
    --needsRollingWindow                   need to decimate to align to spectrogram?
    --selector=selector                    filename if using a unit selector
    --resetHDF                             delete the h5 file if it exists?
    --controlSet                           regular data, or control?
"""

import logging
logging.captureWarnings(True)
import sys
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

if arguments['plotting']:
    import matplotlib, os
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if 'CCV_HEADLESS' in os.environ:
        matplotlib.use('PS')   # generate postscript output
    else:
        matplotlib.use('QT5Agg')   # generate interactive output
#
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set(
        context='talk', style='dark',
        palette='dark', font='sans-serif',
        font_scale=1.5, color_codes=True)

import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.hampel as hf_hampel
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
import traceback
from copy import deepcopy
idxSl = pd.IndexSlice
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
if arguments['alignFolderName'] == 'stim':
    if blockExperimentType == 'proprio-motionOnly':
        print('skipping block {} (no stim)'.format(arguments['blockIdx']))
        sys.exit()
if arguments['alignFolderName'] == 'motion':
    if blockExperimentType == 'proprio-miniRC':
        print('skipping block {} (no movement)'.format(arguments['blockIdx']))
        sys.exit()

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
dataFramesFolder = os.path.join(
    analysisSubFolder, 'dataframes'
    )  ## must exist from previous call to calcTestTrainSplit.py
cvIteratorSubfolder = os.path.join(
    alignSubFolder, 'testTrainSplits')
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
    print('Loading cv iterator from\n{}\n'.format(iteratorPath))
with open(iteratorPath, 'rb') as f:
    loadingMeta = pickle.load(f)
iteratorsBySegment = loadingMeta['iteratorsBySegment']
iteratorOpts = loadingMeta['iteratorOpts']

listOfDataFrames = []

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))

alignedAsigsKWargs = loadingMeta['alignedAsigsKWargs'].copy()
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['verbose'] = arguments['verbose']
if 'procFun' in iteratorOpts:
    if arguments['selectionName'] in iteratorOpts['procFun']:
        if iteratorOpts['procFun'][arguments['selectionName']] is not None:
            print('Applying processing to signal: {}'.format(iteratorOpts['procFun'][arguments['selectionName']]))
            alignedAsigsKWargs['procFun'] = eval(iteratorOpts['procFun'][arguments['selectionName']])
#
outlierResultsPathCSV = os.path.join(
    scratchFolder, 'outlierTrials', arguments['alignFolderName'],
    blockBaseName + '_{}_outliers.csv'.format(arguments['window']))
# outlierTrialsForReference = pd.read_csv(outlierResultsPathCSV).set_index(['segment', 't'])
outlierTrialsForReference = ash.processOutlierTrials(
    scratchFolder, blockBaseName,
    maskOutlierBlocks=True,
    invertOutlierBlocks=False,
    window=arguments['window'], alignFolderName=arguments['alignFolderName'],
    includeDeviation=True)
#
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
nSeg = len(dataBlock.segments)
nSegLoading = len(loadingMeta['iteratorsBySegment'])
blockWasChunked = (nSeg > nSegLoading)
# pdb.set_trace()
loadingMeta['alignedAsigsKWargs'] = alignedAsigsKWargs.copy()
for segIdx in range(nSegLoading):
    if arguments['verbose']:
        prf.print_memory_usage('extracting data on segment {}'.format(segIdx))
    aakwa = deepcopy(alignedAsigsKWargs)
    aakwa['verbose'] = False
    if 'listOfROIMasks' in loadingMeta:
        aakwa['finalIndexMask'] = loadingMeta['listOfROIMasks'][segIdx]
    if arguments['verbose']:
        prf.print_memory_usage('Loading {}'.format(triggeredPath))
    if blockWasChunked:
        whichSegments = None
        overrideSegIdx = segIdx
    else:
        whichSegments = [segIdx]
        overrideSegIdx = None
    dataDF = ns5.alignedAsigsToDF(
        dataBlock, whichSegments=whichSegments, overrideSegIdx=overrideSegIdx, **aakwa)
    if 'postLoadProcFun' in iteratorOpts:
        if arguments['selectionName'] in iteratorOpts['postLoadProcFun']:
            if iteratorOpts['postLoadProcFun'][arguments['selectionName']] is not None:
                print('Applying processing to signal: {}'.format(iteratorOpts['postLoadProcFun'][arguments['selectionName']]))
                fcn = eval(iteratorOpts['postLoadProcFun'][arguments['selectionName']])
                dataDF = fcn(dataDF, None)
    print('dataDF.index.names = {}'.format(dataDF.index.names))
    print('dataDF.columns = {}'.format(dataDF.columns.to_frame().reset_index(drop=True)))
    #
    colRenamer = {fN: fN.replace('#0', '') for fN in dataDF.columns.get_level_values('feature')}
    dataDF.rename(columns=colRenamer, level='feature', inplace=True)
    #
    if 'listOfExampleIndexes' in loadingMeta:
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        loadedTrialInfo = loadingMeta['listOfExampleIndexes'][segIdx].to_frame().reset_index(drop=True)
        targetAnns = np.intersect1d(trialInfo.columns, loadedTrialInfo.columns)
        targetAnns = [
            ta
            for ta in targetAnns
            if ta not in [
                'stimDelay', 'unitAnnotations',
                'detectionDelay', 'originalIndex',
                'freqBandName', 'xCoords', 'yCoords', 'parentFeature']]
        try:
            metaDataMatches = (trialInfo.loc[:, targetAnns] == loadedTrialInfo.loc[:, targetAnns])
            assert metaDataMatches.all(axis=None)
        except Exception:
            traceback.print_exc()
            print('(this data) trialInfo.loc[:, targetAnns] =\n{}'.format(trialInfo.loc[:, targetAnns]))
            print('(loaded from iterator) loadedTrialInfo.loc[:, targetAnns] =\n{}'.format(loadedTrialInfo.loc[:, targetAnns]))
            print('(trialInfo.loc[:, targetAnns] == loadedTrialInfo.loc[:, targetAnns]).all(axis=0) =\n{}'.format(metaDataMatches.all(axis=0)))
            nonMatchingAnns = metaDataMatches.columns[~metaDataMatches.all(axis=0)]
            nonMatchingIndex = metaDataMatches.index[~metaDataMatches.all(axis=1)]
            print('(contradictory this data)\n{}'.format(trialInfo.drop_duplicates('t').loc[:, nonMatchingAnns]))
            print('(contradictory loaded from iterator)\n{}'.format(loadedTrialInfo.drop_duplicates('t').loc[:, nonMatchingAnns]))
            pdb.set_trace()
    listOfDataFrames.append(dataDF)
if arguments['verbose']:
    prf.print_memory_usage('Done loading')
if arguments['lazy']:
    dataReader.file.close()

exportDF = pd.concat(listOfDataFrames)
outputDFPath = os.path.join(
    dataFramesFolder,
    '{}_{}_df{}.h5'.format(
        blockBaseName,
        arguments['window'],
        iteratorSuffix))
if arguments['resetHDF']:
    if os.path.exists(outputDFPath):
        os.remove(outputDFPath)
#
if arguments['controlSet']:
    trialInfo = exportDF.index.to_frame().reset_index(drop=True)
    for sCN in stimulusConditionNames:
        trialInfo.loc[:, sCN] = ns5.metaFillerLookup[sCN]
    trialInfo.loc[:, 'originalIndex'] = trialInfo['originalIndex'] + int(1e6)
    exportDF.index = pd.MultiIndex.from_frame(trialInfo)
    exportKey = '/{}/control'.format(arguments['selectionName'])
    outputLoadingMetaPath = os.path.join(
        dataFramesFolder,
        '{}_{}_df{}_ctrl_{}_meta.pickle'.format(
            blockBaseName,
            arguments['window'],
            iteratorSuffix, arguments['selectionName']))
else:
    outputLoadingMetaPath = os.path.join(
        dataFramesFolder,
        '{}_{}_df{}_{}_meta.pickle'.format(
            blockBaseName,
            arguments['window'],
            iteratorSuffix, arguments['selectionName']))
    exportKey = '/{}/data'.format(arguments['selectionName'])
if arguments['verbose']:
    prf.print_memory_usage(
        'Saving {} to {}'.format(exportKey, outputDFPath))

trialInfo = exportDF.index.to_frame().reset_index(drop=True)
trialInfo.loc[:, 't'] = np.round(trialInfo['t'], decimals=9)
if outlierTrialsForReference is not None:
    trialInfo.loc[:, 'isOutlierTrial'] = trialInfo.set_index(['segment', 't']).index.map(outlierTrialsForReference['rejectBlock'])
    trialInfo.loc[:, 'outlierDeviation'] = trialInfo.set_index(['segment', 't']).index.map(outlierTrialsForReference['deviation'])
else:
    trialInfo.loc[:, 'isOutlierTrial'] = False
    trialInfo.loc[:, 'outlierDeviation'] = 0
exportDF.index = pd.MultiIndex.from_frame(trialInfo)
##
ddfColumns = exportDF.columns.to_frame().reset_index(drop=True)
ddfColumns.loc[ddfColumns['parentFeature'] == 'NA', 'parentFeature'] = ddfColumns.loc[ddfColumns['parentFeature'] == 'NA', 'feature']
exportDF.columns = pd.MultiIndex.from_frame(ddfColumns)
exportDF.to_hdf(outputDFPath, exportKey, mode='a')
##
featureGroupNames = [cN for cN in exportDF.columns.names]
maskList = []
haveAllGroup = False
allGroupIdx = pd.MultiIndex.from_tuples(
    [tuple('all' for fgn in featureGroupNames)],
    names=featureGroupNames)
allMask = pd.Series(True, index=exportDF.columns).to_frame()
allMask.columns = allGroupIdx
maskList.append(allMask.T)
nFreqBands = exportDF.columns.get_level_values('freqBandName').unique().size
makeMaskPerFreqBand = False
if 'maskEachFreqBand' in iteratorOpts:
    if (iteratorOpts['maskEachFreqBand']) and (nFreqBands > 1):
        makeMaskPerFreqBand = True
if makeMaskPerFreqBand:
    # each freq band
    for name, group in exportDF.groupby('freqBandName', axis='columns'):
        attrValues = ['all' for fgn in featureGroupNames]
        attrValues[featureGroupNames.index('freqBandName')] = name
        thisMask = pd.Series(
            exportDF.columns.isin(group.columns),
            index=exportDF.columns).to_frame()
        if np.all(thisMask):
            haveAllGroup = True
            thisMask.columns = allGroupIdx
        else:
            thisMask.columns = pd.MultiIndex.from_tuples(
                (attrValues, ), names=featureGroupNames)
        maskList.append(thisMask.T)
# each lag
for name, group in exportDF.groupby('lag', axis='columns'):
    attrValues = ['all' for fgn in featureGroupNames]
    attrValues[featureGroupNames.index('lag')] = name
    thisMask = pd.Series(
        exportDF.columns.isin(group.columns),
        index=exportDF.columns).to_frame()
    if not np.all(thisMask):
        # all group already covered
        thisMask.columns = pd.MultiIndex.from_tuples(
            (attrValues, ), names=featureGroupNames)
        maskList.append(thisMask.T)
'''
# each parent feature
for name, group in exportDF.groupby('parentFeature', axis='columns'):
    attrValues = ['all' for fgn in featureGroupNames]
    attrValues[featureGroupNames.index('parentFeature')] = name
    thisMask = pd.Series(
        exportDF.columns.isin(group.columns),
        index=exportDF.columns).to_frame()
    if np.all(thisMask):
        haveAllGroup = True
        thisMask.columns = allGroupIdx
    else:
        thisMask.columns = pd.MultiIndex.from_tuples(
            (attrValues, ), names=featureGroupNames)
    maskList.append(thisMask.T)'''
#
maskDF = pd.concat(maskList)
maskParams = [
    {k: v for k, v in zip(maskDF.index.names, idxItem)}
    for idxItem in maskDF.index
    ]
maskParamsStr = [
    '{}'.format(idxItem).replace("'", '')
    for idxItem in maskParams]
maskDF.loc[:, 'maskName'] = maskParamsStr
maskDF.set_index('maskName', append=True, inplace=True)
masksKey = '/{}/featureMasks'.format(arguments['selectionName'])
if arguments['verbose']:
    prf.print_memory_usage(
        'Saving {} to {}'.format(masksKey, outputDFPath))
maskDF.to_hdf(outputDFPath, masksKey, mode='a')
#
print('saving loading meta to \n{}\n'.format(outputLoadingMetaPath))
if os.path.exists(outputLoadingMetaPath):
    os.remove(outputLoadingMetaPath)
## update loadingMeta['arguments'] re: feature names
for kN in ['selectionName', 'unitQuery']:
    if kN in arguments:
        loadingMeta['arguments'][kN] = arguments[kN]
## update loadingMeta['alignedAsigsKW'] re: feature names
with open(outputLoadingMetaPath, 'wb') as _f:
    pickle.dump(loadingMeta, _f)

print('\n' + '#' * 50 + '\n{}\nComplete\n'.format(__file__) + '#' * 50 + '\n')
