"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --profile                              print time and mem diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: short]
    --winStart=winStart                    start of window
    --winStop=winStop                      end of window
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --estimatorName=estimatorName          estimator filename
    --datasetName=datasetName              dataset used to train estimator (use to get loading arguments)
    --selectionName=selectionName          dataset used to train estimator (use to get loading arguments)
    --datasetExp=datasetExp                dataset used to train estimator (use to get loading arguments)
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
"""

import logging
logging.captureWarnings(True)
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import os
import quantities as pq
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import dill as pickle
import sys
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from dataAnalysis.analysis_code.namedQueries import namedQueries
from docopt import docopt

for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'unitQuery': None, 'profile': False, 'winStop': '1000', 'lazy': False, 'window': 'XL',
        'inputBlockSuffix': 'lfp_CAR_spectral', 'datasetName': 'Block_XL_df_ca', 'blockIdx': '3',
        'selectionName': 'lfp_CAR_spectral', 'estimatorName': 'mahal', 'datasetExp': '202101281100-Rupert',
        'winStart': '-600', 'alignFolderName': 'motion', 'verbose': False, 'maskOutlierBlocks': True,
        'alignQuery': 'starting', 'processAll': False, 'inputBlockPrefix': 'Block', 'exp': 'exp202101281100',
        'analysisName': 'hiRes'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')

'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchFolder, blockBaseName, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, metaDataToCategories=False,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=essentialMetadataFields, decimate=1))
if 'windowSize' not in alignedAsigsKWargs:
    alignedAsigsKWargs['windowSize'] = [ws for ws in rasterOpts['windowSizes'][arguments['window']]]
if 'winStart' in arguments:
    if arguments['winStart'] is not None:
        alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
if 'winStop' in arguments:
    if arguments['winStop'] is not None:
        alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)
alignedAsigsKWargs['verbose'] = arguments['verbose']
alignedAsigsKWargs['getFeatureMetaData'] = ['xCoords', 'yCoords', 'freqBandName', 'parentFeature']

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))
###
oldWay = False
if oldWay:
    # alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    #     namedQueries, scratchFolder, **arguments)
    #
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        arguments['estimatorName'] + '.joblib')
    estimator = jb.load(estimatorPath)
    alignedAsigsKWargs.update(estimatorMetadata['alignedAsigsKWargs'])
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    normalizeDataset = None
    extendedFeatureMeta = None
else:
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
    estimatorName = arguments['estimatorName']
    fullEstimatorName = '{}_{}_{}'.format(
        estimatorName, datasetName, selectionName)
    #
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    if arguments['datasetExp'] is not None:
        estimatorsSubFolder = estimatorsSubFolder.replace(
            experimentName, arguments['datasetExp'])
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    if arguments['datasetExp'] is not None:
        dataFramesFolder = dataFramesFolder.replace(experimentName, arguments['datasetExp'])
    '''datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )'''
    scoresPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.joblib'
        )
    estimatorMetadataPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_meta.pickle'
        )
    estimator = jb.load(estimatorPath)
    with open(estimatorMetadataPath, 'rb') as _f:
        estimatorMeta = pickle.load(_f)
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        for discardEntry in ['plotting', 'showFigures']:
            _ = loadingMeta['arguments'].pop(discardEntry, None)
    if 'normalizeDataset' in loadingMeta:
        normalizeDataset = loadingMeta['normalizeDataset']
        normalizationParams = loadingMeta['normalizationParams']
    else:
        normalizeDataset = None
    matchLoadingArgs = [
        'getMetaData', 'concatOn', 'transposeToColumns',
        'addLags', 'procFun', 'getFeatureMetaData',
        'decimate', 'rollingWindow']
    for aakwaEntry in matchLoadingArgs:
        if aakwaEntry in loadingMeta['alignedAsigsKWargs']:
            alignedAsigsKWargs[aakwaEntry] = loadingMeta['alignedAsigsKWargs'][aakwaEntry]
    #
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    print('alignedAsigsKWargs[dataQuery] = {}'.format(alignedAsigsKWargs['dataQuery']))
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **loadingMeta['arguments'])
#
with open(
    os.path.join(
        estimatorPath.replace('.joblib', '_meta.pickle')),
        'rb') as f:
    estimatorMetadata = pickle.load(f)
#
alignedAsigsKWargs['verbose'] = arguments['verbose']
#
outputPath = os.path.join(
    alignSubFolder,
    blockBaseName + inputBlockSuffix + '_{}_{}'.format(
        estimatorMetadata['name'], arguments['window']))
#
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
#
if arguments['verbose']:
    prf.print_memory_usage('Loading {}'.format(triggeredPath))

alignedAsigsDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
print(alignedAsigsDF.columns)
'''if extendedFeatureMeta is not None:
    featureMeta = alignedAsigsDF.columns.to_frame().reset_index(drop=True)
    loadedMeta = featureMeta.loc[:, ['feature', 'lag']]
    comparisonMeta = extendedFeatureMeta.loc[:, ['feature', 'lag']]
    comparisonErrorMsg = "Error: loaded data columns\n\n{}\n\ndoes not equal dataset columns\n\n{}\n\n".format(featureMeta.loc[:, ['feature', 'lag']], extendedFeatureMeta.loc[:, ['feature', 'lag']])
    assert (loadedMeta == comparisonMeta).all(axis=None), comparisonErrorMsg
    alignedAsigsDF.columns = pd.MultiIndex.from_frame(extendedFeatureMeta)'''
if normalizeDataset is not None:
    alignedAsigsDF = normalizeDataset(alignedAsigsDF, normalizationParams)
alignedAsigsDF.rename(columns=lambda x: x.replace('#0', ''), level='feature', inplace=True)
# pdb.set_trace()
if hasattr(estimator, 'transform'):
    features = estimator.transform(alignedAsigsDF)
if arguments['profile']:
    prf.print_memory_usage('after estimator.transform')
#
if 'outputFeatures' in estimatorMetadata:
    if isinstance(estimatorMetadata['outputFeatures'], pd.MultiIndex):
        featureNames = pd.Index(estimatorMetadata['outputFeatures'].get_level_values('feature'), dtype=str)
    elif isinstance(estimatorMetadata['outputFeatures'], pd.Index):
        featureNames = estimatorMetadata['outputFeatures']
    else:
        featureNames = pd.Index(estimatorMetadata['outputFeatures'])
        featureNames.name = 'feature'
else:
    featureNames = pd.Index([
        estimatorMetadata['name'] + '{:0>3}#0'.format(i)
        for i in range(features.shape[1])])
#
trialTimes = np.unique(alignedAsigsDF.index.get_level_values('t'))
tBins = np.unique(alignedAsigsDF.index.get_level_values('bin'))
alignedFeaturesDF = pd.DataFrame(
    features, index=alignedAsigsDF.index,
    # columns=featureNames
    columns=estimatorMetadata['outputFeatures']
    )
alignedFeaturesDF.columns.name = 'feature'
del alignedAsigsDF
#
spikeTrainMeta = {
    'units': pq.s,
    'wvfUnits': pq.dimensionless,
    'left_sweep': (-1) * tBins[0] * pq.s,
    't_start': min(0, trialTimes[0]) * pq.s,
    't_stop': trialTimes[-1] * pq.s,
    'sampling_rate': ((tBins[1] - tBins[0]) ** (-1)) * pq.Hz
    }
masterBlock = ns5.alignedAsigDFtoSpikeTrain(
    alignedFeaturesDF, spikeTrainMeta=spikeTrainMeta, matchSamplingRate=False)
if arguments['lazy']:
    dataReader.file.close()
if os.path.exists(outputPath + '.nix'):
    os.remove(outputPath + '.nix')
masterBlock = ns5.purgeNixAnn(masterBlock)
print('Writing {}.nix...'.format(outputPath))
writer = ns5.NixIO(filename=outputPath + '.nix', mode='ow')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
