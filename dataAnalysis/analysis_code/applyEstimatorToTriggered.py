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
    --matchDownsampling                    match downsampling? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: short]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --estimatorName=estimatorName          estimator filename
    --datasetName=datasetName              dataset used to train estimator (use to get loading arguments)
    --datasetExp=datasetExp                dataset used to train estimator (use to get loading arguments)
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
"""
#
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from namedQueries import namedQueries
import os
import quantities as pq
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import dill as pickle
import sys
from currentExperiment import parseAnalysisOptions
from docopt import docopt

for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)

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
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], arguments['datasetName'])
    #
    estimatorsSubFolder = os.path.join(
        alignSubFolder, 'estimators')
    if arguments['datasetExp'] is not None:
        estimatorsSubFolder = estimatorsSubFolder.replace(experimentName, arguments['datasetExp'])
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(alignSubFolder, 'dataframes')
    if arguments['datasetExp'] is not None:
        dataFramesFolder = dataFramesFolder.replace(experimentName, arguments['datasetExp'])
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    scoresPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.joblib'
        )
    estimator = jb.load(estimatorPath)
    with open(datasetPath.replace('.h5', '_meta.pickle'), 'rb') as _f:
        loadingMeta = pickle.load(_f)
        for discardEntry in ['plotting', 'showFigures']:
            _ = loadingMeta['arguments'].pop(discardEntry, None)
    if 'normalizeDataset' in loadingMeta:
        normalizeDataset = loadingMeta['normalizeDataset']
        normalizationParams = loadingMeta['normalizationParams']
    else:
        normalizeDataset = None
    featureMasks = pd.read_hdf(datasetPath, datasetName + '_featureMasks')
    extendedFeatureMeta = featureMasks.columns.to_frame().reset_index(drop=True)
    for aakwaEntry in ['getMetaData', 'concatOn', 'transposeToColumns', 'addLags', 'procFun']:
        if aakwaEntry in loadingMeta['alignedAsigsKWargs']:
            alignedAsigsKWargs[aakwaEntry] = loadingMeta['alignedAsigsKWargs'][aakwaEntry]
    if arguments['matchDownsampling']:
        if 'decimate' in loadingMeta['alignedAsigsKWargs']:
            alignedAsigsKWargs['decimate'] = loadingMeta['alignedAsigsKWargs']['decimate']
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    print('alignedAsigsKWargs[dataQuery] = {}'.format(alignedAsigsKWargs['dataQuery']))
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(namedQueries, scratchFolder, **arguments)
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
if extendedFeatureMeta is not None:
    featureMeta = alignedAsigsDF.columns.to_frame().reset_index(drop=True)
    assert (featureMeta.loc[:, ['feature', 'lag']] == extendedFeatureMeta.loc[:, ['feature', 'lag']]).all(axis=None)
    alignedAsigsDF.columns = pd.MultiIndex.from_frame(extendedFeatureMeta)
if normalizeDataset is not None:
    alignedAsigsDF = normalizeDataset(alignedAsigsDF, normalizationParams)
if hasattr(estimator, 'transform'):
    features = estimator.transform(alignedAsigsDF)
elif hasattr(estimator, 'mahalanobis'):
    features = estimator.mahalanobis(alignedAsigsDF)
if arguments['profile']:
    prf.print_memory_usage('after estimator.transform')
#
if 'outputFeatures' in estimatorMetadata:
    featureNames = estimatorMetadata['outputFeatures']
else:
    featureNames = [
        estimatorMetadata['name'] + '{:0>3}'.format(i)
        for i in range(features.shape[1])]
trialTimes = np.unique(alignedAsigsDF.index.get_level_values('t'))
tBins = np.unique(alignedAsigsDF.index.get_level_values('bin'))
alignedFeaturesDF = pd.DataFrame(
    features, index=alignedAsigsDF.index, columns=featureNames)
if isinstance(alignedFeaturesDF.columns, pd.MultiIndex):
    alignedFeaturesDF.columns = alignedFeaturesDF.columns.get_level_values('feature')
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
masterBlock = ns5.purgeNixAnn(masterBlock)
print('Writing {}.nix...'.format(outputPath))
writer = ns5.NixIO(filename=outputPath + '.nix')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
