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
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --estimatorName=estimatorName          estimator filename
    --originDatasetName=originDatasetName  dataset used to train estimator (use to get loading arguments)
    --datasetName=datasetName              dataset used to train estimator (use to get loading arguments)
    --selectionName=selectionName          dataset used to train estimator (use to get loading arguments)
    --originDatasetExp=originDatasetExp    dataset used to train estimator (use to get loading arguments)
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
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
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
originDatasetName = arguments['originDatasetName']
datasetName = arguments['datasetName']
selectionName = arguments['selectionName']
estimatorName = arguments['estimatorName']
fullEstimatorName = '{}_{}_{}'.format(
    estimatorName, originDatasetName, selectionName)
outputSelectionName = '{}_{}'.format(
    selectionName, estimatorName)
estimatorsSubFolder = os.path.join(
    analysisSubFolder, 'estimators')
if arguments['originDatasetExp'] is not None:
    estimatorsSubFolder = estimatorsSubFolder.replace(
        experimentName, arguments['originDatasetExp'])
if not os.path.exists(estimatorsSubFolder):
    os.makedirs(estimatorsSubFolder)
originDataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
if arguments['originDatasetExp'] is not None:
    originDataFramesFolder = originDataFramesFolder.replace(experimentName, arguments['originDatasetExp'])
originDatasetPath = os.path.join(
    originDataFramesFolder,
    originDatasetName + '.h5'
    )
dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
datasetPath = os.path.join(
    dataFramesFolder,
    datasetName + '.h5'
    )
#
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.joblib'
    )
estimator = jb.load(estimatorPath)
#
originLoadingMetaPath = os.path.join(
    originDataFramesFolder,
    originDatasetName + '_{}'.format(outputSelectionName) + '_meta.pickle'
    )
with open(originLoadingMetaPath, 'rb') as _f:
    originFeatureLoadingMeta = pickle.load(_f)
#
loadingMetaPath = os.path.join(
    dataFramesFolder,
    datasetName + '_{}'.format(selectionName) + '_meta.pickle'
    )
with open(loadingMetaPath, 'rb') as _f:
    dataLoadingMeta = pickle.load(_f)
#
outputLoadingMetaPath = os.path.join(
    dataFramesFolder,
    datasetName + '_{}'.format(outputSelectionName) + '_meta.pickle'
    )
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
#featureMasks
dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
#
if hasattr(estimator, 'transform'):
    features = estimator.transform(dataDF)
elif hasattr(estimator, 'mahalanobis'):
    features = estimator.mahalanobis(dataDF)
if arguments['profile']:
    prf.print_memory_usage('after estimator.transform')
#
if 'outputFeatures' in estimatorMetadata:
    featureNames = estimatorMetadata['outputFeatures']
else:
    featureNames = pd.Index([
        estimatorMetadata['name'] + '{:0>3}#0'.format(i)
        for i in range(features.shape[1])])
#
featuresDF = pd.DataFrame(
    features, index=dataDF.index,
    columns=featureNames
    )
del dataDF
featuresDF.to_hdf(
    datasetPath,
    '/{}/data'.format(outputSelectionName),
    mode='a')
###
featureMasksDF = pd.read_hdf(originDatasetPath, '/{}/data'.format(outputSelectionName))
featureMasksDF.to_hdf(
    datasetPath,
    '/{}/featureMasks'.format(outputSelectionName),
    mode='a')
###
loadingMeta = originFeatureLoadingMeta.copy()
for key in ['iteratorsBySegment', 'iteratorOpts']:
    loadingMeta[key] = dataLoadingMeta[key]
with open(outputLoadingMetaPath, 'wb') as f:
    pickle.dump(loadingMeta, f)