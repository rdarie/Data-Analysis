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
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
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
import pandas as pd
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
else:
    datasetName = arguments['datasetName']
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], arguments['datasetName'])
    #
    estimatorsSubFolder = os.path.join(
        alignSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(alignSubFolder, 'dataframes')
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
            _ = loadingMeta['arguments'].pop(discardEntry)
    '''for loadingEntry in ['unitQuery']:
        if loadingEntry in loadingMeta['arguments']:
            arguments[loadingEntry] = loadingMeta['arguments'][loadingEntry]'''
    for aakwaEntry in ['getMetaData', 'concatOn', 'transposeToColumns']:
        if aakwaEntry in loadingMeta['alignedAsigsKWargs']:
            alignedAsigsKWargs[aakwaEntry] = loadingMeta['alignedAsigsKWargs'][aakwaEntry]
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(namedQueries, scratchFolder, **arguments)
#
with open(
    os.path.join(
        estimatorPath.replace('.joblib', '_meta.pickle')),
        'rb') as f:
    estimatorMetadata = pickle.load(f)
#
# Reduce time sample even further
# alignedAsigsKWargs.update(dict(getMetaData=True, decimate=20))
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
# pdb.set_trace()
if hasattr(estimator, 'transform'):
    features = estimator.transform(alignedAsigsDF.to_numpy())
elif hasattr(estimator, 'mahalanobis'):
    features = estimator.mahalanobis(alignedAsigsDF.to_numpy())
if arguments['profile']:
    prf.print_memory_usage('after estimator.transform')
#
if 'outputFeatures' in estimatorMetadata:
    featureNames = estimatorMetadata['outputFeatures']
else:
    featureNames = [
        estimatorMetadata['name'] + '{:0>3}'.format(i)
        for i in range(features.shape[1])]
#
# colNames = pd.MultiIndex.from_arrays(
#     [featureNames, [0 for fN in featureNames]],
#     names=['feature', 'lag'])
alignedFeaturesDF = pd.DataFrame(
    features, index=alignedAsigsDF.index, columns=featureNames)
alignedFeaturesDF.columns.name = 'feature'
del alignedAsigsDF
#
masterBlock = ns5.alignedAsigDFtoSpikeTrain(
    alignedFeaturesDF, dataBlock=dataBlock)
if arguments['lazy']:
    dataReader.file.close()
masterBlock = ns5.purgeNixAnn(masterBlock)
print('Writing {}.nix...'.format(outputPath))
writer = ns5.NixIO(filename=outputPath + '.nix')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
