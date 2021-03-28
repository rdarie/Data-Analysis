"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --estimatorName=estimatorName          filename for resulting estimator [default: pca]
    --selector=selector                    filename if using a unit selector
"""

import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from namedQueries import namedQueries
import pdb
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import SparsePCA
from sklearn.pipeline import make_pipeline, Pipeline
import joblib as jb
import dill as pickle
import gc
from currentExperiment import parseAnalysisOptions
from docopt import docopt

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchPath, blockBaseName, **arguments)

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=True, decimate=1))
alignedAsigsKWargs['verbose'] = arguments['verbose']

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))

fullEstimatorName = '{}_{}_{}_{}'.format(
    blockBaseName,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])

estimatorSubFolder = os.path.join(
    analysisSubFolder, 'estimators'
    )

if not os.path.exists(estimatorSubFolder):
    os.makedirs(estimatorSubFolder)

estimatorPath = os.path.join(
    estimatorSubFolder,
    arguments['estimatorName'] + '.joblib')
    # fullEstimatorName + '.joblib')

if arguments['verbose']:
    prf.print_memory_usage('before load data')

print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])

nSeg = len(dataBlock.segments)
dataList = []
for segIdx in range(nSeg):
    if arguments['verbose']:
        prf.print_memory_usage('fitting on segment {}'.format(segIdx))
    dataDF = ns5.alignedAsigsToDF(
        dataBlock,
        whichSegments=[segIdx],
        **alignedAsigsKWargs)
    dataList.append(dataDF)
allDataDF = pd.concat(dataList)
# TODO: use trial metadata to ensure balanced dataset?
# trialInfo = dataDF.index.to_frame().reset_index(drop=True)
prf.print_memory_usage('just loaded data, fitting')
# nComp = len(alignedAsigsKWargs['unitNames'])
nComp = dataDF.columns.shape[0]
estimator = SparsePCA(
    n_components=None)
estimator.fit(dataDF.to_numpy())
saveUnitNames = [cN[0] for cN in dataDF.columns]
del dataDF
gc.collect()
#
prf.print_memory_usage('Done fitting')

if arguments['lazy']:
    dataReader.file.close()
jb.dump(estimator, estimatorPath)

alignedAsigsKWargs['unitNames'] = saveUnitNames
alignedAsigsKWargs['unitQuery'] = None
alignedAsigsKWargs.pop('dataQuery')
estimatorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(estimatorPath),
    'name': arguments['estimatorName'],
    'inputBlockSuffix': inputBlockSuffix,
    'blockBaseName': blockBaseName,
    'inputFeatures': saveUnitNames,
    'alignedAsigsKWargs': alignedAsigsKWargs
    }

with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
    pickle.dump(
        estimatorMetadata, f)
