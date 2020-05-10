"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --window=window                        process with short window? [default: long]
    --verbose                              print diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --estimatorName=estimatorName          filename for resulting estimator [default: pca]
    --inputBlockName=inputBlockName        filename for resulting estimator [default: fr_sqrt]
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --selector=selector                    filename if using a unit selector
"""

import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
#  import numpy as np
#  import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  import neo
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
import joblib as jb
import dill as pickle
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import gc
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

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=False, getMetaData=False, decimate=5))

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
fullEstimatorName = '{}_{}_{}_{}'.format(
    prefix,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])
estimatorPath = os.path.join(
    analysisSubFolder,
    fullEstimatorName + '.joblib')
prf.print_memory_usage('before load data')
dataReader, dataBlock = ns5.blockFromPath(triggeredPath, lazy=arguments['lazy'])

masterSpikeMat = ns5.alignedAsigsToDF(
    dataBlock,
    **alignedAsigsKWargs).to_numpy()
prf.print_memory_usage('just loaded firing rates')
if arguments['lazy']:
    dataReader.file.close()
#  free up memory
del dataBlock
gc.collect()

nComp = masterSpikeMat.shape[1]
pca = IncrementalPCA(
    n_components=nComp,
    batch_size=int(3 * nComp))
estimator = Pipeline([('dimred', pca)])
prf.print_memory_usage('starting fit')

estimator.fit(masterSpikeMat)

jb.dump(estimator, estimatorPath)

estimatorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(estimatorPath),
    'name': arguments['estimatorName'],
    'inputFeatures': masterSpikeMat.columns.to_list(),
    'alignedAsigsKWargs': alignedAsigsKWargs
    }

with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
    pickle.dump(
        estimatorMetadata, f)