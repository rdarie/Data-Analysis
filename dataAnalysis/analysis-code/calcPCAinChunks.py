"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --window=window                        process with short window? [default: long]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --inputBlockName=inputBlockName        filename for input block [default: fr_sqrt]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --estimatorName=estimatorName          filename for resulting estimator [default: pca]
    --selector=selector                    filename if using a unit selector
"""

import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import pdb
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, IncrementalPCA
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

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=False, getMetaData=False, decimate=5))
alignedAsigsKWargs['verbose'] = arguments['verbose']

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
fullEstimatorName = '{}_{}_{}_{}'.format(
    prefix,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])
estimatorPath = os.path.join(
    scratchFolder,
    fullEstimatorName + '.joblib')

if arguments['verbose']:
    prf.print_memory_usage('before load data')
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])

nSeg = len(dataBlock.segments)
nComp = len(alignedAsigsKWargs['unitNames'])
estimator = IncrementalPCA(
    n_components=nComp,
    batch_size=int(5 * nComp))

for segIdx in range(nSeg):
    if arguments['verbose']:
        prf.print_memory_usage('fitting on segment {}'.format(segIdx))
    alignedAsigsDF = ns5.alignedAsigsToDF(
        dataBlock,
        whichSegments=[segIdx],
        **alignedAsigsKWargs)
    prf.print_memory_usage('just loaded firing rates, fitting')
    estimator.partial_fit(alignedAsigsDF.to_numpy())
    saveColumns = alignedAsigsDF.columns.to_list()
    del alignedAsigsDF
    gc.collect()

if arguments['lazy']:
    dataReader.file.close()
jb.dump(estimator, estimatorPath)

alignedAsigsKWargs['unitNames'] = saveColumns
alignedAsigsKWargs.pop('unitQuery')
estimatorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(estimatorPath),
    'name': arguments['estimatorName'],
    'inputBlockName': arguments['inputBlockName'],
    'inputFeatures': saveColumns,
    'alignedAsigsKWargs': alignedAsigsKWargs
    }

with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
    pickle.dump(
        estimatorMetadata, f)
