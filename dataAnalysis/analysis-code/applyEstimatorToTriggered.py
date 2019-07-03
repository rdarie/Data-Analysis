"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --window=window                        process with short window? [default: short]
    --estimator=estimator                  estimator filename
    --unitQuery=unitQuery                  how to restrict channels?
"""

import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import os
import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

estimatorPath = os.path.join(
    scratchFolder,
    arguments['estimator'] + '.joblib')
with open(
    os.path.join(
        scratchFolder,
        arguments['estimator'] + '_meta.pickle'),
        'rb') as f:
    estimatorMetadata = pickle.load(f)
estimator = jb.load(
    os.path.join(scratchFolder, estimatorMetadata['path']))

alignedAsigsKWargs.update(estimatorMetadata['alignedAsigsKWargs'])
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs.update(dict(getMetaData=True, decimate=1))
alignedAsigsKWargs['verbose'] = arguments['verbose']

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        estimatorMetadata['inputBlockName'], arguments['window']))
outputPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}'.format(
        estimatorMetadata['name'], arguments['window']))

if arguments['verbose']:
    prf.print_memory_usage('before load dataBlock')
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
if arguments['verbose']:
    prf.print_memory_usage('after load dataBlock')

if arguments['verbose']:
    prf.print_memory_usage('before load firing rates')
alignedAsigsDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
if arguments['verbose']:
    prf.print_memory_usage('after load firing rates')

features = estimator.transform(alignedAsigsDF.to_numpy())
if arguments['verbose']:
    prf.print_memory_usage('after estimator.transform')
featureNames = [
    estimatorMetadata['name'] + '{:0>3}'.format(i)
    for i in range(features.shape[1])]

alignedFeaturesDF = pd.DataFrame(
    features, index=alignedAsigsDF.index, columns=featureNames)
alignedFeaturesDF.columns.name = 'feature'
del alignedAsigsDF

masterBlock = ns5.alignedAsigDFtoSpikeTrain(alignedFeaturesDF, dataBlock)
if arguments['lazy']:
    dataReader.file.close()
masterBlock = ns5.purgeNixAnn(masterBlock)
writer = ns5.NixIO(filename=outputPath + '.nix')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
