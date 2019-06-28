"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --exp=exp                              which experimental day to analyze
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --window=window                        process with short window? [default: short]
    --estimator=estimator                  estimator filename
    --chanQuery=chanQuery                  how to restrict channels?
"""
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.profiling as prf
import os
#  import seaborn as sns
#  import numpy as np
#  import quantities as pq
import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  from neo.io.proxyobjects import (
#      AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import joblib as jb
import pickle
#  import gc

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

verbose = True

estimatorPath = os.path.join(
    scratchFolder,
    arguments['--estimator'] + '.joblib')

with open(
    os.path.join(
        scratchFolder,
        arguments['--estimator'] + '_meta.pickle'),
        'rb') as f:
    estimatorMetadata = pickle.load(f)

estimator = jb.load(
    os.path.join(scratchFolder, estimatorMetadata['path']))

if arguments['--processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        estimatorMetadata['inputBlockName'], arguments['--window']))
outputPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}'.format(
        estimatorMetadata['name'], arguments['--window']))

if verbose:
    prf.print_memory_usage('before load data')
dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

if arguments['--alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['--alignQuery']
    ])

if verbose:
    prf.print_memory_usage('before load firing rates')
unitNames = estimatorMetadata['inputFeatures']
alignedAsigsDF = ns5.alignedAsigsToDF(
    dataBlock, unitNames, dataQuery,
    **estimatorMetadata['alignedAsigsKWargs'], verbose=True)
if verbose:
    prf.print_memory_usage('after load firing rates')

features = estimator.transform(alignedAsigsDF.to_numpy())
if verbose:
    prf.print_memory_usage('after estimator.transform')
featureNames = [
    estimatorMetadata['name'] + '{:0>3}'.format(i)
    for i in range(features.shape[1])]
#
alignedFeaturesDF = pd.DataFrame(
    features, index=alignedAsigsDF.index, columns=featureNames)
alignedFeaturesDF.columns.name = 'feature'

del alignedAsigsDF
masterBlock = ns5.alignedAsigDFtoSpikeTrain(alignedFeaturesDF, dataBlock)
dataReader.file.close()
#  print('memory usage: {:.1f} MB'.format(prf.memory_usage_psutil()))
masterBlock = ns5.purgeNixAnn(masterBlock)
writer = ns5.NixIO(filename=outputPath + '.nix')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
