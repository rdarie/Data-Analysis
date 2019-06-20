"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --exp=exp                              which experimental day to analyze
    --alignQuery=alignQuery                choose a subset of the data?
    --window=window                        process with short window? [default: short]
    --estimator=estimator                  estimator filename
    --chanQuery=chanQuery                  how to restrict channels?
"""
import os
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.profiling as prf
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
import neo
#  import gc
import joblib as jb
import pickle

print('about to import exp opts, memory usage: {:.2f} MB'.format(
    prf.memory_usage_psutil()))

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

estimatorPath = os.path.join(
    scratchFolder,
    arguments['--estimator'] + '.joblib')

with open(
    os.path.join(
        scratchFolder,
        arguments['--estimator'] + '_meta.pickle'),
        'rb') as f:
    estimatorMetadata = pickle.load(f)

print('about to load data, memory usage: {:.2f} MB'.format(
    prf.memory_usage_psutil()))

if arguments['--processAll']:
    triggeredPath = os.path.join(
        scratchFolder,
        experimentName + '_trig_fr_sqrt_{}.nix'.format(
            arguments['--window']))
    outputPath = os.path.join(
        scratchFolder,
        experimentName + '_trig_{}_{}'.format(
            estimatorMetadata['name'], arguments['--window']))
else:
    triggeredPath = os.path.join(
        scratchFolder,
        ns5FileName + '_trig_fr_sqrt_{}.nix'.format(
            arguments['--window']))
    outputPath = os.path.join(
        scratchFolder,
        ns5FileName + '_trig_{}_{}'.format(
            estimatorMetadata['name'], arguments['--window']))

dataReader = neo.io.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

estimator = jb.load(
    os.path.join(scratchFolder, estimatorMetadata['path']))

if arguments['--alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['--alignQuery']
    ])

print('about to extract waveforms, memory usage: {:.2f} MB'.format(
    prf.memory_usage_psutil()))

unitNames = estimatorMetadata['inputFeatures']
masterSpikeMat = ns5.alignedAsigsToDF(
    dataBlock, unitNames, dataQuery,
    **estimatorMetadata['alignedAsigsKWargs'])
print(
    'just loaded data, memory usage: {:.2f} MB'
    .format(prf.memory_usage_psutil()))

features = estimator.transform(masterSpikeMat.values)
featureNames = [
    estimatorMetadata['name'] + '{}'.format(i)
    for i in range(features.shape[1])]
featuresDF = pd.DataFrame(
    features, index=masterSpikeMat.index, columns=featureNames)
featuresDF.columns.name = 'feature'
allWaveforms = featuresDF.stack().unstack('bin')

masterBlock = ns5.alignedAsigDFtoSpikeTrain(allWaveforms, dataBlock)
dataReader.file.close()
#  print('memory usage: {:.2f} MB'.format(prf.memory_usage_psutil()))
masterBlock = ns5.purgeNixAnn(masterBlock)
writer = neo.io.NixIO(filename=outputPath + '.nix')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
