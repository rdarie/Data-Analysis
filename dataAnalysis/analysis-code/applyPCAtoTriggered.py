"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --window=window                        process with short window? [default: shortWindow]
    --estimator=estimator                  filename for resulting estimator
"""
import os
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.preproc.ns5 as preproc
import seaborn as sns
import numpy as np
import quantities as pq
import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import neo
import gc
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
import joblib as jb
import pickle
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

print('about to load data, memory usage: {}'.format(
    hf.memory_usage_psutil()))

if arguments['--processAll']:
    triggeredPath = os.path.join(
        scratchFolder,
        experimentName + '_triggered_{}.nix'.format(
            arguments['--window']))
else:
    triggeredPath = os.path.join(
        scratchFolder,
        ns5FileName + '_triggered_{}.nix'.format(
            arguments['--window']))

dataReader = neo.io.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

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
    os.path.join(
        scratchFolder, estimatorMetadata['path']))

if arguments['--alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['--alignQuery']
    ])

unitNames = estimatorMetadata['inputFeatures']
asigWide = ns5.alignedAsigsToDF(
    dataBlock, unitNames, dataQuery,
    **estimatorMetadata['alignedAsigsKWargs'])
masterSpikeMat = asigWide.stack().unstack('feature')[unitNames]

#  free up memory
del asigWide
gc.collect()
print('just freed up memory, memory usage: {}'.format(
    hf.memory_usage_psutil()))

features = estimator.transform(masterSpikeMat.values)
featureNames = [
    estimatorMetadata['name'] + '{}'.format(i)
    for i in range(features.shape[1])]
featuresDF = pd.DataFrame(
    features, index=masterSpikeMat.index, columns=featureNames)
featuresDF.columns.name = 'feature'
allWaveforms = featuresDF.stack().unstack('bin')

masterBlock = ns5.alignedAsigDFtoSpikeTrain(allWaveforms, dataBlock)
fileName = os.path.basename(triggeredPath).replace('.nix', '')
allSegs = list(range(len(masterBlock.segments)))
ns5.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeSpikes=True,
    fileName=fileName,
    folderPath=scratchFolder,
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs)
