"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --estimator=estimator                  filename for resulting estimator
"""
import os
import dataAnalysis.plotting.aligned_signal_plots as asp
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

#  source of signal
if arguments['--processAll']:
    filePath = experimentDataPath
else:
    filePath = analysisDataPath

dataReader = neo.io.nixio_fr.NixIO(
    filename=filePath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

estimatorPath = os.path.join(
    scratchFolder,
    arguments['--estimator'] + '.joblib')
with open(os.path.join(scratchFolder, arguments['--estimator'] + '_meta.pickle'), 'rb') as f:
    estimatorMetadata = pickle.load(f)
estimator = jb.load(os.path.join(scratchFolder, estimatorMetadata['path']))

unitNames = []
for uName in estimatorMetadata['inputFeatures']:
    if uName[-2:] == '#0':
        unitNames.append(uName[:-2])
    else:
        unitNames.append(uName)
unitNames = sorted(unitNames)

interpolatedSpikeMats = {}
blockIdx = 0
checkReferences = False
for segIdx, dataSeg in enumerate(dataBlock.segments):
    asigProxysList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if asigP.name in unitNames]
    if checkReferences:
        for asigP in asigProxysList:
            if checkReferences:
                da = asigP._rawio.da_list['blocks'][blockIdx]['segments'][segIdx]['data']
                print('segIdx {}, asigP.name {}'.format(
                    segIdx, asigP.name))
                print('asigP._global_channel_indexes = {}'.format(
                    asigP._global_channel_indexes))
                print('asigP references {}'.format(
                    da[asigP._global_channel_indexes[0]]))
                try:
                    assert asigP.name in da[asigP._global_channel_indexes[0]].name
                except Exception:
                    traceback.print_exc()
    asigsList = [
        asigP.load()
        for asigP in asigProxysList]
    asigsDF = preproc.analogSignalsToDataFrame(asigsList)
    asigsDF.index = asigsDF['t']
    asigsDF.index.name = 't'
    interpolatedSpikeMats.update(
        {segIdx: asigsDF.drop(columns='t')})

masterSpikeMat = pd.concat(
    interpolatedSpikeMats, names=['segment', 't'], sort=True)
masterSpikeMat.columns.name = 'feature'

#  ensure column order
features = estimator.transform(masterSpikeMat[unitNames].values)
featureNames = [
    estimatorMetadata['name'] + '{}'.format(i)
    for i in range(features.shape[1])]
featuresDF = pd.DataFrame(
    features, index=masterSpikeMat.index, columns=featureNames)

masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']

for segIdx, group in featuresDF.groupby('segment'):
    print('Calculating trajectories for segment {}'.format(segIdx))
    segFeaturesDF = group.reset_index().drop(columns='segment')
    
    dataSeg = dataBlock.segments[segIdx]
    dummyAsig = dataSeg.filter(
        objects=AnalogSignalProxy)[0].load(channel_indexes=[0])
    samplingRate = dummyAsig.sampling_rate
    
    featureBlock = preproc.dataFrameToAnalogSignals(
        segFeaturesDF,
        idxT='t', useColNames=True,
        dataCol=segFeaturesDF.drop(columns='t').columns,
        samplingRate=samplingRate)
    featureBlock.name = dataBlock.annotations['neo_name']
    featureBlock.annotate(
        nix_name=dataBlock.annotations['neo_name'])
    featureBlock.segments[0].name = dataSeg.annotations['neo_name']
    featureBlock.segments[0].annotate(
        nix_name=dataSeg.annotations['neo_name'])

    asigList = featureBlock.filter(objects=AnalogSignal)
    for asig in asigList:
        asig.name = 'seg{}_{}'.format(segIdx, asig.name)
        asig.annotate(nix_name=asig.name)
    chanIdxList = featureBlock.filter(objects=ChannelIndex)
    for chanIdx in chanIdxList:
        chanIdx.annotate(nix_name=chanIdx.name)
    masterBlock.merge(featureBlock)

dataReader.file.close()
allSegs = list(range(len(masterBlock.segments)))
fileName = os.path.basename(filePath).replace('.nix', '')
preproc.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeSpikes=False, writeEvents=False,
    fileName=fileName,
    folderPath=scratchFolder,
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )
