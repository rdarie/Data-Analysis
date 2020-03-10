"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --processAll                           process entire experimental day? [default: False]
    --estimator=estimator                  filename for resulting estimator
    --verbose                              print diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
"""
import os
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import seaborn as sns
#  import numpy as np
#  import quantities as pq
import pandas as pd
import pdb
from neo import (
    Block, ChannelIndex, AnalogSignal,
    #  Segment, Event, SpikeTrain, Unit
    )
from neo.io.proxyobjects import (
    AnalogSignalProxy,
    #  SpikeTrainProxy, EventProxy
    )
import neo
import traceback
import joblib as jb
import pickle
from currentExperiment import parseAnalysisOptions
from docopt import docopt
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
#  source of signal
if arguments['processAll']:
    filePath = experimentDataPath.format(arguments['analysisName'])
else:
    filePath = analysisDataPath.format(arguments['analysisName'])

dataReader, dataBlock = ns5.blockFromPath(
    filePath, lazy=arguments['lazy'])
estimatorPath = os.path.join(
    scratchFolder,
    arguments['estimator'] + '.joblib')
with open(os.path.join(scratchFolder, arguments['estimator'] + '_meta.pickle'), 'rb') as f:
    estimatorMetadata = pickle.load(f)
estimator = jb.load(os.path.join(scratchFolder, estimatorMetadata['path']))

unitNames = []
for uName in estimatorMetadata['alignedAsigsKWargs']['unitNames']:
    if uName[-2:] == '#0':
        unitNames.append(uName[:-2])
    else:
        unitNames.append(uName)

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
    asigsDF = ns5.analogSignalsToDataFrame(asigsList)
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
    print('Saving trajectories for segment {}'.format(segIdx))
    segFeaturesDF = group.reset_index().drop(columns='segment')
    
    dataSeg = dataBlock.segments[segIdx]
    dummyAsig = dataSeg.filter(
        objects=AnalogSignalProxy)[0].load(channel_indexes=[0])
    samplingRate = dummyAsig.sampling_rate
    
    featureBlock = ns5.dataFrameToAnalogSignals(
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

if arguments['lazy']:
    dataReader.file.close()
allSegs = list(range(len(masterBlock.segments)))
fileName = os.path.basename(filePath).replace('.nix', '')
ns5.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeSpikes=False, writeEvents=False,
    fileName=fileName,
    folderPath=analysisSubFolder,
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )
