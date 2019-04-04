import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
from copy import copy
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
#  load options
from currentExperiment import *
from joblib import dump, load
from importlib import reload
import quantities as pq

dataReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

interpolatedSpikeMats = {}
blockIdx = 0
checkReferences = True
for segIdx, dataSeg in enumerate(dataBlock.segments):
    asigProxysList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if '_fr' in asigP.name]
    if checkReferences:
        for asigP in asigProxysList:
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
    for asig in asigsList:
        asig.name = asig.name.split('_fr')[0]
    asigsDF = preproc.analogSignalsToDataFrame(asigsList)
    asigsDF.index = asigsDF['t']
    asigsDF.index.name = 'bin'
    interpolatedSpikeMats.update(
        {segIdx: asigsDF.drop(columns='t')})

masterSpikeMat = pd.concat(
    interpolatedSpikeMats, names=['segment', 'bin'])
masterSpikeMat.columns.name = 'unit'

nComp = masterSpikeMat.columns.shape[0]
pca = PCA(n_components=nComp)
compNames = ['PC{}'.format(i+1) for i in range(nComp)]
estimator = Pipeline([('dimred', pca)])

features = estimator.fit_transform(masterSpikeMat.values)
featuresDF = pd.DataFrame(
    features, index=masterSpikeMat.index, columns=compNames)

masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']

for segIdx, group in featuresDF.groupby('segment'):
    print('Loading FRs for segment {}'.format(segIdx))
    segFeaturesDF = group.xs(segIdx, axis='index')
    segFeaturesDF.reset_index(inplace=True)

    dataSeg = dataBlock.segments[segIdx]
    dummyAsig = dataSeg.filter(
        objects=AnalogSignalProxy)[0].load(channel_indexes=[0])
    samplingRate = dummyAsig.sampling_rate

    pcBlockInterp = preproc.dataFrameToAnalogSignals(
        segFeaturesDF,
        idxT='bin', useColNames=True,
        dataCol=segFeaturesDF.drop(columns='bin').columns,
        samplingRate=samplingRate)
    pcBlockInterp.name = dataBlock.annotations['neo_name']
    pcBlockInterp.annotate(
        nix_name=dataBlock.annotations['neo_name'])
    pcBlockInterp.segments[0].name = dataSeg.annotations['neo_name']
    pcBlockInterp.segments[0].annotate(
        nix_name=dataSeg.annotations['neo_name'])

    asigList = pcBlockInterp.filter(objects=AnalogSignal)
    for asig in asigList:
        asig.annotate(binWidth=rasterOpts['binWidth'])
        asig.name = 'seg{}_{}'.format(segIdx, asig.name)
        asig.annotate(nix_name=asig.name)
    chanIdxList = pcBlockInterp.filter(objects=ChannelIndex)
    for chanIdx in chanIdxList:
        chanIdx.annotate(nix_name=chanIdx.name)

    masterBlock.merge(pcBlockInterp)
dataReader.file.close()

allSegs = list(range(len(masterBlock.segments)))
preproc.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeSpikes=False, writeEvents=False,
    fileName=trialFilesStim['ins']['experimentName'] + '_analyze',
    folderPath=os.path.join(
        trialFilesStim['ins']['folderPath'],
        trialFilesStim['ins']['experimentName']),
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )

dump(estimator, estimatorPath)