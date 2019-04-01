import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
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
dataReader.parse_header()
#  check units
#  [i.sampling_rate for i in dataBlock.filter(objects=AnalogSignalProxy)]
#  asig = dataBlock.filter(objects=AnalogSignal, name='elec95#0_fr')
#  plt.plot(asig[0].magnitude[:1000]); plt.show()
interpolatedSpikeMats = {}
blockIdx = 0
for segIdx in range(dataReader.header['nb_segment'][blockIdx]):
    dataSeg = dataReader.read_segment(
        block_index=0, seg_index=segIdx, lazy=True,
        signal_group_mode='split-all')
    asigProxysList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if '_fr' in asigP.name]
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
dataReader.file.close()

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

for segIdx, group in featuresDF.groupby('segment'):
    segFeaturesDF = group.xs(segIdx, axis='index')
    segFeaturesDF.reset_index(inplace=True)
    pcBlockInterp = preproc.dataFrameToAnalogSignals(
        segFeaturesDF,
        idxT='bin', useColNames=True,
        dataCol=segFeaturesDF.drop(columns='bin').columns,
        samplingRate=asig.sampling_rate)
    for asig in pcBlockInterp.filter(objects=AnalogSignal):
        asig.annotate(binWidth=rasterOpts['binWidth'])
        #  asig.name = asig.name + '_PC'
    preproc.addBlockToNIX(
        pcBlockInterp, segIdx=0,
        writeSpikes=False, writeEvents=False,
        fileName=trialFilesStim['ins']['experimentName'] + '_analyze',
        folderPath=os.path.join(
            trialFilesStim['ins']['folderPath'],
            trialFilesStim['ins']['experimentName']),
        nixBlockIdx=0, nixSegIdx=segIdx,
        )

dump(estimator, estimatorPath)