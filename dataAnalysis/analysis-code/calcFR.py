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

masterSpikeMats, _ = preproc.loadSpikeMats(
    experimentBinnedSpikePath, rasterOpts,
    loadAll=True)

for segIdx, segSpikeMat in masterSpikeMats.items():
    spikeMatDF = segSpikeMat.reset_index().rename(
        columns={'bin': 't'})
    dataReader = neo.io.nixio_fr.NixIO(
        filename=experimentDataPath)
    dataSeg = dataReader.read_segment(
        block_index=0, seg_index=segIdx, lazy=True,
        signal_group_mode=None)
    dummyAsig = dataSeg.filter(
        objects=AnalogSignalProxy)[0].load(channel_indexes=[0])
    dataReader.file.close()
    samplingRate = dummyAsig.sampling_rate
    
    segSpikeMatInterp = hf.interpolateDF(
        spikeMatDF, pd.Series(dummyAsig.times.magnitude),
        kind='linear', fill_value=(0, 0),
        x='t')
    spikeMatBlockInterp = preproc.dataFrameToAnalogSignals(
        segSpikeMatInterp,
        idxT='t', useColNames=True,
        dataCol=segSpikeMatInterp.drop(columns='t').columns,
        samplingRate=samplingRate, nameSuffix='_fr')
    #  testAsigs = spikeMatBlockInterp.filter(objects=AnalogSignal, name='elec95#0_fr')
    #  plt.plot(testAsigs[0].magnitude[:1000]); plt.show()
    asigList = spikeMatBlockInterp.filter(objects=AnalogSignal)
    for asig in asigList:
        asig.annotate(binWidth=rasterOpts['binWidth'])
    preproc.addBlockToNIX(
        spikeMatBlockInterp, segIdx=0,
        writeSpikes=False, writeEvents=False,
        fileName=trialFilesStim['ins']['experimentName'] + '_analyze',
        folderPath=os.path.join(
            trialFilesStim['ins']['folderPath'],
            trialFilesStim['ins']['experimentName']),
        nixBlockIdx=0, nixSegIdx=segIdx,
        )
