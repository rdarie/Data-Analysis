"""  10b: Calculate Firing Rates
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
"""

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
#  from currentExperiment import *
from joblib import dump, load
from importlib import reload
import quantities as pq

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']),
    arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['--processAll']:
    masterSpikeMats, _ = preproc.loadSpikeMats(
        experimentBinnedSpikePath, rasterOpts,
        loadAll=True, checkReferences=False)
    dataReader = neo.io.nixio_fr.NixIO(
        filename=experimentDataPath)
else:
    masterSpikeMats, _ = preproc.loadSpikeMats(
        binnedSpikePath, rasterOpts,
        loadAll=True, checkReferences=False)
    dataReader = neo.io.nixio_fr.NixIO(
        filename=analysisDataPath)

dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']

for segIdx, segSpikeMat in masterSpikeMats.items():
    print('Calculating FR for segment {}'.format(segIdx))
    spikeMatDF = segSpikeMat.reset_index().rename(
        columns={'bin': 't'})

    dataSeg = dataBlock.segments[segIdx]
    dummyAsig = dataSeg.filter(
        objects=AnalogSignalProxy)[0].load(channel_indexes=[0])
    samplingRate = dummyAsig.sampling_rate
    newT = dummyAsig.times.magnitude
    spikeMatDF['t'] = spikeMatDF['t'] + newT[0]
    
    segSpikeMatInterp = hf.interpolateDF(
        spikeMatDF, pd.Series(newT),
        kind='linear', fill_value=(0, 0),
        x='t')
    spikeMatBlockInterp = preproc.dataFrameToAnalogSignals(
        segSpikeMatInterp,
        idxT='t', useColNames=True,
        dataCol=segSpikeMatInterp.drop(columns='t').columns,
        samplingRate=samplingRate)
    spikeMatBlockInterp.name = dataBlock.annotations['neo_name']
    spikeMatBlockInterp.annotate(
        nix_name=dataBlock.annotations['neo_name'])
    spikeMatBlockInterp.segments[0].name = dataSeg.annotations['neo_name']
    spikeMatBlockInterp.segments[0].annotate(
        nix_name=dataSeg.annotations['neo_name'])
    
    asigList = spikeMatBlockInterp.filter(objects=AnalogSignal)
    
    for asig in asigList:
        asig.annotate(binWidth=rasterOpts['binWidth'])
        if '_raster' in asig.name:
            asig.name = asig.name.replace('_raster', '_fr')
        asig.name = 'seg{}_{}'.format(segIdx, asig.name)
        asig.annotate(nix_name=asig.name)
    chanIdxList = spikeMatBlockInterp.filter(objects=ChannelIndex)
    for chanIdx in chanIdxList:
        if '_raster' in chanIdx.name:
            chanIdx.name = chanIdx.name.replace('_raster', '_fr')
        chanIdx.annotate(nix_name=chanIdx.name)

    masterBlock.merge(spikeMatBlockInterp)

dataReader.file.close()
allSegs = list(range(len(masterBlock.segments)))

if arguments['--processAll']:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeSpikes=False, writeEvents=False,
        fileName=experimentName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
else:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeSpikes=False, writeEvents=False,
        fileName=ns5FileName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
