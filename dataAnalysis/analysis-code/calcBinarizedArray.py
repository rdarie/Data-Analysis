import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import quantities as pq
import elephant as elph
from elephant.conversion import binarize

from importlib import reload
import pdb

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain)
import neo

#  load options
from currentExperiment import *

experimentDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_analyze.nix')
binnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_binarized.nix')
"""
experimentDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    'Trial001_analyze.nix')
binnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    'Trial001_binarized.nix'
"""
dataBlock = preproc.loadWithArrayAnn(experimentDataPath, fromRaw=False)

spikeMatBlock = Block()
spikeMatBlock.merge_annotations(dataBlock)

allSpikeTrains = [
        i for i in dataBlock.filter(objects=SpikeTrain) if '#' in i.name]
for st in allSpikeTrains:
    if '#' in st.name:
        chanList = spikeMatBlock.filter(
            objects=ChannelIndex, name=st.unit.name)
        if not len(chanList):
            chanIdx = ChannelIndex(name=st.unit.name, index=np.array([0]))
            #  print(chanIdx.name)
            spikeMatBlock.channel_indexes.append(chanIdx)

for segIdx, seg in enumerate(dataBlock.segments):
    newSeg = Segment(name='binned_{}'.format(segIdx))
    newSeg.merge_annotations(seg)
    spikeMatBlock.segments.append(newSeg)
    tStart = seg.analogsignals[0].t_start
    tStop = seg.analogsignals[0].t_stop
    
    # make dummy binary spike train, in case ths chan didn't fire
    segSpikeTrains = [
        i for i in seg.filter(objects=SpikeTrain) if '#' in i.name]
    samplingRate = 1e3 * pq.Hz
    dummyBin = binarize(
        segSpikeTrains[0],
        sampling_rate=samplingRate,
        t_start=tStart,
        t_stop=tStop) * 0

    for chanIdx in spikeMatBlock.channel_indexes:
        #  print(chanIdx.name)
        stList = seg.filter(
            objects=SpikeTrain,
            name='{}_{}'.format(chanIdx.name, segIdx)
            )
        if len(stList):
            st = stList[0]
            print(st.name)
            stBin = binarize(
                st,
                sampling_rate=samplingRate,
                t_start=tStart,
                t_stop=tStop)
        else:
            stBin = dummyBin

        asig = AnalogSignal(
            stBin * samplingRate,
            name='seg{}_{}'.format(segIdx, st.unit.name),
            sampling_rate=samplingRate,
            dtype=np.int,
            **st.annotations)
        asig.t_start = tStart

        chanIdx.analogsignals.append(asig)
        asig.channel_index = chanIdx
        spikeMatBlock.segments[segIdx].analogsignals.append(asig)

spikeMatBlock.create_relationship()
spikeMatBlock = preproc.purgeNixAnn(spikeMatBlock)

writer = neo.io.NixIO(filename=binnedSpikePath)
writer.write_block(spikeMatBlock)
writer.close()