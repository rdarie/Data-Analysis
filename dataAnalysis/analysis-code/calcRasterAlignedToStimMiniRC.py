import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
#  load options
from currentExperiment import *
import quantities as pq

#  source of events
eventReader = neo.io.nixio_fr.NixIO(
    filename=analysisDataPath)
eventBlock = eventReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in eventBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])

eventName = 'alignTimes'
stimSpikeProxys = [i for i in eventBlock.filter(objects=SpikeTrainProxy) if '#' not in i.name]
stimSpikes = [stP.load() for stP in stimSpikeProxys]
stimSpikes = [i for i in stimSpikes if len(i.times)]
stimSpikes = preproc.loadContainerArrayAnn(trainList=stimSpikes)
arrAnnKeys = stimSpikes[0].array_annotations.keys()

eventTimes = np.concatenate(
    [stSp.times for stSp in stimSpikes]
    ) * pq.s
eventArrAnn = {key: np.array([]) for key in arrAnnKeys}
for key in arrAnnKeys:
    eventArrAnn[key] = np.concatenate(
        [stSp.array_annotations[key] for stSp in stimSpikes])

alignEvents = Event(
    name=eventName,
    times=eventTimes,
    array_annotations=eventArrAnn
    )

newEventBlock = Block()
newEventBlock.segments.append(Segment())
newEventBlock.segments[0].events.append(alignEvents)
#  source of analogsignals
signalReader = neo.io.nixio_fr.NixIO(
    filename=binnedSpikePath)
signalBlock = signalReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToTrigger = np.unique([
    i.name
    for i in signalBlock.filter(objects=AnalogSignalProxy)])
#  chansToTrigger = ['elec96#0_fr', 'elec44#0_fr']

windowSize = [i * pq.s for i in rasterOpts['windowSize']]

preproc.analogSignalsAlignedToEvents(
    eventBlock=newEventBlock, signalBlock=signalBlock,
    chansToTrigger=chansToTrigger, eventName=eventName,
    windowSize=windowSize, appendToExisting=True,
    checkReferences=False,
    fileName=ns5FileName + '_triggered',
    folderPath=os.path.join(
        remoteBasePath, 'processed', experimentName))