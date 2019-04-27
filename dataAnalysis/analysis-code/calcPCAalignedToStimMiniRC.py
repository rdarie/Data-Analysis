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
stimSpikes = []
for stP in stimSpikeProxys:
    try:
        st = stP.load()
        if len(st.times):
            stimSpikes.append(st)
    except Exception:
        traceback.print_exc()

stimSpikes = preproc.loadContainerArrayAnn(trainList=stimSpikes)
arrAnnKeys = stimSpikes[0].array_annotations.keys()

eventTimes = np.concatenate(
    [stSp.times for stSp in stimSpikes]
    ) * pq.s
eventArrAnn = {key: np.array([]) for key in arrAnnKeys}
for key in arrAnnKeys:
    eventArrAnn[key] = np.concatenate(
        [stSp.array_annotations[key] for stSp in stimSpikes])

nameAnn = np.array([])
for stSp in stimSpikes:
    nameAnn = np.concatenate([
        nameAnn,
        np.array([stSp.name for i in range(len(stSp.times))])]
    )
eventArrAnn['t'] = np.concatenate([
    stSp.times for stSp in stimSpikes
    ])
eventArrAnn['program'] = np.array([
    float(i[-1]) for i in nameAnn])
alignEventsDF = pd.DataFrame(
    eventArrAnn
    )
alignEventsDF.rename(columns={
    'amplitudes': 'amplitude',
    'rates': 'RateInHz',
    }, inplace=True)
alignEvents = preproc.eventDataFrameToEvents(
    alignEventsDF, idxT='t',
    annCol=None,
    eventName='seg0_alignTimes', tUnits=pq.s,
    makeList=False)

newEventBlock  = Block()
newEventBlock.name = eventBlock.annotations['neo_name']
newEventBlock.annotate(
    nix_name=eventBlock.annotations['neo_name'])

alignEvents.annotate(nix_name=alignEvents.name)
newSeg = Segment(name='seg0_')
newSeg.annotate(nix_name='seg0_')
newSeg.events.append(alignEvents)
alignEvents.segment = newSeg
newEventBlock.segments.append(newSeg)
#  source of signals
signalReader = neo.io.nixio_fr.NixIO(
    filename=analysisDataPath)
signalBlock = signalReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToTrigger = np.unique([
    i.name
    for i in signalBlock.filter(objects=AnalogSignalProxy)] +
    [i.name for i in signalBlock.filter(objects=AnalogSignal)])

windowSize = [-1 * pq.s, 1 * pq.s]

preproc.analogSignalsAlignedToEvents(
    eventBlock=newEventBlock, signalBlock=signalBlock,
    chansToTrigger=chansToTrigger, eventName=eventName,
    windowSize=windowSize, appendToExisting=False,
    checkReferences=False,
    fileName=ns5FileName + '_triggered',
    folderPath=os.path.join(
        remoteBasePath, 'processed', experimentName))

#  source of analogsignals
rasterReader = neo.io.nixio_fr.NixIO(
    filename=binnedSpikePath)
rasterBlock = rasterReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToTrigger = np.unique([
    i.name
    for i in rasterBlock.filter(objects=AnalogSignalProxy)])
#  chansToTrigger = ['elec96#0_fr', 'elec44#0_fr']

preproc.analogSignalsAlignedToEvents(
    eventBlock=newEventBlock, signalBlock=rasterBlock,
    chansToTrigger=chansToTrigger, eventName=eventName,
    windowSize=windowSize, appendToExisting=True,
    checkReferences=False,
    fileName=ns5FileName + '_triggered',
    folderPath=os.path.join(
        remoteBasePath, 'processed', experimentName))