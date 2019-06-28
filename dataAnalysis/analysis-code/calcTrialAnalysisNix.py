"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcTrialAnalysisNix.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions as hf
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb
from importlib import reload


#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
samplingRate = 1 / rasterOpts['binInterval'] * pq.Hz

nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
nspBlock = preproc.readBlockFixNames(nspReader, block_index=0)
#  nspBlockRaw = nspReader.read_block(
#      block_index=0, lazy=True,
#      signal_group_mode='split-all')
dataBlock = hf.extractSignalsFromBlock(
    nspBlock, keepSpikes=True)
dataBlock = hf.loadBlockProxyObjects(dataBlock)
'''
for ev in dataBlock.segments[0].filter(objects=Event):
    if len(ev.times):
        print('{}.t_start={}'.format(ev.name, ev.times[0]))
    else:
        print('{} has no times'.format(ev.name))
for st in dataBlock.segments[0].filter(objects=SpikeTrain):
    if len(st.times):
        print('{}.t_start={}'.format(st.name, st.t_start))
    else:
        print('{} has no times'.format(st.name))
for st in nspBlock.segments[0].filter(objects=SpikeTrainProxy):
    print('{}.t_start={}'.format(st.name, st.t_start))
    
for asig in dataBlock.segments[0].filter(objects=AnalogSignal):
        print('{}.t_start={}'.format(asig.name, asig.t_start))
'''
allSpikeTrains = dataBlock.filter(objects=SpikeTrain)

for segIdx, dataSeg in enumerate(dataBlock.segments):
    tStart = nspReader.get_signal_t_start(
        block_index=0, seg_index=segIdx) * pq.s
    fs = nspReader.get_signal_sampling_rate(
        channel_indexes=[0])
    sigSize = nspReader.get_signal_size(
        block_index=0, seg_index=segIdx
        )
    tStop = (sigSize / fs) * pq.s + tStart
    #
    spikeList = dataSeg.filter(objects=SpikeTrain)
    spikeList = preproc.loadContainerArrayAnn(trainList=spikeList)
    '''
    for st in dataSeg.filter(objects=SpikeTrain):
        if len(st.times):
            st.t_start = min(tStart, st.times[0] * 0.999)
            st.t_stop = min(tStop, st.times[-1] * 1.001)
            validMask = st < st.t_stop
            if ~validMask.all():
                print('Deleted some spikes')
                st = st[validMask]
                for key in st.array_annotations.keys():
                    st.array_annotations[key] = st.array_annotations[key][validMask]
                if 'arrayAnnNames' in st.annotations.keys():
                    for key in st.annotations['arrayAnnNames']:
                        st.annotations[key] = np.array(st.annotations[key])[validMask]
        else:
            st.t_start = tStart
            st.t_stop = tStop
        if 'arrayAnnNames' in st.annotations.keys():
            for key in st.annotations['arrayAnnNames']:
                #  fromRaw, the ann come back as tuple, need to recast
                st.array_annotations.update(
                   {key: np.array(st.annotations[key])})
        st.sampling_rate = samplingRate
        if st.waveforms is None:
            st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
    '''


#  tests...
#  [i.unit.channel_index.name for i in insBlockJustSpikes.filter(objects=SpikeTrain)]
#  [i.channel_index.name for i in nspBlock.filter(objects=AnalogSignalProxy)]
#  [i.name for i in nspBlock.filter(objects=AnalogSignalProxy)]
#  [i.channel_index.name for i in dataBlock.filter(objects=AnalogSignal)]
#  [i.channel_index.name for i in dataBlock.filter(objects=AnalogSignal)]

#  merge events
evList = []
for key in ['property', 'value']:
    #  key = 'property'
    insProp = dataBlock.filter(
        objects=Event,
        name='seg0_ins_' + key
        )[0]
    rigProp = dataBlock.filter(
        objects=Event,
        name='seg0_rig_' + key
        )
    if len(rigProp):
        rigProp = rigProp[0]
        allProp = insProp.merge(rigProp)
        allProp.name = 'seg0_' + key

        evSortIdx = np.argsort(allProp.times, kind='mergesort')
        allProp = allProp[evSortIdx]
        evList.append(allProp)
    else:
        #  mini RC's don't have rig_events
        allProp = insProp
        allProp.name = 'seg0_' + key
        evList.append(insProp)

#  make concatenated event, for viewing
concatLabels = np.array([
    (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
        elphpdb._convert_value_safe(evList[1].labels[i])) for
    i in range(len(evList[0]))
    ])
concatEvent = Event(
    name='seg0_' + 'concatenated_updates',
    times=allProp.times,
    labels=concatLabels
    )
concatEvent.merge_annotations(allProp)
evList.append(concatEvent)
dataBlock.segments[0].events = evList

dataBlock = preproc.purgeNixAnn(dataBlock)
writer = neo.io.NixIO(filename=analysisDataPath)
writer.write_block(dataBlock, use_obj_names=True)
writer.close()

spikeMatBlock = preproc.calcBinarizedArray(
    dataBlock, samplingRate,
    binnedSpikePath, saveToFile=True)

#  save ins time series
if miniRCTrial:
    tdChanNames = [
        i.name for i in nspBlock.filter(objects=AnalogSignalProxy)
        if 'ins_td' in i.name]
else:
    tdChanNames = [
        i.name for i in nspBlock.filter(objects=AnalogSignalProxy)
        if 'ins_td' in i.name] + ['seg0_position', 'seg0_velocityCat']

tdBlock = hf.extractSignalsFromBlock(
    nspBlock, keepSpikes=False, keepSignals=tdChanNames)
tdBlock = hf.loadBlockProxyObjects(tdBlock)

ins_events = [
    i for i in tdBlock.filter(objects=Event)
    if 'ins_' in i.name]
tdDF = preproc.analogSignalsToDataFrame(
    tdBlock.filter(objects=AnalogSignal))
newT = pd.Series(
    spikeMatBlock.filter(objects=AnalogSignal)[0].times.magnitude)
tdInterp = hf.interpolateDF(
    tdDF, newT,
    kind='linear', fill_value=(0, 0),
    x='t', columns=tdChanNames)

expandCols = [
        'RateInHz', 'therapyStatus',
        'activeGroup', 'program', 'trialSegment']
deriveCols = ['amplitudeRound', 'amplitude']
progAmpNames = rcsa_helpers.progAmpNames

stimStSer = preproc.eventsToDataFrame(
    ins_events, idxT='t'
    )
stimStatus = hf.stimStatusSerialtoLong(
    stimStSer, idxT='t',  namePrefix='seg0_ins_',
    expandCols=expandCols,
    deriveCols=deriveCols, progAmpNames=progAmpNames)
columnsToBeAdded = ['amplitude', 'program']
infoFromStimStatus = hf.interpolateDF(
    stimStatus, tdInterp['t'],
    x='t', columns=columnsToBeAdded, kind='previous')
infoFromStimStatus.rename(
    columns={
        'amplitude': 'seg0_amplitude',
        'program': 'seg0_program'
    }, inplace=True)
tdInterp = pd.concat((
    tdInterp,
    infoFromStimStatus.drop(columns='t')),
    axis=1)
tdBlockInterp = preproc.dataFrameToAnalogSignals(
    tdInterp,
    idxT='t', useColNames=True,
    dataCol=tdInterp.drop(columns='t').columns,
    samplingRate=samplingRate)
for chanIdx in tdBlockInterp.channel_indexes:
    chanIdx.name = preproc.childBaseName(chanIdx.name, 'seg0')
preproc.addBlockToNIX(
    tdBlockInterp, neoSegIdx=[0],
    writeSpikes=False, writeEvents=False,
    purgeNixNames=False,
    fileName=ns5FileName + '_analyze',
    folderPath=scratchFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )
