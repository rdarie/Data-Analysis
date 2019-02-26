# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""
import neo
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import quantities as pq
from quantities import mV, kHz, s, uV
import matplotlib, math, pdb
from copy import copy
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.helperFunctions.motor_encoder as mea
from brPY.brpylib import NsxFile, NevFile, brpylib_ver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import pickle
from copy import *
import traceback
import h5py
from scipy import signal
import rcsanalysis.packet_func as rcsa_helpers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def spikeDictToSpikeTrains(
        spikes, block=None, seg=None,
        probeName='insTD', t_stop=None,
        waveformUnits=pq.uV,
        sampling_rate=3e4 * pq.Hz):

    if block is None:
        assert seg is None
        block = Block()
        seg = Segment(name=probeName + ' segment')
        block.segments.append(seg)

    if t_stop is None:
        t_stop = hf.getLastSpikeTime(spikes) + 1

    for idx, chanName in enumerate(spikes['ChannelID']):
        #  unique units on this channel
        unitsOnThisChan = pd.unique(spikes['Classification'][idx])
        nixChanName = probeName + '{}'.format(chanName)
        chanIdx = ChannelIndex(
            name=nixChanName,
            index=np.array([idx]),
            channel_names=np.array([nixChanName]))
        block.channel_indexes.append(chanIdx)
        
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            unitMask = spikes['Classification'][idx] == unitName
            # this unit's spike timestamps
            theseTimes = spikes['TimeStamps'][idx][unitMask]
            # this unit's waveforms
            if len(spikes['Waveforms'][idx].shape) == 3:
                theseWaveforms = spikes['Waveforms'][idx][unitMask, :, :]
                theseWaveforms = np.swapaxes(theseWaveforms, 1, 2)
            elif len(spikes['Waveforms'][idx].shape) == 2:
                theseWaveforms = (
                    spikes['Waveforms'][idx][unitMask, np.newaxis, :])
            else:
                raise(Exception('spikes[Waveforms] has bad shape'))

            unitName = '{}#{}'.format(nixChanName, unitIdx)
            unit = Unit(name=unitName)
            unit.channel_index = chanIdx
            chanIdx.units.append(unit)

            train = SpikeTrain(
                times=theseTimes, t_stop=t_stop, units='sec',
                name=unitName, sampling_rate=sampling_rate,
                waveforms=theseWaveforms*waveformUnits,
                left_sweep=0, dtype=np.float32)
            unit.spiketrains.append(train)
            seg.spiketrains.append(train)

            unit.create_relationship()
        chanIdx.create_relationship()
    seg.create_relationship()
    block.create_relationship()
    return block


def spikeTrainsToSpikeDict(spiketrains):
    nCh = len(spiketrains)
    spikes = {
        'ChannelID': [i for i in range(nCh)],
        'Classification': [np.array([]) for i in range(nCh)],
        'NEUEVWAV_HeaderIndices': [None for i in range(nCh)],
        'TimeStamps': [np.array([]) for i in range(nCh)],
        'Units': 'uV',
        'Waveforms': [np.array([]) for i in range(nCh)],
        'basic_headers': {'TimeStampResolution': 3e4},
        'extended_headers': []
        }
    for idx, st in enumerate(spiketrains):
        #  pdb.set_trace()
        spikes['ChannelID'][idx] = st.name
        if len(spikes['TimeStamps'][idx]):
            spikes['TimeStamps'][idx] = np.stack((
                spikes['TimeStamps'][idx],
                st.times.magnitude), axis=-1)
        else:
            spikes['TimeStamps'][idx] = st.times.magnitude
        
        theseWaveforms = np.swapaxes(st.waveforms, 1, 2)
        if len(spikes['Waveforms'][idx]):
            spikes['Waveforms'][idx] = np.stack((
                spikes['Waveforms'][idx],
                theseWaveforms), axis=-1)
        else:
            spikes['Waveforms'][idx] = theseWaveforms
        
        classVals = st.times.magnitude ** 0 * idx
        if len(spikes['Classification'][idx]):
            spikes['Classification'][idx] = np.stack((
                spikes['Classification'][idx],
                classVals), axis=-1)
        else:
            spikes['Classification'][idx] = classVals
    return spikes

def analogSignalsToDataFrame(analogsignals, idxT='t'):
    asigList = []
    for asig in analogsignals:
        if asig.shape[1] == 1:
            colNames = [asig.name]
        else:
            colNames = [
                asig.name +
                '_{}'.format(i) for i in
                asig.channel_index.channel_ids
                ]
        asigList.append(
            pd.DataFrame(
                asig.magnitude, columns=colNames,
                index=range(asig.shape[0])))
    asigList.append(
        pd.DataFrame(
            asig.times.magnitude, columns=[idxT],
            index=range(asig.shape[0])))
    return pd.concat(asigList, axis=1)


def dataFrameToAnalogSignals(
        df,
        block=None, seg=None,
        idxT='NSPTime',
        probeName='insTD', samplingRate=500*pq.Hz,
        timeUnits=pq.s, measureUnits=pq.mV,
        dataCol=['channel_0', 'channel_1'],
        useColNames=False, forceColNames=None, namePrefix=''):

    if block is None:
        assert seg is None
        block = Block()
        seg = Segment(name=probeName + ' segment')
        block.segments.append(seg)

    for idx, colName in enumerate(dataCol):
        if forceColNames is not None:
            chanName = forceColNames[idx]
        elif useColNames:
            chanName = namePrefix + colName
        else:
            chanName = namePrefix + (probeName.lower() + '{}'.format(idx))

        arrayAnn = {
            'channel_names': np.array([chanName]),
            'channel_ids': np.array([idx], dtype=np.int)
            }
        chanIdx = ChannelIndex(
            name=chanName,
            index=np.array([idx]),
            channel_names=np.array([chanName]))
        block.channel_indexes.append(chanIdx)
        asig = AnalogSignal(
            df[colName].values*measureUnits,
            name=chanName,
            sampling_rate=samplingRate,
            dtype=np.float32,
            array_annotations=arrayAnn)
        asig.t_start = df[idxT].iloc[0]*timeUnits
        asig.channel_index = chanIdx
        # assign ownership to containers
        chanIdx.analogsignals.append(asig)
        seg.analogsignals.append(asig)
        chanIdx.create_relationship()
    # assign parent to children
    # block.create_relationship()
    # seg.create_relationship()
    return block


def eventDataFrameToEvents(
        eventDF, idxT=None,
        annCol=None,
        eventName='', tUnits=pq.s
        ):
    eventList = []
    for colName in annCol:
        event = Event(
            name=eventName + colName,
            times=eventDF[idxT].values * tUnits,
            labels=eventDF[colName].values
            )
        #  pdb.set_trace()
        originalDType = type(eventDF[colName].values[0]).__name__
        event.annotate(originalDType=originalDType)
        eventList.append(event)
    return eventList

def eventsToDataFrame(
        events, idxT='t'
        ):
    eventDict = {}
    
    for event in events:
        if len(event.times):
            #  print(event.name)
            values = event.array_annotations['labels']
            if isinstance(values[0], bytes):
                #  event came from hdf, need to recover dtype
                originalDType = eval('np.' + event.annotations['originalDType'])
                values = np.array(values, dtype=originalDType)
            #  print(values.dtype)
            eventDict.update({
                event.name: pd.Series(values)})
    eventDict.update({idxT: pd.Series(event.times.magnitude)})
    return pd.concat(eventDict, axis=1)

def unpackAnalysisBlock(block, interpolateToTimeSeries=False):
    tdAsig = block.filter(
        objects=AnalogSignal
        )
    tdDF = analogSignalsToDataFrame(tdAsig)
    
    insProp = block.filter(
        objects=Event,
        name='ins_property'
        )[0]
    insVal = block.filter(
        objects=Event,
        name='ins_value'
        )[0]

    insEvDF = eventsToDataFrame(
        [insProp, insVal], idxT='t'
        )
    rigEvDF = eventsToDataFrame(
        block.filter(
            objects=Event,
            name='Label'), idxT='t')

    moveOnMask = rigEvDF['Label'] == 'Movement Onset'
    moveOffMask = rigEvDF['Label'] == 'Movement Offset'
    moveMask = moveOnMask | moveOffMask
    moveEvents = pd.DataFrame(
        rigEvDF.loc[moveMask, ['Label', 't']],
        columns=['Label', 't'])
    moveEvents['ins_property'] = 'movement'
    moveEvents['ins_value'] = (
        moveEvents['Label'] == 'Movement Onset').astype(np.float32)
    moveEvents.drop(columns=['Label'], inplace=True)
    moveEvents.rename(
        columns={
            'rig_property': 'ins_property',
            'rig_value': 'ins_value'},
        inplace=True)
    #  TODO use epochs for amplitude and movement!
    stimStSer = pd.concat(
        (insEvDF, moveEvents),
        axis=0, ignore_index=True
        ).sort_values('t').reset_index(drop=True)
    #  pdb.set_trace()
    #  serialize stimStatus
    expandCols = [
        'RateInHz', 'therapyStatus',
        'activeGroup', 'program', 'trialSegment', 'movement']
    deriveCols = ['amplitudeRound', 'movementRound']
    progAmpNames = rcsa_helpers.progAmpNames
    stimStatus = hf.stimStatusSerialtoLong(
        stimStSer, idxT='t', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    #  add stim info to traces
    #  pdb.set_trace()

    if interpolateToTimeSeries:
        columnsToBeAdded = (
            expandCols + deriveCols + progAmpNames)
        infoFromStimStatus = hf.interpolateDF(
            stimStatus, tdDF['t'],
            x='t', columns=columnsToBeAdded, kind='previous')
        infoFromStimStatus['amplitudeIncrease'] = (
            infoFromStimStatus['amplitudeRound'].diff().fillna(0))
        tdDF = pd.concat((
            tdDF,
            infoFromStimStatus.drop(columns='t')),
            axis=1)
    return tdDF, stimStatus

def findSegsIncluding(block, timeSlice=None):
    segBoundsList = []
    for segIdx, seg in enumerate(block.segments):
        segBoundsList.append(pd.DataFrame({
            't_start': seg.t_start,
            't_stop': seg.t_stop
            }, index=[segIdx]))

    segBounds = pd.concat(segBoundsList)
    if timeSlice[0] is not None:
        segMask = (segBounds['t_start'] * s >= timeSlice[0]) & (
            segBounds['t_stop'] * s <= timeSlice[1])
        requestedSegs = segBounds.loc[segMask, :]
    else:
        timeSlice = (None, None)
        requestedSegs = segBounds
    return segBounds, requestedSegs


def findSegsIncluded(block, timeSlice=None):
    segBoundsList = []
    for segIdx, seg in enumerate(block.segments):
        segBoundsList.append(pd.DataFrame({
            't_start': seg.t_start,
            't_stop': seg.t_stop
            }, index=[segIdx]))

    segBounds = pd.concat(segBoundsList)
    if timeSlice[0] is not None:
        segMask = (segBounds['t_start'] * s <= timeSlice[0]) | (
            segBounds['t_stop'] * s >= timeSlice[1])
        requestedSegs = segBounds.loc[segMask, :]
    else:
        timeSlice = (None, None)
        requestedSegs = segBounds
    return segBounds, requestedSegs


def getElecLookupTable(block, elecIds=None):
    lookupTableList = []
    for metaIdx, chanIdx in enumerate(block.channel_indexes):
        if chanIdx.analogsignals:
            #  print(chanIdx.name)
            lookupTableList.append(pd.DataFrame({
                'channelNames': np.array(chanIdx.channel_names, dtype=np.str),
                'index': chanIdx.index,
                'metaIndex': metaIdx * chanIdx.index**0,
                'localIndex': (
                    list(range(chanIdx.analogsignals[0].shape[1])))
                }))
    lookupTable = pd.concat(lookupTableList, ignore_index=True)

    if elecIds is None:
        requestedIndices = lookupTable
    else:
        if isinstance(elecIds[0], str):
            idxMask = lookupTable['channelNames'].isin(elecIds)
            requestedIndices = lookupTable.loc[idxMask, :]
    return lookupTable, requestedIndices


def getNIXData(
        fileName=None,
        folderPath=None,
        reader=None, blockIdx=0,
        elecIds=None, startTime_s=None,
        dataLength_s=None, downsample=1,
        signal_group_mode='group-by-same-units',
        closeReader=False):
    #  Open file and extract headers
    if reader is None:
        assert (fileName is not None) and (folderPath is not None)
        filePath = os.path.join(folderPath, fileName) + '.nix'
        reader = neo.io.nixio_fr.NixIO(filename=filePath)

    block = reader.read_block(
        block_index=blockIdx, lazy=True,
        signal_group_mode=signal_group_mode)

    for segIdx, seg in enumerate(block.segments):
        seg.events = [i.load() for i in seg.events]
        seg.epochs = [i.load() for i in seg.epochs]

    # find elecIds
    lookupTable, requestedIndices = getElecLookupTable(
        block, elecIds=elecIds)

    # find segments that contain the requested times
    if dataLength_s is not None:
        assert startTime_s is not None
        timeSlice = (
            startTime_s * s,
            (startTime_s + dataLength_s) * s)
    else:
        timeSlice = (None, None)
    segBounds, requestedSegs = findSegsIncluding(block, timeSlice)
    
    data = pd.DataFrame(columns=elecIds + ['t'])
    for segIdx in requestedSegs.index:
        seg = block.segments[segIdx]
        if dataLength_s is not None:
            timeSlice = (
                max(timeSlice[0], seg.t_start),
                min(timeSlice[1], seg.t_stop)
                )
        else:
            timeSlice = (seg.t_start, seg.t_stop)
        segData = pd.DataFrame()
        for metaIdx in pd.unique(requestedIndices['metaIndex']):
            metaIdxMatch = requestedIndices['metaIndex'] == metaIdx
            theseRequestedIndices = requestedIndices.loc[
                metaIdxMatch, :]
            theseElecIds = theseRequestedIndices['channelNames']
            asig = seg.analogsignals[metaIdx]
            thisTimeSlice = (
                max(timeSlice[0], asig.t_start),
                min(timeSlice[1], asig.t_stop)
                )
            #  pdb.set_trace()
            reqData = asig.load(
                time_slice=thisTimeSlice,
                channel_indexes=theseRequestedIndices['localIndex'].values)
            segData = pd.concat((
                    segData,
                    pd.DataFrame(
                        reqData.magnitude, columns=theseElecIds.values)),
                axis=1)
        segT = reqData.times
        segData['t'] = segT
        data = pd.concat(
            (data, segData),
            axis=0, ignore_index=True)
    channelData = {
        'data': data,
        't': data['t']
        }
    if closeReader:
        reader.file.close()
        block = None
        # closing the reader breaks its connection to the block
    return channelData, block


def blockToNix(
        block, writer, chunkSize,
        segInitIdx,
        pruneOutUnits=True,
        fillOverflow=False,
        eventInfo=None,
        removeJumps=False, trackMemory=True
        ):
    idx = segInitIdx
    #  prune out nev spike placeholders
    #  (will get added back on a chunk by chunk basis,
    #  if not pruning units)
    for chanIdx in block.channel_indexes:
        if chanIdx.units:
            for unit in chanIdx.units:
                if unit.spiketrains:
                    unit.spiketrains = []
            if pruneOutUnits:
                chanIdx.units = []

    #  if not keeping the spiketrains, trim down the channel_index list
    if pruneOutUnits:
        block.channel_indexes = (
            [chanIdx for chanIdx in block.channel_indexes if (
                chanIdx.analogsignals)])

    #  delete asig and irsig proxies from channel index list
    for metaIdx, chanIdx in enumerate(block.channel_indexes):
        if chanIdx.analogsignals:
            chanIdx.analogsignals = []
        if chanIdx.irregularlysampledsignals:
            chanIdx.irregularlysampledsignals = []
    
    #  precalculate new segments
    newSegList = []
    oldSegList = block.segments
    #  keep track of which oldSeg newSegs spawn from
    segParents = {}
    for segIdx, seg in enumerate(block.segments):
        segLen = seg.analogsignals[0].shape[0] / (
            seg.analogsignals[0].sampling_rate)
        nChunks = math.ceil(segLen / chunkSize)
        actualChunkSize = (segLen / nChunks).magnitude
        segParents.update({segIdx: {}})
        for chunkIdx in range(nChunks):
            #  for chunkIdx in [0, 1]:
            tStart = chunkIdx * actualChunkSize
            tStop = (chunkIdx + 1) * actualChunkSize

            newSeg = Segment(
                    index=idx, name=seg.name,
                    description=seg.description,
                    file_origin=seg.file_origin,
                    file_datetime=seg.file_datetime,
                    rec_datetime=seg.rec_datetime,
                    **seg.annotations
                )
                
            newSegList.append(newSeg)
            segParents[segIdx].update(
                {chunkIdx: newSegList.index(newSeg)})
            idx += 1
    block.segments = newSegList
    block, nixblock = writer.write_block_meta(block)

    # descend into Segments
    for segIdx, seg in enumerate(oldSegList):
        segLen = seg.analogsignals[0].shape[0] / (
            seg.analogsignals[0].sampling_rate)
        nChunks = math.ceil(segLen / chunkSize)
        actualChunkSize = (segLen / nChunks).magnitude
        for chunkIdx in range(nChunks):
            #  for chunkIdx in [0, 1]:
            tStart = chunkIdx * actualChunkSize
            tStop = (chunkIdx + 1) * actualChunkSize
            print(
                'PreprocNs5: starting chunk %d of %d' % (
                    chunkIdx + 1, nChunks))
            if trackMemory:
                print('memory usage: {}'.format(
                    hf.memory_usage_psutil()))
            newSeg = block.segments[segParents[segIdx][chunkIdx]]
            newSeg, nixgroup = writer._write_segment_meta(newSeg, nixblock)
            
            if eventInfo is not None:
                #  process trial related events
                analogData = []
                for key, value in eventInfo['inputIDs'].items():
                    ainpAsig = seg.filter(
                        objects=AnalogSignalProxy,
                        name=value)[0]
                    ainpData = ainpAsig.load(
                        time_slice=(tStart, tStop),
                        magnitude_mode='rescaled')
                    analogData.append(
                        pd.DataFrame(ainpData.magnitude, columns=[key]))
                    del ainpData
                motorData = pd.concat(analogData, axis=1)
                del analogData
                motorData = mea.processMotorData(
                    motorData, ainpAsig.sampling_rate.magnitude)
                plotExample = False
                if plotExample:
                    exampleTrace = motorData.loc[12e6:13e6, 'position']
                    exampleTrace.plot()
                    plt.show()
                keepCols = [
                    'position', 'velocity', 'velocityCat',
                    'rightBut_int', 'leftBut_int',
                    'rightLED_int', 'leftLED_int', 'simiTrigs_int']
                for colName in keepCols:
                    if trackMemory:
                        print('writing motorData memory usage: {}'.format(
                            hf.memory_usage_psutil()))
                    chanIdx = ChannelIndex(
                        name=colName,
                        index=np.array([0]),
                        channel_names=np.array([0]))
                    block.channel_indexes.append(chanIdx)
                    motorAsig = AnalogSignal(
                        motorData[colName].values * pq.mV,
                        name=colName,
                        sampling_rate=ainpAsig.sampling_rate,
                        dtype=np.float32)
                    motorAsig.t_start = ainpAsig.t_start
                    motorAsig.channel_index = chanIdx
                    # assign ownership to containers
                    chanIdx.analogsignals.append(motorAsig)
                    newSeg.analogsignals.append(motorAsig)
                    chanIdx.create_relationship()
                    newSeg.create_relationship()
                    # write out to file
                    motorAsig = writer._write_analogsignal(
                        motorAsig, nixblock, nixgroup)
                    del motorAsig
                _, trialEvents = mea.getTrialsNew(
                    motorData, ainpAsig.sampling_rate.magnitude,
                    tStart, trialType=None)
                trialEvents.fillna(0)
                trialEvents.rename(
                    columns={
                        'Label': 'rig_property',
                        'Details': 'rig_value'}, inplace=True)
                #  pdb.set_trace()
                del motorData
                eventList = eventDataFrameToEvents(
                    trialEvents,
                    idxT='Time',
                    annCol=['rig_property', 'rig_value'])
                for event in eventList:
                    #  pdb.set_trace()
                    if trackMemory:
                        print('writing motor events memory usage: {}'.format(
                            hf.memory_usage_psutil()))
                    event.segment = newSeg
                    newSeg.events.append(event)
                    newSeg.create_relationship()
                    # write out to file
                    event = writer._write_event(event, nixblock, nixgroup)
                    del event
                del trialEvents, eventList

            if not pruneOutUnits:
                for stIdx, stProxy in enumerate(seg.spiketrains):
                    if trackMemory:
                        print('writing spiketrains mem usage: {}'.format(
                            hf.memory_usage_psutil()))
                    unit = stProxy.unit
                    st = stProxy.load(
                        time_slice=(tStart, tStop),
                        magnitude_mode='rescaled',
                        load_waveforms=True)
                    #  link SpikeTrain and ID providing unit
                    st.unit = unit
                    # assign ownership to containers
                    unit.spiketrains.append(st)
                    newSeg.spiketrains.append(st)
                    # assign parent to children
                    unit.create_relationship()
                    newSeg.create_relationship()
                    # write out to file
                    st = writer._write_spiketrain(st, nixblock, nixgroup)
                    del st

            for aSigIdx, aSigProxy in enumerate(seg.analogsignals):
                if trackMemory:
                    print('writing asigs memory usage: {}'.format(
                        hf.memory_usage_psutil()))
                chanIdx = aSigProxy.channel_index
                asig = aSigProxy.load(
                    time_slice=(tStart, tStop),
                    magnitude_mode='rescaled')
                #  link AnalogSignal and ID providing channel_index
                asig.channel_index = chanIdx
                #  perform requested preproc operations
                if fillOverflow:
                    # fill in overflow:
                    '''
                    timeSection['data'], overflowMask = hf.fillInOverflow(
                        timeSection['data'], fillMethod = 'average')
                    badData.update({'overflow': overflowMask})
                    '''
                    pass
                if removeJumps:
                    # find unusual jumps in derivative or amplitude
                    '''
                    timeSection['data'], newBadData = hf.fillInJumps(timeSection['data'],
                        timeSection['samp_per_s'], smoothing_ms = 0.5, nStdDiff = 50,
                        nStdAmp = 100)
                    badData.update(newBadData)
                    '''
                    pass

                # assign ownership to containers
                chanIdx.analogsignals.append(asig)
                newSeg.analogsignals.append(asig)
                # assign parent to children
                chanIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                asig = writer._write_analogsignal(asig, nixblock, nixgroup)
                del asig

            for irSigIdx, irSigProxy in enumerate(
                    seg.irregularlysampledsignals):
                chanIdx = irSigProxy.channel_index

                isig = irSigProxy.load(
                    time_slice=(tStart, tStop),
                    magnitude_mode='rescaled')
                #  link irregularlysampledSignal
                #  and ID providing channel_index
                isig.channel_index = chanIdx
                # assign ownership to containers
                chanIdx.irregularlysampledsignals.append(isig)
                newSeg.irregularlysampledsignals.append(isig)
                # assign parent to children
                chanIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                isig = writer._write_irregularlysampledsignal(
                    isig, nixblock, nixgroup)
                del isig

            for eventProxy in seg.events:
                event = eventProxy.load(
                    time_slice=(tStart, tStop))
                event.segment = newSeg
                newSeg.events.append(event)
                newSeg.create_relationship()
                # write out to file
                event = writer._write_event(event, nixblock, nixgroup)
                del event

            for epochProxy in seg.epochs:
                epoch = epochProxy.load(
                    time_slice=(tStart, tStop))
                epoch.segment = newSeg
                newSeg.events.append(epoch)
                newSeg.create_relationship()
                # write out to file
                epoch = writer._write_epoch(epoch, nixblock, nixgroup)
                del epoch

    # descend into ChannelIndexes
    for chanIdx in block.channel_indexes:
        chanIdx = writer._write_channelindex(chanIdx, nixblock)
    writer._create_source_links(block, nixblock)
    return idx


#  TODO: write code that merges a dataframe and a spikesdict to a block
def addBlockToNIX(
        newBlock, segIdx=0,
        writeAsigs=True, writeSpikes=True,
        fileName=None,
        folderPath=None,
        nixBlockIdx=0, nixSegIdx=0,
        ):
    #  base file name
    trialBasePath = os.path.join(folderPath, fileName)
    
    # peek at file to ensure compatibility
    reader = neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix')
    tempBlock = reader.read_block(
        block_index=nixBlockIdx,
        lazy=True)

    '''
    #  future dev: find segment automatically
    for segIdx, seg in enumerate(tempBlock.segments):
        seg.events = [i.load() for i in seg.events]
        seg.epochs = [i.load() for i in seg.epochs]

    timeSlice = (
        newBlock.segments[0].t_start,
        newBlock.segments[0].t_stop)

    _, segsWriteTo = preproc.findSegsIncluded(tempBlock, timeSlice)
    '''

    maxIdx = 0
    for chanIdx in tempBlock.channel_indexes:
        if chanIdx.analogsignals:
            maxIdx = max(maxIdx, max(chanIdx.index))
    
    tempAsig = tempBlock.channel_indexes[0].analogsignals[0]

    forceType = tempAsig.dtype
    forceShape = tempAsig.shape[0]  # ? docs say shape[1], but that's confusing
    forceFS = tempAsig.sampling_rate
    reader.file.close()
    #  if newBlock was loaded from a nix file, strip the old nix_names away:
    
    for child in newBlock.children_recur:
        child.annotations.pop('nix_name', None)
        child.annotations.pop('neo_name', None)
    for child in newBlock.data_children_recur:
        child.annotations.pop('nix_name', None)
        child.annotations.pop('neo_name', None)
    
    writer = neo.io.NixIO(filename=trialBasePath + '.nix')
    nixblock = writer.nix_file.blocks[nixBlockIdx]
    nixblockName = nixblock.name
    newBlock.annotate(nix_name=nixblockName)

    #  for idx, segIdx in enumerate(segsWriteTo.index):
    nixgroup = nixblock.groups[nixSegIdx]
    nixSegName = nixgroup.name
    newBlock.segments[segIdx].annotate(nix_name=nixSegName)
    #  TODO: double check that you can't add the same thing twice
    for event in newBlock.segments[segIdx].events:
        event = writer._write_event(event, nixblock, nixgroup)
    if writeAsigs:
        for asig in newBlock.segments[segIdx].analogsignals:
            assert asig.dtype == forceType
            assert asig.sampling_rate == forceFS
            assert asig.shape[0] == forceShape
            asig = writer._write_analogsignal(asig, nixblock, nixgroup)
        for isig in newBlock.segments[segIdx].irregularlysampledsignals:
            isig = writer._write_irregularlysampledsignal(
                isig, nixblock, nixgroup)
    
    if writeSpikes:
        alreadyWrittenSpikeTrainNames = []
        for st in newBlock.segments[segIdx].spiketrains:
            #  pdb.set_trace()
            if st.name not in alreadyWrittenSpikeTrainNames:
                alreadyWrittenSpikeTrainNames.append(st.name)
                st = writer._write_spiketrain(st, nixblock, nixgroup)
            #  print('already wrote {}'.format(alreadyWrittenSpikeTrainNames))
    
    alreadyWrittenChanNames = []
    for chanIdx in newBlock.channel_indexes:
        if chanIdx.name not in alreadyWrittenChanNames:
            chanIdx = writer._write_channelindex(chanIdx, nixblock)
            alreadyWrittenChanNames.append(chanIdx.name)
            #  print('already wrote {}'.format(alreadyWrittenChanNames))
        #  auto descends into units inside of _write_channelindex
    writer._create_source_links(newBlock, nixblock)
    writer.close()
    return newBlock


def preproc(
        fileName='Trial001',
        folderPath='./',
        fillOverflow=True, removeJumps=True,
        eventInfo=None,
        pruneOutUnits=True,
        chunkSize=1800,
        signal_group_mode='split-all', trialInfo=None
        ):

    #  base file name
    trialBasePath = os.path.join(folderPath, fileName)
    #  instantiate reader, get metadata
    reader = neo.io.BlackrockIO(
        filename=trialBasePath, nsx_to_load=5)
    reader.parse_header()
    metadata = reader.header
    
    #  instantiate writer
    writer = neo.io.NixIO(filename=trialBasePath + '.nix')
    #  absolute section index
    idx = 0
    for blkIdx in range(metadata['nb_block']):
        #  blkIdx = 0
        block = reader.read_block(
            block_index=blkIdx, lazy=True,
            signal_group_mode=signal_group_mode)
        idx = blockToNix(
            block, writer, chunkSize,
            segInitIdx=idx,
            fillOverflow=fillOverflow,
            removeJumps=removeJumps,
            eventInfo=eventInfo,
            pruneOutUnits=pruneOutUnits
            )

    writer.close()

    return neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix')


def calcSpikeMatsAndSave(block):

    return block

def purgeNixAnn(block):

    block.annotations.pop('nix_name', None)
    block.annotations.pop('neo_name', None)
    for child in block.children_recur:
        child.annotations.pop('nix_name', None)
        child.annotations.pop('neo_name', None)
    for child in block.data_children_recur:
        child.annotations.pop('nix_name', None)
        child.annotations.pop('neo_name', None)

    return block