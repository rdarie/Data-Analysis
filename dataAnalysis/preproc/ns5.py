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
import sys, os, gc
import pickle
from copy import *
import traceback
import h5py
import re
from scipy import signal
import rcsanalysis.packet_func as rcsa_helpers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from elephant.conversion import binarize


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


def spikeTrainsToSpikeDict(
        spiketrains):
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
        spikes['ChannelID'][idx] = st.name
        if len(spikes['TimeStamps'][idx]):
            spikes['TimeStamps'][idx] = np.stack((
                spikes['TimeStamps'][idx],
                st.times.magnitude), axis=-1)
        else:
            spikes['TimeStamps'][idx] = st.times.magnitude
        
        theseWaveforms = np.swapaxes(
            st.waveforms, 1, 2)
        theseWaveforms = np.atleast_2d(np.squeeze(
            theseWaveforms))
            
        if len(spikes['Waveforms'][idx]):
            spikes['Waveforms'][idx] = np.stack((
                spikes['Waveforms'][idx],
                theseWaveforms.magnitude), axis=-1)
        else:
            spikes['Waveforms'][idx] = theseWaveforms.magnitude
        
        classVals = st.times.magnitude ** 0 * idx
        if len(spikes['Classification'][idx]):
            spikes['Classification'][idx] = np.stack((
                spikes['Classification'][idx],
                classVals), axis=-1)
        else:
            spikes['Classification'][idx] = classVals
    return spikes


def channelIndexesToSpikeDict(
        channel_indexes):
    nCh = len(channel_indexes)
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
    #  allocate fields for annotations
    dummyUnit = channel_indexes[0].units[0]
    dummySt = [
        st
        for st in dummyUnit.spiketrains
        if len(st.times)][0]
    #  allocate fields for array annotations (per spike)
    if dummySt.array_annotations:
        for key in dummySt.array_annotations.keys():
            spikes.update({key: [np.array([]) for i in range(nCh)]})
        
    maxUnitIdx = 0
    for idx, chIdx in enumerate(channel_indexes):
        spikes['ChannelID'][idx] = chIdx.name
        for unitIdx, thisUnit in enumerate(chIdx.units):
            for stIdx, st in enumerate(thisUnit.spiketrains):
                print(
                    'unit {} has {} spiketrains'.format(
                        thisUnit.name,
                        len(thisUnit.spiketrains)))
                if len(spikes['TimeStamps'][idx]):
                    spikes['TimeStamps'][idx] = np.concatenate((
                        spikes['TimeStamps'][idx],
                        st.times.magnitude), axis=0)
                else:
                    spikes['TimeStamps'][idx] = st.times.magnitude
                #  reshape waveforms to comply with BRM convention
                theseWaveforms = np.swapaxes(
                    st.waveforms, 1, 2)
                theseWaveforms = np.atleast_2d(np.squeeze(
                    theseWaveforms))
                #  append waveforms
                if len(spikes['Waveforms'][idx]):
                    spikes['Waveforms'][idx] = np.concatenate((
                        spikes['Waveforms'][idx],
                        theseWaveforms.magnitude), axis=0)
                else:
                    spikes['Waveforms'][idx] = theseWaveforms.magnitude
                #  give each unit a global index
                classVals = st.times.magnitude ** 0 * maxUnitIdx
                st.array_annotations.update({'Classification': classVals})
                #  expand array_annotations into spikes dict
                for key, value in st.array_annotations.items():
                    if len(spikes[key][idx]):
                        spikes[key][idx] = np.concatenate((
                            spikes[key][idx],
                            value), axis=0)
                    else:
                        spikes[key][idx] = value
                for key, value in st.annotations.items():
                    if key not in spikes['basic_headers']:
                        spikes['basic_headers'].update({key: {}})
                    if value:
                        spikes['basic_headers'][key].update({maxUnitIdx: value})
                maxUnitIdx += 1
    return spikes


def spikeTrainArrayAnnToDF(
        spiketrains):
    fullAnnotationsDict = {}
    for st in spiketrains:
        theseAnnDF = pd.DataFrame(st.array_annotations)
        theseAnnDF['times'] = st.times
        fullAnnotationsDict.update({childBaseName(st.name, 'seg'): theseAnnDF})
    
    annotationsDF = pd.concat(
        fullAnnotationsDict, names=['name', 'index'])
    return annotationsDF


def spikeTrainWaveformsToDF(
        spiketrains):
    waveformsDict = {}
    for st in spiketrains:
        wfDF = pd.DataFrame(np.squeeze(st.waveforms))
        annDict = {
            k: pd.Series(v)
            for k, v in st.array_annotations.items()}
        annDict.update({'t': pd.Series(st.times.magnitude)})
        annDF = pd.concat(annDict, axis=1).reset_index()
        fullDF = pd.concat([wfDF, annDF], axis=1)
        fullDF.set_index(annDF.columns.tolist(), inplace=True)
        fullDF.columns.name = 'bin'
        waveformsDict.update({childBaseName(st.name, 'seg'): fullDF})
    waveformsDF = pd.concat(
        waveformsDict, names=['feature'] + annDF.columns.tolist())
    return waveformsDF


def analogSignalsToDataFrame(
        analogsignals, idxT='t', useChanNames=False):
    asigList = []
    for asig in analogsignals:
        if asig.shape[1] == 1:
            if useChanNames:
                colNames = [asig.channel_index.name]
            else:
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


def analogSignalsAlignedToEvents(
        eventBlock=None, signalBlock=None,
        chansToTrigger=None, eventName=None,
        windowSize=None, appendToExisting=False,
        checkReferences=True,
        fileName=None,
        folderPath=None
        ):
    if signalBlock is None:
        signalBlock = eventBlock
    masterBlock = Block()
    masterBlock.name = signalBlock.annotations['neo_name']
    masterBlock.annotate(nix_name=signalBlock.annotations['neo_name'])
    #  make channels and units for triggered time series
    for chanName in chansToTrigger:
        chanIdx = ChannelIndex(name=chanName + '#0', index=[0])
        chanIdx.annotate(nix_name=chanIdx.name)
        thisUnit = Unit(name=chanIdx.name)
        thisUnit.annotate(nix_name=chanIdx.name)
        chanIdx.units.append(thisUnit)
        thisUnit.channel_index = chanIdx
        masterBlock.channel_indexes.append(chanIdx)

    for segIdx, eventSeg in enumerate(eventBlock.segments):
        signalSeg = signalBlock.segments[segIdx]
        # seg to contain triggered time series
        newSeg = Segment(name=signalSeg.annotations['neo_name'])
        newSeg.annotate(nix_name=signalSeg.annotations['neo_name'])
        masterBlock.segments.append(newSeg)

        alignEvents = ([
            i.load()
            for i in eventSeg.filter(
                objects=EventProxy, name=eventName)] +
            eventSeg.filter(objects=Event))
        alignEvents = loadContainerArrayAnn(trainList=alignEvents)[0]
        for chanName in chansToTrigger:
            print('extracting channel {}'.format(chanName))
            asigP = signalSeg.filter(
                objects=AnalogSignalProxy, name=chanName)[0]
            if checkReferences:
                da = (
                    asigP
                    ._rawio
                    .da_list['blocks'][0]['segments'][segIdx]['data'])
                print('segIdx {}, asigP.name {}'.format(
                    segIdx, asigP.name))
                print('asigP._global_channel_indexes = {}'.format(
                    asigP._global_channel_indexes))
                print('asigP references {}'.format(
                    da[asigP._global_channel_indexes[0]]))
                try:
                    assert (
                        asigP.name
                        in da[asigP._global_channel_indexes[0]].name)
                except Exception:
                    traceback.print_exc()
            validMask = (
                ((alignEvents + windowSize[1]) < asigP.t_stop) &
                ((alignEvents + windowSize[0]) > asigP.t_start)
                )
            alignEvents = alignEvents[validMask]
            rawWaveforms = [
                asigP.load(time_slice=(t + windowSize[0], t + windowSize[1]))
                for t in alignEvents]

            samplingRate = asigP.sampling_rate
            waveformUnits = rawWaveforms[0].units
            #  fix length
            minLen = min([rW.shape[0] for rW in rawWaveforms])
            rawWaveforms = [rW[:minLen] for rW in rawWaveforms]

            spikeWaveforms = (
                np.hstack([rW.magnitude for rW in rawWaveforms])
                .transpose()[:, np.newaxis, :] * waveformUnits
                )

            thisUnit = masterBlock.filter(
                objects=Unit, name=chanName + '#0')[0]
            stAnn = {
                k: v
                for k, v in alignEvents.annotations.items()
                if k not in ['nix_name', 'neo_name']
                }
            st = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=alignEvents.times,
                waveforms=spikeWaveforms,
                t_start=asigP.t_start, t_stop=asigP.t_stop,
                left_sweep=windowSize[0] * (-1),
                sampling_rate=samplingRate,
                array_annotations=alignEvents.array_annotations,
                **stAnn
                )
            st.annotate(nix_name=st.name)
            thisUnit.spiketrains.append(st)
            newSeg.spiketrains.append(st)
            st.unit = thisUnit
    try:
        eventBlock.filter(
            objects=EventProxy)[0]._rawio.file.close()
    except Exception:
        traceback.print_exc()
    if signalBlock is not eventBlock:
        signalBlock.filter(
            objects=AnalogSignalProxy)[0]._rawio.file.close()
    if appendToExisting:
        allSegs = list(range(len(masterBlock.segments)))
        addBlockToNIX(
            masterBlock, neoSegIdx=allSegs,
            writeSpikes=True,
            fileName=fileName,
            folderPath=folderPath,
            purgeNixNames=False,
            nixBlockIdx=0, nixSegIdx=allSegs)
    else:
        experimentTriggeredPath = os.path.join(
            folderPath, fileName + '.nix')
        masterBlock = purgeNixAnn(masterBlock)
        writer = neo.io.NixIO(filename=experimentTriggeredPath)
        writer.write_block(masterBlock, use_obj_names=True)
        writer.close()

    return masterBlock


def dataFrameToAnalogSignals(
        df,
        block=None, seg=None,
        idxT='NSPTime',
        probeName='insTD', samplingRate=500*pq.Hz,
        timeUnits=pq.s, measureUnits=pq.mV,
        dataCol=['channel_0', 'channel_1'],
        useColNames=False, forceColNames=None,
        namePrefix='', nameSuffix=''):

    if block is None:
        assert seg is None
        block = Block()
        seg = Segment(name=probeName + ' segment')
        block.segments.append(seg)

    for idx, colName in enumerate(dataCol):
        if forceColNames is not None:
            chanName = forceColNames[idx]
        elif useColNames:
            chanName = namePrefix + colName + nameSuffix
        else:
            chanName = namePrefix + (probeName.lower() + '{}'.format(idx)) + nameSuffix

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
    block.create_relationship()
    seg.create_relationship()
    return block


def eventDataFrameToEvents(
        eventDF, idxT=None,
        annCol=None,
        eventName='', tUnits=pq.s,
        makeList=True
        ):
    if makeList:
        eventList = []
        for colName in annCol:
            originalDType = type(eventDF[colName].values[0]).__name__
            event = Event(
                name=eventName + colName,
                times=eventDF[idxT].values * tUnits,
                labels=eventDF[colName].astype(originalDType).values
                )
            event.annotate(originalDType=originalDType)
            eventList.append(event)
        return eventList
    else:
        if annCol is None:
            annCol = eventDF.drop(columns=idxT).columns
        event = Event(
            name=eventName,
            times=eventDF[idxT].values * tUnits,
            labels=np.array(eventDF.index)
            )
        event.annotations.update(
            {
                'arrayAnnNames': [],
                'arrayAnnDTypes': []
                })
        for colName in annCol:
            originalDType = type(eventDF[colName].values[0]).__name__
            arrayAnn = eventDF[colName].astype(originalDType).values
            event.array_annotations.update(
                {colName: arrayAnn})
            event.annotations['arrayAnnNames'].append(colName)
            event.annotations['arrayAnnDTypes'].append(originalDType)
            event.annotations.update(
                {colName: arrayAnn})
        return event


def eventsToDataFrame(
        events, idxT='t', names=None
        ):
    eventDict = {}
    
    calculatedT = False
    for event in events:
        if names is not None:
            if event.name not in names:
                continue
        if len(event.times):
            if not calculatedT:
                t = pd.Series(event.times.magnitude)
                calculatedT = True
            #  print(event.name)
            values = event.array_annotations['labels']
            if isinstance(values[0], bytes):
                #  event came from hdf, need to recover dtype
                dtypeStr = event.annotations['originalDType'].split(';')[-1]
                if 'np.' not in dtypeStr:
                    dtypeStr = 'np.' + dtypeStr
                originalDType = eval(dtypeStr)
                values = np.array(values, dtype=originalDType)
            #  print(values.dtype)
            eventDict.update({
                event.name: pd.Series(values)})
    eventDict.update({idxT: t})
    return pd.concat(eventDict, axis=1)


'''
def unpackAnalysisBlock(
        block, interpolateToTimeSeries=False,
        binnedSpikePath=None):
    
    tdDFList = []
    for seg in block.segments:
        tdAsig = seg.filter(
            objects=AnalogSignal
            )
        tdDFList.append(
            analogSignalsToDataFrame(tdAsig, useChanNames=True))
    tdDF = pd.concat(tdDFList, ignore_index=True, sort=True)
    
    if binnedSpikePath is not None:
        reader = neo.io.nixio_fr.NixIO(
            filename=binnedSpikePath)
        dummyBlock = reader.read_block(
            block_index=0, lazy=True)
        tStart = dummyBlock.segments[0].analogsignals[0].t_start
        fs = dummyBlock.segments[0].analogsignals[0].sampling_rate
        tStop = dummyBlock.segments[-1].analogsignals[0].t_stop
        newX = np.arange(tStart, tStop, 1 / fs)

        tdDF = hf.interpolateDF(
            tdDF, newX, kind='linear',
            x='t')
    #  TODO use epochs for amplitude and movement!
    
    stimStSerList = []
    for segIdx, seg in enumerate(block.segments):
        eventDF = eventsToDataFrame(
            seg.events, idxT='t',
            names=[
                'seg{}_property'.format(segIdx),
                'seg{}_value'.format(segIdx)]
            )
        eventDF.rename(
            columns={
                'seg{}_property'.format(segIdx): 'property',
                'seg{}_value'.format(segIdx): 'value'
            }, inplace=True)
        stimStSerList.append(eventDF)
    stimStSer = pd.concat(stimStSerList, ignore_index=True, sort=True)        
    #  serialize stimStatus
    expandCols = [
        'RateInHz', 'therapyStatus',
        'activeGroup', 'trialSegment', 'movement']
    deriveCols = ['amplitudeRound', 'movementRound']
    progAmpNames = rcsa_helpers.progAmpNames
    stimStatus = hf.stimStatusSerialtoLong(
        stimStSer, idxT='t', namePrefix='', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    #  add stim info to traces
    debugPlot = False
    if debugPlot:
        stimStatus.loc[:, ['program'] + progAmpNames].plot()
        plt.show()
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
        #  tdDF.loc[3e5:4e5, ['program', 'amplitude']].plot(); plt.show()
    return tdDF, stimStatus
'''


def loadSpikeMats(
        dataPath, rasterOpts,
        alignTimes=None, chans=None, loadAll=False,
        absoluteBins=False, transposeSpikeMat=False,
        checkReferences=False):

    reader = neo.io.nixio_fr.NixIO(filename=dataPath)
    chanNames = reader.header['signal_channels']['name']
    
    if chans is not None:
        sigMask = np.isin(chanNames, chans)
        chanNames = chanNames[sigMask]
    chanIdx = reader.channel_name_to_index(chanNames)
    
    if not loadAll:
        assert alignTimes is not None
        spikeMats = {i: None for i in alignTimes.index}
        validTrials = pd.Series(True, index=alignTimes.index)
    else:
        spikeMats = {
            i: None for i in range(reader.segment_count(block_index=0))}
        validTrials = None
    
    for segIdx in range(reader.segment_count(block_index=0)):
        if checkReferences:
            for i, cIdx in enumerate(chanIdx):
                da = reader.da_list['blocks'][
                    0]['segments'][segIdx]['data'][cIdx]
                print('name {}, da.name {}'.format(
                    chanNames[i], da.name
                    ))
                try:
                    assert chanNames[i] in da.name, 'reference problem!!'
                except Exception:
                    pass
        tStart = reader.get_signal_t_start(
            block_index=0, seg_index=segIdx)
        fs = reader.get_signal_sampling_rate(
            channel_indexes=chanIdx
            )
        sigSize = reader.get_signal_size(
            block_index=0, seg_index=segIdx
            )
        tStop = sigSize / fs + tStart
        # convert to indices early to avoid floating point problems
        
        intervalIdx = int(round(rasterOpts['binInterval'] * fs))
        #halfIntervalIdx = int(round(intervalIdx / 2))
        
        widthIdx = int(3 * round(rasterOpts['binWidth'] * fs))
        halfWidthIdx = int(round(widthIdx / 2))
        
        theBins = None

        if not loadAll:
            winStartIdx = int(round(rasterOpts['windowSize'][0] * fs))
            winStopIdx = int(round(rasterOpts['windowSize'][1] * fs))
            timeMask = (alignTimes > tStart) & (alignTimes < tStop)
            maskedTimes = alignTimes[timeMask]
        else:
            #  irrelevant, will load all
            maskedTimes = pd.Series(np.nan)

        for idx, tOnset in maskedTimes.iteritems():
            if not loadAll:
                idxOnset = int(round((tOnset - tStart) * fs))
                #  can't not be ints
                iStart = idxOnset + winStartIdx - halfWidthIdx
                iStop = idxOnset + winStopIdx + halfWidthIdx
            else:
                winStartIdx = 0
                iStart = 0
                iStop = sigSize

            if iStart < 0:
                #  near the first edge
                validTrials[idx] = False
            elif (sigSize < iStop):
                #  near the ending edge
                validTrials[idx] = False
            else:
                #  valid slices
                rawSpikeMat = pd.DataFrame(
                    reader.get_analogsignal_chunk(
                        block_index=0, seg_index=segIdx,
                        i_start=iStart, i_stop=iStop,
                        channel_names=chanNames))
                procSpikeMat = rawSpikeMat.rolling(
                    window=widthIdx, center=True,
                    win_type='gaussian'
                    ).mean(std=widthIdx/6).dropna().iloc[::intervalIdx, :]
                    
                procSpikeMat.columns = chanNames
                procSpikeMat.columns.name = 'unit'
                if theBins is None:
                    theBins = np.array(
                        procSpikeMat.index + winStartIdx) / fs
                if absoluteBins:
                    procSpikeMat.index = theBins + idxOnset / fs
                else:
                    procSpikeMat.index = theBins
                procSpikeMat.index.name = 'bin'
                if loadAll:
                    smIdx = segIdx
                else:
                    smIdx = idx
                    
                spikeMats[smIdx] = procSpikeMat
                if transposeSpikeMat:
                    spikeMats[smIdx] = spikeMats[smIdx].transpose()
            #  plt.imshow(rawSpikeMat.values, aspect='equal'); plt.show()
    return spikeMats, validTrials


def findSegsIncluding(
        block, timeSlice=None):
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


def findSegsIncluded(
        block, timeSlice=None):
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


def getElecLookupTable(
        block, elecIds=None):
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
    
    #  for stp in block.filter(objects=SpikeTrainProxy):
    #      print('original tstart: '.format(stp.t_start))
    #      print('original tstop: '.format(stp.t_stop))
    if closeReader:
        reader.file.close()
        block = None
        # closing the reader breaks its connection to the block
    return channelData, block


def childBaseName(
        childName, searchTerm):
    if searchTerm in childName:
        baseName = '_'.join(childName.split('_')[1:])
    else:
        baseName = childName
    return baseName


def readBlockFixNames(
        rawioReader,
        block_index=0, signal_group_mode='split-all',
        lazy=True):

    dataBlock = rawioReader.read_block(
        block_index=block_index, lazy=lazy,
        signal_group_mode=signal_group_mode)
    
    #  
    if dataBlock.name is None:
        dataBlock.name = dataBlock.annotations['neo_name']
    
    #  on first segment, rename the chan_indexes and units
    seg0 = dataBlock.segments[0]
    asigLikeList = (
        seg0.filter(objects=AnalogSignalProxy) +
        seg0.filter(objects=AnalogSignal)
        )
    for asig in asigLikeList:
        if 'seg' in asig.name:
            asigBaseName = '_'.join(asig.name.split('_')[1:])
        else:
            asigBaseName = asig.name
        if 'Channel group ' in asig.channel_index.name:
            asig.channel_index.name = asigBaseName
    
    spikeTrainLikeList = (
        seg0.filter(objects=SpikeTrainProxy) +
        seg0.filter(objects=SpikeTrain)
        )
    for stp in spikeTrainLikeList:
        # todo: use childBaseName function
        if 'seg' in stp.name:
            stpBaseName = '_'.join(stp.name.split('_')[1:])
        else:
            stpBaseName = stp.name
        '''
        if isinstance(rawioReader, neo.io.blackrockio.BlackrockIO):
            unit = stp.unit
            nameParser = re.search(r'ch(\d*)#(\d*)', unit.name)
            chanId = nameParser.group(1)
            unitId = nameParser.group(2)
            chanIdx = [
                i for i in dataBlock.channel_indexes
                if int(chanId) in i.channel_ids][0]
            stpBaseName = '{}#{}'.format(chanIdx.name, unitId)
            stp.name = stpBaseName
        '''
        if 'ChannelIndex for ' in stp.unit.channel_index.name:
            stp.unit.name = stpBaseName
            stp.unit.channel_index.name = stpBaseName
    
    #  rename the children
    typesNeedRenaming = [
        SpikeTrainProxy, AnalogSignalProxy, EventProxy,
        SpikeTrain, AnalogSignal, Event]
    for segIdx, seg in enumerate(dataBlock.segments):
        if seg.name is None:
            seg.name = 'seg{}_'.format(segIdx)
        else:
            if 'seg{}_'.format(segIdx) not in seg.name:
                seg.name = 'seg{}_{}'.format(segIdx, seg.name)
        for objType in typesNeedRenaming:
            for child in seg.filter(objects=objType):
                if 'seg{}_'.format(segIdx) not in child.name:
                    child.name = 'seg{}_{}'.format(segIdx, child.name)
                #  todo: decide if below is needed
                #  elif 'seg' in child.name:
                #      childBaseName = '_'.join(child.name.split('_')[1:])
                #      child.name = 'seg{}_{}'.format(segIdx, childBaseName)
    return dataBlock


def addBlockToNIX(
        newBlock, neoSegIdx=[0],
        writeAsigs=True, writeSpikes=True, writeEvents=True,
        purgeNixNames=False,
        fileName=None,
        folderPath=None,
        nixBlockIdx=0, nixSegIdx=[0],
        ):
    #  base file name
    trialBasePath = os.path.join(folderPath, fileName)
    # peek at file to ensure compatibility
    reader = neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix')
    tempBlock = reader.read_block(
        block_index=nixBlockIdx,
        lazy=True, signal_group_mode='split-all')
    
    checkCompatible = {i: False for i in nixSegIdx}
    forceShape = {i: None for i in nixSegIdx}
    forceType = {i: None for i in nixSegIdx}
    forceFS = {i: None for i in nixSegIdx}
    for nixIdx in nixSegIdx:
        tempAsigList = tempBlock.segments[nixIdx].filter(
            objects=AnalogSignalProxy)
        if len(tempAsigList) > 0:
            tempAsig = tempAsigList[0]
            checkCompatible[nixIdx] = True
            forceType[nixIdx] = tempAsig.dtype
            forceShape[nixIdx] = tempAsig.shape[0]  # ? docs say shape[1], but that's confusing
            forceFS[nixIdx] = tempAsig.sampling_rate
    reader.file.close()
    #  if newBlock was loaded from a nix file, strip the old nix_names away:
    #  todo: replace with function from this module
    if purgeNixNames:
        newBlock = purgeNixAnn(newBlock)
    
    writer = neo.io.NixIO(filename=trialBasePath + '.nix')
    nixblock = writer.nix_file.blocks[nixBlockIdx]
    nixblockName = nixblock.name
    if 'nix_name' in newBlock.annotations.keys():
        try:
            assert newBlock.annotations['nix_name'] == nixblockName
        except Exception:
            newBlock.annotations['nix_name'] = nixblockName
    else:
        newBlock.annotate(nix_name=nixblockName)

    for idx, segIdx in enumerate(neoSegIdx):
        nixIdx = nixSegIdx[idx]
        newSeg = newBlock.segments[segIdx]
        nixgroup = nixblock.groups[nixIdx]
        
        nixSegName = nixgroup.name
        if 'nix_name' in newSeg.annotations.keys():
            try:
                assert newSeg.annotations['nix_name'] == nixSegName
            except Exception:
                newSeg.annotations['nix_name'] = nixSegName
        else:
            newSeg.annotate(nix_name=nixSegName)
        
        if writeEvents:
            eventList = newSeg.events
            eventOrder = np.argsort([i.name for i in eventList])
            for event in [eventList[i] for i in eventOrder]:
                event = writer._write_event(event, nixblock, nixgroup)
        
        if writeAsigs:
            asigList = newSeg.filter(objects=AnalogSignal)
            asigOrder = np.argsort([i.name for i in asigList])
            for asig in [asigList[i] for i in asigOrder]:
                if checkCompatible[nixIdx]:
                    assert asig.dtype == forceType[nixIdx]
                    assert asig.sampling_rate == forceFS[nixIdx]
                    assert asig.shape[0] == forceShape[nixIdx]
                asig = writer._write_analogsignal(asig, nixblock, nixgroup)
            #  for isig in newSeg.filter(objects=IrregularlySampledSignal):
            #      isig = writer._write_irregularlysampledsignal(
            #          isig, nixblock, nixgroup)

        if writeSpikes:
            stList = newSeg.filter(objects=SpikeTrain)
            stOrder = np.argsort([i.name for i in stList])
            for st in [stList[i] for i in stOrder]:
                st = writer._write_spiketrain(st, nixblock, nixgroup)

    for chanIdx in newBlock.filter(objects=ChannelIndex):
        chanIdx = writer._write_channelindex(chanIdx, nixblock)
        #  auto descends into units inside of _write_channelindex
    writer._create_source_links(newBlock, nixblock)
    writer.close()
    return newBlock


def blockToNix(
        block, writer, chunkSize,
        segInitIdx,
        fillOverflow=False,
        eventInfo=None,
        removeJumps=False, trackMemory=True,
        spikeSource='', spikeBlock=None
        ):
    idx = segInitIdx
    #  prune out nev spike placeholders
    #  (will get added back on a chunk by chunk basis,
    #  if not pruning units)
    if spikeSource == 'nev':
        pruneOutUnits = False
    else:
        pruneOutUnits = True

    for chanIdx in block.channel_indexes:
        if chanIdx.units:
            for unit in chanIdx.units:
                if unit.spiketrains:
                    unit.spiketrains = []
            if pruneOutUnits:
                chanIdx.units = []

    if spikeBlock is not None:
        for chanIdx in spikeBlock.channel_indexes:
            if chanIdx.units:
                for unit in chanIdx.units:
                    if unit.spiketrains:
                        unit.spiketrains = []

    #  remove chanIndexes assigned to units; makes more sense to
    #  only use chanIdx for asigs and spikes on that asig
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
        
        if spikeBlock is not None:
            spikeSeg = spikeBlock.segments[segIdx]
        else:
            spikeSeg = seg

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
                gc.collect()

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
                gc.collect()

            if spikeSource:
                for stIdx, stProxy in enumerate(spikeSeg.spiketrains):
                    if trackMemory:
                        print('writing spiketrains mem usage: {}'.format(
                            hf.memory_usage_psutil()))
                    unit = stProxy.unit
                    try:
                        st = stProxy.load(
                            time_slice=(tStart, tStop),
                            magnitude_mode='rescaled',
                            load_waveforms=True)
                    except Exception:
                        st = stProxy.load(
                            time_slice=(tStart, tStop),
                            magnitude_mode='rescaled',
                            load_waveforms=False)
                        st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV

                    st.annotate(fromNSP=True)
                    #  !!! TODO, mark by origin (utah, nForm, ainp)

                    #  rename chanIndexes TODO: replace in readBlockFixNames
                    if spikeSource == 'nev':
                        nameParser = re.search(r'ch(\d*)#(\d*)', unit.name)
                        if nameParser is not None:
                            # first time at this unit, rename it
                            chanId = nameParser.group(1)
                            unitId = nameParser.group(2)
                            chanIdx = [
                                    i for i in block.channel_indexes
                                    if int(chanId) in i.channel_ids][0]
                            unit.channel_index = chanIdx
                            unit.name = '{}#{}'.format(chanIdx.name, unitId)
                            chanIdx.units.append(unit)
                    elif spikeSource == 'tdc':
                        #  tdc may or may not have the same channel ids, but it will have
                        #  consistent channel names
                        #  TODO: fix the fact that units inherit their st's name on load
                        if not (unit in chanIdx.units):
                            nameParser = re.search(
                                r'([a-zA-Z0-9]*)#(\d*)', unit.name)
                            chanLabel = nameParser.group(1)
                            unitId = nameParser.group(2)
                            chanIdx = [
                                i for i in block.channel_indexes
                                if chanLabel.encode() in i.channel_names][0]
                            # first time at this unit, rename it
                            unit.channel_index = chanIdx
                            chanIdx.units.append(unit)
                            unit.name = '{}#{}'.format(chanIdx.name, unitId)

                    st.name = 'seg{}_{}'.format(segIdx, unit.name)

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

            if eventInfo is not None:
                #  process trial related events
                analogData = []
                for key, value in eventInfo['inputIDs'].items():
                    searchName = 'seg{}_'.format(segIdx) + value
                    ainpAsig = seg.filter(
                        objects=AnalogSignalProxy,
                        name=searchName)[0]
                    ainpData = ainpAsig.load(
                        time_slice=(tStart, tStop),
                        magnitude_mode='rescaled')
                    analogData.append(
                        pd.DataFrame(ainpData.magnitude, columns=[key]))
                    del ainpData
                    gc.collect()
                motorData = pd.concat(analogData, axis=1)
                del analogData
                gc.collect()
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
                    gc.collect()
                _, trialEvents = mea.getTrialsNew(
                    motorData, ainpAsig.sampling_rate.magnitude,
                    tStart, trialType=None)
                trialEvents.fillna(0)
                trialEvents.rename(
                    columns={
                        'Label': 'rig_property',
                        'Details': 'rig_value'},
                    inplace=True)
                del motorData
                gc.collect()
                eventList = eventDataFrameToEvents(
                    trialEvents,
                    idxT='Time',
                    annCol=['rig_property', 'rig_value'])
                for event in eventList:
                    if trackMemory:
                        print('writing motor events memory usage: {}'.format(
                            hf.memory_usage_psutil()))
                    event.segment = newSeg
                    newSeg.events.append(event)
                    newSeg.create_relationship()
                    # write out to file
                    event = writer._write_event(event, nixblock, nixgroup)
                    del event
                    gc.collect()
                del trialEvents, eventList

            for eventProxy in seg.events:
                event = eventProxy.load(
                    time_slice=(tStart, tStop))
                event.segment = newSeg
                newSeg.events.append(event)
                newSeg.create_relationship()
                # write out to file
                event = writer._write_event(event, nixblock, nixgroup)
                del event
                gc.collect()

            for epochProxy in seg.epochs:
                epoch = epochProxy.load(
                    time_slice=(tStart, tStop))
                epoch.segment = newSeg
                newSeg.events.append(epoch)
                newSeg.create_relationship()
                # write out to file
                epoch = writer._write_epoch(epoch, nixblock, nixgroup)
                del epoch
                gc.collect()

    # descend into ChannelIndexes
    for chanIdx in block.channel_indexes:
        if chanIdx.analogsignals or chanIdx.units:
            chanIdx = writer._write_channelindex(chanIdx, nixblock)
    writer._create_source_links(block, nixblock)
    return idx


def preproc(
        fileName='Trial001',
        folderPath='./',
        fillOverflow=True, removeJumps=True,
        eventInfo=None,
        spikeSource='',
        chunkSize=1800, writeMode='rw',
        signal_group_mode='split-all', trialInfo=None
        ):
    #  base file name
    trialBasePath = os.path.join(folderPath, fileName)
    #  instantiate reader, get metadata
    reader = neo.io.BlackrockIO(
        filename=trialBasePath, nsx_to_load=5)
    reader.parse_header()
    metadata = reader.header
    #  instantiate spike reader if requested
    if spikeSource == 'tdc':
        spikePath = os.path.join(
            folderPath, 'tdc_' + fileName,
            'tdc_' + fileName + '.nix')
        spikeReader = neo.io.nixio_fr.NixIO(filename=spikePath)
    else:
        spikeReader = None

    #  instantiate writer
    writer = neo.io.NixIO(
        filename=trialBasePath + '.nix', mode=writeMode)
    #  absolute section index
    idx = 0
    for blkIdx in range(metadata['nb_block']):
        #  blkIdx = 0
        block = readBlockFixNames(
            reader,
            block_index=blkIdx, lazy=True,
            signal_group_mode=signal_group_mode)
        if spikeReader is not None:
            spikeBlock = readBlockFixNames(
                spikeReader, block_index=blkIdx, lazy=True,
                signal_group_mode=signal_group_mode)
            spikeBlock = purgeNixAnn(spikeBlock)
        else:
            spikeBlock = None

        idx = blockToNix(
            block, writer, chunkSize,
            segInitIdx=idx,
            fillOverflow=fillOverflow,
            removeJumps=removeJumps,
            eventInfo=eventInfo,
            spikeSource=spikeSource,
            spikeBlock=spikeBlock
            )

    writer.close()

    return neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix')


def purgeNixAnn(
        block, annNames=['nix_name', 'neo_name']):
    for annName in annNames:
        block.annotations.pop(annName, None)
    for child in block.children_recur:
        if child.annotations:
            child.annotations = {
                k: v
                for k, v in child.annotations.items()
                if k not in annNames}
    for child in block.data_children_recur:
        if child.annotations:
            child.annotations = {
                k: v
                for k, v in child.annotations.items()
                if k not in annNames}
    return block


def loadContainerArrayAnn(
        container=None, trainList=None
        ):
    assert (container is not None) or (trainList is not None)

    spikesAndEvents = []
    returnObj = []
    if container is not None:
        #  need the line below! (RD: don't remember why, consider removing)
        container.create_relationship()
        
        spikesAndEvents += (
            container.filter(objects=SpikeTrain) +
            container.filter(objects=Event)
            )
        returnObj.append(container)
    if trainList is not None:
        spikesAndEvents += trainList
        returnObj.append(trainList)
    
    if len(returnObj) == 1:
        returnObj = returnObj[0]
    else:
        returnObj = tuple(returnObj)

    for st in spikesAndEvents:
        if 'arrayAnnNames' in st.annotations.keys():
            #  pdb.set_trace()
            if isinstance(st.annotations['arrayAnnNames'], str):
                st.annotations['arrayAnnNames'] = [st.annotations['arrayAnnNames']]
            for key in st.annotations['arrayAnnNames']:
                #  fromRaw, the ann come back as tuple, need to recast
                try:
                    if len(st.times) == 1:
                        st.annotations[key] = [st.annotations[key]]
                    st.array_annotations.update(
                        {key: np.array(st.annotations[key])})
                except Exception:
                    traceback.print_exc()
                    #pdb.set_trace()
        if hasattr(st, 'waveforms'):
            if st.waveforms is None:
                st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
            elif not len(st.waveforms):
                st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
    return returnObj


def loadWithArrayAnn(
        dataPath, fromRaw=False):
    if fromRaw:
        reader = neo.io.nixio_fr.NixIO(filename=dataPath)
        block = readBlockFixNames(reader, lazy=False)
    else:
        reader = neo.io.NixIO(filename=dataPath)
        block = reader.read_block()
    
    #  block.create_relationship()  # need this!

    block = loadContainerArrayAnn(container=block)
    
    if fromRaw:
        reader.file.close()
    else:
        reader.close()
    return block


def calcBinarizedArray(
        dataBlock, samplingRate, binnedSpikePath, saveToFile=True):
    
    spikeMatBlock = Block(name=dataBlock.name + ' binarized')
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
        newSeg = Segment(name='seg{}_'.format(segIdx))
        newSeg.merge_annotations(seg)
        spikeMatBlock.segments.append(newSeg)
        
        #  tStart = dataBlock.segments[0].t_start
        #  tStop = dataBlock.segments[0].t_stop
        tStart = seg.t_start
        tStop = seg.t_stop

        # make dummy binary spike train, in case ths chan didn't fire
        segSpikeTrains = [
            i for i in seg.filter(objects=SpikeTrain) if '#' in i.name]
        
        dummyBin = binarize(
            segSpikeTrains[0],
            sampling_rate=samplingRate,
            t_start=tStart,
            t_stop=tStop) * 0
        for chanIdx in spikeMatBlock.channel_indexes:
            #  print(chanIdx.name)
            stList = seg.filter(
                objects=SpikeTrain,
                name='seg{}_{}'.format(segIdx, chanIdx.name)
                )
            if len(stList):
                st = stList[0]
                print('binarizing {}'.format(st.name))
                stBin = binarize(
                    st,
                    sampling_rate=samplingRate,
                    t_start=tStart,
                    t_stop=tStop)
                spikeMatBlock.segments[segIdx].spiketrains.append(st)
                #  to do: link st to spikematblock's chidx and units
            else:
                print('{} has no spikes'.format(st.name))
                stBin = dummyBin

            asig = AnalogSignal(
                stBin * samplingRate,
                name='seg{}_{}_raster'.format(segIdx, st.unit.name),
                sampling_rate=samplingRate,
                dtype=np.int,
                **st.annotations)
            asig.t_start = tStart
            asig.annotate(binWidth=1 / samplingRate.magnitude)
            chanIdx.analogsignals.append(asig)
            asig.channel_index = chanIdx
            spikeMatBlock.segments[segIdx].analogsignals.append(asig)
    
    for chanIdx in spikeMatBlock.channel_indexes:
        chanIdx.name = chanIdx.name + '_raster'

    spikeMatBlock.create_relationship()
    spikeMatBlock = purgeNixAnn(spikeMatBlock)
    if saveToFile:
        writer = neo.io.NixIO(filename=binnedSpikePath)
        writer.write_block(spikeMatBlock, use_obj_names=True)
        writer.close()
    return spikeMatBlock
