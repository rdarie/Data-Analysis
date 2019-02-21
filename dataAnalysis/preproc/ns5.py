# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""
import neo
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal)
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
import argparse
import h5py
from scipy import signal
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable




def dataFrameToNeo(
        df,
        idxT='NSPTime',
        probeName='insTD', samplingRate=500*pq.Hz,
        timeUnits=pq.s, measureUnits=pq.mV,
        dataCol=['channel_0', 'channel_1'],
        useColNames=False, forceColNames=None, namePrefix=''):
    block = Block()
    seg = Segment(name=probeName + ' segment')
    block.segments.append(seg)

    for idx, colName in enumerate(dataCol):
        if useColNames:
            chanName = namePrefix + colName
        elif forceColNames is not None:
            chanName = namePrefix + forceColNames[idx]
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
    block.create_relationship()
    seg.create_relationship()
    return block


def addBlockToNIX(
        newBlock, segIdx=0,
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
    
    writer = neo.io.NixIO(filename=trialBasePath + '.nix')
    nixblock = writer.nix_file.blocks[nixBlockIdx]
    nixblockName = nixblock.name
    newBlock.annotate(nix_name=nixblockName)

    #  for idx, segIdx in enumerate(segsWriteTo.index):
    nixgroup = nixblock.groups[nixSegIdx]
    nixSegName = nixgroup.name
    newBlock.segments[segIdx].annotate(nix_name=nixSegName)
    #  TODO: double check that you can't add the same thing twice
    for asig in newBlock.segments[segIdx].analogsignals:
        #  asig.annotations.update({
        #      'channel_ids': asig.channel_index.channel_ids + maxIdx+1
        #      })
        assert asig.dtype == forceType
        assert asig.sampling_rate == forceFS
        assert asig.shape[0] == forceShape
        asig = writer._write_analogsignal(asig, nixblock, nixgroup)
    for chanIdx in newBlock.channel_indexes:
        chanIdx.index = chanIdx.index + maxIdx + 1
        chanIdx.channel_ids = chanIdx.index
        chanIdx = writer._write_channelindex(chanIdx, nixblock)
    writer._create_source_links(newBlock, nixblock)
    writer.close()
    return


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
        dataLength_s=None, downsample=1):
    #  Open file and extract headers
    if reader is None:
        assert (fileName is not None) and (folderPath is not None)
        filePath = os.path.join(folderPath, fileName) + '.nix'
        reader = neo.io.nixio_fr.NixIO(filename=filePath)

    block = reader.read_block(
        block_index=blockIdx, lazy=True,
        signal_group_mode='group-by-same-units')

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

            reqData = asig.load(
                time_slice=timeSlice,
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
    reader.file.close()
    return channelData


def blockToNix(
        block, writer, chunkSize,
        segInitIdx,
        pruneOutUnits=True,
        fillOverflow=False,
        removeJumps=False
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
            newSeg = block.segments[segParents[segIdx][chunkIdx]]
            newSeg, nixgroup = writer._write_segment_meta(newSeg, nixblock)
            
            if not pruneOutUnits:
                for stIdx, stProxy in enumerate(seg.spiketrains):
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
                    time_slice=(tStart, tStop),
                    magnitude_mode='rescaled')
                event.segment = newSeg
                newSeg.events.append(event)
                newSeg.create_relationship()
                # write out to file
                event = writer._write_event(event, nixblock, nixgroup)

            for epochProxy in seg.epochs:
                epoch = epochProxy.load(
                    time_slice=(tStart, tStop),
                    magnitude_mode='rescaled')
                epoch.segment = newSeg
                newSeg.events.append(epoch)
                newSeg.create_relationship()
                # write out to file
                epoch = writer._write_epoch(epoch, nixblock, nixgroup)

    # descend into ChannelIndexes
    for chanIdx in block.channel_indexes:
        chanIdx = writer._write_channelindex(chanIdx, nixblock)
    writer._create_source_links(block, nixblock)
    return idx


def preproc(
        fileName='Trial001',
        folderPath='./',
        fillOverflow=True, removeJumps=True,
        pruneOutUnits=True,
        chunkSize=1800,
        signal_group_mode='split-all'
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
            pruneOutUnits=pruneOutUnits
            )

    writer.close()

    return reader


def loadTimeSeries(
        filePath=None, elecIds=None,
        startTime_s=None, dataLength_s=None):
    channelData = None
    return channelData
