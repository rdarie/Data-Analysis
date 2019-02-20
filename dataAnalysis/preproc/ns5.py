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


def getNSxData(
        fileName=None,
        folderPath=None,
        reader=None, elecIds=None, startTime_s=None,
        dataLength_s=None, downsample=1):

    #  Open file and extract headers
    if reader is None:
        assert (fileName is not None) and (folderPath is not None)
        filePath = os.path.join(folderPath, fileName) + '.nix'
        reader = neo.io.nixio.NixIO(filename=filePath)
    
    idx = 0
    reader.parse_header()
    metadata = reader.header
    
    #  pdb.set_trace()
    for blkIdx in range(metadata['nb_block']):
        #  blkIdx = 0
        block = reader.read_block(
            block_index=blkIdx, lazy=True,
            signal_group_mode='group-by-same-units')
        for segIdx in range(metadata['nb_segment'][blkIdx]):
            #  segIdx = 0
            #  you will get an error with t_start unless you load events
            seg = block.segments[segIdx]
            seg.events = [i.load() for i in seg.events]
    testPlot = False
    if testPlot:
        data = seg.analogsignals[0].load(channel_indexes=[8])
        plt.plot(data.magnitude)
        plt.show()
    channelData = None
    return reader


def blockToNix(
        block, writer, chunkSize,
        segInitIdx,
        fillOverflow=False,
        removeJumps=False
        ):
    idx = segInitIdx
    #  prune out nev spike placeholders
    for chanIdx in block.channel_indexes:
        if chanIdx.units:
            for unit in chanIdx.units:
                #  print(unit.spiketrains)
                if unit.spiketrains:
                    for st in unit.spiketrains:
                        del st
                    unit.spiketrains = []
                del unit
            chanIdx.units = []
            #  chanIdx.name += ' ' + chanIdx.units[0].name
    block.channel_indexes = (
        [chanIdx for chanIdx in block.channel_indexes if (
            chanIdx.analogsignals)])
    #  delete asig proxies from channel index list
    for metaIdx, chanIdx in enumerate(block.channel_indexes):
        chanIdx.analogsignals = []
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
            newSeg.events = [
                i.load(time_slice=(tStart, tStop)) for i in seg.events]
            newSeg.epochs = [
                i.load(time_slice=(tStart, tStop)) for i in seg.epochs]
                
            newSegList.append(newSeg)
            segParents[segIdx].update(
                {chunkIdx: newSegList.index(newSeg)})
            idx += 1
    block.segments = newSegList
    #  print(block.children_recur)
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
            
            for aSigIdx, aSigProxy in enumerate(seg.analogsignals):
                chanIdx = aSigProxy.channel_index
                #  print(chanIdx)
                a = aSigProxy.load(
                    time_slice=(tStart, tStop))
                #  link AnalogSignal and ID providing channel_index
                a.channel_index = chanIdx
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

                chanIdx.analogsignals.append(a)
                chanIdx.create_relationship()
                newSeg.analogsignals.append(a)
                newSeg.create_relationship()

            for iaSigIdx, iaSigProxy in enumerate(
                    seg.irregularlysampledsignals):
                chanIdx = iaSigProxy.channel_index
                tStart = chunkIdx * actualChunkSize
                tStop = (chunkIdx + 1) * actualChunkSize
                ia = iaSigProxy.load(
                    time_slice=(tStart, tStop))
                #  link AnalogSignal and ID providing channel_index
                ia.channel_index = chanIdx
                chanIdx.irregularlysampledsignals.append(ia)
                newSeg.irregularlysampledsignals.append(ia)
            #  print(newSeg.children_recur)
            newSeg = writer._write_segment(newSeg, nixblock)
    # descend into ChannelIndexes
    for chanIdx in block.channel_indexes:
        chanIdx = writer._write_channelindex(chanIdx, nixblock)
    writer._create_source_links(block, nixblock)
    return idx


def preproc(
        fileName='Trial001',
        folderPath='./',
        fillOverflow=True, removeJumps=True,
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
            removeJumps=removeJumps
            )

    writer.close()

    channelData = getNSxData(
        reader=neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix'),
        elecIds=None, startTime_s=None,
        dataLength_s=None, downsample=1)
    
    return channelData

def loadTimeSeries(
        filePath=None, elecIds=None,
        startTime_s=None, dataLength_s=None):
    channelData = None
    return channelData
