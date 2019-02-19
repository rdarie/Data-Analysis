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
        reader = neo.io.nixio_fr.NixIO(filename=filePath)
    
    idx = 0
    reader.parse_header()
    metadata = reader.header
    #  pdb.set_trace()
    for blkIdx in range(metadata['nb_block']):
        #  blkIdx = 0      
        blockMeta = reader.read_block(block_index=blkIdx, lazy=True)  
        for segIdx in range(metadata['nb_segment'][blkIdx]):
            #  segIdx = 0
            nSamp = reader.get_signal_size(blkIdx, segIdx)
            blockMeta = reader.read_block(block_index=blkIdx, lazy=True)
            segMeta = blockMeta.segments[segIdx]
            print(segMeta.t_start)
    channelData = None
    return channelData


def preproc(
        fileName='Trial001',
        folderPath='./',
        fillOverflow=True, removeJumps=True,
        chunkSize=1800
        ):

    #  base file name
    trialBasePath = os.path.join(folderPath, fileName)
    #  instantiate reader, get metadata
    reader = neo.io.BlackrockIO(
        filename=trialBasePath, nsx_to_load=5)
    reader.parse_header()
    metadata = reader.header
    fs = reader.get_signal_sampling_rate()
    #  instantiate writer
    writer = neo.io.NixIO(filename=trialBasePath + '.nix')
    #  absolute section index
    idx = 0
    for blkIdx in range(metadata['nb_block']):
        #  blkIdx = 0
        blockMeta = reader.read_block(block_index=blkIdx, lazy=True)
        #  prune out nev spike placeholders
        for metaIdx, chanIdx in enumerate(blockMeta.channel_indexes):
            if chanIdx.units:
                for unitIdx, unit in enumerate(chanIdx.units):
                    print(unit.spiketrains)
                    unit.spiketrains = []
                chanIdx.units = []
                #  chanIdx.name += ' ' + chanIdx.units[0].name
        blockMeta.channel_indexes = (
            [chanIdx for chanIdx in blockMeta.channel_indexes if (
                chanIdx.analogsignals)])

        for segIdx in range(metadata['nb_segment'][blkIdx]):
            #  segIdx = 0
            nSamp = reader.get_signal_size(blkIdx, segIdx)
            segLen = nSamp / fs
            nChunks = int(math.ceil(segLen / chunkSize))
            for curChunk in range(nChunks):
                print(
                    'PreprocNs5: starting chunk %d of %d' % (
                        curChunk + 1, nChunks))

                blockMeta = reader.read_block(block_index=blkIdx, lazy=True)
                segMeta = blockMeta.segments[segIdx]
                for metaIdx, chanIdx in enumerate(blockMeta.channel_indexes):
                    if 'Unit' in chanIdx.name:
                        blockMeta.channel_indexes[metaIdx].name += '{}'.format(metaIdx)
                #  see https://neo.readthedocs.io/en/latest/_images/simple_generated_diagram.png
                #  seg is a new object in memory, swap it out
                #  for the metadata placeholder

                i_start = int(curChunk * chunkSize * fs)
                i_stop = i_start + int(chunkSize * fs)
                chunkData = reader.get_analogsignal_chunk(
                    block_index=blkIdx, seg_index=segIdx,
                    i_start=i_start, i_stop=i_stop)
                # recalc i_stop in case there wasn't enough data
                i_stop = i_start + chunkData.shape[0]
                # actualChunkDur = chunkData.shape[0] / fs 
                #  actually do the preprocessing on the chunkData
                if fillOverflow:
                    # fill in overflow:
                    pass
                    '''
                    timeSection['data'], overflowMask = hf.fillInOverflow(
                        timeSection['data'], fillMethod = 'average')
                    #  todo: add epoch showing problematic period
                    '''
                if removeJumps:
                    # find unusual jumps in derivative or amplitude
                    pass
                    '''
                    timeSection['data'], newBadData = hf.fillInJumps(timeSection['data'],
                        timeSection['samp_per_s'], smoothing_ms = 0.5, nStdDiff = 50,
                        nStdAmp = 100)
                    #  todo: add epoch showing problematic period
                    '''
                segMeta.spiketrains = []
                segMeta.epochs = []
                segMeta.events = []
                assert not segMeta.irregularlysampledsignals

                #  replace analogsignal proxies with the chunk data
                for aIdx, aSigProxy in enumerate(segMeta.analogsignals):
                    chanIdx = aSigProxy.channel_index
                    aSig = AnalogSignal(
                        chunkData[:, chanIdx.index],
                        units=aSigProxy.units, dtype=aSigProxy.dtype,
                        t_start=aSigProxy.t_start + curChunk * chunkSize * s,
                        sampling_rate=aSigProxy.sampling_rate,
                        sampling_period=aSigProxy.sampling_period,
                        name=aSigProxy.name, file_origin=aSigProxy.file_origin,
                        description=aSigProxy.description,
                        array_annotations=aSigProxy.array_annotations,
                        **aSigProxy.annotations)

                    #  fix the channel index pointers
                    assert len(chanIdx.analogsignals) == 1
                    chanIdx.analogsignals[0] = aSig
                    #  purge the spiketrains
                    for unit in chanIdx.units:
                        unit.spiketrains = []
                    chanIdx.units = []
                    #  save the data
                    segMeta.analogsignals[aIdx] = aSig
                
                # purge units on remaining channels
                for chanIdx in blockMeta.channel_indexes:
                    for unit in chanIdx.units:
                        unit.spiketrains = []
                        #  print(chanIdx.channel_names)
                    chanIdx.units = []
                
                segMeta.index = idx
                blockMeta.index = idx
                idx += 1
                writer.write_block(blockMeta, use_obj_names=True)

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
