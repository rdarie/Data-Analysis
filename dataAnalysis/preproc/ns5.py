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
        filePath, elecIds, startTime_s,
        dataLength_s, spikeStruct, downsample=1):

    elecCorrespondence = spikeStruct.loc[elecIds, 'nevID']
    elecCorrespondence = elecCorrespondence.astype(int)
    #  Open file and extract headers
    reader = neo.io.BlackrockIO(filename=filePath)

    nsx_file = NsxFile(filePath)
    #  Extract data - note: data will be returned based on *SORTED* nevIds, see cont_data['elec_ids']
    #  pdb.set_trace()
    channelData = nsx_file.getdata(
        list(elecCorrespondence.values), startTime_s, dataLength_s, downsample)

    rowIndex = range(
        int(channelData['start_time_s'] * channelData['samp_per_s']),
        int((channelData['start_time_s'] + channelData['data_time_s']) *
            channelData['samp_per_s']))
    channelData['data'] = pd.DataFrame(
        channelData['data'].transpose(),
        index=rowIndex, columns=elecCorrespondence.sort_values().index.values)

    channelData['t'] = channelData['start_time_s'] + np.arange(channelData['data'].shape[0]) / channelData['samp_per_s']
    channelData['t'] = pd.Series(
        channelData['t'], index=channelData['data'].index)
    channelData['badData'] = {}
    channelData['basic_headers'] = nsx_file.basic_header
    channelData['extended_headers'] = nsx_file.extended_headers
    # Close the nsx file now that all data is out
    nsx_file.close()
    return channelData


def preproc(
        fileName='Trial001',
        folderPath='./',
        elecIds=range(1, 97), startTimeS=0, dataTimeS='all',
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
        #  load block metadata only
        #  blockMeta = reader.read_block(block_index=blkIdx, lazy=True)
        #  make a copy and strip it of placeholders
        #  blockMeta.segments = []
        #  blockMeta.channel_indexes = []
        #  writer.write_block(blockMeta)
        
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
                segMeta.index = idx
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
                #  actually do the preprocessing on the chunkData
                
                segMeta.spiketrains = []
                segMeta.epochs = []
                segMeta.events = []
                assert not segMeta.irregularlysampledsignals

                #  replace analogsignal proxies with the chunk data
                for aIdx, aSigProxy in enumerate(segMeta.analogsignals):
                    aSig = AnalogSignal(
                        chunkData[:, aSigProxy.channel_index.index],
                        units=aSigProxy.units, dtype=aSigProxy.dtype,
                        t_start=aSigProxy.t_start + curChunk * chunkSize * s,
                        sampling_rate=aSigProxy.sampling_rate,
                        sampling_period=aSigProxy.sampling_period,
                        name=aSigProxy.name, file_origin=aSigProxy.file_origin,
                        description=aSigProxy.description,
                        array_annotations=aSigProxy.array_annotations,
                        **aSigProxy.annotations)
                    #  fix the channel index references
                    aSigProxy.channel_index = aSig.channel_index
                    #  save the data
                    segMeta.analogsignals[aIdx] = aSig
                idx += 1
                writer.write_block(blockMeta)
    return writer

def preprocSection(
    fileName='Trial001',
    folderPath = './',
    elecIds = range(1, 97), startTimeS = 0, dataTimeS = 900,
    chunkSize = 900,
    curSection = 0, sectionsTotal = 1,
    fillOverflow = False, removeJumps = True):

    filePath     = os.path.join(folderPath, fileName + '.h5')
    timeSection  = hf.getNSxData(filePath, elecIds, startTimeS, dataTimeS)

    origDataPath = os.path.join(folderPath, 'dataAnalysisPreproc',
        fileName + '_orig_%dof%d.npy' % (curSection + 1, sectionsTotal))
    np.save(origDataPath, timeSection['data'].values)
    badData = {}

    if fillOverflow:
        # fill in overflow:
        timeSection['data'], overflowMask = hf.fillInOverflow(
            timeSection['data'], fillMethod = 'average')
        badData.update({'overflow': overflowMask})

    if removeJumps:
        # find unusual jumps in derivative or amplitude
        timeSection['data'], newBadData = hf.fillInJumps(timeSection['data'],
            timeSection['samp_per_s'], smoothing_ms = 0.5, nStdDiff = 50,
            nStdAmp = 100)
        badData.update(newBadData)

    timeSection['badData'] = badData

    print('Saving clean data')
    #
    dataPath = os.path.join(folderPath, 'dataAnalysisPreproc',
        fileName + '_clean.h5')
    with h5py.File(dataPath, "a") as f:
        #  print(timeSection['data'])
        #  pdb.set_trace()
        sectionIndex = slice(
            int(curSection * chunkSize * timeSection['samp_per_s']),
            int((curSection * chunkSize + dataTimeS) * timeSection['samp_per_s'])
            )
        f['data'][sectionIndex, :] = timeSection['data'].values
        f['index'][sectionIndex] = timeSection['data'].index
        f['t'][sectionIndex] = timeSection['t'].values

    del timeSection['data'], timeSection['t']

    metaDataPath = os.path.join(folderPath, 'dataAnalysisPreproc',
        fileName + '_clean_metadata_%dof%d.pickle' % (curSection + 1, sectionsTotal))

    with open(metaDataPath, "wb" ) as f:
        pickle.dump(timeSection, f, protocol=4 )

    print('Done cleaning data')
    return section

def preprocNs5Spectrum(stepLen_s = 0.05, winLen_s = 0.1,
    fr_start = 5, fr_stop = 1000):
    pass


def loadTimeSeries(
        filePath=None, elecIds=None,
        startTime_s=None, dataLength_s=None):
    channelData = None
    return channelData