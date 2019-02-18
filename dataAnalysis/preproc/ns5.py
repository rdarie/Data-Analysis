# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""
import neo
import matplotlib, math, pdb
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


def getNSxData(
        filePath, elecIds, startTime_s,
        dataLength_s, spikeStruct, downsample=1):

    elecCorrespondence = spikeStruct.loc[elecIds, 'nevID']
    elecCorrespondence = elecCorrespondence.astype(int)
    #  Open file and extract headers
    reader = neo.io.BlackrockIO(filename=filePath)
    #  read the blocks
    blks = reader.read(lazy=False)
    
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
    chunkSize=900
    ):

    filePath = os.path.join(folderPath, fileName) + '.ns5'
    dummyChannelData = hf.getNSxData(filePath, elecIds[0], startTimeS, dataTimeS)

    if dataTimeS == 'all':
        dataTimeS = dummyChannelData['data_time_s']
        print('Recording is %4.2f seconds long' % dataTimeS)

    nSamples = int(dataTimeS * dummyChannelData['samp_per_s'])
    nChannels = len(elecIds)
    nChunks = dataTimeS // chunkSize
    # add a chunk if division isn't perfect
    if dataTimeS / chunkSize > dataTimeS // chunkSize:
        nChunks += 1
    nChunks = int(nChunks)

    if not os.path.exists(os.path.join(folderPath, 'dataAnalysisPreproc')):
        os.makedirs(os.path.join(folderPath, 'dataAnalysisPreproc'))

    dataPath = os.path.join(
        folderPath, 'dataAnalysisPreproc',
        fileName + '_clean.h5')

    with h5py.File(dataPath, "w") as f:
        #pdb.set_trace()
        f.create_dataset("data", (nSamples, nChannels), dtype='float32',
            chunks=True)
        f.create_dataset("channels", (nChannels,), data=list(elecIds),
            dtype='int32')
        f.create_dataset("index", (nSamples,), dtype='int32')
        f.create_dataset("t", (nSamples,), dtype='float32')

    for curSection in range(nChunks):
        print('PreprocNs5: starting chunk %d of %d' % (curSection + 1, nChunks))

        if curSection == nChunks - 1:
            thisDataTime = dataTimeS - chunkSize * curSection
        else:
            thisDataTime = chunkSize

        preprocSection(
            fileName = fileName,
            folderPath = folderPath,
            elecIds = elecIds, startTimeS = startTimeS + curSection * chunkSize,
            dataTimeS = thisDataTime,
            chunkSize = chunkSize,
            curSection = curSection, sectionsTotal = nChunks,
            fillOverflow = fillOverflow, removeJumps = removeJumps)
    return channelData

def preprocSection(
    fileName = 'Trial001',
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