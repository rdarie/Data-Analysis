# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""

import matplotlib, math, pdb
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.helperFunctions.motor_encoder as mea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import pickle
from copy import *
import argparse
import h5py
from scipy import signal

def preprocNs5(
    fileName = 'Trial001',
    folderPath = './',
    elecIds = range(1, 97), startTimeS = 0, dataTimeS = 'all',
    fillOverflow = True, removeJumps = True,
    chunkSize = 900
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

    dataPath = os.path.join(folderPath, 'dataAnalysisPreproc',
        fileName + '_clean.h5')

    with h5py.File(dataPath, "w") as f:
        #pdb.set_trace()

        f.create_dataset("data", (nSamples,nChannels), dtype='float32',
            chunks=True)
        f.create_dataset("channels", (nChannels,), data = list(elecIds),
            dtype='int32')
        f.create_dataset("index", (nSamples,), dtype='int32')
        f.create_dataset("t", (nSamples,), dtype='float32')

    for curSection in range(nChunks):
        print('PreprocNs5: starting chunk %d of %d' % (curSection + 1, nChunks))

        if curSection == nChunks - 1:
            thisDataTime = dataTimeS - chunkSize * curSection
        else:
            thisDataTime = chunkSize

        preprocNs5Section(
            fileName = fileName,
            folderPath = folderPath,
            elecIds = elecIds, startTimeS = startTimeS + curSection * chunkSize,
            dataTimeS = thisDataTime,
            chunkSize = chunkSize,
            curSection = curSection, sectionsTotal = nChunks,
            fillOverflow = fillOverflow, removeJumps = removeJumps)

def preprocNs5Section(
    fileName = 'Trial001',
    folderPath = './',
    elecIds = range(1, 97), startTimeS = 0, dataTimeS = 900,
    chunkSize = 900,
    curSection = 0, sectionsTotal = 1,
    fillOverflow = False, removeJumps = True):

    filePath     = os.path.join(folderPath, fileName +'.ns5')
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
        #print(timeSection['data'])
        #pdb.set_trace()
        sectionIndex = slice(int(curSection * chunkSize * timeSection['samp_per_s']),
            int((curSection * chunkSize + dataTimeS) * timeSection['samp_per_s']))
        f['data'][sectionIndex, :] = timeSection['data'].values
        f['index'][sectionIndex] = timeSection['data'].index
        f['t'][sectionIndex] = timeSection['t'].values

    del timeSection['data'], timeSection['t']

    metaDataPath = os.path.join(folderPath, 'dataAnalysisPreproc',
        fileName + '_clean_metadata_%dof%d.pickle' % (curSection + 1, sectionsTotal))

    with open(metaDataPath, "wb" ) as f:
        pickle.dump(timeSection, f, protocol=4 )

    print('Done cleaning data')

def preprocOpenEphysFolder(
    folderPath,
    chIds = 'all', adcIds = 'all',
    chanNames = None,
    notchFreq = 60, notchWidth = 5, notchOrder = 2, nNotchHarmonics = 1,
    highPassFreq = None, highPassOrder = 2,
    lowPassFreq = None, lowPassOrder = 2,
    startTimeS = 0, dataTimeS = 900,
    chunkSize = 900,
    curSection = 0, sectionsTotal = 1,
    fillOverflow = False, removeJumps = True):

    channelData = hf.getOpenEphysFolder(folderPath, adcIds = adcIds, chanNames = chanNames)
    cleanData = preprocOpenEphys(channelData,
        notchFreq = notchFreq, notchWidth = notchWidth, notchOrder = notchOrder, nNotchHarmonics = nNotchHarmonics,
        highPassFreq = highPassFreq, highPassOrder = highPassOrder,
        lowPassFreq = lowPassFreq, lowPassOrder = lowPassOrder,
        startTimeS =startTimeS, dataTimeS = dataTimeS,
        chunkSize = chunkSize ,
        curSection = curSection, sectionsTotal = sectionsTotal,
        fillOverflow = fillOverflow, removeJumps = removeJumps)

    nSamples = channelData['data'].shape[0]
    nChannels = channelData['data'].shape[1]
    isChan = np.logical_not(channelData['basic_headers']['isADC'])

    with h5py.File(os.path.join(folderPath, 'processed.h5'), "w") as f:
        f.create_dataset("data", data = channelData['data'].values, dtype='float32',
            chunks=True)
        f.create_dataset("cleanData", data = cleanData.values, dtype='float32')
        f.create_dataset("t", data = channelData['t'].values, dtype='float32')

    with open(os.path.join(folderPath, 'metadata.p'), "wb" ) as f:
        pickle.dump(channelData['basic_headers'], f, protocol=4 )

    return channelData, cleanData

def preprocOpenEphys(channelData,
    notchFreq = 60, notchWidth = 5, notchOrder = 2, nNotchHarmonics = 1,
    highPassFreq = None, highPassOrder = 2,
    lowPassFreq = None, lowPassOrder = 2,
    startTimeS = 0, dataTimeS = 900,
    chunkSize = 900,
    curSection = 0, sectionsTotal = 1,
    fillOverflow = False, removeJumps = True):

    isChan = np.logical_not(channelData['basic_headers']['isADC'])
    cleanData = channelData['data'].loc[:,isChan]

    if notchFreq is not None:
        for harmonicOrder in range(1, nNotchHarmonics + 1):
            print('notch filtering harmonic order: {}'.format(harmonicOrder))
            notchLowerBound = harmonicOrder * notchFreq - notchWidth / 2
            notchUpperBound = harmonicOrder * notchFreq + notchWidth / 2

            y, x = signal.iirfilter(notchOrder,
                (2 * notchLowerBound / channelData['samp_per_s'],
                2 * notchUpperBound / channelData['samp_per_s']), rp=1, rs=50, btype = 'bandstop', ftype='ellip')

            def filterFun(sig):
                return signal.filtfilt(y, x, sig, method="gust")

            cleanData = cleanData.apply(filterFun)

    if highPassFreq is not None:
        print('high pass filtering: {}'.format(highPassFreq))
        y, x = signal.iirfilter(highPassOrder,
            2 * highPassFreq / channelData['samp_per_s'],
            rp=1, rs=50, btype = 'high', ftype='ellip')

        def filterFun(sig):
            return signal.filtfilt(y, x, sig, method="gust")

        cleanData = cleanData.apply(filterFun)

    if lowPassFreq is not None:
        print('low pass filtering: {}'.format(lowPassFreq))
        y, x = signal.iirfilter(lowPassOrder,
            2 * lowPassFreq / channelData['samp_per_s'],
            rp=1, rs=50, btype = 'low', ftype='ellip')

        def filterFun(sig):
            return signal.filtfilt(y, x, sig, method="gust")

        cleanData = cleanData.apply(filterFun)

    print('Done cleaning Data')
    return cleanData

def preprocNs5Spectrum(stepLen_s = 0.05, winLen_s = 0.1,
    fr_start = 5, fr_stop = 1000):
    pass
