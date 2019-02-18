import openEphysAnalysis.OpenEphys as oea
from scipy import signal
import numpy as np
import pandas as pd
import os
import collections


def getOpenEphysFolder(
        folderPath, chIds='all', adcIds='all', chanNames=None,
        startTime_s=0, dataLength_s='all', downsample=1):

    # load one file to get the metadata
    source = '100'
    session = '0'
    dummyChFileName = None
    if chIds is not None:
        chprefix = 'CH'
        if chIds == 'all':
            chIds = oea._get_sorted_channels(folderPath, 'CH', session, source)
            dummyChFileName = os.path.join(
                folderPath, source + '_'+chprefix + '1.continuous')
        else:  # already gave a list
            if not isinstance(chIds, collections.Iterable):
                chIds = [chIds]
            dummyChFileName = os.path.join(
                folderPath, source + '_' +
                chprefix + str(chIds[0]) + '.continuous')

        recordingData = oea.loadFolderToArray(folderPath, channels=chIds)
    else:
        recordingData = None

    if adcIds is not None:
        if adcIds == 'all':
            adcIds = oea._get_sorted_channels(
                folderPath, 'ADC', session, source)
            if dummyChFileName is None:
                chprefix = 'ADC'
                dummyChFileName = os.path.join(
                    folderPath, source + '_' +
                    chprefix + '1.continuous')
        else:  # already gave a list
            if not isinstance(adcIds, collections.Iterable):
                adcIds = [adcIds]
            if dummyChFileName is None:
                chprefix = 'ADC'
                dummyChFileName = os.path.join(
                    folderPath, source + '_' +
                    chprefix + str(chIds[0]) + '.continuous')
        adcData = oea.loadFolderToArray(
            folderPath, channels=adcIds, chprefix='ADC')
    else:
        adcData = None

    dummyChannelData = oea.loadContinuous(dummyChFileName)

    t = np.arange(
        dummyChannelData['data'].shape[0]) / (
        float(dummyChannelData['header']['sampleRate']))

    if startTime_s > 0:
        dataMask = t > startTime_s
        t = t[dataMask]
        if adcData is not None:
            adcData = adcData[dataMask, :]
        if recordingData is not None:
            recordingData = recordingData[dataMask, :]

    if dataLength_s == 'all':
        dataLength_s = t[-1]
    else:
        if dataLength_s < t[-1]:
            dataMask = t < dataLength_s
            t = t[dataMask]
            if adcData is not None:
                adcData = adcData[dataMask,:]
            if recordingData is not None:
                recordingData = recordingData[dataMask,:]
        else:
            dataLength_s = t[-1]

    #  pdb.set_trace()
    data = pd.DataFrame(np.concatenate((recordingData, adcData), axis=1), columns=chanNames)
    isADC = np.ones(len(data.columns), dtype=np.bool)
    isADC[:len(chIds)] = False

    channelData = {
        'data' : data,
        't' : pd.Series(t, index=data.index),
        'basic_headers' :dummyChannelData['header'],
        'badData' : {},
        'start_time_s' : startTime_s,
        'data_time_s' : dataLength_s,
        'samp_per_s' : float(dummyChannelData['header']['sampleRate'])
        }

    channelData['basic_headers']['isADC'] = isADC
    channelData['basic_headers']['chanNames'] = chanNames
    return channelData


def preprocOpenEphysFolder(
        folderPath,
        chIds='all', adcIds='all',
        chanNames=None,
        notchFreq=60, notchWidth=5, notchOrder=2, nNotchHarmonics=1,
        highPassFreq=None, highPassOrder=2,
        lowPassFreq=None, lowPassOrder=2,
        startTimeS=0, dataTimeS=900,
        chunkSize=900,
        curSection=0, sectionsTotal=1,
        fillOverflow=False, removeJumps=True):

    channelData = getOpenEphysFolder(folderPath, adcIds=adcIds, chanNames=chanNames)
    cleanData = preprocOpenEphys(
        channelData,
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

            y, x = signal.iirfilter(
                notchOrder,
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