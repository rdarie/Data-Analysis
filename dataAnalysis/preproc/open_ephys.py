import openEphysAnalysis.OpenEphys as oea
from scipy import signal
import numpy as np
import pandas as pd
import os
import collections
import pdb
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from neo.io import OpenEphysIO, NixIO, nixio_fr
import quantities as pq
import dataAnalysis.preproc.ns5 as ns5


def openEphysFolderToNixBlock(
        folderPath, chIds='all', adcIds='all', chanNames=None,
        startTime_s=0, dataLength_s='all', downsample=1, ):
    
    dataReader = OpenEphysIO(folderPath)
    
    dataBlock = ns5.readBlockFixNames(dataReader, lazy=False)
    dataBlock.name = os.path.basename(folderPath)
    dataBlock.annotate(nix_name=dataBlock.name)

    for segIdx, seg in enumerate(dataBlock.segments):
        seg.name = 'seg{}_{}'.format(segIdx, dataBlock.name)
        seg.annotate(nix_name=seg.name)
    for chanIdx in dataBlock.channel_indexes:
        chanIdx.annotate(nix_name=chanIdx.name)
        if not chIds == 'all':
            if not isinstance(chIds, collections.Iterable):
                chIds = [chIds]
            if (chanIdx.name not in chIds) and ('CH' in chanIdx.name):
                # TODO: delete unwanted
                pass
        if not adcIds == 'all':
            if not isinstance(adcIds, collections.Iterable):
                adcIds = [adcIds]
            if (chanIdx.name not in adcIds) and ('ADC' in chanIdx.name):
                # TODO: delete unwanted
                pass
        if chanIdx.name in chanNames:
            chanIdx.annotate(label=chanNames[chanIdx.name])
        else:
            chanIdx.annotate(label='NA')
        # TODO: implement startime end time
    return dataBlock


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
                folderPath, source + '_' + chprefix + '1.continuous')
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

    rawBlock = openEphysFolderToNixBlock(
        folderPath, adcIds=adcIds, chanNames=chanNames)
    outputPath = os.path.join(folderPath, rawBlock.name + '.nix')
    writer = NixIO(filename=outputPath)
    writer.write_block(rawBlock, use_obj_names=True)
    writer.close()

    rawReader = nixio_fr.NixIO(filename=outputPath)
    cleanDataBlock = ns5.readBlockFixNames(rawReader, lazy=False)
    cleanDataBlock = preprocOpenEphysBlock(
        cleanDataBlock,
        notchFreq=notchFreq, notchWidth=notchWidth,
        notchOrder=notchOrder, nNotchHarmonics=nNotchHarmonics,
        highPassFreq=highPassFreq, highPassOrder=highPassOrder,
        lowPassFreq=lowPassFreq, lowPassOrder=lowPassOrder,
        startTimeS=startTimeS, dataTimeS=dataTimeS,
        chunkSize=chunkSize,
        curSection=curSection, sectionsTotal=sectionsTotal,
        fillOverflow=fillOverflow, removeJumps=removeJumps)
    
    cleanDataBlock = ns5.purgeNixAnn(cleanDataBlock)
    outputPath = os.path.join(folderPath, rawBlock.name + '_filtered.nix')
    writer = NixIO(filename=outputPath)
    writer.write_block(cleanDataBlock, use_obj_names=True)
    writer.close()
    return


def preprocOpenEphysBlock(
    dataBlock,
    notchFreq=60, notchWidth=5, notchOrder=2,
    nNotchHarmonics=1,
    highPassFreq=None, highPassOrder=2,
    lowPassFreq=None, lowPassOrder=2,
    startTimeS=0, dataTimeS=900,
    chunkSize=900,
    curSection=0, sectionsTotal=1,
    fillOverflow=False, removeJumps=True):

    bTot = np.array([])
    aTot = np.array([])
    samplingRate = dataBlock.filter(objects=AnalogSignal)[0].sampling_rate.magnitude
    if notchFreq is not None:
        for harmonicOrder in range(1, nNotchHarmonics + 1):
            # print('notch filtering harmonic order: {}'.format(harmonicOrder))
            notchLowerBound = harmonicOrder * notchFreq - notchWidth / 2
            notchUpperBound = harmonicOrder * notchFreq + notchWidth / 2

            bNotch, aNotch = signal.iirfilter(
                notchOrder,
                (
                    2 * notchLowerBound / samplingRate,
                    2 * notchUpperBound / samplingRate),
                rp=1, rs=50, btype='bandstop', ftype='ellip')
            bTot = np.concatenate((bTot, bNotch))
            aTot = np.concatenate((aTot, aNotch))

    if highPassFreq is not None:
        bHigh, aHigh = signal.iirfilter(
            highPassOrder,
            2 * highPassFreq / samplingRate,
            rp=1, rs=50, btype='high', ftype='ellip')
        bTot = np.concatenate((bTot, bHigh))
        aTot = np.concatenate((aTot, aHigh))

    if lowPassFreq is not None:
        bLow, aLow = signal.iirfilter(
            lowPassOrder,
            2 * lowPassFreq / samplingRate,
            rp=1, rs=50, btype='low', ftype='ellip')
        bTot = np.concatenate((bTot, bLow))
        aTot = np.concatenate((aTot, aLow))

    def filterFun(sig, b=None, a=None):
        return signal.filtfilt(b, a, sig, method="gust")

    for chanIdx in dataBlock.channel_indexes:
        if 'CH' not in chanIdx.name:
            continue
        for segIdx, seg in enumerate(dataBlock.segments):
            asigList = [a for a in chanIdx.filter(objects=AnalogSignal) if a.segment is seg]
            assert len(asigList) == 1
            asig = asigList[0]
            asig[:] = filterFun(asig, b=bTot, a=aTot) * asig.units
    return dataBlock


def preprocOpenEphys(channelData,
    notchFreq=60, notchWidth=5, notchOrder=2,
    nNotchHarmonics=1,
    highPassFreq=None, highPassOrder=2,
    lowPassFreq=None, lowPassOrder=2,
    startTimeS=0, dataTimeS=900,
    chunkSize=900,
    curSection=0, sectionsTotal=1,
    fillOverflow=False, removeJumps=True):

    isChan = np.logical_not(channelData['basic_headers']['isADC'])
    cleanData = channelData['data'].loc[:, isChan]

    if notchFreq is not None:
        for harmonicOrder in range(1, nNotchHarmonics + 1):
            print('notch filtering harmonic order: {}'.format(harmonicOrder))
            notchLowerBound = harmonicOrder * notchFreq - notchWidth / 2
            notchUpperBound = harmonicOrder * notchFreq + notchWidth / 2

            y, x = signal.iirfilter(
                notchOrder,
                (
                    2 * notchLowerBound / channelData['samp_per_s'],
                    2 * notchUpperBound / channelData['samp_per_s']),
                rp=1, rs=50, btype='bandstop', ftype='ellip')

            def filterFun(sig):
                return signal.filtfilt(y, x, sig, method="gust")

            cleanData = cleanData.apply(filterFun)

    if highPassFreq is not None:
        print('high pass filtering: {}'.format(highPassFreq))
        y, x = signal.iirfilter(highPassOrder,
            2 * highPassFreq / channelData['samp_per_s'],
            rp=1, rs=50, btype='high', ftype='ellip')

        def filterFun(sig):
            return signal.filtfilt(y, x, sig, method="gust")

        cleanData = cleanData.apply(filterFun)

    if lowPassFreq is not None:
        print('low pass filtering: {}'.format(lowPassFreq))
        y, x = signal.iirfilter(lowPassOrder,
            2 * lowPassFreq / channelData['samp_per_s'],
            rp=1, rs=50, btype='low', ftype='ellip')

        def filterFun(sig):
            return signal.filtfilt(y, x, sig, method="gust")

        cleanData = cleanData.apply(filterFun)

    print('Done cleaning Data')
    cleanData = pd.concat([cleanData, channelData['data'].loc[:, channelData['basic_headers']['isADC']]], axis='columns')
    return cleanData