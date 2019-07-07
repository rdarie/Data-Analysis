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
import dataAnalysis.helperFunctions.helper_functions_new as hf
from neo.io import OpenEphysIO, NixIO, nixio_fr
import quantities as pq
import dataAnalysis.preproc.ns5 as ns5
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
from copy import deepcopy

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
        # print(chanIdx.name)
        chanIdx.annotate(nix_name=chanIdx.name)
        if not (chIds == 'all'):
            if not isinstance(chIds, collections.Iterable):
                chIds = [chIds]
            if (chanIdx.name not in chIds) and ('CH' in chanIdx.name):
                # TODO: delete unwanted
                pass
        if not (adcIds == 'all'):
            if not isinstance(adcIds, collections.Iterable):
                adcIds = [adcIds]
            if (chanIdx.name not in adcIds) and ('ADC' in chanIdx.name):
                # TODO: delete unwanted
                pass
        #
        if chanIdx.name in chanNames.keys():
            thisLabel = chanNames[chanIdx.name]
        else:
            thisLabel = 'NA'
        chanIdx.annotate(label=thisLabel)
        for asig in chanIdx.analogsignals:
            asig.annotate(label=thisLabel)
        # TODO: implement startime end time
    return dataBlock


def preprocOpenEphysFolder(
        folderPath,
        chIds='all', adcIds='all',
        chanNames=None, overwriteExisting=False,
        filterOpts={},
        startTimeS=0, dataTimeS=900,
        chunkSize=900,
        curSection=0, sectionsTotal=1, plotting=False,
        fillOverflow=False, removeJumps=True):

    outputPath = os.path.join(
        folderPath, os.path.basename(folderPath) + '.nix')
    if os.path.exists(outputPath) and overwriteExisting:
        os.remove(outputPath)
    cleanOutputPath = os.path.join(
        folderPath, os.path.basename(folderPath) + '_filtered.nix')
    if os.path.exists(cleanOutputPath) and overwriteExisting:
        os.remove(cleanOutputPath)
    #
    if not os.path.exists(outputPath):
        rawBlock = openEphysFolderToNixBlock(
            folderPath, adcIds=adcIds, chanNames=chanNames)
        writer = NixIO(filename=outputPath)
        writer.write_block(rawBlock, use_obj_names=True)
        writer.close()
    else:
        rawBlock = ns5.loadWithArrayAnn(outputPath, fromRaw=False)
    
    if not os.path.exists(cleanOutputPath):
        cleanDataBlock = preprocOpenEphysBlock(
            rawBlock,
            filterOpts=filterOpts,
            startTimeS=startTimeS, dataTimeS=dataTimeS,
            chunkSize=chunkSize,
            curSection=curSection, sectionsTotal=sectionsTotal,
            plotting=plotting,
            fillOverflow=fillOverflow, removeJumps=removeJumps)
        cleanDataBlock = ns5.purgeNixAnn(cleanDataBlock)
        writer = NixIO(filename=cleanOutputPath)
        writer.write_block(cleanDataBlock, use_obj_names=True)
        writer.close()
    return


def preprocOpenEphysBlock(
        dataBlock,
        filterOpts={},
        startTimeS=0, dataTimeS=900,
        chunkSize=900,
        curSection=0, sectionsTotal=1,
        plotting=False,
        fillOverflow=False, removeJumps=True):
    #filterSOSList = []
    filterCoeffs = np.ndarray(shape=(0, 6))
    samplingRate = dataBlock.filter(objects=AnalogSignal)[0].sampling_rate.magnitude
    fOpts = deepcopy(filterOpts)
    if 'bandstop' in fOpts:
        nNotchHarmonics = fOpts['bandstop'].pop('nHarmonics')
        notchFreq = fOpts['bandstop'].pop('Wn')
        notchQ = fOpts['bandstop'].pop('Q')
        fOpts['bandstop']['fs'] = samplingRate
        for harmonicOrder in range(1, nNotchHarmonics + 1):
            # bNotch, aNotch = signal.iirnotch(
            #     2 * harmonicOrder * notchFreq / samplingRate,
            #     notchQ
            # )
            w0 = harmonicOrder * notchFreq
            bw = w0/notchQ
            fOpts['bandstop']['Wn'] = [w0 - bw/2, w0 + bw/2]
            sos = signal.iirfilter(
                    **fOpts['bandstop'], output='sos')
            #filterSOSList.append(sos)
            filterCoeffs = np.concatenate([filterCoeffs, sos])
            if plotting:
                hf.plotFilterOptsResponse(fOpts['bandstop'])
    #
    if 'matched' in fOpts:
        matchType = fOpts['matched']['type']
        matchFreq = fOpts['matched']['Wn']
        nCycles = fOpts['matched']['nCycles']
        nMatchSamples = int(nCycles * samplingRate / matchFreq)
        noiseKernels = []
        if plotting:
            tK = np.linspace(0, nCycles / matchFreq, nMatchSamples)
            fig, axK = plt.subplots()
        if matchType == 'sin':
            nMatchHarmonics = fOpts['matched']['nHarmonics']
            for harmonicOrder in range(1, nMatchHarmonics + 1):
                kernel = np.sin(
                    np.linspace(
                        0,
                        harmonicOrder * nCycles * 2 * m.pi,
                        nMatchSamples) - m.pi / 2)
                kernel = kernel / np.sum(np.abs(kernel))
                noiseKernels.append(kernel)
                if plotting:
                    axK.plot(
                        tK, kernel,
                        label='model order {}'.format(harmonicOrder))
            numKernels = len(noiseKernels)
        else:
            numKernels = 1
        if plotting:
            axK.set_title('matched filter kernels')
            axK.set_xlabel('Time (sec)')
    
    if 'high' in fOpts:
        fOpts['high']['fs'] = samplingRate
        sos = signal.iirfilter(
                **fOpts['high'], output='sos')
        #filterSOSList.append(sos)
        filterCoeffs = np.concatenate([filterCoeffs, sos])
        if plotting:
            hf.plotFilterOptsResponse(fOpts['high'])
    #
    if 'low' in fOpts:
        fOpts['low']['fs'] = samplingRate
        sos = signal.iirfilter(
                **fOpts['low'], output='sos')
        #filterSOSList.append(sos)
        filterCoeffs = np.concatenate([filterCoeffs, sos])
        if plotting:
            hf.plotFilterOptsResponse(fOpts['low'])

    def filterFun(sig, filterCoeffs=None):
        sig[:] = signal.sosfiltfilt(
            filterCoeffs, sig.magnitude.flatten())[:, np.newaxis] * sig.units
        return sig

    for chanIdx in dataBlock.channel_indexes:
        if 'CH' not in chanIdx.name:
            continue
        for segIdx, seg in enumerate(dataBlock.segments):
            asigList = [a for a in chanIdx.filter(objects=AnalogSignal) if a.segment is seg]
            assert len(asigList) == 1
            asig = asigList[0]
            if plotting:
                # maxIdx = min(asig.shape[0], 2*int(asig.sampling_rate))
                tMask = (asig.times > 419.9) & (asig.times < 420.9)
                if 'matched' in fOpts:
                    fig, (axBA, axM) = plt.subplots(2, sharex=True)
                    cPalette = sns.color_palette(n_colors=numKernels)
                else:
                    fig, axBA = plt.subplots()
                axBA.plot(
                    asig.times[tMask], asig.magnitude[tMask],
                    label='Original')
                #  envelope = np.abs(signal.hilbert(asig.magnitude))
                #  envelope -= np.mean(envelope)
                #  axBA.plot(
                #      asig.times[tMask], envelope[tMask],
                #      label='Hilbert')
                #
            if 'matched' in fOpts:
                if matchType == 'sin':
                    for hOrder, kernel in enumerate(noiseKernels):
                        noiseModel = signal.correlate(
                            asig.flatten(), kernel,
                            mode='same')[:, np.newaxis]
                        if plotting:
                            axM.plot(
                                asig.times[tMask], noiseModel[tMask], '--',
                                label='noise model order {}'.format(hOrder + 1),
                                c=cPalette[hOrder])
                            axM.plot(
                                asig.times[tMask], asig.magnitude[tMask], '-',
                                label='before match order {}'.format(hOrder + 1),
                                c=cPalette[hOrder])
                        asig[:] = (asig - noiseModel * asig.units)
                        if plotting:
                            axBA.plot(
                                asig.times[tMask], asig.magnitude[tMask],
                                label='after match order {}'.format(hOrder + 1))
                if matchType == 'empirical':
                    # find phase
                    shortKernel = asig.flatten()[:int(samplingRate / notchFreq)]
                    phaseIdx = np.argmin(shortKernel)
                    # pdb.set_trace()
                    kernel = asig.flatten()[phaseIdx:phaseIdx + nMatchSamples]
                    kernel = kernel / np.sum(np.abs(kernel))
                    noiseModel = signal.correlate(
                        asig.flatten(), kernel,
                        mode='same')[:, np.newaxis]
                    if plotting:
                        axK.plot(
                            tK, kernel,
                            label='noise Kernel')
                        axK.set_xlabel('Time (sec)')
                        axK.legend()
                        axM.plot(
                            asig.times[tMask], noiseModel[tMask], '--',
                            label='noise model',
                            c=cPalette[0])
                        axM.plot(
                            asig.times[tMask], asig.magnitude[tMask], '-',
                            label='before match',
                            c=cPalette[0])
                    asig[:] = (asig - noiseModel * asig.units)
                    if plotting:
                        axBA.plot(
                            asig.times[tMask], asig.magnitude[tMask],
                            label='after match')
            #
            asig[:] = filterFun(asig, filterCoeffs=filterCoeffs)
            #
            if plotting:
                axBA.plot(
                    asig.times[tMask], asig.magnitude[tMask],
                    label='after filtering')
                axBA.legend()
                axBA.set_xlabel('Time ({})'.format(asig.times.units))
                axBA.set_title('preprocOpenEphysBlock({})'.format(asig.name))
                if 'matched' in fOpts:
                    axM.legend()
                    axM.set_xlabel('Time ({})'.format(asig.times.units))
                plt.show()
    return dataBlock

