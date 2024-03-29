# -*- coding: utf-8 -*-
"""
@author: Radu Darie
"""
from neo.io import NixIO, nixio_fr, BlackrockIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
import quantities as pq
#  from quantities import mV, kHz, s, uV
import math, pdb
from scipy import stats, signal, fftpack
from scipy.ndimage import median_filter
from copy import copy, deepcopy
import dataAnalysis.helperFunctions.helper_functions_new as hf
from dataAnalysis.helperFunctions.hampel import hampel
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.motor_encoder_new as mea
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
#
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timezone
import os, gc
import traceback
import json
from functools import reduce
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
#  import h5py
import re, sys
#  import rcsanalysis.packet_func as rcsa_helpers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from collections import defaultdict
sys.stderr = open(os.devnull, "w")  # silence stderr
from elephant.conversion import binarize
sys.stderr = sys.__stderr__  # unsilence stderr


metaFillerLookup = defaultdict(lambda: 'NA')
metaFillerLookup['program'] = 999.
metaFillerLookup['trialAmplitude'] = 0.
metaFillerLookup['amplitude'] = 0.
metaFillerLookup['amplitudeRound'] = 999.
metaFillerLookup['activeGroup'] = 0.
metaFillerLookup['RateInHz'] = 0.
metaFillerLookup['trialRateInHz'] = 0.
metaFillerLookup['detectionDelay'] = np.nan
metaFillerLookup['stimDelay'] = np.nan


def analogSignalsToDataFrame(
        analogsignals, idxT='t', useChanNames=False):
    asigList = []
    for asig in analogsignals:
        if asig.shape[1] == 1:
            if useChanNames:
                colNames = [str(asig.channel_index.name)]
            else:
                colNames = [str(asig.name)]
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


def listChanNames(
        dataBlock, chanQuery,
        objType=AnalogSignalProxy, condition=None):
    allChanList = [
        i.name
        for i in dataBlock.filter(objects=objType)]
    if condition == 'hasAsigs':
        allChanList = [
            i
            for i in allChanList
            if len(dataBlock.filter(objects=objType, name=i)[0].analogsignals)
        ]
    chansToTrigger = pd.DataFrame(
        np.unique(allChanList),
        columns=['chanName'])
    if chanQuery is not None:
        chansToTrigger = chansToTrigger.query(
            chanQuery, engine='python')['chanName'].to_list()
    else:
        chansToTrigger = chansToTrigger['chanName'].to_list()
    return chansToTrigger


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
            index=np.asarray([idx]),
            channel_names=np.asarray([nixChanName]))
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
                left_sweep=0, dtype=float)
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
        'Classification': [np.asarray([]) for i in range(nCh)],
        'NEUEVWAV_HeaderIndices': [None for i in range(nCh)],
        'TimeStamps': [np.asarray([]) for i in range(nCh)],
        'Units': 'uV',
        'Waveforms': [np.asarray([]) for i in range(nCh)],
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
        'Classification': [np.asarray([]) for i in range(nCh)],
        'NEUEVWAV_HeaderIndices': [None for i in range(nCh)],
        'TimeStamps': [np.asarray([]) for i in range(nCh)],
        'Units': 'uV',
        'Waveforms': [np.asarray([]) for i in range(nCh)],
        'basic_headers': {'TimeStampResolution': 3e4},
        'extended_headers': []
        }
    #  allocate fields for annotations
    for dummyCh in channel_indexes:
        if len(dummyCh.units):
            dummyUnit = dummyCh.units[0]
            if len(dummyUnit.spiketrains):
                if len(dummyUnit.spiketrains[0].times):
                    break
    dummySt = [
        st
        for st in dummyUnit.spiketrains
        if len(st.times)][0]
    #  allocate fields for array annotations (per spike)
    if dummySt.array_annotations:
        for key in dummySt.array_annotations.keys():
            spikes.update({key: [np.asarray([]) for i in range(nCh)]})
        
    maxUnitIdx = 0
    for idx, chIdx in enumerate(channel_indexes):
        spikes['ChannelID'][idx] = chIdx.name
        for unitIdx, thisUnit in enumerate(chIdx.units):
            for stIdx, st in enumerate(thisUnit.spiketrains):
                if not len(st.times):
                    continue
                #  print(
                #      'unit {} has {} spiketrains'.format(
                #          thisUnit.name,
                #          len(thisUnit.spiketrains)))
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
                    try:
                        spikes['Waveforms'][idx] = np.concatenate((
                            spikes['Waveforms'][idx],
                            theseWaveforms.magnitude), axis=0)
                    except Exception:
                        traceback.print_exc()
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
                    try:
                        spikes['basic_headers'][key].update({maxUnitIdx: value})
                    except Exception:
                        pass
                maxUnitIdx += 1
    return spikes


def unitSpikeTrainArrayAnnToDF(
        spikeTrainContainer, columnNames=None):
    #  list contains different segments
    if isinstance(spikeTrainContainer, ChannelIndex):
        assert len(spikeTrainContainer.units) == 0
        spiketrains = spikeTrainContainer.units[0].spiketrains
    elif isinstance(spikeTrainContainer, Unit):
        spiketrains = spikeTrainContainer.spiketrains
    elif isinstance(spikeTrainContainer, list):
        spiketrains = spikeTrainContainer
    doNotFillList = ['segment', 'originalIndex', 't', 'feature', 'bin']
    fullAnnotationsDict = {}
    for segIdx, st in enumerate(spiketrains):
        theseAnnDF = pd.DataFrame(st.array_annotations)
        theseAnnDF.loc[:, 't'] = st.times.magnitude
        if columnNames is not None:
            fieldsNeedFiller = [
                mdn
                for mdn in columnNames
                if (mdn not in doNotFillList) and (mdn not in theseAnnDF.columns)]
            for mdName in fieldsNeedFiller:
                if mdName in st.annotations:
                    theseAnnDF.loc[:, mdName] = st.annotations[mdName]
                elif mdName == 'parentFeature':
                    theseAnnDF.loc[:, mdName] = childBaseName(st.name, 'seg')
                else:
                    theseAnnDF.loc[:, mdName] = metaFillerLookup[mdName]
        fullAnnotationsDict.update({segIdx: theseAnnDF})
    annotationsDF = pd.concat(
        fullAnnotationsDict, names=['segment', 'index'], sort=True)
    # # if columnNames is not None:
    # #     fieldsNeedFiller = [
    # #         mdn
    # #         for mdn in columnNames
    # #         if (mdn not in doNotFillList) and (mdn not in annotationsDF.columns)]
    # #     for mdName in fieldsNeedFiller:
    # #         #
    # #         # if mdName == 'parentFeature':
    # #         #     annotationsDF.loc[:, mdName] = annotationsDF.loc[:, 'feature']
    # #         # else:
    # #         #     annotationsDF.loc[:, mdName] = metaFillerLookup[mdName]
    # #         annotationsDF.loc[:, mdName] = metaFillerLookup[mdName]
    return annotationsDF


def getSpikeDFMetadata(spikeDF, metaDataCols):
    spikeDF.reset_index(inplace=True)
    metaDataCols = np.atleast_1d(metaDataCols)
    spikeDF.index.name = 'metaDataIdx'
    metaDataDF = spikeDF.loc[:, metaDataCols].copy()
    newSpikeDF = spikeDF.drop(columns=metaDataCols).reset_index()
    return newSpikeDF, metaDataDF


def transposeSpikeDF(
        spikeDF, transposeToColumns,
        fastTranspose=False):
    newColumnNames = np.atleast_1d(transposeToColumns).tolist()
    originalColumnNames = np.atleast_1d(spikeDF.columns.names)
    metaDataCols = np.setdiff1d(spikeDF.index.names, newColumnNames).tolist()
    if fastTranspose:
        #  fast but memory inefficient
        return spikeDF.stack().unstack(transposeToColumns)
    else:
        raise(Warning('Caution! transposeSpikeDF might not be working, needs testing RD 06252019'))
        #  stash annotations, transpose, recover annotations
        newSpikeDF, metaDataDF = getSpikeDFMetadata(spikeDF, metaDataCols)
        del spikeDF
        gc.collect()
        #
        newSpikeDF = newSpikeDF.stack().unstack(newColumnNames)
        newSpikeDF.reset_index(inplace=True)
        #  set the index
        newIdxLabels = np.concatenate(
            [originalColumnNames, metaDataCols]).tolist()
        newSpikeDF.loc[:, metaDataCols] = (
            metaDataDF
            .loc[newSpikeDF['metaDataIdx'].to_list(), metaDataCols]
            .to_numpy())
        newSpikeDF = (
            newSpikeDF
            .drop(columns=['metaDataIdx'])
            .set_index(newIdxLabels))
        return newSpikeDF


def concatenateBlocks(
        asigBlocks, spikeBlocks, eventBlocks, chunkingMetadata,
        samplingRate, chanQuery, lazy, trackMemory, verbose,
        clipSignals=False, clippingOpts={},
        filterSignals=False, analysisFilterOpts={},
        medianFilterSignals=False, analysisMedianFilterOpts={},
        hampelFilterSignals=False, analysisHampelFilterOpts={},
        diagnosticPlots=False
        ):
    # Scan ahead through all files and ensure that
    # spikeTrains and units are present across all assembled files
    channelIndexCache = {}
    unitCache = {}
    asigCache = []
    asigAnnCache = {}
    spiketrainCache = {}
    eventCache = {}
    # get list of channels and units
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        gc.collect()
        chunkIdx = int(chunkIdxStr)
        asigBlock = asigBlocks[chunkIdx]
        asigSeg = asigBlock.segments[0]
        spikeBlock = spikeBlocks[chunkIdx]
        eventBlock = eventBlocks[chunkIdx]
        eventSeg = eventBlock.segments[0]
        for chIdx in asigBlock.filter(objects=ChannelIndex):
            chAlreadyThere = (chIdx.name in channelIndexCache.keys())
            if not chAlreadyThere:
                newChIdx = copy(chIdx)
                newChIdx.analogsignals = []
                newChIdx.units = []
                channelIndexCache[chIdx.name] = newChIdx
        for unit in (spikeBlock.filter(objects=Unit)):
            if lazy:
                theseSpiketrains = []
                for stP in unit.spiketrains:
                    st = loadStProxy(stP)
                    if len(st.times) > 0:
                        theseSpiketrains.append(st)
            else:
                theseSpiketrains = [
                    st
                    for st in unit.spiketrains
                    if len(st.times)
                    ]
            for st in theseSpiketrains:
                st = loadObjArrayAnn(st)
                if len(st.times):
                    st.magnitude[:] = st.times.magnitude + spikeBlock.annotations['chunkTStart']
                    st.t_start = min(0 * pq.s, st.times[0] * 0.999)
                    st.t_stop = max(
                        st.t_stop + spikeBlock.annotations['chunkTStart'] * pq.s,
                        st.times[-1] * 1.001)
                else:
                    st.t_start += spikeBlock.annotations['chunkTStart'] * pq.s
                    st.t_stop += spikeBlock.annotations['chunkTStart'] * pq.s
            uAlreadyThere = (unit.name in unitCache.keys())
            if not uAlreadyThere:
                newUnit = copy(unit)
                newUnit.spiketrains = []
                newUnit.annotations['parentChanName'] = unit.channel_index.name
                unitCache[unit.name] = newUnit
                spiketrainCache[unit.name] = theseSpiketrains
            else:
                spiketrainCache[unit.name] = spiketrainCache[unit.name] + theseSpiketrains
        #
        if lazy:
            evList = [
                evP.load()
                for evP in eventSeg.events]
        else:
            evList = eventSeg.events
        for event in evList:
            event.magnitude[:] = event.magnitude + eventBlock.annotations['chunkTStart']
            if event.name in eventCache.keys():
                eventCache[event.name].append(event)
            else:
                eventCache[event.name] = [event]
        # take the requested analog signal channels
        if lazy:
            tdChanNames = listChanNames(
                asigBlock, chanQuery, objType=AnalogSignalProxy)
            #############
            # tdChanNames = ['seg0_utah1', 'seg0_utah10']
            ##############
            asigList = []
            for asigP in asigSeg.analogsignals:
                if asigP.name in tdChanNames:
                    asig = asigP.load()
                    asig.channel_index = asigP.channel_index
                    asigList.append(asig)
                    if trackMemory:
                        print('loading {} from proxy object. memory usage: {:.1f} MB'.format(
                            asigP.name, prf.memory_usage_psutil()))
        else:
            tdChanNames = listChanNames(
                asigBlock, chanQuery, objType=AnalogSignal)
            asigList = [
                asig
                for asig in asigSeg.analogsignals
                if asig.name in tdChanNames
                ]
        for asig in asigList:
            if asig.size > 0:
                dummyAsig = asig
        if idx == 0:
            outputBlock = Block(
                name=asigBlock.name,
                file_origin=asigBlock.file_origin,
                file_datetime=asigBlock.file_datetime,
                rec_datetime=asigBlock.rec_datetime,
                **asigBlock.annotations
            )
            newSeg = Segment(
                index=0, name=asigSeg.name,
                description=asigSeg.description,
                file_origin=asigSeg.file_origin,
                file_datetime=asigSeg.file_datetime,
                rec_datetime=asigSeg.rec_datetime,
                **asigSeg.annotations
            )
            outputBlock.segments = [newSeg]
            for asig in asigList:
                asigAnnCache[asig.name] = asig.annotations
                asigAnnCache[asig.name]['parentChanName'] = asig.channel_index.name
            asigUnits = dummyAsig.units
        tdDF = analogSignalsToDataFrame(asigList)
        del asigList  # asigs saved to dataframe, no longer needed
        tdDF.loc[:, 't'] += asigBlock.annotations['chunkTStart']
        tdDF.set_index('t', inplace=True)
        if samplingRate != dummyAsig.sampling_rate:
            if filterSignals:
                if not len(analysisFilterOpts):
                    analysisFilterOpts = {
                        'low': {
                            'Wn': float(samplingRate / 3),
                            'N': 8,
                            'btype': 'low',
                            'ftype': 'butter'
                        },
                        'high': {
                            'Wn': .1,
                            'N': 8,
                            'btype': 'high',
                            'ftype': 'butter'
                        }
                    }
            else:
                analysisFilterOpts = {}
            newT = pd.Series(
                np.arange(
                    dummyAsig.t_start + asigBlock.annotations['chunkTStart'] * pq.s,
                    dummyAsig.t_stop + asigBlock.annotations['chunkTStart'] * pq.s,
                    1/samplingRate))
            if samplingRate < dummyAsig.sampling_rate:
                if filterSignals:
                    filterCoeffs = hf.makeFilterCoeffsSOS(
                        analysisFilterOpts, float(dummyAsig.sampling_rate))
                '''
                ### check that axis=0 is the correct option
                dummyDF = tdDF.loc[:, 'seg0_utah90'].copy()
                filteredAsigs0 = signal.sosfiltfilt( filterCoeffs, dummyDF.to_numpy(), axis=0)
                filteredAsigs1 = signal.sosfiltfilt( filterCoeffs, dummyDF.to_numpy(), axis=1)
                dummyDF = tdDF.loc[:, ['seg0_utah90']].iloc[6467800:6520000, :].copy(); 
                medianFilteredAsigs = median_filter(dummyDF.to_numpy(), size=(30, 1))
                fig, ax = plt.subplots(); ax.plot(dummyDF.to_numpy()); ax.plot(medianFilteredAsigs); plt.show()
                ###
                '''
                if medianFilterSignals:
                    if not len(analysisMedianFilterOpts):
                        analysisMedianFilterOpts = dict(size=(30, 1))
                    print('Applying median filter, analysisMedianFilterOpts={}'.format(analysisMedianFilterOpts))
                    medianFilteredAsigs = median_filter(tdDF.to_numpy(), **analysisMedianFilterOpts)
                    tdDF = pd.DataFrame(
                        medianFilteredAsigs,
                        index=tdDF.index,
                        columns=tdDF.columns)
                elif hampelFilterSignals:
                    if not len(analysisHampelFilterOpts):
                        analysisHampelFilterOpts = dict()
                    print('Applying hampel filter, analysisHampelFilterOpts={}'.format(analysisHampelFilterOpts))
                    analysisHampelFilterOpts.update({'debug_plots': diagnosticPlots})
                    tdDF, tdSAD, (fig, ax) = hampel(tdDF, **analysisHampelFilterOpts)
                    del tdSAD
                #
                if filterSignals:
                    if trackMemory:
                        print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                            prf.memory_usage_psutil()))
                    filteredAsigs = signal.sosfiltfilt(
                        filterCoeffs, tdDF.to_numpy(),
                        axis=0)
                    tdDF = pd.DataFrame(
                        filteredAsigs,
                        index=tdDF.index, columns=tdDF.columns)
                    if diagnosticPlots and hampelFilterSignals:
                        t = tdDF.index.to_numpy()
                        palette1=['b', 'g', 'r']
                        palette2=['c', 'y', 'm']
                        for cIdx, cN in enumerate(['seg0_utah78', 'seg0_utah1']):
                            filtered = tdDF.loc[:, cN].to_numpy()
                            ax[0].plot(t, filtered, lw=1., c=palette1[cIdx], label='{} (filtered)'.format(cN))
                        ax[0].legend(loc='upper left')
                        plt.show()
                    if trackMemory:
                        print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                            prf.memory_usage_psutil()))
            tdInterp = hf.interpolateDF(
                tdDF, newT,
                kind='linear', fill_value='extrapolate',
                verbose=verbose)
            # free up memory used by full resolution asigs
            del tdDF
        else:
            tdInterp = tdDF
        #
        asigCache.append(tdInterp)
        #
        print('Finished chunk {}'.format(chunkIdxStr))
    allTdDF = pd.concat(asigCache)
    del asigCache
    #
    if clipSignals:
        for clpN, clpOpt in clippingOpts.items():
            colMask = allTdDF.columns.str.contains(clpN)
            useIQR = clpOpt.pop('IQR', False)
            if useIQR:
                tdMedian = np.median(allTdDF.loc[:, colMask])
                qVal = clpOpt.pop('quantileVal')
                multi = clpOpt.pop('quantileMultiple')
                tdIQR = (
                    np.percentile(allTdDF.loc[:, colMask], 100 * (1 - qVal)) -
                    np.percentile(allTdDF.loc[:, colMask], 100 * qVal))
                allTdDF.loc[:, colMask] = allTdDF.loc[:, colMask].clip(
                    lower=tdMedian - multi * tdIQR,
                    upper=tdMedian + multi * tdIQR)
            absoluteLimit = clpOpt.pop('absoluteLimit', False)
            if absoluteLimit:
                allTdDF.loc[:, colMask] = allTdDF.loc[:, colMask].clip(
                    lower=absoluteLimit * (-1),
                    upper=absoluteLimit)
    # TODO: check for nans, if, for example a signal is partially missing
    allTdDF.fillna(method='bfill', inplace=True)
    allTdDF.fillna(method='ffill', inplace=True)
    for asigName in allTdDF.columns:
        newAsig = AnalogSignal(
            allTdDF[asigName].to_numpy() * asigUnits,
            name=asigName,
            sampling_rate=samplingRate,
            dtype=float,
            **asigAnnCache[asigName])
        chIdxName = asigAnnCache[asigName]['parentChanName']
        chIdx = channelIndexCache[chIdxName]
        # cross-assign ownership to containers
        chIdx.analogsignals.append(newAsig)
        newSeg.analogsignals.append(newAsig)
        newAsig.channel_index = chIdx
        newAsig.segment = newSeg
    #
    for uName, unit in unitCache.items():
        # concatenate spike times, waveforms, etc.
        if len(spiketrainCache[unit.name]):
            consolidatedTimes = np.concatenate([
                    st.times.magnitude
                    for st in spiketrainCache[unit.name]
                ])
            # TODO:   decide whether to include this step
            #         which snaps the spike times to the nearest
            #         *sampled* data point
            #
            # consolidatedTimes, timesIndex = hf.closestSeries(
            #     takeFrom=pd.Series(consolidatedTimes),
            #     compareTo=pd.Series(allTdDF.index))
            #
            # find an example spiketrain with array_annotations
            for st in spiketrainCache[unit.name]:
                if len(st.times):
                    dummySt = st
                    break
            consolidatedAnn = {
                key: np.array([])
                for key, value in dummySt.array_annotations.items()
                }
            for key, value in consolidatedAnn.items():
                consolidatedAnn[key] = np.concatenate([
                    st.annotations[key]
                    for st in spiketrainCache[unit.name]
                ])
            consolidatedWaveforms = np.concatenate([
                st.waveforms
                for st in spiketrainCache[unit.name]
                ])
            spikeTStop = max([
                st.t_stop
                for st in spiketrainCache[unit.name]
                ])
            spikeTStart = max([
                st.t_start
                for st in spiketrainCache[unit.name]
                ])
            spikeAnnotations = {
                key: value
                for key, value in dummySt.annotations.items()
                if key not in dummySt.annotations['arrayAnnNames']
            }
            newSt = SpikeTrain(
                name=dummySt.name,
                times=consolidatedTimes, units='sec', t_stop=spikeTStop,
                waveforms=consolidatedWaveforms * dummySt.waveforms.units,
                left_sweep=dummySt.left_sweep,
                sampling_rate=dummySt.sampling_rate,
                t_start=spikeTStart, **spikeAnnotations,
                array_annotations=consolidatedAnn)
            # cross-assign ownership to containers
            unit.spiketrains.append(newSt)
            newSt.unit = unit
            newSeg.spiketrains.append(newSt)
            newSt.segment = newSeg
            # link chIdxes and Units
            if unit.annotations['parentChanName'] in channelIndexCache:
                chIdx = channelIndexCache[unit.annotations['parentChanName']]
                if unit not in chIdx.units:
                    chIdx.units.append(unit)
                    unit.channel_index = chIdx
            else:
                newChIdx = ChannelIndex(
                    name=unit.annotations['parentChanName'], index=0)
                channelIndexCache[unit.annotations['parentChanName']] = newChIdx
                if unit not in newChIdx.units:
                    newChIdx.units.append(unit)
                    unit.channel_index = newChIdx
    #
    for evName, eventList in eventCache.items():
        consolidatedTimes = np.concatenate([
            ev.times.magnitude
            for ev in eventList
            ])
        consolidatedLabels = np.concatenate([
            ev.labels
            for ev in eventList
            ])
        newEvent = Event(
            name=evName,
            times=consolidatedTimes * pq.s,
            labels=consolidatedLabels
            )
        # if len(newEvent):
        newEvent.segment = newSeg
        newSeg.events.append(newEvent)
    for chIdxName, chIdx in channelIndexCache.items():
        if len(chIdx.analogsignals) or len(chIdx.units):
            outputBlock.channel_indexes.append(chIdx)
            chIdx.block = outputBlock
    #
    outputBlock = purgeNixAnn(outputBlock)
    createRelationship = False
    if createRelationship:
        outputBlock.create_relationship()
    return outputBlock


'''
def concatenateEventsContainerV2(eventContainer, newSegIdx=0):
    if isinstance(eventContainer, dict):
        allListOfEvents = list(eventContainer.values())
    else:
        allListOfEvents = eventContainer
    listOfEvents = [ev for ev in allListOfEvents if len(ev.times)]
    if not len(listOfEvents) > 0:
        return allListOfEvents[0]
    masterEvent = None
    for evIdx, ev in enumerate(listOfEvents):
        masterEvent = ev
        if len(masterEvent.times):
            break
    if evIdx > len(listOfEvents) - 1:
        for ev in listOfEvents[evIdx+1:]:
            masterEvent = masterEvent.merge(ev)
    if masterEvent.array_annotations is not None:
        arrayAnnNames = list(masterEvent.array_annotations.keys())
        masterEvent.annotations.update(masterEvent.array_annotations)
        masterEvent.annotations['arrayAnnNames'] = arrayAnnNames
    return masterEvent
'''


def concatenateEventsContainer(
        eventContainer, linkParents=True):
    if isinstance(eventContainer, dict):
        listOfEvents = list(eventContainer.values())
    else:
        listOfEvents = eventContainer
    nonEmptyEvents = []
    masterEvent = None
    for evIdx, ev in enumerate(listOfEvents):
        if len(ev.times):
            nonEmptyEvents.append(ev)
            if masterEvent is None:
                masterEvent = ev
                startIdx = evIdx + 1
    if not len(nonEmptyEvents) > 0:
        return listOfEvents[0]
    for evIdx, ev in enumerate(listOfEvents[startIdx:]):
        try:
            if hasattr(masterEvent, 't_stop'):
                if masterEvent.t_stop < ev.t_stop:
                    masterEvent.t_stop = ev.t_stop
                elif masterEvent.t_stop > ev.t_stop:
                    ev.t_stop = masterEvent.t_stop
            if hasattr(masterEvent, 't_start'):
                if masterEvent.t_start > ev.t_start:
                    masterEvent.t_start = ev.t_start
                elif masterEvent.t_start < ev.t_start:
                    ev.t_start = masterEvent.t_start
            if ev.size:
                masterEvent = masterEvent.merge(ev)
        except Exception:
            traceback.print_exc()
            # [arr.shape for key, arr in masterEvent.array_annotations.items()]
    if masterEvent.array_annotations is not None:
        arrayAnnNames = list(masterEvent.array_annotations.keys())
        masterEvent.annotations.update(masterEvent.array_annotations)
        masterEvent.annotations['arrayAnnNames'] = arrayAnnNames
    if linkParents:
        masterEvent.segment = listOfEvents[0].segment
        if isinstance(masterEvent, SpikeTrain):
            masterEvent.unit = listOfEvents[0].unit
    return masterEvent

'''
def concatenateEventsContainer(eventContainer, newSegIdx=0):
    if isinstance(eventContainer, dict):
        listOfEvents = list(eventContainer.values())
    else:
        listOfEvents = eventContainer
    listOfEvents = [ev for ev in listOfEvents if len(ev.times)]
    for ev in listOfEvents:
        dummyEvent = ev
        if len(dummyEvent.times):
            break
    consolidatedTimes = np.concatenate([st.times for st in listOfEvents]) * dummyEvent.units
    arrayAnnNames = list(dummyEvent.array_annotations.keys())
    if len(arrayAnnNames):
        consolidatedArrayAnn = {k: None for k in arrayAnnNames}
        for annNm in arrayAnnNames:
            consolidatedArrayAnn[annNm] = np.concatenate(
                [st.array_annotations[annNm] for st in listOfEvents])
    else:
        consolidatedArrayAnn = None
    if isinstance(dummyEvent, SpikeTrain):
        consolidatedWaveforms = np.concatenate(
            [st.waveforms for st in listOfEvents if hasattr(st, 'waveforms')],
            axis=0) * dummyEvent.waveforms.units
        tStop = max(st.t_stop for st in listOfEvents if hasattr(st, 't_stop'))
        tStart = min(st.t_start for st in listOfEvents if hasattr(st, 't_start'))
        if consolidatedWaveforms.size == 0:
            consolidatedWaveforms = None
            leftSweep = None
        else:
            dummyEvent.left_sweep
        outObj = SpikeTrain(
            name='seg{}_{}'.format(newSegIdx, dummyEvent.unit.name),
            times=consolidatedTimes,
            waveforms=consolidatedWaveforms,
            t_start=tStart, t_stop=tStop,
            left_sweep=leftSweep, **dummyEvent.annotations
            )
        outObj.unit = dummyEvent.unit
    if isinstance(dummyEvent, Event):
        consolidatedLabels = np.concatenate([st.labels for st in listOfEvents])
        outObj = Event(
            name='seg{}_{}'.format(
                newSegIdx, childBaseName(dummyEvent.name, 'seg')),
            labels=consolidatedLabels,
            times=consolidatedTimes,
            **dummyEvent.annotations
            )
    outObj.segment = dummyEvent.segment
    if consolidatedArrayAnn is not None:
        outObj.array_annotations = consolidatedArrayAnn
        outObj.annotations.update(consolidatedArrayAnn)
        outObj.annotations['arrayAnnNames'] = arrayAnnNames
    return outObj
'''

#  renamed spikeTrainWaveformsToDF to unitSpikeTrainWaveformsToDF
def unitSpikeTrainWaveformsToDF(
        spikeTrainContainer,
        dataQuery=None,
        transposeToColumns='bin', fastTranspose=True,
        lags=None, decimate=1, rollingWindow=None,
        getMetaData=True, verbose=False, getFeatureMetaData=None,
        whichSegments=None, windowSize=None, procFun=None):
    #  list contains different segments from *one* unit
    if isinstance(spikeTrainContainer, ChannelIndex):
        assert len(spikeTrainContainer.units) == 0
        spiketrains = spikeTrainContainer.units[0].spiketrains
    elif isinstance(spikeTrainContainer, Unit):
        spiketrains = spikeTrainContainer.spiketrains
    else:
        raise(Exception('not a valid container'))
    # TODO check if really need to assert uniqueness?
    uniqueSpiketrains = []
    for st in spiketrains:
        if not np.any([st is i for i in uniqueSpiketrains]):
            uniqueSpiketrains.append(st)
    #  subsampling options
    decimate = int(decimate)
    if whichSegments is not None:
        uniqueSpiketrains = [
            uniqueSpiketrains[i]
            for i in whichSegments
        ]
    waveformsList = []
    #
    getMetaDataWorking = deepcopy(getMetaData)
    if getFeatureMetaData is not None:
        if isinstance(getMetaDataWorking, Iterable):
            for annName in getFeatureMetaData:
                if annName not in getMetaDataWorking:
                    getMetaDataWorking.append(annName)
        elif not(getMetaDataWorking):
            getMetaDataWorking = deepcopy(getFeatureMetaData)
    for segIdx, stIn in enumerate(uniqueSpiketrains):
        if verbose:
            print('extracting spiketrain from {}'.format(stIn.segment))
        #  make sure is not a proxyObj
        if isinstance(stIn, SpikeTrainProxy):
            st = loadStProxy(stIn)
            if (getMetaDataWorking) or (dataQuery is not None):
                # if there's a query, get metadata temporarily to resolve it
                st = loadObjArrayAnn(st)
        else:
            st = stIn
        #  extract bins spaced by decimate argument
        if not st.times.any():
            continue
        if verbose:
            print('extracting wf from {}'.format(stIn.segment))
        wf = np.asarray(
            np.squeeze(st.waveforms),
            dtype=float)
        if wf.ndim == 3:
            print('Waveforms from more than one channel!')
            if wf.shape[1] > 0:
                wf = wf[:, 0, :]
        wfDF = pd.DataFrame(wf)
        # samplingRate = st.sampling_rate
        samplingPeriod = float(st.sampling_period) * st.sampling_period.units
        leftSweep = float(st.left_sweep) * st.left_sweep.units
        bins = (
            np.arange(wfDF.shape[1], dtype=float) * samplingPeriod -
            leftSweep)
        wfDF.columns = bins.magnitude
        if windowSize is not None:
            winMask = (
                (wfDF.columns >= windowSize[0]) &
                (wfDF.columns <= windowSize[1]))
            wfDF = wfDF.loc[:, winMask]
        wfDF.columns = np.around(wfDF.columns, decimals=6)
        if procFun is not None:
            wfDF = procFun(wfDF, st)
        idxLabels = ['segment', 'originalIndex', 't']
        wfDF.loc[:, 't'] = np.asarray(st.times.magnitude.astype(np.float64))
        if (getMetaDataWorking) or (dataQuery is not None):
            # if there's a query, get metadata temporarily to resolve it
            annDict = {}
            for k, values in st.array_annotations.items():
                if isinstance(getMetaDataWorking, Iterable):
                    # if selecting metadata fields, check that
                    # the key is in the provided list
                    if k not in getMetaDataWorking:
                        continue
                if isinstance(values[0], str):
                    v = np.asarray(values, dtype='str')
                else:
                    v = np.asarray(values)
                annDict.update({k: v})
            skipAnnNames = (
                st.annotations['arrayAnnNames'] +
                [
                    'arrayAnnNames', 'arrayAnnDTypes',
                    'nix_name', 'neo_name', 'id', 'templateDist',
                    'cell_label', 'cluster_label', 'max_on_channel', 'binWidth']
                )
            annDF = pd.DataFrame(annDict)
            for k, value in st.annotations.items():
                if isinstance(getMetaDataWorking, Iterable):
                    # if selecting metadata fields, check that
                    # the key is in the provided list
                    if k not in getMetaDataWorking:
                        continue
                if k not in skipAnnNames:
                    annDF.loc[:, k] = value
            #
            if isinstance(getMetaDataWorking, Iterable):
                doNotFillList = idxLabels + ['feature', 'bin']
                fieldsNeedFiller = [
                    mdn
                    for mdn in getMetaDataWorking
                    if (mdn not in doNotFillList) and (mdn not in annDF.columns)]
                for mdName in fieldsNeedFiller:
                    if mdName == 'parentFeature':
                        annDF.loc[:, mdName] = childBaseName(st.name, 'seg')
                    else:
                        annDF.loc[:, mdName] = metaFillerLookup[mdName]
            annColumns = annDF.columns.to_list()
            if getMetaDataWorking:
                for annNm in annColumns:
                    if annNm not in idxLabels:
                        idxLabels.append(annNm)
                # idxLabels += annColumns
            spikeDF = annDF.join(wfDF)
        else:
            spikeDF = wfDF
            del wfDF, st
        spikeDF.loc[:, 'segment'] = segIdx
        spikeDF.loc[:, 'originalIndex'] = spikeDF.index
        spikeDF.columns.name = 'bin'
        if dataQuery is not None:
            spikeDF.query(dataQuery, inplace=True)
            if not getMetaDataWorking:
                spikeDF.drop(columns=annColumns, inplace=True)
        waveformsList.append(spikeDF)
    #
    zeroLagWaveformsDF = pd.concat(waveformsList, axis='index')
    if verbose:
        prf.print_memory_usage('before transposing waveforms')
    metaDF = zeroLagWaveformsDF.loc[:, idxLabels].copy()
    zeroLagWaveformsDF.drop(columns=idxLabels, inplace=True)
    zeroLagWaveformsDF.columns = zeroLagWaveformsDF.columns.astype(float)
    if lags is None:
        lags = [0]
    laggedWaveformsDict = {
        (spikeTrainContainer.name, k): None for k in lags}
    laggedMetaDict = {
        (spikeTrainContainer.name, k): None for k in lags}
    for lag in lags:
        if isinstance(lag, int):
            shiftedWaveform = zeroLagWaveformsDF.shift(
                lag, axis='columns')
            if rollingWindow is not None:
                halfRollingWin = int(np.ceil(rollingWindow/2))
                # seekIdx = slice(
                #     halfRollingWin, -halfRollingWin+1, decimate)
                seekIdx = slice(rollingWindow, None, decimate)
                shiftedWaveform = (
                    shiftedWaveform
                    .T
                    .rolling(
                        window=rollingWindow,
                        win_type='hann',
                        # win_type='gaussian',
                        center=True)
                    .mean(sym=True).T)  # std=halfRollingWin
            else:  # rollingWindow is None
                halfRollingWin = 0
                seekIdx = slice(None, None, decimate)
                if False:
                    oldShiftedWaveform = zeroLagWaveformsDF.shift(
                        lag, axis='columns')
                    plt.plot(oldShiftedWaveform.iloc[0, :])
                    plt.plot(shiftedWaveform.iloc[0, :])
                    plt.show()
            thisLaggedWvf = shiftedWaveform.iloc[:, seekIdx].copy()
            laggedWaveformsDict[
                (spikeTrainContainer.name, lag)] = thisLaggedWvf
        if isinstance(lag, tuple):
            halfRollingWin = int(np.ceil(lag[1]/2))
            '''seekIdx = slice(
                halfRollingWin, -halfRollingWin+1, decimate)'''
            seekIdx = slice(lag[1], None, decimate)
            shiftedWaveform = (
                zeroLagWaveformsDF
                .shift(lag[0], axis='columns')
                .T
                .rolling(
                    window=lag[1],
                    win_type='hann',
                    # win_type='gaussian',
                    center=True)
                .mean(sym=True).T)  # std=halfRollingWin
            laggedWaveformsDict[
                (spikeTrainContainer.name, lag)] = (
                    shiftedWaveform.iloc[:, seekIdx].copy())
        laggedMetaDict[
            (spikeTrainContainer.name, lag)] = metaDF.copy()
    # plt.plot(zeroLagWaveformsDF.columns, zeroLagWaveformsDF.mean())
    # plt.plot(shiftedWaveform.columns, shiftedWaveform.mean()); plt.show()
    stackedMetaDF = pd.concat(laggedMetaDict, names=['feature', 'lag', 'originalDummy']).reset_index().drop(columns=['originalDummy'])
    # e.g. getFeatureMetaData = ['freqBandName', 'parentFeature']
    if getFeatureMetaData is not None:
        featureMetaDF = stackedMetaDF.loc[:, ['feature', 'lag'] + getFeatureMetaData].drop_duplicates().reset_index(drop=True)
        stackedMetaDF.drop(columns=getFeatureMetaData, inplace=True)
    else:
        featureMetaDF = None
    if transposeToColumns == 'feature':
        # stack the bin, name the feature column
        for idx, (key, value) in enumerate(laggedWaveformsDict.items()):
            if idx == 0:
                stackedIndexDF = pd.concat(
                    [metaDF, value], axis='columns')
                stackedIndexDF.set_index(idxLabels, inplace=True)
                # don't drop nans for now - might need to keep track of them
                # if we need to equalize to another array later
                newIndex = stackedIndexDF.stack(dropna=False).index
                idxLabels.append('bin')
            laggedWaveformsDict[key] = value.stack(dropna=False).to_frame(name=key).reset_index(drop=True)
        waveformsDF = pd.concat(
            laggedWaveformsDict.values(),
            axis='columns')
        waveformsDF.columns.names = ['feature', 'lag']
        waveformsDF.index = newIndex
        waveformsDF.columns.name = 'feature'
    elif transposeToColumns == 'bin':
        # add the feature column
        waveformsDF = pd.concat(
            laggedWaveformsDict,
            names=['feature', 'lag', 'originalDummy']).reset_index()
        waveformsDF = pd.concat(
            [
                stackedMetaDF.drop(columns=['feature', 'lag']),
                waveformsDF.drop(columns='originalDummy')],
            axis='columns')
        idxLabels += ['feature', 'lag']
        waveformsDF.columns.name = 'bin'
        waveformsDF.set_index(idxLabels, inplace=True)
    #
    if transposeToColumns != waveformsDF.columns.name:
        waveformsDF = transposeSpikeDF(
            waveformsDF, transposeToColumns,
            fastTranspose=fastTranspose)
    return waveformsDF, featureMetaDF


def concatenateUnitSpikeTrainWaveformsDF(
        units, dataQuery=None,
        transposeToColumns='bin', concatOn='index',
        fastTranspose=True, getMetaData=True,
        getFeatureMetaData=None,
        verbose=False, addLags=None, decimate=1, rollingWindow=None,
        metaDataToCategories=False, windowSize=None,
        whichSegments=None, procFun=None):
    allUnits = []
    for thisUnit in units:
        hasAnySpikes = []
        for stIn in thisUnit.spiketrains:
            if isinstance(stIn, SpikeTrainProxy):
                st = stIn.load(
                    magnitude_mode='rescaled',
                    load_waveforms=False)
            else:
                st = stIn
            hasAnySpikes.append(st.times.any())
        if np.any(hasAnySpikes):
            allUnits.append(thisUnit)
    waveformsList = []
    featureMetaList = []
    for idx, thisUnit in enumerate(allUnits):
        if verbose:
            print('concatenating unitDF {}'.format(thisUnit.name))
        lags = None
        if addLags is not None:
            if 'all' in addLags:
                lags = addLags['all']
            elif thisUnit.name in addLags:
                lags = addLags[thisUnit.name]
        unitWaveforms, featureMetaDF = unitSpikeTrainWaveformsToDF(
            thisUnit, dataQuery=dataQuery,
            transposeToColumns=transposeToColumns,
            fastTranspose=fastTranspose, getMetaData=getMetaData,
            getFeatureMetaData=getFeatureMetaData,
            lags=lags, decimate=decimate, rollingWindow=rollingWindow,
            verbose=verbose, windowSize=windowSize,
            whichSegments=whichSegments, procFun=procFun)
        if idx == 0:
            idxLabels = unitWaveforms.index.names
        else:
            assert (idxLabels == unitWaveforms.index.names)
        if (concatOn == 'columns') and (idx > 0):
            # i.e. if the columns represent features,
            # other than first time, we already have the trial metadata
            unitWaveforms.reset_index(drop=True, inplace=True)
        else:
            # first time, or if concatenating indices,
            # keep the the metadata
            unitWaveforms.reset_index(inplace=True)
            if metaDataToCategories:
                # convert metadata to categoricals to free memory
                #
                unitWaveforms[idxLabels] = (
                    unitWaveforms[idxLabels]
                    .astype('category')
                    )
        waveformsList.append(unitWaveforms)
        if getFeatureMetaData is not None:
            featureMetaList.append(featureMetaDF)
        del unitWaveforms
        if verbose:
            print('memory usage: {:.1f} MB'.format(prf.memory_usage_psutil()))
    if verbose:
        print(
            'about to join all, memory usage: {:.1f} MB'
            .format(prf.memory_usage_psutil()))
    #  if concatenating indexes, reset the index of the result
    #  ignoreIndex = (concatOn == 'index')
    try:
        assert len(waveformsList) > 0, 'Error: dataquery ({}) yields empty list!'.format(dataQuery)
    except:
        traceback.print_exc()
    allWaveforms = pd.concat(waveformsList, axis=concatOn)
    # allWaveforms.loc[allWaveforms.isna().any(axis=1), :]
    if getFeatureMetaData is not None:
        allFeatureMetaDF = pd.concat(featureMetaList).drop_duplicates().set_index(['feature', 'lag'])
    else:
        allFeatureMetaDF = None
    del waveformsList
    if verbose:
        print(
            'finished concatenating, memory usage: {:.1f} MB'
            .format(prf.memory_usage_psutil()))
    try:
        # idxLabels = sorted(idxLabels)
        allWaveforms.set_index(idxLabels, inplace=True)
        sortIndexBy = ['segment', 't'] # , 'originalIndex'
        for cN in ['bin']:
            if cN in allWaveforms.index.names:
                sortIndexBy = sortIndexBy + [cN]
        for cN in ['lag', 'feature']:
            if cN in allWaveforms.index.names:
                sortIndexBy = [cN] + sortIndexBy
        allWaveforms.sort_index(
            level=sortIndexBy,
            axis='index', inplace=True,
            kind='mergesort', sort_remaining=False)
        allWaveforms.sort_index(
            axis='columns', inplace=True,
            kind='mergesort', sort_remaining=False)
    except Exception:
        traceback.print_exc()
    return allWaveforms, allFeatureMetaDF


def alignedAsigsToDF(
        dataBlock, unitNames=None,
        unitQuery=None, dataQuery=None,
        collapseSizes=False, verbose=False,
        duplicateControlsByProgram=False,
        amplitudeColumn='trialAmplitude', programColumn='program', electrodeColumn='electrode',
        transposeToColumns='bin', concatOn='index', fastTranspose=True,
        addLags=None, decimate=1, rollingWindow=None,
        whichSegments=None, windowSize=None,
        getMetaData=True, getFeatureMetaData=None,
        metaDataToCategories=False,
        outlierTrials=None, invertOutlierMask=False,
        makeControlProgram=False, removeFuzzyName=False,
        procFun=None, dropNaNs=True, finalIndexMask=None, overrideSegIdx=None):
    #  channels to trigger
    if unitNames is None:
        unitNames = listChanNames(dataBlock, unitQuery, objType=Unit)
    allUnits = []
    for uName in unitNames:
        allUnits += dataBlock.filter(objects=Unit, name=uName)
    allWaveforms, columnsMeta = concatenateUnitSpikeTrainWaveformsDF(
        allUnits, dataQuery=dataQuery,
        transposeToColumns=transposeToColumns, concatOn=concatOn,
        fastTranspose=fastTranspose,
        addLags=addLags, decimate=decimate, rollingWindow=rollingWindow,
        verbose=verbose, whichSegments=whichSegments,
        windowSize=windowSize, procFun=procFun,
        getMetaData=getMetaData, getFeatureMetaData=getFeatureMetaData,
        metaDataToCategories=metaDataToCategories)
    #
    waveformsIndexDF = allWaveforms.index.to_frame().reset_index(drop=True)
    roundFactor = 9
    waveformsIndexDF.loc[:, 't'] = np.around(waveformsIndexDF['t'], decimals=roundFactor)
    # np.unique(waveformsIndexDF['t'])
    if overrideSegIdx is not None:
        waveformsIndexDF.loc[:, 'segment'] = overrideSegIdx
    allWaveforms.index = pd.MultiIndex.from_frame(waveformsIndexDF)
    manipulateIndex = np.any(
        [
            collapseSizes, duplicateControlsByProgram,
            makeControlProgram, removeFuzzyName
            ])
    if outlierTrials is not None:
        outlierTrialsIndex = outlierTrials.index.to_frame().reset_index(drop=True)
        outlierTrialsIndex.loc[:, 't'] = np.around(outlierTrialsIndex['t'], decimals=roundFactor)
        # pd.concat({'outlierT':  pd.Series(np.unique(outlierTrialsIndex['t'])), 'dataT': pd.Series(np.unique(waveformsIndexDF['t']))}, axis='columns')
        outlierTrials.index = pd.MultiIndex.from_frame(outlierTrialsIndex)
        outlierMask = (
                waveformsIndexDF
                .set_index(outlierTrials.index.names)
                .index
                .map(outlierTrials)
            ).to_numpy(dtype=bool)
        #  waveformsIndexDF.set_index(outlierTrials.index.names).index.map(outlierTrials)
        if invertOutlierMask:
            outlierMask = ~outlierMask
        allWaveforms = allWaveforms.loc[~outlierMask, :]
    if manipulateIndex and getMetaData:
        idxLabels = allWaveforms.index.names
        allWaveforms.reset_index(inplace=True)
        # 
        if collapseSizes:
            try:
                allWaveforms.loc[allWaveforms['pedalSizeCat'] == 'XL', 'pedalSizeCat'] = 'L'
                allWaveforms.loc[allWaveforms['pedalSizeCat'] == 'XS', 'pedalSizeCat'] = 'S'
            except Exception:
                traceback.print_exc()
        if makeControlProgram:
            try:
                allWaveforms.loc[allWaveforms[amplitudeColumn] == 0, programColumn] = metaFillerLookup[programColumn]
                allWaveforms.loc[allWaveforms[amplitudeColumn] == 0, electrodeColumn] = metaFillerLookup[electrodeColumn]
            except Exception:
                traceback.print_exc()
        if duplicateControlsByProgram:
            noStimWaveforms = (
                allWaveforms
                .loc[allWaveforms[amplitudeColumn] == 0, :]
                )
            stimWaveforms = (
                allWaveforms
                .loc[allWaveforms[amplitudeColumn] != 0, :]
                .copy()
                )
            uniqProgs = stimWaveforms[programColumn].unique()
            progElecLookup = {}
            for progIdx in uniqProgs:
                theseStimDF = stimWaveforms.loc[
                    stimWaveforms[programColumn] == progIdx,
                    electrodeColumn]
                elecIdx = theseStimDF.iloc[0]
                progElecLookup.update({progIdx: elecIdx})
            #
            if makeControlProgram:
                uniqProgs = np.append(uniqProgs, metaFillerLookup[programColumn])
                progElecLookup.update({metaFillerLookup[programColumn]: metaFillerLookup[electrodeColumn]})
            #
            for progIdx in uniqProgs:
                dummyWaveforms = noStimWaveforms.copy()
                dummyWaveforms.loc[:, programColumn] = progIdx
                dummyWaveforms.loc[:, electrodeColumn] = progElecLookup[progIdx]
                stimWaveforms = pd.concat([stimWaveforms, dummyWaveforms])
            stimWaveforms.reset_index(drop=True, inplace=True)
            allWaveforms = stimWaveforms
        #
        if removeFuzzyName:
            fuzzyNamesBase = [
                i.replace('Fuzzy', '')
                for i in idxLabels
                if 'Fuzzy' in i]
            colRenamer = {n + 'Fuzzy': n for n in fuzzyNamesBase}
            fuzzyNamesBasePresent = [
                i
                for i in fuzzyNamesBase
                if i in allWaveforms.columns]
            allWaveforms.drop(columns=fuzzyNamesBasePresent, inplace=True)
            allWaveforms.rename(columns=colRenamer, inplace=True)
            idxLabels = np.unique(
                [i.replace('Fuzzy', '') for i in idxLabels])
        allWaveforms.set_index(
            list(idxLabels),
            inplace=True)
        if isinstance(allWaveforms.columns, pd.MultiIndex):
            allWaveforms.columns = allWaveforms.columns.remove_unused_levels()
    #
    if transposeToColumns == 'feature':
        zipNames = zip(pd.unique(allWaveforms.columns.get_level_values('feature')).tolist(), unitNames)
        try:
            assert np.all([i == j for i, j in zipNames]), 'columns out of requested order!'
        except Exception:
            traceback.print_exc()
            allWaveforms.reindex(columns=unitNames)
        #
        if getFeatureMetaData is not None:
            assert np.all(allWaveforms.columns == columnsMeta.index)
            allWaveforms.columns = pd.MultiIndex.from_frame(columnsMeta.reset_index())
            allWaveforms = allWaveforms.droplevel(getFeatureMetaData, axis='index')
    if dropNaNs:
        allNaNIndex = allWaveforms.index[allWaveforms.isna().all(axis='columns')]
        # allNaNIndex.to_frame().reset_index(drop=True)
        if allNaNIndex.size > 0:
            allWaveforms.drop(index=allNaNIndex, inplace=True)
        if transposeToColumns == 'bin':
            allWaveforms.dropna(inplace=True, axis='columns')
        elif transposeToColumns == 'feature':
            allWaveforms.dropna(inplace=True, axis='index')
    ##
    if isinstance(allWaveforms.columns, pd.MultiIndex):
        allWaveforms.columns = allWaveforms.columns.remove_unused_levels()
    if 'bin' in allWaveforms.index.names:
        sortIndexBy = ['segment', 't', 'bin']
    else:
        sortIndexBy = ['feature', 'lag', 'segment', 't']
    allWaveforms.sort_index(
        level=sortIndexBy,
        axis='index', inplace=True,
        kind='mergesort', sort_remaining=False)
    allWaveforms.sort_index(
        axis='columns', inplace=True,
        kind='mergesort', sort_remaining=False)
    ##
    if finalIndexMask is not None:
        if True:
            tInfo1 = allWaveforms.index.to_frame().reset_index(drop=True).set_index(['segment', 't'])
            tInfo1 = tInfo1.loc[~tInfo1.index.duplicated(keep='first'), :]
            tInfo2 = finalIndexMask.index.to_frame().reset_index(drop=True).set_index(['segment', 't'])
            tInfo2 = tInfo2.loc[~tInfo2.index.duplicated(keep='first'), :]
            tInfo1.index[~tInfo1.index.duplicated(keep='first')]
        try:
            allWaveforms = allWaveforms.loc[finalIndexMask.to_numpy(), :]
        except  Exception:
            pdb.set_trace()
    # if postProcFun is not None:
    #     allWaveforms = postProcFun(allWaveforms)
    return allWaveforms


def getAsigsAlignedToEvents(
        eventBlock=None, signalBlock=None,
        chansToTrigger=None, chanQuery=None,
        eventName=None, windowSize=None,
        minNReps=None,
        appendToExisting=False,
        checkReferences=True, verbose=False,
        fileName=None, folderPath=None,
        breakdownFolderPath=None,
        chunkSize=None
        ):
    #  get signals from same block as events?
    if signalBlock is None:
        signalBlock = eventBlock
    #  channels to trigger
    if chansToTrigger is None:
        chansToTrigger = listChanNames(
            signalBlock, chanQuery, objType=ChannelIndex, condition='hasAsigs')
    #  allocate block for spiketrains
    masterBlock = Block()
    try:
        masterBlock.name = signalBlock.annotations['neo_name']
        masterBlock.annotate(nix_name=signalBlock.annotations['neo_name'])
    except Exception:
        masterBlock.name = signalBlock.name
        masterBlock.annotate(neo_name=signalBlock.name)
        masterBlock.annotate(nix_name=signalBlock.name)
    #  make channels and units for triggered time series
    for chanName in chansToTrigger:
        chanIdx = ChannelIndex(name=chanName, index=[0])
        chanIdx.annotate(nix_name=chanIdx.name)
        thisUnit = Unit(name=chanName + '#0')
        thisUnit.annotate(nix_name=chanName + '#0')
        chanIdx.units.append(thisUnit)
        thisUnit.channel_index = chanIdx
        masterBlock.channel_indexes.append(chanIdx)
        chanIdx.block = masterBlock
        sigChanIdxList = signalBlock.filter(
            objects=ChannelIndex, name=chanName)
        if len(sigChanIdxList):
            sigChanIdx = sigChanIdxList[0]
            if sigChanIdx.coordinates is not None:
                coordUnits = sigChanIdx.coordinates[0][0].units
                chanIdx.coordinates = np.asarray(sigChanIdx.coordinates) * coordUnits
                thisUnit.annotations['parentChanXCoords'] = float(chanIdx.coordinates[:, 0].magnitude)
                thisUnit.annotations['parentChanYCoords'] = float(chanIdx.coordinates[:, 1].magnitude)
                thisUnit.annotations['parentChanCoordinateUnits'] = '{}'.format(coordUnits)
    totalNSegs = 0
    #  print([evSeg.events[3].name for evSeg in eventBlock.segments])
    allAlignEventsList = []
    for segIdx, eventSeg in enumerate(eventBlock.segments):
        thisEventName = 'seg{}_{}'.format(segIdx, eventName)
        try:
            assert len(eventSeg.filter(name=thisEventName)) == 1
        except Exception:
            traceback.print_exc()
        allEvIn = eventSeg.filter(name=thisEventName)[0]
        if isinstance(allEvIn, EventProxy):
            allAlignEvents = loadObjArrayAnn(allEvIn.load())
        elif isinstance(allEvIn, Event):
            allAlignEvents = allEvIn
        else:
            raise(Exception(
                '{} must be an Event or EventProxy!'
                .format(eventName)))
        allAlignEventsList.append(allAlignEvents)
    allAlignEventsDF = unitSpikeTrainArrayAnnToDF(allAlignEventsList)
    breakDownData, breakDownText, breakDownHtml = hf.calcBreakDown(allAlignEventsDF, breakDownBy=minNReps['categories'])
    breakDownData = breakDownData['count']
    try:
        if breakdownFolderPath is not None:
            breakDownInspectPath = os.path.join(
                breakdownFolderPath,
                fileName.split('_')[0] + '_trialsBreakDown.html')
        else:
            breakDownInspectPath = os.path.join(
                folderPath,
                fileName.split('_')[0] + '_trialsBreakDown.html')
        with open(breakDownInspectPath, 'w') as _f:
            _f.write(breakDownHtml)
    except Exception:
        traceback.print_exc()
    allAlignEventsDF.loc[:, 'keepMask'] = False
    for name, group in allAlignEventsDF.groupby(minNReps['categories']):
        allAlignEventsDF.loc[group.index, 'keepMask'] = (
            breakDownData[name] > minNReps['n'])
    for segIdx, group in allAlignEventsDF.groupby('segment'):
        allAlignEventsList[segIdx].array_annotations['keepMask'] = group['keepMask'].to_numpy()
    #
    for segIdx, eventSeg in enumerate(eventBlock.segments):
        if verbose:
            print(
                'getAsigsAlignedToEvents on segment {} of {}'
                .format(segIdx + 1, len(eventBlock.segments)))
        allAlignEvents = allAlignEventsList[segIdx]
        if chunkSize is None:
            alignEventGroups = [allAlignEvents]
        else:
            nChunks = max(
                int(np.floor(allAlignEvents.shape[0] / chunkSize)),
                1)
            alignEventGroups = []
            for i in range(nChunks):
                if not (i == (nChunks - 1)):
                    # not last one
                    alignEventGroups.append(
                        allAlignEvents[i * chunkSize: (i + 1) * chunkSize])
                else:
                    alignEventGroups.append(
                        allAlignEvents[i * chunkSize:])
        signalSeg = signalBlock.segments[segIdx]
        for subSegIdx, alignEvents in enumerate(alignEventGroups):
            # seg to contain triggered time series
            if verbose:
                print(
                    'getAsigsAlignedToEvents on subSegment {} of {}'
                    .format(subSegIdx + 1, len(alignEventGroups)))
            if not alignEvents.shape[0] > 0:
                continue
            newSeg = Segment(name='seg{}_'.format(int(totalNSegs)))
            newSeg.annotate(nix_name=newSeg.name)
            masterBlock.segments.append(newSeg)
            for chanName in chansToTrigger:
                asigName = 'seg{}_{}'.format(segIdx, chanName)
                if verbose:
                    print(
                        'getAsigsAlignedToEvents on channel {}'
                        .format(chanName))
                assert len(signalSeg.filter(name=asigName)) == 1
                asig = signalSeg.filter(name=asigName)[0]
                nominalWinLen = int(
                    (windowSize[1] - windowSize[0]) *
                    asig.sampling_rate)
                validMask = (
                    ((
                        alignEvents + windowSize[1] +
                        # asig.sampling_rate ** (-1)) < asig.t_stop) &
                        asig.sampling_period) < asig.t_stop) &
                    ((
                        alignEvents + windowSize[0] -
                        # asig.sampling_rate ** (-1)) >= asig.t_start)
                        asig.sampling_period) >= asig.t_start)
                    )
                thisKeepMask = alignEvents.array_annotations['keepMask']
                fullMask = (validMask & thisKeepMask)
                alignEvents = alignEvents[fullMask]
                # array_annotations get sliced with the event, but regular anns do not
                for annName in alignEvents.annotations['arrayAnnNames']:
                    alignEvents.annotations[annName] = (
                        alignEvents.array_annotations[annName])
                if isinstance(asig, AnalogSignalProxy):
                    if checkReferences:
                        da = (
                            asig
                            ._rawio
                            .da_list['blocks'][0]['segments'][segIdx]['data'])
                        print('segIdx {}, asig.name {}'.format(
                            segIdx, asig.name))
                        print('asig._global_channel_indexes = {}'.format(
                            asig._global_channel_indexes))
                        print('asig references {}'.format(
                            da[asig._global_channel_indexes[0]]))
                        try:
                            assert (
                                asig.name
                                in da[asig._global_channel_indexes[0]].name)
                        except Exception:
                            traceback.print_exc()
                    rawWaveforms = [
                        asig.load(
                            time_slice=(
                                t + windowSize[0] - asig.sampling_period,
                                t + windowSize[1]))
                        for t in alignEvents]
                    if any([rW.shape[0] < nominalWinLen for rW in rawWaveforms]):
                        rawWaveforms = [
                            asig.load(
                                time_slice=(
                                    t + windowSize[0] - asig.sampling_period,
                                    t + windowSize[1] + asig.sampling_period))
                            for t in alignEvents]
                elif isinstance(asig, AnalogSignal):
                    rawWaveforms = []
                    for t in alignEvents:
                        asigMask = (
                            (asig.times >= t + windowSize[0]) &
                            (asig.times < t + windowSize[1])
                            )
                        rawWaveforms.append(asig[asigMask[:, np.newaxis]])
                else:
                    raise(Exception('{} must be an AnalogSignal or AnalogSignalProxy!'.format(asigName)))
                #
                samplingRate = asig.sampling_rate
                waveformUnits = rawWaveforms[0].units
                #  fix length if roundoff error
                #  minLen = min([rW.shape[0] for rW in rawWaveforms])
                rawWaveforms = [rW[:nominalWinLen] for rW in rawWaveforms]
                #
                spikeWaveforms = (
                    np.hstack([rW.magnitude for rW in rawWaveforms])
                    .transpose()[:, np.newaxis, :] * waveformUnits
                    )
                #
                thisUnit = masterBlock.filter(
                    objects=Unit, name=chanName + '#0')[0]
                skipEventAnnNames = (
                    ['nix_name', 'neo_name']
                    )
                stAnn = {
                    k: v
                    for k, v in alignEvents.annotations.items()
                    if k not in skipEventAnnNames
                    }
                skipAsigAnnNames = (
                    ['channel_id', 'nix_name', 'neo_name']
                    )
                stAnn.update({
                    k: v
                    for k, v in asig.annotations.items()
                    if k not in skipAsigAnnNames
                })
                thisStName = 'seg{}_{}'.format(int(totalNSegs), thisUnit.name)
                if verbose:
                    print('    making spiketrain {}'.format(thisStName))
                st = SpikeTrain(
                    name=thisStName,
                    times=alignEvents.times,
                    waveforms=spikeWaveforms,
                    t_start=asig.t_start, t_stop=asig.t_stop,
                    left_sweep=windowSize[0] * (-1),
                    sampling_rate=samplingRate,
                    **stAnn)
                st.annotate(nix_name=st.name)
                st.annotations['unitAnnotations'] = json.dumps(
                    thisUnit.annotations.copy())
                thisUnit.spiketrains.append(st)
                newSeg.spiketrains.append(st)
                st.unit = thisUnit
            totalNSegs += 1
    try:
        eventBlock.filter(
            objects=EventProxy)[0]._rawio.file.close()
    except Exception:
        traceback.print_exc()
    if signalBlock is not eventBlock:
        try:
            signalBlock.filter(
                objects=AnalogSignalProxy)[0]._rawio.file.close()
        except Exception:
            traceback.print_exc()
    triggeredPath = os.path.join(
        folderPath, fileName + '.nix')
    if not os.path.exists(triggeredPath):
        appendToExisting = False
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
        if os.path.exists(triggeredPath):
            os.remove(triggeredPath)
        masterBlock = purgeNixAnn(masterBlock)
        writer = NixIO(filename=triggeredPath)
        writer.write_block(masterBlock, use_obj_names=True)
        writer.close()
    return masterBlock


def alignedAsigDFtoSpikeTrain(
        allWaveforms, spikeTrainMeta=None,
        dataBlock=None, matchSamplingRate=True, verbose=False):
    masterBlock = Block()
    if dataBlock is not None:
        masterBlock.name = dataBlock.annotations['neo_name']
        masterBlock.annotate(nix_name=dataBlock.annotations['neo_name'])
    else:
        masterBlock.name = 'block'
    if isinstance(allWaveforms.columns, pd.MultiIndex):
        extraSpikeTrainAnnotations = allWaveforms.columns.to_frame().reset_index(drop=True)
        if 'bin' in extraSpikeTrainAnnotations:
            extraSpikeTrainAnnotations = None
        else:
            assert 'feature' in extraSpikeTrainAnnotations
            featureNames = pd.Index(allWaveforms.columns.get_level_values('feature'), dtype=str)
            allWaveforms.columns = featureNames
            allWaveforms.columns.name = 'feature'
            extraSpikeTrainAnnotations.set_index('feature', inplace=True)
            if 'lag' in extraSpikeTrainAnnotations:
                extraSpikeTrainAnnotations.drop(columns=['lag'], inplace=True)
    else:
        extraSpikeTrainAnnotations = None
    for segIdx, group in allWaveforms.groupby('segment'):
        print('Saving spiketrains for segment {}'.format(segIdx))
        # seg to contain triggered time series
        newSeg = Segment(name='seg{}_'.format(segIdx))
        # newSeg.annotate(nix_name=dataSeg.annotations['neo_name'])
        masterBlock.segments.append(newSeg)
        if dataBlock is not None:
            dataSeg = dataBlock.segments[segIdx]
            exSt = dataSeg.spiketrains[0]
            if isinstance(exSt, SpikeTrainProxy):
                print(
                    'alignedAsigDFtoSpikeTrain basing seg {} on {}'
                    .format(segIdx, exSt.name))
                stProxy = exSt
                exSt = loadStProxy(stProxy)
                exSt = loadObjArrayAnn(exSt)
            if verbose:
                print('exSt.left_sweep is {}'.format(exSt.left_sweep))
            wfBins = ((np.arange(exSt.waveforms.shape[2]) / (exSt.sampling_rate)) - exSt.left_sweep).magnitude
            #
            tUnits = exSt.units
            wvfUnits = exSt.waveforms.units
            t_start = exSt.t_start
            t_stop = exSt.t_stop
            left_sweep = exSt.left_sweep
            sampling_rate = exSt.sampling_rate
        else:
            assert spikeTrainMeta is not None
            tUnits = spikeTrainMeta['units']
            wvfUnits = spikeTrainMeta['wvfUnits']
            t_start = spikeTrainMeta['t_start']
            t_stop = spikeTrainMeta['t_stop']
            left_sweep = spikeTrainMeta['left_sweep']
            sampling_rate = spikeTrainMeta['sampling_rate']
        if isinstance(group.columns, pd.MultiIndex):
            if 'feature' in group.columns.names:
                grouper = group.groupby(['feature', 'lag'], sort=False)
        elif isinstance(group.columns, pd.Index):
            if group.columns.name == 'bin':
                grouper = group.groupby('feature', sort=False)
                colsAre = 'bin'
            elif group.columns.name == 'feature':
                grouper = group.iteritems()
                colsAre = 'feature'
        for featName, featGroup in grouper:
            if verbose:
                print('Saving {}...'.format(featName))
            if featName[-2:] == '#0':
                cleanFeatName = featName
            else:
                cleanFeatName = featName + '#0'
            if segIdx == 0:
                #  allocate units
                chanIdx = ChannelIndex(
                    name=cleanFeatName, index=[0])
                chanIdx.annotate(nix_name=chanIdx.name)
                thisUnit = Unit(name=chanIdx.name)
                thisUnit.annotate(nix_name=chanIdx.name)
                if extraSpikeTrainAnnotations is not None:
                    theseExtraAnn = extraSpikeTrainAnnotations.loc[thisUnit.name, :].to_dict()
                    thisUnit.annotations.update(theseExtraAnn)
                chanIdx.units.append(thisUnit)
                thisUnit.channel_index = chanIdx
                masterBlock.channel_indexes.append(chanIdx)
            else:
                thisUnit = masterBlock.filter(
                    objects=Unit, name=cleanFeatName)[0]
            if colsAre == 'bin':
                spikeWaveformsDF = featGroup
            elif colsAre == 'feature':
                if isinstance(featGroup, pd.Series):
                    featGroup = featGroup.to_frame(name=featName)
                    featGroup.columns.name = 'feature'
                ########
                spikeWaveformsDF = transposeSpikeDF(
                    featGroup, 'bin', fastTranspose=True)
                sortIndexBy = ['feature', 'segment', 't']
                spikeWaveformsDF.sort_index(
                    level=sortIndexBy,
                    axis='index', inplace=True,
                    kind='mergesort', sort_remaining=False)
            if matchSamplingRate:
                if len(spikeWaveformsDF.columns) != len(wfBins):
                    wfDF = spikeWaveformsDF.reset_index(drop=True).T
                    wfDF = hf.interpolateDF(wfDF, wfBins)
                    spikeWaveformsDF = wfDF.T.set_index(spikeWaveformsDF.index)
            spikeWaveforms = spikeWaveformsDF.to_numpy()[:, np.newaxis, :]
            arrAnnDF = spikeWaveformsDF.index.to_frame()
            spikeTimes = arrAnnDF['t']
            arrAnnDF.drop(columns='t', inplace=True)
            arrAnn = {}
            # colsToKeep = arrAnnDF.columns.drop(['originalIndex', 'feature', 'segment', 'lag'])
            colsToKeep = [
                cN
                for cN in arrAnnDF.columns
                if cN not in ['originalIndex', 'feature', 'segment', 'lag']]
            for cName in colsToKeep:
                values = arrAnnDF[cName].to_numpy()
                if isinstance(values[0], str):
                    # print('Converting {} to unicode'.format(cName))
                    values = values.astype('U')
                arrAnn.update({str(cName): values.flatten()})
            arrayAnnNames = {
                'arrayAnnNames': np.asarray([
                    k
                    for k, v in arrAnn.items()],
                    dtype='U')}
            if extraSpikeTrainAnnotations is not None:
                theseExtraAnn = extraSpikeTrainAnnotations.loc[thisUnit.name, :].to_dict()
                arrAnn.update(theseExtraAnn)
            st = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=spikeTimes.to_numpy() * tUnits,
                waveforms=spikeWaveforms * wvfUnits,
                t_start=t_start, t_stop=t_stop,
                left_sweep=left_sweep,
                sampling_rate=sampling_rate,
                **arrAnn, **arrayAnnNames
                )
            if verbose:
                print('st.left_sweep is {}'.format(st.left_sweep))
            st.annotate(nix_name=st.name)
            thisUnit.spiketrains.append(st)
            newSeg.spiketrains.append(st)
            st.unit = thisUnit
    return masterBlock


def dataFrameToAnalogSignals(
        df,
        block=None, seg=None,
        idxT='NSPTime',
        probeName='insTD', samplingRate=500*pq.Hz,
        timeUnits=pq.s, measureUnits=pq.mV,
        dataCol=['channel_0', 'channel_1'],
        useColNames=False, forceColNames=None,
        namePrefix='', nameSuffix='', verbose=False):
    if block is None:
        assert seg is None
        block = Block(name=probeName)
        seg = Segment(name='seg0_' + probeName)
        block.segments.append(seg)
    if verbose:
        print('in dataFrameToAnalogSignals...')
    for idx, colName in enumerate(dataCol):
        if verbose:
            print('    {}'.format(colName))
        if forceColNames is not None:
            chanName = forceColNames[idx]
        elif useColNames:
            chanName = namePrefix + colName + nameSuffix
        else:
            chanName = namePrefix + (probeName.lower() + '{}'.format(idx)) + nameSuffix
        #
        chanIdx = ChannelIndex(
            name=chanName,
            # index=None,
            index=np.asarray([idx]),
            # channel_names=np.asarray([chanName])
            )
        block.channel_indexes.append(chanIdx)
        asig = AnalogSignal(
            df[colName].to_numpy() * measureUnits,
            name='seg0_' + chanName,
            sampling_rate=samplingRate,
            dtype=float,
            # **ann
            )
        if idxT is not None:
            asig.t_start = df[idxT].iloc[0] * timeUnits
        else:
            asig.t_start = df.index[0] * timeUnits
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
            originalDType = type(eventDF[colName].to_numpy()[0]).__name__
            event = Event(
                name=eventName + colName,
                times=eventDF[idxT].to_numpy() * tUnits,
                labels=eventDF[colName].astype(originalDType).to_numpy()
                )
            event.annotate(originalDType=originalDType)
            eventList.append(event)
        return eventList
    else:
        if annCol is None:
            annCol = eventDF.drop(columns=idxT).columns
        event = Event(
            name=eventName,
            times=eventDF[idxT].to_numpy() * tUnits,
            labels=np.asarray(eventDF.index)
            )
        event.annotations.update(
            {
                'arrayAnnNames': [],
                'arrayAnnDTypes': []
                })
        for colName in annCol:
            originalDType = type(eventDF[colName].to_numpy()[0]).__name__
            arrayAnn = eventDF[colName].astype(originalDType).to_numpy()
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
            try:
                values = event.array_annotations['labels']
            except Exception:
                values = event.labels
            if isinstance(values[0], bytes):
                #  event came from hdf, need to recover dtype
                if 'originalDType' in event.annotations:
                    dtypeStr = event.annotations['originalDType'].split(';')[-1]
                    if 'np.' not in dtypeStr:
                        dtypeStr = 'np.' + dtypeStr
                    originalDType = eval(dtypeStr)
                    values = np.asarray(values, dtype=originalDType)
                else:
                    values = np.asarray(values, dtype=np.str)
            #  print(values.dtype)
            eventDict.update({
                event.name: pd.Series(values)})
    eventDict.update({idxT: t})
    return pd.concat(eventDict, axis=1)


def loadSpikeMats(
        dataPath, rasterOpts,
        alignTimes=None, chans=None, loadAll=False,
        absoluteBins=False, transposeSpikeMat=False,
        checkReferences=False,
        aggregateFun=None):

    reader = nixio_fr.NixIO(filename=dataPath)
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
                da = reader.da_list['blocks'][0]['segments'][segIdx]['data'][cIdx]
                print('name {}, da.name {}'.format(chanNames[i], da.name))
                try:
                    assert chanNames[i] in da.name, 'reference problem!!'
                except Exception:
                    traceback.print_exc()
        tStart = reader.get_signal_t_start(
            block_index=0, seg_index=segIdx)
        fs = reader.get_signal_sampling_rate(
            channel_indexes=chanIdx
            )
        sigSize = reader.get_signal_size(
            block_index=0, seg_index=segIdx
            )
        tStop = sigSize / fs + tStart
        #  convert to indices early to avoid floating point problems
        
        intervalIdx = int(round(rasterOpts['binInterval'] * fs))
        #  halfIntervalIdx = int(round(intervalIdx / 2))
        
        widthIdx = int(round(rasterOpts['binWidth'] * fs))
        halfWidthIdx = int(round(widthIdx / 2))
        
        if rasterOpts['smoothKernelWidth'] is not None:
            kernWidthIdx = int(round(rasterOpts['smoothKernelWidth'] * fs))
        
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
                iStart = idxOnset + winStartIdx - int(3 * halfWidthIdx)
                iStop = idxOnset + winStopIdx + int(3 * halfWidthIdx)
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
                try:
                    rawSpikeMat = pd.DataFrame(
                        reader.get_analogsignal_chunk(
                            block_index=0, seg_index=segIdx,
                            i_start=iStart, i_stop=iStop,
                            channel_names=chanNames))
                except Exception:
                    traceback.print_exc()
                    #
                if aggregateFun is None:
                    procSpikeMat = rawSpikeMat.rolling(
                        window=3 * widthIdx, center=True,
                        win_type='gaussian'
                        ).mean(std=halfWidthIdx)
                else:
                    procSpikeMat = rawSpikeMat.rolling(
                        window=widthIdx, center=True
                        ).apply(
                            aggregateFun,
                            raw=True,
                            kwargs={'fs': fs, 'nSamp': widthIdx})
                #
                if rasterOpts['smoothKernelWidth'] is not None:
                    procSpikeMat = (
                        procSpikeMat
                        .rolling(
                            window=3 * kernWidthIdx, center=True,
                            win_type='gaussian')
                        .mean(std=kernWidthIdx/2)
                        .dropna().iloc[::intervalIdx, :]
                    )
                else:
                    procSpikeMat = (
                        procSpikeMat
                        .dropna().iloc[::intervalIdx, :]
                    )

                procSpikeMat.columns = chanNames
                procSpikeMat.columns.name = 'unit'
                if theBins is None:
                    theBins = np.asarray(
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
            #  plt.imshow(rawSpikeMat.to_numpy(), aspect='equal'); plt.show()
    return spikeMats, validTrials


def synchronizeINStoNSP(
        tapTimestampsNSP=None, tapTimestampsINS=None,
        precalculatedFun=None,
        NSPTimeRanges=(None, None),
        td=None, accel=None, insBlock=None, trialSegment=None, degree=1,
        trimSpiketrains=False
        ):
    print('Trial Segment {}'.format(trialSegment))
    if precalculatedFun is None:
        assert ((tapTimestampsNSP is not None) & (tapTimestampsINS is not None))
        # sanity check that the intervals match
        insDiff = tapTimestampsINS.diff().dropna().values
        nspDiff = tapTimestampsNSP.diff().dropna().values
        print('On the INS, the diff() between taps was\n{}'.format(insDiff))
        print('On the NSP, the diff() between taps was\n{}'.format(nspDiff))
        print('This amounts to a msec difference of\n{}'.format(
            (insDiff - nspDiff) * 1e3))
        if (insDiff - nspDiff > 20e-3).any():
            raise(Exception('Tap trains too different!'))
        #
        if degree > 0:
            synchPolyCoeffsINStoNSP = np.polyfit(
                x=tapTimestampsINS.values, y=tapTimestampsNSP.values,
                deg=degree)
        else:
            timeOffset = tapTimestampsNSP.values - tapTimestampsINS.values
            synchPolyCoeffsINStoNSP = np.array([1, np.mean(timeOffset)])
        timeInterpFunINStoNSP = np.poly1d(synchPolyCoeffsINStoNSP)
    else:
        timeInterpFunINStoNSP = precalculatedFun
    if td is not None:
        td.loc[:, 'NSPTime'] = pd.Series(
            timeInterpFunINStoNSP(td['t']), index=td['t'].index)
        td.loc[:, 'NSPTime'] = timeInterpFunINStoNSP(td['t'].to_numpy())
    if accel is not None:
        accel.loc[:, 'NSPTime'] = pd.Series(
            timeInterpFunINStoNSP(accel['t']), index=accel['t'].index)
    if insBlock is not None:
        # allUnits = [st.unit for st in insBlock.segments[0].spiketrains]
        # [un.name for un in insBlock.filter(objects=Unit)]
        for unit in insBlock.filter(objects=Unit):
            tStart = NSPTimeRanges[0]
            tStop = NSPTimeRanges[1]
            uniqueSt = []
            for st in unit.spiketrains:
                if st not in uniqueSt:
                    uniqueSt.append(st)
                else:
                    continue
                print('Synchronizing {}'.format(st.name))
                if len(st.times):
                    segMaskSt = np.array(
                        st.array_annotations['trialSegment'],
                        dtype=np.int) == trialSegment
                    st.magnitude[segMaskSt] = (
                        timeInterpFunINStoNSP(st.times[segMaskSt].magnitude))
                    if trimSpiketrains:
                        print('Trimming spiketrain')
                        #  kludgey fix for weirdness concerning t_start
                        st.t_start = min(tStart, st.times[0] * 0.999)
                        st.t_stop = min(tStop, st.times[-1] * 1.001)
                        validMask = st < st.t_stop
                        if ~validMask.all():
                            print('Deleted some spikes')
                            st = st[validMask]
                            # delete invalid spikes
                            if 'arrayAnnNames' in st.annotations.keys():
                                for key in st.annotations['arrayAnnNames']:
                                    try:
                                        # st.annotations[key] = np.array(st.array_annotations[key])
                                        st.annotations[key] = np.delete(st.annotations[key], ~validMask)
                                    except Exception:
                                        traceback.print_exc()
                else:
                    if trimSpiketrains:
                        st.t_start = tStart
                        st.t_stop = tStop
        #
        allEvents = [
            ev
            for ev in insBlock.filter(objects=Event)
            if ('ins' in ev.name) and ('concatenate' not in ev.name)]
        concatEvents = [
            ev
            for ev in insBlock.filter(objects=Event)
            if ('ins' in ev.name) and ('concatenate' in ev.name)]
        eventsDF = eventsToDataFrame(allEvents, idxT='t')
        newNames = {i: childBaseName(i, 'seg') for i in eventsDF.columns}
        eventsDF.rename(columns=newNames, inplace=True)
        segMask = hf.getStimSerialTrialSegMask(eventsDF, trialSegment)
        evTStart = eventsDF.loc[segMask, 't'].min() * pq.s
        evTStop = eventsDF.loc[segMask, 't'].max() * pq.s
        # print('allEvents[0].shape = {}'.format(allEvents[0].shape))
        # print('allEvents[0].magnitude[segMask][0] = {}'.format(allEvents[0].magnitude[segMask][0]))
        for event in (allEvents + concatEvents):
            if trimSpiketrains:
                thisSegMask = (event.times >= evTStart) & (event.times <= evTStop)
            else:
                thisSegMask = (event.times >= evTStart) & (event.times < evTStop)
            event.magnitude[thisSegMask] = (
                timeInterpFunINStoNSP(event.times[thisSegMask].magnitude))
        # print('allEvents[0].magnitude[segMask][0] = {}'.format(allEvents[0].magnitude[segMask][0]))
        # if len(concatEvents) > trialSegment:
        #     concatEvents[trialSegment].magnitude[:] = timeInterpFunINStoNSP(
        #         concatEvents[trialSegment].times[:].magnitude)
    return td, accel, insBlock, timeInterpFunINStoNSP


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
        segMask = (segBounds['t_start'] * pq.s >= timeSlice[0]) & (
            segBounds['t_stop'] * pq.s <= timeSlice[1])
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
        segMask = (segBounds['t_start'] * pq.s <= timeSlice[0]) | (
            segBounds['t_stop'] * pq.s >= timeSlice[1])
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
                'channelNames': np.asarray(chanIdx.channel_names, dtype=np.str),
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
        reader = nixio_fr.NixIO(filename=filePath)

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
            startTime_s * pq.s,
            (startTime_s + dataLength_s) * pq.s)
    else:
        timeSlice = (None, None)
    segBounds, requestedSegs = findSegsIncluding(block, timeSlice)
    #
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
                channel_indexes=theseRequestedIndices['localIndex'].to_numpy())
            segData = pd.concat((
                    segData,
                    pd.DataFrame(
                        reqData.magnitude, columns=theseElecIds.to_numpy())),
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
        lazy=True, mapDF=None, reduceChannelIndexes=False,
        loadList=None, purgeNixNames=False
        ):
    headerSignalChan = pd.DataFrame(
        rawioReader.header['signal_channels']).set_index('id')
    headerUnitChan = pd.DataFrame(
        rawioReader.header['unit_channels']).set_index('id')
    dataBlock = rawioReader.read_block(
        block_index=block_index, lazy=lazy,
        signal_group_mode=signal_group_mode)
    if dataBlock.name is None:
        if 'neo_name' in dataBlock.annotations:
            dataBlock.name = dataBlock.annotations['neo_name']
    #  on first segment, rename the chan_indexes and units
    seg0 = dataBlock.segments[0]
    # print(sorted([asi.name for asi in seg0.analogsignals]))
    asigLikeList = (
        seg0.filter(objects=AnalogSignalProxy) +
        seg0.filter(objects=AnalogSignal))
    # asigLikeList = seg0.analogsignals
    if mapDF is not None:
        if headerSignalChan.size > 0:
            headerSignalChan.loc[:, 'renamedLabel'] = np.nan
            asigNameChanger = {}
            for nevID in mapDF['nevID']:
                if int(nevID) in headerSignalChan.index:
                    labelFromMap = (
                        mapDF
                        .loc[mapDF['nevID'] == nevID, 'label']
                        .iloc[0])
                    asigNameChanger[
                        headerSignalChan.loc[int(nevID), 'name']] = labelFromMap
                    headerSignalChan.loc[int(nevID), 'renamedLabel'] = labelFromMap
            headerSignalChan.loc[headerSignalChan['renamedLabel'].isna(), 'renamedLabel'] = (
                headerSignalChan.loc[headerSignalChan['renamedLabel'].isna(), 'name'])
        else:
            asigOrigNames = np.unique(
                [i.split('#')[0] for i in headerUnitChan['name']])
            asigNameChanger = {}
            for origName in asigOrigNames:
                # ripple specific
                formattedName = origName.replace('.', '_').replace(' raw', '')
                if mapDF['label'].str.contains(formattedName).any():
                    asigNameChanger[origName] = formattedName
    else:
        asigNameChanger = dict()
    for asig in asigLikeList:
        asigBaseName = childBaseName(asig.name, 'seg')
        if 'renamedLabel' in headerSignalChan.columns:
            asigChId = asig.channel_index.channel_ids[0]
            print('renamed {} to {}'.format(asig.name, headerSignalChan.loc[asigChId, 'renamedLabel']))
            asig.name = headerSignalChan.loc[asigChId, 'renamedLabel']
        elif asigBaseName in asigNameChanger:
            asig.name = asigNameChanger[asigBaseName]
        else:
            asig.name = asigBaseName
        if mapDF is not None:
            if (mapDF['label'] == asig.name).any():
                asig.annotations['xCoords'] = float(mapDF.loc[mapDF['label'] == asig.name, 'xcoords'].iloc[0])
                asig.annotations['yCoords'] = float(mapDF.loc[mapDF['label'] == asig.name, 'ycoords'].iloc[0])
                asig.annotations['zCoords'] = float(mapDF.loc[mapDF['label'] == asig.name, 'zcoords'].iloc[0])
        if 'Channel group ' in asig.channel_index.name:
            if 'renamedLabel' in headerSignalChan.columns:
                asigChId = asig.channel_index.channel_ids[0]
                newChanName = headerSignalChan.loc[asigChId, 'renamedLabel']
            elif asigBaseName in asigNameChanger:
                newChanName = asigNameChanger[asigBaseName]
            else:
                newChanName = asigBaseName
            asig.channel_index.name = newChanName
            if 'neo_name' in asig.channel_index.annotations:
                asig.channel_index.annotations['neo_name'] = newChanName
            if 'nix_name' in asig.channel_index.annotations:
                asig.channel_index.annotations['nix_name'] = newChanName
            if mapDF is not None:
                try:
                    asig.channel_index.coordinates = np.asarray([
                        asig.annotations['xCoords'], asig.annotations['yCoords'], asig.annotations['zCoords']
                    ])[np.newaxis, :] * pq.um
                except Exception:
                    pass
    spikeTrainLikeList = (
        seg0.filter(objects=SpikeTrainProxy) +
        seg0.filter(objects=SpikeTrain))
    # add channels for channelIndex that has no asigs but has spikes
    # print(sorted([ci.name for ci in dataBlock.filter(objects=ChannelIndex)]))
    nExtraChans = 0
    for stp in spikeTrainLikeList:
        stpBaseName = childBaseName(stp.name, 'seg')
        nameParser = re.search(r'ch(\d+)#(\d+)', stpBaseName)
        if nameParser is not None:
            # first time at this unit, rename it
            chanId = int(nameParser.group(1))
            unitId = int(nameParser.group(2))
            if chanId >= 5121:
                isRippleStimChan = True
                chanId = chanId - 5120
            else:
                isRippleStimChan = False
            ####################
            # asigBaseName = headerSignalChan.loc[chanId, 'name']
            # if mapDF is not None:
            #     if asigBaseName in asigNameChanger:
            #         chanIdLabel = (
            #             asigNameChanger[asigBaseName]
            #             if asigBaseName in asigNameChanger
            #             else asigBaseName)
            #     else:
            #         chanIdLabel = asigBaseName
            # else:
            #     chanIdLabel = asigBaseName
            ###################
            # if swapMaps is not None:
            #     nameCandidates = (swapMaps['to'].loc[swapMaps['to']['nevID'] == chanId, 'label']).to_list()
            # elif mapDF is not None:
            #     nameCandidates = (mapDF.loc[mapDF['nevID'] == chanId, 'label']).to_list()
            # else:
            #     nameCandidates = []
            ##############################
            if mapDF is not None:
                nameCandidates = (
                    mapDF
                    .loc[mapDF['nevID'] == chanId, 'label']
                    .to_list())
            else:
                nameCandidates = []
            if len(nameCandidates) == 1:
                chanIdLabel = nameCandidates[0]
            elif chanId in headerSignalChan:
                chanIdLabel = headerSignalChan.loc[chanId, 'name']
            else:
                chanIdLabel = 'ch{}'.format(chanId)
            #
            if isRippleStimChan:
                stp.name = '{}_stim#{}'.format(chanIdLabel, unitId)
            else:
                stp.name = '{}#{}'.format(chanIdLabel, unitId)
            stp.unit.name = stp.name
        ########################################
        # sanitize ripple names ####
        stp.name = stp.name.replace('.', '_').replace(' raw', '')
        stp.unit.name = stp.unit.name.replace('.', '_').replace(' raw', '')
        ###########################################
        if 'ChannelIndex for ' in stp.unit.channel_index.name:
            newChanName = stp.name.replace('_stim#0', '')
            # remove unit #
            newChanName = re.sub(r'#\d', '', newChanName)
            stp.unit.channel_index.name = newChanName
            # units and analogsignals have different channel_indexes when loaded by nix
            # add them to each other's parent list
            allMatchingChIdx = dataBlock.filter(objects=ChannelIndex, name=newChanName)
            if (len(allMatchingChIdx) > 1) and reduceChannelIndexes:
                try:
                    assert len(allMatchingChIdx) == 2
                except:
                    # [ci.name for ci in allMatchingChIdx]
                    # [ci.name for ci in dataBlock.filter(objects=ChannelIndex)]
                    pass
                targetChIdx = [
                    ch
                    for ch in allMatchingChIdx
                    if ch is not stp.unit.channel_index][0]
                oldChIdx = stp.unit.channel_index
                oldChIdx.name = '{}_duplicate'.format(newChanName)
                targetChIdx.units.append(stp.unit)
                stp.unit.channel_index = targetChIdx
                oldChIdx.units.remove(stp.unit)
                if not (len(oldChIdx.units) or len(oldChIdx.analogsignals)):
                    dataBlock.channel_indexes.remove(oldChIdx)
                del oldChIdx
                targetChIdx.create_relationship()
            elif reduceChannelIndexes:
                if newChanName not in headerSignalChan['name']:
                    stp.unit.channel_index.index = np.asarray(
                        [headerSignalChan['name'].size + nExtraChans])
                    stp.unit.channel_index.channel_ids = np.asarray(
                        [headerSignalChan['name'].size + nExtraChans])
                    stp.unit.channel_index.channel_names = np.asarray(
                        [newChanName])
                    nExtraChans += 1
                if 'neo_name' not in allMatchingChIdx[0].annotations:
                    allMatchingChIdx[0].annotations['neo_name'] = allMatchingChIdx[0].name
                if 'nix_name' not in allMatchingChIdx[0].annotations:
                    allMatchingChIdx[0].annotations['nix_name'] = allMatchingChIdx[0].name
        stp.unit.channel_index.name = stp.unit.channel_index.name.replace('.', '_').replace(' raw', '')
    #  rename the children
    typesNeedRenaming = [
        SpikeTrainProxy, AnalogSignalProxy, EventProxy,
        SpikeTrain, AnalogSignal, Event]
    for segIdx, seg in enumerate(dataBlock.segments):
        if seg.name is None:
            seg.name = 'seg{}_'.format(segIdx)
        else:
            if 'seg{}_'.format(segIdx) not in seg.name:
                seg.name = (
                    'seg{}_{}'
                    .format(
                        segIdx,
                        childBaseName(seg.name, 'seg')))
        for objType in typesNeedRenaming:
            for child in seg.filter(objects=objType):
                if 'seg{}_'.format(segIdx) not in child.name:
                    child.name = (
                        'seg{}_{}'
                        .format(
                            segIdx, childBaseName(child.name, 'seg')))
                #  todo: decide if below is needed
                #  elif 'seg' in child.name:
                #      childBaseName = '_'.join(child.name.split('_')[1:])
                #      child.name = 'seg{}_{}'.format(segIdx, childBaseName)
    # [i.name for i in dataBlock.filter(objects=Unit)]
    # [i.name for i in dataBlock.filter(objects=ChannelIndex)]
    # [i.name for i in dataBlock.filter(objects=SpikeTrain)]
    # [i.name for i in dataBlock.filter(objects=SpikeTrainProxy)]
    if lazy:
        for stP in dataBlock.filter(objects=SpikeTrainProxy):
            if 'unitAnnotations' in stP.annotations:
                unAnnStr = stP.annotations['unitAnnotations']
                #  if ';' in unAnnStr:
                #      try:
                #          raise(Warning('loading unit annotations for spiketrain {}\n{}'.format(stp.name, unAnnStr)))
                #      except:
                #          traceback.print_exc()
                #          pass
                #      unAnnStr = unAnnStr.split(';')[0]
                #
                try:
                    stP.unit.annotations.update(json.loads(unAnnStr))
                except Exception:
                    traceback.print_exc()
                    pdb.set_trace()
    if (loadList is not None) and lazy:
        if 'asigs' in loadList:
            loadAsigList(
                dataBlock, listOfAsigProxyNames=loadList['asigs'],
                replaceInParents=True)
        if 'events' in loadList:
            loadEventList(
                dataBlock,
                listOfEventNames=loadList['events'],
                replaceInParents=True)
        if 'spiketrains' in loadList:
            loadSpikeTrainList(
                dataBlock,
                listOfSpikeTrainNames=loadList['spiketrains'],
                replaceInParents=True)
    if purgeNixNames:
        dataBlock = purgeNixAnn(dataBlock)
    return dataBlock


def loadSpikeTrainList(
        dataBlock, listOfSpikeTrainNames=None,
        replaceInParents=True):
    listOfSpikeTrains = []
    if listOfSpikeTrainNames is None:
        listOfSpikeTrainNames = [
            stp.name
            for stp in dataBlock.filter(objects=SpikeTrainProxy)]
    for stP in dataBlock.filter(objects=SpikeTrainProxy):
        if stP.name in listOfSpikeTrainNames:
            st = loadObjArrayAnn(loadStProxy(stP))
            listOfSpikeTrains.append(st)
            if replaceInParents:
                seg = stP.segment
                segStNames = [s.name for s in seg.spiketrains]
                idxInSeg = segStNames.index(stP.name)
                seg.spiketrains[idxInSeg] = st
                #
                unit = stP.unit
                unitStNames = [s.name for s in unit.spiketrains]
                st.unit = unit
                idxInUnit = unitStNames.index(stP.name)
                unit.spiketrains[idxInUnit] = st
    return listOfSpikeTrains


def loadEventList(
        dataBlock,
        listOfEventNames=None, replaceInParents=True):
    listOfEvents = []
    if listOfEventNames is None:
        listOfEventNames = [
            evp.name
            for evp in dataBlock.filter(objects=EventProxy)]
    for evP in dataBlock.filter(objects=EventProxy):
        if evP.name in listOfEventNames:
            ev = loadObjArrayAnn(evP.load())
            listOfEvents.append(ev)
            if replaceInParents:
                seg = evP.segment
                segEvNames = [e.name for e in seg.events]
                idxInSeg = segEvNames.index(evP.name)
                seg.events[idxInSeg] = ev
    return listOfEvents


def loadAsigList(
        dataBlock, listOfAsigProxyNames=None, replaceInParents=True):
    listOfAsigs = []
    if listOfAsigProxyNames is None:
        listOfAsigProxyNames = [
            asigp.name
            for asigp in dataBlock.filter(objects=AnalogSignalProxy)]
    for asigP in dataBlock.filter(objects=AnalogSignalProxy):
        if asigP.name in listOfAsigProxyNames:
            asig = asigP.load()
            asig.annotations = asigP.annotations.copy()
            listOfAsigs.append(asig)
            #
            if replaceInParents:
                seg = asigP.segment
                segAsigNames = [ag.name for ag in seg.analogsignals]
                asig.segment = seg
                idxInSeg = segAsigNames.index(asigP.name)
                seg.analogsignals[idxInSeg] = asig
                #
                chIdx = asigP.channel_index
                chIdxAsigNames = [ag.name for ag in chIdx.analogsignals]
                asig.channel_index = chIdx
                idxInChIdx = chIdxAsigNames.index(asigP.name)
                chIdx.analogsignals[idxInChIdx] = asig
    return listOfAsigs


def addBlockToNIX(
        newBlock, neoSegIdx=[0],
        writeAsigs=True, writeSpikes=True, writeEvents=True,
        asigNameList=None,
        purgeNixNames=False,
        fileName=None,
        folderPath=None,
        nixBlockIdx=0, nixSegIdx=[0],
        ):
    #  base file name
    trialBasePath = os.path.join(folderPath, fileName)
    if writeAsigs:
        # peek at file to ensure compatibility
        reader = nixio_fr.NixIO(filename=trialBasePath + '.nix')
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
    #
    writer = NixIO(filename=trialBasePath + '.nix')
    nixblock = writer.nix_file.blocks[nixBlockIdx]
    nixblockName = nixblock.name
    if 'nix_name' in newBlock.annotations.keys():
        try:
            assert newBlock.annotations['nix_name'] == nixblockName
        except Exception:
            newBlock.annotations['nix_name'] = nixblockName
    else:
        newBlock.annotate(nix_name=nixblockName)
    #
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
        #
        if writeEvents:
            eventList = newSeg.events
            eventOrder = np.argsort([i.name for i in eventList])
            for event in [eventList[i] for i in eventOrder]:
                event = writer._write_event(event, nixblock, nixgroup)
        #
        if writeAsigs:
            asigList = newSeg.filter(objects=AnalogSignal)
            asigOrder = np.argsort([i.name for i in asigList])
            for asig in [asigList[i] for i in asigOrder]:
                if checkCompatible[nixIdx]:
                    assert asig.dtype == forceType[nixIdx]
                    assert asig.sampling_rate == forceFS[nixIdx]
                    #  print('asig.shape[0] = {}'.format(asig.shape[0]))
                    #  print('forceShape[nixIdx] = {}'.format(forceShape[nixIdx]))
                    assert asig.shape[0] == forceShape[nixIdx]
                asig = writer._write_analogsignal(asig, nixblock, nixgroup)
            #  for isig in newSeg.filter(objects=IrregularlySampledSignal):
            #      isig = writer._write_irregularlysampledsignal(
            #          isig, nixblock, nixgroup)
        #
        if writeSpikes:
            stList = newSeg.filter(objects=SpikeTrain)
            stOrder = np.argsort([i.name for i in stList])
            for st in [stList[i] for i in stOrder]:
                st = writer._write_spiketrain(st, nixblock, nixgroup)
    #
    for chanIdx in newBlock.filter(objects=ChannelIndex):
        chanIdx = writer._write_channelindex(chanIdx, nixblock)
        #  auto descends into units inside of _write_channelindex
    writer._create_source_links(newBlock, nixblock)
    writer.close()
    print('Done adding block to Nix.')
    return newBlock


def loadStProxy(stProxy):
    try:
        st = stProxy.load(
            magnitude_mode='rescaled',
            load_waveforms=True)
    except Exception:
        st = stProxy.load(
            magnitude_mode='rescaled',
            load_waveforms=False)
        st.waveforms = np.asarray([]).reshape((0, 0, 0))*pq.mV
    return st


def preproc(
        fileName='Trial001',
        rawFolderPath='./',
        outputFolderPath='./', mapDF=None,
        # swapMaps=None,
        electrodeArrayName='utah',
        fillOverflow=True, removeJumps=True,
        removeMeanAcross=False,
        linearDetrend=False,
        interpolateOutliers=False, calcOutliers=False,
        outlierMaskFilterOpts=None,
        outlierThreshold=1,
        calcArtifactTrace=False,
        motorEncoderMask=None,
        calcAverageLFP=False,
        eventInfo=None,
        spikeSourceType='', spikePath=None,
        chunkSize=1800, equalChunks=True, chunkList=None, chunkOffset=0,
        writeMode='rw',
        signal_group_mode='split-all', trialInfo=None,
        asigNameList=None, ainpNameList=None, nameSuffix='',
        saveFromAsigNameList=True,
        calcRigEvents=True, normalizeByImpedance=False,
        LFPFilterOpts=None, encoderCountPerDegree=180e2,
        outlierRemovalDebugFlag=False, impedanceFilePath=None
        ):
    #  base file name
    rawBasePath = os.path.join(rawFolderPath, fileName)
    outputFilePath = os.path.join(
        outputFolderPath,
        fileName + nameSuffix + '.nix')
    if os.path.exists(outputFilePath):
        os.remove(outputFilePath)
    #  instantiate reader, get metadata
    print('Loading\n{}\n'.format(rawBasePath))
    reader = BlackrockIO(
        filename=rawBasePath, nsx_to_load=5)
    reader.parse_header()
    # metadata = reader.header
    #  absolute section index
    dummyBlock = readBlockFixNames(
        reader,
        block_index=0, lazy=True,
        signal_group_mode=signal_group_mode,
        mapDF=mapDF, reduceChannelIndexes=True,
        )
    segLen = dummyBlock.segments[0].analogsignals[0].shape[0] / (
        dummyBlock.segments[0].analogsignals[0].sampling_rate)
    nChunks = math.ceil(segLen / chunkSize)
    #
    if equalChunks:
        actualChunkSize = (segLen / nChunks).magnitude
    else:
        actualChunkSize = chunkSize
    if chunkList is None:
        chunkList = range(nChunks)
    chunkingMetadata = {}
    for chunkIdx in chunkList:
        print('preproc on chunk {}'.format(chunkIdx))
        #  instantiate spike reader if requested
        if spikeSourceType == 'tdc':
            if spikePath is None:
                spikePath = os.path.join(
                    outputFolderPath, 'tdc_' + fileName,
                    'tdc_' + fileName + '.nix')
            print('loading {}'.format(spikePath))
            spikeReader = nixio_fr.NixIO(filename=spikePath)
        else:
            spikeReader = None
        #  absolute section index
        block = readBlockFixNames(
            reader,
            block_index=0, lazy=True,
            signal_group_mode=signal_group_mode,
            mapDF=mapDF, reduceChannelIndexes=True,
            purgeNixNames=True
            # swapMaps=swapMaps
            )
        if spikeReader is not None:
            spikeBlock = readBlockFixNames(
                spikeReader, block_index=0, lazy=True,
                signal_group_mode=signal_group_mode,
                mapDF=mapDF, reduceChannelIndexes=True,
                purgeNixNames=True
                # swapMaps=swapMaps
                )
            spikeBlock = purgeNixAnn(spikeBlock)
        else:
            spikeBlock = None
        #
        #  instantiate writer
        if (nChunks == 1) or (len(chunkList) == 1):
            partNameSuffix = ""
            thisChunkOutFilePath = outputFilePath
        else:
            partNameSuffix = '_pt{:0>3}'.format(chunkIdx)
            thisChunkOutFilePath = (
                outputFilePath
                .replace('.nix', partNameSuffix + '.nix'))
        #
        if os.path.exists(thisChunkOutFilePath):
            os.remove(thisChunkOutFilePath)
        writer = NixIO(
            filename=thisChunkOutFilePath, mode=writeMode)
        chunkTStart = chunkIdx * actualChunkSize + chunkOffset
        chunkTStop = (chunkIdx + 1) * actualChunkSize + chunkOffset
        chunkingMetadata[chunkIdx] = {
            'filename': thisChunkOutFilePath,
            'partNameSuffix': partNameSuffix,
            'chunkTStart': chunkTStart,
            'chunkTStop': chunkTStop}
        block.annotate(chunkTStart=chunkTStart)
        block.annotate(chunkTStop=chunkTStop)
        block.annotate(
            recDatetimeStr=(
                block
                .rec_datetime
                .replace(tzinfo=timezone.utc)
                .isoformat())
            )
        #
        preprocBlockToNix(
            block, writer,
            chunkTStart=chunkTStart,
            chunkTStop=chunkTStop,
            fillOverflow=fillOverflow,
            removeJumps=removeJumps,
            interpolateOutliers=interpolateOutliers,
            calcOutliers=calcOutliers,
            outlierThreshold=outlierThreshold,
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            calcArtifactTrace=calcArtifactTrace,
            linearDetrend=linearDetrend,
            motorEncoderMask=motorEncoderMask,
            electrodeArrayName=electrodeArrayName,
            calcAverageLFP=calcAverageLFP,
            eventInfo=eventInfo,
            asigNameList=asigNameList, ainpNameList=ainpNameList,
            saveFromAsigNameList=saveFromAsigNameList,
            spikeSourceType=spikeSourceType,
            spikeBlock=spikeBlock,
            calcRigEvents=calcRigEvents,
            normalizeByImpedance=normalizeByImpedance,
            removeMeanAcross=removeMeanAcross,
            LFPFilterOpts=LFPFilterOpts,
            encoderCountPerDegree=encoderCountPerDegree,
            outlierRemovalDebugFlag=outlierRemovalDebugFlag,
            impedanceFilePath=impedanceFilePath,
            )
        #### diagnostics
        diagnosticFolder = os.path.join(
            outputFolderPath,
            'figures', 'preprocDiagnostics',
            # fileName + nameSuffix + partNameSuffix
            )
        if not os.path.exists(diagnosticFolder):
            os.makedirs(diagnosticFolder, exist_ok=True)
        asigDiagnostics = {}
        outlierDiagnostics = {}
        diagnosticText = ''
        for asig in block.filter(objects=AnalogSignal):
            annNames = ['mean_removal_r2', 'mean_removal_group']
            for annName in annNames:
                if annName in asig.annotations:
                    if asig.name not in asigDiagnostics:
                        asigDiagnostics[asig.name] = {}
                    asigDiagnostics[asig.name].update({
                        annName: asig.annotations[annName]})
            annNames = [
                'outlierProportion', 'nDim',
                'noveltyThreshold', 'outlierThreshold'
                ]
            for annName in annNames:
                if annName in asig.annotations:
                    if asig.name not in outlierDiagnostics:
                        outlierDiagnostics[asig.name] = {}
                    outlierDiagnostics[asig.name].update({
                        annName: '{}'.format(asig.annotations[annName])
                    })
        if removeMeanAcross:
            asigDiagnosticsDF = pd.DataFrame(asigDiagnostics).T
            asigDiagnosticsDF.sort_values(by='mean_removal_r2', inplace=True)
            diagnosticText += '<h2>LFP Diagnostics</h2>\n'
            diagnosticText += asigDiagnosticsDF.to_html()
            fig, ax = plt.subplots()
            sns.distplot(asigDiagnosticsDF['mean_removal_r2'], ax=ax)
            ax.set_ylabel('Count of analog signals')
            ax.set_xlabel('R^2 of regressing mean against signal')
            fig.savefig(os.path.join(
                    diagnosticFolder,
                    fileName + nameSuffix + partNameSuffix + '_meanRemovalR2.png'
                ))
        if interpolateOutliers:
            outlierDiagnosticsDF = pd.DataFrame(outlierDiagnostics).T
            diagnosticText += '<h2>Outlier Diagnostics</h2>\n'
            diagnosticText += outlierDiagnosticsDF.to_html()
        diagnosticTextPath = os.path.join(
            diagnosticFolder,
            fileName + nameSuffix + partNameSuffix + '_asigDiagnostics.html'
            )
        with open(diagnosticTextPath, 'w') as _f:
            _f.write(diagnosticText)
        writer.close()
    chunkingInfoPath = os.path.join(
        outputFolderPath,
        fileName + nameSuffix +
        '_chunkingInfo.json'
        )
    if os.path.exists(chunkingInfoPath):
        os.remove(chunkingInfoPath)
    with open(chunkingInfoPath, 'w') as f:
        json.dump(chunkingMetadata, f)
    return


def preprocBlockToNix(
        block, writer,
        chunkTStart=None,
        chunkTStop=None,
        eventInfo=None,
        fillOverflow=False, calcAverageLFP=False,
        interpolateOutliers=False, calcOutliers=False,
        calcArtifactTrace=False,
        outlierMaskFilterOpts=None,
        useMeanToCenter=False,   # mean center? median center?
        linearDetrend=False,
        zScoreEachTrace=False,
        outlierThreshold=1,
        motorEncoderMask=None,
        electrodeArrayName='utah',
        removeJumps=False, trackMemory=True,
        asigNameList=None, ainpNameList=None,
        saveFromAsigNameList=True,
        spikeSourceType='', spikeBlock=None,
        calcRigEvents=True,
        normalizeByImpedance=True,
        impedanceFilePath=None,
        removeMeanAcross=False,
        LFPFilterOpts=None, encoderCountPerDegree=180e2,
        outlierRemovalDebugFlag=False,
        ):
    #  prune out nev spike placeholders
    #  (will get added back on a chunk by chunk basis,
    #  if not pruning units)
    if spikeSourceType == 'nev':
        pruneOutUnits = False
    else:
        pruneOutUnits = True
    #
    for chanIdx in block.channel_indexes:
        if chanIdx.units:
            for unit in chanIdx.units:
                if unit.spiketrains:
                    unit.spiketrains = []
            if pruneOutUnits:
                chanIdx.units = []
    #
    if spikeBlock is not None:
        for chanIdx in spikeBlock.channel_indexes:
            if chanIdx.units:
                for unit in chanIdx.units:
                    if unit.spiketrains:
                        unit.spiketrains = []
    #  precalculate new segment
    seg = block.segments[0]
    #  remove chanIndexes assigned to units; makes more sense to
    #  only use chanIdx for asigs and spikes on that asig
    #  block.channel_indexes = (
    #      [chanIdx for chanIdx in block.channel_indexes if (
    #          chanIdx.analogsignals)])
    if calcAverageLFP:
        lastIndex = len(block.channel_indexes)
        lastID = block.channel_indexes[-1].channel_ids[0] + 1
        if asigNameList is None:
            asigNameList = [
                [
                    childBaseName(a.name, 'seg')
                    for a in seg.analogsignals
                    if not (('ainp' in a.name) or ('analog' in a.name))]
                ]
        nMeanChans = len(asigNameList)
        #
        meanChIdxList = []
        for meanChIdx in range(nMeanChans):
            tempChIdx = ChannelIndex(
                index=[lastIndex + meanChIdx],
                channel_names=['{}_rawAverage_{}'.format(electrodeArrayName, meanChIdx)],
                channel_ids=[lastID + meanChIdx],
                name='{}_rawAverage_{}'.format(electrodeArrayName, meanChIdx),
                file_origin=block.channel_indexes[-1].file_origin
                )
            tempChIdx.merge_annotations(block.channel_indexes[-1])
            tempChIdx.annotations['nix_name'] = '{}_rawAverage_{}'.format(electrodeArrayName, meanChIdx)
            block.channel_indexes.append(tempChIdx)
            meanChIdxList.append(tempChIdx)
            lastIndex += 1
            lastID += 1
        lastIndex = len(block.channel_indexes)
        lastID = block.channel_indexes[-1].channel_ids[0] + 1
        # if calcArtifactTrace:
        if True:
            artChIdxList = []
            for artChIdx in range(nMeanChans):
                tempChIdx = ChannelIndex(
                    index=[lastIndex + artChIdx],
                    channel_names=['{}_artifact_{}'.format(electrodeArrayName, artChIdx)],
                    channel_ids=[lastID + artChIdx],
                    name='{}_artifact_{}'.format(electrodeArrayName, artChIdx),
                    file_origin=block.channel_indexes[-1].file_origin
                    )
                tempChIdx.merge_annotations(block.channel_indexes[-1])
                tempChIdx.annotations['nix_name'] = '{}_artifact_{}'.format(electrodeArrayName, artChIdx)
                block.channel_indexes.append(tempChIdx)
                artChIdxList.append(tempChIdx)
                lastIndex += 1
                lastID += 1
        # if calcOutliers:
        if True:
            devChIdxList = []
            for devChIdx in range(nMeanChans):
                tempChIdx = ChannelIndex(
                    index=[lastIndex + devChIdx],
                    channel_names=['{}_deviation_{}'.format(electrodeArrayName, devChIdx)],
                    channel_ids=[lastID + devChIdx],
                    name='{}_deviation_{}'.format(electrodeArrayName, devChIdx),
                    file_origin=block.channel_indexes[-1].file_origin
                    )
                tempChIdx.merge_annotations(block.channel_indexes[-1])
                tempChIdx.annotations['nix_name'] = '{}_deviation_{}'.format(electrodeArrayName, devChIdx)
                block.channel_indexes.append(tempChIdx)
                devChIdxList.append(tempChIdx)
                lastIndex += 1
                lastID += 1
            smDevChIdxList = []
            for devChIdx in range(nMeanChans):
                tempChIdx = ChannelIndex(
                    index=[lastIndex + devChIdx],
                    channel_names=['{}_smoothed_deviation_{}'.format(electrodeArrayName, devChIdx)],
                    channel_ids=[lastID + devChIdx],
                    name='{}_smoothed_deviation_{}'.format(electrodeArrayName, devChIdx),
                    file_origin=block.channel_indexes[-1].file_origin
                    )
                tempChIdx.merge_annotations(block.channel_indexes[-1])
                tempChIdx.annotations['nix_name'] = '{}_smoothed_deviation_{}'.format(electrodeArrayName, devChIdx)
                block.channel_indexes.append(tempChIdx)
                smDevChIdxList.append(tempChIdx)
                lastIndex += 1
                lastID += 1
            outMaskChIdxList = []
            for outMaskChIdx in range(nMeanChans):
                tempChIdx = ChannelIndex(
                    index=[lastIndex + outMaskChIdx],
                    channel_names=['{}_outlierMask_{}'.format(
                        electrodeArrayName, outMaskChIdx)],
                    channel_ids=[lastID + outMaskChIdx],
                    name='{}_outlierMask_{}'.format(
                        electrodeArrayName, outMaskChIdx),
                    file_origin=block.channel_indexes[-1].file_origin
                    )
                tempChIdx.merge_annotations(block.channel_indexes[-1])
                tempChIdx.annotations['nix_name'] = '{}_outlierMask_{}'.format(electrodeArrayName, outMaskChIdx)
                block.channel_indexes.append(tempChIdx)
                outMaskChIdxList.append(tempChIdx)
                lastIndex += 1
                lastID += 1
    #  delete asig and irsig proxies from channel index list
    for metaIdx, chanIdx in enumerate(block.channel_indexes):
        if chanIdx.analogsignals:
            chanIdx.analogsignals = []
        if chanIdx.irregularlysampledsignals:
            chanIdx.irregularlysampledsignals = []
    newSeg = Segment(
            index=0, name=seg.name,
            description=seg.description,
            file_origin=seg.file_origin,
            file_datetime=seg.file_datetime,
            rec_datetime=seg.rec_datetime,
            **seg.annotations
        )
    block.segments = [newSeg]
    block, nixblock = writer.write_block_meta(block)
    # descend into Segments
    if impedanceFilePath is not None:
        try:
            impedances = prb_meta.getLatestImpedance(
                block=block, impedanceFilePath=impedanceFilePath)
            averageImpedance = impedances['impedance'].median()
        except Exception:
            traceback.print_exc()
    # for segIdx, seg in enumerate(oldSegList):
    if spikeBlock is not None:
        spikeSeg = spikeBlock.segments[0]
    else:
        spikeSeg = seg
    #
    if trackMemory:
        print('memory usage: {:.1f} MB'.format(
            prf.memory_usage_psutil()))
    newSeg, nixgroup = writer._write_segment_meta(newSeg, nixblock)
    #  trim down list of analog signals if necessary
    asigNameListSeg = []
    if (removeMeanAcross or calcAverageLFP):
        meanGroups = {}
    for subListIdx, subList in enumerate(asigNameList):
        subListSeg = [
            'seg{}_{}'.format(0, a)
            for a in subList]
        asigNameListSeg += subListSeg
        if (removeMeanAcross or calcAverageLFP):
            meanGroups[subListIdx] = subListSeg
    aSigList = []
    # [asig.name for asig in seg.analogsignals]
    for a in seg.analogsignals:
        # if np.any([n in a.name for n in asigNameListSeg]):
        if a.name in asigNameListSeg:
            aSigList.append(a)
    if ainpNameList is not None:
        ainpNameListSeg = [
            'seg{}_{}'.format(0, a)
            for a in ainpNameList]
        ainpList = []
        for a in seg.analogsignals:
            if np.any([n == a.name for n in ainpNameListSeg]):
                ainpList.append(a)
    else:
        ainpList = [
            a
            for a in seg.analogsignals
            if (('ainp' in a.name) or ('analog' in a.name))]
        ainpNameListSeg = [a.name for a in aSigList]
    nAsigs = len(aSigList)
    # [asi.name for asi in aSigList]
    if LFPFilterOpts is not None:
        def filterFun(sig, filterCoeffs=None):
            # sig[:] = signal.sosfiltfilt(
            sig[:] = signal.sosfilt(
                filterCoeffs, sig.magnitude.flatten())[:, np.newaxis] * sig.units
            return sig
        filterCoeffs = hf.makeFilterCoeffsSOS(
            LFPFilterOpts, float(seg.analogsignals[0].sampling_rate))
        '''if False:
            fig, ax1, ax2 = hf.plotFilterResponse(
                filterCoeffs,
                float(seg.analogsignals[0].sampling_rate))
            fig2, ax3, ax4 = hf.plotFilterImpulseResponse(
                LFPFilterOpts,
                float(seg.analogsignals[0].sampling_rate))
            plt.show()'''
    # first pass through asigs, if removing mean across channels
    if (removeMeanAcross or calcAverageLFP):
        # [asigP.name for asigP in seg.analogsignals]
        for aSigIdx, aSigProxy in enumerate(seg.analogsignals):
            if aSigIdx == 0:
                # check bounds
                tStart = max(chunkTStart * pq.s, aSigProxy.t_start)
                tStop = min(chunkTStop * pq.s, aSigProxy.t_stop)
            loadThisOne = (aSigProxy in aSigList)
            if loadThisOne:
                if trackMemory:
                    print(
                        'Extracting asig for mean, memory usage: {:.1f} MB'.format(
                            prf.memory_usage_psutil()))
                chanIdx = aSigProxy.channel_index
                asig = aSigProxy.load(
                    time_slice=(tStart, tStop),
                    magnitude_mode='rescaled')
                if 'tempLFPStore' not in locals():
                    try:
                        tempLFPStore = pd.DataFrame(
                            np.zeros(
                                (asig.shape[0], nAsigs),
                                dtype=float),
                            columns=asigNameListSeg)
                    except:
                        traceback.print_exc()
                        pdb.set_trace()
                if 'dummyAsig' not in locals():
                    dummyAsig = asig.copy()
                #  perform requested preproc operations
                #  if LFPFilterOpts is not None:
                #      asig[:] = filterFun(
                #          asig, filterCoeffs=filterCoeffs)
                if normalizeByImpedance:
                    elNmMatchMsk = impedances['elec'] == chanIdx.name
                    '''
                    asig.magnitude[:] = (
                        (asig.magnitude - np.median(asig.magnitude)) /
                        np.min(
                            impedances.loc[elNmMatchMsk, 'impedance']
                            ))
                    '''
                    asig.magnitude[:] = (
                        (asig.magnitude) * averageImpedance /
                        np.min(
                            impedances.loc[elNmMatchMsk, 'impedance']
                            ))
                # if fillOverflow:
                #     # fill in overflow:
                #     '''
                #     timeSection['data'], overflowMask = hf.fillInOverflow(
                #         timeSection['data'], fillMethod = 'average')
                #     badData.update({'overflow': overflowMask})
                #     '''
                #     pass
                # if removeJumps:
                #     # find unusual jumps in derivative or amplitude
                #     '''
                #     timeSection['data'], newBadData = hf.fillInJumps(timeSection['data'],
                #     timeSection['samp_per_s'], smoothing_ms = 0.5, nStdDiff = 50,
                #     nStdAmp = 100)
                #     badData.update(newBadData)
                #     '''
                #     pass
                tempLFPStore.loc[:, aSigProxy.name] = asig.magnitude.flatten()
                del asig
                gc.collect()
        # end of first pass
        if (removeMeanAcross or calcAverageLFP):
            centerLFP = np.zeros(
                (tempLFPStore.shape[0], len(asigNameList)),
                dtype=float)
            spreadLFP = np.zeros(
                (tempLFPStore.shape[0], len(asigNameList)),
                dtype=float)
            # if calcOutliers:
            if True:
                if outlierMaskFilterOpts is not None:
                    filterCoeffsOutlierMask = hf.makeFilterCoeffsSOS(
                        outlierMaskFilterOpts, float(dummyAsig.sampling_rate))
                lfpDeviation = np.zeros(
                    (tempLFPStore.shape[0], len(asigNameList)),
                    dtype=float)
                smoothedDeviation = np.zeros(
                    (tempLFPStore.shape[0], len(asigNameList)),
                    dtype=float)
                outlierMask = np.zeros(
                    (tempLFPStore.shape[0], len(asigNameList)),
                    dtype=bool)
                outlierMetadata = {}
            # if calcArtifactTrace:
            if True:
                artifactSignal = np.zeros(
                    (tempLFPStore.shape[0], len(asigNameList)),
                    dtype=float)
            ###############
            # tempLFPStore.iloc[:, 0] = np.nan  # for debugging axes
            #############
            plotDevFilterDebug = False
            if plotDevFilterDebug:
                try:
                    devFiltDebugMask = (dummyAsig.times > 90 * pq.s) & (dummyAsig.times < 92 * pq.s)
                except Exception:
                    traceback.print_exc()
                plotColIdx = 1
                ddfFig, ddfAx = plt.subplots(len(asigNameList), 1)
                ddfFig2, ddfAx2 = plt.subplots()
                ddfFig3, ddfAx3 = plt.subplots(
                    1, len(asigNameList),
                    sharey=True)
                if len(asigNameList) == 1:
                    ddfAx = np.asarray([ddfAx])
                    ddfAx3 = np.asarray([ddfAx3])
            for subListIdx, subList in enumerate(asigNameList):
                columnsForThisGroup = meanGroups[subListIdx]
                if trackMemory:
                    print(
                        'asig group {}: calculating mean, memory usage: {:.1f} MB'.format(
                            subListIdx, prf.memory_usage_psutil()))
                    print('this group contains\n{}'.format(columnsForThisGroup))
                if plotDevFilterDebug:
                    ddfAx3[subListIdx].plot(
                        dummyAsig.times[devFiltDebugMask],
                        tempLFPStore.loc[:, columnsForThisGroup].iloc[devFiltDebugMask, plotColIdx],
                        label='original ch'
                        )
                if fillOverflow:
                    print('Filling overflow...')
                    # fill in overflow:
                    tempLFPStore.loc[:, columnsForThisGroup], pltHandles = hf.fillInOverflow2(
                        tempLFPStore.loc[:, columnsForThisGroup].to_numpy(),
                        overFlowFillType='average',
                        overFlowThreshold=8000,
                        debuggingPlots=plotDevFilterDebug
                        )
                    if plotDevFilterDebug:
                        pltHandles['ax'].set_title('ch grp {}'.format(subListIdx))
                        ddfAx3[subListIdx].plot(
                            dummyAsig.times[devFiltDebugMask],
                            tempLFPStore.loc[:, columnsForThisGroup].iloc[devFiltDebugMask, plotColIdx],
                            label='filled ch'
                            )
                # zscore of each trace
                if zScoreEachTrace:
                    print('About to calculate zscore of each trace (along columns) for prelim outlier detection')
                    columnZScore = pd.DataFrame(
                        stats.zscore(
                            tempLFPStore.loc[:, columnsForThisGroup],
                            axis=1),
                        index=tempLFPStore.index,
                        columns=columnsForThisGroup
                        )
                    excludeFromMeanMask = columnZScore.abs() > 6
                    if useMeanToCenter:
                        centerLFP[:, subListIdx] = (
                            tempLFPStore
                            .loc[:, columnsForThisGroup]
                            .mask(excludeFromMeanMask)
                            .mean(axis=1).to_numpy()
                            )
                    else:
                        centerLFP[:, subListIdx] = (
                            tempLFPStore
                            .loc[:, columnsForThisGroup]
                            .mask(excludeFromMeanMask)
                            .median(axis=1).to_numpy()
                            )
                else:
                    if useMeanToCenter:
                        centerLFP[:, subListIdx] = (
                            tempLFPStore
                            .loc[:, columnsForThisGroup]
                            .mean(axis=1).to_numpy()
                            )
                    else:
                        centerLFP[:, subListIdx] = (
                            tempLFPStore
                            .loc[:, columnsForThisGroup]
                            .median(axis=1).to_numpy()
                            )
                if calcArtifactTrace:
                    print('Applying hampel filter for artifact detection')
                    outlier_mask, scaled_absolute_deviation, (fig, ax) = hampel(
                        tempLFPStore, window_size=31, thresh=3, imputation=False,
                        average_across_channels=True, debug_plots=False
                        )
                    artifactSignal[:, subListIdx] = scaled_absolute_deviation.to_numpy()
                    '''
                    if LFPFilterOpts is not None:
                        print('applying LFPFilterOpts to cached asigs for artifact ID')
                        # tempLFPStore.loc[:, columnsForThisGroup] = signal.sosfilt(
                        try:
                            tempLFPStore.loc[:, columnsForThisGroup] = signal.sosfiltfilt(
                                filterCoeffs, tempLFPStore.loc[:, columnsForThisGroup].to_numpy(),
                                axis=0)
                        except:
                            traceback.print_exc()
                    if useMeanToCenter:
                        tempCenter = (
                            tempLFPStore
                            .loc[:, columnsForThisGroup]
                            .mean(axis=1).diff().fillna(0)
                            )
                    else:
                        tempCenter = (
                            tempLFPStore
                            .loc[:, columnsForThisGroup]
                            .median(axis=1).diff().fillna(0)
                            )
                    artifactSignal[:, subListIdx] = np.abs(stats.zscore(tempCenter.to_numpy()))
                    '''
                if calcOutliers:
                    if plotDevFilterDebug:
                        ddfAx[subListIdx].plot(
                            dummyAsig.times[devFiltDebugMask],
                            centerLFP[devFiltDebugMask, subListIdx],
                            label='mean of ch group'
                            )
                    # filter the traces, if needed
                    if LFPFilterOpts is not None:
                        print('applying LFPFilterOpts to cached asigs before outlier detection')
                        # tempLFPStore.loc[:, columnsForThisGroup] = signal.sosfiltfilt(
                        tempLFPStore.loc[:, columnsForThisGroup] = signal.sosfilt(
                            filterCoeffs, tempLFPStore.loc[:, columnsForThisGroup],
                            axis=0)
                        if plotDevFilterDebug:
                            ddfAx3[subListIdx].plot(
                                dummyAsig.times[devFiltDebugMask],
                                tempLFPStore.loc[:, columnsForThisGroup].iloc[devFiltDebugMask, plotColIdx],
                                label='filtered ch'
                                )
                    ##################################
                    print('Whitening cached traces before outlier detection')
                    whitenByPCA = False
                    whitenByMahalanobisDist = False
                    if whitenByPCA:
                        projector = PCA(
                            n_components=None, whiten=True)
                        pcs = projector.fit_transform(
                            tempLFPStore.loc[:, columnsForThisGroup])
                        explVarMask = (
                            np.cumsum(projector.explained_variance_ratio_) < 1 - 1e-2)
                        explVarMask[0] = True  # (keep at least 1)
                        pcs = pcs[:, explVarMask]
                        nDim = pcs.shape[1]
                        lfpDeviation[:, subListIdx] = (pcs ** 2).sum(axis=1)
                    elif whitenByMahalanobisDist:  # whiten by mahalanobis distance
                        est = EmpiricalCovariance()
                        est.fit(tempLFPStore.loc[:, columnsForThisGroup].to_numpy())
                        lfpDeviation[:, subListIdx] = est.mahalanobis(
                            tempLFPStore.loc[:, columnsForThisGroup].to_numpy())
                        nDim = tempLFPStore.loc[:, columnsForThisGroup].shape[1]
                    else: # use sum of artifact signal
                        lfpDeviation[:, subListIdx] = artifactSignal[:, subListIdx]
                        # nDim = tempLFPStore.loc[:, columnsForThisGroup].shape[1]
                        nDim = 1
                    #
                    transformedDeviation = stats.norm.isf(stats.chi2.sf(lfpDeviation[:, subListIdx], nDim))
                    infMask = np.isinf(transformedDeviation)
                    if infMask.any():
                        transformedDeviation[infMask] = transformedDeviation[~infMask].max()
                    debugProbaTrans = False
                    if debugProbaTrans:
                        fig, ax = plt.subplots()
                        tAx = ax.twinx()
                        plotMask = (dummyAsig.times >= 60 * pq.s) & (dummyAsig.times < 95 * pq.s)
                        ax.plot(dummyAsig.times[plotMask], transformedDeviation[plotMask], c='b', label='transformed deviation')
                        tAx.plot(dummyAsig.times[plotMask], lfpDeviation[plotMask, subListIdx], c='r', label='original deviation')
                        ax.legend(loc='upper left')
                        tAx.legend(loc='upper right')
                        plt.show()
                    lfpDeviation[:, subListIdx] = transformedDeviation
                    noveltyThreshold = stats.norm.interval(outlierThreshold)[1]
                    # chi2Bounds = stats.chi2.interval(outlierThreshold, nDim)
                    # lfpDeviation[:, subListIdx] = lfpDeviation[:, subListIdx] / chi2Bounds[1]
                    # print('nDim = {}, chi2Lim = {}'.format(nDim, chi2Bounds))
                    # noveltyThreshold = 1
                    #
                    outlierMetadata[subListIdx] = {
                        'nDim': nDim,
                        'noveltyThreshold': noveltyThreshold,
                        'outlierThreshold': outlierThreshold
                        }
                    # smoothedDeviation = signal.sosfilt(
                    print('Smoothing deviation')
                    tempSmDev = signal.sosfiltfilt(
                        filterCoeffsOutlierMask, lfpDeviation[:, subListIdx])
                    smoothedDeviation[:, subListIdx] = tempSmDev
                    if plotDevFilterDebug:
                        ddfAx[subListIdx].plot(
                            dummyAsig.times[devFiltDebugMask],
                            lfpDeviation[devFiltDebugMask, subListIdx],
                            label='original deviation (ch grp {})'.format(subListIdx))
                        ddfAx[subListIdx].plot(
                            dummyAsig.times[devFiltDebugMask],
                            smoothedDeviation[devFiltDebugMask, subListIdx],
                            label='filtered deviation (ch grp {})'.format(subListIdx))
                    ##
                    print('Calculating outlier mask')
                    outlierMask[:, subListIdx] = (
                        smoothedDeviation[:, subListIdx] > noveltyThreshold)
                    if plotDevFilterDebug:
                        ddfAx[subListIdx].axhline(noveltyThreshold, c='r')
            if plotDevFilterDebug and calcOutliers:
                for subListIdx, subList in enumerate(asigNameList):
                    ddfAx[subListIdx].legend(loc='upper right')
                    ddfAx[subListIdx].set_title('Deviation')
                    ddfAx3[subListIdx].legend(loc='upper right')
                    ddfAx3[subListIdx].set_title('Example channel')
                    ddfAx2.plot(
                        dummyAsig.times[devFiltDebugMask],
                        smoothedDeviation[devFiltDebugMask, subListIdx],
                        label='ch grp {}'.format(subListIdx))
                    ddfAx2.set_title('Smoothed Deviation')
                ddfAx2.legend(loc='upper right')
                plt.show()
            #############
            del tempLFPStore
            gc.collect()
    if (removeMeanAcross or calcAverageLFP):
        for mIdx, meanChIdx in enumerate(meanChIdxList):
            meanAsig = AnalogSignal(
                centerLFP[:, mIdx],
                units=dummyAsig.units,
                sampling_rate=dummyAsig.sampling_rate,
                # name='seg{}_{}'.format(idx, meanChIdx.name)
                name='seg{}_{}'.format(0, meanChIdx.name),
                t_start=tStart
            )
            # assign ownership to containers
            meanChIdx.analogsignals.append(meanAsig)
            newSeg.analogsignals.append(meanAsig)
            # assign parent to children
            meanChIdx.create_relationship()
            newSeg.create_relationship()
            # write out to file
            if LFPFilterOpts is not None:
                meanAsig[:] = filterFun(
                    meanAsig, filterCoeffs=filterCoeffs)
            meanAsig = writer._write_analogsignal(
                meanAsig, nixblock, nixgroup)
        # if calcArtifactTrace:
        if True:
            for mIdx, artChIdx in enumerate(artChIdxList):
                artAsig = AnalogSignal(
                    artifactSignal[:, mIdx],
                    units=dummyAsig.units,
                    sampling_rate=dummyAsig.sampling_rate,
                    # name='seg{}_{}'.format(idx, devChIdx.name)
                    name='seg{}_{}'.format(0, artChIdx.name),
                    t_start=tStart
                    )
                # assign ownership to containers
                artChIdx.analogsignals.append(artAsig)
                newSeg.analogsignals.append(artAsig)
                # assign parent to children
                artChIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                artAsig = writer._write_analogsignal(
                    artAsig, nixblock, nixgroup)
                #########################################################
        # if calcOutliers:
        if True:
            for mIdx, devChIdx in enumerate(devChIdxList):
                devAsig = AnalogSignal(
                    lfpDeviation[:, mIdx],
                    units=dummyAsig.units,
                    sampling_rate=dummyAsig.sampling_rate,
                    # name='seg{}_{}'.format(idx, devChIdx.name)
                    name='seg{}_{}'.format(0, devChIdx.name),
                    t_start=tStart
                    )
                # assign ownership to containers
                devChIdx.analogsignals.append(devAsig)
                newSeg.analogsignals.append(devAsig)
                # assign parent to children
                devChIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                devAsig = writer._write_analogsignal(
                    devAsig, nixblock, nixgroup)
                #########################################################
            for mIdx, smDevChIdx in enumerate(smDevChIdxList):
                smDevAsig = AnalogSignal(
                    smoothedDeviation[:, mIdx],
                    units=dummyAsig.units,
                    sampling_rate=dummyAsig.sampling_rate,
                    # name='seg{}_{}'.format(idx, devChIdx.name)
                    name='seg{}_{}'.format(0, smDevChIdx.name),
                    t_start=tStart
                    )
                # assign ownership to containers
                smDevChIdx.analogsignals.append(smDevAsig)
                newSeg.analogsignals.append(smDevAsig)
                # assign parent to children
                smDevChIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                smDevAsig = writer._write_analogsignal(
                    smDevAsig, nixblock, nixgroup)
                #########################################################
            for mIdx, outMaskChIdx in enumerate(outMaskChIdxList):
                outMaskAsig = AnalogSignal(
                    outlierMask[:, mIdx],
                    units=dummyAsig.units,
                    sampling_rate=dummyAsig.sampling_rate,
                    # name='seg{}_{}'.format(idx, outMaskChIdx.name)
                    name='seg{}_{}'.format(0, outMaskChIdx.name),
                    t_start=tStart, dtype=float
                    )
                outMaskAsig.annotations['outlierProportion'] = np.mean(outlierMask[:, mIdx])
                if calcOutliers:
                    outMaskAsig.annotations.update(outlierMetadata[mIdx])
                # assign ownership to containers
                outMaskChIdx.analogsignals.append(outMaskAsig)
                newSeg.analogsignals.append(outMaskAsig)
                # assign parent to children
                outMaskChIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                outMaskAsig = writer._write_analogsignal(
                    outMaskAsig, nixblock, nixgroup)
        #
        w0 = 60
        bandQ = 20
        bw = w0/bandQ
        noiseSos = signal.iirfilter(
            N=8, Wn=[w0 - bw/2, w0 + bw/2],
            btype='band', ftype='butter',
            analog=False, fs=float(dummyAsig.sampling_rate),
            output='sos')
        # signal.hilbert does not have an option to zero pad
        nextLen = fftpack.helper.next_fast_len(dummyAsig.shape[0])
        deficit = int(nextLen - dummyAsig.shape[0])
        lDef = int(np.floor(deficit / 2))
        rDef = int(np.ceil(deficit / 2)) + 1
        temp = np.pad(
            dummyAsig.magnitude.flatten(),
            (lDef, rDef), mode='constant')
        # lineNoise = signal.sosfiltfilt(
        lineNoise = signal.sosfilt(
            noiseSos, temp, axis=0)
        lineNoiseH = signal.hilbert(lineNoise)
        lineNoise = lineNoise[lDef:-rDef]
        lineNoiseH = lineNoiseH[lDef:-rDef]
        lineNoisePhase = np.angle(lineNoiseH)
        lineNoisePhaseDF = pd.DataFrame(
            lineNoisePhase,
            index=dummyAsig.times,
            columns=['phase']
            )
        plotHilbert = False
        if plotHilbert:
            lineNoiseFreq = (
                np.diff(np.unwrap(lineNoisePhase)) /
                (2.0*np.pi) * float(dummyAsig.sampling_rate))
            lineNoiseEnvelope = np.abs(lineNoiseH)
            i1 = 300000; i2 = 330000
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(dummyAsig.times[devFiltDebugMask], dummyAsig.magnitude[devFiltDebugMask, :])
            ax[0].plot(dummyAsig.times[devFiltDebugMask], lineNoise[devFiltDebugMask])
            ax[0].plot(dummyAsig.times[devFiltDebugMask], lineNoiseEnvelope[devFiltDebugMask])
            axFr = ax[1].twinx()
            ax[1].plot(
                dummyAsig.times[devFiltDebugMask], lineNoisePhase[devFiltDebugMask],
                c='r', label='phase')
            ax[1].legend()
            axFr.plot(
                dummyAsig.times[devFiltDebugMask], lineNoiseFreq[devFiltDebugMask],
                label='freq')
            axFr.set_ylim([59, 61])
            axFr.legend()
            plt.show()
    # second pass through asigs, to save
    for aSigIdx, aSigProxy in enumerate(seg.analogsignals):
        if aSigIdx == 0:
            # check bounds
            tStart = max(chunkTStart * pq.s, aSigProxy.t_start)
            tStop = min(chunkTStop * pq.s, aSigProxy.t_stop)
        loadThisOne = (
            (saveFromAsigNameList and (aSigProxy in aSigList)) or
            (aSigProxy in ainpList)
            )
        if loadThisOne:
            if trackMemory:
                print('writing asig {} ({}) memory usage: {:.1f} MB'.format(
                    aSigIdx, aSigProxy.name, prf.memory_usage_psutil()))
            chanIdx = aSigProxy.channel_index
            asig = aSigProxy.load(
                time_slice=(tStart, tStop),
                magnitude_mode='rescaled')
            #  link AnalogSignal and ID providing channel_index
            asig.channel_index = chanIdx
            #  perform requested preproc operations
            if 'impedances' in locals():
                elNmMatchMsk = impedances['elec'] == chanIdx.name
                if elNmMatchMsk.any():
                    originalImpedance = np.min(
                        impedances.loc[elNmMatchMsk, 'impedance']
                        )
                    asig.annotations['originalImpedance'] = originalImpedance
                    if normalizeByImpedance and (aSigProxy not in ainpList):
                        '''
                        asig.magnitude[:] = (
                            (asig.magnitude - np.median(asig.magnitude)) /
                            np.min(
                                impedances.loc[elNmMatchMsk, 'impedance']
                                )
                            )
                        '''
                        print('Normalizing {} by {} kOhms'.format(asig.name, originalImpedance))
                        asig.magnitude[:] = (
                            (asig.magnitude * averageImpedance) / originalImpedance
                            )
            if fillOverflow:
                # fill in overflow:
                asig.magnitude[:], _ = hf.fillInOverflow2(
                    asig.magnitude[:],
                    overFlowFillType='average',
                    overFlowThreshold=8000,
                    debuggingPlots=False
                    )
            if removeJumps:
                # find unusual jumps in derivative or amplitude
                '''
                timeSection['data'], newBadData = hf.fillInJumps(timeSection['data'],
                timeSection['samp_per_s'], smoothing_ms = 0.5, nStdDiff = 50,
                nStdAmp = 100)
                badData.update(newBadData)
                '''
                pass
            if calcAverageLFP and (aSigProxy not in ainpList):
                for k, cols in meanGroups.items():
                    if asig.name in cols:
                        whichColumnToSubtract = k
                noiseModel = np.polyfit(
                    centerLFP[:, whichColumnToSubtract],
                    asig.magnitude.flatten(), 1, full=True)
                rSq = 1 - noiseModel[1][0] / np.sum(asig.magnitude.flatten() ** 2)
                asig.annotations['mean_removal_r2'] = rSq
                asig.annotations['mean_removal_group'] = whichColumnToSubtract
                if linearDetrend:
                    noiseTerm = np.polyval(
                        noiseModel[0],
                        centerLFP[:, whichColumnToSubtract])
                else:
                    noiseTerm = centerLFP[:, whichColumnToSubtract]
                ###
                plotMeanSubtraction = False
                if plotMeanSubtraction:
                    i1 = 300000; i2 = 330000
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(asig.times[devFiltDebugMask], asig.magnitude[devFiltDebugMask, :], label='channel')
                    ax.plot(asig.times[devFiltDebugMask], centerLFP[devFiltDebugMask, whichColumnToSubtract], label='mean')
                    ax.plot(asig.times[devFiltDebugMask], noiseTerm[devFiltDebugMask], label='adjusted mean')
                    ax.legend()
                    plt.show()
                ###
                if removeMeanAcross:
                    asig.magnitude[:] = np.atleast_2d(
                        asig.magnitude.flatten() - noiseTerm).transpose()
                    # asig.magnitude[:] = (
                    #     asig.magnitude - np.median(asig.magnitude))
            if (LFPFilterOpts is not None) and (aSigProxy not in ainpList):
                asig.magnitude[:] = filterFun(asig, filterCoeffs=filterCoeffs)
            if (interpolateOutliers) and (aSigProxy not in ainpList) and (not outlierRemovalDebugFlag):
                for k, cols in meanGroups.items():
                    if asig.name in cols:
                        whichColumnToSubtract = k
                tempSer = pd.Series(asig.magnitude.flatten())
                tempSer.loc[outlierMask[:, whichColumnToSubtract]] = np.nan
                tempSer = (
                    tempSer
                    .interpolate(method='linear', limit_area='inside')
                    .fillna(method='ffill')
                    .fillna(method='bfill')
                    )
                asig.magnitude[:, 0] = tempSer.to_numpy()
            if (aSigProxy in aSigList) or (aSigProxy in ainpList):
                # assign ownership to containers
                chanIdx.analogsignals.append(asig)
                newSeg.analogsignals.append(asig)
                # assign parent to children
                chanIdx.create_relationship()
                newSeg.create_relationship()
                # write out to file
                asig = writer._write_analogsignal(
                    asig, nixblock, nixgroup)
            del asig
            gc.collect()
    for irSigIdx, irSigProxy in enumerate(
            seg.irregularlysampledsignals):
        chanIdx = irSigProxy.channel_index
        #
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
    #
    if len(spikeSourceType):
        for stIdx, stProxy in enumerate(spikeSeg.spiketrains):
            if trackMemory:
                print('writing spiketrains mem usage: {}'.format(
                    prf.memory_usage_psutil()))
            unit = stProxy.unit
            st = loadStProxy(stProxy)
            #  have to manually slice tStop and tStart because
            #  array annotations are not saved natively in the nix file
            #  (we're getting them as plain annotations)
            timeMask = np.asarray(
                (st.times >= tStart) & (st.times < tStop),
                dtype=bool)
            try:
                if 'arrayAnnNames' in st.annotations:
                    for key in st.annotations['arrayAnnNames']:
                        st.annotations[key] = np.asarray(
                            st.annotations[key])[timeMask]
                st = st[timeMask]
                st.t_start = tStart
                st.t_stop = tStop
            except Exception:
                traceback.print_exc()
            #  tdc may or may not have the same channel ids, but
            #  it will have consistent channel names
            nameParser = re.search(
                r'([a-zA-Z0-9]*)#(\d*)', unit.name)
            chanLabel = nameParser.group(1)
            unitId = nameParser.group(2)
            #
            chIdxName = unit.name.replace('_stim', '').split('#')[0]
            chanIdx = block.filter(objects=ChannelIndex, name=chIdxName)[0]
            # [i.name for i in block.filter(objects=ChannelIndex)]
            # [i.name for i in spikeBlock.filter(objects=Unit)]
            #  print(unit.name)
            if not (unit in chanIdx.units):
                # first time at this unit, add to its chanIdx
                unit.channel_index = chanIdx
                chanIdx.units.append(unit)
            #  except Exception:
            #      traceback.print_exc()
            st.name = 'seg{}_{}'.format(0, unit.name)
            # st.name = 'seg{}_{}'.format(idx, unit.name)
            #  link SpikeTrain and ID providing unit
            if calcAverageLFP:
                if 'arrayAnnNames' in st.annotations:
                    st.annotations['arrayAnnNames'] = list(st.annotations['arrayAnnNames'])
                else:
                    st.annotations['arrayAnnNames'] = []
                st.annotations['arrayAnnNames'].append('phase60hz')
                phase60hz = hf.interpolateDF(
                    lineNoisePhaseDF,
                    newX=st.times, columns=['phase']).to_numpy().flatten()
                st.annotations.update({'phase60hz': phase60hz})
                plotPhaseDist = False
                if plotPhaseDist:
                    sns.distplot(phase60hz)
                    plt.show()
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
    #  process proprio trial related events
    if calcRigEvents:
        print('Processing rig events...')
        analogData = []
        for key, value in eventInfo['inputIDs'].items():
            searchName = 'seg{}_'.format(0) + value
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
        if motorEncoderMask is not None:
            ainpData = ainpAsig.load(
                time_slice=(tStart, tStop),
                magnitude_mode='rescaled')
            ainpTime = ainpData.times.magnitude
            meTimeMask = np.zeros_like(ainpTime, dtype=bool)
            for meTimeBounds in motorEncoderMask:
                meTimeMask = (
                    meTimeMask |
                    (
                        (ainpTime > meTimeBounds[0]) &
                        (ainpTime < meTimeBounds[1])
                        )
                    )
            columnsToOverride = ['A-', 'A+', 'B-', 'B+', 'Z-', 'Z+']
            for colName in columnsToOverride:
                motorData.loc[~meTimeMask, colName] = motorData.loc[:, colName].quantile(q=0.05)
            del ainpData, ainpTime
            gc.collect()
        motorData = mea.processMotorData(
            motorData, ainpAsig.sampling_rate.magnitude,
            encoderCountPerDegree=encoderCountPerDegree
            )
        keepCols = [
            'position', 'velocity', 'velocityCat',
            'rightBut_int', 'leftBut_int',
            'rightLED_int', 'leftLED_int', 'simiTrigs_int']
        for colName in keepCols:
            if trackMemory:
                print('writing motorData memory usage: {:.1f} MB'.format(
                    prf.memory_usage_psutil()))
            chanIdx = ChannelIndex(
                name=colName,
                index=np.asarray([0]),
                channel_names=np.asarray([0]))
            block.channel_indexes.append(chanIdx)
            motorAsig = AnalogSignal(
                motorData[colName].to_numpy() * pq.mV,
                name=colName,
                sampling_rate=ainpAsig.sampling_rate,
                dtype=float)
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
        _, trialEvents = mea.getTrials(
            motorData, ainpAsig.sampling_rate.magnitude,
            float(tStart.magnitude), trialType=None)
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
                print(
                    'writing motor events memory usage: {:.1f} MB'
                    .format(prf.memory_usage_psutil()))
            event.segment = newSeg
            newSeg.events.append(event)
            newSeg.create_relationship()
            # write out to file
            event = writer._write_event(event, nixblock, nixgroup)
            del event
            gc.collect()
        del trialEvents, eventList
    #
    for eventProxy in seg.events:
        event = eventProxy.load(
            time_slice=(tStart, tStop))
        event.t_start = tStart
        event.t_stop = tStop
        event.segment = newSeg
        newSeg.events.append(event)
        newSeg.create_relationship()
        # write out to file
        event = writer._write_event(event, nixblock, nixgroup)
        del event
        gc.collect()
    #
    for epochProxy in seg.epochs:
        epoch = epochProxy.load(
            time_slice=(tStart, tStop))
        epoch.t_start = tStart
        epoch.t_stop = tStop
        epoch.segment = newSeg
        newSeg.events.append(epoch)
        newSeg.create_relationship()
        # write out to file
        epoch = writer._write_epoch(epoch, nixblock, nixgroup)
        del epoch
        gc.collect()
    #
    chanIdxDiscardNames = []
    # descend into ChannelIndexes
    for chanIdx in block.channel_indexes:
        if chanIdx.analogsignals or chanIdx.units:
            try:
                chanIdx = writer._write_channelindex(chanIdx, nixblock)
                print('Wrote ch {}'.format(chanIdx.name))
            except:
                traceback.print_exc()
                pdb.set_trace()
        else:
            print('      Did not write ch {}'.format(chanIdx.name))
            chanIdxDiscardNames.append(chanIdx.name)
    block.channel_indexes = [
        i
        for i in block.channel_indexes
        if i.name not in chanIdxDiscardNames
        ]
    writer._create_source_links(block, nixblock)
    return


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
    #
    spikesAndEvents = []
    returnObj = []
    if container is not None:
        #  need the line below! (RD: don't remember why, consider removing)
        container.create_relationship()
        #
        spikesAndEvents += (
            container.filter(objects=SpikeTrain) +
            container.filter(objects=Event)
            )
        returnObj.append(container)
    if trainList is not None:
        spikesAndEvents += trainList
        returnObj.append(trainList)
    #
    if len(returnObj) == 1:
        returnObj = returnObj[0]
    else:
        returnObj = tuple(returnObj)
    #
    for st in spikesAndEvents:
        st = loadObjArrayAnn(st)
    return returnObj


def loadObjArrayAnn(st):
    if 'arrayAnnNames' in st.annotations.keys():
        if isinstance(st.annotations['arrayAnnNames'], str):
            st.annotations['arrayAnnNames'] = [st.annotations['arrayAnnNames']]
        elif isinstance(st.annotations['arrayAnnNames'], tuple):
            st.annotations['arrayAnnNames'] = [i for i in st.annotations['arrayAnnNames']]
        #
        arrayAnnNames = copy(st.annotations['arrayAnnNames'])
        for key in arrayAnnNames:
            if key not in st.annotations:
                print('Warning! {} not found in st.annotations'.format(key))
                st.annotations['arrayAnnNames'].remove(key)
                continue
            #  fromRaw, the ann come back as tuple, need to recast
            try:
                if len(st.times) == 1:
                    st.annotations[key] = np.atleast_1d(st.annotations[key]).flatten()
                st.array_annotations.update(
                    {key: np.asarray(st.annotations[key])})
                st.annotations[key] = np.asarray(st.annotations[key])
            except Exception:
                print('Error with {}'.format(st.name))
                traceback.print_exc()
    if hasattr(st, 'waveforms'):
        if st.waveforms is None:
            st.waveforms = np.asarray([]).reshape((0, 0, 0)) * pq.mV
        elif not len(st.waveforms):
            st.waveforms = np.asarray([]).reshape((0, 0, 0)) * pq.mV
    return st


def loadWithArrayAnn(
        dataPath, fromRaw=False,
        mapDF=None, reduceChannelIndexes=False):
    if fromRaw:
        reader = nixio_fr.NixIO(filename=dataPath)
        block = readBlockFixNames(
            reader, lazy=False,
            mapDF=mapDF,
            reduceChannelIndexes=reduceChannelIndexes)
    else:
        reader = NixIO(filename=dataPath)
        block = reader.read_block()
        # [un.name for un in block.filter(objects=Unit)]
        # [len(un.spiketrains) for un in block.filter(objects=Unit)]
    #
    block = loadContainerArrayAnn(container=block)
    #
    if fromRaw:
        reader.file.close()
    else:
        reader.close()
    return block


def blockFromPath(
        dataPath, lazy=False, mapDF=None,
        reduceChannelIndexes=False, loadList=None,
        purgeNixNames=False, chunkingInfoPath=None):
    print('blockFromPath(lazy={}) loading {} ...'.format(lazy, dataPath))
    chunkingMetadata = None
    if chunkingInfoPath is not None:
        if os.path.exists(chunkingInfoPath):
            with open(chunkingInfoPath, 'r') as f:
                chunkingMetadata = json.load(f)
    if chunkingMetadata is None:
        chunkingMetadata = {
            '0': {
                'filename': dataPath,
                'partNameSuffix': '',
                'chunkTStart': 0,
                'chunkTStop': 'NaN'
            }}
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        print('blockFromPath: chunkIdx = {}'.format(chunkIdxStr))
        thisDataPath = chunkMeta['filename']
        assert os.path.exists(thisDataPath), '{}\nDoes not exist!'.format(thisDataPath)
        if idx == 0:
            if lazy:
                dataReader = nixio_fr.NixIO(
                    filename=thisDataPath)
                dataBlock = readBlockFixNames(
                    dataReader, lazy=lazy, mapDF=mapDF,
                    reduceChannelIndexes=reduceChannelIndexes,
                    purgeNixNames=purgeNixNames, loadList=loadList)
            else:
                dataReader = None
                dataBlock = loadWithArrayAnn(thisDataPath)
        else:
            if lazy:
                dataReader2 = nixio_fr.NixIO(
                    filename=thisDataPath)
                dataBlock2 = readBlockFixNames(
                    dataReader2, lazy=lazy, mapDF=mapDF,
                    reduceChannelIndexes=reduceChannelIndexes, loadList=loadList)
            else:
                dataReader2 = None
                dataBlock2 = loadWithArrayAnn(thisDataPath)
            maxSegIdx = len(dataBlock.segments)
            typesNeedRenaming = [
                SpikeTrainProxy, AnalogSignalProxy, EventProxy,
                SpikeTrain, AnalogSignal, Event]
            for segIdx, seg in enumerate(dataBlock2.segments):
                if seg.name is None:
                    seg.name = 'seg{}_'.format(maxSegIdx + segIdx)
                else:
                    if 'seg{}_'.format(maxSegIdx + segIdx) not in seg.name:
                        seg.name = (
                            'seg{}_{}'
                            .format(
                                maxSegIdx + segIdx,
                                childBaseName(seg.name, 'seg')))
                for objType in typesNeedRenaming:
                    for child in seg.filter(objects=objType):
                        if 'seg{}_'.format(maxSegIdx + segIdx) not in child.name:
                            child.name = (
                                'seg{}_{}'
                                .format(
                                    maxSegIdx + segIdx, childBaseName(child.name, 'seg')))
            dataBlock.merge(dataBlock2)
    return dataReader, dataBlock


def calcBinarizedArray(
        dataBlock, samplingRate,
        binnedSpikePath=None,
        saveToFile=True, matchT=None):
    #
    spikeMatBlock = Block(name=dataBlock.name + '_binarized')
    spikeMatBlock.merge_annotations(dataBlock)
    #
    allSpikeTrains = [
        i for i in dataBlock.filter(objects=SpikeTrain)]
    #
    for st in allSpikeTrains:
        chanList = spikeMatBlock.filter(
            objects=ChannelIndex, name=st.unit.name)
        if not len(chanList):
            chanIdx = ChannelIndex(name=st.unit.name, index=np.asarray([0]))
            #  print(chanIdx.name)
            spikeMatBlock.channel_indexes.append(chanIdx)
            thisUnit = Unit(name=st.unit.name)
            chanIdx.units.append(thisUnit)
            thisUnit.channel_index = chanIdx
    #
    for segIdx, seg in enumerate(dataBlock.segments):
        newSeg = Segment(name='seg{}_{}'.format(segIdx, spikeMatBlock.name))
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
            t_stop=tStop + samplingRate ** -1) * 0
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
                    t_stop=tStop + samplingRate ** -1)
                spikeMatBlock.segments[segIdx].spiketrains.append(st)
                #  to do: link st to spikematblock's chidx and units
                assert len(chanIdx.filter(objects=Unit)) == 1
                thisUnit = chanIdx.filter(objects=Unit)[0]
                thisUnit.spiketrains.append(st)
                st.unit = thisUnit
                st.segment = spikeMatBlock.segments[segIdx]
            else:
                print('{} has no spikes'.format(st.name))
                stBin = dummyBin
            skipStAnnNames = [
                'nix_name', 'neo_name', 'arrayAnnNames']
            if 'arrayAnnNames' in st.annotations:
                skipStAnnNames += list(st.annotations['arrayAnnNames'])
            asigAnn = {
                k: v
                for k, v in st.annotations.items()
                if k not in skipStAnnNames
                }
            asig = AnalogSignal(
                stBin * samplingRate,
                name='seg{}_{}_raster'.format(segIdx, st.unit.name),
                sampling_rate=samplingRate,
                dtype=np.int,
                **asigAnn)
            if matchT is not None:
                asig = asig[:matchT.shape[0], :]
            asig.t_start = tStart
            asig.annotate(binWidth=1 / samplingRate.magnitude)
            chanIdx.analogsignals.append(asig)
            asig.channel_index = chanIdx
            spikeMatBlock.segments[segIdx].analogsignals.append(asig)
    #
    for chanIdx in spikeMatBlock.channel_indexes:
        chanIdx.name = chanIdx.name + '_raster'
    #
    spikeMatBlock.create_relationship()
    spikeMatBlock = purgeNixAnn(spikeMatBlock)
    if saveToFile:
        if os.path.exists(binnedSpikePath):
            os.remove(binnedSpikePath)
        writer = NixIO(filename=binnedSpikePath)
        writer.write_block(spikeMatBlock, use_obj_names=True)
        writer.close()
    return spikeMatBlock


def calcFR(
        binnedPath, dataPath,
        suffix='fr', aggregateFun=None,
        chanNames=None, rasterOpts=None, verbose=False
        ):
    print('Loading rasters...')
    masterSpikeMats, _ = loadSpikeMats(
        binnedPath, rasterOpts,
        aggregateFun=aggregateFun,
        chans=chanNames,
        loadAll=True, checkReferences=False)
    print('Loading data file...')
    dataReader = nixio_fr.NixIO(
        filename=dataPath)
    dataBlock = dataReader.read_block(
        block_index=0, lazy=True,
        signal_group_mode='split-all')
    masterBlock = Block()
    masterBlock.name = dataBlock.annotations['neo_name']
    #
    for segIdx, segSpikeMat in masterSpikeMats.items():
        print('Calculating FR for segment {}'.format(segIdx))
        spikeMatDF = segSpikeMat.reset_index().rename(
            columns={'bin': 't'})

        dataSeg = dataBlock.segments[segIdx]
        dummyAsig = dataSeg.filter(
            objects=AnalogSignalProxy)[0].load(channel_indexes=[0])
        samplingRate = dummyAsig.sampling_rate
        newT = dummyAsig.times.magnitude
        spikeMatDF['t'] = spikeMatDF['t'] + newT[0]

        segSpikeMatInterp = hf.interpolateDF(
            spikeMatDF, pd.Series(newT),
            kind='linear', fill_value=(0, 0),
            x='t')
        spikeMatBlockInterp = dataFrameToAnalogSignals(
            segSpikeMatInterp,
            idxT='t', useColNames=True,
            dataCol=segSpikeMatInterp.drop(columns='t').columns,
            samplingRate=samplingRate)
        spikeMatBlockInterp.name = dataBlock.annotations['neo_name']
        spikeMatBlockInterp.annotate(
            nix_name=dataBlock.annotations['neo_name'])
        spikeMatBlockInterp.segments[0].name = dataSeg.annotations['neo_name']
        spikeMatBlockInterp.segments[0].annotate(
            nix_name=dataSeg.annotations['neo_name'])
        asigList = spikeMatBlockInterp.filter(objects=AnalogSignal)
        for asig in asigList:
            asig.annotate(binWidth=rasterOpts['binWidth'])
            if '_raster' in asig.name:
                asig.name = asig.name.replace('_raster', '_' + suffix)
            asig.name = 'seg{}_{}'.format(segIdx, childBaseName(asig.name, 'seg'))
            asig.annotate(nix_name=asig.name)
        chanIdxList = spikeMatBlockInterp.filter(objects=ChannelIndex)
        for chanIdx in chanIdxList:
            if '_raster' in chanIdx.name:
                chanIdx.name = chanIdx.name.replace('_raster', '_' + suffix)
            chanIdx.annotate(nix_name=chanIdx.name)

        # masterBlock.merge(spikeMatBlockInterp)
        frBlockPath = dataPath.replace('_analyze.nix', '_fr.nix')
        writer = NixIO(filename=frBlockPath)
        writer.write_block(spikeMatBlockInterp, use_obj_names=True)
        writer.close()
    #
    dataReader.file.close()
    return masterBlock
