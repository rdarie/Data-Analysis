"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx               which trial to analyze
    --exp=exp                         which experimental day to analyze
    --analysisName=analysisName       append a name to the resulting blocks? [default: default]
    --chanQuery=chanQuery             how to restrict channels if not providing a list? [default: fr]
    --samplingRate=samplingRate       subsample the result??
    --plotting                        run diagnostic plots? [default: False]
    --commitResults                   save results to data partition? [default: False]
"""
import matplotlib
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.ns5 as ns5
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb
import traceback
from importlib import reload
import json
import shutil
from copy import deepcopy
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
alignTimeBounds = alignTimeBoundsLookup[int(arguments['blockIdx'])]

def calcISIBlockAnalysisNix():
    arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
        namedQueries, scratchFolder, **arguments)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
        )
    if not os.path.exists(analysisSubFolder):
        os.makedirs(analysisSubFolder, exist_ok=True)
    if arguments['samplingRate'] is not None:
        samplingRate = float(arguments['samplingRate']) * pq.Hz
    else:
        samplingRate = float(1 / rasterOpts['binInterval']) * pq.Hz
    #
    if not os.path.exists(trialBasePath):
        trialProcessedPath = os.path.join(
            processedFolder, ns5FileName + '.nix')
        # will throw an error if file was never processed
        shutil.copyfile(trialProcessedPath, trialBasePath)
    #
    nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
    mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
    nspBlock = ns5.readBlockFixNames(
        nspReader, block_index=0,
        reduceChannelIndexes=True
        )
    #
    spikesBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSpikes=True)
    spikesBlock = hf.loadBlockProxyObjects(spikesBlock)
    #  save ins time series
    tdChanNames = ns5.listChanNames(
        nspBlock, arguments['chanQuery'],
        objType=AnalogSignalProxy)
    #
    allSpikeTrains = [
        i
        for i in spikesBlock.filter(objects=SpikeTrain)
        if '#' in i.name]
    if len(allSpikeTrains):
        for segIdx, dataSeg in enumerate(spikesBlock.segments):
            spikeList = dataSeg.filter(objects=SpikeTrain)
            spikeList = ns5.loadContainerArrayAnn(trainList=spikeList)
    # calc binarized and get new time axis
    allStimTrains = [
        i
        for i in spikesBlock.filter(objects=SpikeTrain)
        if '_stim' in i.name]

    tdBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSpikes=False, keepSignals=tdChanNames)
    tdBlock = hf.loadBlockProxyObjects(tdBlock)

    #  
    # if len(allStimTrains):
    #     for segIdx, dataSeg in enumerate(spikesBlock.segments):
    #         spikeList = [
    #             st
    #             for st in dataSeg.filter(objects=SpikeTrain)
    #             if '_stim' in st.name]
    #         for stIdx, st in enumerate(spikeList):
    #             chanName = st.unit.channel_index.name
    #             matchingAsig = tdBlock.filter(objects=AnalogSignal, name='seg0_' + chanName)
    #             if len(matchingAsig):
    #                 stitchStimArtifact = True
    #                 if stitchStimArtifact:
    #                     tIdx = 10
    #                     winSize = st.sampling_period * st.waveforms.shape[-1]
    #                     wvfT = np.arange(
    #                         st.times[tIdx],
    #                         st.times[tIdx] + winSize,
    #                         st.sampling_period) * st.sampling_period.units
    #                     wvfT = wvfT[:st.waveforms.shape[-1]]
    #                     asigTMask = (
    #                         (matchingAsig[0].times >= wvfT[0]) &
    #                         (matchingAsig[0].times < wvfT[0] + winSize))
    #                     plotAsig = np.squeeze(matchingAsig[0])[asigTMask]
    #                     plotAsigT = matchingAsig[0].times[asigTMask]
    #                     plotWvf = np.squeeze(st.waveforms[tIdx, :, :]) * 1e-3
    #                     fig, ax = plt.subplots()
    #                     ax.plot(wvfT, plotWvf, 'c.-')
    #                     twAx = ax.twinx()
    #                     twAx.plot(plotAsigT, plotAsig, 'm.-')
    #                     # ax.plot(plotAsigT, plotAsig - plotWvf, '.-')
    #                     plt.show()
    if len(allStimTrains):
        mustDoubleSpikeWvfLen = True

        def fixRippleStimWvf(sourceArr, destArr, whichIdx, fixFirst=1):
            if fixFirst:
                for jj in range(fixFirst):
                    destArr[:, :, jj] = destArr[:, :, fixFirst]
            for ii in range(destArr.shape[0]):
                destArr[ii, :, :] = destArr[ii, :, :] - sourceArr[ii, :, whichIdx]
            return destArr

        for stIdx, st in enumerate(allStimTrains):
            if stIdx == 0:
                originalSpikeWvfLen = st.waveforms.shape[-1]
            theseTimes = pd.Series(st.times)
            # if a stim train is longer than 1.7 msec
            # it gets split into two spikes
            maskContinued = theseTimes.diff() < 1.8e-3
            #
            if maskContinued.any():
                # mustDoubleSpikeWvfLen = True
                maskContinuedSources = maskContinued.shift(-1).fillna(False)
                assert maskContinued.sum() == maskContinuedSources.sum()
                secondVolIdx = maskContinued.index[maskContinued]
                notADuplicateMask = (~maskContinued).to_numpy()
                firstVolIdx = maskContinuedSources.index[maskContinuedSources]
                # fix inconsistency in first sample sourceArr, destArr, whichIdx
                wvf = pd.DataFrame(np.atleast_2d(
                    np.squeeze(st.waveforms[notADuplicateMask, :, :])))
                wvfDiff = wvf.diff(-1, axis=1).fillna(0)
                wvfDiffAbs = wvfDiff.abs()
                #
                rawMaxIdx = wvfDiffAbs.iloc[:, :5].idxmax(axis=1)
                #
                firstValidIdx, _ = stats.mode(rawMaxIdx, axis=None)
                firstValidIdx = int(firstValidIdx[-1] + 1)
                #
                st.waveforms[notADuplicateMask, :, :] = fixRippleStimWvf(
                    sourceArr=st.waveforms[notADuplicateMask, :, :],
                    destArr=st.waveforms[notADuplicateMask, :, :],
                    whichIdx=firstValidIdx, fixFirst=firstValidIdx)
                st.waveforms[secondVolIdx, :, :] = fixRippleStimWvf(
                    sourceArr=st.waveforms[secondVolIdx, :, :],
                    destArr=st.waveforms[secondVolIdx, :, :],
                    whichIdx=firstValidIdx, fixFirst=firstValidIdx)
                st.waveforms[secondVolIdx, :, :] = fixRippleStimWvf(
                    sourceArr=(-1) * st.waveforms[firstVolIdx, :, :],
                    destArr=st.waveforms[secondVolIdx, :, :], whichIdx=-1, fixFirst=False)
                filledWaveforms = np.concatenate(
                    [
                        st.waveforms[firstVolIdx, :, :],
                        st.waveforms[secondVolIdx, :, :]],
                    axis=-1) * st.waveforms.units
                # expand all, to catch single size spikes
                #
                padding = np.concatenate([
                    st.waveforms[:, :, -1]
                    for i in range(st.waveforms.shape[-1])], axis=-1)
                newWaveforms = np.concatenate(
                    [
                        st.waveforms, padding[:, np.newaxis, :]],
                    axis=-1) * st.waveforms.units
                newWaveforms[firstVolIdx, :, :] = filledWaveforms
                newWaveforms = newWaveforms[notADuplicateMask, :, :]
                #
                unit = st.unit
                uIdx = np.flatnonzero([
                    np.all(i == st)
                    for i in unit.spiketrains])[0]
                seg = st.segment
                segIdx = np.flatnonzero([
                    np.all(i == st)
                    for i in seg.spiketrains])[0]
                #
                newSt = deepcopy(st[notADuplicateMask])
                newSt.waveforms = newWaveforms
                for k in newSt.array_annotations.keys():
                    newSt.array_annotations[k] = st.array_annotations[k][notADuplicateMask]
                    if k in st.annotations:
                        newSt.annotations[k] = st.array_annotations[k][notADuplicateMask]
                unit.spiketrains[uIdx] = newSt
                newSt.unit = unit
                seg.spiketrains[segIdx] = newSt
                newSt.segment = seg
                allStimTrains[stIdx] = newSt
                del st
                unit.create_relationship()
                seg.create_relationship()
            else:
                # fix inconsistency in first sample sourceArr, destArr, whichIdx
                wvf = pd.DataFrame(np.atleast_2d(
                    np.squeeze(st.waveforms)))
                wvfDiff = wvf.diff(-1, axis=1).fillna(0)
                wvfDiffAbs = wvfDiff.abs()
                #
                rawMaxIdx = wvfDiffAbs.iloc[:, :5].idxmax(axis=1)
                firstValidIdx, _ = stats.mode(rawMaxIdx, axis=None)
                firstValidIdx = int(firstValidIdx[-1] + 1)
                #
                st.waveforms = fixRippleStimWvf(
                    st.waveforms, st.waveforms,
                    whichIdx=firstValidIdx, fixFirst=firstValidIdx)
                print(
                    'on spiketrain {}, waveforms.shape = {}'
                    .format(st.name, st.waveforms.shape))
            #
        if mustDoubleSpikeWvfLen:
            for stIdx, st in enumerate(spikesBlock.filter(objects=SpikeTrain)):
                if st.waveforms.shape[-1] == originalSpikeWvfLen:
                    st.waveforms = np.concatenate(
                        [
                            st.waveforms, np.zeros_like(st.waveforms)],
                        axis=-1) * st.waveforms.units
    if len(allSpikeTrains):
        spikeMatBlock = ns5.calcBinarizedArray(
            deepcopy(spikesBlock), samplingRate,
            binnedSpikePath.format(arguments['analysisName']),
            saveToFile=True)
        newT = pd.Series(
            spikeMatBlock.filter(
                objects=AnalogSignal)[0].times.magnitude)
    else:
        dummyT = nspBlock.filter(objects=AnalogSignalProxy)[0]
        newT = pd.Series(
            np.arange(
                dummyT.t_start,
                dummyT.t_stop + 1/samplingRate,
                1/samplingRate))
    #
    etpJsonPath = './isiElectrodeProgramLookup.json'
    if os.path.exists(etpJsonPath):
        with open(etpJsonPath, 'r') as f:
            electrodeToProgramLookup = json.load(f)
            latestProgram = len(electrodeToProgramLookup.keys())
    else:
        electrodeToProgramLookup = {}
        latestProgram = 0
    # Start parsing autologger info
    jsonPath = trialBasePath.replace('.nix', '_autoStimLog.json')
    if os.path.exists(jsonPath):
        with open(jsonPath, 'r') as f:
            stimLog = json.load(f)
        stimResLookup = {4: 10 * pq.uA, 5: 20 * pq.uA}
        stimDict = {
            't': [],
            'elec': [],
            # 'nominalWaveform': [],
            'nominalCurrent': [],
            'RateInHz': [],
            'stimPeriod': [],
            'trainDur': [],
            'firstPW': [],
            'interPhase': [],
            'secondPW': [],
            'totalPW': []
            }
        eventUnits = {
            'nominalCurrent': pq.uA,
            'RateInHz': pq.Hz,
            'stimPeriod': pq.Hz,
            'trainDur': pq.s,
            'firstPW': pq.s,
            'interPhase': pq.s,
            'secondPW': pq.s,
            'totalPW': pq.s
            }
        allNominalWaveforms = []
        for idx, entry in enumerate(stimLog):
            t = entry['t']
            if 'stimCmd' in entry:
                allStimCmd = entry['stimCmd']
                if isinstance(allStimCmd, dict):
                    # if only one electrode
                    allStimCmd = [allStimCmd]
                for stimCmd in allStimCmd:
                    # each stimCmd represents one electrode
                    nominalWaveform = []
                    lastAmplitude = 0
                    totalLen = 0
                    if 'stimRes' in stimCmd:
                        ampQuanta = stimResLookup[stimCmd['stimRes']]
                    else:
                        ampQuanta = 20 * pq.uA
                    for phase in stimCmd['seq']:
                        if phase['enable']:
                            phAmp = ampQuanta * phase['ampl'] * (-1) * ((-1) ** phase['pol'])
                            phaseWaveform = [phAmp for i in range(31 * phase['length'])]
                        else:
                            phaseWaveform = [0 for i in range(31 * phase['length'])]
                        phaseWaveform[:phase['delay']] = [lastAmplitude for i in range(phase['delay'])]
                        lastAmplitude = phaseWaveform[-1]
                        nominalWaveform += phaseWaveform
                        totalLen += phase['length']
                    stimDict['t'].append(t)
                    stimDict['firstPW'].append(
                        (stimCmd['seq'][0]['length'] / (3e4)) * pq.s)
                    stimDict['interPhase'].append(
                        (stimCmd['seq'][1]['length'] / (3e4)) * pq.s)
                    stimDict['secondPW'].append(
                        (stimCmd['seq'][2]['length'] / (3e4)) * pq.s)
                    stimDict['totalPW'].append((totalLen / (3e4)) * pq.s)
                    stimDict['elec'].append(stimCmd['elec'] * pq.dimensionless)
                    allNominalWaveforms.append(np.asarray(nominalWaveform))
                    nominalIdxMax = np.argmax(np.abs(np.asarray(nominalWaveform)))
                    stimDict['nominalCurrent'].append(nominalWaveform[nominalIdxMax])
                    thisStimPeriod = ((stimCmd['period'] / (3e4)) * pq.s)
                    stimDict['stimPeriod'].append(thisStimPeriod)
                    #  
                    # stimDict['RateInHz'].append(np.round(thisStimPeriod ** (-1), decimals=1))
                    stimDict['RateInHz'].append(thisStimPeriod ** (-1))
                    stimDict['trainDur'].append((stimCmd['repeats'] - 1) * thisStimPeriod)
            else:
                stimStr = entry['stimString']
                stimStrDictRaw = {}
                for stimSubStr in stimStr.split(';'):
                    if len(stimSubStr):
                        splitStr = stimSubStr.split('=')
                        stimStrDictRaw[splitStr[0]] = splitStr[1]
                stimStrDict = {}
                for key, val in stimStrDictRaw.items():
                    stimStrDict[key] = [float(st) for st in val.split(',') if len(st)]
                stimStrDF = pd.DataFrame(stimStrDict)
                stimStrDF['Elect'] = stimStrDF['Elect'].astype(np.int)
                stimStrDF.loc[stimStrDF['PL'] == 1, 'Amp'] = stimStrDF.loc[stimStrDF['PL'] == 1, 'Amp'] * (-1)
                for rIdx, row in stimStrDF.iterrows():
                    stimDict['t'].append(t)
                    stimDict['firstPW'].append(row['Dur'] * 1e-3 * pq.s)
                    stimDict['secondPW'].append(row['Dur'] * 1e-3 * pq.s)
                    stimDict['interPhase'].append(2 * ((3e4) ** -1) * pq.s)  # per page 16 of xippmex manual
                    stimDict['totalPW'].append(2 * (row['Dur'] * 1e-3 + ((3e4) ** -1)) * pq.s)
                    stimDict['nominalCurrent'].append(row['Amp'] * ampQuanta)
                    stimDict['RateInHz'].append(row['Freq'] * pq.Hz)
                    stimDict['stimPeriod'].append(row['Freq'] ** -1)
                    stimDict['trainDur'].append(row['TL'] * 1e-3 * pq.s)
                    stimDict['elec'].append(row['Elect'] * pq.dimensionless)
        stimDict['labels'] = np.asarray([
            'stim update {}'.format(i)
            for i in range(len(stimDict['elec']))])
        # (np.asarray(stimDict['t'])/3e4 <= 1).any()
        rawStimEventTimes = np.asarray(stimDict.pop('t')) / (30000) * pq.s
        # rawStimEventTimes = rawStimEventTimes - rawStimEventTimes[0] + activeTimes.min() * pq.s
        # rawStimEventTimes = rawStimEventTimes.magnitude * rawStimEventTimes.units.simplified
        stimEvents = Event(
            name='seg0_stimEvents',
            times=rawStimEventTimes,
            labels=stimDict.pop('labels'))
        stimEvents.annotations['arrayAnnNames'] = [k for k in stimDict.keys()]
        stimEvents.annotations['nix_name'] = stimEvents.name
        spikesBlock.segments[0].events.append(stimEvents)
        stimEvents.segment = spikesBlock.segments[0]
        #
        for k in stimEvents.annotations['arrayAnnNames']:
            stimEvents.array_annotations[k] = stimDict[k]
            stimEvents.annotations[k] = stimDict.pop(k)
    else:
        stimEvents = None
    # stimEvents.annotations['nominalWaveforms'] = np.vstack(allNominalWaveforms)
    if len(allStimTrains):
        for segIdx, dataSeg in enumerate(spikesBlock.segments):
            spikeList = [
                st
                for st in dataSeg.filter(objects=SpikeTrain)
                if '_stim' in st.name]
            stimRasters = [
                sr
                for sr in spikeMatBlock.segments[segIdx].analogsignals
                if '_stim' in sr.name]
            stimRastersDF = ns5.analogSignalsToDataFrame(
                stimRasters, idxT='t', useChanNames=True)
            stimRastersDF.columns = [
                cn.replace('_stim#0_raster', '')
                for cn in stimRastersDF.columns]
            keepStimRasterList = []
            for stIdx, st in enumerate(spikeList):
                chanName = st.unit.channel_index.name
                matchingAsig = nspBlock.filter(objects=AnalogSignalProxy, name='seg0_' + chanName)
                if len(matchingAsig):
                    keepStimRasterList.append(chanName)
            stimActive = stimRastersDF[keepStimRasterList].sum(axis=1) > 0
            activeTimes = stimRastersDF.loc[stimActive, 't']
            #
            if stimEvents is not None:
                stimEvents[:] = (
                    stimEvents.times -
                    stimEvents.times[0] -
                    10e-3 * pq.s +  # Fudge factor to account for delay between execution and matlab save
                    activeTimes.min() * pq.s)
                # pdb.set_trace()
                stimEventsDF = pd.DataFrame(stimEvents.array_annotations)
                stimEventsDF['t'] = stimEvents.times
                stimEventsDF.to_csv(os.path.join(
                    analysisSubFolder, ns5FileName + '_exported_stim_updates.csv'
                    ))
            #
            for stIdx, st in enumerate(spikeList):
                # annotate ripple stim spikes with info from json log
                chanName = st.unit.channel_index.name
                # matchingChIdx = nspBlock.filter(objects=ChannelIndex, name=chanName)
                rippleChanNum = int(mapDF.loc[mapDF['label'] == chanName, 'nevID'] + 1)
                if stimEvents is not None:
                    # find which events in the stim log reference this spiketrain
                    thisStEventsMask = stimEvents.array_annotations['elec'] == rippleChanNum
                    theseUpdates = pd.DataFrame({
                        k: v[thisStEventsMask]
                        for k, v in stimEvents.array_annotations.items()
                        })
                    theseUpdates.index = stimEvents[thisStEventsMask].times
                    theseUpdates.index.name = 't'
                    # NOTE: the line below is a workaround for an edge case where the same electrode is
                    # requested twice in the same command, it should not be needed normally
                    theseUpdates = theseUpdates.loc[~theseUpdates.index.duplicated(), :]
                    # create entries for each pulse of the spiketrain
                    newIndex = np.unique(np.concatenate([
                        stimEvents[thisStEventsMask].times.magnitude,
                        st.times.magnitude]))
                    #  
                    # updateTimes = pd.Series(theseUpdates.index)
                    # nonMonotonicTimes = updateTimes.diff().fillna(1) <= 0
                    # updateTimes[nonMonotonicTimes][0]
                    # theseUpdate.loc[theseUpdates.index > updateTimes[nonMonotonicTimes][0], :]
                    try:
                        allUpdates = theseUpdates.reindex(newIndex, method='ffill')
                    except Exception:
                        traceback.print_exc()
                         
                    stAnnotations = allUpdates.loc[
                        allUpdates.index.isin(st.times.magnitude), :]
                #
                wvf = pd.DataFrame(np.atleast_2d(np.squeeze(st.waveforms)))
                wvfDiff = wvf.diff(-1, axis=1).fillna(0)
                wvfDiffAbs = wvfDiff.abs()
                if stimEvents is not None:
                    lastValidIdx = int(stAnnotations['totalPW'].min() * 3e4) - 1
                    idxPeak = int(stAnnotations['firstPW'].min() * 3e4)
                    wvf.iloc[:, lastValidIdx:] = np.nan
                    wvf.fillna(method='ffill', axis=1, inplace=True)
                    wvfDiff = wvf.diff(-1, axis=1).fillna(0)
                    wvfDiffAbs = wvfDiff.abs()
                else:
                    rawMaxIdx = wvfDiffAbs.idxmax(axis=1)
                    #
                    if (rawMaxIdx > 2).any():
                        lastValidIdx, _ = stats.mode(
                            rawMaxIdx[rawMaxIdx > 2], axis=None)
                        lastValidIdx = int(lastValidIdx[-1]) - 2
                    else:
                        lastValidIdx = wvf.shape[-1] - 1
                    #
                    print(
                        'On spikeTrain {}, last valid index is {}'
                        .format(st.name, lastValidIdx))
                    #
                    wvf.iloc[:, lastValidIdx:] = np.nan
                    wvf.fillna(method='ffill', axis=1, inplace=True)
                    wvfDiff = wvf.diff(-1, axis=1).fillna(0)
                    wvfDiffAbs = wvfDiff.abs()
                    #
                    scaler = StandardScaler()
                    scaler.fit(wvfDiffAbs.iloc[:, 1:lastValidIdx].to_numpy().reshape(-1, 1))
                    transformWvfDiff = lambda x: np.squeeze(scaler.transform(x.reshape(-1, 1)))
                    wvfDiffStd = wvfDiffAbs.apply(transformWvfDiff, axis=1, raw=True)
                    # if arguments['plotting']:
                    #     plt.plot(wvfDiffStd.T, 'o-'); plt.title('{} standardized abs diff'.format(st.name)); plt.show()
                    # TODO: check if it's necessary to exclude some samples from being centered
                    # samplesNeedFix = wvfDiffStd.abs().iloc[:, 0] > 0
                    # print('{} out of {} samples need fixing'.format(samplesNeedFix.sum(), samplesNeedFix.size))
                    # wvf.loc[samplesNeedFix, 0] = np.nan
                    # wvf.fillna(method='bfill', axis=1, inplace=True)
                    # wvfDiff.loc[samplesNeedFix, 0] = np.nan
                    # wvfDiff.fillna(method='bfill', axis=1, inplace=True)
                    # wvfDiffStd.loc[samplesNeedFix, 0] = np.nan
                    # wvfDiffStd.fillna(method='bfill', axis=1, inplace=True)
                    # wvf = wvf.apply(lambda x: x - x[0], axis=1, raw=True)
                    allPeakIdx = wvfDiffStd.iloc[:, :lastValidIdx - 5].idxmax(axis=1)
                    if (allPeakIdx > 2).any():
                        idxPeak, _ = stats.mode(allPeakIdx[allPeakIdx > 2], axis=None)
                        idxPeak = int(idxPeak[0])
                    else:
                        idxPeak = int(lastValidIdx/2)
                #
                amplitudes = wvf.apply(
                    lambda x: (x[idxPeak] - x[0]) * 1e-6,
                    axis=1, raw=True).to_numpy() * pq.V
                st.annotations['amplitude'] = amplitudes
                st.array_annotations['amplitude'] = amplitudes
                if 'arrayAnnNames' in st.annotations:
                    st.annotations['arrayAnnNames'].append('amplitude')
                else:
                    st.annotations['arrayAnnNames'] = ['amplitude']
                #
                ampWithinSpec = np.abs(amplitudes) < 4
                #
                plotMask = st.times > 0 # < 1360
                if arguments['plotting']:
                    plt.plot(st.sampling_period * np.arange(wvf.shape[1]), wvf.iloc[plotMask, :].T * 1e-6, 'o-'); plt.title('{} fixed wvf peak at {}'.format(st.name, idxPeak*st.sampling_period)); plt.show()
                    plt.plot(st.sampling_period * np.arange(wvf.shape[1]), (wvfDiffAbs).iloc[:, :].T * 1e-6, 'o-');
                    plt.plot(st.sampling_period * np.arange(wvf.shape[1]), (wvfDiffAbs).iloc[:, :].mean().T * 1e-6, 'o-', lw=3); plt.title('{} fixed diff peak at {}'.format(st.name, idxPeak*st.sampling_period)); plt.show()
                if stimEvents is None:
                    pws = amplitudes ** 0 * idxPeak * st.sampling_period
                    st.annotations['firstPW'] = pws
                    st.array_annotations['firstPW'] = pws
                    st.annotations['arrayAnnNames'].append('firstPW')
                    #
                    secPws = amplitudes ** 0 * (lastValidIdx - idxPeak) * st.sampling_period
                    st.annotations['secondPW'] = secPws
                    st.array_annotations['secondPW'] = secPws
                    st.annotations['arrayAnnNames'].append('secondPW')
                    #
                    interPhases = 2 * amplitudes ** 0 * st.sampling_period
                    st.annotations['interPhase'] = interPhases
                    st.array_annotations['interPhase'] = interPhases
                    st.annotations['arrayAnnNames'].append('interPhase')
                    #
                    totalPws = pws + secPws
                    st.annotations['totalPW'] = totalPws
                    st.array_annotations['totalPW'] = totalPws
                    st.annotations['arrayAnnNames'].append('totalPW')
                    # try to estimate current
                    matchingAsig = nspBlock.filter(objects=AnalogSignalProxy, name='seg0_' + chanName)
                    if len(matchingAsig):
                        # keepStimRasterList.append(chanName)
                        elecImpedance = (
                            impedancesRipple
                            .loc[impedancesRipple['elec'] == chanName, 'impedance'])
                        currents = amplitudes / (elecImpedance.iloc[0] * pq.kOhm)
                        st.annotations['nominalCurrent'] = currents
                        st.array_annotations['nominalCurrent'] = currents
                        if 'arrayAnnNames' in st.annotations:
                            st.annotations['arrayAnnNames'].append('nominalCurrent')
                        else:
                            st.annotations['arrayAnnNames'] = ['nominalCurrent']
                else:
                    for annName in stAnnotations.drop('elec', axis='columns'):
                        st.annotations['arrayAnnNames'].append(annName)
                        st.annotations[annName] = (
                            stAnnotations[annName].to_numpy() *
                            eventUnits[annName])
                        st.array_annotations[annName] = (
                            stAnnotations[annName].to_numpy() *
                            eventUnits[annName])
            # detect stimulation trains
            peakIdx, _, trainStartIdx, trainEndIdx = hf.findTrains(
                peakTimes=activeTimes, minDistance=5e-3, maxDistance=200e-3)
            #  
            trainDurations = trainEndIdx - trainStartIdx
            #
            if len(trainStartIdx):
                startCategories = pd.DataFrame(
                    activeTimes[trainStartIdx].to_numpy(),
                    # index=range(activeTimes[trainStartIdx].size),
                    columns=['t'])
                startCategories = startCategories.reindex(columns=[
                    # 'amplitude',
                    'nominalCurrent', 'program',
                    'activeGroup', 'firstPW', 'secondPW', 'interPhase',
                    'totalPW', 'electrode',
                    'RateInHz', 'stimPeriod', 'trainDur', 't'])
                #
                for idx, (idxStart, idxEnd) in enumerate(
                        zip(trainStartIdx, trainEndIdx)):
                    stimRasterRow = (
                        stimRastersDF
                        .loc[idxStart, keepStimRasterList])
                    activeChans = stimRasterRow.index[stimRasterRow > 0]
                    if not activeChans.empty:
                        stimRasterAmplitude = pd.Series(
                            np.nan, index=activeChans)
                        stimRasterCurrent = pd.Series(
                            np.nan, index=activeChans)
                        for activeChanIdx, activeChan in enumerate(activeChans):
                            st = [
                                i
                                for i in spikeList
                                if i.unit.channel_index.name == activeChan][0]
                            theseTimesMask = (
                                (st.times >= (
                                    stimRastersDF.loc[idxStart, 't'] * pq.s -
                                    1.1 * samplingRate ** (-1) / 2)) &
                                (st.times <= (
                                    stimRastersDF.loc[idxEnd, 't'] * pq.s +
                                    1.1 * samplingRate ** (-1) / 2))
                                )
                            theseTimes = st.times[theseTimesMask]
                            if not theseTimesMask.sum():
                                pdb.set_trace()
                            stimRasterAmplitude[activeChan] = np.mean(
                                st.annotations['amplitude'][theseTimesMask])
                            stimRasterCurrent[activeChan] = np.mean(
                                st.annotations['nominalCurrent'][theseTimesMask])
                            if activeChanIdx == 0:
                                if stimEvents is None:
                                    if theseTimes.size == 1:
                                        startCategories.loc[
                                            idx, 'trainDur'] = 0
                                        startCategories.loc[
                                            idx, 'RateInHz'] = 0
                                        startCategories.loc[
                                            idx, 'stimPeriod'] = 1000
                                    else:
                                        startCategories.loc[
                                            idx, 'trainDur'] = (
                                                theseTimes[-1] -
                                                theseTimes[0])
                                        stimPeriod = np.round(np.diff(theseTimes).median(), decimals=6)
                                        startCategories.loc[
                                            idx, 'stimPeriod'] = stimPeriod
                                        startCategories.loc[
                                            idx, 'RateInHz'] = stimPeriod ** -1
                                else:
                                    nominalRate = np.median(st.annotations['RateInHz'][theseTimesMask])
                                    observedRate = np.median(np.diff(theseTimes)) ** (-1)
                                    rateMismatch = np.abs(nominalRate - observedRate)
                                    if not rateMismatch < 1e-6:
                                        print(
                                            'Rate mismatch warning on {} at time {}: off by {} Hz'
                                            .format(st.name, theseTimes[0], rateMismatch))
                                        pdb.set_trace()
                                    nominalTrainDur = np.mean(st.annotations['trainDur'][theseTimesMask])
                                    observedTrainDur = (theseTimes[-1] - theseTimes[0])
                                    if not np.abs(nominalTrainDur - observedTrainDur) < 1e-6:
                                        print('train Dur Warning on {} at time {}'.format(st.name, theseTimes[0]))
                                    # assert np.diff(theseTimes).mean()
                                    startCategories.loc[idx, 'trainDur'] = nominalTrainDur
                                    startCategories.loc[idx, 'RateInHz'] = nominalRate
                                    startCategories.loc[idx, 'stimPeriod'] = nominalRate ** -1
                                startCategories.loc[
                                    idx, 'secondPW'] = np.round(np.mean(
                                        st.annotations['secondPW'][theseTimesMask]), decimals=9)
                                startCategories.loc[
                                    idx, 'firstPW'] = np.round(np.mean(
                                        st.annotations['firstPW'][theseTimesMask]), decimals=9)
                                startCategories.loc[
                                    idx, 'interPhase'] = np.round(np.mean(
                                        st.annotations['interPhase'][theseTimesMask]), decimals=9)
                                startCategories.loc[
                                    idx, 'totalPW'] = np.round(np.mean(
                                        st.annotations['totalPW'][theseTimesMask]), decimals=9)
                        startCategories.loc[idx, 'activeGroup'] = 1
                        electrodeShortHand = ''
                        negativeAmps = stimRasterCurrent < 0
                        #
                        if (negativeAmps).any():
                            electrodeShortHand += '-'
                            totalCathode = stimRasterCurrent[negativeAmps].sum()
                            startCategories.loc[idx, 'nominalCurrent'] = totalCathode
                            averageImpedance = np.mean(
                                impedancesRipple.loc[impedancesRipple['elec'].isin(
                                    stimRasterCurrent[negativeAmps].index), 'impedance'])
                            # startCategories.loc[idx, 'amplitude'] = totalCathode * averageImpedance
                            for cName in stimRasterCurrent[negativeAmps].index:
                                electrodeShortHand += cName[:-2]
                        positiveAmps = stimRasterCurrent > 0
                        if (positiveAmps).any():
                            electrodeShortHand += '+'
                            totalAnode = stimRasterCurrent[positiveAmps].sum()
                            for cName in stimRasterCurrent[positiveAmps].index:
                                electrodeShortHand += cName[:-2]
                            if np.isnan(startCategories.loc[idx, 'nominalCurrent']):
                                startCategories.loc[idx, 'nominalCurrent'] = totalAnode
                        startCategories.loc[idx, 'electrode'] = electrodeShortHand
                        if (electrodeShortHand not in electrodeToProgramLookup):
                            electrodeToProgramLookup[electrodeShortHand] = latestProgram
                            latestProgram += 1
                        startCategories.loc[idx, 'program'] = electrodeToProgramLookup[electrodeShortHand]
                #
                currCats = pd.cut(
                    startCategories['nominalCurrent'],
                    np.arange(-2, 2, 0.2))
                startCategories['nominalCurrentCat'] = currCats.astype('str')
                startCategories['RateInHz'] = np.round(startCategories['RateInHz'], decimals=1)
                stopCategories = startCategories.copy()
                #
                stopCategories['t'] = (
                    activeTimes[trainEndIdx].to_numpy() +
                    (
                        stopCategories['firstPW'] +
                        stopCategories['interPhase'] +
                        stopCategories['secondPW']
                    ).to_numpy() * 1e-6)
                # maxAmp = startCategories['amplitude'].max()
                # minAmp = startCategories['amplitude'].min()
                # ampBinRes = 0.2
                # ampBins = np.arange(
                #     (np.floor(minAmp / ampBinRes) - 1) * ampBinRes,
                #     (np.ceil(maxAmp / ampBinRes) + 1) * ampBinRes,
                #     ampBinRes)
                # ampBins[0] -= 0.01
                # ampBins[-1] += 0.01
                # ampCats = pd.cut(startCategories['amplitude'], ampBins)
                # startCategories['amplitudeCat'] = ampCats.astype(np.str)
                # stopCategories['amplitudeCat'] = ampCats.astype(np.str)
                startCategories['stimCat'] = 'stimOn'
                stopCategories['stimCat'] = 'stimOff'
                startCategories.dropna(inplace=True)
                stopCategories.dropna(inplace=True)
        #
        with open(etpJsonPath, 'w') as f:
            json.dump(electrodeToProgramLookup, f)
        alignEventsDF = pd.concat((
            startCategories, stopCategories),
            axis=0, ignore_index=True, sort=True)
        # remove events outside manually identified time bounds
        keepMask = pd.Series(False, index=alignEventsDF.index)
        for atb in alignTimeBounds:
            keepMask = (
                keepMask |
                (
                    (alignEventsDF['t'] >= atb[0]) &
                    (alignEventsDF['t'] <= atb[1])))
        alignEventsDF.drop(
            index=alignEventsDF.index[~keepMask], inplace=True)
        #
        if not alignEventsDF.empty:
            alignEventsDF.sort_values('t', inplace=True, kind='mergesort')
            alignEvents = ns5.eventDataFrameToEvents(
                alignEventsDF, idxT='t',
                annCol=None,
                eventName='seg{}_stimAlignTimes'.format(segIdx),
                tUnits=pq.s, makeList=False)
            alignEvents.annotate(nix_name=alignEvents.name)
            #
            concatLabelsDF = alignEventsDF
            concatLabels = np.array([
                '{}'.format(row)
                for rowIdx, row in concatLabelsDF.iterrows()])
            concatEvents = Event(
                name='seg{}_stimAlignTimesConcatenated'.format(segIdx),
                times=alignEvents.times,
                labels=concatLabels
                )
            dataSeg.events.append(alignEvents)
            dataSeg.events.append(concatEvents)
            alignEvents.segment = dataSeg
            concatEvents.segment = dataSeg
    #  Delete stim trains, because they won't be consistent across assembled files
    # if len(allStimTrains):
    #     for seg in spikesBlock.segments:
    #         for st in allStimTrains:
    #             if st in seg.spiketrains:
    #                 seg.spiketrains.remove(st)
    #     allStimUnits = [un for un in spikesBlock.filter(objects=Unit) if '_stim' in un.name]
    #     del allStimTrains
    #     # delChanIndices = []
    #     for chIdx in spikesBlock.channel_indexes:
    #         for stUn in allStimUnits:
    #             if stUn in chIdx.units:
    #                 chIdx.units.remove(stUn)
    #     del allStimUnits
    #
    #  
    aSigList = tdBlock.filter(objects=AnalogSignal)
    tdDF = ns5.analogSignalsToDataFrame(aSigList)
    #
    currentSamplingRate = tdBlock.filter(
        objects=AnalogSignal)[0].sampling_rate
    #
    emgCols = [cn for cn in tdDF.columns if 'Emg' in cn]
     
    if len(emgCols):
        # fix for bug affecting the mean of the channel
        keepMaskAsig = pd.Series(False, index=tdDF.index)
        for atb in alignTimeBounds:
            keepMaskAsig = (
                keepMaskAsig |
                (
                    (tdDF['t'] >= atb[0]) &
                    (tdDF['t'] <= atb[1])))
        sosHP = signal.butter(
            2, 40, 'high',
            fs=float(currentSamplingRate), output='sos')
        cornerFrequencyLP = 40
        sosLP = signal.butter(
            2, cornerFrequencyLP, 'low',
            fs=float(currentSamplingRate), output='sos')
        if False:
            t = np.arange(0, .1, currentSamplingRate.magnitude ** (-1))
            x = np.zeros_like(t)
            x[int(x.size/2)] = 1
            y = signal.sosfiltfilt(sosLP, x)
            plt.plot(t, y); plt.show()
        for cName in emgCols:
            procName = cName.replace('Emg', 'EmgEnv')
            # weird units hack, TODO check
            tdDF.loc[:, cName] = tdDF.loc[:, cName] * 1e6
            preprocEmg = signal.sosfiltfilt(
                sosHP,
                (tdDF[cName] - tdDF.loc[keepMaskAsig, cName].median()).to_numpy())
            # 
            tdDF[procName] = signal.sosfiltfilt(
                sosLP, np.abs(preprocEmg))
            # break
            # if True:
            #     plt.plot(tdDF.loc[keepMaskAsig, cName])
            #     plt.plot(tdDF.loc[keepMaskAsig, procName])
            #     plt.show()
            tdChanNames.append(procName)
            #
    if len(allStimTrains):
        # fill in blank period
        stimMask = (stimRastersDF.drop(columns='t') > 0).any(axis='columns')
        # blankingDur = 0.5e-3 + np.round(stAnnotations['totalPW'].max(), decimals=3) - 2 * currentSamplingRate.magnitude ** (-1)
        # blankingDur = stAnnotations['totalPW'].max() + 5 * currentSamplingRate.magnitude ** (-1)
        blankingDur = stAnnotations['totalPW'].max()
        #  TODO: get fixed part from metadata and make robust to
        #  different blanks per stim config stAnnotations['secondPW']
        kernelT = np.arange(
            # -blankingDur,
            -blankingDur + currentSamplingRate.magnitude ** (-1),
            # blankingDur,
            blankingDur + currentSamplingRate.magnitude ** (-1),
            currentSamplingRate.magnitude ** (-1))
        kernel = np.zeros_like(kernelT)
        kernel[kernelT > 0] = 1
        blankMask = (
            np.convolve(kernel, stimMask, 'same') > 0)[:tdDF.shape[0]]
        checkBlankMask = False
        if checkBlankMask:
            plotIdx = slice(2000000, 2020000)
            fig, ax = plt.subplots()
            twAx = ax.twinx()
            ax.plot(
                tdDF['t'].iloc[plotIdx],
                tdDF.iloc[plotIdx, 1], 'b.-', lw=2)
        spinalLfpChans = [
            cN
            for cN in tdDF.columns
            if 'rostral' in cN or 'caudal' in cN]
        # tdDF.loc[
        #     blankMask, spinalLfpChans] = np.nan
        # tdDF.interpolate(axis=0, method='cubic', inplace=True)
        # tdDF.loc[
        #     blankMask, spinalLfpChans] = 0
        if checkBlankMask:
            ax.plot(
                tdDF['t'].iloc[plotIdx],
                tdDF.iloc[plotIdx, 1].interpolate(axis=0, method='cubic'), 'g--', lw=2)
            twAx.plot(
                tdDF['t'].iloc[plotIdx],
                blankMask[plotIdx], 'r')
            plt.show()
    
    if samplingRate != currentSamplingRate:
        print("Reinterpolating...")
        tdInterp = hf.interpolateDF(
            tdDF, newT,
            kind='linear', fill_value=(0, 0),
            x='t', columns=tdChanNames)
    else:
        tdInterp = tdDF
    #
    tdInterp.columns = [i.replace('seg0_', '') for i in tdInterp.columns]
    tdInterp.sort_index(axis='columns', inplace=True)
    tdBlockInterp = ns5.dataFrameToAnalogSignals(
        tdInterp,
        idxT='t', useColNames=True, probeName='',
        dataCol=tdInterp.drop(columns='t').columns,
        samplingRate=samplingRate)
    #
    for aSig in tdBlockInterp.filter(objects=AnalogSignal):
        chName = aSig.channel_index.name
        chIdxList = spikesBlock.filter(objects=ChannelIndex, name=chName)
        if not len(chIdxList):
            lastIndex = len(spikesBlock.channel_indexes)
            if len(spikesBlock.channel_indexes[-1].channel_ids):
                lastID = spikesBlock.channel_indexes[-1].channel_ids[0] + 1
            else:
                lastID = 1
            chIdx = ChannelIndex(
                index=[lastIndex],
                channel_names=[chName],
                channel_ids=[lastID],
                name=chName,
                file_origin=spikesBlock.channel_indexes[-1].file_origin
                )
            chIdx.merge_annotations(spikesBlock.channel_indexes[-1])
            spikesBlock.channel_indexes.append(chIdx)
        else:
            chIdx = chIdxList[0]
        chIdx.analogsignals.append(aSig)
        aSig.channel_index = chIdx
        segName = aSig.segment.name
        segList = spikesBlock.filter(objects=Segment, name=segName)
        seg=segList[0]
        seg.analogsignals.append(aSig)
        aSig.segment = seg
    #
    spikesBlock = ns5.purgeNixAnn(spikesBlock)
    #
    spikesBlock.create_relationship()
    outPathName = analysisDataPath.format(arguments['analysisName'])
    if os.path.exists(outPathName):
        os.remove(outPathName)
    writer = neo.io.NixIO(filename=outPathName)
    writer.write_block(spikesBlock, use_obj_names=True)
    writer.close()
    if arguments['commitResults']:
        analysisProcessedSubFolder = os.path.join(
            processedFolder, arguments['analysisName']
            )
        if not os.path.exists(processedFolder):
            os.makedirs(processedFolder, exist_ok=True)
        processedOutPath = os.path.join(
            processedFolder, ns5FileName + '_analyze.nix')
        shutil.copyfile(outPathName, processedOutPath)
    # ns5.addBlockToNIX(
    #     tdBlockInterp, neoSegIdx=[0],
    #     writeSpikes=False, writeEvents=False,
    #     purgeNixNames=False,
    #     fileName=ns5FileName + '_analyze',
    #     folderPath=analysisSubFolder,
    #     nixBlockIdx=0, nixSegIdx=[0],
    #     )
    return


if __name__ == "__main__":
    runProfiler = False
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        if arguments['lazy']:
            nameSuffix = 'lazy'
        else:
            nameSuffix = 'not_lazy'
        prf.profileFunction(
            topFun=calcISIBlockAnalysisNix,
            modulesToProfile=[ash, ns5, hf],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        calcISIBlockAnalysisNix()