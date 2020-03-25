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
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import numpy as np
import pandas as pd
from scipy import stats
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
    nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
    nspBlock = ns5.readBlockFixNames(nspReader, block_index=0)
    #
    spikesBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSpikes=True)
    spikesBlock = hf.loadBlockProxyObjects(spikesBlock)
    #  save ins time series
    tdChanNames = ns5.listChanNames(
        nspBlock, arguments['chanQuery'],
        objType=AnalogSignalProxy)
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
    if len(allStimTrains):
        mustDoubleSpikeWvfLen = False
        for stIdx, st in enumerate(allStimTrains):
            if stIdx == 0:
                originalSpikeWvfLen = st.waveforms.shape[-1]
            theseTimes = pd.Series(st.times)
            # if a stim train is longer than 1.7 msec
            # it gets split into two spikes
            maskContinued = theseTimes.diff() < 1.8e-3
            #
            if maskContinued.any():
                mustDoubleSpikeWvfLen = True
                maskContinuedSources = maskContinued.shift(-1).fillna(False)
                assert maskContinued.sum() == maskContinuedSources.sum()
                secondVolIdx = maskContinued.index[maskContinued]
                notADuplicateMask = (~maskContinued).to_numpy()
                firstVolIdx = maskContinuedSources.index[maskContinuedSources]
                newWaveforms = np.concatenate(
                    [
                        st.waveforms[firstVolIdx, :, :],
                        st.waveforms[secondVolIdx, :, :]],
                    axis=-1) * st.waveforms.units
                # expand all, to catch single size spikes
                st.waveforms = np.concatenate(
                    [
                        st.waveforms, np.zeros_like(st.waveforms)],
                    axis=-1) * st.waveforms.units
                unit = st.unit
                uIdx = np.flatnonzero([np.all(i == st) for i in unit.spiketrains])[0]
                seg = st.segment
                segIdx = np.flatnonzero([np.all(i == st) for i in seg.spiketrains])[0]
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
                pass
            pdb.set_trace()
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
    # Start parsing autologger info
    jsonPath = trialBasePath.replace('.nix', '_autoStimLog.json')
    with open(jsonPath, 'r') as f:
        stimLog = json.load(f)
    stimDict = {
        't': [],
        'elec': [],
        'nominalWaveform': [],
        'nominalCurrent': [],
        'RateInHz': [],
        'trainDur': [],
        'firstPW': [],
        'secondPW': [],
        }
    allNominalWaveforms = []
    for idx, entry in enumerate(stimLog):
        t = entry['t']
        allStimCmd = entry['stimCmd']
        ampQuanta = 20 * pq.uA  # TODO: read from settings
        for stimCmd in allStimCmd:
            # each stimCmd represents one electrode
            nominalWaveform = []
            lastAmplitude = 0
            for phase in stimCmd['seq']:
                if phase['enable']:
                    phAmp = ampQuanta * phase['ampl'] * (-1) * ((-1) ** phase['pol'])
                    phaseWaveform = [phAmp for i in range(31 * phase['length'])]
                else:
                    phaseWaveform = [0 for i in range(31 * phase['length'])]
                phaseWaveform[:phase['delay']] = [lastAmplitude for i in range(phase['delay'])]
                lastAmplitude = phaseWaveform[-1]
                nominalWaveform += phaseWaveform
            stimDict['t'].append(t)
            stimDict['firstPW'].append(stimCmd['seq'][0]['length'] / (30000 * pq.Hz))
            stimDict['secondPW'].append(stimCmd['seq'][2]['length'] / (30000 * pq.Hz))
            stimDict['elec'].append(stimCmd['elec'] * pq.dimensionless)
            allNominalWaveforms.append(np.asarray(nominalWaveform))
            nominalIdxMax = np.argmax(np.abs(np.asarray(nominalWaveform)))
            stimDict['nominalCurrent'].append(nominalWaveform[nominalIdxMax])
            thisStimPeriod = (stimCmd['period'] / (30000 * pq.Hz))
            stimDict['RateInHz'].append(thisStimPeriod ** (-1))
            stimDict['trainDur'].append((stimCmd['repeats']) * thisStimPeriod)
    
    stimDict['labels'] = np.asarray([
        'stim update {}'.format(i)
        for i in range(len(stimDict['elec']))])
    stimEvents = Event(
        name='seg0_stimEvents',
        times=np.asarray(stimDict.pop('t')) / (30000 * pq.Hz),
        labels=stimDict.pop('labels'))
    stimEvents.annotations['arrayAnnNames'] = [k for k in stimDict.keys()]
    stimEvents.annotations['nix_name'] = stimEvents.name
    spikesBlock.segments[0].events.append(stimEvents)
    stimEvents.segment = spikesBlock.segments[0]
    #
    for k in stimEvents.annotations['arrayAnnNames']:
        stimEvents.array_annotations[k] = stimDict[k]
        stimEvents.annotations[k] = stimDict.pop(k)
    stimEvents.annotations['nominalWaveforms'] = np.vstack(allNominalWaveforms)
    #
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
            stimRastersDF.columns = [cn.replace('_stim#0_raster', '') for cn in stimRastersDF.columns]
            keepStimRasterList = []
            for stIdx, st in enumerate(spikeList):
                chanName = st.unit.name.replace('_stim#0', '')
                wvf = pd.DataFrame(np.squeeze(st.waveforms))
                lastValidIdx = 30
                wvf.iloc[:, lastValidIdx:] = np.nan
                wvf.fillna(method='ffill', axis=1, inplace=True)
                #
                wvfDiff = wvf.diff(-1, axis=1).fillna(0)
                wvfDiffSign = wvfDiff.apply(np.sign)
                wvfDiffAbs = wvfDiff.abs()
                scaler = StandardScaler()
                scaler.fit(
                    wvfDiffAbs.iloc[:, 1:lastValidIdx]
                    .to_numpy().reshape(-1, 1))
                transformWvfDiff = lambda x: np.squeeze(scaler.transform(x.reshape(-1, 1)))
                wvfDiffStd = wvfDiffAbs.apply(transformWvfDiff, axis=1, raw=True)
                if arguments['plotting']:
                    plt.plot(wvfDiffStd.T, 'o-'); plt.title('{} standardized abs diff'.format(st.name)); plt.show()
                # TODO: check if it's necessary to exclude some samples from being centered
                samplesNeedFix = wvfDiffStd.abs().iloc[:, 0] > 0
                print('{} out of {} samples need fixing'.format(samplesNeedFix.sum(), samplesNeedFix.size))
                wvf.loc[samplesNeedFix, 0] = np.nan
                wvf.fillna(method='bfill', axis=1, inplace=True)
                wvfDiff.loc[samplesNeedFix, 0] = np.nan
                wvfDiff.fillna(method='bfill', axis=1, inplace=True)
                wvfDiffStd.loc[samplesNeedFix, 0] = np.nan
                wvfDiffStd.fillna(method='bfill', axis=1, inplace=True)
                wvf = wvf.apply(lambda x: x - x[0], axis=1, raw=True)
                #
                idxPeak, _ = stats.mode(wvfDiffStd.idxmax(axis=1), axis=None)
                idxPeak = int(idxPeak)
                amplitudes = wvf.apply(lambda x: (x[idxPeak] - x[0]) * 1e-6, axis=1, raw=True).to_numpy() * pq.V
                st.annotations['amplitude'] = amplitudes
                st.array_annotations['amplitude'] = amplitudes
                if 'arrayAnnNames' in st.annotations:
                    st.annotations['arrayAnnNames'].append('amplitude')
                else:
                    st.annotations['arrayAnnNames'] = ['amplitude']
                pws = amplitudes ** 0 * idxPeak * st.sampling_period
                st.annotations['pw'] = pws
                st.array_annotations['pw'] = pws
                if 'arrayAnnNames' in st.annotations:
                    st.annotations['arrayAnnNames'].append('pw')
                else:
                    st.annotations['arrayAnnNames'] = ['pw']
                #
                ampWithinSpec = np.abs(amplitudes) < 4
                plotMask = st.times > 0 # < 1360
                if arguments['plotting']:
                    plt.plot(st.sampling_period * np.arange(wvf.shape[1]), wvf.iloc[plotMask, :].T * 1e-6, 'o-'); plt.title('{} fixed wvf peak at {}'.format(st.name, idxPeak)); plt.show()
                    plt.plot(st.sampling_period * np.arange(wvf.shape[1]), (wvfDiffStd).iloc[:, :].T * 1e-6, 'o-');
                    plt.plot(st.sampling_period * np.arange(wvf.shape[1]), (wvfDiffStd).iloc[:, :].mean().T * 1e-6, 'o-', lw=3); plt.title('{} fixed diff peak at {}'.format(st.name, idxPeak)); plt.show()
                matchingAsig = nspBlock.filter(objects=AnalogSignalProxy, name='seg0_' + chanName)
                if len(matchingAsig):
                    keepStimRasterList.append(chanName)
                    elecImpedance = (
                        impedancesRipple
                        .loc[impedancesRipple['elec'] == chanName, 'impedance'])
                    currents = amplitudes / (elecImpedance.iloc[0] * pq.kOhm)
                    st.annotations['current'] = currents
                    st.array_annotations['current'] = currents
                    if 'arrayAnnNames' in st.annotations:
                        st.annotations['arrayAnnNames'].append('current')
                    else:
                        st.annotations['arrayAnnNames'] = ['current']
            stimActive = stimRastersDF[keepStimRasterList].sum(axis=1) > 0
            activeTimes = stimRastersDF.loc[stimActive, 't']
            #
            stimEvents[:] = (
                stimEvents.times -
                stimEvents.times[0] +
                activeTimes.min() * pq.s)
            #
            peakIdx, _, trainStartIdx, trainEndIdx = hf.findTrains(
                peakTimes=activeTimes, iti=10e-3)
            trainDurations = trainEndIdx - trainStartIdx
            #
            if len(trainStartIdx):
                startCategories = pd.DataFrame(
                    activeTimes[trainStartIdx].to_numpy(),
                    # index=range(activeTimes[trainStartIdx].size),
                    columns=['t'])
                startCategories = startCategories.reindex(columns=[
                    'amplitude', 'current', 'program',
                    'activeGroup', 'pw', 'electrode',
                    'RateInHz', 'trainDuration', 't'])
                #
                latestProgram = 0
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
                            # pdb.set_trace()
                            st = [
                                i
                                for i in spikeList
                                if i.unit.channel_index.name == activeChan][0] # + '_stim#0'
                            theseTimesMask = (
                                (st.times >= stimRastersDF.loc[idxStart, 't'] * pq.s - 1.1 * samplingRate ** (-1) / 2) &
                                (st.times <= stimRastersDF.loc[idxEnd, 't'] * pq.s + 1.1 * samplingRate ** (-1) / 2)
                                )
                            theseTimes = st.times[theseTimesMask]
                            if not theseTimesMask.sum():
                                pdb.set_trace()
                            stimRasterAmplitude[activeChan] = np.mean(
                                st.annotations['amplitude'][theseTimesMask])
                            stimRasterCurrent[activeChan] = np.mean(
                                st.annotations['current'][theseTimesMask])
                            if activeChanIdx == 0:
                                if theseTimes.size == 1:
                                    startCategories.loc[
                                        idx, 'trainDuration'] = 0
                                    startCategories.loc[
                                        idx, 'RateInHz'] = 0
                                else:
                                    startCategories.loc[
                                        idx, 'trainDuration'] = (
                                            theseTimes[-1] -
                                            theseTimes[0])
                                    startCategories.loc[
                                        idx, 'RateInHz'] = (
                                            np.diff(theseTimes).mean() ** (-1))
                                startCategories.loc[
                                    idx, 'pw'] = np.mean(
                                        st.annotations['pw'][theseTimesMask])
                        startCategories.loc[idx, 'activeGroup'] = 1
                        electrodeShortHand = ''
                        negativeAmps = stimRasterCurrent < 0
                        if (negativeAmps).any():
                            electrodeShortHand += '-'
                            totalCathode = stimRasterCurrent[negativeAmps].sum()
                            startCategories.loc[idx, 'current'] = totalCathode
                            averageImpedance = np.mean(impedancesRipple.loc[impedancesRipple['elec'].isin(stimRasterCurrent[negativeAmps].index), 'impedance'])
                            startCategories.loc[idx, 'amplitude'] = totalCathode * averageImpedance
                            for cName in stimRasterCurrent[negativeAmps].index:
                                electrodeShortHand += cName[:-2]
                        positiveAmps = stimRasterCurrent > 0
                        if (positiveAmps).any():
                            electrodeShortHand += '-'
                            # totalAnode = stimRasterCurrent[positiveAmps].sum()
                            for cName in stimRasterCurrent[positiveAmps].index:
                                electrodeShortHand += cName[:-2]
                        startCategories.loc[idx, 'electrode'] = electrodeShortHand
                        if electrodeShortHand not in startCategories['electrode'].unique().tolist() and idx != 0:
                            latestProgram += 1
                        startCategories.loc[idx, 'program'] = latestProgram
                #
                stopCategories = startCategories.copy()
                stopCategories['t'] = activeTimes[trainEndIdx].to_numpy()
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
                pdb.set_trace()
                startCategories.dropna(inplace=True)
                stopCategories.dropna(inplace=True)
        #
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
    #
    spikesBlock = ns5.purgeNixAnn(spikesBlock)
    writer = neo.io.NixIO(
        filename=analysisDataPath.format(arguments['analysisName']))
    writer.write_block(spikesBlock, use_obj_names=True)
    writer.close()
    #
    tdBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSpikes=False, keepSignals=tdChanNames)
    tdBlock = hf.loadBlockProxyObjects(tdBlock)
    #
    tdDF = ns5.analogSignalsToDataFrame(
        tdBlock.filter(objects=AnalogSignal))
    #
    currentSamplingRate = tdBlock.filter(
        objects=AnalogSignal)[0].sampling_rate
    if samplingRate != currentSamplingRate:
        tdInterp = hf.interpolateDF(
            tdDF, newT,
            kind='linear', fill_value=(0, 0),
            x='t', columns=tdChanNames)
    else:
        tdInterp = tdDF
    #
    tdInterp.columns = [i.replace('seg0_', '') for i in tdInterp.columns]
    #
    tdBlockInterp = ns5.dataFrameToAnalogSignals(
        tdInterp,
        idxT='t', useColNames=True,
        dataCol=tdInterp.drop(columns='t').columns,
        samplingRate=samplingRate)
    ns5.addBlockToNIX(
        tdBlockInterp, neoSegIdx=[0],
        writeSpikes=False, writeEvents=False,
        purgeNixNames=False,
        fileName=ns5FileName + '_analyze',
        folderPath=analysisSubFolder,
        nixBlockIdx=0, nixSegIdx=[0],
        )
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