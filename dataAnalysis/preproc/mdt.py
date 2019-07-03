import pandas as pd
import os, pdb
from neo import (
    AnalogSignal, Event, Block,
    Segment, ChannelIndex, SpikeTrain, Unit)
import neo
import elephant as elph
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import matplotlib.pyplot as plt
import numpy as np
import traceback
#  import sys
#  import pickle
#  from copy import *
import quantities as pq
#  import argparse, linecache
from scipy import stats, signal, ndimage
import peakutils
import seaborn as sns
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler


def preprocINS(
        trialFilesStim,
        insDataFilename,
        plottingFigures=False,
        plotBlocking=True):
    print('Preprocessing')
    jsonBaseFolder = trialFilesStim['folderPath']
    jsonSessionNames = trialFilesStim['jsonSessionNames']

    td = hf.getINSTDFromJson(
        jsonBaseFolder, jsonSessionNames, getInterpolated=True,
        forceRecalc=trialFilesStim['forceRecalc'])

    elecConfiguration, senseInfo = (
        hf.getINSDeviceConfig(jsonBaseFolder, jsonSessionNames[0])
        )
    renamer = {}
    tdDataCols = []
    for colName in td['data'].columns:
        if 'channel_' in colName:
            idx = int(colName[-1])
            updatedName = 'channel_{}'.format(senseInfo.loc[idx, 'senseChan'])
            tdDataCols.append(updatedName)
            renamer.update({colName: updatedName})
    td['data'].rename(columns=renamer, inplace=True)

    accel = hf.getINSAccelFromJson(
        jsonBaseFolder, jsonSessionNames, getInterpolated=True,
        forceRecalc=trialFilesStim['forceRecalc'])

    timeSync = hf.getINSTimeSyncFromJson(
        jsonBaseFolder, jsonSessionNames,
        forceRecalc=True)

    stimStatusSerial = hf.getINSStimLogFromJson(
        jsonBaseFolder, jsonSessionNames)
    #  packets are aligned to hf.INSReferenceTime, for convenience
    #  (otherwise the values in ['t'] would be huge)
    #  align them to the more reasonable minimum first timestamp across packets

    #  System Tick seconds before roll over
    rolloverSeconds = pd.to_timedelta(6.5535, unit='s')

    for trialSegment in pd.unique(td['data']['trialSegment']):

        accelSegmentMask = accel['data']['trialSegment'] == trialSegment
        accelGroup = accel['data'].loc[accelSegmentMask, :]
        tdSegmentMask = td['data']['trialSegment'] == trialSegment
        tdGroup = td['data'].loc[tdSegmentMask, :]
        timeSyncSegmentMask = timeSync['trialSegment'] == trialSegment
        timeSyncGroup = timeSync.loc[timeSyncSegmentMask, :]

        streamInitTimestamps = pd.Series({
            'td': tdGroup['time_master'].iloc[0],
            'accel': accelGroup['time_master'].iloc[0],
            'timeSync': timeSyncGroup['time_master'].iloc[0],
            })
        print('streamInitTimestamps\n{}'.format(streamInitTimestamps))
        streamInitSysTicks = pd.Series({
            'td': tdGroup['microseconds'].iloc[0],
            'accel': accelGroup['microseconds'].iloc[0],
            'timeSync': timeSyncGroup['microseconds'].iloc[0],
            })
        print('streamInitTimestamps\n{}'.format(streamInitSysTicks))
        # get first timestamp in session for each source
        sessionMasterTime = min(streamInitTimestamps)
        sessionTimeRef = streamInitTimestamps - sessionMasterTime

        #  get first systemTick in session for each source
        #  Note: only look within the master timestamp
        happenedBeforeSecondTimestamp = (
            sessionTimeRef == pd.Timedelta(0)
            )
        ticksInFirstSecond = streamInitSysTicks.loc[
            happenedBeforeSecondTimestamp]
        if len(ticksInFirstSecond) == 1:
            sessionMasterTick = ticksInFirstSecond.iloc[0]
        else:
            #  if there are multiple candidates for first systick, need to
            #  make sure there was no rollover
            lowestTick = min(ticksInFirstSecond)
            highestTick = max(ticksInFirstSecond)
            if (highestTick - lowestTick) > pd.Timedelta(1, unit='s'):
                #  the counter rolled over in the first second
                rolloverMask = ticksInFirstSecond > pd.Timedelta(1, unit='s')
                underRollover = ticksInFirstSecond.index[rolloverMask]

                sessionMasterTick = min(ticksInFirstSecond[underRollover])
                #  refTo = ticksInFirstSecond[underRollover].idxmin()
            else:
                # no rollover in first second
                sessionMasterTick = min(ticksInFirstSecond)
                #  refTo = ticksInFirstSecond.idxmin()

        #  now check for rollover between first systick and anything else
        #  look for other streams that should be within rollover of start
        rolloverCandidates = ~(sessionTimeRef > rolloverSeconds)
        #  are there any streams that appear to start before the first stream?
        rolledOver = (streamInitSysTicks < sessionMasterTick) & (
            rolloverCandidates)

        if trialSegment == 0:
            absoluteRef = sessionMasterTime
            alignmentFactor = pd.Series(
                - sessionMasterTick, index=streamInitSysTicks.index)
        else:
            alignmentFactor = pd.Series(
                sessionMasterTime-absoluteRef-sessionMasterTick,
                index=streamInitSysTicks.index)
        #  correct any roll-over        
        alignmentFactor[rolledOver] += rolloverSeconds
        print('alignmentFactor\n{}'.format(alignmentFactor))
    
        accel = hf.realignINSTimestamps(
            accel, trialSegment, alignmentFactor.loc['accel'])
        td = hf.realignINSTimestamps(
            td, trialSegment, alignmentFactor.loc['td'])
        timeSync = hf.realignINSTimestamps(
            timeSync, trialSegment, alignmentFactor.loc['timeSync'])

    #  Check timeSync
    checkTimeSync = False
    if checkTimeSync and plottingFigures:
        for name, group in timeSync.groupby('trialSegment'):
            print('Segment {} head:'.format(name))
            print(group.loc[:, ('time_master', 'microseconds', 't')].head())
            print('Segment {} tail:'.format(name))
            print(group.loc[:, ('time_master', 'microseconds', 't')].tail())

        plt.plot(
            np.linspace(0, 1, len(timeSync['t'])),
            timeSync['t'], 'o', label='timeSync')
        plt.plot(
            np.linspace(0, 1, len(td['t'])), td['t'],
            'o', label='td')
        plt.plot(
            np.linspace(0, 1, len(accel['t'])), accel['t'],
            'o', label='accel')
        plt.legend()
        plt.show()
        plt.plot(np.linspace(
            0, 1, len(timeSync['microseconds'])),
            timeSync['microseconds'], 'o',label='timeSync')
        plt.plot(np.linspace(
            0, 1, len(td['data']['microseconds'])),
            td['data']['microseconds'], 'o',label='td')
        plt.plot(np.linspace(
            0, 1, len(accel['data']['microseconds'])),
            accel['data']['microseconds'], 'o',label='accel')
        plt.legend()
        plt.show()
        plt.plot(np.linspace(
            0, 1, len(timeSync['time_master'])),
            timeSync['time_master'], 'o',label='timeSync')
        plt.plot(np.linspace(
            0, 1, len(td['data']['time_master'])),
            td['data']['time_master'], 'o',label='td')
        plt.plot(np.linspace(
            0, 1, len(accel['data']['time_master'])),
            accel['data']['time_master'], 'o',label='accel')
        plt.legend()
        plt.show()

    progAmpNames = rcsa_helpers.progAmpNames
    expandCols = (
        ['RateInHz', 'therapyStatus', 'trialSegment'] +
        progAmpNames)
    deriveCols = ['amplitudeRound']
    stimStatus = hf.stimStatusSerialtoLong(
        stimStatusSerial, idxT='HostUnixTime', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)

    interpFunHUTtoINS = hf.getHUTtoINSSyncFun(
        timeSync, degree=1, syncTo='PacketGenTime')
    for trialSegment in pd.unique(td['data']['trialSegment']):
        stimStatus = hf.synchronizeHUTtoINS(
            stimStatus, trialSegment, interpFunHUTtoINS[trialSegment])
        stimStatusSerial = hf.synchronizeHUTtoINS(
            stimStatusSerial, trialSegment, interpFunHUTtoINS[trialSegment])
    #  sync Host PC Unix time to NSP
    HUTtoINSPlotting = True
    if HUTtoINSPlotting and plottingFigures:
        plottingColumns = deriveCols + expandCols + progAmpNames
        plotStimStatusList = []
        for trialSegment in pd.unique(td['data']['trialSegment']):
            stimGroupMask = stimStatus['trialSegment'] == trialSegment
            stimGroup = stimStatus.loc[stimGroupMask, :]
            thisHUTtoINS = interpFunHUTtoINS[trialSegment]
            #
            plottingRange = np.arange(
                stimGroup['HostUnixTime'].min(),
                stimGroup['HostUnixTime'].max(), 10)  # every 2msec
            temp = hf.interpolateDF(
                stimGroup, plottingRange,
                x='HostUnixTime', columns=plottingColumns, kind='previous')
            temp['amplitudeIncrease'] = temp['amplitudeRound'].diff().fillna(0)
            temp['INSTime'] = pd.Series(
                thisHUTtoINS(temp['HostUnixTime']),
                index=temp['HostUnixTime'].index)
            plotStimStatusList.append(temp)
        plotStimStatus = pd.concat(
            plotStimStatusList,
            ignore_index=True)

    if HUTtoINSPlotting and plottingFigures:
        tStartTD = 200
        tStopTD = 300
        hf.plotHUTtoINS(
            td, accel, plotStimStatus,
            tStartTD, tStopTD,
            sharex=True, dataCols=tdDataCols,
            plotBlocking=plotBlocking
            )

    block = insDataToBlock(
        td, accel, stimStatusSerial,
        senseInfo, trialFilesStim,
        tdDataCols=tdDataCols)

    #  stim detection
    if trialFilesStim['detectStim']:
        block = getINSStimOnset(
            block, elecConfiguration,
            **trialFilesStim['getINSkwargs'])
        #  if we did stim detection, recalculate stimStatusSerial
        stimSpikes = block.filter(objects=SpikeTrain)
        stimSpikes = ns5.loadContainerArrayAnn(trainList=stimSpikes)
        stimSpikesDF = ns5.unitSpikeTrainArrayAnnToDF(stimSpikes)
        stimSpikesDF['ratePeriod'] = stimSpikesDF['RateInHz'] ** (-1)
        onsetEvents = pd.melt(
            stimSpikesDF,
            id_vars=['t'],
            value_vars=['program', 'amplitude', 'RateInHz', 'pulseWidth'],
            var_name='ins_property', value_name='ins_value')
        
        onsetEvents.rename(columns={'t': 'INSTime'}, inplace=True)
        offsetEvents = pd.melt(
            stimSpikesDF,
            id_vars=['endTime'],
            value_vars=['program', 'amplitude'],
            var_name='ins_property', value_name='ins_value')
        offsetEvents.loc[
            offsetEvents['ins_property'] == 'amplitude',
            'ins_value'] = 0
        offsetEvents.rename(columns={'endTime': 'INSTime'}, inplace=True)
        firstNonZeroAmplitudeMask = (
            (stimStatusSerial['ins_property'] == 'amplitude') &
            (stimStatusSerial['ins_value'] > 0)
        )
        firstNonZeroAmplitudeTime = stimStatusSerial.iloc[
            firstNonZeroAmplitudeMask
            .index[firstNonZeroAmplitudeMask][0]
            ]['INSTime']
        keepMask = (
            (
                stimStatusSerial['ins_property']
                .isin(['trialSegment', 'activeGroup', 'therapyStatus'])) |
            (stimStatusSerial['INSTime'] < firstNonZeroAmplitudeTime)
            )
        newStimStatusSerial = stimStatusSerial.loc[
            keepMask,
            ['INSTime', 'ins_property', 'ins_value']]
        # pdb.set_trace()
        newStimStatusSerial = pd.concat(
            [newStimStatusSerial, onsetEvents, offsetEvents])
        newStimStatusSerial = (
            newStimStatusSerial
            .sort_values('INSTime', kind='mergesort')
            .reset_index(drop=True)
            )
        #  pdb.set_trace()
        newStimEvents = ns5.eventDataFrameToEvents(
            newStimStatusSerial, idxT='INSTime',
            annCol=['ins_property', 'ins_value']
            )
        block.segments[0].events = newStimEvents
        block.create_relationship()
    #
    writer = neo.io.NixIO(filename=insDataFilename, mode='ow')
    writer.write_block(block, use_obj_names=True)
    writer.close()
    return block


def getINSStimOnset(
        block, elecConfiguration,
        cyclePeriod=0, minDist=0, minDur=0,
        gaussWid=600e-3,
        timeInterpFunINStoNSP=None,
        maxSpikesPerGroup=None,
        correctThresholdWithAmplitude=True,
        recalculateExpectedOffsets=False,
        fixedDelay=0e-3, delayByFreqMult=0,
        cyclePeriodCorrection=18e-3,
        stimDetectOptsByChannel=None,
        plotAnomalies=False,
        spikeWindow=[-32, 64],
        plotting=[]):

    segIdx = 0
    seg = block.segments[segIdx]
    fs = seg.analogsignals[0].sampling_rate
    tdDF, accelDF, stimStatus = unpackINSBlock(block)

    #  assume a fixed delay between onset and stim
    fixedDelayIdx = int(fixedDelay * fs)
    print('Using a fixed delay of {} samples'.format(fixedDelayIdx))
    defaultOptsDict = {
        'detectChannels': [i for i in tdDF.columns if 'ins_td' in i]}

    if stimDetectOptsByChannel is None:
        stimDetectOptsByChannel = {
            grpIdx: {progIdx: defaultOptsDict for progIdx in range(4)}
            for grpIdx in range(4)}
    
    #  allocate units for each group/program pair
    for groupIdx in range(4):
        for progIdx in range(4):
            electrodeCombo = 'g{:d}p{:d}'.format(
                groupIdx, progIdx)
            mdtIdx = int(4 * groupIdx + progIdx)
            chanIdx = ChannelIndex(
                name=electrodeCombo,
                index=[mdtIdx])
            block.channel_indexes.append(chanIdx)

            thisElecConfig = elecConfiguration[
                groupIdx][progIdx]
            thisUnit = Unit(name=electrodeCombo)
            thisUnit.annotate(group=groupIdx)
            thisUnit.annotate(program=progIdx)
            thisUnit.annotate(cathodes=thisElecConfig['cathodes'])
            thisUnit.annotate(anodes=thisElecConfig['anodes'])
            try:
                theseDetOpts = stimDetectOptsByChannel[
                    groupIdx][progIdx]
            except Exception:
                theseDetOpts = defaultOptsDict
            for key, value in theseDetOpts.items():
                thisUnit.annotations.update({key: value})
            chanIdx.units.append(thisUnit)
            thisUnit.channel_index = chanIdx
    
    spikeTStop = tdDF['t'].iloc[-1]
    spikeTStart = tdDF['t'].iloc[0]
    
    for name, group in tdDF.groupby('amplitudeRound'):
        anomalyOccured = False
        #  pad with paddingDuration msec to ensure robust z-score
        paddingDuration = 150e-3
        
        tStart = group['t'].iloc[0]
        tStop = group['t'].iloc[-1]
        
        if (tStop - tStart) < minDur:
            print('group {} (tStop - tStart) < minDur'.format(name))
            continue
        
        if (group['amplitude'] > 0).any():
            ampOnMask = group['amplitude'] > 0
        else:
            print('Amplitude never turned on!')
            continue

        activeState = (
            bool(
                group
                .loc[ampOnMask, 'therapyStatus']
                .value_counts()
                .idxmax()))
        if not activeState:
            print('group {} Therapy not active!'.format(name))
            continue

        activeProgram = (
            int(
                group
                .loc[ampOnMask, 'program']
                .value_counts()
                .idxmax()))
        ampColName = 'program{}_amplitude'.format(activeProgram)
        thisAmplitude = group[ampColName].max()
        #  stim artifacts will be in this temporal subset of the recording
        groupAmpMask = (
            (group[ampColName] == thisAmplitude) &
            (group['therapyStatus'] > 0))
        
        tStartPadded = max(
            0,
            group.loc[ampOnMask, 't'].iloc[0] - paddingDuration)
        tStopPadded = min(
            group.loc[ampOnMask, 't'].iloc[-1] + paddingDuration,
            tdDF['t'].iloc[-1])            
        plotMaskTD = (tdDF['t'] > tStartPadded) & (tdDF['t'] < tStopPadded)

        activeGroup = (
            int(
                group
                .loc[ampOnMask, 'activeGroup']
                .value_counts()
                .idxmax()))
        thisTrialSegment = (
            int(
                group
                .loc[ampOnMask, 'trialSegment']
                .value_counts()
                .idxmax()))
        stimRate = (
            group
            .loc[ampOnMask, 'RateInHz']
            .value_counts()
            .idxmax())
        
        stimPW = (
            group
            .loc[ampOnMask, 'pulseWidth']
            .value_counts()
            .idxmax())
        #  changes to stim happen at least a full period after the request
        delayByFreq = (delayByFreqMult / stimRate)
        delayByFreqIdx = int(fs * delayByFreq)
        #  load the appropriate detection options
        theseDetectOpts = stimDetectOptsByChannel[activeGroup][activeProgram]
        #  calculate signal used for stim artifact detection
        tdSeg = (tdDF.loc[
            plotMaskTD,
            theseDetectOpts['detectChannels'] + ['t']
            ])
        #  how far after the nominal start and stop of stim should we look?
        ROIPadding = [-gaussWid, gaussWid]
        #  make sure we didn't overextend onset estimates
        lastValidOnsetIdx = group.loc[groupAmpMask, 't'].index[-1]
        #  make sure we didn't overextend offset estimates
        lastValidOffsetIdx = min(
            lastValidOnsetIdx +
            delayByFreqIdx + fixedDelayIdx,
            tdSeg.index[-1])

        # use the HUT derived stim onset to favor detection
        expectedOnsetIdx = np.atleast_1d(np.array(
            max(
                group.index[0] + fixedDelayIdx + delayByFreqIdx,
                tdSeg.index[0])
            ))
        expectedOffsetIdx = np.atleast_1d(
            np.array(lastValidOffsetIdx))
        # if we know cycle value, use it to predict onsets
        thisElecConfig = elecConfiguration[activeGroup][activeProgram]
        if thisElecConfig['cyclingEnabled']:
            cyclePeriod = (
                thisElecConfig['cycleOffTime']['time'] +
                thisElecConfig['cycleOnTime']['time']) / 10
            cycleOnTime = (
                thisElecConfig['cycleOnTime']['time']) / 10
            cycleOnTimeIdx = int(fs * cycleOnTime)

            #  cycle period lags behind!
            cyclePeriodIdx = int(fs*(cyclePeriod + cyclePeriodCorrection))
            expectedOnsetIdx = np.arange(
                expectedOnsetIdx[0], lastValidOnsetIdx + cyclePeriodIdx,
                cyclePeriodIdx
                )
            expectedOnsetIdx = (
                expectedOnsetIdx[expectedOnsetIdx <= lastValidOnsetIdx])

            expectedOffsetIdx = (expectedOnsetIdx + cycleOnTimeIdx)
            expectedOffsetIdx = (
                expectedOffsetIdx[expectedOffsetIdx < lastValidOffsetIdx])
            if len(expectedOffsetIdx) == (len(expectedOnsetIdx) - 1):
                expectedOffsetIdx = np.append(
                    expectedOffsetIdx, [lastValidOffsetIdx])
        else:
            cyclePeriod = -1  # placeholder for no cycling

        #
        tStartOnset = max(
            tdSeg['t'].iloc[0],
            tdSeg.loc[expectedOnsetIdx[0], 't'] + ROIPadding[0])
        tStopOnset = min(
            tdSeg.loc[expectedOnsetIdx[-1], 't'] + ROIPadding[1],
            group['t'].iloc[-1])
        ROIMaskOnset = (
            (tdSeg['t'] > tStartOnset) &
            (tdSeg['t'] < tStopOnset))

        tStartOffset = max(
            group['t'].iloc[0],
            tdSeg.loc[expectedOffsetIdx[0], 't'] + ROIPadding[0])
        #  tStop can't be off group
        tStopOffset = min(
            tdSeg.loc[expectedOffsetIdx[-1], 't'] + ROIPadding[1],
            group['t'].iloc[-1])
        ROIMaskOffset = (
            (tdSeg['t'] > tStartOffset) &
            (tdSeg['t'] < tStopOffset))
        
        (
            onsetIdx, theseOnsetTimestamps, onsetDifferenceFromExpected,
            onsetDifferenceFromLogged,
            anomalyOccured, originalOnsetIdx, correctionFactorOnset,
            onsetDetectSignalFull, currentThresh) = extractArtifactTimestamps(
                tdSeg.drop(columns='t'), tdDF, ROIMaskOnset,
                fs, gaussWid,
                stimRate=stimRate,
                closeThres=gaussWid,
                expectedIdx=expectedOnsetIdx,
                enhanceEdges=True,
                enhanceExpected=True,
                correctThresholdWithAmplitude=True,
                thisAmplitude=thisAmplitude,
                name=name, plotting=plotting,
                plotAnomalies=plotAnomalies, anomalyOccured=anomalyOccured,
                minDist=minDist, theseDetectOpts=theseDetectOpts,
                maxSpikesPerGroup=maxSpikesPerGroup,
                fixedDelayIdx=fixedDelayIdx, delayByFreqIdx=delayByFreqIdx,
                keep_what='max', plotDetection=False,
            )

        # recalculate expected off times based on detected on times
        if recalculateExpectedOffsets:
            # expectedOnsetTimestamps = pd.Series(
            #     tdSeg['t'].loc[expectedOnsetIdx])
            expectedOnsetTimestamps = theseOnsetTimestamps
            expectedOffsetTimestamps = pd.Series(
                tdSeg['t'].loc[expectedOffsetIdx])
            if len(expectedOnsetTimestamps) == len(expectedOffsetTimestamps):
                expectedTrainDurations = (
                    expectedOffsetTimestamps.values -
                    expectedOnsetTimestamps.values)
            else:
                dummyOffsets, _ = hf.closestSeries(
                    expectedOnsetTimestamps,
                    expectedOffsetTimestamps, strictly='greater')
                expectedTrainDurations = (
                    dummyOffsets.values -
                    expectedOnsetTimestamps.values)
            newExpectedOffsetTimestamps, _ = hf.closestSeries(
                theseOnsetTimestamps + expectedTrainDurations, tdSeg['t'])
            expectedOffsetIdx = np.array(
                tdSeg['t'].loc[tdSeg['t'].isin(newExpectedOffsetTimestamps.values)].index)
        else:
            expectedOffsetIdx = np.array(expectedOffsetIdx)
        #  double check that we don't look overboard
        if expectedOffsetIdx[-1] > lastValidOffsetIdx:
            expectedOffsetIdx[-1] = lastValidOffsetIdx
        (
            offsetIdx, theseOffsetTimestamps, offsetDifferenceFromExpected,
            offsetDifferenceFromLogged,
            anomalyOccured, originalOffsetIdx, correctionFactorOffset,
            offsetDetectSignalFull, currentThresh) = extractArtifactTimestamps(
                tdSeg.drop(columns='t'), tdDF, ROIMaskOffset,
                fs, gaussWid,
                stimRate=stimRate,
                closeThres=gaussWid,
                expectedIdx=expectedOffsetIdx,
                enhanceEdges=True,
                enhanceExpected=True,
                correctThresholdWithAmplitude=True,
                thisAmplitude=thisAmplitude,
                name=name, plotting=plotting,
                plotAnomalies=plotAnomalies, anomalyOccured=anomalyOccured,
                minDist=minDist, theseDetectOpts=theseDetectOpts,
                maxSpikesPerGroup=maxSpikesPerGroup,
                fixedDelayIdx=fixedDelayIdx, delayByFreqIdx=delayByFreqIdx,
                keep_what='max', plotDetection=False
            )

        if timeInterpFunINStoNSP is not None:
            # synchronize stim timestamps with INS timestamps
            theseOnsetTimestamps.iloc[:] = timeInterpFunINStoNSP[thisTrialSegment](
                theseOnsetTimestamps.values)
        
        closestOffsets, _ = hf.closestSeries(
            theseOnsetTimestamps,
            theseOffsetTimestamps, strictly='greater')
        theseOnsetTimestamps = theseOnsetTimestamps.values * pq.s
        closestOffsets = closestOffsets.values * pq.s
        electrodeCombo = 'g{:d}p{:d}'.format(
            activeGroup, activeProgram)
        
        if len(theseOnsetTimestamps):
            thisUnit = block.filter(
                objects=Unit,
                name=electrodeCombo
                )[0]

            ampList = theseOnsetTimestamps ** 0 * 100 * thisAmplitude * pq.uA
            rateList = theseOnsetTimestamps ** 0 * stimRate * pq.Hz
            pwList = theseOnsetTimestamps ** 0 * 10 * stimPW * pq.ms
            programList = theseOnsetTimestamps ** 0 * activeProgram * pq.dimensionless
            tSegList = theseOnsetTimestamps ** 0 * thisTrialSegment
            
            arrayAnn = {
                'amplitude': ampList, 'RateInHz': rateList,
                'pulseWidth': pwList,
                'trialSegment': tSegList,
                'endTime': closestOffsets,
                'program': programList,
                'offsetFromExpected': onsetDifferenceFromExpected,
                'offsetFromLogged': onsetDifferenceFromLogged}
        
            st = SpikeTrain(
                times=theseOnsetTimestamps, t_stop=spikeTStop,
                t_start=spikeTStart,
                name=thisUnit.name,
                array_annotations=arrayAnn,
                **arrayAnn)
            #  st.annotate(amplitude=thisAmplitude * 100 * pq.uA)
            #  st.annotate(rate=stimRate * pq.Hz)
            thisUnit.spiketrains.append(st)
        #
        if (name in plotting) or anomalyOccured:
            print('About to Plot')
            print('{}'.format(onsetIdx))

            fig, ax = plt.subplots(3, 1, sharex=True)
            '''
            if accelDF is not None:
                plotMaskAccel = (accelDF['t'] > tStart) & (
                    accelDF['t'] < tStop)
                ax[0].plot(
                    accelDF['t'].loc[plotMaskAccel],
                    stats.zscore(
                        accelDF.loc[plotMaskAccel, 'ins_accinertia']),
                    '-', label='inertia')
            '''
            for channelName in theseDetectOpts['detectChannels']:
                ax[0].plot(
                    tdDF['t'].loc[plotMaskTD],
                    stats.zscore(tdDF.loc[plotMaskTD, channelName]),
                    '-', label=channelName)
            ax[0].legend(loc='best')
            ax[0].set_title('INS Accel and TD')

            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                onsetDetectSignalFull,
                'k-', lw=0.25)
            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                offsetDetectSignalFull,
                'k-', lw=0.25)
            ax[1].plot(
                tdDF['t'].loc[ROIMaskOnset[ROIMaskOnset].index],
                onsetDetectSignalFull.loc[ROIMaskOnset],
                'b-', label='onsetDetectSignal')
            ax[1].plot(
                tdDF['t'].loc[ROIMaskOffset[ROIMaskOffset].index],
                offsetDetectSignalFull.loc[ROIMaskOffset],
                'r--', label='offsetDetectSignal')
                
            ax[1].plot(
                tdDF['t'].loc[onsetIdx],
                onsetDetectSignalFull.loc[onsetIdx],
                'bo', markersize=10, label='detected onset')
            ax[1].plot(
                tdDF['t'].loc[offsetIdx],
                offsetDetectSignalFull.loc[offsetIdx],
                'ro', markersize=10, label='detected offset')

            try:
                ax[1].plot(
                    tdDF['t'].loc[originalOnsetIdx],
                    onsetDetectSignalFull.loc[originalOnsetIdx],
                    'y^', label='original detected onset')
            except Exception:
                pass
            try:
                ax[1].plot(
                    tdDF['t'].loc[originalOffsetIdx],
                    offsetDetectSignalFull.loc[originalOffsetIdx],
                    'y^', label='original detected offset')
            except Exception:
                pass

            ax[1].plot(
                tdDF['t'].loc[expectedOnsetIdx],
                onsetDetectSignalFull.loc[expectedOnsetIdx],
                'c*', label='expected onset')
            ax[1].plot(
                tdDF['t'].loc[expectedOffsetIdx],
                onsetDetectSignalFull.loc[expectedOffsetIdx],
                'm*', label='expected offset')
            ax[1].axhline(
                currentThresh,
                color='r', label='detection Threshold')

            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                correctionFactorOnset,
                'k-', label='correctionFactorOnset')
            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                correctionFactorOffset,
                'k--', label='correctionFactorOffset')

            ax[1].legend(loc='best')
            ax[1].set_title('INS TD Measurements')
            progAmpNames = rcsa_helpers.progAmpNames
            for columnName in progAmpNames:
                ax[2].plot(
                    tdDF.loc[plotMaskTD, 't'],
                    tdDF.loc[plotMaskTD, columnName],
                    '-', label=columnName, lw=2.5)

            statusAx = ax[2].twinx()
            statusAx.plot(
                tdDF.loc[plotMaskTD, 't'],
                tdDF.loc[plotMaskTD, 'therapyStatus'],
                '--', label='therapyStatus', lw=1.5)
            
            statusAx.plot(
                tdDF.loc[plotMaskTD, 't'],
                tdDF.loc[plotMaskTD, 'amplitudeIncrease'],
                'c--', label='amplitudeIncrease', lw=1.5)
            
            ax[2].legend(loc='upper left')    
            statusAx.legend(loc='upper right')
            ax[2].set_ylabel('Stim Amplitude (mA)')
            ax[2].set_xlabel('NSP Time (sec)')

            for thisAx in ax:
                for linePos in tdDF['t'].loc[onsetIdx].values:
                    thisAx.axvline(
                        linePos,
                        color='b')
                for linePos in tdDF['t'].loc[offsetIdx].values:
                    thisAx.axvline(
                        linePos,
                        color='r')
                for linePos in tdDF['t'].loc[expectedOnsetIdx].values:
                    thisAx.axvline(
                        linePos,
                        color='c')
                for linePos in tdDF['t'].loc[expectedOffsetIdx].values:
                    thisAx.axvline(
                        linePos,
                        color='m')
            plt.suptitle(
                'stimAmp: {} stimRate: {}'.format(thisAmplitude, stimRate))
            plt.show()
    #
    createRelationship = True
    for thisUnit in block.filter(objects=Unit):
        if len(thisUnit.spiketrains) == 0:
            st = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=[], units='sec', t_stop=spikeTStop,
                t_start=spikeTStart)
            thisUnit.spiketrains.append(st)
            seg.spiketrains.append(st)
        else:
            #  consolidate spiketrains
            consolidatedTimes = np.array([])
            consolidatedAnn = {
                'amplitude': np.array([]),
                'RateInHz': np.array([]),
                'program': np.array([]),
                'pulseWidth': np.array([]),
                'endTime': np.array([]),
                'trialSegment': np.array([]),
                'offsetFromExpected': np.array([]),
                'offsetFromLogged': np.array([])
                }
            arrayAnnNames = {'arrayAnnNames': list(consolidatedAnn.keys())}
            for idx, st in enumerate(thisUnit.spiketrains):
                consolidatedTimes = np.concatenate((
                    consolidatedTimes,
                    st.times.magnitude
                ))
                for key, value in consolidatedAnn.items():
                    consolidatedAnn[key] = np.concatenate((
                        consolidatedAnn[key],
                        st.annotations[key]
                        ))
            #
            unitDetectedOn = thisUnit.annotations['detectChannels']
            consolidatedTimes, timesIndex = hf.closestSeries(
                takeFrom=pd.Series(consolidatedTimes),
                compareTo=tdDF['t'])
            timesIndex = np.array(timesIndex.values, dtype=np.int)
            left_sweep_samples = spikeWindow[0] * (-1)
            left_sweep = left_sweep_samples / fs
            right_sweep_samples = spikeWindow[1] - 1
            right_sweep = right_sweep_samples / fs
            #  spike_duration = left_sweep + right_sweep
            spikeWaveforms = np.zeros(
                (
                    timesIndex.shape[0], len(unitDetectedOn),
                    left_sweep_samples + right_sweep_samples + 1),
                dtype=np.float)
            for idx, tIdx in enumerate(timesIndex):
                thisWaveform = (
                    tdDF.loc[
                        tIdx - int(left_sweep * fs):tIdx + int(right_sweep * fs),
                        unitDetectedOn])
                if spikeWaveforms[idx, :, :].shape == thisWaveform.shape:
                    spikeWaveforms[idx, :, :] = np.swapaxes(
                        thisWaveform.values, 0, 1)
                elif tIdx - int(left_sweep * fs) < tdDF.index[0]:
                    padLen = tdDF.index[0] - tIdx + int(left_sweep * fs)
                    padVal = np.pad(
                        thisWaveform.values,
                        ((padLen, 0), (0, 0)), 'edge')
                    spikeWaveforms[idx, :, :] = np.swapaxes(
                        padVal, 0, 1)
                elif tIdx + int(right_sweep * fs) > tdDF.index[-1]:
                    padLen = tIdx + int(right_sweep * fs) - tdDF.index[-1]
                    padVal = np.pad(
                        thisWaveform.values,
                        ((0, padLen), (0, 0)), 'edge')
                    spikeWaveforms[idx, :, :] = np.swapaxes(
                        padVal, 0, 1)
            newSt = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=consolidatedTimes, units='sec', t_stop=spikeTStop,
                waveforms=spikeWaveforms * pq.mV, left_sweep=left_sweep,
                sampling_rate=fs,
                t_start=spikeTStart, **consolidatedAnn, **arrayAnnNames)
            assert (consolidatedTimes.shape[0] == spikeWaveforms.shape[0])
            newSt.unit = thisUnit
            thisUnit.spiketrains = [newSt]
            if createRelationship:
                thisUnit.create_relationship()
            seg.spiketrains.append(newSt)

    if createRelationship:
        for chanIdx in block.channel_indexes:
            chanIdx.create_relationship()
        seg.create_relationship()
        block.create_relationship()

    return block


def extractArtifactTimestamps(
        tdSeg, tdDF, ROIMask,
        fs, gaussWid,
        stimRate=100,
        closeThres=200e-3,
        expectedIdx=None,
        enhanceEdges=True,
        enhanceExpected=True,
        correctThresholdWithAmplitude=True,
        thisAmplitude=None,
        name=None, plotting=None, 
        plotAnomalies=None, anomalyOccured=None,
        minDist=None, theseDetectOpts=None,
        maxSpikesPerGroup=None,
        fixedDelayIdx=None, delayByFreqIdx=None,
        keep_what=None, plotDetection=False, plotKernel=False
        ):
    '''
    tdPow = hf.filterDF(
        tdSeg, fs,
        lowPass=stimRate * 1.75, highPass=stimRate * 0.75,
        highOrder=5, notch=True)
    tdPow = tdPow.abs().sum(axis=1)
    '''
    if plotDetection:
        for colName, tdCol in tdSeg.iteritems():
            plt.plot(
                tdCol.values,
                'b-', label='original signal {}'.format(colName))
    # convolve with a future facing kernel
    if enhanceEdges:
        edgeEnhancer = pd.Series(0, index=tdSeg.index)
        for colName, tdCol in tdSeg.iteritems():
            thisEdgeEnhancer = (
                hf.noisyTriggerCorrection(
                    tdCol, fs, gaussWid, order=2, plotKernel=plotKernel))
            edgeEnhancer += thisEdgeEnhancer
            if plotDetection:
                plt.plot(
                    thisEdgeEnhancer,
                    'k-', label='edge enhancer {}'.format(colName))
        #
        edgeEnhancer = pd.Series(
            MinMaxScaler(feature_range=(1e-2, 1))
            .fit_transform(edgeEnhancer.values.reshape(-1, 1))
            .squeeze(),
            index=tdSeg.index)
    else:
        edgeEnhancer = pd.Series(
            1, index=tdSeg.index)
    if plotDetection:
        plt.plot(
            edgeEnhancer.values,
            'k-', label='final edge enhancer')
    if enhanceExpected:
        assert expectedIdx is not None
        expectedEnhancer = hf.gaussianSupport(
            tdSeg, expectedIdx, gaussWid, fs)
        #
        correctionFactor = edgeEnhancer * expectedEnhancer
        correctionFactor = pd.Series(
            MinMaxScaler(feature_range=(1e-2, 1))
            .fit_transform(correctionFactor.values.reshape(-1, 1))
            .squeeze(),
            index=tdSeg.index)
        if plotDetection:
            plt.plot(expectedEnhancer.values, 'k--', label='expected enhancer')
    tdPow = tdSeg.abs().sum(axis=1)
    detectSignalFull = hf.enhanceNoisyTriggers(
        tdPow, correctionFactor)
    if plotDetection:
        plt.plot(detectSignalFull.values, 'g-', label='detect signal')
    detectSignal = detectSignalFull.loc[ROIMask]
    
    #  threshold proportional to amplitude
    if correctThresholdWithAmplitude:
        amplitudeCorrection = thisAmplitude / 20
        if (name in plotting) or anomalyOccured:
            print(
                'amplitude threshold correction = {}'
                .format(amplitudeCorrection))
        currentThresh = theseDetectOpts['thres'] + amplitudeCorrection
    else:
        currentThresh = theseDetectOpts['thres']
    
    if plotDetection:
        ax = plt.gca()
        ax.axhline(currentThresh, color='r')
        plt.legend()
        plt.show()
    idxLocal = peakutils.indexes(
        detectSignal.values,
        thres=currentThresh,
        min_dist=int(minDist * fs), thres_abs=True,
        keep_what=keep_what)

    if not len(idxLocal):
        if plotAnomalies:
            anomalyOccured = True
        if (name in plotting) or anomalyOccured:
            print(
                'After peakutils.indexes, no peaks found! ' +
                'Using JSON times...')
        peakIdx = expectedIdx
    else:
        peakIdx = detectSignal.index[idxLocal]

    expectedTimestamps = pd.Series(
        tdDF['t'].loc[expectedIdx])
    if maxSpikesPerGroup > 0:
        peakIdx = peakIdx[:maxSpikesPerGroup]
        expectedIdx = expectedIdx[:maxSpikesPerGroup]
        expectedTimestamps = pd.Series(
            tdDF['t'].loc[expectedIdx])

    theseTimestamps = pd.Series(
        tdDF['t'].loc[peakIdx])

    if (name in plotting) or anomalyOccured:
        print('Before checking against expectation')
        print('{}'.format(theseTimestamps))
    
    # check whether the found spikes are off from expected
    assert len(theseTimestamps) > 0, 'nothing found!'
    
    if theseTimestamps.shape[0] >= expectedTimestamps.shape[0]:
        if theseTimestamps.shape[0] > expectedTimestamps.shape[0]:
            if plotAnomalies:
                anomalyOccured = True
                print('More timestamps than expected!')
        closestExpected, _ = hf.closestSeries(
            theseTimestamps,
            expectedTimestamps)
        offsetFromExpected = (
            theseTimestamps.values -
            closestExpected.values)
        offsetTooFarMask = (
            np.abs(offsetFromExpected) > closeThres)
        if offsetTooFarMask.any():
            if plotAnomalies:
                anomalyOccured = True
            #  just use the expected locations
            #  if we failed to find anything better
            newTimestamps = theseTimestamps.copy()
            newTimestamps.loc[offsetTooFarMask] = (
                closestExpected.loc[offsetTooFarMask])
    elif theseTimestamps.shape[0] < expectedTimestamps.shape[0]:
        if plotAnomalies:
            anomalyOccured = True
            print('fewer timestamps than expected!')
        #  fewer detections than expected
        closestFound, _ = hf.closestSeries(
            expectedTimestamps,
            theseTimestamps
            )
        offsetFromExpected = (
            expectedTimestamps.values -
            closestFound.values)
        offsetTooFarMask = (
            np.abs(offsetFromExpected) > closeThres)
        #  start with expected and fill in with
        #  timestamps that were fine
        newTimestamps = expectedTimestamps.copy()
        newTimestamps.loc[~offsetTooFarMask] = (
            closestFound.loc[~offsetTooFarMask])
    '''
    if (
            (offsetTooFarMask.any()) &
            (cyclePeriod > 0) &
            (np.sum(~offsetTooFarMask) >= 2)):
        #  if there are at least 2 good detections
        #  and we can assume they are cycling
        #  interpolate the rest
        dummyT = newTimestamps.reset_index(name='t')
        interpFun = interpolate.interp1d(
            dummyT.loc[~offsetTooFarMask].index,
            dummyT.loc[~offsetTooFarMask, 't'].values,
            kind='linear', fill_value='extrapolate',
            bounds_error=False)
        newTimestamps.loc[offsetTooFarMask] = (
            interpFun(dummyT.loc[offsetTooFarMask].index))
    '''
    if offsetTooFarMask.any():
        closestTimes, _ = hf.closestSeries(
            newTimestamps,
            tdDF['t'],
            )
        originalPeakIdx = theseTimestamps.index
        peakIdx = tdDF['t'].index[tdDF['t'].isin(closestTimes)]
        theseTimestamps = pd.Series(
            tdDF['t'].loc[peakIdx])
    else:
        originalPeakIdx = None
        
    # save closest for diagnostics
    closestExpected, _ = hf.closestSeries(
        theseTimestamps,
        expectedTimestamps)
    offsetFromExpected = (theseTimestamps - closestExpected).values * pq.s

    deadjustedIndices = expectedIdx - fixedDelayIdx - delayByFreqIdx
    loggedTimestamps = pd.Series(
        tdDF['t'].loc[deadjustedIndices])
    closestToLogged, _ = hf.closestSeries(
        theseTimestamps,
        loggedTimestamps)
    offsetFromLogged = (theseTimestamps - closestToLogged).values * pq.s
    return (
        peakIdx, theseTimestamps, offsetFromExpected, offsetFromLogged,
        anomalyOccured, originalPeakIdx, correctionFactor, detectSignalFull,
        currentThresh)


def insDataToBlock(
        td, accel, stimStatusSerial,
        senseInfo, trialFilesStim,
        tdDataCols=None):
    #  ins data to block
    accelColumns = (
        ['accel_' + i for i in ['x', 'y', 'z']]) + (
        ['inertia'])
    accelNixColNames = ['x', 'y', 'z', 'inertia']

    #  assume constant sampleRate
    sampleRate = senseInfo.loc[0, 'sampleRate'] * pq.Hz
    fullX = np.arange(
        0,
        #  td['data']['t'].iloc[0],
        td['data']['t'].iloc[-1] + 1/sampleRate.magnitude,
        1/sampleRate.magnitude
        )
    tdInterp = hf.interpolateDF(
        td['data'], fullX,
        kind='linear', fill_value=(0, 0),
        x='t', columns=tdDataCols)
    accelInterp = hf.interpolateDF(
        accel['data'], fullX,
        kind='linear', fill_value=(0, 0),
        x='t', columns=accelColumns)
    tStart = fullX[0]

    blockName = trialFilesStim['experimentName'] + ' ins data'
    block = Block(name=blockName)
    block.annotate(jsonSessionNames=trialFilesStim['jsonSessionNames'])
    seg = Segment(name='seg0_')
    block.segments.append(seg)

    for idx, colName in enumerate(tdDataCols):
        sigName = 'ins_td{}'.format(senseInfo.loc[idx, 'senseChan'])
        asig = AnalogSignal(
            tdInterp[colName].values*pq.mV,
            name='seg0_' + sigName,
            sampling_rate=sampleRate,
            dtype=np.float32,
            **senseInfo.loc[idx, :].to_dict())
        #
        asig.annotate(td=True, accel=False)
        asig.t_start = tStart*pq.s
        seg.analogsignals.append(asig)

        minIn = int(senseInfo.loc[idx, 'minusInput'])
        plusIn = int(senseInfo.loc[idx, 'plusInput'])
        chanIdx = ChannelIndex(
            name=sigName,
            index=np.array([0]),
            channel_names=np.array(['p{}m{}'.format(minIn, plusIn)]),
            channel_ids=np.array([plusIn])
            )
        block.channel_indexes.append(chanIdx)
        chanIdx.analogsignals.append(asig)
        asig.channel_index = chanIdx

    for idx, colName in enumerate(accelColumns):
        if colName == 'inertia':
            accUnits = (pq.N)
        else:
            accUnits = (pq.m/pq.s**2)
        sigName = 'ins_acc{}'.format(accelNixColNames[idx])
        #  print(sigName)
        asig = AnalogSignal(
            accelInterp[colName].values*accUnits,
            name='seg0_' + sigName,
            sampling_rate=sampleRate,
            dtype=np.float32)
        asig.annotate(td=False, accel=True)
        asig.t_start = tStart*pq.s
        seg.analogsignals.append(asig)
        #
        chanIdx = ChannelIndex(
            name=sigName,
            index=np.array([0]),
            channel_names=np.array([sigName])
            )
        block.channel_indexes.append(chanIdx)
        chanIdx.analogsignals.append(asig)
        asig.channel_index = chanIdx
    #
    stimEvents = ns5.eventDataFrameToEvents(
        stimStatusSerial, idxT='INSTime',
        annCol=['ins_property', 'ins_value']
        )
    seg.events = stimEvents
    block.create_relationship()
    #
    return block


def unpackINSBlock(block, unpackAccel=True):
    tdAsig = block.filter(
        objects=AnalogSignal,
        td=True
        )
    tdDF = ns5.analogSignalsToDataFrame(tdAsig, useChanNames=True)
    if unpackAccel:
        accelAsig = block.filter(
            objects=AnalogSignal,
            accel=True
            )
        accelDF = ns5.analogSignalsToDataFrame(accelAsig, useChanNames=True)
    events = block.filter(
        objects=Event
        )
    stimStSer = ns5.eventsToDataFrame(
        events, idxT='t'
        )
    #  serialize stimStatus
    expandCols = [
        'RateInHz', 'therapyStatus', 'pulseWidth',
        'activeGroup', 'program', 'trialSegment']
    deriveCols = ['amplitudeRound', 'amplitude']
    progAmpNames = rcsa_helpers.progAmpNames
    stimStatus = hf.stimStatusSerialtoLong(
        stimStSer, idxT='t', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    #  add stim info to traces
    columnsToBeAdded = (
        expandCols + deriveCols + progAmpNames)
    infoFromStimStatus = hf.interpolateDF(
        stimStatus, tdDF['t'],
        x='t', columns=columnsToBeAdded, kind='previous')
    infoFromStimStatus['amplitudeIncrease'] = (
        infoFromStimStatus['amplitudeRound'].diff().fillna(0))
    tdDF = pd.concat((
        tdDF,
        infoFromStimStatus.drop(columns='t')),
        axis=1)
    if unpackAccel:
        accelDF = pd.concat((
            accelDF,
            infoFromStimStatus['trialSegment']),
            axis=1)
    else:
        accelDF = False
    return tdDF, accelDF, stimStatus
