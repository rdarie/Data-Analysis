import pandas as pd
import os, pdb
from neo import (
    AnalogSignal, Event, Block,
    Segment, ChannelIndex, SpikeTrain, Unit)
import neo
import elephant as elph
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import matplotlib.pyplot as plt
import numpy as np
#  import sys
#  import pickle
#  from copy import *
import quantities as pq
#  import argparse, linecache
from scipy import stats, signal, ndimage
import peakutils
import seaborn as sns

def preprocINS(
        trialFilesStim,
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

    #  stimStatus = hf.getINSStimLogFromJson(
    #      jsonBaseFolder, jsonSessionNames, logForm='long')
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
        timeSync, degree=1, syncTo='HostUnixTime')
    for trialSegment in pd.unique(td['data']['trialSegment']):
        stimStatus = hf.synchronizeHUTtoINS(
            stimStatus, trialSegment, interpFunHUTtoINS[trialSegment])
        stimStatusSerial = hf.synchronizeHUTtoINS(
            stimStatusSerial, trialSegment, interpFunHUTtoINS[trialSegment])

    #  pdb.set_trace()
    #  sync Host PC Unix time to NSP
    HUTtoINSPlotting = True
    if HUTtoINSPlotting and plottingFigures:
        plottingColumns = deriveCols + expandCols + progAmpNames
        plotStimStatusList = []
        for trialSegment in pd.unique(td['data']['trialSegment']):
            stimGroupMask = stimStatus['trialSegment'] == trialSegment
            stimGroup = stimStatus.loc[stimGroupMask, :]
            thisHUTtoINS = interpFunHUTtoINS[trialSegment]

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

    tStartTD = 200
    tStopTD = 300

    if HUTtoINSPlotting and plottingFigures:
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
        
    # export function
    orcaExpPath = os.path.join(
        trialFilesStim['folderPath'],
        trialFilesStim['experimentName']
    )
    if not os.path.exists(orcaExpPath):
        os.makedirs(orcaExpPath)

    insDataFilename = os.path.join(
        orcaExpPath, trialFilesStim['ns5FileName'] + '_ins.nix')

    writer = neo.io.NixIO(filename=insDataFilename, mode='ow')
    writer.write_block(block)
    writer.close()
    return block


def getINSStimOnset(
        block, elecConfiguration,
        stimIti=0, minDist=0, minDur=0, thres=.5,
        gaussWid=500e-3, gaussKerWid=150e-3,
        timeInterpFunINStoNSP=None,
        maxSpikesPerGroup=None,
        fixedDelay=15e-3,
        stimDetectOpts=None,
        plotting=[]):

    segIdx = 0
    seg = block.segments[segIdx]
    defaultOptsDict = {
        'detectChannels': ['ins_td0', 'ins_td2'],
        'thres': thres,
        'keep_max': False}

    if stimDetectOpts is None:
        stimDetectOpts = {
            grpIdx: {progIdx: defaultOptsDict for progIdx in range(4)}
            for grpIdx in range(4)}
    
    #  allocate channels for each physical contact
    #  for mdtIdx in range(17):
    #      mdtChanName = 'ins_ch{}'.format(mdtIdx)
    #      chanIdx = ChannelIndex(
    #          name=mdtChanName,
    #          index=[mdtIdx])
    #      block.channel_indexes.append(chanIdx)
    
    #  allocate units for each group/program pair
    for groupIdx in range(4):
        for progIdx in range(4):
            electrodeCombo = 'g{:d}p{:d}'.format(
                groupIdx, progIdx)
            mdtIdx = int(4 * groupIdx + progIdx)
            chanIdx = ChannelIndex(
                name=electrodeCombo + '_ch',
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
                theseDetOpts = stimDetectOpts[
                    groupIdx][progIdx]
            except Exception:
                theseDetOpts = defaultOptsDict
            for key, value in theseDetOpts.items():
                thisUnit.annotations.update({key: value})
            chanIdx.units.append(thisUnit)
            #  for mdtIdx in thisElecConfig['cathodes']:
            #      chanIdx = block.filter(
            #          objects=ChannelIndex,
            #          index=[mdtIdx])[0]
            #      chanIdx.units.append(thisUnit)
            #  for mdtIdx in thisElecConfig['anodes']:
            #      chanIdx = block.filter(
            #          objects=ChannelIndex,
            #          index=[mdtIdx])[0]
            #      chanIdx.units.append(thisUnit)
    tdDF, accelDF, stimStatus = unpackINSBlock(block)
    
    spikeTStop = tdDF['t'].iloc[-1]
    spikeTStart = tdDF['t'].iloc[0]
    fs = seg.analogsignals[0].sampling_rate
    for name, group in tdDF.groupby('amplitudeRound'):
        activeProgram = int(group['program'].value_counts().idxmax())
        activeGroup = int(group['activeGroup'].value_counts().idxmax())
        thisTrialSegment = int(group['trialSegment'].value_counts().idxmax())
        activeState = bool(group['therapyStatus'].value_counts().idxmax())
        stimRate = group['RateInHz'].value_counts().idxmax()
        #  changes to stim wait for a full period
        delayByFreq = (1.5 / stimRate)
        delayByFreqIdx = int(fs * delayByFreq)
        
        ampColName = 'program{}_amplitude'.format(activeProgram)
        thisAmplitude = group[ampColName].max()
        
        #  pad with 150 msec to capture first pulse
        tStart = max(0, group['t'].iloc[0] - 150e-3)
        tStop = min(group['t'].iloc[-1], tdDF['t'].iloc[-1])
        #  pad with 100 msec to *avoid* first pulse
        #  tStart = min(tStop, group['t'].iloc[0] + 0.1)
        plotMaskTD = (tdDF['t'] > tStart) & (tdDF['t'] < tStop)
        
        if (tStop - tStart) < minDur:
            #  print('tStart ={}'.format(tStart))
            #  print('tStop ={}'.format(tStop))
            print('group {} (tStop - tStart) < minDur'.format(name))
            continue
        
        if not activeState:
            print('group {} Therapy not active!'.format(name))
            continue
                
        theseDetectOpts = stimDetectOpts[activeGroup][activeProgram]
        #  pdb.set_trace()
        tdSeg = (tdDF.loc[plotMaskTD, theseDetectOpts['detectChannels']])
        tdPow = hf.filterDF(
            tdSeg, fs,
            lowPass=stimRate+10, highPass=stimRate - 10,
            highOrder=5, notch=True)
        tdPow = tdPow.abs().sum(axis=1)
        #  tdPow = (tdDF.loc[
        #      plotMaskTD, theseDetectOpts['detectChannels']] ** 2).sum(axis=1)
        
        # convolve with a future facing kernel
        fancyCorrection = True
        if fancyCorrection:
            correctionFactor = hf.noisyTriggerCorrection(tdPow, fs)
        else:
            correctionFactor = pd.Series(
                tdPow**0,
                index=tdPow.index)

        # use the HUT derived stim onset to favor detection
        hutPeakIdx = hf.getTriggers(
            tdDF.loc[plotMaskTD, ampColName], thres=5,
            iti=minDist, keep_max=theseDetectOpts['keep_max'])
        
        onsetIndices = hutPeakIdx
        if len(onsetIndices):
            #  assume a fixed delay of 5 msec between onset and stim
            fixedDelayIdx = int(fixedDelay * fs)
            onsetIndices = (
                onsetIndices + fixedDelayIdx +
                delayByFreqIdx)
                
        onsetTimestamps = pd.Series(
            tdDF['t'].loc[onsetIndices].values,
            index=tdDF['t'].loc[onsetIndices].index)
        
        # if we know cycle value, use it to predict onsets
        if stimIti == 0:
            thisElecConfig = elecConfiguration[activeGroup][activeProgram]
            if thisElecConfig['cyclingEnabled']:
                stimIti = (
                    thisElecConfig['cycleOffTime']['time'] +
                    thisElecConfig['cycleOnTime']['time']) / 10
        
        if len(onsetIndices) == 1 and stimIti > 0:
            segmentEnding = tdDF.loc[plotMaskTD, 't'].index[-1]
            onsetIndices = np.arange(
                onsetIndices[0], segmentEnding,
                int(fs*(stimIti+delayByFreq)))
            onsetTimestamps = pd.Series(
                tdDF['t'].loc[onsetIndices].values,
                index=tdDF['t'].loc[onsetIndices].index)
      
        if len(onsetTimestamps):
            gaussWidIdx = min(int(gaussWid * fs), len(tdPow.index)-1)
            gaussKerIdx = int(gaussKerWid * fs)
            gaussKern = signal.gaussian(
                gaussWidIdx, gaussKerIdx)
            support = pd.Series(0, index=tdPow.index)
            support.loc[onsetIndices] = 1
            support.iloc[:] = np.convolve(
                support.values,
                gaussKern, mode='same'
                )
            support = support + 1
            correctionFactor = correctionFactor * support

        stimDetectSignal = hf.enhanceNoisyTriggers(tdPow, correctionFactor)
        if thisAmplitude == 0:
            peakIdx = np.array([])
        else:
            peakIdx = peakutils.indexes(
                stimDetectSignal.values, thres=theseDetectOpts['thres'],
                min_dist=int(minDist * fs), thres_abs=True,
                keep_max=theseDetectOpts['keep_max'])
            peakIdx = tdPow.index[peakIdx]
            if name in plotting:
                print('After peakutils.indexes, before check for amplitude')
                print('{}'.format(peakIdx))
            # check for amplitude
            keepMask = (tdDF.loc[peakIdx, ampColName] == thisAmplitude).values
            peakIdx = peakIdx[keepMask]
        
        if name in plotting:
            print('Before if maxSpikesPerGroup is not None')
            print('{}'.format(peakIdx))

        if maxSpikesPerGroup > 0:
            peakIdx = peakIdx[:maxSpikesPerGroup]

        theseTimestamps = pd.Series(
            tdDF['t'].loc[peakIdx].values,
            index=tdDF['t'].loc[peakIdx].index)

        if name in plotting:
            print('Before if stimIti > 0')
            print('{}'.format(peakIdx))
        
        if stimIti > 0:
            stimBackDiff = theseTimestamps.diff().fillna(method='bfill')
            stimFwdDiff = (-1) * theseTimestamps.diff(-1).fillna(method='ffill')
            stimDiff = (stimBackDiff + stimFwdDiff) / 2
            offBy = (stimDiff - stimIti - delayByFreq).abs()
            if (name in plotting) and (len(offBy) > 1):
                #      ax = sns.distplot(stimDiff)
                #      plt.plot()
                #  else:
                print('stim iti off by:')
                print('{}'.format(offBy))
            keepMask = offBy < 0.3
            theseTimestamps = theseTimestamps.loc[keepMask]
            peakIdx = peakIdx[keepMask]
        
        if timeInterpFunINStoNSP is not None:
            # synchronize stim timestamps with INS timestamps
            theseTimestamps.iloc[:] = timeInterpFunINStoNSP[thisTrialSegment](
                theseTimestamps.values)
        
        theseTimestamps = theseTimestamps.values * pq.s
        electrodeCombo = 'g{:d}p{:d}'.format(
            activeGroup, activeProgram)
        
        if len(theseTimestamps):
            thisUnit = block.filter(
                objects=Unit,
                name=electrodeCombo
                )[0]

            ampList = theseTimestamps ** 0 * 100 * thisAmplitude * pq.uA
            rateList = theseTimestamps ** 0 * stimRate * pq.Hz
            tSegList = theseTimestamps ** 0 * thisTrialSegment
            arrayAnn = {
                'amplitudes': ampList, 'rates': rateList,
                'trialSegments': tSegList}
        
            st = SpikeTrain(
                times=theseTimestamps, t_stop=spikeTStop,
                t_start=spikeTStart,
                name=thisUnit.name,
                array_annotations=arrayAnn,
                **arrayAnn)
            #  st.annotate(amplitude=thisAmplitude * 100 * pq.uA)
            #  st.annotate(rate=stimRate * pq.Hz)
            thisUnit.spiketrains.append(st)
        
        if name in plotting:
            #  pdb.set_trace()
            print('About to Plot')
            print('{}'.format(peakIdx))

            fig, ax = plt.subplots(3, 1, sharex=True)
            if accelDF is not None:
                plotMaskAccel = (accelDF['t'] > tStart) & (
                    accelDF['t'] < tStop)
                ax[0].plot(
                    accelDF['t'].loc[plotMaskAccel],
                    stats.zscore(
                        accelDF.loc[plotMaskAccel, 'ins_accinertia']),
                    '-', label='inertia')
            for channelName in theseDetectOpts['detectChannels']:
                ax[0].plot(
                    tdDF['t'].loc[plotMaskTD],
                    stats.zscore(tdDF.loc[plotMaskTD, channelName]),
                    '-', label=channelName)
            ax[0].legend()
            ax[0].set_title('INS Accel and TD')
            
            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                correctionFactor,
                '-', label='correctionFactor')
                
            #ax[1].plot(
            #    tdDF['t'].loc[plotMaskTD],
            #    sobelFiltered,
            #    '-', label='sobelFiltered')
            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                stimDetectSignal,
                '-', label='stimDetectSignal')
            ax[1].plot(
                tdDF['t'].loc[peakIdx],
                stimDetectSignal.loc[peakIdx],
                'o', label='stimOnset')
            ax[1].plot(
                tdDF['t'].loc[onsetIndices],
                stimDetectSignal.loc[onsetIndices],
                '*', label='stimOnset')
                                  
            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                stimDetectSignal ** 0 * theseDetectOpts['thres'],
                'r-', label='detection Threshold')
            ax[1].legend()
            ax[1].set_title('INS TD Measurements')
            progAmpNames = rcsa_helpers.progAmpNames
            for columnName in progAmpNames:
                ax[2].plot(
                    tdDF.loc[plotMaskTD, 't'],
                    tdDF.loc[plotMaskTD, columnName],
                    '-', label=columnName, lw = 2.5)

            statusAx = ax[2].twinx()
            statusAx.plot(
                tdDF.loc[plotMaskTD, 't'],
                tdDF.loc[plotMaskTD, 'therapyStatus'],
                '--', label='therapyStatus', lw = 1.5)
            
            statusAx.plot(
                tdDF.loc[plotMaskTD, 't'],
                tdDF.loc[plotMaskTD, 'amplitudeIncrease'],
                'c--', label='amplitudeIncrease', lw = 1.5)
            
            ax[2].legend(loc='upper left')    
            statusAx.legend(loc='upper right')
            ax[2].set_ylabel('Stim Amplitude (mA)')
            ax[2].set_xlabel('NSP Time (sec)')
            plt.suptitle('Stim State')
            plt.show()
    
    createRelationship = True
    for thisUnit in block.list_units:
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
                'amplitudes': np.array([]),
                'rates': np.array([]),
                'trialSegments': np.array([])
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
            #  pdb.set_trace()
            newSt = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=consolidatedTimes, units='sec', t_stop=spikeTStop,
                t_start=spikeTStart, **consolidatedAnn, **arrayAnnNames)
            
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
        #td['data']['t'].iloc[0],
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
    seg = Segment()
    block.segments.append(seg)

    for idx, colName in enumerate(tdDataCols):
        sigName = 'ins_td{}'.format(senseInfo.loc[idx, 'senseChan'])
        asig = AnalogSignal(
            tdInterp[colName].values*pq.mV,
            name=sigName,
            sampling_rate=sampleRate,
            dtype=np.float32,
            **senseInfo.loc[idx, :].to_dict())
        #  pdb.set_trace()
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
            name=sigName,
            sampling_rate=sampleRate,
            dtype=np.float32)
        asig.annotate(td=False, accel=True)
        asig.t_start = tStart*pq.s
        seg.analogsignals.append(asig)
    
        chanIdx = ChannelIndex(
            name=sigName,
            index=np.array([0]),
            channel_names=np.array([sigName])
            )
        block.channel_indexes.append(chanIdx)
        chanIdx.analogsignals.append(asig)
        asig.channel_index = chanIdx

    stimEvents = ns5.eventDataFrameToEvents(
        stimStatusSerial, idxT='INSTime',
        annCol=['ins_property', 'ins_value']
        )
    #  pdb.set_trace()
    seg.events = stimEvents
    block.create_relationship()
        
    return block


def unpackINSBlock(block):
    tdAsig = block.filter(
        objects=AnalogSignal,
        td=True
        )
    tdDF = ns5.analogSignalsToDataFrame(tdAsig)
    accelAsig = block.filter(
        objects=AnalogSignal,
        accel=True
        )
    accelDF = ns5.analogSignalsToDataFrame(accelAsig)

    events = block.filter(
        objects=Event
        )

    stimStSer = ns5.eventsToDataFrame(
        events, idxT='t'
        )
    #  pdb.set_trace()
    #  serialize stimStatus
    expandCols = [
        'RateInHz', 'therapyStatus',
        'activeGroup', 'program', 'trialSegment']
    deriveCols = ['amplitudeRound', 'amplitude']
    progAmpNames = rcsa_helpers.progAmpNames

    stimStatus = hf.stimStatusSerialtoLong(
        stimStSer, idxT='t', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    #  add stim info to traces
    #  pdb.set_trace()
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
    accelDF = pd.concat((
        accelDF,
        infoFromStimStatus['trialSegment']),
        axis=1)
    return tdDF, accelDF, stimStatus