import pandas as pd
import os, pdb
from neo import (
    AnalogSignal, Event, Block,
    Segment, ChannelIndex, SpikeTrain, Unit)
import neo
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

def preprocINS(trialFilesStim,
        plottingFigures=False,
        plotBlocking=True):
    jsonBaseFolder = trialFilesStim['folderPath']
    jsonSessionNames = trialFilesStim['jsonSessionNames']

    td = hf.getINSTDFromJson(
        jsonBaseFolder, jsonSessionNames, getInterpolated=True,
        forceRecalc=True)

    elecStatus, elecType, elecConfiguration, senseInfo = (
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
        forceRecalc=True)

    timeSync = hf.getINSTimeSyncFromJson(
        jsonBaseFolder, jsonSessionNames,
        forceRecalc=True)

    stimStatus = hf.getINSStimLogFromJson(
        jsonBaseFolder, jsonSessionNames, logForm='long')
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

    interpFunHUTtoINS = hf.getHUTtoINSSyncFun(
        timeSync, degree=1, syncTo='HostUnixTime')
    for trialSegment in pd.unique(td['data']['trialSegment']):
        stimStatus = hf.synchronizeHUTtoINS(
            stimStatus, trialSegment, interpFunHUTtoINS[trialSegment])
        stimStatusSerial = hf.synchronizeHUTtoINS(
            stimStatusSerial, trialSegment, interpFunHUTtoINS[trialSegment])

    #  sync Host PC Unix time to NSP
    HUTtoINSPlotting = True
    if HUTtoINSPlotting:
        progAmpNames = rcsa_helpers.progAmpNames
        
        plottingColumns = [
            'frequency', 'therapyStatus', 'trialSegment',
            'amplitudeChange', 'amplitudeRound'] +\
            progAmpNames
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
            temp['amplitudeChange'] = (temp[
                'amplitudeRound'].diff().fillna(0) > 0).astype(float)

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

    #  ins data to block

    accelColumns = (
        ['accel_' + i for i in ['x', 'y', 'z']]) + (
        ['inertia'])
    accelNixColNames = ['x', 'y', 'z', 'inertia']
    accelInterp = hf.interpolateDF(
        accel['data'], td['data']['t'],
        kind='linear', fill_value=(0, 0),
        x='t', columns=accelColumns)

    tStart = td['data']['t'].iloc[0]
    blockName = trialFilesStim['experimentName'] + ' ins data'
    block = Block(name=blockName)
    block.annotate(jsonSessionNames=trialFilesStim['jsonSessionNames'])
    seg = Segment()
    block.segments.append(seg)

    #  assume constant sampleRate
    sampleRate = senseInfo.loc[0, 'sampleRate'] * pq.Hz
    for idx, colName in enumerate(tdDataCols):
        sigName = 'ins_td{}'.format(senseInfo.loc[idx, 'senseChan'])
        asig = AnalogSignal(
            td['data'][colName].values*pq.mV,
            name=sigName,
            sampling_rate=sampleRate,
            dtype=np.float32,
            **senseInfo.loc[idx, :].to_dict())
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
        sigName = 'ins_acc{}'.format(accelNixColNames[idx])
        #  print(sigName)
        asig = AnalogSignal(
            accelInterp[colName].values*pq.mV,
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
        annCol=['property', 'group', 'program', 'value', 'trialSegment']
        )
    #  pdb.set_trace()
    seg.events = stimEvents
    '''
    seg.create_relationship()
    for chanIdx in block.channel_indexes:
        chanIdx.create_relationship()
    block.create_relationship()
    '''
    #  stim detection
    orcaExpPath = os.path.join(
        trialFilesStim['folderPath'],
        trialFilesStim['experimentName']
    )
    if not os.path.exists(orcaExpPath):
        os.makedirs(orcaExpPath)

    block = getINSStimOnset(
        block, elecConfiguration,
        **trialFilesStim['getINSkwargs'])
        
    # export function
    insDataFilename = os.path.join(
        orcaExpPath, trialFilesStim['ns5FileName'] + '_ins.nix')

    writer = neo.io.NixIO(filename=insDataFilename)
    writer.write_block(block)
    writer.close()
    return block


def getINSStimOnset(
        block, elecConfiguration,
        fs=500, stimIti=0, minDist=0, minDur=0, thres=.5,
        timeInterpFunINStoNSP=None,
        maxSpikesPerGroup=None,
        stimFreq=None, fixedDelay=15e-3,
        stimDetectOpts=None,
        plotting=[]):

    seg = block.segments[0]
    defaultOptsDict = {
        'detectChannels': ['ins_td0', 'ins_td2'],
        'thres': thres,
        'keep_max': False}

    if stimDetectOpts is None:
        stimDetectOpts = {
            grpIdx: {progIdx: defaultOptsDict for progIdx in range(4)}
            for grpIdx in range(4)}
    #  allocate channels for each physical contact
    for mdtIdx in range(17):
        mdtChanName = 'ins_ch{}'.format(mdtIdx)
        chanIdx = ChannelIndex(
            name=mdtChanName,
            index=[mdtIdx])
        block.channel_indexes.append(chanIdx)
    #  allocate units for each combination of group and program
    
    tdDF, accelDF, stimStatus = unpackINSNeoBlock(block)
    t_stop = tdDF['t'].iloc[-1]
    for name, group in tdDF.groupby('amplitudeRound'):
        activeProgram = int(group['program'].value_counts().idxmax())
        activeGroup = int(group['group'].value_counts().idxmax())
        ampColName = 'program{}_amplitude'.format(activeProgram)
        thisAmplitude = group[ampColName].max()
        thisTrialSegment = group['trialSegment'].value_counts().idxmax()
        
        #  pad with 150 msec to capture first pulse
        tStart = max(0, group['t'].iloc[0] - 150e-3)
        tStop = min(group['t'].iloc[-1], tdDF['t'].iloc[-1])
        
        #  pad with 100 msec to *avoid* first pulse
        #  tStart = min(tStop, group['t'].iloc[0] + 0.1)
        
        if (tStop - tStart) < minDur:
            print('tStart ={}'.format(tStart))
            print('tStop ={}'.format(tStop))
            print('(tStop - tStart) < minDur')
            continue

        plotMaskTD = (tdDF['t'] > tStart) & (tdDF['t'] < tStop)
        thisTherapyStatus = tdDF.loc[plotMaskTD, 'therapyStatus']
        activeState = bool(thisTherapyStatus.value_counts().idxmax())

        if not activeState:
            print('Therapy not active!')
            continue
                
        theseDetectOpts = stimDetectOpts[activeGroup][activeProgram]
        
        tdPow = (tdDF.loc[
            plotMaskTD, theseDetectOpts['detectChannels']] ** 2).sum(axis = 1)
        
        # convolve with a rectangular kernel
        # then shift forward and backward to get forward and backward sum
        kernDur = 0.2
        kernNSamp = int(kernDur * fs)
        if kernNSamp > len(tdPow):
            kernNSamp = round(len(tdPow) - 2)
        kern = np.concatenate((
            np.linspace(0, 1, round(kernNSamp/2)),
            np.array([0]),
            np.linspace(-1, 0, round(kernNSamp/2))
            ))
        
        # use the HUT derived stim onset to favor detection
        hutPeakIdx = hf.getTriggers(
            tdDF.loc[plotMaskTD, ampColName], thres=5,
            iti=minDist, keep_max=theseDetectOpts['keep_max'])
        
        onsetIndices = hutPeakIdx
        if len(onsetIndices):
            #  assume a fixed delay of 5 msec between onset and stim
            fixedDelayIdx = int(fixedDelay * fs)
            onsetIndices = onsetIndices + fixedDelayIdx
            if stimFreq is not None:
                onsetIndices = onsetIndices + int(fs / stimFreq)
                
        onsetTimestamps = pd.Series(
            tdDF['t'].loc[onsetIndices].values,
            index = tdDF['t'].loc[onsetIndices].index)
        # if we know cycle value, use it to predict onsets
        if len(onsetIndices) == 1 and stimIti > 0:
            segmentEnding = tdDF.loc[plotMaskTD, 't']
            onsetIndices = np.arange(
                onsetIndices[0], segmentEnding.index[-1], fs*stimIti)
            onsetTimestamps = pd.Series(
                tdDF['t'].loc[onsetIndices].values,
                index = tdDF['t'].loc[onsetIndices].index)

        correctionFactor = pd.Series(
            np.convolve(kern, tdPow, mode='same'),
            index = tdPow.index)
        
        correctionFactor = correctionFactor - correctionFactor.min()
        correctionFactor = (correctionFactor / correctionFactor.max()) + 1
        
        if len(onsetTimestamps):
            gaussDur = 0.15
            gaussNSamp = int(gaussDur * fs)
            gaussKern = signal.gaussian(
                3 * gaussNSamp, gaussNSamp)
            #  pdb.set_trace()
            support = pd.Series(0, index = tdPow.index)
            support.loc[onsetIndices] = 1
            support.iloc[:] = np.convolve(
                support.values,
                gaussKern, mode = 'same'
                )
            support = support + 1
            #  correctionFactor = support
            #  pdb.set_trace()
            correctionFactor = correctionFactor * support
        sobelFiltered = pd.Series(
            ndimage.sobel(tdPow, mode='reflect'),
            index=tdPow.index)

        stimDetectSignal = sobelFiltered * correctionFactor
        stimDetectSignal = stimDetectSignal.fillna(0)
        stimDetectSignal.iloc[:] = stats.zscore(sobelFiltered)
        stimDetectSignal = stimDetectSignal.abs()

        if thisAmplitude == 0:
            #  pdb.set_trace()
            #  peakIdx = hf.getTriggers(
            #      tdDF.loc[plotMaskTD, 'amplitudeChange'].astype(np.float), thres=5,
            #      iti=minDist, keep_max = theseDetectOpts['keep_max'])
            peakIdx = np.array([])
        else:
            peakIdx = peakutils.indexes(
                stimDetectSignal.values, thres=theseDetectOpts['thres'],
                min_dist=int(minDist * fs), thres_abs=True,
                keep_max = theseDetectOpts['keep_max'])
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

        if maxSpikesPerGroup is not None:
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
            offBy = (stimDiff - stimIti).abs()
            if name in plotting:
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
        electrodeCombo = 'gr{}pr{}_{}'.format(
            activeGroup, activeProgram, int(thisAmplitude * 100))
        
        if len(theseTimestamps):
            thisUnitList = block.filter(
                objects=Unit,
                name=electrodeCombo
                )
            if len(thisUnitList):
                #  unit already exists
                thisUnit = thisUnitList[0]
            else:
                #  allocate unit
                thisElecConfig = elecConfiguration[
                    activeGroup][activeProgram]
            
                thisUnit = Unit(name=electrodeCombo)
                thisUnit.annotate(cathodes=thisElecConfig['cathodes'])
                thisUnit.annotate(anodes=thisElecConfig['anodes'])
                thisUnit.annotate(amplitude=thisAmplitude * 100 * pq.uA)

                try:
                    theseDetOpts = stimDetectOpts[
                        activeGroup][activeProgram]
                except Exception:
                    theseDetOpts = defaultOptsDict

                for key, value in theseDetOpts.items():
                    thisUnit.annotations.update({key: value})
            
                for mdtIdx in thisElecConfig['cathodes']:
                    chanIdx = block.filter(
                        objects=ChannelIndex,
                        index=[mdtIdx])[0]
                    chanIdx.units.append(thisUnit)
                for mdtIdx in thisElecConfig['anodes']:
                    chanIdx = block.filter(
                        objects=ChannelIndex,
                        index=[mdtIdx])[0]
                    chanIdx.units.append(thisUnit)
            ampList = theseTimestamps ** 0 * 100 * thisAmplitude * pq.mA
            arrayAnn = {'stimAmplitude': ampList}
        
            st = SpikeTrain(
                times=theseTimestamps, t_stop=t_stop,
                name=thisUnit.name,
                array_annotations=arrayAnn)
            st.annotate(amplitude=thisAmplitude * 100 * pq.uA)
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
                
            ax[1].plot(
                tdDF['t'].loc[plotMaskTD],
                sobelFiltered,
                '-', label='sobelFiltered')
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
            '''
            statusAx.plot(
                tdDF.loc[plotMaskTD, 't'],
                tdDF.loc[plotMaskTD, 'amplitudeChange'],
                'c--', label='amplitudeChange', lw = 1.5)
            '''
            ax[2].legend(loc = 'upper left')    
            statusAx.legend(loc = 'upper right')
            ax[2].set_ylabel('Stim Amplitude (mA)')
            ax[2].set_xlabel('NSP Time (sec)')
            plt.suptitle('Stim State')
            plt.show()
    
    createRelationship = False
    for thisUnit in block.list_units:
        if not len(thisUnit.spiketrains):
            st = SpikeTrain(
                name=thisUnit.name,
                times=[], units='sec',
                t_stop=t_stop)
            thisUnit.spiketrains.append(st)
            seg.spiketrains.append(st)
        else:
            #  consolidate spiketrains
            consolidatedTimes = np.array([])
            for idx, st in enumerate(thisUnit.spiketrains):
                consolidatedTimes = np.concatenate((
                    consolidatedTimes,
                    st.times.magnitude
                ))
            newSt = SpikeTrain(
                name=thisUnit.name,
                times=consolidatedTimes, units='sec',
                t_stop=t_stop)
            thisUnit.spiketrains = [newSt]
            seg.spiketrains.append(newSt)

        if createRelationship:
            thisUnit.create_relationship()

    if createRelationship:
        for chanIdx in block.channel_indexes:
            chanIdx.create_relationship()
        seg.create_relationship()
        block.create_relationship()
    
    return block


def unpackINSNeoBlock(block):
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
    expandCols = ['RateInHz', 'therapyStatus']
    deriveCols = ['amplitudeRound']
    progAmpNames = rcsa_helpers.progAmpNames

    stimStatus = hf.stimStatusSerialtoLong(
        stimStSer, idxT='t', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    #  add stim info to traces
    #  pdb.set_trace()
    columnsToBeAdded = (
        expandCols + deriveCols + progAmpNames +
        ['group', 'program', 'trialSegment'])
    infoFromStimStatus = hf.interpolateDF(
        stimStatus, tdDF['t'],
        x='t', columns=columnsToBeAdded, kind='previous')
    tdDF = pd.concat((
        tdDF,
        infoFromStimStatus.drop(columns='t')),
        axis=1)
    return tdDF, accelDF, stimStatus

'''
parser = argparse.ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--stepLen', default = 0.05, type = float)
parser.add_argument('--winLen', default = 0.1, type = float)

args = parser.parse_args()

argFile = args.file
argFile = 'W:/ENG_Neuromotion_Shared/group/BSI/Shepherd/Recordings/201711161108-Shepherd-Treadmill/ORCA Logs/Session1510849056349/DeviceNPC700199H/RawDataTD.json'
fileDir = '/'.join(argFile.split('/')[:-1])
fileName = argFile.split('/')[-1]
fileType = fileName.split('.')[-1]

stepLen_s = args.stepLen
stepLen_s = 0.05
winLen_s = args.winLen
winLen_s = 0.1

elecID = [4, 5, 6, 7]

elecLabel = ['Mux4', 'Mux5', 'Mux6', 'Mux7']

# which channel to plot
whichChan = 4

# either txt file from David's code, or ORCA json object
if fileType == 'txt':
    data = {'raw' : pd.read_table(argFile, skiprows = 2, header = 0)}
    data.update({'elec_ids' : elecID})
    data.update(
        {
            'ExtendedHeaderIndices' : [0, 1, 2, 3],
            'extended_headers' : [{'Units' : elecUnits, 'ElectrodeLabel' : elecLabel[i]} for i in [0, 1, 2, 3]],
        }
        )

    data.update({'data' : data['raw'].loc[:, [' SenseChannel1 ',
        ' SenseChannel2 ', ' SenseChannel3 ', ' SenseChannel4 ']]})
    data['data'].columns = [0,1,2,3]
    data.update({'t' : data['raw'].loc[:,  ' Timestamp '].values - 636467827693977000})

    sR = linecache.getline(argFile, 2)
    sR = [int(s) for s in sR.split() if s.isdigit()][0]
    data.update({'samp_per_s' : sR})
    data.update({'start_time_s' : 0})

elif fileType == 'json':
    RawDataTD = pd.read_json(fileDir + '/RawDataTD.json')
    data = {'raw' : RawDataTD}

    RawDataFFT = pd.read_json(fileDir + '/RawDataFFT.json')
    RawDataPower = pd.read_json(fileDir + '/RawDataPower.json')
    RawDataAccel = pd.read_json(fileDir + '/RawDataAccel.json')
    AdaptiveLog = pd.read_json(fileDir + '/AdaptiveLog.json')

    logs = [RawDataTD, RawDataFFT, RawDataPower, RawDataAccel]
    logDataFields = ['TimeDomainData', 'FftData', 'PowerDomainData', 'AccelData']
    firstGoodTimeAll = [np.nan for i in logs] + [np.nan]

    for idx, log in enumerate(logs):
        if not log[logDataFields[idx]].empty:
            for packet in log[logDataFields[idx]][0]:
                if packet['PacketGenTime'] > 0:
                    firstGoodTimeAll[idx] = packet['PacketGenTime']
                    break

    if not AdaptiveLog.empty:
        for log in AdaptiveLog:
            if log['AdaptiveUpdate']['PacketGenTime'] > 0:
                firstGoodTimeAll[4] = log['AdaptiveUpdate']['PacketGenTime']
                break

    firstGoodTime = min(firstGoodTimeAll)
    # This will be used as a reference for all plots when plotting based off Packet Gen
    # Time. It is the end of the first streamed packet. The first received sample is
    # considered the zero reference.
    TimeSync = pd.read_json(fileDir + '/TimeSync.json')
    logs = logs + [TimeSync]
    logDataFields = logDataFields + ['TimeSyncData']
    masterTickArray = np.asarray([np.nan for i in logs] + [np.nan])
    masterTimeStampArray = np.asarray([np.nan for i in logs] + [np.nan])

    for idx, log in enumerate(logs):
        if not log[logDataFields[idx]].empty:
            masterTickArray[idx] = log[logDataFields[idx]][0][0]['Header']['systemTick']
            masterTimeStampArray[idx] = log[logDataFields[idx]][0][0]['Header']['timestamp']['seconds']

    if not AdaptiveLog.empty:
        masterTickArray[idx] = AdaptiveLog[0]['Header']['systemTick']
        masterTimeStampArray[idx] = AdaptiveLog[0]['Header']['timestamp']['seconds']

    masterTimeStamp = min(masterTimeStampArray)
    I = [i for i, x in enumerate(masterTimeStampArray) if x == masterTimeStamp]
    masterTick = min(masterTickArray[I])

    # This (masterTimeStamp and masterTick) will be used as a reference for all plots
    # when plotting based off System Tick. It is the end of the first streamed packet.
    # The first received sample is considered the zero reference.

    rolloverseconds = 6.5535 # System Tick seconds before roll over

    # get first packet to unpack some information about packets to come
    packets = iter(data['raw']['TimeDomainData'][0])
    packet = next(packets)

    elecUnits = packet['Units']
    numberOfChannels = len(packet['ChannelSamples'])
    assert numberOfChannels == len(elecLabel)
    numberOfEvokedMarkers = len(packet['EvokedMarker'])
    sampleRate = sampleRateDict[packet['SampleRate']]

    #get first channel data
    channels = iter(packet['ChannelSamples'])
    channel = next(channels)

    if I == [0]:
        endtime1 = (len(channel['Value']) - 1) / sampleRate
        # Sets the first packet end time in seconds based off samples and sample rate.
        # This is used as a reference for FFT, Power, and Adaptive data because these are
        # calculated after the TD data has been acquired. Accel data generates its own
        # first packet end time in seconds.
    else:
        endtime1 = 0
    #preallocate data containers
    channelData = pd.DataFrame(index = [], columns = elecLabel)
    packetSize = []
    evokedMarker = pd.DataFrame(index = [], columns = elecLabel)
    evokedIndex = pd.DataFrame(index = [], columns = elecLabel)
    tvec = pd.Series(index = [])
    missedPacketGaps = 0 #Initializes missed packet count

    seconds = 0 #Initializes seconds addition due to looping

    # Determine corrective time offset-----------------------------------------------
    # if only plotting time domain packets, these will always be zero, but I'm
    # leaving this in here for future expansion
    tickRef = packet['Header']['systemTick'] - masterTick
    timeStampRef = packet['Header']['timestamp']['seconds'] - masterTimeStamp

    if timeStampRef > 6:
        seconds = seconds + timeStampRef # if time stamp differs by 7 or more seconds make correction

    elif tickRef < 0 and timeStampRef > 0:
        seconds = seconds + rolloverseconds # adds initial loop time if needed


    #--------------------------------------------------------------------------------
    loopCount = 0 #Initializes loop count
    loopTimeStamp = [] #Initializes loop time stamp index

    # plot based off system tick if True, else plot based off Packet Gen Time:
    timing = True
    #linearly space packet data points if True, else space packet data points based off sample rate:
    spacing = False

    if ('stp', 'var') in locals():
        del stp

    packets = data['raw']['TimeDomainData'][0]
    #idx, packet = next( enumerate(packets) )
    for idx, packet in enumerate(packets):
        #print(str(idx))
        #type(packet['ChannelSamples']) == list, each item is a channel
        if idx != 0:
            if packets[idx - 1]['Header']['dataTypeSequence'] == 255:
                if packet['Header']['dataTypeSequence'] != 0:
                    missedPacketGaps = missedPacketGaps + 1
            else:
                if packet['Header']['dataTypeSequence'] != packets[idx - 1]['Header']['dataTypeSequence'] + 1:
                    missedPacketGaps = missedPacketGaps + 1

        if timing:
            #plotting based off system tick***********************************************
            if idx == 0:
                   endtime = (packet['Header']['systemTick'] - masterTick)*0.0001 + endtime1 + seconds # adjust the endtime of the first packet according to the masterTick
                   endtimeold = endtime - (len(packet['ChannelSamples'][0]['Value']) - 1) / sampleRate # plot back from endtime based off of sample rate
            else:
                   endtimeold = endtime
                   if packets[idx - 1]['Header']['systemTick'] < packets[idx]['Header']['systemTick']:
                       endtime = (packet['Header']['systemTick'] - masterTick)*0.0001 + endtime1 + seconds
                   else:
                       #systemTick has rolled over, add the appropriate number of seconds
                       seconds = seconds + rolloverseconds
                       endtime = (packets[idx]['Header']['systemTick'] - masterTick)*0.0001 + endtime1 + seconds

            #-------------------------------------------------------------------------
            channels = enumerate(packet['ChannelSamples'])
            tempChannelData = pd.DataFrame()

            for chIdx, channel in channels: #Construct Raw TD Data Structure
                tempChannelData = tempChannelData.append(pd.Series(channel['Value'], name = elecLabel[chIdx]))

            channelData = channelData.append(tempChannelData.transpose(), ignore_index = True)

            if spacing:
                #linearly spacing data between packet system ticks------------------------
                if idx != 0:
                    tvec = tvec.iloc[:-1]
                    tvec = tvec.append(pd.Series(np.linspace(endtimeold,endtime,len(packet['ChannelSamples'][0]['Value']) + 1)), ignore_index = True) # Linearly spacing between packet end times
                else:
                    tvec = tvec.iloc[:-1]
                    tvec = tvec.append(pd.Series(np.linspace(endtimeold,endtime,len(packet['ChannelSamples'][0]['Value']))), ignore_index = True)
            else:
                #sample rate spacing data between packet system ticks---------------------
                newTimes = pd.Series(np.arange(endtime-(len(packet['ChannelSamples'][0]['Value']) - 1)/sampleRate,endtime + 1/sampleRate,1/sampleRate))
                #pdb.set_trace()

                #if len(newTimes) != tempChannelData.shape[1]:
                #    pdb.set_trace()
                # TODO: something (round-off error?) is causing newTimes to have an extra entry on occasion. This is a kludgey fix, should revisit
                newTimes = newTimes[newTimes <= endtime]
                tvec = tvec.append(newTimes, ignore_index = True)

            packetSize.append(len(channel['Value']))
        else:
        #plotting based off packet gen times******************************************
            if packet['PacketGenTime'] > 0: # Check for Packet Gen Time
                if spacing:
                    #linearly spacing data between packet gen times-----------------------
                    if ('stp', 'var') not in locals():
                        stp = 1
                        endtime = (packet['PacketGenTime']-FirstGoodTime)/1000 + endtime1 # adjust the endtime of the first packet according to the FirstGoodTime
                        endtimeold = endtime - (len(packet['ChannelSamples'][0]['Value']) - 1)/sampleRate # plot back from endtime based off of sample rate
                        tvec = tvec.append(pd.Series(np.linspace(endtimeold,endtime,len(packet['ChannelSamples'][0]['Value']))), ignore_index = True)
                    else:
                        endtimeold = endtime
                        endtime = (packet['PacketGenTime']-FirstGoodTime)/1000 + endtime1
                        tvec = tvec.append(pd.Series(np.arange(endtime-(len(packet['ChannelSamples'][0]['Value']) - 1)/sampleRate,endtime + 1/sampleRate,1/sampleRate)), ignore_index = True)
                else:
                    #sample rate spacing data between packet gen times--------------------
                    endtime = (packet['PacketGenTime']-FirstGoodTime)/1000 + endtime1
                    tvec = tvec.append(pd.Series(np.arange(endtime-(len(packet['ChannelSamples'][0]['Value']) - 1)/sampleRate,endtime + 1/sampleRate,1/sampleRate)), ignore_index = True)
                #*****************************************************************************
                channels = enumerate(packet['ChannelSamples'])
                tempChannelData = pd.DataFrame()

                for chIdx, channel in channels: #Construct Raw TD Data Structure
                    tempChannelData = tempChannelData.append(pd.Series(channel['Value'], name = elecLabel[chIdx]))

                channelData = channelData.append(tempChannelData.transpose(), ignore_index = True)
                packetSize.append(len(channel['Value']))

        if idx != 0:
            if packets[idx-1]['Header']['systemTick'] > packet['Header']['systemTick']:
                loopCount = loopCount + 1
                loopTimeStamp.append(packet['Header']['timestamp']['seconds'])

        data.update({'elec_ids' : elecID})
        data.update(
            {
                'ExtendedHeaderIndices' : [0, 1, 2, 3],
                'extended_headers' : [{'Units' : elecUnits, 'ElectrodeLabel' : elecLabel[i]} for i in [0, 1, 2, 3]],
            }
            )

        #pdb.set_trace()
    data.update({'dataRaw' : channelData})
    data['dataRaw'].columns = [0,1,2,3]
    data.update({'tRaw' : tvec.values})

    data.update({'samp_per_s' : sampleRate})
    data.update({'start_time_s' : 0})

    nonIncreasingMask = np.concatenate((np.diff(data['tRaw']) <= 0, [False]))
    nonIncreasingTimes = deepcopy(data['tRaw'][nonIncreasingMask])
    nonIncreasingMagnitude = np.diff(data['tRaw'])[np.diff(data['tRaw']) <= 0]

    assert all(nonIncreasingMagnitude < 1e-10)
    data['tRaw'] = data['tRaw'][~nonIncreasingMask]
    data['dataRaw'] = data['dataRaw'].iloc[~nonIncreasingMask, :]

    data['t'] = np.arange(min(data['tRaw']), max(data['tRaw']), 1/sampleRate)
    data['data'] = pd.DataFrame(index = data['t'], columns = data['ExtendedHeaderIndices'])

    for idx, column in data['dataRaw'].items():
        f = interpolate.interp1d(data['tRaw'], column.values)
        data['data'][data['ExtendedHeaderIndices'][idx]] = f(data['t'])

def increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

#assert increasing(data['t'])
f, ax = plt.subplots()
#plt.show()
plotChan(data, whichChan, label = 'Raw data', mask = None,
    show = False, prevFig = f)
ax.plot(nonIncreasingTimes , nonIncreasingTimes * 0, 'ro')
plt.legend()

plotName = fileName.split('.')[0] + '_' +\
    data['extended_headers'][0]['ElectrodeLabel'] +\
    '_plot'
plt.savefig(fileDir + '/' + plotName + '.png')

with open(fileDir + '/' + plotName + '.pickle', 'wb') as File:
    pickle.dump(f, File)

plt.show()
### Get the Spectrogram
R = 30 # target bandwidth for spectrogram

data['spectrum'] = getSpectrogram(
    data, winLen_s, stepLen_s, R, 100, whichChan, plotting = True)

plotName = fileName.split('.')[0] + '_' +\
    data['extended_headers'][0]['ElectrodeLabel'] +\
    '_spectrum_plot'

plt.savefig(fileDir + '/' + plotName + '.png')
with open(fileDir + '/' + plotName + '.pickle', 'wb') as File:
    pickle.dump(plt.gcf(), File)

plt.show()
data.update({'winLen' : winLen_s, 'stepLen' : stepLen_s})

with open(fileDir + '/' + fileName.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
    pickle.dump(data, f, protocol=4 )
'''