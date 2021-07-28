import pandas as pd
import numpy as np
import pdb
import dataAnalysis.helperFunctions.helper_functions_new as hf
insTimeRollover = 65536


def loadSip(basePath, msecPerSample=2):
    insDataPath = basePath + '-Data.txt'
    insTimeSeries = pd.read_csv(insDataPath, sep='\t', skiprows=2)
    insTimeSeries.rename(columns=str.strip, inplace=True)
    if insTimeSeries.isnull().any().any():
        numNull = insTimeSeries.isnull().any(axis='index').sum()
        print(Warning('Error in loadSip(); assuming missing SenseChans'))
        senseChanNames = [i for i in insTimeSeries.columns if 'Sense' in i]
        dropSenseChans = senseChanNames[len(senseChanNames) - numNull:]
        newChanNames = (
            insTimeSeries.columns.drop(dropSenseChans).tolist() +
            dropSenseChans)
        insTimeSeries.columns = newChanNames
        insTimeSeries.fillna(0, inplace=True)
    packetNumDiff = insTimeSeries['PacketNumber'].diff().fillna(0)
    packetNumDiff[packetNumDiff == -255] += 256
    # packetIntervals = insTimeSeries['Timestamp'].diff()
    # packetIntervals[packetIntervals < 0] += insTimeRollover
    packetRolledOver = packetNumDiff != 0
    packetEnd = (insTimeSeries.index[packetRolledOver] - 1).tolist() + [insTimeSeries.index[-1]]
    packetRolloverGroup = packetRolledOver.cumsum()
    metadata = insTimeSeries.loc[packetEnd, ['PacketNumber', 'Timestamp']]
    allFrameLengths = metadata['Timestamp'].diff()
    frameLength = allFrameLengths.value_counts().idxmax()
    metadata['rolloverGroup'] = (allFrameLengths < 0).cumsum()
    for name, group in metadata.groupby('rolloverGroup'):
        # duplicateSysTick = group.duplicated('systemTick')
        duplicateSysTick = group['Timestamp'].diff() == 0
        if duplicateSysTick.any():
            duplicateIdxs = duplicateSysTick.index[np.flatnonzero(duplicateSysTick)]
            for duplicateIdx in duplicateIdxs:
                sysTickVal = group.loc[duplicateIdx, 'Timestamp']
                allOccurences = group.loc[group['Timestamp'] == sysTickVal, :]
                assert len(allOccurences) == 2
                #  'dataTypeSequence' rolls over, correct for it
                specialCase = (
                    (allOccurences['PacketNumber'] == 255).any() &
                    (allOccurences['PacketNumber'] == 0).any()
                )
                if specialCase:
                    idxNeedsChanging = allOccurences['PacketNumber'].idxmax()
                else:
                    idxNeedsChanging = allOccurences['PacketNumber'].idxmin()
                metadata.loc[idxNeedsChanging, 'Timestamp'] -= frameLength
    metadata['PacketSize'] = packetRolloverGroup.value_counts().sort_index(kind='mergesort').to_numpy(dtype='int')
    metadata['FirstTimestamp'] = metadata['Timestamp'] - 10 * msecPerSample * (metadata['PacketSize'] - 1)
    metadata['SampleOverlap'] = (metadata['Timestamp'] - metadata['FirstTimestamp'].shift(-1).values).fillna(- 10 * msecPerSample)
    metadata['packetsOverlapFuture'] = metadata['SampleOverlap'] > 0
    idxNeedsChanging = metadata.index[metadata['packetsOverlapFuture']]
    ilocNeedsChanging = np.flatnonzero(metadata['packetsOverlapFuture'])
    nextGoodSysTicks = metadata['Timestamp'].iloc[ilocNeedsChanging + 1]
    nextGoodPacketSizes = metadata['PacketSize'].iloc[ilocNeedsChanging + 1]
    correctedSysTicks = (
        nextGoodSysTicks.to_numpy() -
        nextGoodPacketSizes.to_numpy() * msecPerSample)
    metadata.loc[idxNeedsChanging, 'Timestamp'] = correctedSysTicks
    for name in packetRolloverGroup.unique():
        indices = packetRolloverGroup.index[packetRolloverGroup == name]
        newT = metadata.iloc[name]['Timestamp']
        insTimeSeries.loc[indices, 'Timestamp'] = newT
    timestampNRollover = 0
    for idx, dfIdx in enumerate(packetEnd):
        if idx == 0:
            curTimestamp = insTimeSeries.loc[dfIdx, 'Timestamp']
            lastTimestamp = curTimestamp
            prevIdx = insTimeSeries.index[0]
        else:
            curTimestamp = (
                insTimeSeries.loc[dfIdx, 'Timestamp'] +
                insTimeRollover * timestampNRollover)
            prevIdx = packetEnd[idx - 1]
            if curTimestamp < lastTimestamp:
                timestampNRollover += 1
                curTimestamp += insTimeRollover
            lastTimestamp = curTimestamp
        nSamples = insTimeSeries.loc[prevIdx:dfIdx, :].shape[0]
        offsets = np.arange(
            - (nSamples - 1) * msecPerSample,
            msecPerSample,
            msecPerSample)
        insTimeSeries.loc[
            prevIdx:dfIdx,
            'calculatedTimestamp'] = curTimestamp * .1 + offsets
    # 
    insTimeSeries.drop_duplicates(
        subset=['calculatedTimestamp'], keep='first',
        inplace=True)
    insTimeSeries.sort_values(by=['calculatedTimestamp'], inplace=True)
    newT = np.arange(
        insTimeSeries['calculatedTimestamp'].iloc[0],
        insTimeSeries['calculatedTimestamp'].iloc[-1] + msecPerSample, msecPerSample)
    dataCol = [i for i in insTimeSeries.columns if 'Sense' in i]
    insTimeSeries = hf.interpolateDF(
        insTimeSeries, newT, x='calculatedTimestamp',
        columns=dataCol)
    insTimeSeries['calculatedTimestamp'] -= insTimeSeries['calculatedTimestamp'].iloc[0]
    insTimeSeries['calculatedTimestamp'] /= 1e3  # convert to msec
    return insTimeSeries
