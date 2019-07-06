import pandas as pd
import numpy as np
import pdb
import dataAnalysis.helperFunctions.helper_functions_new as hf
insTimeRollover = 65536


def loadSip(basePath, msecPerSample=2):
    insDataPath = basePath + '-Data.txt'
    insTimeSeries = pd.read_csv(insDataPath, sep='	', skiprows=2)
    if insTimeSeries.isnull().sum().sum() > 0:
        print('Error in loadSip() Trying Again')
        insTimeSeries = pd.read_csv(
            insDataPath, sep='	', skiprows=3,
            names=[
                'SampleNumber ', ' SenseChannel1 ',
                ' SenseChannel2 ', ' StimulationClass ',
                ' PacketNumber ', ' Timestamp ', ' IsDroppedPacket '])
    insTimeSeries.rename(columns=str.strip, inplace=True)
    packetNumDiff = insTimeSeries['PacketNumber'].diff(periods=-1).fillna(-1)
    packetNumDiff[packetNumDiff == 255] -= 256
    # packetIntervals = insTimeSeries['Timestamp'].diff()
    # packetIntervals[packetIntervals < 0] += insTimeRollover
    packetEnd = insTimeSeries.index[packetNumDiff != 0].tolist()
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
        offsets = np.arange(- (nSamples - 1) * msecPerSample, msecPerSample, msecPerSample)
        insTimeSeries.loc[
            prevIdx:dfIdx,
            'calculatedTimestamp'] = curTimestamp * .1 + offsets
    # pdb.set_trace()
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
