# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:07:44 2019

@author: Radu
"""

from matplotlib import pyplot as plt
from scipy import stats
from importlib import reload

import matplotlib, pdb, pickle, traceback
import numpy as np
import pandas as pd
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.motor_encoder as mea
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.helperFunctions.estimateElectrodeImpedances as eti
import h5py
import os
import math as m
import seaborn as sns
import scipy.interpolate as intrp
import json
import rcsanalysis.packetizer as rcsa
import rcsanalysis.packet_func as rcsa_helpers
import math as m
from datetime import datetime
from currentExperiment import *

matplotlib.rcParams['agg.path.chunksize'] = 10000
#matplotlib.use('PS')   # generate interactive output by default
#matplotlib.use('TkAgg')   # generate interactive output by default

jsonSessionNames = {
    0: 'Session1548006829607',
 #   1: 'Session1549127148513',
 #   2: 'Session1549132573173',
 #   3: '',
 #   4: 'Session1548518982240'
    }
jsonSessionNames = {0: 'Session1548087177984', 1: 'Session1548087855083'}
print('\n\n\n\nsessions in this analysis {}\n'.format(jsonSessionNames))
progLocationLookup = {
    'rostral': 2,
    'caudal': 0,
    'midline': 1,
    'nostim': 3
    }

#  blockIdx = 1
plotStimStatusAll = pd.DataFrame()
plotErrorStatusAll = pd.DataFrame()
runningTotalAverageBlocks = 0
seshList = list(jsonSessionNames.keys())

for blockIdx in seshList:
    jsonPath = os.path.join(insFolder, jsonSessionNames[blockIdx], deviceName)

    with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
        deviceSettings = json.load(f)
    with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
        stimLog = json.load(f)
    with open(os.path.join(jsonPath, 'ErrorLog.json'), 'r') as f:
        errorLog = json.load(f)
    with open(os.path.join(jsonPath, 'TimeSync.json'), 'r') as f:
        timeSync = json.load(f)

    elecConfiguration, senseInfo = (
        hf.getINSDeviceConfig(insFolder, jsonSessionNames[blockIdx])
        )
    print('Sensing options:')
    print(senseInfo)
    try:
        with open(os.path.join(insFolder, jsonSessionNames[blockIdx], '.MDT_SummitTest', jsonSessionNames[blockIdx] + '.json'), 'r') as f:
            commentsLog = json.load(f)
            print(commentsLog)
    except Exception:
        #traceback.print_exc()
        print('{} no commment, '.format(jsonSessionNames[blockIdx]))

    sessionTimeStamp = float(
        jsonSessionNames[blockIdx].split('Session')[-1]
        ) / 1e3

    sessionTime = datetime.fromtimestamp(sessionTimeStamp)
    print('Session {} Started at {}'.format(jsonSessionNames[blockIdx], sessionTime))
    # added to hf.getINSDeviceConfig
    group0Programs = deviceSettings[0]['TherapyConfigGroup0']['programs']
    electrodeStatus = pd.DataFrame(index=range(4), columns=range(17))
    electrodeType = pd.DataFrame(index=range(4), columns=range(17))
    electrodeConfiguration = [{'cathodes': [], 'anodes': []} for i in range(4)]

    for progIdx in range(4):
        for elecIdx in range(17):
            electrodeStatus.loc[progIdx, elecIdx] =\
                not group0Programs[progIdx]['electrodes']['electrodes'][elecIdx]['isOff']
            electrodeType.loc[progIdx, elecIdx] =\
                group0Programs[progIdx]['electrodes']['electrodes'][elecIdx]['electrodeType']
            if electrodeStatus.loc[progIdx, elecIdx]:
                if electrodeType.loc[progIdx, elecIdx] == 1:
                    electrodeConfiguration[progIdx]['anodes'].append(elecIdx)
                else:
                    electrodeConfiguration[progIdx]['cathodes'].append(elecIdx)
    print('electrode configuration was {}'.format(electrodeConfiguration))
    ###
    # Extract stim log
    progAmpNames = ['program{}_amplitude'.format(progIdx) for progIdx in range(4)]
    progPWNames = ['program{}_pw'.format(progIdx) for progIdx in range(4)]

    stripProgName = lambda x: int(x.split('program')[-1].split('_')[0])

    stimStatus = pd.DataFrame(
        columns=['HostUnixTime', 'therapyStatus', 'activeGroup', 'frequency'] +
                ['amplitudeChange', 'pwChange'] + progAmpNames + progPWNames
        )

    activeGroup = np.nan
    for entry in stimLog:
        if 'RecordInfo' in entry.keys():
            entryData = {'HostUnixTime': entry['RecordInfo']['HostUnixTime']}

        if 'therapyStatusData' in entry.keys():
            entryData.update(entry['therapyStatusData'])
            if 'activeGroup' in entry['therapyStatusData'].keys():
                activeGroup = entry['therapyStatusData']['activeGroup']

        activeGroupSettings = 'TherapyConfigGroup{}'.format(activeGroup)
        thisAmplitude = None
        thisPW = None
        ampChange = False
        pwChange = False
        if activeGroupSettings in entry.keys():
            if 'RateInHz' in entry[activeGroupSettings]:
                entryData.update(
                    {'frequency': entry[activeGroupSettings]['RateInHz']}
                    )

            for progIdx in range(4):
                programName = 'program{}'.format(progIdx)
                if programName in entry[activeGroupSettings].keys():
                    if 'amplitude' in entry[activeGroupSettings][programName]:
                        thisAmplitude = entry[activeGroupSettings][programName][
                            'AmplitudeInMilliamps']
                        entryData.update(
                            {
                                programName + '_amplitude': thisAmplitude
                            }
                        )
                        if thisAmplitude > 0:
                            ampChange = True
                    if 'pulseWidth' in entry[activeGroupSettings][programName]:
                        thisPW = entry[activeGroupSettings][programName][
                            'PulseWidthInMicroseconds']
                        entryData.update(
                            {
                                programName + '_pw': thisPW
                            }
                        )
                        if thisPW > 0:
                            pwChange = True

        entryData.update({'amplitudeChange': ampChange})
        entryData.update({'pwChange': pwChange})

        #  was there an amplitude change?
        entrySeries = pd.Series(entryData)
        stimStatus = stimStatus.append(entrySeries, ignore_index=True, sort=True)

    stimStatus.fillna(method='ffill', axis=0, inplace=True)
    # package above into rcsAnalysis
    errorStatus = pd.DataFrame(
        columns=['HostUnixTime', 'CrcError', 'MaxStackDueToOorUnreliable',
                 'OOR', 'PorHasOccurred', 'SequenceError',
                 'SequenceNumber', 'StatusChange1'
                 ]
        )
    
    for entry in errorLog:
        if 'Error' in entry.keys():
            entryData = {'HostUnixTime': entry['Error']['unixRxTime']}
            entryData.update(entry['Error']['TheLinkStatus'])
            entrySeries = pd.Series(entryData)
            errorStatus = errorStatus.append(entrySeries, ignore_index=True, sort=True)
            
    errorStatus['HostUnixDateTime'] =\
        ( errorStatus['HostUnixTime'] / 1e3).apply(datetime.fromtimestamp)
        
    stimStatus['activeProgram'] = np.nan
    stimStatus['amplitude'] = np.nan

    ampChangeMask = stimStatus['amplitudeChange']
    stimMaxAmplitude = stimStatus.loc[ampChangeMask, progAmpNames].max(axis=1)
    stimStatus.loc[ampChangeMask, 'amplitude'] = stimMaxAmplitude

    stimmingMask = stimStatus.loc[:, progAmpNames].max(axis=1) > 0
    stimAtActive = stimStatus.loc[stimmingMask, progAmpNames]
    activeProgram = stimAtActive.idxmax(axis=1).apply(stripProgName)
    stimStatus['activeProgram'].update(activeProgram)
    stimStatus.fillna(method='ffill', axis=0, inplace=True)

    stimStatus['HostUnixDateTime'] =\
        ( stimStatus['HostUnixTime'] / 1e3).apply(datetime.fromtimestamp)

    stimDuration = stimStatus['HostUnixDateTime'].iloc[-1] - stimStatus['HostUnixDateTime'].iloc[0]
    print('{} duration was {}'.format(jsonSessionNames[blockIdx], stimDuration))

    countMultiplier = {
        'caudal': 0.25,
        'rostral': 0.25,
        'midline': 0.5,
        'nostim':0
        }

    trialCounts = []
    for name, group in stimStatus.loc[ampChangeMask, :].groupby('frequency'):
        stimStartCount = group['activeProgram'].value_counts()
        print(' ')
        for location in ['caudal', 'rostral', 'midline']:
            
            if progLocationLookup[location] in stimStartCount.index:
                nBlocks = (
                    stimStartCount[
                        progLocationLookup[location]] * 
                        countMultiplier[location])
                print('Frequency {}, category {}: {} trials'.format(name, location, nBlocks))
                trialCounts.append(nBlocks)
        
    averageNBlocks = np.nanmean(trialCounts)
    runningTotalAverageBlocks += averageNBlocks
    print('Average {} trials per condition'.format(averageNBlocks))

    #  Plot stim status over time for this trial
    plottingRange = np.arange(
        stimStatus['HostUnixTime'].min(),
        stimStatus['HostUnixTime'].max(), 200)
    plottingColumns = ['HostUnixTime', 'frequency', 'therapyStatus', 'amplitudeChange'] + progAmpNames

    plottingEntries = pd.DataFrame(columns=plottingColumns)
    plottingEntries['HostUnixTime'] = plottingRange
    plotStimStatus = pd.concat([
        stimStatus.loc[:, plottingColumns], plottingEntries
        ])
    plotStimStatus.sort_values('HostUnixTime', inplace=True)
    plotStimStatus.fillna(method='ffill', axis=0, inplace=True)

    plotStimStatus['t'] =\
        ( (plotStimStatus['HostUnixTime']) / 1e3)
    plotStimStatus.set_index('HostUnixTime', inplace = True)
    
    errorStatus['t'] =\
        ( (errorStatus['HostUnixTime']) / 1e3)
    errorStatus.set_index('HostUnixTime', inplace = True)
        
    plotStimStatusAll = pd.concat((plotStimStatusAll, plotStimStatus))
    plotErrorStatusAll = pd.concat((plotErrorStatusAll, errorStatus))

fig, ax = plt.subplots(3, 1, sharex=True)
plotStimStatusAll.plot(
    x='t', y='frequency',
    ax=ax[1], label='Current Frequency')
ax[1].set_ylabel('Frequency (Hz)')

plotStimStatusAll.plot(
    x='t', y='therapyStatus',
    ax=ax[0], label='therapyStatus')
plotStimStatusAll['amplitudeChange'] = plotStimStatusAll['amplitudeChange'].astype(float)
plotStimStatusAll.plot(
    x='t', y='amplitudeChange',
    ax=ax[0], label='amplitudeChange')

plotErrorStatusAll['OOR'] = plotErrorStatusAll['OOR'].astype(float)
plotErrorStatusAll.plot(
    x='t', y='OOR',
    ax=ax[0], label='OOR', style = ['o'])

for columnName in progAmpNames:
    plotStimStatusAll.plot(
        x='t', y=columnName,
        ax=ax[2], label=columnName)
    
ax[2].set_ylabel('Stim Amplitude (mA)')
ax[2].set_xlabel('Host Unix Time (sec)')
plt.suptitle('Stim State')
plt.show()
print(' ')
print('Total average trials = {}'.format(runningTotalAverageBlocks))
print('Total average trials per condition = {}'.format(
    round(
        runningTotalAverageBlocks / 3)
        )
    )