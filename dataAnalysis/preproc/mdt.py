import pandas as pd
import matplotlib, math
from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
from copy import *
import argparse, linecache
from dataAnalysis.helperFunctions.mdt_constants import *
from scipy import interpolate

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
