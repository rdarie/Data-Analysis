from brPY.brpylib import NsxFile, NevFile, brpylib_ver
try:
    import plotly
    import plotly.plotly as py
    import plotly.tools as tls
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
except:
    pass

import matplotlib, pdb, sys, itertools, os, pickle, gc, random, string,\
subprocess, collections, traceback, peakutils, math, argparse

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib import cm

import numpy as np
import pandas as pd
import math as m
import tables as pt
import seaborn as sns
import rcsanalysis.packet_func as rcsa_helpers

import openEphysAnalysis.OpenEphys as oea

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, make_scorer, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
import sklearn.pipeline
from sklearn.feature_selection import RFE,RFECV

import datetime
from datetime import datetime as dt
import json

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
    from tkFileDialog import askopenfilenames, askopenfilename, asksaveasfilename
except ImportError:
    # for Python3
    from tkinter import filedialog   ## notice lowercase 't' in tkinter here

try:
    import libtfr
    HASLIBTFR = True
except:
    import scipy.signal
    HASLIBTFR = False

from scipy import interpolate
from scipy import stats
from copy import *
import matplotlib.backends.backend_pdf
from fractions import gcd
import h5py
LABELFONTSIZE = 5

def getPlotOpts(names):
    """
        Gets the colors to assign to n traces, where n is the length of the
        names list, which contains the names of the n traces.
    """
    viridis_cmap = matplotlib.cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=0, vmax=255)
    colorsRgb = [colors.colorConverter.to_rgb(viridis_cmap(norm(i)))
                    for i in range(0, 255)]

    plotColPlotly, plotColMPL = [], []
    h = 1.0/(len(names)-1)

    for k in range(len(names)):
        C = list(map(np.uint8, np.array(viridis_cmap(k*h)[:3])*255))
        plotColPlotly.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        plotColMPL.append(norm(C))

    names2int = list(range(len(names)))
    line_styles = ['-', '--', ':', '-.']
    return viridis_cmap, norm, colorsRgb, plotColPlotly, plotColMPL, names2int, line_styles


def getNEVData(filePath, elecIds):
    # Version control
    brpylib_ver_req = "1.3.1"
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

    # Open file and extract headers
    nev_file = NevFile(filePath)
    # Extract data and separate out spike data
    spikes = nev_file.getdata(elecIds)

    spikes = spikes['spike_events']

    spikes['basic_headers'] = nev_file.basic_header
    spikes['extended_headers'] = nev_file.extended_headers
    # Close the nev file now that all data is out
    nev_file.close()
    return spikes

def getOpenEphysFolder(folderPath, chIds = 'all', adcIds = 'all', chanNames = None, startTime_s = 0, dataLength_s = 'all', downsample = 1):

    # load one file to get the metadata
    source = '100'
    session = '0'
    dummyChFileName = None
    if chIds is not None:
        chprefix = 'CH'
        if chIds == 'all':
            chIds = oea._get_sorted_channels(folderPath, 'CH', session, source)
            dummyChFileName = os.path.join(folderPath, source + '_'+chprefix + '1.continuous')
        else: # already gave a list
            if not isinstance(chIds, collections.Iterable):
                chIds = [chIds]
            dummyChFileName = os.path.join(folderPath, source + '_'+ chprefix + str(chIds[0]) + '.continuous')

        recordingData = oea.loadFolderToArray(folderPath, channels = chIds)
    else:
        recordingData = None

    if adcIds is not None:
        if adcIds == 'all':
            adcIds = oea._get_sorted_channels(folderPath, 'ADC', session, source)
            if dummyChFileName is None:
                chprefix = 'ADC'
                dummyChFileName = os.path.join(folderPath, source + '_'+ chprefix + '1.continuous')
        else: # already gave a list
            if not isinstance(adcIds, collections.Iterable):
                adcIds = [adcIds]
            if dummyChFileName is None:
                chprefix = 'ADC'
                dummyChFileName = os.path.join(folderPath, source + '_'+ chprefix + str(chIds[0]) + '.continuous')
        adcData = oea.loadFolderToArray(folderPath, channels = adcIds, chprefix = 'ADC')
    else:
        adcData = None

    dummyChannelData = oea.loadContinuous(dummyChFileName)

    t = np.arange(dummyChannelData['data'].shape[0]) / float(dummyChannelData['header']['sampleRate'])

    if startTime_s > 0:
        dataMask = t > startTime_s
        t = t[dataMask]
        if adcData is not None:
            adcData = adcData[dataMask,:]
        if recordingData is not None:
            recordingData = recordingData[dataMask,:]

    if dataLength_s == 'all':
        dataLength_s = t[-1]
    else:
        if dataLength_s < t[-1]:
            dataMask = t < dataLength_s
            t = t[dataMask]
            if adcData is not None:
                adcData = adcData[dataMask,:]
            if recordingData is not None:
                recordingData = recordingData[dataMask,:]
        else:
            dataLength_s = t[-1]

    #pdb.set_trace()
    data = pd.DataFrame(np.concatenate((recordingData, adcData), axis = 1), columns = chanNames)
    isADC = np.ones(len(data.columns), dtype = np.bool)
    isADC[:len(chIds)] = False

    channelData = {
        'data' : data,
        't' : pd.Series(t, index = data.index),
        'basic_headers' :dummyChannelData['header'],
        'badData' : {},
        'start_time_s' : startTime_s,
        'data_time_s' : dataLength_s,
        'samp_per_s' : float(dummyChannelData['header']['sampleRate'])
        }

    channelData['basic_headers']['isADC'] = isADC
    channelData['basic_headers']['chanNames'] = chanNames
    return channelData


def getNSxData(filePath, elecIds, startTime_s, dataLength_s, spikeStruct, downsample=1):
    # Version control
    brpylib_ver_req = "1.3.1"
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

    elecCorrespondence = spikeStruct.loc[elecIds, 'nevID'].astype(int)
    # Open file and extract headers

    nsx_file = NsxFile(filePath)
    # Extract data - note: data will be returned based on *SORTED* nevIds, see cont_data['elec_ids']
    # pdb.set_trace()
    channelData = nsx_file.getdata(
        list(elecCorrespondence.values), startTime_s, dataLength_s, downsample)

    rowIndex = range(
        int(channelData['start_time_s'] * channelData['samp_per_s']),
        int((channelData['start_time_s'] + channelData['data_time_s']) *
            channelData['samp_per_s']))
    channelData['data'] = pd.DataFrame(
        channelData['data'].transpose(),
        index=rowIndex, columns=elecCorrespondence.sort_values().index.values)

    channelData['t'] = channelData['start_time_s'] + np.arange(channelData['data'].shape[0]) / channelData['samp_per_s']
    channelData['t'] = pd.Series(
        channelData['t'], index=channelData['data'].index)
    channelData['badData'] = {}
    channelData['basic_headers'] = nsx_file.basic_header
    channelData['extended_headers'] = nsx_file.extended_headers
    # Close the nsx file now that all data is out
    nsx_file.close()
    return channelData


def getINSTDFromJson(
    folderPath, sessionName, deviceName='DeviceNPC700373H', fs=500
    ):

    jsonPath = os.path.join(folderPath, sessionName, deviceName)
    try:
        #  raise(Exception('Debugging, always extract fresh'))
        tdData = pd.read_csv(
            os.path.join(jsonPath, 'RawDataTD.csv'))
        tdData['microseconds'] = pd.to_timedelta(
            tdData['microseconds'])
    except Exception:
        traceback.print_exc()

        with open(os.path.join(jsonPath, 'RawDataTD.json'), 'r') as f:
            timeDomainJson = json.load(f)
            
        intersampleTickCount = (1/fs) / (100e-6)

        timeDomainMeta = rcsa_helpers.extract_td_meta_data(timeDomainJson)
        # assume n channels is constant
        nChan = int(timeDomainMeta[0, 8])

        timeDomainMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
            timeDomainMeta)

        timeDomainMeta, packetsNeededFixing =\
            rcsa_helpers.correct_meta_matrix_time_displacement(
                timeDomainMeta, intersampleTickCount)

        num_real_points, num_macro_rollovers, loss_as_scalar =\
            rcsa_helpers.calculate_statistics(
                timeDomainMeta, intersampleTickCount)
        
        timeDomainValues = rcsa_helpers.unpacker_td(
            timeDomainMeta, timeDomainJson, intersampleTickCount)
        
        tdData = rcsa_helpers.save_to_disk(
            timeDomainValues, os.path.join(
                jsonPath, 'RawDataTD.csv'),
            time_format='full', data_type='td', num_cols=nChan)

    tdData['t'] = tdData['microseconds'] / datetime.timedelta(seconds=1)
    tdDataSorted = tdData.drop_duplicates(
        ['t']
        ).sort_values('t').reset_index(drop=True)
    td = {
    'data': tdDataSorted,
    't': tdDataSorted['microseconds'] / datetime.timedelta(seconds=1)
    }
    td['data']['INSTime'] = td['t']
    td['INSTime'] = td['t']
    return td

def getINSAccelFromJson(folderPath, sessionName, deviceName = 'DeviceNPC700373H', fs = 64):
    jsonPath = os.path.join(folderPath, sessionName, deviceName)
    try:
        #  raise(Exception('Debugging, always extract fresh'))
        accelData = pd.read_csv(os.path.join(jsonPath, 'RawDataAccel.csv'))
        accelData['microseconds'] = pd.to_timedelta(accelData['microseconds'])
    except Exception:
        traceback.print_exc()

        with open(os.path.join(jsonPath, 'RawDataAccel.json'), 'r') as f:
            accelJson = json.load(f)

        intersampleTickCount = (1/fs) / (100e-6)
        accelMeta = rcsa_helpers.extract_accel_meta_data(accelJson)

        accelMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
            accelMeta)

        accelMeta, packetsNeededFixing =\
            rcsa_helpers.correct_meta_matrix_time_displacement(
                accelMeta, intersampleTickCount)

        accelDataValues = rcsa_helpers.unpacker_accel(
            accelMeta, accelJson, intersampleTickCount)
        accelData = rcsa_helpers.save_to_disk(
            accelDataValues, os.path.join(
                jsonPath, 'RawDataAccel.csv'),
            time_format='full', data_type='accel')

    accelData['t'] = accelData['microseconds'] / datetime.timedelta(seconds=1)
    accelDataSorted = accelData.drop_duplicates(
        ['t']
        ).sort_values('t').reset_index(drop=True)
    accel = {
        'data': accelDataSorted,
        't': accelDataSorted['microseconds'] / datetime.timedelta(seconds=1)
        }
    inertia = accel['data']['accel_x']**2 +\
        accel['data']['accel_y']**2 +\
        accel['data']['accel_z']**2
    inertia = inertia.apply(np.sqrt)
    accel['data']['inertia'] = inertia

    accel['INSTime'] = accel['t']
    accel['data']['INSTime'] = accel['t']
    return accel

def getINSTapTimestamp(
    td, accel,
    tStart = 0, tStop = 30, iti = 1/4,
    visibleInChannel = False,
    ):
    def deviance(x, timeMask):
        return np.abs(stats.zscore(x[timeMask]))
        
    plotMask = (accel['t'] > tStart) & (accel['t'] < tStop)
    plotMaskTD = (td['t'] > tStart) & (td['t'] < tStop)

    dev = pd.Series(
        (
            deviance(accel['data']['inertia'], plotMask)
        ))

    accelPeakIdx = getTriggers(
        dev, iti=iti, fs=64, thres=5,
        edgeType='rising', minAmp=None, minTrainLength=2*iti,
        expectedTime=None, keep_max=False, plotting=False)

    if visibleInChannel:
        tapDetectSignal = td['data'].loc[plotMaskTD, visibleInChannel]

        tStart = accel['t'].loc[accelPeakIdx[0]] - .1
        tStop = accel['t'].loc[accelPeakIdx[-1]] + .1

        plotMask = (accel['t'] > tStart) & (accel['t'] < tStop)
        plotMaskTD = (td['t'] > tStart) & (td['t'] < tStop)

        peakIdx = getTriggers(
            tapDetectSignal.loc[plotMaskTD], iti=iti, fs=500, thres=3,
            edgeType='rising', minAmp=None, minTrainLength=2*iti,
            expectedTime=None, keep_max=True, plotting=False)

        tdPeakIdx = tapDetectSignal.loc[plotMaskTD].index[peakIdx]
        tapTimestamps = td['t'].loc[tdPeakIdx]
        peakIdx = tdPeakIdx
    else:
        tapTimestamps = accel['t'].loc[accelPeakIdx]
        peakIdx = accelPeakIdx

    return tapTimestamps, peakIdx

def synchronizeINStoNSP(
    tapSpikes, tapTimestamps,
    td = None, accel = None
    ):
    # sanity check that the intervals match
    print('On the INS, the diff() between taps was\n{}'.format(
    tapTimestamps.diff().dropna()
    ))
    print('On the NSP, the diff() between taps was\n{}'.format(
        np.diff(tapSpikes['TimeStamps'][0])
        ))
    print('This amounts to a difference of {}'.format(
        tapTimestamps.diff().dropna().values -
        np.diff(tapSpikes['TimeStamps'][0])
        ))

    synchPolyCoeffsINStoNSP = np.polyfit(
        x=tapTimestamps.values, y=tapSpikes['TimeStamps'][0], deg=1)
    timeInterpFunINStoNSP = np.poly1d(synchPolyCoeffsINStoNSP)
    if td is not None:
        td['NSPTime'] = pd.Series(
            timeInterpFunINStoNSP(td['t']), index=td['t'].index)
    # accel['originalTime'] = accel['t']
    if accel is not None:
        accel['NSPTime'] = pd.Series(
            timeInterpFunINStoNSP(accel['t']), index=accel['t'].index)

    return td, accel, timeInterpFunINStoNSP

def synchronizeHUTtoINS(folderPath, sessionName, deviceName = 'DeviceNPC700373H'):
    jsonPath = os.path.join(folderPath, sessionName, deviceName)
    with open(os.path.join(jsonPath, 'TimeSync.json'), 'r') as f:
        timeSync = json.load(f)[0]
        
    timeSyncData = rcsa_helpers.extract_time_sync_meta_data(timeSync)
    synchPolyCoeffsHUTtoINS = np.polyfit(
        x=timeSyncData['HostUnixTime'].values,
        y=timeSyncData['microseconds'].values * 1e-6,
        deg=1)
    timeInterpFunHUTtoINS = np.poly1d(synchPolyCoeffsHUTtoINS)
    return timeInterpFunHUTtoINS

def serializeINSStimLog(folderPath, sessionName, deviceName = 'DeviceNPC700373H'):
    jsonPath = os.path.join(folderPath, sessionName, deviceName)
    with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
        stimLog = json.load(f)
    progAmpNames = ['program{}_amplitude'.format(progIdx) for progIdx in range(4)]
    progPWNames = ['program{}_pw'.format(progIdx) for progIdx in range(4)]

    stimStatus = rcsa_helpers.extract_stim_meta_data(stimLog)
    stripProgName = lambda x: int(x.split('program')[-1].split('_')[0])
    stimStatus['activeProgram'] = stimStatus.loc[:,progAmpNames].idxmax(axis = 1).apply(stripProgName)
    stimStatus['maxAmp'] = stimStatus.loc[:,progAmpNames].max(axis = 1)

    try:
        timeInterpFunHUTtoINS = synchronizeHUTtoINS(
            folderPath, sessionName, deviceName)
        stimStatus['INSTime'] = pd.Series(
            timeInterpFunHUTtoINS(stimStatus['HostUnixTime']),
            index=stimStatus['HostUnixTime'].index)
    except Exception:
        traceback.print_exc()

    return stimStatus

def getINSDeviceConfig(folderPath, sessionName, deviceName = 'DeviceNPC700373H'):
    jsonPath = os.path.join(folderPath, sessionName, deviceName)

    with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
        deviceSettings = json.load(f)
    with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
        stimLog = json.load(f)

    progIndices = list(range(4))
    groupIndices = list(range(4))
    elecIndices = list(range(17))

    electrodeConfiguration = [[{'cathodes': [], 'anodes': []} for i in progIndices] for j in groupIndices]
    
    dfIndex = pd.MultiIndex.from_product([groupIndices, progIndices], names=['group', 'program'])

    electrodeStatus = pd.DataFrame(index=dfIndex, columns=elecIndices)
    electrodeType = pd.DataFrame(index=dfIndex, columns=elecIndices)

    for groupIdx in groupIndices:
        groupPrograms = deviceSettings[0]['TherapyConfigGroup{}'.format(groupIdx)]['programs']
        for progIdx in progIndices:
            for elecIdx in elecIndices:
                electrodeStatus.loc[(groupIdx, progIdx), elecIdx] =\
                    not groupPrograms[progIdx]['electrodes']['electrodes'][elecIdx]['isOff']
                electrodeType.loc[(groupIdx, progIdx), elecIdx] =\
                    groupPrograms[progIdx]['electrodes']['electrodes'][elecIdx]['electrodeType']
                if electrodeStatus.loc[(groupIdx, progIdx), elecIdx]:
                    if electrodeType.loc[(groupIdx, progIdx), elecIdx] == 1:
                        electrodeConfiguration[groupIdx][progIdx]['anodes'].append(elecIdx)
                    else:
                        electrodeConfiguration[groupIdx][progIdx]['cathodes'].append(elecIdx)
    return electrodeStatus, electrodeType, electrodeConfiguration

def getBadSpikesMask(spikes, nStd = 5, whichChan = 0, plotting = False, deleteBad = False):
    """
        where spikes comes from an NSX file extracted with brpy
    """
    spikesBar = [np.mean(sp, axis = 0) for sp in spikes['Waveforms']]
    spikesStd = [np.std (sp, axis = 0) for sp in spikes['Waveforms']]

    t = np.arange(spikesBar[0].shape[0])
    if plotting:
        fi = plt.figure()
        plt.plot(t, spikesBar[whichChan])
        plt.fill_between(t, spikesBar[whichChan]+spikesStd[whichChan],
                         spikesBar[whichChan]-spikesStd[whichChan],facecolor='blue',
                         alpha = 0.3, label = 'mean(spike)')

    badMask = []
    for idx, sp in enumerate(spikes['Waveforms']):
        maxAcceptable = np.abs(spikesBar[idx]) + nStd*spikesStd[idx]
        outliers = [(np.abs(row) > maxAcceptable).any() for row in sp]
        badMask.append(np.array(outliers, dtype = bool))
    #
    if deleteBad:
        for idx, sp in enumerate(spikes['Waveforms']):
            spikes['Waveforms'][idx] = sp[np.logical_not(badMask[idx])]
            spikes['Classification'][idx] = np.array(spikes['Classification'][idx])[np.logical_not(badMask[idx])]
            spikes['TimeStamps'][idx] = np.array(spikes['TimeStamps'][idx])[np.logical_not(badMask[idx])]
    return badMask

def fillInOverflow(channelData, plotting = False, fillMethod = 'constant'):
    # TODO merge this into getBadContinuousMask
    overflowMask = pd.DataFrame(False, index = channelData.index,
        columns = channelData.columns,
        dtype = np.bool).to_sparse(fill_value=False)
    #pdb.set_trace()
    columnList = list(channelData.columns)
    nChan = len(columnList)
    for idx, row in channelData.iteritems():

        ch_idx  = columnList.index(idx)
        col_idx = channelData.columns.get_loc(idx)

        if os.fstat(0) == os.fstat(1):
            endChar = '\r'
            print("Running fillInOverflow: %d%%" % int((ch_idx + 1) * 100 / nChan), end = endChar)
        else:
            print("Running fillInOverflow: %d%%" % int((ch_idx + 1) * 100 / nChan))

        # get the first difference of the row
        rowDiff = row.diff()
        rowDiff.fillna(0, inplace = True)
        # negative dips are the start of the dip

        dipCutoff = 3e3 # 3000 uV jumps are probably caused by artifact
        #pdb.set_trace()
        # peakutils works on a percentage of the range of values
        dipThresh = (dipCutoff - rowDiff.min()) / (rowDiff.max() - rowDiff.min())
        dipStarts = peakutils.indexes(-rowDiff, thres=dipThresh)
        dipEnds = []

        if dipStarts.any() and plotting:
            #pdb.set_trace()
            plt.figure()
            plt.plot(row)
            plt.plot(rowDiff)
            plt.plot(dipStarts, row.iloc[dipStarts], '*')
            plt.plot(dipEnds, row.iloc[dipEnds], '*')
            plt.show()

        if dipStarts.any():
            nDips = len(dipStarts)
            fixedRow = row.copy()
            maskRow = np.full(len(overflowMask.index), False, dtype = np.bool)
            for dipIdx, dipStartIdx in enumerate(dipStarts):
                try:
                    assert row.iloc[dipStartIdx - 1] > 8e3 # 8191 mV is equivalent to 32768
                    nextIdx = dipStarts[dipIdx + 1] if dipIdx < nDips - 1 else len(row)
                    thisSection = rowDiff.iloc[dipStartIdx:nextIdx].values
                    dipEndIdx = peakutils.indexes(thisSection, thres=dipThresh)
                    if not dipEndIdx and thisSection[-1] > dipCutoff:
                        # if the end is the peak, peakutils won't catch it; add it in manually
                        dipEndIdx = [len(thisSection)]
                    dipEndIdx = dipEndIdx[0] + dipStartIdx
                    dipEnds.append(dipEndIdx)
                    if dipEndIdx == nextIdx:
                        # avoid double counting the last point
                        dipEndIdx -= 1
                    assert dipEndIdx > dipStartIdx
                except:
                    continue

                maskRow[dipStartIdx:dipEndIdx] = True
                if fillMethod == 'average':
                    fixValue = (-rowDiff.iloc[dipStartIdx] + rowDiff.iloc[dipEndIdx]) / 2
                elif fillMethod == 'constant':
                    fixValue = 8191

                try:
                    #assert fixedRow.iloc[dipStartIdx:dipEndIdx].mean() < 5e3
                    fixedRow.iloc[dipStartIdx:dipEndIdx] = \
                        row.iloc[dipStartIdx:dipEndIdx].values + \
                        fixValue
                except Exception as e:
                    print(e)
                    continue

            if dipStarts.any() and plotting:
                plt.figure()
                plt.plot(row, label = 'Original')
                plt.plot(fixedRow, label = 'Fixed')
                plt.plot(dipStarts, row.iloc[dipStarts], '*')
                plt.plot(dipEnds, row.iloc[dipEnds], '*')
                plt.legend()
                plt.show()

            channelData.loc[:, idx] = fixedRow
            overflowMask.loc[:, idx] = maskRow

    print('\nFinished finding boolean overflow')
    return channelData, overflowMask

def fillInJumps(channelData, samp_per_s, smoothing_ms = 1, badThresh = 1e-3,
    consecLen = 30, nStdDiff = 20, nStdAmp = 20):
    #Allocate bad data mask as dict
    badMask = {'general':None,
        'perChannelAmp' : pd.DataFrame(False, index = channelData.index, columns = channelData.columns, dtype = np.bool).to_sparse(fill_value=False),
        'perChannelDer' : pd.DataFrame(False, index = channelData.index, columns = channelData.columns, dtype = np.bool).to_sparse(fill_value=False)
        }

    #per channel, only smooth a couple of samples
    shortSmoothKern = np.ones((5))
    # per channel look for abberantly large jumps
    cumDiff = pd.Series(np.zeros(len(channelData.index)), index = channelData.index)
    columnList = list(channelData.columns)
    nChan = len(columnList)
    for idx, row in channelData.iteritems():

        ch_idx  = columnList.index(idx)
        sys.stdout.write("Finding Signal Jumps: %d%%\r" % int((ch_idx + 1) * 100 / nChan))
        sys.stdout.flush()

        if len(row.index) < 3 * 10e4:
            rowVals = row.values
        else:
            rowVals =  row.values[:int(10 * channelData['samp_per_s'])]

        rowBar  = rowVals.mean()
        rowStd  = rowVals.std()
        #Look for abnormally high values in the first difference of each channel
        # how many standard deviations should we keep?
        maxAcceptable = rowBar + nStdAmp * rowStd
        minAcceptable = rowBar - nStdAmp * rowStd

        outliers = np.logical_or(row > maxAcceptable,row < minAcceptable)
        outliers = np.convolve(outliers, shortSmoothKern, 'same') > 0

        badMask['perChannelAmp'].loc[:, idx]=np.array(outliers, dtype = bool)

        # on the derivative of the data
        dRow = row.diff()
        dRow.fillna(0, inplace = True)
        if len(row.index) < 10 * channelData['samp_per_s']:
            dRowVals = dRow.values
        else:
            dRowVals =  dRow.values[:int(10 * channelData['samp_per_s'])]

        dRowBar  = dRowVals.mean()
        dRowStd  = dRowVals.std()
        dMaxAcceptable = dRowBar + nStdDiff * dRowStd
        dMinAcceptable = dRowBar - nStdDiff * dRowStd
        #pdb.set_trace()

        # append to previous list of outliers
        dOutliers = np.logical_or(dRow > dMaxAcceptable, dRow < dMinAcceptable)
        dOutliers = np.convolve(dOutliers, shortSmoothKern, 'same') > 0
        #pdb.set_trace()
        badMask['perChannelDer'].loc[:, idx]=np.array(dOutliers, dtype = bool)

        #add to the cummulative derivative (will check it at the end for flat periods signifying signal cutout)
        cumDiff = cumDiff + dRow.abs() / len(channelData.columns)

    # Look for unchanging signal across channels

    # convolve with step function to find consecutive
    # points where the derivative is identically zero across electrodes
    kern = np.ones((int(consecLen)))
    cumDiff = pd.Series(np.convolve(cumDiff.values, kern, 'same'))
    badFlat = cumDiff < badThresh

    #smooth out the bad data mask
    smoothKernLen = smoothing_ms * 1e-3 * samp_per_s
    smoothKern = np.ones((int(smoothKernLen)))

    badFlat = np.convolve(badFlat, smoothKern, 'same') > 0
    print('\nFinished finding abnormal signal jumps')

    for idx, row in channelData.iteritems():

        ch_idx  = columnList.index(idx)
        sys.stdout.write("Fixing Signal Jumps: %d%%\r" % int((ch_idx + 1) * 100 / nChan))
        sys.stdout.flush()
        #correct the row
        mask = np.logical_or(badMask['perChannelAmp'].loc[:, idx].to_dense(),
            badMask['perChannelDer'].loc[:, idx].to_dense())
        mask = np.logical_or(mask, badFlat)
        channelData.loc[:,idx] = replaceBad(row, mask, typeOpt = 'interp')

    badMask['general'] = pd.Series(np.array(badFlat, dtype = bool),
        index = channelData.index, dtype = np.bool).to_sparse(fill_value=False)
    print('\nFinished processing abnormal signal jumps')

    return channelData, badMask

def confirmTriggersPlot(peakIdx, dataSeries, fs, whichPeak = 0, nSec = 2):

    indent = peakIdx[whichPeak]

    dataSlice = slice(int(indent-.25*fs),int(indent+nSec*fs)) # 5 sec after first peak
    peakSlice = np.where(np.logical_and(peakIdx > indent - .25*fs, peakIdx < indent + nSec*fs))

    f = plt.figure()
    plt.plot(dataSeries.index[dataSlice] - indent, dataSeries.iloc[dataSlice])
    plt.plot(peakIdx[peakSlice] - indent, dataSeries.iloc[peakIdx[peakSlice]], 'r*')
    plt.show(block = False)

    f = plt.figure()
    sns.distplot(np.diff(peakIdx))
    plt.xlabel('distance between triggers (# samples)')
    plt.show(block = True)

def compareClocks(foundTime, expectedTime, thresh):
    timeMismatch = foundTime[:len(expectedTime)] - expectedTime

    expectedClockDrift = np.polyfit(x=range(len(expectedTime)), y=timeMismatch, deg=1)
    expectedTimeMismatchFun = np.poly1d(expectedClockDrift)
    #expectedTimeMismatch =
    # # TODO: FIXME by looking at older checkpoint, I accidentally got deleted

    correctedTimeMismatch = timeMismatch - expectedTimeMismatch
    whereOff = np.flatnonzero(abs(correctedTimeMismatch) > thresh)

    return correctedTimeMismatch, whereOff, expectedTimeMismatchFun

def getThresholdCrossings(dataSeries, thresh, absVal = False, edgeType = 'rising',
    fs = 3e4, iti = None):
    if absVal:
        dsToSearch = dataSeries.abs()
    else:
        dsToSearch = dataSeries
    nextDS = dsToSearch.shift(1)

    if edgeType == 'rising':
        crossMask = ((dsToSearch >= thresh) & (nextDS < thresh)) |\
            ((dsToSearch > thresh) & (nextDS <= thresh))
    else:
        crossMask = ((dsToSearch <= thresh) & (nextDS > thresh)) |\
            ((dsToSearch < thresh) & (nextDS >= thresh))

    crossIdx = dataSeries.index[crossMask]

    if iti is not None:
        itiWiggle = 0.05
        width = int(fs * iti * (1 - itiWiggle))
        rejectMask = pd.Series(crossIdx).diff() < width
        crossMask[crossIdx[rejectMask]] = False
        crossIdx = crossIdx[~rejectMask]
        #pdb.set_trace()
    return crossIdx, crossMask

def findTrains(dataSeries, peakIdx, iti, fs, minTrainLength, maxDistance = 1.5, maxTrain = False):
    peakMask = dataSeries.index.isin(peakIdx)
    foundTime = pd.Series((dataSeries.index[peakIdx] - dataSeries.index[peakIdx[0]]) / fs)
    #
    # identify trains of peaks
    itiWiggle = 0.05
    minPeriod = iti * (1 - itiWiggle)

    peakDiff = foundTime.diff()
    #pdb.set_trace()
    peakDiff.iloc[0] = iti * 2e3 #fudge it so that the first one is taken
    trainStartIdx = foundTime.index[peakDiff > (iti * maxDistance)]
    trainStarts = foundTime[trainStartIdx]

    peakFwdDiff = foundTime.diff(periods = -1).abs()
    peakFwdDiff.iloc[-1] = iti * 2e3 #fudge it so that the last one is taken
    trainEndIdx = foundTime.index[peakFwdDiff > (iti * maxDistance)]
    trainEnds = foundTime[trainEndIdx]

    #pdb.set_trace()
    trainLengths = (trainEnds.values - trainStarts.values)
    validTrains = trainLengths > minTrainLength
    keepPeaks = pd.Series(False, index = foundTime.index)
    nTrains = 0
    for idx, idxIntoPeaks in enumerate(trainStartIdx):
        #print('{}, {}'.format(idx, idxIntoPeaks))
        #pdb.set_trace()
        thisMeanPeriod = peakDiff.loc[idxIntoPeaks:trainEndIdx[idx]].iloc[1:].mean()
        if validTrains[idx] and thisMeanPeriod > minPeriod:
            keepPeaks.loc[idxIntoPeaks:trainEndIdx[idx]] = True
            nTrains += 1
            #pdb.set_trace()
            if maxTrain:
                if nTrains > maxTrain: break;
        else:
            validTrains[idx] = False

    #pdb.set_trace()
    trainStartPeaks = peakIdx[trainStartIdx[validTrains]]
    trainEndPeaks = peakIdx[trainEndIdx[validTrains]]
    foundTime = foundTime[keepPeaks]
    peakIdx = peakIdx[keepPeaks]

    return peakIdx, trainStartPeaks,trainEndPeaks

def getTriggers(dataSeries, iti = .01, fs = 3e4, thres = 2.58,
    edgeType = 'rising',
    minAmp = None,
    minTrainLength = None,
    expectedTime = None, keep_max = True, plotting = False):
    # iti: expected inter trigger interval

    # minimum distance between triggers (units of samples), 5% wiggle room
    itiWiggle = 0.05
    width = int(fs * iti * (1 - itiWiggle))
    # first difference of triggers
    triggersPrime = dataSeries.diff()
    triggersPrime.fillna(0, inplace = True)
    # z-score the derivative
    triggersPrime = pd.Series(stats.zscore(triggersPrime), index = triggersPrime.index)

    if plotting:
        f = plt.figure()
        plt.plot(triggersPrime)
        plt.show(block = False)

    if edgeType == 'falling':
        triggersPrime = - triggersPrime

    # moments when camera capture occured
    triggersPrimeVals = triggersPrime.values.squeeze()
    peakIdx = peakutils.indexes(triggersPrimeVals, thres=thres,
        min_dist=width, thres_abs=True, keep_max = keep_max)

    # check that the # of triggers matches the number of frames
    if expectedTime is not None:
        foundTime = (dataSeries.index[peakIdx] - dataSeries.index[peakIdx[0]]) / fs
        if len(foundTime) > len(expectedTime):
            # more triggers found than frames in the video

            # check for inconsistencies beyond normal clock drift
            foundTime = foundTime[:len(expectedTime)]
            correctedTimeMismatch, whereOff, expectedTimeMismatchFun = compareClocks(
                foundTime, expectedTime, 1e-3)

            if whereOff.any():
                raise Exception('triggers do not match simi record')

            peakIdx = peakIdx[:len(expectedTime)]

        elif len(foundTimes) > len(expectedTime):
            raise Exception('table contains {} entries and there are {} triggers'.format(len(raw.index), len(trigTimes)))

    if minAmp is not None:
        triggersZScore = pd.Series(stats.zscore(dataSeries), index = dataSeries.index)
        insetIntoTrig = int(fs * iti * 1e-2)
        peakIdx = np.array(peakIdx[triggersZScore.loc[peakIdx+insetIntoTrig] > minAmp])
        #pdb.set_trace()

    if minTrainLength is not None:
        peakMask = dataSeries.index.isin(peakIdx)
        foundTime = pd.Series((dataSeries.index[peakIdx] - dataSeries.index[peakIdx[0]]) / fs)
        #
        # identify trains of peaks
        peakDiff = foundTime.diff()
        peakDiff.iloc[0] = iti * 2e3 #fudge it so that the first one is taken
        #pdb.set_trace()
        trainStartIdx = foundTime.index[peakDiff > (iti * 2)]
        trainStarts = foundTime[trainStartIdx]

        peakFwdDiff = foundTime.diff(periods = -1).abs()
        peakFwdDiff.iloc[-1] = iti * 2e3 #fudge it so that the last one is taken
        trainEndIdx = foundTime.index[peakFwdDiff > (iti * 2)]
        trainEnds = foundTime[trainEndIdx]

        trainLengths = (trainEnds.values - trainStarts.values)
        validTrains = trainLengths > minTrainLength
        keepPeaks = pd.Series(False, index = foundTime.index)
        nTrains = 0
        maxTrain = 3
        for idx, idxIntoPeaks in enumerate(trainStartIdx):
            #print('{}, {}'.format(idx, idxIntoPeaks))
            if validTrains[idx]:
                keepPeaks.loc[idxIntoPeaks:trainEndIdx[idx]] = True
                nTrains += 1
                #pdb.set_trace()
                if nTrains > maxTrain: break;

        #pdb.set_trace()
        foundTime = foundTime[keepPeaks]
        peakIdx = peakIdx[keepPeaks]

    if plotting:
        confirmTriggersPlot(peakIdx, dataSeries, fs)
        if minAmp is not None:
            plt.plot(triggersZScore)
            plt.plot(triggersZScore[peakIdx], '*')
            plt.title('Z scored trace')
            plt.show(block = True)

    return peakIdx

def getCameraTriggers(simiData, thres = 2.58, expectedTime = None, plotting = False):
    # sample rate
    fs = simiData['samp_per_s']
    # get camera triggers
    triggers = simiData['data']
    # get Triggers
    peakIdx = getTriggers(triggers, iti = .01, fs = fs, thres = thres,
        edgeType = 'falling', expectedTime = expectedTime, plotting = plotting)

    # get time of first simi frame in NSP time:
    trigTimes = simiData['t'][peakIdx]
    # timeOffset = trigTimes[0]
    return peakIdx, trigTimes

def getAngles(anglesFile,
    trigTimes = None,
    simiData = None, thres = None,
    selectHeaders = None, selectTime = None,
    reIndex = None, lowCutoff = None):

    raw = pd.read_table(anglesFile, index_col = 0, skiprows = [1])

    if trigTimes is None:
        peakIdx, trigTimes = getCameraTriggers(simiData, thres = thres, expectedTime = raw.index)

    headings = sorted(raw.columns) # get column names

    # reorder alphabetically by columns
    raw = raw.reindex(columns = headings)

    proc = pd.DataFrame(raw.values, columns = raw.columns, index = trigTimes)

    proc.index = np.around(proc.index, 3)
    if selectTime:
        proc = proc.loc[slice(selectTime[0], selectTime[1]), :]

    if lowCutoff is not None:
        fr = 1 / 0.01
        Wn = 2 * lowCutoff / fr
        b, a = signal.butter(12, Wn, analog=False)
        for column in proc:
            proc.loc[:, column] = signal.filtfilt(b, a, proc.loc[:, column])

    return proc

def getTensTrigs(tensThresh, ampThresh, channelData, referenceTimes, plotting = False):

    iti = .1
    minTrainLength = 5 * iti * 1e3

    peakIdx = getTriggers(channelData['data']['TensSync'], iti = iti, fs = channelData['samp_per_s'],
        thres = tensThresh, keep_max = False, minAmp = ampThresh, plotting = plotting)
    peakMask = channelData['data'].index.isin(peakIdx)

    peakTimes = channelData['t'][peakMask] * 1e3

    # identify trains of peaks
    peakDiff = peakTimes.diff()
    peakDiff.iloc[0] = iti * 2e3 #fudge it so that the first one is taken
    trainStartIdx = peakTimes.index[peakDiff > iti * 1.1 * 1e3]
    trainStarts = peakTimes[trainStartIdx]

    peakFwdDiff = peakTimes.diff(periods = -1).abs()
    peakFwdDiff.iloc[-1] = iti * 2e3 #fudge it so that the first one is taken
    trainEndIdx = peakTimes.index[peakFwdDiff > iti * 1.1 * 1e3]
    trainEnds = peakTimes[trainEndIdx]

    if plotting:
        plt.plot(peakTimes, peakTimes ** 0, 'mo', label = 'unaligned OpE TENS Pulses')

    trainLengths = (trainEnds.values - trainStarts.values)
    validTrains = trainLengths > minTrainLength
    keepPeaks = pd.Series(False, index = peakTimes.index)
    nTrains = 0
    maxTrain = 3
    for idx, idxIntoPeaks in enumerate(trainStartIdx):
        #print('{}, {}'.format(idx, idxIntoPeaks))
        if validTrains[idx]:
            keepPeaks.loc[idxIntoPeaks:trainEndIdx[idx]] = True
            nTrains += 1
            #pdb.set_trace()
            if nTrains > maxTrain: break;

    peakTimes = peakTimes[keepPeaks]
    peakIdx = np.array(peakTimes.index)

    if plotting:
        plt.plot(peakTimes, peakTimes ** 0, 'co', label = 'unaligned OpE TENS Pulses (valid Trains)')
        plt.plot(trainStarts, trainStarts ** 0 -1, 'go', label = 'unaligned OpE TENS Train Start')
        plt.plot(trainEnds, trainEnds ** 0 +1, 'ro', label = 'unaligned OpE TENS Train End')
        plt.legend()
        plt.show()

    nDetectedMismatch = len(peakIdx) - len(referenceTimes)
    print('peakIdx is {} samples long, referenceTimes is {} samples long'.format(len(peakIdx), len(referenceTimes)))
    if nDetectedMismatch != 0:
        if nDetectedMismatch > 0:
            shorterTimes = referenceTimes
            longerTimes = peakTimes
        else:
            shorterTimes = peakTimes
            longerTimes = referenceTimes

        rmsDiff = np.inf
        keepTimes = np.array(longerTimes[:len(shorterTimes)])
        for correctTimes in itertools.combinations(longerTimes, len(shorterTimes)):
            #pdb.set_trace()
            alignedCorrectTimes = pd.Series(correctTimes) - correctTimes[0]
            alignedShorterTimes = shorterTimes - shorterTimes.iloc[0]
            newRmsDiff = np.mean((alignedCorrectTimes.values - alignedShorterTimes.values) ** 2)
            if newRmsDiff < rmsDiff:
                keepTimes = correctTimes
        #pdb.set_trace()

        if nDetectedMismatch > 0:
            peakTimes = peakTimes[peakTimes.isin(keepTimes)]
        else:
            referenceTimes = referenceTimes[referenceTimes.isin(keepTimes)]

    peakIdx = np.array(peakTimes.index)
    peakMask = channelData['data'].index.isin(peakIdx)

    if plotting:
        plt.plot(peakTimes, peakTimes ** 0, 'co', label = 'unaligned OpE TENS Pulses')
        plt.plot(trainStarts, trainStarts ** 0 -1, 'go', label = 'unaligned OpE TENS Train Start')
        plt.plot(trainEnds, trainEnds ** 0 +1, 'ro', label = 'unaligned OpE TENS Train End')
        plt.plot(referenceTimes, referenceTimes ** 0, 'ko', label = 'unaligned INS TENS Pulses')
        plt.legend()
        plt.show()
        #pdb.set_trace()

    return peakIdx, peakTimes / 1e3, peakMask, referenceTimes

def optimizeTensThresholds(diffThresholds, ampThresholds, channelData, plotting = False):
    dv, av = np.meshgrid(diffThresholds, ampThresholds, sparse=False, indexing='ij')
    mismatch = pd.DataFrame(np.nan, index = diffThresholds, columns = ampThresholds)
    for i in range(len(diffThresholds)):
        for j in range(len(ampThresholds)):
            # treat xv[i,j], yv[i,j]
            print('Evaluating {}, {}'.format(i,j))
            peakIdx, openEphysTensTimes, peakMask = getTensTrigs(dv[i,j], av[i,j], channelData)
            if len(peakIdx) != len(insTensTimes[trialIdx]):
                #no good, not the same number
                mismatch.loc[diffThresholds[i], ampThresholds[j]] = 1e9
            else:
                # coarsely align
                minNSamp = min(len(peakIdx) , len(insTensTimes[trialIdx]))
                coarseOpenEphysTensTimes = (openEphysTensTimes.iloc[:minNSamp] - openEphysTensTimes.iloc[0]) * 1e3
                coarseInsTensTimes = insTensTimes[trialIdx][:minNSamp] - insTensTimes[trialIdx][0]
                coarseDiff = (coarseOpenEphysTensTimes - coarseInsTensTimes).abs().sum()
                mismatch.loc[diffThresholds[i], ampThresholds[j]] = coarseDiff
    if plotting:
        ax = sns.heatmap(mismatch, vmax = 1e4, cmap="YlGnBu")
        plt.show()
    bestDiff = mismatch.idxmin(axis = 0).iloc[0]
    bestAmp = mismatch.idxmin(axis = 1).iloc[0]
    return mismatch, bestDiff, bestAmp

def loadAngles(folderPath, fileName, kinAngleOpts = {
    'ns5FileName' : '',
    'selectHeaders' : None,
    'reIndex' : None,
    'flip' : None,
    'lowCutoff': None
    }, forceRecalc = False):

    setPath = os.path.join(folderPath, fileName + '.h5')
    if not forceRecalc:
        try:
            angles = pd.read_hdf(setPath, 'angles')
        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('Angles not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:

        motorDataPath = os.path.join(folderPath,
            kinAngleOpts['ns5FileName'] + '_eventInfo.h5')
        motorData= pd.read_hdf(motorDataPath, 'motorData')

        simiData = {
        'samp_per_s' : 3e4,
        'data' : motorData['simiTrigs'],
        't' : np.array(motorData['simiTrigs'].index) / 3e4
        }
        anglesFile = os.path.join(folderPath,fileName + '-angles.txt')

        # keep trying thresholds until the # of trigs matches the # of samples
        for thres in [3.29, 2.58, 2.33, 2.05]:
            try:
                angles = getAngles(anglesFile, simiData = simiData, thres = thres, selectHeaders = kinAngleOpts['selectHeaders'])
                break
            except Exception:
                traceback.print_exc()

        angles.to_hdf(setPath, 'angles')

    return angles

def getKinematics(kinematicsFile,
    trigTimes = None,
    simiData = None, thres = None,
    selectHeaders = None, selectTime = None,
    flip = None, reIndex = None, lowCutoff = None):
    # TODO: unify with mujoco stuff
    raw = pd.read_table(kinematicsFile, index_col = 0, skiprows = [1])

    if trigTimes is None:
        peakIdx, trigTimes = getCameraTriggers(simiData, thres = thres, expectedTime = raw.index)

    raw = pd.DataFrame(raw.values, index = trigTimes, columns = raw.columns)
    raw.index = np.around(raw.index, 3)

    if selectTime:
        raw = raw.loc[slice(selectTime[0], selectTime[1]), :]

    headings = sorted(raw.columns) # get column names
    coordinates = ['x', 'y', 'z']

    # reorder alphabetically by columns
    raw = raw.reindex(columns = headings)
    #pdb.set_trace()

    if reIndex is not None:
        uniqueHeadings = set([name[:-2] for name in headings])
        oldIndex = raw.columns
        newIndex = np.array(oldIndex)
        for name in uniqueHeadings:
            for reindexPair in reIndex:
                mask = [
                    oldIndex.str.contains(name + ' ' + reindexPair[0]),
                    oldIndex.str.contains(name + ' ' + reindexPair[1])
                    ]
                temp = newIndex[mask[0]]
                newIndex[mask[0]] = newIndex[mask[1]]
                newIndex[mask[1]] = temp
        raw.columns = newIndex
        headings = sorted(newIndex)
        raw = raw.reindex_axis(headings, axis = 1)

    if flip is not None:
        uniqueHeadings = set([name[:-2] for name in headings])
        for name in uniqueHeadings:
            for flipAxis in flip:
                raw.loc[:,name + ' ' + flipAxis] = -raw.loc[:,name + ' ' + flipAxis]
    # create multiIndex column names

    if selectHeaders is not None:
        expandedHeadings = [
            name + ' X' for name in selectHeaders
        ] + [
            name + ' Y' for name in selectHeaders
        ] + [
            name + ' Z' for name in selectHeaders
        ]
        raw = raw[sorted(expandedHeadings)]
        indexPD = pd.MultiIndex.from_product([sorted(selectHeaders),
            coordinates], names=['joint', 'coordinate'])
    else:
        uniqueHeadings = set([name[:-2] for name in headings])
        indexPD = pd.MultiIndex.from_product([sorted(uniqueHeadings),
            coordinates], names=['joint', 'coordinate'])

    proc = pd.DataFrame(raw.values, columns = indexPD, index = raw.index)

    if lowCutoff is not None:
        fr = 1 / 0.01
        Wn = 2 * lowCutoff / fr
        b, a = signal.butter(12, Wn, analog=False)
        for column in proc:
            proc.loc[:, column] = signal.filtfilt(b, a, proc.loc[:, column])

    return proc

def loadKinematics(folderPath, fileName, kinPosOpts = {
    'ns5FileName' : '',
    'selectHeaders' : None,
    'reIndex' : None,
    'flip' : None,
    'lowCutoff': None
    },
    forceRecalc = False):

    setPath = os.path.join(folderPath, fileName + '.h5')
    if not forceRecalc:
        try:
            kinematics = pd.read_hdf(setPath, 'kinematics')
        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('kinematics not pickled. Recalculating...')
            forceRecalc = True
    if forceRecalc:
        motorData   = pd.read_pickle(os.path.join(folderPath,
            kinPosOpts['ns5FileName'] + '_motorData.pickle'))
        simiData = {
        'samp_per_s' : 3e4,
        'data' : motorData['simiTrigs'],
        't' : np.array(motorData['simiTrigs'].index) / 3e4
        }
        peakIdx, trigTimes = getCameraTriggers(simiData)
        kinematicsFile = os.path.join(folderPath,fileName + '-positions.txt')

        # keep trying thresholds until the # of trigs matches the # of samples
        for thres in [3.29, 2.58, 2.33, 2.05]:
            try:
                kinematics = getKinematics(kinematicsFile, simiData = simiData, thres = thres, selectHeaders = kinPosOpts['selectHeaders'],
                    flip = kinPosOpts['flip'], reIndex = kinPosOpts['reIndex'], lowCutoff = kinPosOpts['lowCutoff'])

                break
            except Exception:
                traceback.print_exc()

        kinematics.to_hdf(setPath, 'kinematics')

    return kinematics

def getGaitEvents(trigTimes, simiTable, whichColumns =  ['ToeUp_Left Y', 'ToeDown_Left Y'],
    plotting = False, fudge = 2, CameraFs = 100):
    # NSP time of first camera trigger
    timeOffset = trigTimes[0]
    simiTable['Time'] = simiTable['Time'] + timeOffset
    simiDf = pd.DataFrame(simiTable[whichColumns])
    simiDf = simiDf.notnull()
    simiDf['simiTime'] = simiTable['Time']
    # max time recorded on NSP
    # Note: Clocks drift slightly between Simi Computer and NSP. Allow timeMax
    # to go a little bit beyond so that we don't drop the last simi frame, i.e.
    # add fudge times the time increment TODO: FIX THIS.

    # depeding on which acquisition system stopped first
    if trigTimes[-1] > simiTable['Time'].max():
        # NSP stopped after SIMI
        timeMax = simiTable['Time'].max() + fudge/CameraFs
        trigTimes = trigTimes[:simiDf.shape[0]]
    else:
        # SIMI stopped after NSP
        timeMax = trigTimes[-1] + fudge/CameraFs
        simiDf.drop(simiDf[simiDf['simiTime'] >= timeMax].index, inplace = True)


    debugging = False
    if debugging:
        nNSPTrigs = len(trigTimes)
        simiHist, simiBins = np.histogram(np.diff(simiDf['simiTime'].values))
        totalSimiTime = np.diff(simiDf['simiTime'].values).sum()
        nSimiTrigs = len(simiDf['simiTime'].values)
        trigHist, trigBins = np.histogram(np.diff(trigTimes))
        totalNSPTime = np.diff(trigTimes).sum()
        np.savetxt('getGaitEvents-trigTimes.txt', trigTimes, fmt = '%4.4f', delimiter = ' \n')
        pdb.set_trace()
    #
    simiDf['NSPTime'] = pd.Series(trigTimes, index = simiDf.index)

    simiDfPadded = deepcopy(simiDf)
    for idx, row in simiDfPadded.iterrows():

        if row[whichColumns[0]]:
            newSimiTime = row['simiTime'] + 1/CameraFs
            newNSPTime = row['NSPTime'] + 1/CameraFs
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: False, whichColumns[1]: False, 'simiTime': row['simiTime'] - 1e-6, 'NSPTime': row['NSPTime'] - 1e-6}), ignore_index = True)
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: True, whichColumns[1]: False, 'simiTime': newSimiTime -1e-6, 'NSPTime': newNSPTime -1e-6}), ignore_index = True)
            #simiDfPadded = simiDfPadded.append(pd.Series({'ToeUp_Left Y': False, 'ToeDown_Left Y': False, 'simiTime': newSimiTime + 1e-6, 'NSPTime': newNSPTime + 1e-6}), ignore_index = True)

        if row[whichColumns[1]]:
            newSimiTime = row['simiTime'] + 1/CameraFs
            newNSPTime = row['NSPTime'] + 1/CameraFs
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: False, whichColumns[1]: False, 'simiTime': row['simiTime'] - 1e-6, 'NSPTime': row['NSPTime'] - 1e-6}), ignore_index = True)
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: False, whichColumns[1]: True, 'simiTime': newSimiTime -1e-6, 'NSPTime': newNSPTime -1e-6}), ignore_index = True)
            #simiDfPadded = simiDfPadded.append(pd.Series({'ToeUp_Left Y': False, 'ToeDown_Left Y': False, 'simiTime': newSimiTime + 1e-6, 'NSPTime': newNSPTime + 1e-6}), ignore_index = True)

    simiDfPadded.sort_values('simiTime', inplace = True)
    down = (simiDfPadded[whichColumns[1]].values * 1)
    up = (simiDfPadded[whichColumns[0]].values * 1)
    gait = up.cumsum() - down.cumsum()

    if plotting:
        f = plt.figure()
        plt.plot(simiDfPadded['simiTime'], gait)
        plt.plot(simiDfPadded['simiTime'], down, 'g*')
        plt.plot(simiDfPadded['simiTime'], up, 'r*')
        ax = plt.gca()
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([6.8, 7.8])
        plt.show(block = False)

    gaitLabelFun = interpolate.interp1d(simiDfPadded['simiTime'], gait, bounds_error = False, fill_value = 'extrapolate')
    downLabelFun = interpolate.interp1d(simiDfPadded['simiTime'], down, bounds_error = False, fill_value = 0)
    upLabelFun   = interpolate.interp1d(simiDfPadded['simiTime'], up  , bounds_error = False, fill_value = 0)
    # TODO: check that the padded Df oesn't inerfere with thiese labelrs
    simiDf['Labels'] = pd.Series(['Swing' if x > 0 else 'Stance' for x in gaitLabelFun(simiDf['simiTime'])], index = simiDf.index)
    return simiDf, gaitLabelFun, downLabelFun, upLabelFun

def assignLabels(timeVector, lbl, fnc, CameraFs = 100, oversizeWindow = None):
    dt = np.mean(np.diff(timeVector))

    if dt < 1/CameraFs:
        # sampling faster than the original data! Interpolate!
        labels = pd.Series([lbl if x > 0 else 'Neither' for x in fnc(timeVector)])
        #
    else:
        #sampling slower than the original data! Histogram!
        # create a false time vector with 10x as many samples as the camera sample rate
        pseudoTime = np.arange(timeVector[0], timeVector[-1] + dt, 0.1/CameraFs)
        oversampledFnc = fnc(pseudoTime)
        #pdb.set_trace()
        if oversizeWindow is None:
            timeBins = np.append(timeVector, timeVector[-1] + dt)
            #
            histo,_ = np.histogram(pseudoTime[oversampledFnc > 0], timeBins)
            labels = pd.Series([lbl if x > 0 else 'Neither' for x in histo])
        else:
            binInterval = dt
            binWidth = oversizeWindow
            timeStart = timeVector[0] - binWidth / 2

            timeEnd = timeVector[-1] + binWidth / 2
            #pdb.set_trace()
            mat, binCenters, binLeftEdges = binnedEvents(
                [pseudoTime[oversampledFnc > 0]], [0], [0],
                binInterval, binWidth, timeStart, timeEnd)
            mat = np.squeeze(mat)
            """
            It's possible mat will have an extra entry, because we don't know
            what the original timeEnd was.
            timeVector[-1] <= originalTimeEnd - binWidth/2, because we're not
            guaranteed that the time window is divisible by the binInterval.
            Therefore, delete the last entry if need be:
            """
            if len(mat) != len(timeVector):
                mat = mat[:-1]
            labels = pd.Series([lbl if x > 0 else 'Neither' for x in mat])
            #pdb.set_trace()
    return labels

def getSpectrogram(channelData, elec_ids, samp_per_s, start_time_s, dataT, winLen_s, stepLen_s = 0.02, R = 20, fr_start = None, fr_stop = None, whichChan = 1, plotting = False):

    Fs = samp_per_s
    nChan = channelData.shape[1]
    nSamples = channelData.shape[0]

    delta = 1 / Fs

    winLen_samp = int(winLen_s * Fs)
    stepLen_samp = int(stepLen_s * Fs)

    NFFT = nextpowof2(winLen_samp)
    nw = winLen_s * R # time bandwidth product based on 0.1 sec windows and 200 Hz bandwidth
    nTapers = m.ceil(nw / 2) # L < nw - 1
    nWindows = m.floor((nSamples - NFFT + 1) / stepLen_samp)

    fr_samp = int(NFFT / 2) + 1
    fr = np.arange(fr_samp) * samp_per_s / (2 * fr_samp)

    #pdb.set_trace()
    if fr_start is not None:
        fr_start_idx = np.where(fr > fr_start)[0][0]
    else:
        fr_start_idx = 0

    if fr_stop is not None:
        fr_stop_idx = np.where(fr < fr_stop)[0][-1]
    else:
        fr_stop_idx = -1

    #pdb.set_trace()
    fr = fr[fr_start_idx : fr_stop_idx]
    fr_samp = len(fr)

    columnList = list(channelData.columns)
    if HASLIBTFR:
        origin = 'libtfr'
        #pdb.set_trace()
        t = start_time_s + np.arange(nWindows + 1) * stepLen_s + NFFT / Fs * 0.5
        spectrum = np.zeros((nChan, nWindows + 1, fr_samp))
        # generate a transform object with size equal to signal length and ntapers tapers
        D = libtfr.mfft_dpss(NFFT, nw, nTapers, NFFT)
        #pdb.set_trace()
        for col,signal in channelData.iteritems():
            #pdb.set_trace()
            idx = columnList.index(col)
            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
                print("Running getSpectrogram: %d%%" % int(idx * 100 / nChan + 1), end = endChar)
            else:
                print("Running getSpectrogram: %d%%" % int(idx * 100 / nChan + 1))


            P_libtfr = D.mtspec(signal, stepLen_samp).transpose()
            P_libtfr = P_libtfr[np.newaxis,:,fr_start_idx:fr_stop_idx]

            spectrum[idx,:,:] = P_libtfr
    else:
        origin = 'scipy'
        spectrum = np.zeros((nChan, nWindows, fr_samp))
        for col,signal in channelData.iteritems():
            idx = columnList.index(col)
            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
                print("Running getSpectrogram: %d%%" % int(idx * 100 / nChan + 1), end = endChar)
            else:
                print("Running getSpectrogram: %d%%" % int(idx * 100 / nChan + 1))


            overlap_samp = NFFT - stepLen_samp
            _, t, P_scipy = scipy.signal.spectrogram(signal,mode='magnitude',
                window = 'boxcar', nperseg = NFFT, noverlap = overlap_samp, fs = Fs)
            P_scipy = P_scipy.transpose()[np.newaxis,:,fr_start_idx:fr_stop_idx]
            spectrum[idx,:,:] = P_scipy
        t = start_time_s + t

    if plotting:
        ch_idx  = elec_ids.index(whichChan)

        #TODO: implement passing elecID to plotSpectrum
        #hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]

        P = spectrum[ch_idx,:,:]
        #pdb.set_trace()
        plotSpectrum(P, Fs, start_time_s, dataT[-1], fr = fr, t = t, show = True, fr_start = fr_start, fr_stop = fr_stop)

    #pdb.set_trace()

    return {'PSD' : pd.Panel(spectrum, items = elec_ids, major_axis = t, minor_axis = fr),
            'fr' : fr,
            't' : t,
            'origin' : origin
            }

def nextpowof2(x):
    return 2**(m.ceil(m.log(x, 2)))

def cleanNEVSpikes(spikes, badData):
    pass

def replaceBad(dfSeries, mask, typeOpt = 'nans'):
    dfSeries[mask] = float('nan')
    if typeOpt == 'nans':
        pass
    elif typeOpt == 'interp':
        dfSeries.interpolate(method = 'linear', inplace = True)
        if dfSeries.isnull().any(): # For instance, if begins with bad data, there is nothing there to linearly interpolate
            dfSeries.fillna(method = 'backfill', inplace = True)
            dfSeries.fillna(method = 'ffill', inplace = True)

    return dfSeries

def plotChan(channelData, dataT, whichChan, recordingUnits = 'uV', electrodeLabel = '', label = " ", mask = None, maskLabel = " ",
    show = False, prevFig = None, zoomAxis = True, timeRange = (0,-1)):
    # Plot the data channel
    ch_idx  = whichChan

    #hdr_idx = channelData['ExtendedHeaderIndices'][channelData['elec_ids'].index(whichChan)]
    if not prevFig:
        #plt.figure()
        f, ax = plt.subplots()
    else:
        f = prevFig
        ax = prevFig.axes[0]

    #pdb.set_trace()
    #channelDataForPlotting = channelData.drop(['Labels', 'LabelsNumeric'], axis = 1) if 'Labels' in channelData.columns else channelData
    channelDataForPlotting = channelData.loc[:,ch_idx]
    tMask = np.logical_and(dataT > timeRange[0], dataT < timeRange[1])
    tSlice = slice(np.flatnonzero(tMask)[0], np.flatnonzero(tMask)[-1])
    tPlot = dataT[tSlice]

    #pdb.set_trace()
    ax.plot(tPlot, channelDataForPlotting[tSlice], label = label)

    if np.any(mask):
        for idx, thisMask in enumerate(mask):
            thisMask = thisMask[tSlice]
            ax.plot(tPlot[thisMask], channelDataForPlotting[tSlice][thisMask], 'o', label = maskLabel[idx])
    #pdb.set_trace()
    #channelData['data'][ch_idx].fillna(0, inplace = True)

    if zoomAxis:
        ax.set_xlim((tPlot[0], tPlot[-1]))
        ax.set_ylim((min(channelDataForPlotting[tSlice]),max(channelDataForPlotting[tSlice])))

    ax.locator_params(axis = 'y', nbins = 20)
    plt.xlabel('Time (s)')
    plt.ylabel("Extracellular voltage (" + recordingUnits + ")")
    plt.title(electrodeLabel)
    plt.tight_layout()
    if show:
        plt.show(block = False)

    return f, ax

def plotChanWithSpikesandStim(\
    spikes, spikesStim, channelData,
    spikeChanName, stimChanName,
    startTimeS, dataTimeS
    ):

    stimChIdx = spikesStim['ChannelID'].index(stimChanName)
    spikeChIdx = spikes['ChannelID'].index(spikeChanName)

    unitsOnThisChan = np.unique(spikes['Classification'][spikeChIdx])
    mask = [np.full(len(channelData['data'].index), False) for i in unitsOnThisChan]
    maskLabel = [" " for i in unitsOnThisChan]

    stimUnitsOnThisChan = np.unique(spikesStim['Classification'][stimChIdx])
    stimMask = [np.full(len(channelData['data'].index), False) for i in stimUnitsOnThisChan]
    stimMaskLabel = [" " for i in stimUnitsOnThisChan]

    try:
        spikeSnippetLen = spikes['Waveforms'][0].shape[1] / spikes['basic_headers']['TimeStampResolution']
    except:
        spikeSnippetLen = 64 / spikes['basic_headers']['TimeStampResolution']

    for unitIdx, unitName in enumerate(unitsOnThisChan):
        unitMask = spikes['Classification'][spikeChIdx] == unitName
        timeMask = np.logical_and(spikes['TimeStamps'][spikeChIdx] > startTimeS, spikes['TimeStamps'][spikeChIdx] < startTimeS + dataTimeS)
        theseTimes = spikes['TimeStamps'][spikeChIdx][np.logical_and(unitMask, timeMask)]
        for spikeTime in theseTimes:
            timeMaskContinuous = np.logical_and(channelData['t'] > spikeTime, channelData['t'] < spikeTime + spikeSnippetLen)
            mask[unitIdx][timeMaskContinuous] = True
            maskLabel[unitIdx] = 'unit %d' % unitName

    try:
        spikeSnippetLen = spikesStim['Waveforms'][0].shape[1] / spikesStim['basic_headers']['TimeStampResolution']
    except:
        spikeSnippetLen = 64 / spikes['basic_headers']['TimeStampResolution']

    for unitIdx, unitName in enumerate(stimUnitsOnThisChan):
        unitMask = spikesStim['Classification'][stimChIdx] == unitName
        timeMask = np.logical_and(spikesStim['TimeStamps'][stimChIdx] > startTimeS, spikesStim['TimeStamps'][stimChIdx] < startTimeS + dataTimeS)
        #pdb.set_trace()
        theseTimes = spikesStim['TimeStamps'][stimChIdx][np.logical_and(unitMask, timeMask)]
        for spikeTime in theseTimes:
            timeMaskContinuous = np.logical_and(channelData['t'] > spikeTime, channelData['t'] < spikeTime + spikeSnippetLen)
            stimMask[unitIdx][timeMaskContinuous] = True
            stimMaskLabel[unitIdx] = 'stim category %d' % unitIdx
        #theseTimes = spikes['TimeStamps'][spikeChIdx][unitMask]
    plotChan(channelData['data'], channelData['t'].values, spikeChanName,
        mask = mask + stimMask, maskLabel = maskLabel + stimMaskLabel,
        timeRange = (startTimeS, startTimeS + dataTimeS))
    plt.legend()
    plt.show()

def pdfReport(cleanData, origData, badData = None,
    pdfFilePath = 'pdfReport.pdf', spectrum = False, cleanSpectrum = None,
    origSpectrum = None, fr_start = 5, fr_stop = 3000, nSecPlot = 30):

    with matplotlib.backends.backend_pdf.PdfPages(pdfFilePath) as pdf:
        nChan = cleanData['data'].shape[1]

        if spectrum:
            fr = clean_data_spectrum['fr']
            t = clean_data_spectrum['t']
            Fs = origData['samp_per_s']

        columnList = list(cleanData['data'].columns)
        for idx, row in cleanData['data'].iteritems():
            ch_idx  = columnList.index(idx)
            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
                print("Running pdfReport: %d%%" % int((ch_idx + 1) * 100 / nChan), end = endChar)
            else:
                print("Running pdfReport: %d%%" % int((ch_idx + 1) * 100 / nChan))

            hdr_idx = cleanData['ExtendedHeaderIndices'][cleanData['elec_ids'].index(idx)]
            electrodeLabel = cleanData['extended_headers'][hdr_idx]['ElectrodeLabel']

            f,_ = plotChan(origData, cleanData['t'], idx,
                electrodeLabel = electrodeLabel, label = 'Raw data', zoomAxis = False,
                mask = None, show = False, timeRange = (cleanData['start_time_s'],
                (cleanData['start_time_s'] + nSecPlot) ))

            plotChan(cleanData['data'], cleanData['t'], idx, electrodeLabel = electrodeLabel,
                mask = [badData["general"].to_dense(), badData["perChannelAmp"].loc[:,idx].to_dense(),
                badData["perChannelDer"].loc[:,idx].to_dense(), badData["overflow"].loc[:,idx].to_dense()],
                label = 'Clean data', timeRange = (cleanData['start_time_s'],
                cleanData['start_time_s'] + nSecPlot),
                maskLabel = ["Flatline Dropout", "Amp Out of Bounds Dropout",
                "Derrivative Out of Bounds Dropout", "Overflow"], show = False,
                prevFig = f)

            plt.tight_layout()
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

            if spectrum:
                #pdb.set_trace()
                P =  clean_data_spectrum["PSD"][idx,:,:]
                #pdb.set_trace()
                f = plotSpectrum(P, Fs, origData['start_time_s'], origData['start_time_s'] + origData['data_time_s'], fr = fr, t = t, timeRange = (origData['start_time_s'],origData['start_time_s']+nSecPlot), show = False, fr_start = fr_start, fr_stop = fr_stop)
                pdf.savefig(f)
                plt.close(f)

        #pdb.set_trace()
        generateLastPage = False
        print('\nOn last page of pdf report')
        if generateLastPage:
            for idx, row in origData['data'].iteritems():
                if idx == 0:
                    f,_ = plotChan(origData['data'], origData['t'], idx, electrodeLabel = electrodeLabel, mask = None, show = False)
                elif idx == origData['data'].shape[1] - 1:
                    f,_ = plotChan(origData['data'], origData['t'], idx, electrodeLabel = electrodeLabel, mask = badData['general'], show = True, prevFig = f)
                    plt.tight_layout()
                    pdf.savefig(f)
                    plt.close(f)
                else:
                    f,_ = plotChan(origData['data'], origData['t'], idx, electrodeLabel = electrodeLabel, mask = None, show = False, prevFig = f)

def plotSpectrum(P, fs, start_time_s, end_time_s, fr_start = 10, fr_stop = 600, fr = None, t = None, show = False, timeRange = None):

    if fr is None:
        fr = np.arange(P.shape[1]) / P.shape[1] * fs / 2
    if t is None:
        t = start_time_s + np.arange(P.shape[0]) * (end_time_s-start_time_s) / P.shape[0]

    if type(P) is pd.DataFrame:
        P = P.values

    if timeRange is not None:
        tMask = np.logical_and(t > timeRange[0], t < timeRange[1])
        P = P[tMask, :]
        t = t[tMask]
        #pdb.set_trace()

    zMin, zMax = P.min(), P.max()

    f = plt.figure()
    plt.pcolormesh(t,fr,P.transpose(), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
        vmin=zMin, vmax=zMax))

    plt.axis([t.min(), t.max(), max(fr_start, fr.min()), min(fr_stop, fr.max())])
    #plt.colorbar()
    #plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Time (s)')
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    if show:
        plt.show(block = False)
    return f

def catSpikesGenerator(nBins = None, type = 'RMS', timePoint = None, subSet = slice(None, None), bounds = None):
    # creates a function that takes in spikes, chanIdx and categorizes them by the rule
    calcBounds = bounds is None
    if type == 'RMS':
        def catSpikes(spikes, chanIdx):
            rms = np.sqrt(np.mean(np.array(spikes['Waveforms'][chanIdx][:,subSet], dtype = np.float32) ** 2, axis = 1))
            if calcBounds:
                bounds = np.linspace(rms.min() * 0.99, rms.max() * 1.01, nBins + 1)
            return np.digitize(rms, bounds)
        return catSpikes

    if type == 'atTime':
        assert timePoint is not None
        def catSpikes(spikes, chanIdx):
            val = spikes['Waveforms'][chanIdx][:, timePoint]
            if calcBounds:
                bounds = np.linspace(val.min(), val.max(), nBins + 1)
            return np.digitize(val, bounds)
        return catSpikes

    if type == 'minPeak':
        def catSpikes(spikes, chanIdx):
            val = abs(spikes['Waveforms'][chanIdx][:,subSet].min(axis = 1))
            #pdb.set_trace()
            if calcBounds:
                bounds = np.linspace(val.min() * 0.99, val.max() * 1.01, nBins + 1)
            return np.digitize(val, bounds)
        return catSpikes

    if type == 'Classification':
        def catSpikes(spikes, chanIdx):
            return spikes['Classification'][chanIdx]
        return catSpikes

    return

def correctSpikeAlignment(spikes, w_pre):
    alignmentCorrection = w_pre / spikes['basic_headers']['TimeStampResolution'] # seconds
    for idx, spikeTimes in enumerate(spikes['TimeStamps']):
        spikes['TimeStamps'][idx] += alignmentCorrection
    return spikes

def getBinCenters(timeStart, timeEnd, binWidth, binInterval):
    #won't include last one unless I add binInterval to the end
    binCenters = np.arange(timeStart + binWidth / 2, timeEnd - binWidth / 2 + binInterval, binInterval)

    return binCenters

def getEventBins(timeStart, timeEnd, binInterval, binWidth):

    binCenters = getBinCenters(timeStart, timeEnd, binWidth, binInterval)

    binRes = gcd(math.floor(binWidth * 1e9 / 2), math.floor(binInterval * 1e9)) * 1e-9 # greatest common denominator
    fineBins = np.arange(timeStart, timeEnd + binRes, binRes)

    fineBinsPerWindow = int(binWidth / binRes)
    fineBinsPerInterval = int(binInterval / binRes)
    fineBinsTotal = len(fineBins) - 1

    centerIdx = np.arange(0, fineBinsTotal - fineBinsPerWindow + fineBinsPerInterval, fineBinsPerInterval)

    binLeftEdges = centerIdx * binRes

    return binCenters, fineBins, fineBinsPerWindow, binWidth, centerIdx, binLeftEdges

def binnedEvents(timeStamps, chans,
    timeStart = None, timeEnd = None, binInterval = None,
    binWidth = None, binCenters = None, fineBins = None, fineBinsPerWindow = None, centerIdx = None, binLeftEdges = None,
    ):

    if any((binWidth is None, binCenters is None, fineBins is None, fineBinsPerWindow is None, centerIdx is None, binLeftEdges is None)):
        binCenters, fineBins, fineBinsPerWindow, binWidth, centerIdx, binLeftEdges = getEventBins(timeStart, timeEnd, binInterval, binWidth)

    nChans = len(timeStamps)
    spikeMat = np.zeros([len(binCenters), nChans])
    for idx, chan in enumerate(chans):
        histo, _ = np.histogram(timeStamps[idx], fineBins)
        try:
            spikeMat[:, idx] = np.array(
                [histo[x:x+fineBinsPerWindow].sum() / binWidth for x in centerIdx]
            )
        except Exception:
            traceback.print_exc()
            pdb.set_trace()

    spikeMatDF = pd.DataFrame(spikeMat.transpose(), index = chans, columns = binCenters)
    return spikeMatDF, binCenters, binLeftEdges

def binnedSpikesAligned(spikes, alignTimes, binInterval, binWidth, channel,
    windowSize = [-0.25, 1], timeStampUnits = 'samples', discardEmpty = False):

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][ChanIdx])
    
    if unitsOnThisChan is not None:
        #peek at the binCenters. Note that this must be identical to the ones in binnedEvents
        binCenters = getBinCenters(windowSize[0], windowSize[1], binWidth, binInterval)
        
        spikeMats = [pd.DataFrame(0, index = alignTimes.index,
            columns = binCenters) for unit in unitsOnThisChan]
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            unitMask = spikes['Classification'][ChanIdx] == unitName
            # spike times in seconds
            allSpikeTimes = np.array(spikes['TimeStamps'][ChanIdx][unitMask])

            timeStampsToAnalyze = []
            trialID = []
            for rowIdx, startTime in alignTimes.items():
                idx = alignTimes.index.get_loc(rowIdx)
                try:
                    #print('Calculating raster for trial %s' % idx)
                    # allSpikeTimes is in seconds
                    # startTime is also in seconds
                    trialTimeMask = np.logical_and(allSpikeTimes > startTime + windowSize[0],
                        allSpikeTimes < startTime + windowSize[-1])
                    if trialTimeMask.sum() != 0:
                        trialSpikeTimes = allSpikeTimes[trialTimeMask] - startTime
                        timeStampsToAnalyze.append(trialSpikeTimes)
                        trialID.append(rowIdx)
                        #pdb.set_trace()
                    else:
                        if not discardEmpty:
                            timeStampsToAnalyze.append(np.array([]))
                            trialID.append(rowIdx)
                        else:
                            pass
                except Exception:
                    print('In binSpikesAligned: Error getting firing rate for trial %s' % idx)
                    traceback.print_exc()
                    spikeMats[unitIdx].iloc[idx, :] = np.nan

            spikeMats[unitIdx], binCenters, binLeftEdges = binnedEvents(
                timeStampsToAnalyze, trialID, timeStart = windowSize[0], timeEnd = windowSize[-1],
                binInterval = binInterval, binWidth = binWidth)
        #for idx, x in enumerate(spikeMats):
        #    spikeMats[idx].dropna(inplace = True)
    else: #no units on this chan
        spikeMats = []
    return spikeMats

def spikeAlignmentTimes(spikesTo,spikesToIdx,
    separateByFun = catSpikesGenerator(type = 'Classification'),
    timeRange = None, maxSpikesTo = None, discardEmpty = False):

    # get spike firing times to align to
    categories = None
    ChanIdxTo = spikesTo['ChannelID'].index(spikesToIdx['chan'])
    unitsOnThisChanTo = np.unique(spikesTo['Classification'][ChanIdxTo])

    if unitsOnThisChanTo is not None:
        alignTimes = pd.Series(spikesTo['TimeStamps'][ChanIdxTo])
        if separateByFun is not None:
            categories = pd.Series(separateByFun(spikesTo, ChanIdxTo), index = alignTimes.index)

    if timeRange is not None: # timeRange in seconds
        timeMask = np.logical_and(alignTimes > timeRange[0],
            alignTimes < timeRange[1])
        alignTimes = alignTimes.loc[timeMask]
        if separateByFun is not None:
            categories = categories.loc[timeMask]

    selectedIndices = None
    if maxSpikesTo is not None:
        if len(alignTimes.index) > maxSpikesTo:
            alignTimes = alignTimes.sample(n = maxSpikesTo)
            selectedIndices = alignTimes.index
            if separateByFun is not None:
                categories = categories.loc[selectedIndices]

    return alignTimes, categories, selectedIndices

def binnedSpikesAlignedToSpikes(spikesFrom, spikesTo,
    spikesFromIdx, spikesToIdx,
    binInterval, binWidth, windowSize = [-0.25, 1],
    separateByFun = catSpikesGenerator(type = 'Classification'),
    timeRange = None, maxSpikesTo = None, discardEmpty = False):

    # get spike firing times to align to
    alignTimes, categories, selectedIndices = spikeAlignmentTimes(spikesTo,spikesToIdx,
        separateByFun = separateByFun,
        timeRange = timeRange, maxSpikesTo = maxSpikesTo, discardEmpty =  discardEmpty)
    
    spikeMats = binnedSpikesAligned(spikesFrom, alignTimes, binInterval,
        binWidth, spikesFromIdx['chan'], windowSize = windowSize,
        discardEmpty = discardEmpty)

    return spikeMats, categories, selectedIndices

def binnedSpikesAlignedToTrial(spikes, binInterval, binWidth, trialStats,
    alignTo, channel, separateBy = None, windowSize = [-0.25, 1], timeRange = None,
    maxTrial = None, discardEmpty = False):

    #trialStats[alignedTo] gets converted from samples to seconds
    #spikes[TimesStamps] is already in seconds
    alignTimes = trialStats[alignTo] / 3e4

    if timeRange is not None:
        timeMask = np.logical_and(trialStats['FirstOnset'] > timeRange[0] * 3e4,
            trialStats['ChoiceOnset'] < timeRange[1] * 3e4)
        trialStats = trialStats.loc[timeMask, :]

    selectedIndices = None
    if maxTrial is not None:
        maxTrial = min(len(trialStats.index), maxTrial)
        trialStats = trialStats.iloc[:maxTrial, :]
        selectedIndices = trialStats.index

    if separateBy is not None:
        categories = trialStats[separateBy]
    else:
        categories = None

    spikeMats = binnedSpikesAligned(spikes, alignTimes, binInterval,
        binWidth, channel, windowSize = windowSize, discardEmpty = discardEmpty)

    return spikeMats, categories, selectedIndices

def binnedArray(spikes, rasterOpts, timeStart, chans = None):
    # bins all spikes at a particular point in time

    parseAll = True
    if chans is not None:
        parseAll = False

    timeStamps = []
    ChannelID = []

    for channel in spikes['ChannelID']:
        if parseAll:
            parseThis = True
        else:
            parseThis = channel in chans

        if  parseThis:
            #expand list to account for different units on each channel
            idx = spikes['ChannelID'].index(channel)
            unitsOnThisChan = np.unique(spikes['Classification'][idx])

            if unitsOnThisChan.any():
                for unitName in unitsOnThisChan:
                    unitMask = spikes['Classification'][idx] == unitName
                    timeStamps.append(spikes['TimeStamps'][idx][unitMask])
                    ChannelID.append(unitName)

    spikeMats = {i:None for i in timeStart.index}
    binCenters, fineBins, fineBinsPerWindow, binWidth, centerIdx, binLeftEdges =\
        getEventBins(rasterOpts['windowSize'][0],
            rasterOpts['windowSize'][-1],
            rasterOpts['binInterval'], rasterOpts['binWidth'])

    for idx, thisStartTime in timeStart.iteritems():
        try:

            spikeMats[idx], binCenters, binLeftEdges = binnedEvents(timeStamps, ChannelID,
                binWidth = rasterOpts['binWidth'],
                binCenters = binCenters + thisStartTime, fineBins = fineBins + thisStartTime,
                fineBinsPerWindow = fineBinsPerWindow,
                centerIdx = centerIdx, binLeftEdges = binLeftEdges + thisStartTime
                )

        except Exception:
            traceback.print_exc()
            pdb.set_trace()
    return spikeMats

def trialBinnedArray(spikes, rasterOpts, trialStats, chans = None):

    validMask = trialStats[rasterOpts['alignTo']].notnull()
    if rasterOpts['endOn'] is not None:
        validMask = np.logical_and(validMask,
            trialStats[rasterOpts['endOn']].notnull())
        endTimes = trialStats[rasterOpts['endOn']] - trialStats[rasterOpts['alignTo']]
        rasterOpts['windowSize'][1] = rasterOpts['windowSize'][1] + endTimes.max() / 3e4 # samples to seconds conversion
    # time units of samples
    timeStart = trialStats.loc[validMask, rasterOpts['alignTo']] / 3e4 # conversion to seconds
    spikeMats = {i : None for i in trialStats.index}
    spikeMats.update(binnedArray(spikes, rasterOpts, timeStart))

    return spikeMats

def binnedSpikes(spikes, binInterval, binWidth, timeStart, timeEnd,
    timeStampUnits = 'samples', chans = None):
    # bins all spikes at a particular point in time
    parseAll = True
    if chans is not None:
        parseAll = False

    timeStamps = []
    ChannelID = []
    if timeStampUnits == 'samples':
        allTimeStamps = [x / spikes['basic_headers']['TimeStampResolution'] for x in spikes['TimeStamps']]
    elif timeStampUnits == 'seconds':
        allTimeStamps = spikes['TimeStamps']

    for channel in spikes['ChannelID']:
        if parseAll:
            parseThis = True
        else:
            parseThis = channel in chans

        if  parseThis:
            #expand list to account for different units on each channel
            idx = spikes['ChannelID'].index(channel)
            unitsOnThisChan = np.unique(spikes['Classification'][idx])

            if unitsOnThisChan.any():
                for unitName in unitsOnThisChan:
                    unitMask = spikes['Classification'][idx] == unitName
                    timeStamps.append(allTimeStamps[idx][unitMask])
                    ChannelID.append(unitName)

    #pdb.set_trace()
    spikeMat, binCenters, binLeftEdges = binnedEvents(timeStamps, ChannelID,
        timeStart = timeStart, timeEnd = timeEnd, binInterval = binInterval, binWidth = binWidth)

    return spikeMat, binCenters, binLeftEdges

def plotBinnedSpikes(spikeMat, show = True, normalizationType = 'linear',
    zAxis = None, ax = None, labelTxt = 'Spk/s'):

    if ax is None:
        fi, ax = plt.subplots()
    else:
        fi = ax.figure

    #pdb.set_trace()
    if zAxis is not None:
        zMin, zMax = zAxis
    else:
        zMin, zMax = spikeMat.min().min(), spikeMat.max().max()

    if normalizationType == 'linear':
        nor = colors.Normalize(vmin=zMin, vmax=zMax)
    elif normalizationType == 'SymLogNorm':
        nor = colors.SymLogNorm(linthresh= 1, linscale=1, vmin=zMin, vmax=zMax)
    elif normalizationType == 'LogNorm':
        #print('zMin is {}'.format(zMin))
        nor = colors.LogNorm(vmin=max(zMin, 1e-6), vmax=zMax)
    elif normalizationType == 'Power':
        nor = colors.PowerNorm(gamma=0.5)

    #fi, ax = plt.subplots()
    cbar_kws = {'label' : labelTxt}
    axPos = ax.get_position()
    cbAxPos = [axPos.x0 + axPos.width - 0.075, axPos.y0,
        axPos.width / 50, axPos.height]
    cbAx = fi.add_axes(cbAxPos)
    #ax = sns.heatmap(spikeMat, ax = ax, robust = False,
    #    vmin = zMin, vmax = zMax, cbar_kws = cbar_kws, cbar_ax = cbAx)
    chanIdx = np.arange(len(spikeMat.index)+1)
    timeVals = np.linspace(spikeMat.columns[0], spikeMat.columns[-1],
        len(spikeMat.columns) + 1)
    #pdb.set_trace()
    im = ax.pcolormesh(timeVals, chanIdx, spikeMat, norm = nor, cmap = 'plasma')
    ax.set_xlim([timeVals[0], timeVals[-1]])
    #ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    cbar = fi.colorbar(im, cax=cbAx)
    cbar.set_label(labelTxt)
    labelFontSize = LABELFONTSIZE
    ax.set_xlabel('Time (s)', fontsize = labelFontSize,
        labelpad = - 3 * labelFontSize)
    ax.set_ylabel("Unit (#)", fontsize = labelFontSize,
        labelpad = - 3 * labelFontSize)
    #plt.show()

    ax.tick_params(left=False, top=False, right=False, bottom=False,
        labelleft=False, labeltop=False, labelright=False,
        labelbottom=False)
    if show:
        plt.show()
    return fi, im

def plot_spikes(spikes, chans, ignoreUnits = []):
    # Initialize plots
    #pdb.set_trace()
    colors      =  sns.color_palette(n_colors = 40)
    line_styles = ['-', '--', ':', '-.']
    f, axarr    = plt.subplots(len(chans))
    if len(chans) == 1:
        axarr = [axarr]
    samp_per_ms = spikes['basic_headers']['TimeStampResolution'] / 1000.0

    for i in range(len(chans)):

        # Extract the channel index, then use that index to get unit ids, extended header index, and label index
        ch_idx      = spikes['ChannelID'].index(chans[i])

        units       = sorted(list(set(spikes['Classification'][ch_idx])))
        unitIds     = list(range(len(units)))
        #ext_hdr_idx = spikes['NEUEVWAV_HeaderIndices'][ch_idx]
        #lbl_idx     = next(idx for (idx, d) in enumerate(spikes['extended_headers'])
        #                   if d['PacketID'] == 'NEUEVLBL' and d['ElectrodeID'] == chans[i])

        # loop through all spikes and plot based on unit classification
        # note: no classifications in sampleData, i.e., only unit='none' exists in the sample data
        ymin = 0; ymax = 0
        t = np.arange(spikes['Waveforms'][0].shape[1]) / samp_per_ms

        for j in range(len(units)):
            if units[j] in [-1] + ignoreUnits:
                continue
            unit_idxs   = [idx for idx, unit in enumerate(spikes['Classification'][ch_idx]) if unit == units[j]]
            unit_spikes = np.array(spikes['Waveforms'][ch_idx][unit_idxs])

            if units[j] == 'none':
                color_idx = 0; ln_sty_idx = 0
            else:
                color_idx = (unitIds[j] % len(colors)) + 1
                ln_sty_idx = unitIds[j] // len(colors)

            for k in range(unit_spikes.shape[0]):

                try:
                    axarr[i].plot(t, unit_spikes[k], line_styles[ln_sty_idx], c = colors[color_idx])
                    if min(unit_spikes[k]) < ymin: ymin = min(unit_spikes[k])
                    if max(unit_spikes[k]) > ymax: ymax = max(unit_spikes[k])
                except:
                    #pdb.set_trace()
                    pass

        #if lbl_idx: axarr[i].set_ylabel(spikes['extended_headers'][lbl_idx]['Label'] + ' ($\mu$V)')
        #else:       axarr[i].set_ylabel('Channel ' + str(chans[i]) + ' ($\mu$V)')
        axarr[i].set_ylabel('Channel ' + str(chans[i]) + ' ' + spikes['Units'])
        axarr[i].set_ylim((ymin * 1.05, ymax * 1.05))
        axarr[i].locator_params(axis='y', nbins=10)

    axarr[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show(block = False)

def plot_events_raster(eventDf, names, collapse = False, usePlotly = True):
    # Initialize plots
    (viridis_cmap, norm,colorsRgb, plotColPlotly,
    plotColMPL, names2int, line_styles) = getPlotOpts(names)

    if not usePlotly:
        fig, ax = plt.subplots()
    else:
        data = []
        layout = {

            'title' : 'Event Raster Plot',

            'xaxis' : dict(
                title='Time (sec)',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            ),

            'yaxis' : dict(
                title='',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            ),
            'shapes' : []}

    for idx, name in enumerate(names):

        color_idx = (names2int[idx] % len(plotColPlotly))
        ln_sty_idx = 0
        event_times = eventDf['Time'][eventDf['Label'] == name]

        if collapse:
            idx = 0

        if not usePlotly:

            ax.vlines(event_times, idx, idx + 1, colors = plotColMPL[color_idx],
                linestyles = line_styles[ln_sty_idx], label = name)

        else:

            data.append( go.Scatter(
                    x = event_times,
                    y = [idx + 0.5 for i in event_times],
                    mode = 'markers',
                    name = name,
                    marker = dict(
                        color = plotColPlotly[color_idx][1],
                        symbol = "line-ns-open",
                        line = dict(
                            width = 4,
                            color = plotColPlotly[color_idx][1]
                        )
                    )
                )
            )

            layout['shapes'] = layout['shapes'] + [
                {
                    'type' : 'line',
                    'x0' :  t,
                    'y0' :  idx,
                    'x1' :  t,
                    'y1' :  idx + 1,
                    'line' :  dict(
                        color = plotColPlotly[color_idx][1],
                        width = 5
                    )
                } for t in event_times
            ]

    if not usePlotly:
        plt.legend()
        plt.xlabel('Times (sec)')
        ax.get_yaxis().set_visible(False)
        rasterFig = fig

    else:
        rasterFig = go.Figure(data=data,layout=layout)
        #py.iplot(fig, filename='eventRaster')
    return rasterFig

def plotPeristimulusTimeHistogram(eventDf, stimulus, names,
    preInterval = 5, postInterval = 5, deltaInterval = 100e-3,
    collapse = False, usePlotly = True):
    """
    https://en.wikipedia.org/wiki/Peristimulus_time_histogram
    """
    nBins = int((preInterval + postInterval)/deltaInterval + 1)
    (viridis_cmap, norm,colorsRgb, plotColPlotly,
        plotColMPL, names2int, line_styles) = getPlotOpts(names)

    if not usePlotly:
        fig, ax = plt.subplots()
        listHistograms = []
    else:
        data = []
        layout = go.Layout(barmode='overlay')
        layout['title'] = 'Peristimulus Time Histogram'
        layout['titlefont'] =dict(
                family='Arial',
                size=25,
                color='#7f7f7f'
            )
        layout['xaxis'] = dict(
                title='Time (sec)',
                titlefont=dict(
                    family='Arial',
                    size=25,
                    color='#7f7f7f'
                ),
                tickfont = dict(
                    size = 20
                )
            )
        layout['yaxis'] = dict(
                title='Count',
                titlefont=dict(
                    family='Arial',
                    size=25,
                    color='#7f7f7f'
                ),
                tickfont = dict(
                    size = 20
                )
            )
        layout['legend'] = dict(
            font = dict(
                size = 20
            )
        )

    maxTime = max(eventDf['Time'])
    for idx, name in enumerate(names):

        color_idx = (names2int[idx] % len(plotColPlotly))
        ln_sty_idx = 0
        event_times = eventDf['Time'][eventDf['Label'] == name]

        eventTimes = pd.Series(index = ['Time'])
        allEventTimes = eventDf[eventDf['Label'] == name]['Time']

        for stimTime in eventDf[eventDf['Label'].isin(stimulus)]['Time']:
            startTime = max(stimTime - preInterval, 0)
            endTime = min(stimTime + postInterval, maxTime)

            thisMask = np.logical_and(allEventTimes > startTime,
                allEventTimes < endTime)
            theseEventTimes = allEventTimes[thisMask] - stimTime
            #theseEventTimes.dropna(inplace = True)
            eventTimes = eventTimes.append(theseEventTimes, ignore_index = True)

        eventTimes.dropna(inplace = True)
        if not usePlotly:
            #store list of these values
            listHistograms.append(eventTimes.values)
        else:
            data.append( go.Histogram(
                    x = eventTimes,
                    name = name,
                    opacity = 0.75,
                    nbinsx = nBins,
                    marker = dict(
                        color = plotColPlotly[color_idx][1],
                        line = dict(
                            width = 4,
                            color = plotColPlotly[color_idx][1]
                        )
                    )
                )
            )

    if not usePlotly:
        ax.hist(listHistograms, nBins, histtype='bar')
        plt.legend(names)
        plt.xlabel('Times (sec)')
        ax.get_yaxis().set_visible(False)
        psthFig = fig
    else:
        layout['shapes'] = [
            {
                'type' : 'line',
                'x0' :  0,
                'y0' :  0,
                'x1' :  0,
                'y1' :  np.max([np.max(np.histogram(datum['x'], bins = nBins)[0]) for datum in data]),
                'line' :  dict(
                    color = 'rgb(0,0,0)',
                    width = 5
                )
            }
        ]

        psthFig = go.Figure(data=data,layout=layout)

    return psthFig

def plot_trial_stats(trialStatsDf, usePlotly = True, separate = None, debugging = False, title = 'Global outcomes'):
    #trialStatsDf = trialStats

    conditionShortNames = trialStatsDf['Condition'].unique()
    typeShortNames = conditionStats['Type'].unique()
    directionShortNames = conditionStats['Direction'].unique()

    if usePlotly:
        data = []

        for conditionName in conditionShortNames:
            conditionStats = trialStatsDf[trialStatsDf['Condition'] == conditionName]
            if separate == 'leftRight':
                #typeName = next(iter(np.unique(conditionStats['Type'])))
                for typeName in typeShortNames:
                    typeStats = conditionStats[conditionStats['Type'] == typeName]
                    y = [typeStats[typeStats['Outcome'] == on].size \
                        for on in sorted(typeStats['Outcome'].unique())]
                    if debugging:
                        print(y)
                    x = [outcomeLongName[on] for on in sorted(typeStats['Outcome'].unique())]

                    data.append(go.Bar(
                        x=x,
                        y= [i / sum(y) for i in y],
                        name=conditionLongName[conditionName] + ' ' + typeLongName[typeName] + ' ' + str(len(typeStats)) + ' total trials'
                        ))
            elif separate == 'forwardBack':
                for directionName in directionShortNames:
                    directionStats = conditionStats[conditionStats['Direction'] == directionName]
                    y = [directionStats[directionStats['Outcome'] == on].size \
                        for on in sorted(directionStats['Outcome'].unique())]
                    if debugging:
                        print(y)
                    x = [outcomeLongName[on] for on in sorted(directionStats['Outcome'].unique())]
                    data.append(go.Bar(
                        x=x,
                        y= [i / sum(y) for i in y],
                        name=conditionLongName[conditionName] + ' ' + directionName + ' ' + str(len(directionStats)) + ' total trials'
                        ))
            else:
                y = [conditionStats[conditionStats['Outcome'] == on].size \
                    for on in sorted(conditionStats['Outcome'].unique())]
                if debugging:
                    print(y)
                x = [outcomeLongName[on] for on in sorted(conditionStats['Outcome'].unique())]
                #pdb.set_trace()
                data.append(go.Bar(
                    x=x,
                    y= [i / sum(y) for i in y],
                    name=conditionLongName[conditionName] + ' ' + str(len(conditionStats)) + ' trials'
                    ))

        #plotTitle = ['%d %s trials' % (totals[cn], conditionLongName[cn]) for cn in conditionShortNames]
        #plotTitle = ' '.join(plotTitle)
        layout = {
            'barmode' : 'group',
            'title' : title,
            'xaxis' : {
                'title' : 'Outcome',
                'titlefont' : {
                    'family' : 'Arial',
                    'size' : 30,
                    'color' : '#7f7f7f'
                    },
                'tickfont' : {
                    'size' : 20
                    }
                },
            'yaxis' : {
                'title' : 'Percentage',
                'titlefont' : {
                    'family' : 'Arial',
                    'size' : 25,
                    'color' : '#7f7f7f'
                    },
                'tickfont' : {
                    'size' : 20
                    }
                },
            'legend' : {
                'font' : {
                    'size' : 20
                }
            }
            }

        fig = go.Figure(data=data, layout=layout)
        return fig, data, layout
    else:

        return None

def runScriptAllFiles(scriptPath, folderPath):
    onlyfiles = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
    for f in onlyfiles:
        try:
            subprocess.check_output('python3 ' + scriptPath + ' --file '  + '\"' + f  + '\"' + ' --folder \"' + folderPath + '\"', shell=True)
        except:
            try:
                subprocess.check_output('python ' + scriptPath + ' --file '  + '\"' + f  + '\"' + ' --folder \"' + folderPath + '\"', shell=True)
            except:
                print("issue with :" + f)

def fileId_from_url(url):
    """Return fileId from a url."""
    index = url.find('~')
    fileId = url[index + 1:]
    local_id_index = fileId.find('/')

    share_key_index = fileId.find('?share_key')
    if share_key_index == -1:
        return fileId.replace('/', ':')
    else:
        return fileId[:share_key_index].replace('/', ':')

def sharekey_from_url(url):
    """Return the sharekey from a url."""
    index = url.find('share_key=')
    return url[index + len('share_key='):]

def plot_raster(spikes, chans):
    # Initialize plots
    colors      = 'kbgrm'
    line_styles = ['-', '--', ':', '-.']
    f, ax    = plt.subplots()
    samp_per_s = spikes['basic_headers']['SampleTimeResolution']

    timeMax = max([max(x) for x in spikes['TimeStamps']])

    for idx, chan in enumerate(chans):
        # Extract the channel index, then use that index to get unit ids, extended header index, and label index
        #pdb.set_trace()
        ch_idx      = spikes['ChannelID'].index(chan)
        units       = sorted(list(set(spikes['Classification'][ch_idx])))
        ext_hdr_idx = spikes['NEUEVWAV_HeaderIndices'][ch_idx]
        lbl_idx     = next(idx for (idx, d) in enumerate(spikes['extended_headers'])
                           if d['PacketID'] == 'NEUEVLBL' and d['ElectrodeID'] == chan)

        # loop through all spikes and plot based on unit classification
        # note: no classifications in sampleData, i.e., only unit='none' exists in the sample data

        t = np.arange(timeMax) / samp_per_s

        for j in range(len(units)):
            unit_idxs   = [idx for idx, unit in enumerate(spikes['Classification'][ch_idx]) if unit == units[j]]
            unit_spike_times = np.array(spikes['TimeStamps'][ch_idx])[unit_idxs] / samp_per_s

            if units[j] == 'none':
                color_idx = 0; ln_sty_idx = 0
            else:
                color_idx = (units[j] % len(colors)) + 1
                ln_sty_idx = units[j] // len(colors)

            for spt in unit_spike_times:
                ax.vlines(spt, idx, idx + 1, colors = colors[color_idx],  linestyles = line_styles[ln_sty_idx])

        ax.set_ylim((0, len(chans)))
        ax.locator_params(axis='y', nbins=10)

    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show(block = False)

def readPiLog(filePaths, names = None, zeroTime = False, fixMovedToError = [False]):
    logs = pd.DataFrame()
    trialStatsAll = pd.DataFrame()
    for idx, filePath in enumerate(filePaths):
        if names is not None:
            log = pd.read_table(filePath, header = None, names = names)
        else:
            log = pd.read_table(filePath)

        if zeroTime:
            if 'Time' in log.columns.values:
                log['Time'] = log['Time'] - log['Time'][0]
            else:
                print('No time column found to zero!')

        keepIdx = None
        if fixMovedToError[idx]:
            for idx, row in log.iterrows():
                if row['Label'] == 'turnPedalRandom_1' or row['Label'] == 'turnPedalRandom_2':
                    keepIdx = idx
                if row['Label'] == 'moved to: ':
                    log.loc[keepIdx, 'Details'] = row['Time']
                    keepIdx = None
                    log.drop([idx], inplace = True)

        for idx, row in log.iterrows():
            if row['Label'] == 'button timed out!':
                log.loc[idx, 'Label'] = 'button timed out'
            if row['Label'] == 'correct_button':
                log.loc[idx, 'Label'] = 'correct button'
            if row['Label'] == 'incorrect_button':
                log.loc[idx, 'Label'] = 'incorrect button'

        log.drop_duplicates(subset = 'Time', inplace = True)
        mask = log['Label'].str.contains('turnPedalRandom_1') | \
            log['Label'].str.contains('turnPedalRandom_2') | \
            log['Label'].str.contains('easy') | \
            log['Label'].str.contains('hard') | \
            log['Label'].str.endswith('correct button') | \
            log['Label'].str.endswith('incorrect button') | \
            log['Label'].str.endswith('button timed out')
        trialRelevant = pd.DataFrame(log[mask]).reset_index()

        # TODO: kludge, to avoid wait for corect button substring. fix
        # later note: what does the above even mean?

        # if the last trial didn't have time to end, remove its entries from the list of events
        while not trialRelevant.iloc[-1, :]['Label'] in ['correct button', 'incorrect button', 'button timed out']:
            trialRelevant.drop(trialRelevant.index[len(trialRelevant)-1], inplace = True)

        def magnitude_lookup_table(difference):
            if difference > 2e4:
                return ('Long', 'Extension')
            if difference > 0:
                return ('Short', 'Extension')
            if difference > -2e4:
                return ('Short', 'Flexion')
            else:
                return ('Long', 'Flexion')

        trialStartIdx = trialRelevant.index[trialRelevant['Label'].str.contains('turnPedalRandom_1')]
        trialStats = pd.DataFrame(index = trialStartIdx, columns = ['First', 'Second', 'Magnitude', 'Direction', 'Condition', 'Type', 'Stimulus Duration', 'Outcome', 'Choice'])

        for idx in trialStartIdx:
            assert trialRelevant.loc[idx, 'Label'] == 'turnPedalRandom_1'
            movementStartTime = trialRelevant.loc[idx, 'Time']

            trialStats.loc[idx, 'First'] = float(trialRelevant.loc[idx, 'Details'])

            assert trialRelevant.loc[idx + 1, 'Label'] == 'turnPedalRandom_2'

            trialStats.loc[idx, 'Second'] = float(trialRelevant.loc[idx + 1, 'Details'])
            trialStats.loc[idx, 'Type'] = 0 if abs(trialStats.loc[idx, 'First']) < abs(trialStats.loc[idx, 'Second']) else 1
            trialStats.loc[idx, 'Magnitude'] = magnitude_lookup_table(trialStats.loc[idx, 'First'] - trialStats.loc[idx, 'Second'])[0]
            trialStats.loc[idx, 'Direction'] = magnitude_lookup_table(trialStats.loc[idx, 'First'] - trialStats.loc[idx, 'Second'])[1]
            assert (trialRelevant.loc[idx + 2, 'Label'] == 'easy') | (trialRelevant.loc[idx + 2, 'Label'] == 'hard')
            movementEndTime = trialRelevant.loc[idx + 2, 'Time']

            trialStats.loc[idx, 'Stimulus Duration'] = movementEndTime - movementStartTime
            trialStats.loc[idx, 'Condition'] = trialRelevant.loc[idx + 2, 'Label']
            assert (trialRelevant.loc[idx + 3, 'Label'] == 'correct button') | \
                (trialRelevant.loc[idx + 3, 'Label'] == 'incorrect button') | \
                (trialRelevant.loc[idx + 3, 'Label'] == 'button timed out')

            trialStats.loc[idx, 'Outcome'] = trialRelevant.loc[idx + 3, 'Label']
            if trialRelevant.loc[idx + 3, 'Label'] == 'button timed out':
                trialStats.loc[idx, 'Choice'] = 'button timed out'
            else:
                trialStats.loc[idx, 'Choice'] = trialRelevant.loc[idx + 3, 'Details']

        logs = pd.concat([logs, log], ignore_index = True)
        trialStatsAll = pd.concat([trialStatsAll, trialStats], ignore_index = True)
        print(trialStatsAll.shape)
    return logs, trialStatsAll

def readPiJsonLog(filePaths, zeroTime = False):
    logs = pd.DataFrame()
    trialStatsAll = pd.DataFrame()

    timeOffset = 0
    for idx, filePath in enumerate(filePaths):
        #idx, filePath = next(enumerate(filePaths))
        try:
            log = pd.read_json(filePath, orient = 'records')
        except:
            with open(filePath, 'a+') as f:
                f.write('{}]')
            log = pd.read_json(filePath, orient = 'records')
        log.drop_duplicates(subset = 'Time', inplace = True)
        trialStartTime = 0
        if zeroTime:
            #zeroTime = True
            if 'Time' in log.columns.values:
                trialStartTime = log['Time'][0]
                log['Time'] = log['Time'] - trialStartTime + timeOffset
                timeOffset = timeOffset + log['Time'].iloc[-1]
            else:
                print('No time column found to zero!')


        mask = log['Label'].str.contains('turnPedalPhantomCompound') | \
            log['Label'].str.contains('turnPedalCompound') | \
            log['Label'].str.contains('goEasy') | \
            log['Label'].str.contains('goHard') | \
            log['Label'].str.endswith('correct button') | \
            log['Label'].str.endswith('incorrect button') | \
            log['Label'].str.endswith('button timed out')
        trialRelevant = pd.DataFrame(log[mask]).reset_index()

        # TODO: kludge, to avoid wait for corect button substring. fix
        # later note: what does the above even mean?

        # if the last trial didn't have time to end, remove its entries from the list of events
        print('On file %s' % filePath)
        #pdb.set_trace()
        while not trialRelevant.iloc[-1, :]['Label'] in ['correct button', 'incorrect button', 'button timed out']:
            trialRelevant.drop(trialRelevant.index[len(trialRelevant)-1], inplace = True)

        def magnitude_lookup_table(difference):
            if difference > 2e4:
                return ('Long', 'Extension')
            if difference > 0:
                return ('Short', 'Extension')
            if difference > -2e4:
                return ('Short', 'Flexion')
            else:
                return ('Long', 'Flexion')

        trialStartIdx = trialRelevant.index[trialRelevant['Label'].str.contains('turnPedalPhantomCompound')] | trialRelevant.index[trialRelevant['Label'].str.contains('turnPedalCompound')]
        trialStats = pd.DataFrame(index = trialStartIdx, columns = ['First', 'Second', 'Magnitude', 'Direction', 'Condition', 'Type', 'Stimulus Duration', 'Outcome', 'Choice'])

        for idx in trialStartIdx:
            #idx = trialStartIdx[0]
            assert trialRelevant.loc[idx, 'Label'] == 'turnPedalPhantomCompound' or trialRelevant.loc[idx, 'Label'] == 'turnPedalCompound'
            movementStartTime = trialRelevant.loc[idx, 'Details']['movementOnset']

            trialStats.loc[idx, 'First'] = float(trialRelevant.loc[idx, 'Details']['firstThrow'])
            trialStats.loc[idx, 'Second'] = float(trialRelevant.loc[idx, 'Details']['secondThrow'])
            trialStats.loc[idx, 'Type'] = 0 if abs(trialStats.loc[idx, 'First']) < abs(trialStats.loc[idx, 'Second']) else 1
            trialStats.loc[idx, 'Magnitude'] = magnitude_lookup_table(trialStats.loc[idx, 'First'] - trialStats.loc[idx, 'Second'])[0]
            trialStats.loc[idx, 'Direction'] = magnitude_lookup_table(trialStats.loc[idx, 'First'] - trialStats.loc[idx, 'Second'])[1]
            assert (trialRelevant.loc[idx + 1, 'Label'] == 'goEasy') | (trialRelevant.loc[idx + 1, 'Label'] == 'goHard')
            movementEndTime = trialRelevant.loc[idx, 'Details']['movementOff']

            trialStats.loc[idx, 'Stimulus Duration'] = movementEndTime - movementStartTime

            conditionLookUp = {
                'goEasy' : 'easy',
                'goHard' : 'hard'
                }
            trialStats.loc[idx, 'Condition'] = conditionLookUp[trialRelevant.loc[idx + 1, 'Label']]
            assert (trialRelevant.loc[idx + 2, 'Label'] == 'correct button') | \
                (trialRelevant.loc[idx + 2, 'Label'] == 'incorrect button') | \
                (trialRelevant.loc[idx + 2, 'Label'] == 'button timed out')

            trialStats.loc[idx, 'Outcome'] = trialRelevant.loc[idx + 2, 'Label']
            if trialRelevant.loc[idx + 2, 'Label'] == 'button timed out':
                trialStats.loc[idx, 'Choice'] = 'button timed out'
            else:
                trialStats.loc[idx, 'Choice'] = trialRelevant.loc[idx + 2, 'Details']

        logs = pd.concat([logs, log], ignore_index = True)
        trialStatsAll = pd.concat([trialStatsAll, trialStats], ignore_index = True)
        #print(trialStatsAll.shape)
    return logs, trialStatsAll

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Modified from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    fi = plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #np.set_printoptions(precision=2)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:4.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.colorbar()
    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fi

def plotFeature(X, y):
    fi = plt.figure()

    t = np.array(range(X.shape[0]))
    dummyVar = np.ones(X.shape[0]) * 1

    zMin = X.min().min()
    zMax = X.max().max()

    plt.pcolormesh(X.transpose(),
        norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
        vmin=zMin, vmax=zMax))
    ax = fi.axes[0]
    clrs = ['b', 'r', 'g']
    #pdb.set_trace()
    for idx, label in enumerate(np.unique(y)):
        if label != 0: ax.plot(t[y.values == label], dummyVar[y.values == label], 'o', c = clrs[idx])
    #plt.show()
    return fi

def freqDownSampler(nChan, factor):
    def downSample(X):
        support = np.arange(X.shape[1] / nChan) # range of original frequencies
        downSampledSupport = np.linspace(0, support[-1], m.floor(len(support) / factor)) # range of new frequencies

        XReshaped = np.reshape(X.values, (X.shape[0], nChan, -1)) # electrodes by frequency by time bin
        XDownSampled = np.zeros((XReshaped.shape[0], XReshaped.shape[1], m.floor(XReshaped.shape[2] / factor)))# electrodes by new frequency by time bin

        for idx in range(nChan):
            oldX = XReshaped[:, idx, :]
            interpFun = interpolate.interp1d(support, oldX, kind = 'cubic', axis = -1)
            XDownSampled[:, idx, :] = interpFun(downSampledSupport)

        XDownSampledFlat = pd.DataFrame(np.reshape(XDownSampled, (X.shape[0], -1)))
        return XDownSampledFlat
    return downSample

def freqDownSample(X, **kwargs):
    keepChans = kwargs['keepChans']
    whichChans = kwargs['whichChans']
    strategy = kwargs['strategy']

    nChan = len(whichChans)
    newNChan = len(keepChans)

    support = np.arange(X.shape[1] / nChan) # range of original frequencies
    nTimePoints = X.shape[0]

    XReshaped = np.reshape(X, (nTimePoints, nChan, -1)) # electrodes by frequency by time bin
    del(X)
    XReshaped = XReshaped[:, keepChans, :] #remove unwanted contacts

    if strategy == 'interpolate':
        freqFactor = kwargs['freqFactor']
        newFreqRes = m.floor(len(support) / freqFactor)
        downSampledSupport = np.linspace(support[0], support[-1], newFreqRes) # range of new frequencies

        XDownSampled = np.zeros((XReshaped.shape[0], newNChan, m.floor(XReshaped.shape[2] / freqFactor)))# electrodes by new frequency by time bin

        for idx in range(newNChan):
            oldX = XReshaped[:, idx, :]
            interpFun = interpolate.interp1d(support, oldX, kind = 'linear', axis = -1)
            XDownSampled[:, idx, :] = interpFun(downSampledSupport)
            del(interpFun)
            #gc.collect()
        del(XReshaped)
    else:
        bands = kwargs['bands']
        maxFreq = kwargs['maxFreq']
        XDownSampled = np.zeros((XReshaped.shape[0], newNChan, len(bands)))# electrodes by new frequency by time bin
        support = np.linspace(0, maxFreq, len(support))
        bandIdx = [np.logical_and(support > band[0], support < band[1]) for band in bands]

        for idx in range(newNChan):
            oldX = XReshaped[:, idx, :]

            for frIdx, band in enumerate(bands):

                if not np.any(bandIdx[frIdx]):
                    print("WARNING. no evaluations in band.")
                    print(band)
                    closeIdx = (np.abs(support-np.mean(band))).argmin()
                    bandIdx[frIdx] = support == support[closeIdx]

                XDownSampled[:, idx, frIdx] = np.mean(oldX[:, bandIdx[frIdx]], axis = 1)
        del(XReshaped)
    XDownSampledFlat = pd.DataFrame(np.reshape(XDownSampled, (nTimePoints, -1)))
    del(XDownSampled)
    return XDownSampledFlat

def selectFromX(X, support):
    return X[:,support]

def ROCAUC_ScoreFunction(estimator, X, y):
    #pdb.set_trace()
    binarizer = LabelBinarizer()
    binarizer.fit([0,1,2])
    average = 'macro'
    if hasattr(estimator, 'decision_function'):
        score = roc_auc_score(binarizer.transform(y), estimator.decision_function(X), average = average)
    elif hasattr(estimator, 'predict_proba'):
        #pdb.set_trace()
        score = roc_auc_score(binarizer.transform(y), estimator.predict_proba(X), average = average)
    else: # default to f1 score
        score = f1_score(y, estimator.predict(X), average = average)
    return score

def getSpectrumXY(dataNames, whichChans, maxFreq):
    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

    X, y, trueLabels = pd.DataFrame(), pd.Series(), pd.Series()

    for idx, dataName in enumerate(dataNames):
        dataFile = localDir + dataName

        ns5Data = pd.read_pickle(dataFile)
        spectrum = ns5Data['channel']['spectrum']['PSD']
        whichFreqs = ns5Data['channel']['spectrum']['fr'] < maxFreq

        reducedSpectrum = spectrum[whichChans, :, whichFreqs]
        X = pd.concat((X,reducedSpectrum.transpose(1, 0, 2).to_frame().transpose()))
        y = pd.concat((y,ns5Data['channel']['spectrum']['LabelsNumeric']))
        trueLabels = pd.concat((trueLabels, ns5Data['channel']['spectrum']['Labels']))
    return X, y, trueLabels

def trainSpectralMethod(
    dataNames, whichChans, maxFreq, estimator, skf, parameters,
    outputFileName, memPreallocate = 'n_jobs'):

    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
    X, y, _ = getSpectrumXY(dataNames, whichChans, maxFreq)

    grid=GridSearchCV(estimator, parameters, cv = skf, verbose = 1,
        scoring = ROCAUC_ScoreFunction, pre_dispatch = memPreallocate,
        n_jobs = -1)

    #if __name__ == '__main__':
    grid.fit(X,y)

    bestEstimator={'estimator' : grid.best_estimator_,
        'info' : grid.cv_results_, 'whichChans' : whichChans,
        'maxFreq' : maxFreq}

    with open(localDir + outputFileName, 'wb') as f:
        pickle.dump(bestEstimator, f)

def getEstimator(modelFileName):
    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
    modelFile = localDir + modelFileName
    estimatorDict = pd.read_pickle(modelFile)
    estimator = estimatorDict['estimator']

    try:
        estimatorInfo = estimatorDict['info']
    except:
        estimatorInfo = None

    try:
        whichChans = estimatorDict['whichChans']
    except:
        whichChans = None

    try:
        maxFreq = estimatorDict['maxFreq']
    except:
        maxFreq = None

    return estimator, estimatorInfo, whichChans, maxFreq

def getSpikeXY(dataNames, whichChans):
    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
    X, y, trueLabels = pd.DataFrame(), pd.Series(), pd.Series()
    for idx, dataName in enumerate(dataNames):
        #get all columns of spikemat that aren't the labels
        dataFile = localDir + dataName
        data = pd.read_pickle(dataFile)

        spikeMat = data['spikeMat']
        nonLabelChans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

        X = pd.concat((X,spikeMat[nonLabelChans]))
        y = pd.concat((y,spikeMat['LabelsNumeric']))
        trueLabels = pd.concat((trueLabels, spikeMat['Labels']))
    return X, y, trueLabels

def trainSpikeMethod(dataNames, whichChans, estimator, skf, parameters,
    outputFileName, memPreallocate = 'n_jobs'):

    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
    X, y, _ = getSpikeXY(dataNames, whichChans)

    grid=GridSearchCV(estimator, parameters,
        scoring = ROCAUC_ScoreFunction,
        cv = skf, n_jobs = -1, pre_dispatch = memPreallocate, verbose = 2)

    grid.fit(X,y)

    best_estimator = grid.best_estimator_
    estimator_info = grid.cv_results_

    bestEstimator={'estimator' : best_estimator, 'info' : estimator_info}

    with open(localDir + outputFileName, 'wb') as f:
        pickle.dump(bestEstimator, f)

def getModelName(estimator):
    if type(estimator) == sklearn.pipeline.Pipeline:
        modelName = '_'.join([x[0] for x in estimator.steps])
    else:
        modelName = str(type(estimator)).split('.')[-1]
        #strip away non alphanumeric chars
        modelName = ''.join(c for c in modelName if c in string.ascii_letters)
    return modelName

def fitRFECV(estimator, skf, X, y, nFeatures = None):
    if nFeatures == None:
        selector = RFECV(estimator, step=1, cv=skf, scoring = ROCAUC_ScoreFunction,
            n_jobs = -1, verbose = 1)
    else:
        selector = RFE(estimator, n_features_to_select=nFeatures, step=1, verbose=0)
    selector.fit(X, y)
    return selector

def plotValidationCurve(estimator, estimatorInfo, whichParams = [0,1]):
    modelName = getModelName(estimator)
    keys = sorted([name for name in estimatorInfo.keys() if name[:6] == 'param_'])
    validationParams = {}
    for key in keys:
        if isinstance(estimatorInfo[key].data[0], dict):
            nestedKeys = sorted([name for name in estimatorInfo[key].data[0].keys()])

            if 'downSampler' in modelName.split('_'):
                # These aren't "parameters" for this particular estimator
                if 'whichChans' in nestedKeys: nestedKeys.remove('whichChans')
                if 'maxFreq' in nestedKeys: nestedKeys.remove('maxFreq')
                if 'strategy' in nestedKeys: nestedKeys.remove('strategy')
                #
                whichChans = estimatorInfo['params'][0]['downSampler__kw_args']['whichChans']
                strategy = estimatorInfo['params'][0]['downSampler__kw_args']['strategy']

            for nestedKey in nestedKeys:
                if nestedKey == 'keepChans':
                    theseParams = np.array([len(whichChans) / len(x[nestedKey]) for x in estimatorInfo[key].data])
                else:
                    theseParams = np.array([x[nestedKey] for x in estimatorInfo[key].data])

                validationParams.update({nestedKey : theseParams})
        else:
            validationParams.update({key[6:]: estimatorInfo[key].data})

    keys = sorted([name for name in validationParams.keys()])

    if len(keys) > 2:
        keys = [keys[i] for i in whichParams]

    if len(keys) == 1:
        fi, ax = plt.subplots()
        plt.title("Validation Curve with " + modelName)
        plt.xlabel("Parameter")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2

        nParam1 = len(np.unique(validationParams[keys[0]]))
        param1 = validationParams[keys[0]]
        xTickRange = list(range(nParam1))
        xTickTargets = list(np.unique(param1))
        try:
            xTickLabels = ['{:4.2e}'.format(num) for num in xTickTargets]
        except:
            xTickLabels = xTickTargets
        xTicks = [xTickRange[xTickTargets.index(x)] for x in param1]

        ax.semilogx(param1, estimatorInfo['mean_train_score'], label="Training score",
                     color="darkorange", lw=lw)
        ax.fill_between(param1, estimatorInfo['mean_train_score'] - estimatorInfo['std_train_score'],
                         estimatorInfo['mean_train_score'] + estimatorInfo['std_train_score'], alpha=0.2,
                         color="darkorange", lw=lw)

        ax.semilogx(param1, estimatorInfo['mean_test_score'], label="Cross-validation score",
                     color="navy", lw=lw)
        ax.fill_between(param1, estimatorInfo['mean_test_score'] - estimatorInfo['std_test_score'],
                         estimatorInfo['mean_test_score'] + estimatorInfo['std_test_score'], alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")

    elif len(keys) == 2:
        fi, ax = plt.subplots(nrows = 2, ncols = 1)
        ax[0].set_title("Validation Curve with " + modelName)

        nParam1 = len(np.unique(validationParams[keys[0]]))
        param1 = validationParams[keys[0]]
        nParam2 = len(np.unique(validationParams[keys[1]]))
        param2 = validationParams[keys[1]]

        nParams = [nParam1, nParam2]

        zMin = min(estimatorInfo['mean_test_score'])
        zMax = max(estimatorInfo['mean_test_score'])

        #Create a uniform grid for the scatter plot
        xTickRange = list(range(nParam1))
        xTickTargets = list(np.unique(param1))
        try:
            xTickLabels = ['{:4.2e}'.format(num) for num in xTickTargets]
        except:
            xTickLabels = xTickTargets
        xTicks = [xTickRange[xTickTargets.index(x)] for x in param1]

        yTickRange = list(range(nParam2))
        yTickTargets = list(np.unique(param2))
        try:
            yTickLabels = ['{:4.2e}'.format(num) for num in yTickTargets]
        except:
            yTickLabels = yTickTargets
        yTicks = [yTickRange[yTickTargets.index(y)] for y in param2]

        im = ax[0].scatter(xTicks, yTicks, c = estimatorInfo['mean_test_score'],
            s = 200, norm = colors.Normalize(vmin=zMin, vmax=zMax))
        im = ax[1].scatter(xTicks, yTicks, c = estimatorInfo['mean_train_score'],
            s = 200, norm = colors.Normalize(vmin=zMin, vmax=zMax))

        ax[0].set_xlabel(keys[0])
        ax[0].set_ylabel('Test ' + keys[1])
        ax[0].set_xticks(xTickRange)
        ax[0].set_yticks(yTickRange)
        ax[0].set_xticklabels(xTickLabels)
        ax[0].set_yticklabels(yTickLabels)

        ax[1].set_xlabel(keys[0])
        ax[1].set_ylabel('Train ' + keys[1])
        ax[1].set_xticks(xTickRange)
        ax[1].set_yticks(yTickRange)
        ax[1].set_xticklabels(xTickLabels)
        ax[1].set_yticklabels(yTickLabels)

        fi.subplots_adjust(right = 0.8)
        cbar_ax = fi.add_axes([0.85, 0.15, 0.05, 0.7])
        fi.colorbar(im, cax = cbar_ax)
    return fi

def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None, scoreFun = 'f1_macro',
                        n_jobs=1, pre_dispatch = 'n_jobs', train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.
    Based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fi = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        scoring = scoreFun, train_sizes=train_sizes, pre_dispatch = pre_dispatch)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return fi

def lenientScore(yTrue, yPredicted, oldBinSize, newBinSize, scoreFun, **kwargs):
    if not isinstance(yTrue, np.ndarray):
        yTrue = yTrue.values

    downSampleFactor = m.ceil(newBinSize / oldBinSize)
    nSamples = len(yTrue)
    yTrueLenient = []
    yPredictedLenient = []

    for idx, currY in enumerate(yPredicted):
        if currY == 1 or currY == 2:
            yPredicted[idx + 1 : idx + downSampleFactor + 1] = 0

    for idx in range(nSamples)[0::downSampleFactor]:
        if 1 in yTrue[idx:idx + downSampleFactor]:
            yTrueLenient.append(1)
        elif 2 in yTrue[idx:idx + downSampleFactor]:
            yTrueLenient.append(2)
        else:
            yTrueLenient.append(0)
        if 1 in yPredicted[idx:idx + downSampleFactor]:
            yPredictedLenient.append(1)
        elif 2 in yPredicted[idx:idx + downSampleFactor]:
            yPredictedLenient.append(2)
        else:
            yPredictedLenient.append(0)

    return scoreFun(yTrueLenient, yPredictedLenient, **kwargs), yTrueLenient, yPredictedLenient

def long_form_df(DF, overrideColumns = None):
    longDF = pd.DataFrame(DF.unstack())
    longDF.reset_index(inplace=True)
    if overrideColumns is not None:
        longDF.columns = overrideColumns
    return longDF

def reloadPlot(filePath = None, message = "Please choose file(s)"):
    # get filename
    filename = ''

    if filePath == 'None':
        try:
            startPath = os.environ['DATA_ANALYSIS_LOCAL_DIR']
        except:
            startPath = 'Z:/data/rdarie/tempdata/Data-Analysis'
        filename = filedialog.askopenfilename(title = message, initialdir = startPath) # open window to get file name
    else:
        filename = filePath

    with open(filename, 'rb') as f:
        ax = pickle.load(f)

    #np.set_printoptions(precision=2)
    plt.show()
    return plt.gcf()

def loadRecruitmentCurve(folderPath, ignoreChans = []):

    with open(os.path.join(folderPath, 'metadata.p'), 'rb') as f:
        metadata = pickle.load(f)

    recruitmentCurve = pd.read_hdf(os.path.join(folderPath, 'recruitmentCurve.h5'), 'recruitment')
    dropList = pd.Series()
    for chanName in ignoreChans:
        dropIndices = pd.Series(recruitmentCurve.loc[recruitmentCurve['chanName'] == chanName, :].index)
        dropList = dropList.append(dropIndices, ignore_index = True)
    recruitmentCurve.drop(index = dropList, inplace = True)

    return metadata, recruitmentCurve

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem
