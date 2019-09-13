import pandas as pd
import os, pdb
from neo import (
    AnalogSignal, Event, Block,
    Segment, ChannelIndex, SpikeTrain, Unit)
import neo
import elephant as elph
from collections.abc import Iterable
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import itertools
#  import sys
#  import pickle
from copy import copy
import dataAnalysis.preproc.mdt_constants as mdt_constants
import quantities as pq
#  import argparse, linecache
from scipy import stats, signal, ndimage
import peakutils
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
import datetime
from datetime import datetime as dt
import json
from copy import copy
INSReferenceTime = pd.Timestamp('2018-03-01')

def fixMalformedJson(jsonString, jsonType=''):
    '''
    Adapted from Medtronic RDK Matlab code
    %--------------------------------------------------------------------------
    % Copyright (c) Medtronic, Inc. 2017
    %
    % MEDTRONIC CONFIDENTIAL -- This document is the property of Medtronic
    % PLC, and must be accounted for. Information herein is confidential trade
    % secret information. Do not reproduce it, reveal it to unauthorized 
    % persons, or send it outside Medtronic without proper authorization.
    %--------------------------------------------------------------------------
    %
    % File Name: fixMalformedJson.m
    % Autor: Ben Johnson (johnsb68)
    %
    % Description: This file contains the MATLAB function to fix a malformed 
    % Summit JSON File due to improperly closing the SummitSystem session.
    %
    % -------------------------------------------------------------------------
    %% Check and Apply Appropriate Fixes*************************************************
    '''
    
    jsonString = jsonString.replace('INF', 'Inf')
    
    numOpenSqua = jsonString.count('[')
    numOpenCurl = jsonString.count('{')
    numClosedSqua = jsonString.count(']')
    numClosedCurl = jsonString.count('}')

    if (
            (numOpenSqua != numClosedSqua) and
            (('Log' in jsonType) or ('Settings' in jsonType))):
        jsonStringOut = jsonString + ']'
    elif (numOpenSqua != numClosedSqua) or (numOpenCurl != numClosedCurl):
        nMissingSqua = numOpenSqua-numClosedSqua-1
        nMissingCurl = numOpenCurl-numClosedCurl-1
        jsonStringOut = (
            jsonString +
            '}' * nMissingCurl +
            ']' * nMissingSqua +
            '}]')
    else:
        jsonStringOut = jsonString
    return jsonStringOut


def getINSTDFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H', fs=500,
        forceRecalc=True, getInterpolated=True, upsampleRate=None,
        ):

    if not isinstance(sessionNames, Iterable):
        sessionNames = [sessionNames]

    tdSessions = []
    for idx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        try:
            if forceRecalc:
                raise(Exception('Debugging, always extract fresh'))

            if getInterpolated:
                csvFname = 'RawDataTD_interpolated.csv'
            else:
                csvFname = 'RawDataTD.csv'
            tdData = pd.read_csv(os.path.join(jsonPath, csvFname))

            #  loading from csv removes datetime formatting, recover it:
            tdData['microseconds'] = pd.to_timedelta(
                tdData['microseconds'], unit='us')
            tdData['time_master'] = pd.to_datetime(
                tdData['time_master'])

        except Exception:
            traceback.print_exc()

            try:
                with open(os.path.join(jsonPath, 'RawDataTD.json'), 'r') as f:
                    timeDomainJson = json.load(f)
            except Exception:
                with open(os.path.join(jsonPath, 'RawDataTD.json'), 'r') as f:
                    timeDomainJsonText = f.read()
                    timeDomainJsonText = fixMalformedJson(timeDomainJsonText)
                    timeDomainJson = json.loads(timeDomainJsonText)

            intersampleTickCount = int((1/fs) / (100e-6))

            timeDomainMeta = rcsa_helpers.extract_td_meta_data(timeDomainJson)
            
            # assume n channels is constant for all assembled sessions
            nChan = int(timeDomainMeta[0, 8])
                    
            timeDomainMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
                timeDomainMeta)

            timeDomainMeta, packetsNeededFixing =\
                rcsa_helpers.correct_meta_matrix_time_displacement(
                    timeDomainMeta, intersampleTickCount)
                    
            timeDomainMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
                timeDomainMeta)
            
            #  num_real_points, num_macro_rollovers, loss_as_scalar =\
            #      rcsa_helpers.calculate_statistics(
            #          timeDomainMeta, intersampleTickCount)

            timeDomainValues = rcsa_helpers.unpacker_td(
                timeDomainMeta, timeDomainJson, intersampleTickCount)

            #  save the noninterpolated files to disk
            tdData = rcsa_helpers.save_to_disk(
                timeDomainValues, os.path.join(
                    jsonPath, 'RawDataTD.csv'),
                time_format='full', data_type='td', num_cols=nChan)

            tdData['t'] = (
                tdData['actual_time'] - INSReferenceTime) / (
                    datetime.timedelta(seconds=1))
                    
            
            tdData = tdData.drop_duplicates(
                ['t']
                ).sort_values('t').reset_index(drop=True)

            if getInterpolated:
                if upsampleRate is not None:
                    fs = fs * upsampleRate
                uniformT = np.arange(
                    tdData['t'].iloc[0],
                    tdData['t'].iloc[-1] + float(fs) ** (-1),
                    float(fs) ** (-1))
                channelsPresent = [
                    i for i in tdData.columns if 'channel_' in i]
                channelsPresent += [
                    'time_master', 'microseconds']
                #  convert to floats before interpolating
                tdData['microseconds'] = tdData['microseconds'] / (
                    datetime.timedelta(microseconds=1))
                tdData['time_master'] = (
                    tdData['time_master'] - pd.Timestamp('2000-03-01')) / (
                    datetime.timedelta(seconds=1))
                # pdb.set_trace()
                tdData = hf.interpolateDF(
                    tdData, uniformT, x='t', kind='cubic',
                    columns=channelsPresent, fill_value=(0, 0))
                #  interpolating converts to floats, recover
                tdData['microseconds'] = pd.to_timedelta(
                    tdData['microseconds'], unit='us')
                tdData['time_master'] = pd.to_datetime(
                    tdData['time_master'], unit='s',
                    origin=pd.Timestamp('2000-03-01'))
                #  tdData['actual_time'] = tdData['time_master'] + (
                #      tdData['microseconds'])
                
            tdData['trialSegment'] = idx

            if getInterpolated:
                tdData.to_csv(
                    os.path.join(jsonPath, 'RawDataTD_interpolated.csv'))
            else:
                tdData.to_csv(
                    os.path.join(jsonPath, 'RawDataTD.csv'))

        tdSessions.append(tdData)

    td = {
        'data': pd.concat(tdSessions, ignore_index=True),
        't': None
        }
    
    td['t'] = td['data']['t']

    td['data']['INSTime'] = td['data']['t']
    td['INSTime'] = td['t']
    
    return td


def getINSTimeSyncFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        forceRecalc=True
        ):

    if not isinstance(sessionNames, Iterable):
        sessionNames = [sessionNames]
    tsSessions = []
    for idx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        try:
            if forceRecalc:
                raise(Exception('Debugging, always extract fresh'))

            timeSyncData = pd.read_csv(os.path.join(jsonPath, 'TimeSync.csv'))
            #  loading from csv removes datetime formatting, recover it:
            timeSyncData['microseconds'] = pd.to_timedelta(
                timeSyncData['microseconds'], unit='us')
            timeSyncData['time_master'] = pd.to_datetime(
                timeSyncData['time_master'])
            timeSyncData['actual_time'] = pd.to_datetime(
                timeSyncData['actual_time'])

        except Exception:
            traceback.print_exc()
            try:
                with open(os.path.join(jsonPath, 'TimeSync.json'), 'r') as f:
                    timeSync = json.load(f)[0]
            except Exception:
                with open(os.path.join(jsonPath, 'TimeSync.json'), 'r') as f:
                    timeSyncText = f.read()
                    timeSyncText = fixMalformedJson(timeSyncText)
                    timeSync = json.loads(timeSyncText)[0]
                    
            
            timeSyncData = rcsa_helpers.extract_time_sync_meta_data(timeSync)

            timeSyncData['trialSegment'] = idx
            timeSyncData.to_csv(os.path.join(jsonPath, 'TimeSync.csv'))

        timeSyncData['t'] = (
            timeSyncData['actual_time'] - INSReferenceTime) / (
                datetime.timedelta(seconds=1))
        timeSyncData.to_csv(os.path.join(jsonPath, 'TimeSync.csv'))
        tsSessions.append(timeSyncData)

    allTimeSync = pd.concat(tsSessions, ignore_index=True)
    return allTimeSync


def getINSAccelFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H', fs=64,
        forceRecalc=True, getInterpolated=True
        ):

    if not isinstance(sessionNames, Iterable):
        sessionNames = [sessionNames]

    accelSessions = []
    for idx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        try:
            if forceRecalc:
                raise(Exception('Debugging, always extract fresh'))
                
            if getInterpolated:
                csvFname = 'RawDataAccel_interpolated.csv'
            else:
                csvFname = 'RawDataAccel.csv'
            accelData = pd.read_csv(os.path.join(jsonPath, csvFname))

            #  loading from csv removes datetime formatting, recover it:
            accelData['microseconds'] = pd.to_timedelta(
                accelData['microseconds'], unit='us')
            accelData['time_master'] = pd.to_datetime(
                accelData['time_master'])

        except Exception:
            traceback.print_exc()

            try:
                with open(os.path.join(jsonPath, 'RawDataAccel.json'), 'r') as f:
                    accelJson = json.load(f)
            except Exception:
                with open(os.path.join(jsonPath, 'RawDataAccel.json'), 'r') as f:
                    accelJsonText = f.read()
                    accelJsonText = fixMalformedJson(accelJsonText)
                    accelJson = json.loads(accelJsonText)

            intersampleTickCount = int((1/fs) / (100e-6))
            accelMeta = rcsa_helpers.extract_accel_meta_data(accelJson)

            accelMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
                accelMeta)

            accelMeta, packetsNeededFixing =\
                rcsa_helpers.correct_meta_matrix_time_displacement(
                    accelMeta, intersampleTickCount)

            accelMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
                accelMeta)

            accelDataValues = rcsa_helpers.unpacker_accel(
                accelMeta, accelJson, intersampleTickCount)
                
            #  save the noninterpolated files to disk
            accelData = rcsa_helpers.save_to_disk(
                accelDataValues, os.path.join(
                    jsonPath, 'RawDataAccel.csv'),
                time_format='full', data_type='accel')

            accelData['t'] = (
                accelData['actual_time'] - INSReferenceTime) / (
                    datetime.timedelta(seconds=1))
            accelData = accelData.drop_duplicates(
                ['t']
                ).sort_values('t').reset_index(drop=True)
            
            if getInterpolated:
                uniformT = np.arange(
                    accelData['t'].iloc[0],
                    accelData['t'].iloc[-1] + 1/fs,
                    1/fs)
                channelsPresent = [
                    i for i in accelData.columns if 'accel_' in i]
                channelsPresent += [
                    'time_master', 'microseconds']

                #  convert to floats before interpolating
                accelData['microseconds'] = accelData['microseconds'] / (
                    datetime.timedelta(microseconds=1))
                accelData['time_master'] = (
                    accelData['time_master'] - pd.Timestamp('2000-03-01')) / (
                    datetime.timedelta(seconds=1))
                accelData = hf.interpolateDF(
                    accelData, uniformT, x='t',
                    columns=channelsPresent, fill_value=(0, 0))
                #  interpolating converts to floats, recover
                accelData['microseconds'] = pd.to_timedelta(
                    accelData['microseconds'], unit='us')
                accelData['time_master'] = pd.to_datetime(
                    accelData['time_master'], unit='s',
                    origin=pd.Timestamp('2000-03-01'))

            inertia = accelData['accel_x']**2 +\
                accelData['accel_y']**2 +\
                accelData['accel_z']**2
            inertia = inertia.apply(np.sqrt)
            accelData['inertia'] = inertia

            accelData['trialSegment'] = idx

            if getInterpolated:
                accelData.to_csv(
                    os.path.join(jsonPath, 'RawDataAccel_interpolated.csv'))
            else:
                accelData.to_csv(
                    os.path.join(jsonPath, 'RawDataAccel.csv'))

        accelSessions.append(accelData)

    accel = {
        'data': pd.concat(accelSessions, ignore_index=True),
        't': None
        }

    accel['t'] = accel['data']['t']
    accel['INSTime'] = accel['data']['t']
    accel['data']['INSTime'] = accel['t']
    return accel


def realignINSTimestamps(
        dataStruct, trialSegment, alignmentFactor
        ):

    if isinstance(dataStruct, pd.DataFrame):
        segmentMask = dataStruct['trialSegment'] == trialSegment
        dataStruct.loc[segmentMask, 't'] = (
            dataStruct.loc[segmentMask, 'microseconds'] + alignmentFactor) / (
                pd.Timedelta(1, unit='s'))

        if 'INSTime' in dataStruct.columns:
            dataStruct.loc[
                segmentMask, 'INSTime'] = dataStruct.loc[segmentMask, 't']

    elif isinstance(dataStruct, dict):
        segmentMask = dataStruct['data']['trialSegment'] == trialSegment
        dataStruct['data'].loc[segmentMask, 't'] = (
            dataStruct['data'].loc[
                segmentMask, 'microseconds'] + alignmentFactor) / (
                    pd.Timedelta(1, unit='s'))

        if 'INSTime' in dataStruct['data'].columns:
            dataStruct['data'].loc[
                segmentMask, 'INSTime'] = dataStruct[
                    'data'].loc[segmentMask, 't']

        #  ['INSTime'] is a reference to ['t']  for the dict
        dataStruct['t'].loc[segmentMask] = dataStruct[
            'data']['t'].loc[segmentMask]
        if 'INSTime' in dataStruct.keys():
            dataStruct['INSTime'].loc[
                segmentMask] = dataStruct['t'].loc[segmentMask]
    return dataStruct


def plotHUTtoINS(
        td, accel, plotStimStatus,
        tStartTD, tStopTD,
        tStartStim=None, tStopStim=None,
        tdX='t', stimX='INSTime', dataCols=['channel_0', 'channel_1'],
        sharex=True,
        plotBlocking=True, plotEachPacket=False,
        annotatePacketName=False
        ):
    # check match between INS time and HUT time
    if stimX == 'INSTime':
        fig, ax = plt.subplots(3, 1, sharex=sharex)
        unitsCorrection = 1
        tStartStim = tStartTD
        tStopStim = tStopTD
    elif stimX == 'HostUnixTime':
        fig, ax = plt.subplots(3, 1)
        unitsCorrection = 1e-3

    plotMaskTD = (td[tdX] > tStartTD) & (td[tdX] < tStopTD)
    plotMaskAccel = (accel[tdX] > tStartTD) & (accel[tdX] < tStopTD)
    plotMaskStim = (plotStimStatus[stimX] * unitsCorrection > tStartStim) &\
        (plotStimStatus[stimX] * unitsCorrection < tStopStim)

    if plotEachPacket:
        tdIterator = td['data'].loc[plotMaskTD, :].groupby('packetIdx')
    else:
        tdIterator = enumerate([td['data'].loc[plotMaskTD, :]])

    for name, group in tdIterator:
        for columnName in dataCols:
            ax[0].plot(
                group.loc[:, tdX],
                group.loc[:, columnName],
                '-', label=columnName)
            if annotatePacketName:
                ax[0].text(
                    group[tdX].iloc[-1],
                    group[columnName].iloc[-1],
                    '{}'.format(name))
    #  ax[0].legend()
    ax[0].set_title('INS Data')
    ax[1].set_ylabel('TD Data')

    if plotEachPacket:
        accelIterator = accel['data'].loc[plotMaskAccel, :].groupby('packetIdx')
    else:
        accelIterator = enumerate([accel['data'].loc[plotMaskAccel, :]])

    for name, group in accelIterator:
        for columnName in ['inertia']:
            ax[1].plot(
                group.loc[:, tdX],
                group.loc[:, columnName],
                '-', label=columnName)
    #  ax[1].legend()
    ax[1].set_ylabel('Inertia')
    progAmpNames = [
        'program{}_amplitude'.format(progIdx) for progIdx in range(4)]
    #  progPWNames = [
    #      'program{}_pw'.format(progIdx) for progIdx in range(4)]
    statusAx = ax[2].twinx()
    for columnName in progAmpNames:
        ax[2].plot(
            plotStimStatus[stimX].loc[plotMaskStim]*unitsCorrection,
            plotStimStatus.loc[plotMaskStim, columnName],
            lw=2.5, label=columnName)
    
    statusAx.plot(
        plotStimStatus[stimX].loc[plotMaskStim]*unitsCorrection,
        plotStimStatus.loc[plotMaskStim, 'amplitudeIncrease'],
        'c--', label='amplitudeIncrease')
    statusAx.plot(
        plotStimStatus[stimX].loc[plotMaskStim]*unitsCorrection,
        plotStimStatus.loc[plotMaskStim, 'therapyStatus'],
        '--', label='therapyStatus')
    statusAx.plot(
        plotStimStatus[stimX].loc[plotMaskStim]*unitsCorrection,
        plotStimStatus.loc[plotMaskStim, 'RateInHz']/100,
        '--', label='RateInHz')
    statusAx.legend()
    ax[2].set_ylabel('Stim Amplitude (mA)')
    ax[2].set_xlabel('INS Time (sec)')
    plt.suptitle('Stim State')
    plt.show(block=plotBlocking)
    return


def line_picker(line, mouseevent):
    """
    find the points within a certain distance from the mouseclick in
    data coords and attach some extra attributes, pickx and picky
    which are the data points that were picked
    """
    if mouseevent.xdata is None:
        return False, dict()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    maxd = 0.1
    d = np.sqrt(
        (xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)
    dmask = (d < maxd) & (d == min(d))
    ind, = np.nonzero(dmask)
    
    if len(ind):
        pickx = xdata[ind]
        picky = ydata[ind]
        props = dict(ind=ind, pickx=pickx, picky=picky)
        return True, props
    else:
        return False, dict()


def peekAtTaps(
        td, accel,
        channelData, trialIdx,
        tapDetectOpts, sessionTapRangesNSP,
        onlyPlotDetectChan=True,
        insX='t', plotBlocking=True,
        allTapTimestampsINS=None,
        allTapTimestampsNSP=None,
        segmentsToPlot=None):
    sns.set_style('darkgrid')
    if segmentsToPlot is None:
        segmentsToPlot = pd.unique(td['data']['trialSegment'])

    tempClick = {
        'ins': [],
        'nsp': []
        }
    clickDict = {
        i: tempClick
        for i in segmentsToPlot}
    
    def onpick(event):
        
        if 'INS' in event.artist.axes.get_title():
            tempClick['ins'].append(event.pickx[0])
            print('Clicked on ins {}'.format(event.pickx[0]))
        elif 'NSP' in event.artist.axes.get_title():
            tempClick['nsp'].append(event.pickx[0])
            print('Clicked on nsp {}'.format(event.pickx[0]))

    for trialSegment in segmentsToPlot:
        #  NSP plotting Bounds
        trialSegment = int(trialSegment)
        tStartNSP = (
            sessionTapRangesNSP[trialIdx][trialSegment]['timeRanges'][0])
        tStopNSP = (
            sessionTapRangesNSP[trialIdx][trialSegment]['timeRanges'][1])
        tDiffNSP = tStopNSP - tStartNSP
        #  Set up figures
        fig = plt.figure(tight_layout=True)
        ax = [None for i in range(3)]
        ax[0] = fig.add_subplot(311)
        ax[1] = fig.add_subplot(312, sharex=ax[0])
        if insX == 't':
            # INS plotting bounds
            tStartINS = td['t'].iloc[0]
            tStopINS = td['t'].iloc[-1]
            for thisRange in tapDetectOpts[
                    trialIdx][trialSegment]['timeRanges']:
                tStartINS = max(tStartINS, thisRange[0])
                tStopINS = min(tStopINS, thisRange[1])
            #  make it so that the total extent always matches, for easy comparison
            tDiffINS = max(tStopINS - tStartINS, tDiffNSP)
            tStopINS = tStartINS + tDiffINS
            ax[2] = fig.add_subplot(313)
        else:
            ax[2] = fig.add_subplot(313, sharex=ax[0])
            tStartINS = tStartNSP
            tStopINS = tStopNSP
        
        insTapsAx = ax[1]
        twinAx = ax[2].twiny()
        dataColMask = np.array(['ins_td' in i for i in td['data'].columns])
        if 'accChan' in tapDetectOpts[trialIdx][trialSegment].keys():
            detectOn = 'acc'
            accAx = ax[1]
            tdAx = ax[0]
            accAx.get_shared_x_axes().join(accAx, twinAx)
            accAx.get_shared_y_axes().join(accAx, twinAx)
            accAxLineStyle = '.-'
            tdAxLineStyle = '-'
            if onlyPlotDetectChan:
                accColumnNames = [tapDetectOpts[trialIdx][trialSegment]['accChan']]
            else:
                accColumnNames = [
                    'ins_accx', 'ins_accy',
                    'ins_accz', 'ins_accinertia']
            tdColumnNames = td['data'].columns[dataColMask]
        else:
            detectOn = 'td'
            accAx = ax[0]
            tdAx = ax[1]
            tdAx.get_shared_x_axes().join(tdAx, twinAx)
            tdAx.get_shared_y_axes().join(tdAx, twinAx)
            accAxLineStyle = '-'
            tdAxLineStyle = '.-'
            if onlyPlotDetectChan:
                tdColumnNames = [tapDetectOpts[trialIdx][trialSegment]['tdChan']]
            else:
                tdColumnNames = td['data'].columns[dataColMask]
            accColumnNames = [
                'ins_accx', 'ins_accy',
                'ins_accz', 'ins_accinertia']

        plotMask = (accel[insX] > tStartINS) & (accel[insX] < tStopINS)
        plotMaskTD = (td[insX] > tStartINS) & (td[insX] < tStopINS)

        for columnName in accColumnNames:
            accAx.plot(
                accel[insX].loc[plotMask],
                stats.zscore(accel['data'].loc[plotMask, columnName]),
                accAxLineStyle,
                label=columnName, picker=line_picker)
            if detectOn == 'acc':
                twinAx.plot(
                    accel[insX].loc[plotMask],
                    stats.zscore(accel['data'].loc[plotMask, columnName])
                    )
        accAx.set_title('INS Acc Segment {}'.format(trialSegment))
        accAx.set_ylabel('Z Score (a.u.)')

        for columnName in tdColumnNames:
            tdAx.plot(
                td[insX].loc[plotMaskTD],
                stats.zscore(td['data'].loc[plotMaskTD, columnName]),
                tdAxLineStyle,
                label=columnName, picker=line_picker)
            if detectOn == 'td':
                twinAx.plot(
                    td[insX].loc[plotMaskTD],
                    stats.zscore(td['data'].loc[plotMaskTD, columnName])
                    )
        tdAx.set_ylabel('Z Score (a.u.)')
        tdAx.set_title('INS TD Segment {}'.format(trialSegment))

        if allTapTimestampsINS is not None:
            insTapsAx.plot(
                allTapTimestampsINS[trialSegment],
                allTapTimestampsINS[trialSegment] ** 0 - 1,
                'c*', label='tap peaks')
        
        xmin, xmax = ax[1].get_xlim()
        xTicks = np.arange(xmin, xmax, 0.05)
        ax[0].set_xticks(xTicks)
        ax[0].set_yticks([])
        ax[0].set_xticklabels([])
        ax[0].grid(which='major', color='b', alpha=0.75)
        ax[1].set_xticks(xTicks)
        ax[1].set_yticks([])
        ax[1].set_xticklabels([])
        ax[1].grid(which='major', color='b', alpha=0.75)
        twinAx.set_xticks(xTicks)
        twinAx.set_yticks([])
        twinAx.set_xticklabels([])
        twinAx.grid(which='major', color='b', alpha=0.75)
        tdAx.legend()
        accAx.legend()

        tNSPMask = (channelData['t'] > tStartNSP) & (channelData['t'] < tStopNSP)
        triggerTrace = channelData['data'].loc[tNSPMask, 'ainp7']
        
        ax[2].plot(
            channelData['t'].loc[tNSPMask].iloc[::30],
            stats.zscore(triggerTrace.iloc[::30]),
            'c', label='Analog Sync', picker=line_picker)
        xmin, xmax = ax[2].get_xlim()
        xTicks2 = np.arange(xmin, xmax, 0.05)
        ax[2].set_xticks(xTicks2)
        ax[2].set_yticks([])
        ax[2].set_xticklabels([])
        ax[2].grid(which='major', color='c', alpha=0.75)
        if allTapTimestampsNSP is not None:
            ax[2].plot(
                allTapTimestampsNSP[trialSegment],
                allTapTimestampsNSP[trialSegment] ** 0 - 1,
                'm*', label='tap peaks', picker=line_picker)

        ax[2].legend()
        ax[2].set_title('NSP Data')
        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show(block=plotBlocking)

        for key, value in tempClick.items():
            tempVal = pd.Series(value)
            tempClick[key] = tempVal.loc[tempVal.diff().fillna(1) > 100e-3]
            
        clickDict[trialSegment] = tempClick

        tempClick = {
            'ins': [],
            'nsp': []
            }
    return clickDict


def getINSTapTimestamp(
        td=None, accel=None,
        tapDetectOpts={}, plotting=False
        ):

    if 'timeRanges' in tapDetectOpts.keys():
        timeRanges = tapDetectOpts['timeRanges']
    else:
        timeRanges = None

    if 'keepIndex' in tapDetectOpts.keys():
        keepIndex = tapDetectOpts['keepIndex']
    else:
        keepIndex = slice(None)

    if 'iti' in tapDetectOpts.keys():
        iti = tapDetectOpts['iti']
    else:
        iti = 0.25
    #  itiWiggle = 0.05

    if 'accChan' in tapDetectOpts.keys():
        assert accel is not None

        if not isinstance(accel, dict):
            accel = {
                'data': accel,
                't': accel['t']
                }

        if timeRanges is None:
            tdMask = td['t'] > 0
        else:
            for idx, timeSegment in enumerate(timeRanges):
                if idx == 0:
                    accelMask = (accel['t'] > timeSegment[0]) & (
                        accel['t'] < timeSegment[1])
                else:
                    accelMask = accelMask | (
                        (accel['t'] > timeSegment[0]) & (
                            accel['t'] < timeSegment[1]))

        tapDetectSignal = accel['data'].loc[
            accelMask, tapDetectOpts['accChan']]
        
        # check that we've interpolated accel at this point?
        accelPeakIdx = hf.getTriggers(
            tapDetectSignal, iti=iti, fs=500, thres=tapDetectOpts['accThres'],
            edgeType='both', minAmp=None,
            expectedTime=None, keep_max=False, plotting=plotting)
        '''

        # minimum distance between triggers (units of samples), 5% wiggle room
        width = int(64 * iti * (1 - itiWiggle))
        ilocPeakIdx = peakutils.indexes(
            accel['data']['inertia_z'].loc[accelMask].values, thres=accThres,
            min_dist=width, thres_abs=True, keep_max=True)
        accelPeakIdx = accel['data']['inertia_z'].loc[accelMask].index[ilocPeakIdx]
        '''
        accelPeakIdx = accelPeakIdx[keepIndex]
        print('Accel Timestamps \n{}'.format(accel['t'].loc[accelPeakIdx]))

        tapTimestamps = accel['t'].loc[accelPeakIdx]
        peakIdx = accelPeakIdx

    if 'tdChan' in tapDetectOpts.keys():
        assert td is not None
        if not isinstance(td, dict):
            td = {
                'data': td,
                't': td['t']
                }

        if timeRanges is None:
            tdMask = td['t'] > 0
        else:
            for idx, timeSegment in enumerate(timeRanges):
                if idx == 0:
                    tdMask = (td['t'] > timeSegment[0]) & (
                        td['t'] < timeSegment[1])
                else:
                    tdMask = tdMask | (
                        (td['t'] > timeSegment[0]) & (
                            td['t'] < timeSegment[1]))

        tapDetectSignal = td['data'].loc[tdMask, tapDetectOpts['tdChan']]

        tdPeakIdx = hf.getTriggers(
            tapDetectSignal, iti=iti, fs=500, thres=tapDetectOpts['tdThres'],
            edgeType='both', minAmp=None,
            expectedTime=None, keep_max=False, plotting=plotting)
        '''

        # minimum distance between triggers (units of samples), 5% wiggle room
        width = int(500 * iti * (1 - itiWiggle))
        ilocPeakIdx = peakutils.indexes(
            tapDetectSignal.loc[tdMask].values, thres=tdThres,
            min_dist=width, thres_abs=True, keep_max=True)
        tdPeakIdx = tapDetectSignal.loc[tdMask].index[ilocPeakIdx]
        '''
        tdPeakIdx = tdPeakIdx[keepIndex]
        print('TD Timestamps \n{}'.format(td['t'].loc[tdPeakIdx]))

        tapTimestamps = td['t'].loc[tdPeakIdx]
        peakIdx = tdPeakIdx

    return tapTimestamps, peakIdx


def getHUTtoINSSyncFun(
        timeSyncData,
        degree=1, plotting=False,
        trialSegments=None,
        syncTo='HostUnixTime',
        chunkSize=None,
        ):
    if trialSegments is None:
        trialSegments = pd.unique(timeSyncData['trialSegment'])
    timeInterpFunHUTtoINS = [[] for i in trialSegments]
    timeSyncData['timeChunks'] = 0
    for trialSegment, tsDataSegment in timeSyncData.groupby('trialSegment'):
        if chunkSize is not None:
            nChunks = int(np.ceil(tsDataSegment.shape[0] / chunkSize))
            thisT = tsDataSegment['t']
            for i in range(nChunks):
                if i < (nChunks - 1):
                    tMask = (thisT >= i * chunkSize) & (thisT < (i + 1) * chunkSize)
                else:
                    tMask = (thisT >= i * chunkSize)
                tsDataSegment.loc[tMask, 'timeChunks'] = i
        for timeChunk, tsTimeChunk in tsDataSegment.groupby('timeChunks'):
            if degree > 0:
                synchPolyCoeffsHUTtoINS = np.polyfit(
                    x=tsTimeChunk[syncTo].values,
                    y=tsTimeChunk['t'].values,
                    deg=degree)
            else:
                # assume no clock drift
                timeOffset = (
                    tsTimeChunk['t'].values -
                    tsTimeChunk[syncTo].values * 1e-3)
                synchPolyCoeffsHUTtoINS = np.array([1e-3, np.mean(timeOffset)])
            
            thisFun = np.poly1d(synchPolyCoeffsHUTtoINS)
            thisInterpDict = {
                'fun': thisFun,
                'tStart': tsTimeChunk['t'].iloc[0],
                'tStop': tsTimeChunk['t'].iloc[-1],
                'tStartHUT': tsTimeChunk[syncTo].iloc[0],
                'tStopHUT': tsTimeChunk[syncTo].iloc[-1]
            }
            timeInterpFunHUTtoINS[trialSegment].append(thisInterpDict)
            if plotting:
                plt.plot(
                    tsTimeChunk[syncTo].values * 1e-3,
                    tsTimeChunk['t'].values, 'bo',
                    label='original')
                plt.plot(
                    tsTimeChunk[syncTo].values * 1e-3,
                    thisFun(tsTimeChunk[syncTo].values), 'r-',
                    label='first degree polynomial fit')
                plt.xlabel('Host Unix Time (msec)')
                plt.ylabel('INS Time (sec)')
                plt.title('HUT Synchronization')
                plt.legend()
                plt.show()
                resid = (
                    thisFun(tsTimeChunk[syncTo].values) -
                    tsTimeChunk['t'].values)
                ax = sns.distplot(resid)
                ax.set_title('Residuals from HUT Regression')
                ax.set_xlabel('Time (sec)')
                plt.show()
                plt.plot(
                    tsTimeChunk['t'].values,
                    resid, label='Residuals from HUT Regression')
                plt.title('Residuals from HUT Regression')
                plt.xlabel('Time (sec)')
                plt.ylabel('Time (sec)')
                plt.show()
    return timeInterpFunHUTtoINS


def synchronizeHUTtoINS(
        insDF, trialSegment, interpFunDictList,
        ):
    if 'INSTime' not in insDF.columns:
        insDF['INSTime'] = np.nan
    if 'trialSegment' in insDF.columns:
        segmentMask = insDF['trialSegment'] == trialSegment
    else:
        #  event dataframes don't have an explicit trialSegment
        segmentMask = hf.getStimSerialTrialSegMask(insDF, trialSegment)
    for idx, interpFunDict in enumerate(interpFunDictList):
        interpFun = interpFunDict['fun']
        
        if idx == 0:
            timeChunkIdx = insDF.loc[segmentMask, :].index[
                (insDF.loc[segmentMask, 'HostUnixTime'] < interpFunDict['tStopHUT'])
                ]
        elif idx == (len(interpFunDictList) - 1):
            timeChunkIdx = insDF.loc[segmentMask, :].index[
                (insDF.loc[segmentMask, 'HostUnixTime'] >= interpFunDict['tStartHUT'])
                ]
        else:
            timeChunkIdx = insDF.loc[segmentMask, :].index[
                (insDF.loc[segmentMask, 'HostUnixTime'] >= interpFunDict['tStartHUT']) &
                (insDF.loc[segmentMask, 'HostUnixTime'] < interpFunDict['tStopHUT'])
                ]
        #  if not timeChunkIdx.any():
        #      lookAhead = 1
        #      
        #      while not timeChunkIdx.any():
        #          nextInterpFunDict = interpFunDictList[idx + lookAhead]
        #          interpFun = nextInterpFunDict['fun']
        #          
        #          timeChunkIdx = insDF.loc[segmentMask, :].index[
        #              (insDF.loc[segmentMask, 'HostUnixTime'] <= nextInterpFunDict['tStartHUT'])
        #              ]
        #          lookAhead += 1
        if timeChunkIdx.any():
            insDF.loc[timeChunkIdx, 'INSTime'] = interpFun(
                insDF.loc[timeChunkIdx, 'HostUnixTime'])
    return insDF


def getINSStimLogFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        absoluteStartTime=None, logForm='serial'):
    allStimStatus = []
    for idx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
            stimLog = json.load(f)
        if logForm == 'serial':
            stimStatus = rcsa_helpers.extract_stim_meta_data_events(
                stimLog, trialSegment=idx)
        allStimStatus.append(stimStatus)
    allStimStatusDF = pd.concat(allStimStatus, ignore_index=True)
    return allStimStatusDF


def stimStatusSerialtoLong(
        stimStSer, idxT='t', namePrefix='ins_', expandCols=[],
        deriveCols=[], progAmpNames=[], dropDuplicates=True,
        amplitudeCatBins=4):
    fullExpandCols = copy(expandCols)
    #  fixes issue with 'program' and 'amplitude' showing up unrequested
    if 'program' not in expandCols:
        fullExpandCols.append('program')
    if 'activeGroup' not in expandCols:
        fullExpandCols.append('activeGroup')
    #
    stimStLong = pd.DataFrame(
        index=stimStSer.index, columns=fullExpandCols + [idxT])
    #  fill req columns
    stimStLong[idxT] = stimStSer[idxT]
    for pName in fullExpandCols:
        #  print(pName)
        stimStLong[pName] = np.nan
        pMask = stimStSer[namePrefix + 'property'] == pName
        pValues = stimStSer.loc[pMask, namePrefix + 'value']
        stimStLong.loc[pMask, pName] = pValues
        if pName == 'movement':
            stimStLong[pName].iloc[0] = 0
        stimStLong[pName].fillna(
            method='ffill', inplace=True)
        stimStLong[pName].fillna(
            method='bfill', inplace=True)
    #
    debugPlot = False
    if debugPlot:
        stimCat = pd.concat((stimStLong, stimStSer), axis=1)
    #
    for idx, pName in enumerate(progAmpNames):
        stimStLong[pName] = np.nan
        pMask = (stimStSer[namePrefix + 'property'] == 'amplitude') & (
            stimStLong['program'] == idx)
        stimStLong.loc[pMask, pName] = stimStSer.loc[pMask, namePrefix + 'value']
        stimStLong[pName].fillna(method='ffill', inplace=True)
        stimStLong[pName].fillna(value=0, inplace=True)
    
    if dropDuplicates:
        stimStLong.drop_duplicates(subset=idxT, keep='last', inplace=True)
    #
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        plt.show()
        #
    ampIncrease = pd.Series(False, index=stimStLong.index)
    ampChange = pd.Series(False, index=stimStLong.index)
    for idx, pName in enumerate(progAmpNames):
        ampIncrease = ampIncrease | (stimStLong[pName].diff().fillna(0) > 0)
        ampChange = ampChange | (stimStLong[pName].diff().fillna(0) != 0)
        if debugPlot:
            plt.plot(stimStLong[pName].diff().fillna(0), label=pName)
    #
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        ampIncrease.astype(float).plot(style='ro')
        ampChange.astype(float).plot(style='go')
        plt.legend()
        plt.show()
    #
    if 'amplitudeRound' in deriveCols:
        stimStLong['amplitudeRound'] = (
            ampIncrease.astype(np.float).cumsum())
    if 'movementRound' in deriveCols:
        stimStLong['movementRound'] = (
            stimStLong['movement'].abs().cumsum())
    if 'amplitude' in deriveCols:
        stimStLong['amplitude'] = (
            stimStLong[progAmpNames].sum(axis=1))
    if 'amplitudeCat' in deriveCols:
        ampsForSum = copy(stimStLong[progAmpNames])
        for colName in ampsForSum.columns:
            if ampsForSum[colName].max() > 0:
                ampsForSum[colName] = pd.cut(
                    ampsForSum[colName], bins=amplitudeCatBins, labels=False)
            else:
                ampsForSum[colName] = pd.cut(
                    ampsForSum[colName], bins=1, labels=False)
        stimStLong['amplitudeCat'] = (
            ampsForSum.sum(axis=1))
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        (10 * stimStLong['amplitudeRound'] / (stimStLong['amplitudeRound'].max())).plot()
        plt.show()
    return stimStLong


def getINSDeviceConfig(
        folderPath, sessionName, deviceName='DeviceNPC700373H'):
    jsonPath = os.path.join(folderPath, sessionName, deviceName)

    with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
        deviceSettings = json.load(f)
    with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
        stimLog = json.load(f)

    progIndices = list(range(4))
    groupIndices = list(range(4))
    elecIndices = list(range(17))

    electrodeConfiguration = [
        [{'cathodes': [], 'anodes': []} for i in progIndices] for j in groupIndices]
    
    dfIndex = pd.MultiIndex.from_product(
        [groupIndices, progIndices],
        names=['group', 'program'])

    electrodeStatus = pd.DataFrame(index=dfIndex, columns=elecIndices)
    electrodeType = pd.DataFrame(index=dfIndex, columns=elecIndices)

    for groupIdx in groupIndices:
        groupTherapyConfig = deviceSettings[0][
            'TherapyConfigGroup{}'.format(groupIdx)]
        groupPrograms = groupTherapyConfig['programs']
        for progIdx in progIndices:
            electrodeConfiguration[groupIdx][progIdx].update({
                'cycleOffTime': groupTherapyConfig['cycleOffTime'],
                'cycleOnTime': groupTherapyConfig['cycleOnTime'],
                'cyclingEnabled': groupTherapyConfig['cyclingEnabled']
                })
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
    
    #  process senseInfo
    senseInfo = pd.DataFrame(
        deviceSettings[0]['SensingConfig']['timeDomainChannels'])
    senseInfo['sampleRate'] = senseInfo['sampleRate'].apply(
        lambda x: mdt_constants.sampleRate[x])
    senseInfo['minusInput'] = senseInfo['minusInput'].apply(
        lambda x: mdt_constants.muxIdx[x])
    senseInfo['plusInput'] = senseInfo['plusInput'].apply(
        lambda x: mdt_constants.muxIdx[x])
    senseInfo.loc[(2, 3), ('minusInput', 'plusInput')] += 8
    senseInfo.loc[:, ('minusInput', 'plusInput')].fillna(17, inplace=True)
    senseInfo = senseInfo.loc[senseInfo['sampleRate'].notnull(), :]
    senseInfo.reset_index(inplace=True)
    senseInfo.rename(columns={'index': 'senseChan'}, inplace=True)
    return electrodeConfiguration, senseInfo


def preprocINS(
        trialFilesStim,
        insDataFilename,
        plottingFigures=False,
        plotBlocking=True):
    print('Preprocessing')
    jsonBaseFolder = trialFilesStim['folderPath']
    jsonSessionNames = trialFilesStim['jsonSessionNames']
    #
    stimStatusSerial = getINSStimLogFromJson(
        jsonBaseFolder, jsonSessionNames)
    #
    elecConfiguration, senseInfo = (
        getINSDeviceConfig(jsonBaseFolder, jsonSessionNames[0])
        )
    #
    fsList = senseInfo['sampleRate'].unique()
    assert fsList.size == 1
    fs = fsList[0]
    if 'upsampleRate' in trialFilesStim:
        upsampleRate = trialFilesStim['upsampleRate']
    else:
        upsampleRate = None
    td = getINSTDFromJson(
        jsonBaseFolder, jsonSessionNames, getInterpolated=True,
        fs=fs, upsampleRate=upsampleRate,
        forceRecalc=trialFilesStim['forceRecalc'])
    renamer = {}
    tdDataCols = []
    for colName in td['data'].columns:
        if 'channel_' in colName:
            idx = int(colName[-1])
            updatedName = 'channel_{}'.format(senseInfo.loc[idx, 'senseChan'])
            tdDataCols.append(updatedName)
            renamer.update({colName: updatedName})
    td['data'].rename(columns=renamer, inplace=True)

    accel = getINSAccelFromJson(
        jsonBaseFolder, jsonSessionNames, getInterpolated=True,
        forceRecalc=trialFilesStim['forceRecalc'])

    timeSync = getINSTimeSyncFromJson(
        jsonBaseFolder, jsonSessionNames,
        forceRecalc=True)
    #  packets are aligned to INSReferenceTime, for convenience
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
    
        accel = realignINSTimestamps(
            accel, trialSegment, alignmentFactor.loc['accel'])
        td = realignINSTimestamps(
            td, trialSegment, alignmentFactor.loc['td'])
        timeSync = realignINSTimestamps(
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
    stimStatus = stimStatusSerialtoLong(
        stimStatusSerial, idxT='HostUnixTime', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    HUTChunkSize = 100
    interpFunHUTtoINS = getHUTtoINSSyncFun(
        timeSync, degree=1, syncTo='PacketGenTime', chunkSize=HUTChunkSize)
    for trialSegment in pd.unique(td['data']['trialSegment']):
        stimStatus = synchronizeHUTtoINS(
            stimStatus, trialSegment, interpFunHUTtoINS[trialSegment])
        stimStatusSerial = synchronizeHUTtoINS(
            stimStatusSerial, trialSegment, interpFunHUTtoINS[trialSegment])
    #  sync Host PC Unix time to NSP
    HUTtoINSPlotting = False
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
        plotHUTtoINS(
            td, accel, plotStimStatus,
            tStartTD, tStopTD,
            sharex=True, dataCols=tdDataCols,
            plotBlocking=plotBlocking
            )
    #
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
        stimSpikesDF.sort_values('t', kind='mergesort', inplace=True)
        stimSpikesDF.reset_index(drop=True, inplace=True)
        stimSpikesDF['nextT'] = stimSpikesDF['t'].shift(-1)
        tDiff = (stimSpikesDF['nextT'] - stimSpikesDF['endTime']).fillna(1)
        overShotEnd = tDiff <= 0
        stimSpikesDF.loc[overShotEnd, 'endTime'] = (
            stimSpikesDF.loc[overShotEnd, 'nextT'] -
            stimSpikesDF.loc[overShotEnd, 'ratePeriod'])
        onsetEvents = pd.melt(
            stimSpikesDF,
            id_vars=['t'],
            value_vars=['group', 'program', 'RateInHz', 'pulseWidth', 'amplitude'],
            var_name='ins_property', value_name='ins_value')
        onsetEvents.rename(columns={'t': 'INSTime'}, inplace=True)
        onsetEvents.loc[onsetEvents['ins_property'] == 'group', 'ins_property'] = 'activeGroup'
        #ampOnsets = onsetEvents.loc[onsetEvents['ins_property'] == 'amplitude', :]
        #stimSpikesDF.loc[:, ['t', 'endTime']]
        offsetEvents = pd.melt(
            stimSpikesDF,
            id_vars=['endTime'],
            value_vars=['amplitude'],
            var_name='ins_property', value_name='ins_value')
        offsetEvents.loc[
            offsetEvents['ins_property'] == 'amplitude',
            'ins_value'] = 0
        offsetEvents.rename(columns={'endTime': 'INSTime'}, inplace=True)
        newEvents = (
            pd.concat([onsetEvents, offsetEvents], ignore_index=True)
            .sort_values('INSTime', kind='mergesort')
            .reset_index(drop=True)
            )
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
                .isin(['trialSegment', 'therapyStatus', 'amplitude'])) |
            (stimStatusSerial['INSTime'] < firstNonZeroAmplitudeTime)
            )
        newStimStatusSerial = stimStatusSerial.loc[
            keepMask,
            ['INSTime', 'ins_property', 'ins_value']]
        newStimStatusSerial.loc[newStimStatusSerial['ins_property'] == 'amplitude', 'ins_property'] = 'amplitudeFromJSON'
        newStimStatusSerial = pd.concat(
            [newStimStatusSerial, newEvents])
        newStimStatusSerial = (
            newStimStatusSerial
            .sort_values('INSTime', kind='mergesort')
            .reset_index(drop=True)
            )
        newStimEvents = ns5.eventDataFrameToEvents(
            newStimStatusSerial, idxT='INSTime',
            eventName='seg0_',
            annCol=['ins_property', 'ins_value']
            )
        
        block.segments[0].events = newStimEvents
        for ev in newStimEvents:
            ev.segment = block.segments[0]
        stimStatusSerial = newStimStatusSerial
    else:
        # if not detecting stim onsets, change the filename to reflect that
        insDataFilename = insDataFilename.replace('.nix', '_orig_stim.nix')
    # make labels
    labelNames = [
        'RateInHz', 'program', 'therapyStatus',
        'activeGroup', 'pulseWidth']
    labelIndices = []
    for labelName in labelNames:
        placeHolder = (
            stimStatusSerial
            .loc[
                stimStatusSerial['ins_property'] == labelName,
                'ins_value']
            .reset_index())
        labelIndices += (
            placeHolder
            .loc[placeHolder['ins_value'].diff() != 0, 'index']
            .to_list())
    fullLabelNames = ['amplitude', 'amplitudeFromJSON']
    for labelName in fullLabelNames:
        placeHolder = (
            stimStatusSerial
            .loc[
                stimStatusSerial['ins_property'] == labelName,
                'ins_value']
            .reset_index())
        labelIndices += (
            placeHolder
            .loc[:, 'index']
            .to_list())
    labelIndices = np.sort(labelIndices, kind='mergesort')
    statusLabels = stimStatusSerial.loc[labelIndices, :]
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in statusLabels.loc[:, ['ins_property', 'ins_value']].iterrows()])
    concatEvents = Event(
        name='seg0_concatenatedEvents',
        times=statusLabels['INSTime'].to_numpy() * pq.s,
        labels=concatLabels
        )
    block.segments[0].events.append(concatEvents)
    concatEvents.segment = block.segments[0]
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
        offsetFromPeak=None,
        overrideStartTimes=None,
        cyclePeriodCorrection=18e-3,
        stimDetectOptsByChannel=None,
        plotAnomalies=False,
        predictSlots=True, snapToGrid=False,
        treatAsSinglePulses=False,
        spikeWindow=[-32, 64],
        plotting=[]):

    segIdx = 0
    seg = block.segments[segIdx]
    fs = seg.analogsignals[0].sampling_rate
    # 
    # WIP: treat as single at the stimStatus level
    tdDF, accelDF, stimStatus = unpackINSBlock(
        block, convertStimToSinglePulses=False)
    
    #  assume a fixed delay between onset and stim
    #  fixedDelayIdx = int(fixedDelay * fs)
    #  print('Using a fixed delay of {} samples'.format(fixedDelayIdx))
    defaultOptsDict = {
        'detectChannels': [i for i in tdDF.columns if 'ins_td' in i]}

    if stimDetectOptsByChannel is None:
        stimDetectOptsByChannel = {
            grpIdx: {progIdx: defaultOptsDict for progIdx in range(4)}
            for grpIdx in range(4)}
    
    #  allocate units for each group/program pair
    tempSpiketrainStorage = {}
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
            tempSpiketrainStorage.update({thisUnit.name: []})
    spikeTStop = tdDF['t'].iloc[-1]
    spikeTStart = tdDF['t'].iloc[0]
    allDataCol = [i for i in tdDF.columns if 'ins' in i]
    # calculate therapy status starts and ends
    therapyDiff = tdDF['therapyStatus'].diff().fillna(0)
    therapyOnsetIdx = tdDF.index[therapyDiff == 1]
    # if tdDF['therapyStatus'].iloc[0] == 1:
    #     therapyOnsetIdx = pd.Index(
    #         [tdDF.index[0]] + therapyOnsetIdx.to_list())
    therapyOnTimes = pd.DataFrame({
        'nominalOnIdx': therapyOnsetIdx})
    therapyOnTimes.loc[:, 'on'] = np.nan
    therapyOnTimes.loc[:, 'onIdx'] = np.nan
    for idx, row in therapyOnTimes.iterrows():
        print('Calculating therapy on times for segment {}'.format(idx))
        # figure out what the group rate is at therapy on
        winStartT = tdDF.loc[row['nominalOnIdx'], 't'] - gaussWid / 2
        winStopT = tdDF.loc[row['nominalOnIdx'], 't'] + gaussWid / 2
        tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
        groupRate = tdDF.loc[tMask, 'RateInHz'].value_counts().idxmax()
        # expand search window by group rate
        winStopT += 1/groupRate
        tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
        tdSeg = tdDF.loc[tMask, allDataCol + ['t']]
        tdSegDetect = tdSeg.set_index('t')
        try:
            detectSignal, foundTimestamp, _ = extractArtifactTimestampsNew(
                tdSegDetect, fs,
                gaussWid=gaussWid,
                thresh=2,
                offsetFromPeak=offsetFromPeak,
                enhanceEdges=True,
                plotDetection=False
                )
        except Exception:
            foundTimestamp = [None]
        if foundTimestamp[0] is not None:
            therapyOnTimes.loc[idx, 'on'] = foundTimestamp
            localIdx = [tdSegDetect.index.get_loc(i) for i in foundTimestamp]
            therapyOnTimes.loc[idx, 'onIdx'] = tMask.index[tMask][localIdx]
    therapyOnTimes.dropna(inplace=True)
    trueTherapyDiff = pd.Series(0, index=tdDF.index)
    trueTherapyDiff.loc[therapyOnTimes['onIdx']] = 1
    tdDF.loc[:, 'therapyRound'] = trueTherapyDiff.cumsum()
    # find offsets
    therapyOffsetIdx = tdDF.index[therapyDiff == -1]
    # if tdDF['therapyStatus'].iloc[-1] == 1:
    #     therapyOffsetIdx = pd.Index(
    #         therapyOnsetIdx.to_list() + [tdDF.index[-1]])
    # assert therapyOnsetIdx.size == therapyOffsetIdx.size
    therapyOffTimes = pd.DataFrame({
        'nominalOffIdx': therapyOffsetIdx})
    if therapyOffTimes.size > 0:
        therapyOffTimes.loc[:, 'off'] = np.nan
        therapyOffTimes.loc[:, 'offIdx'] = np.nan
        for idx, row in therapyOffTimes.iterrows():
            print('Calculating therapy off times for segment {}'.format(idx))
            # figure out what the group rate is at therapy on
            winStartT = tdDF.loc[row['nominalOffIdx'], 't'] - gaussWid / 2
            winStopT = tdDF.loc[row['nominalOffIdx'], 't'] + gaussWid / 2
            tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
            groupRate = tdDF.loc[tMask, 'RateInHz'].value_counts().idxmax()
            # expand search window by group rate
            winStopT += 1/groupRate
            tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
            tdSeg = tdDF.loc[tMask, allDataCol + ['t']]
            tdSegDetect = tdSeg.set_index('t')
            try:
                detectSignal, foundTimestamp, _ = extractArtifactTimestampsNew(
                    tdSegDetect, fs,
                    gaussWid=gaussWid,
                    thresh=1,
                    offsetFromPeak=offsetFromPeak,
                    enhanceEdges=True,
                    plotDetection=False,
                    threshMethod='peaks',
                    keepWhat='last'
                    )
            except Exception:
                foundTimestamp = [None]
            if foundTimestamp[0] is not None:
                therapyOffTimes.loc[idx, 'off'] = foundTimestamp
                localIdx = [tdSegDetect.index.get_loc(i) for i in foundTimestamp]
                therapyOffTimes.loc[idx, 'offIdx'] = tMask.index[tMask][localIdx]
        therapyOffTimes.dropna(inplace=True)
    lastTherapyRound = 0
    # lastTrialSegment = 0
    # calculate slots
    tdDF.loc[:, 'slot'] = np.nan
    plottingSlots = False
    resolvedSlots = False
    lastRate = np.nan
    for name, group in tdDF.groupby('amplitudeRound'):
        anomalyOccured = False
        # check that this round is worth analyzing
        groupAmpMask = (
            (group['amplitude'] > 0) &
            (group['therapyStatus'] > 0))
        if not groupAmpMask.any():
            print('Amplitude never turned on!')
            continue
        stimRate = (
            group
            .loc[groupAmpMask, 'RateInHz']
            .value_counts()
            .idxmax())
        slotSize = int(fs/stimRate)
        stimPeriod = stimRate ** (-1)
        activeProgram = (
            int(
                group
                .loc[groupAmpMask, 'program']
                .value_counts()
                .idxmax()))
        activeGroup = (
            int(
                group
                .loc[groupAmpMask, 'activeGroup']
                .value_counts()
                .idxmax()))
        thisTrialSegment = (
            int(
                group
                .loc[groupAmpMask, 'trialSegment']
                .value_counts()
                .idxmax()))
        # if thisTrialSegment != lastTrialSegment:
        #     resolvedSlots = False
        #     lastTrialSegment = thisTrialSegment
        thisTherapyRound = (
            int(
                group
                .loc[groupAmpMask, 'therapyRound']
                .value_counts()
                .idxmax()))
        if thisTherapyRound != lastTherapyRound:
            resolvedSlots = False
            lastTherapyRound = thisTherapyRound
        stimPW = (
            group
            .loc[groupAmpMask, 'pulseWidth']
            .value_counts()
            .idxmax())
        ampColName = 'program{}_amplitude'.format(activeProgram)
        thisAmplitude = group[ampColName].max()
        #  pad with paddingDuration msec to ensure robust z-score
        paddingDuration = max(
            np.around(2 * stimPeriod, decimals=4),
            200e-3)
        tStartPadded = max(
            tdDF['t'].iloc[0],
            group.loc[groupAmpMask, 't'].iloc[0] - 2 * paddingDuration)
        tStopPadded = min(
            tdDF['t'].iloc[-1],
            group.loc[groupAmpMask, 't'].iloc[-1] + 0.25 * paddingDuration)
        plotMaskTD = (tdDF['t'] > tStartPadded) & (tdDF['t'] < tStopPadded)
        #  load the appropriate detection options
        theseDetectOpts = stimDetectOptsByChannel[activeGroup][activeProgram]
        #  calculate signal used for stim artifact detection
        tdSeg = (tdDF.loc[
            plotMaskTD,
            theseDetectOpts['detectChannels'] + ['t']
            ])
        # use the HUT derived stim onset to favor detection
        # minROIWid = 150e-3 # StimLog timestamps only reliable within 50 msec
        gaussWidIdx = int(gaussWid * fs)
        nominalStimOnIdx = group.loc[groupAmpMask, 't'].index[0]
        # pdb.set_trace()
        nominalStimOnT = tdSeg.loc[nominalStimOnIdx, 't']
        # 
        lastAmplitude = tdDF.loc[max(nominalStimOnIdx - 1, 0), ampColName]
        if (lastRate != stimRate):
            # recalculate every time we increment amplitude from zero
            # (these should usually be very visible)
            # in order to mitigate uncertainty about when Rate changes
            # are implemented
            # TODO!!! Only do this when rate changed.
            resolvedSlots = False
        lastRate = stimRate
        useThresh = theseDetectOpts['thres']
        if resolvedSlots:
            ROIWid = gaussWid
            tdSegSlots = tdDF.loc[tdSeg.index, 'slot']
            slotDiff = tdSegSlots.diff()
            # ~ 95% conf interval
            earliestStimOnIdx = max(
                nominalStimOnIdx - int(1.5 * gaussWidIdx),
                tdSeg.index[0])
            latestStimOnIdx = min(
                nominalStimOnIdx + int(1.5 * gaussWidIdx),
                tdSeg.index[-1])
            slotStartMask = (
                (slotDiff == -3) &
                (tdSeg.index >= earliestStimOnIdx) &
                (tdSeg.index <= latestStimOnIdx + int(slotSize))
                )
            possibleSlotStartIdx = tdSeg.index[slotStartMask]
            print('possibleSlotStartIdx is {}\n'.format(possibleSlotStartIdx))
            if len(possibleSlotStartIdx) > 1:
                stimOnUncertainty = pd.Series(
                    stats.norm.cdf(tdSeg['t'], nominalStimOnT, (gaussWid / 2)),
                    index=tdSeg.index)
                possibleSlotStartMask = tdSeg.index.isin(possibleSlotStartIdx)
                uncertaintyValsDF = stimOnUncertainty[possibleSlotStartMask].diff()
                uncertaintyValsDF.iloc[0] = stimOnUncertainty[possibleSlotStartMask].iloc[0]
                uncertaintyVals = uncertaintyValsDF.to_numpy()
                keepMask = uncertaintyVals > .1
                keepMask[np.argmax(uncertaintyVals)] = True
                possibleSlotStartIdx = possibleSlotStartIdx[keepMask]
                uncertaintyVals = uncertaintyVals[keepMask]
            else:
                uncertaintyVals = np.array([1])
            possibleOnsetIdx = (
                possibleSlotStartIdx +
                activeProgram * int(slotSize / 4)
                )
            allPossibleTimestamps = tdSeg.loc[possibleOnsetIdx, 't']
            print('allPossibleTimestamps {}\n'.format(allPossibleTimestamps))
            try:
                assert len(allPossibleTimestamps) > 0
            except Exception:
                traceback.print_exc()
            expectedOnsetIdx = possibleOnsetIdx[np.argmax(uncertaintyVals)]
            expectedTimestamp = tdSeg.loc[expectedOnsetIdx, 't']
            ROIBasis = pd.Series(
                0, index=tdSeg['t'])
            basisMask = tdSeg.index.isin(possibleOnsetIdx)
            ROIBasis.loc[basisMask] = uncertaintyVals
            #  where to look for the onset
            tStartOnset = max(
                tdSeg['t'].iloc[0],
                allPossibleTimestamps.iloc[0] - max(3 * stimPeriod / 16, ROIWid / 2))
            tStopOnset = min(
                tdSeg['t'].iloc[-1],
                allPossibleTimestamps.iloc[-1] + max(3 * stimPeriod / 16, ROIWid / 2))
            print("Expected T = {}".format(expectedTimestamp))
            print("ROI = {}".format((tStartOnset, tStopOnset)))
            ROIMaskOnset = (
                (tdSeg['t'] >= tStartOnset) &
                (tdSeg['t'] <= tStopOnset))
            
            slotMatchesMask = tdDF.loc[tdSeg.index, 'slot'].shift(-int(np.round(slotSize/16))) == activeProgram
            ROIMaskOnset = ROIMaskOnset & slotMatchesMask
        else:
            ROIWid = gaussWid + stimPeriod
            #  on average, the StimLog update will land in the middle of the previous group cycle
            #  adjust the ROIWid to account for this extra variability
            expectedOnsetIdx = (
                nominalStimOnIdx +
                int(slotSize / 2) +
                activeProgram * int(slotSize / 4)
                )
            expectedTimestamp = tdSeg.loc[expectedOnsetIdx, 't']
            if overrideStartTimes is not None:
                ovrTimes = pd.Series(overrideStartTimes)
                ovrMask = (ovrTimes > tdSeg['t'].iloc[0]) & (ovrTimes < tdSeg['t'].iloc[-1])
                ovrTimes = ovrTimes[ovrMask]
                if ovrTimes.any():
                    ROIWid = 3 * stimPeriod / 8
                    expectedTimestamp = ovrTimes.iloc[0]
            print('Expected timestamp is {}'.format(expectedTimestamp))
            ROIBasis = pd.Series(0, index=tdSeg['t'])
            basisMask = (
                (ROIBasis.index >= (expectedTimestamp - stimPeriod / 2)) &
                (ROIBasis.index <= (expectedTimestamp + stimPeriod / 2)))
            ROIBasis[basisMask] = 1
            #  where to look for the onset
            tStartOnset = max(
                tdSeg['t'].iloc[0],
                expectedTimestamp - ROIWid / 2)
            tStopOnset = min(
                tdSeg['t'].iloc[-1],
                expectedTimestamp + ROIWid / 2)
            print("Expected T = {}".format(expectedTimestamp))
            print("ROI = {}".format((tStartOnset, tStopOnset)))
            ROIMaskOnset = (
                (tdSeg['t'] >= tStartOnset) &
                (tdSeg['t'] <= tStopOnset))
        ROIMaskOnset.iloc[0] = True
        ROIMaskOnset.iloc[-1] = True
        tdSegDetect = tdSeg.set_index('t')
        plottingEnabled = (name in plotting) # and (not resolvedSlots)
        detectSignal, foundTimestamp, usedExpectedT = extractArtifactTimestampsNew(
            tdSegDetect,
            fs,
            gaussWid=gaussWid,
            thresh=useThresh,
            stimRate=stimRate,
            threshMethod='peaks',
            keepWhat='max',
            enhanceEdges=True,
            offsetFromPeak=offsetFromPeak,
            expectedTimestamp=expectedTimestamp,
            ROIMask=ROIMaskOnset.to_numpy(),
            ROIBasis=ROIBasis,
            plotDetection=plottingEnabled, plotKernel=False
            )
        usedSlotToDetect = resolvedSlots
        if foundTimestamp[0] is None:
            foundTimestamp = np.atleast_1d(expectedTimestamp)
        localIdx = []
        for i in foundTimestamp:
            localIdx.append(tdSegDetect.index.get_loc(i))
        foundIdx = tdSeg.index[localIdx]
        
        if (not resolvedSlots) and predictSlots and (lastAmplitude == 0):
            # have not resolved phase between slots and recording for this segment
            therSegDF = tdDF.loc[tdDF['therapyRound'] == thisTherapyRound, :]
            # therSegDF = tdDF.loc[tdDF['trialSegment'] == thisTrialSegment, :]
            rateDiff = therSegDF['RateInHz'].diff().fillna(method='bfill')
            rateChangeTimes = therSegDF.loc[rateDiff != 0, 't']
            print('Calculating slots for segment {}'.format(thisTherapyRound))
            
            groupRate = therSegDF.loc[foundIdx, 'RateInHz'].iloc[0]
            groupPeriod = np.float64(groupRate ** (-1))
            rateChangePeriods = (
                therSegDF
                .loc[rateDiff != 0, 'RateInHz']) ** (-1)
            while True:
                rateChangeTimesDiff = rateChangeTimes.diff()
                rateChangeTooFast = rateChangeTimesDiff < rateChangePeriods
                if rateChangeTooFast.any():
                    rateChangePeriods.drop(index=rateChangeTooFast.index[rateChangeTooFast][0], inplace=True)
                    rateChangeTimes.drop(index=rateChangeTooFast.index[rateChangeTooFast][0], inplace=True)
                else:
                    break
            # nominalOffset = groupPeriod * (activeProgram * 1/4 + 1/8)
            nominalOffset = np.ceil((groupPeriod * (activeProgram * 1/4)) * fs.magnitude) / fs.magnitude
            startTime = (foundTimestamp[0] - nominalOffset)
            exitLoop = False
            
            while (not exitLoop):
                startIdx = therSegDF.index[therSegDF['t'] >= startTime][0]
                nextRateChange = rateChangeTimes - startTime
                if (nextRateChange > 0).any():
                    nPeriods = int(np.ceil(nextRateChange[nextRateChange > 0].iloc[0] * groupRate))
                else:
                    nPeriods = int((therSegDF['t'].iloc[-1] - startTime) * groupRate)
                    exitLoop = True
                thisSlotSize = (fs.magnitude/groupRate) # samples
                endIdx = startIdx + int(nPeriods * thisSlotSize) - 1
                timeBase = therSegDF.loc[startIdx:endIdx, 't'].to_numpy()
                timeBase = timeBase - timeBase[0]
                calculatedSlots = (timeBase % groupPeriod) // (groupPeriod / 4)
                tdDF.loc[startIdx:endIdx, 'slot'] = calculatedSlots.astype(np.int)
                timeSeekIdx = min(
                    endIdx + 1,
                    therSegDF.index[-1])
                groupRate = therSegDF.loc[timeSeekIdx, 'RateInHz']
                # oldGroupPeriod = groupPeriod
                groupPeriod = np.float64(groupRate ** (-1))
                startTime = therSegDF.loc[timeSeekIdx, 't']
                # startTime = (
                #     therSegDF.loc[timeSeekIdx, 't'] +
                #     (oldGroupPeriod - groupPeriod) / 8)
            group.loc[:, 'slot'] = tdDF.loc[group.index, 'slot']
            resolvedSlots = True
            if plottingEnabled and plottingSlots:
                # plot all the slots for this therapy round
                sns.set()
                sns.set_style("whitegrid")
                sns.set_color_codes("dark")
                cPal = sns.color_palette('pastel', n_colors=len(allDataCol))
                cLookup = {n: cPal[i] for i, n in enumerate(allDataCol)}
                fig, slotAx = plt.subplots()
                for colName in allDataCol:
                    tdCol = therSegDF[colName]
                    slotAx.plot(
                        therSegDF['t'],
                        tdCol.values,
                        '-', c=cLookup[colName],
                        label='original signal {}'.format(colName))
                slotAx.set_xlabel('Time (sec)')
                slotAx.set_title('Slots for therapy round {}'.format(thisTherapyRound))
                therSegSlotDiff = therSegDF['slot'].diff()
                slotEdges = (
                    therSegDF
                    .loc[therSegSlotDiff.fillna(1) != 0, 't']
                    .reset_index(drop=True))
                theseSlots = (
                    therSegDF
                    .loc[therSegSlotDiff.fillna(1) != 0, 'slot']
                    .reset_index(drop=True))
                for idx, slEdge in slotEdges.iloc[1:].iteritems():
                    try:
                        slotAx.axvspan(
                            slotEdges[idx-1], slEdge,
                            alpha=0.4, facecolor=cPal[int(theseSlots[idx-1])])
                    except Exception:
                        continue
                for t in rateChangeTimes:
                    slotAx.axvline(t)
                slotAx.legend()
        # done resolving slots
        
        if plottingEnabled:
            tdSegSlotDiff = tdDF.loc[plotMaskTD, 'slot'].diff()
            slotEdges = (
                tdSeg
                .loc[tdSegSlotDiff.fillna(1) != 0, 't']
                .reset_index(drop=True))
            theseSlots = (
                tdDF.loc[plotMaskTD, :]
                .loc[tdSegSlotDiff.fillna(1) != 0, 'slot']
                .reset_index(drop=True))
            ax = plt.gca()
            ax.axvline(expectedTimestamp, color='b', linestyle='--', label='expected time')
            ax.axvline(group['t'].iloc[0], color='r', label='stimLog time')
            ax.set_title(
                'Grp {} Prog {} slots: {}'.format(activeGroup, activeProgram, usedSlotToDetect))
            cPal = sns.color_palette(n_colors=4)
            for idx, slEdge in slotEdges.iloc[1:].iteritems():
                try:
                    ax.axvspan(
                        slotEdges[idx-1], slEdge,
                        alpha=0.5, facecolor=cPal[int(theseSlots[idx-1])])
                except Exception:
                    continue
            ax.legend()
            plt.show()
        if snapToGrid and resolvedSlots:
            slotMatchesMask = tdDF.loc[tdSeg.index, 'slot'] == activeProgram # .shift(int(-slotSize/8))
            slotMatchesTime = tdSeg.loc[slotMatchesMask, 't']
            try:
                timeMatchesMask = (slotMatchesTime - foundTimestamp[0]).abs() < stimPeriod / 8
                theseOnsetTimestamps = np.atleast_1d(slotMatchesTime.loc[timeMatchesMask].iloc[0]) * pq.s
            except Exception:
                theseOnsetTimestamps = np.atleast_1d(foundTimestamp[0]) * pq.s
            
        else:
            theseOnsetTimestamps = np.atleast_1d(foundTimestamp[0]) * pq.s
        onsetDifferenceFromExpected = (
            np.atleast_1d(tdSeg.loc[expectedOnsetIdx, 't']) -
            np.atleast_1d(foundTimestamp)) * pq.s
        onsetDifferenceFromLogged = (
            np.atleast_1d(group['t'].iloc[0]) -
            np.atleast_1d(foundTimestamp)) * pq.s
        #
        stimOffIdx = min(
            group.index[-1],
            (
                group.index[groupAmpMask][-1] +
                slotSize +
                activeProgram * int(slotSize/4)
            )
        )
        theseOffsetTimestamps = np.atleast_1d(
            group
            .loc[stimOffIdx, 't']
            ) * pq.s
        #
        electrodeCombo = 'g{:d}p{:d}'.format(activeGroup, activeProgram)
        if len(theseOnsetTimestamps):
            thisUnit = block.filter(
                objects=Unit,
                name=electrodeCombo
                )[0]
            if treatAsSinglePulses:
                tempOnTimes = []
                tempOffTimes = []
                tempOnDiffsE = []
                tempOnDiffsL = []
                for idx, onTime in enumerate(theseOnsetTimestamps):
                    offTime = theseOffsetTimestamps[idx]
                    interPulseInterval = 1 / (stimRate * pq.Hz)
                    pulseOnTimes = np.arange(
                        onTime, offTime,
                        interPulseInterval) * onTime.units
                    pulseOffTimes = pulseOnTimes + 100 * stimPW * pq.us
                    tempOnTimes.append(pulseOnTimes)
                    tempOffTimes.append(pulseOffTimes)
                    onDiffE = onsetDifferenceFromExpected[idx]
                    tempOnDiffsE.append(pulseOnTimes ** 0 * onDiffE)
                    onDiffL = onsetDifferenceFromLogged[idx]
                    tempOnDiffsL.append(pulseOnTimes ** 0 * onDiffL)
                theseOnsetTimestamps = np.concatenate(tempOnTimes)
                theseOffsetTimestamps = np.concatenate(tempOffTimes)
                onsetDifferenceFromExpected = np.concatenate(tempOnDiffsE)
                onsetDifferenceFromLogged = np.concatenate(tempOnDiffsL)
            ampList = theseOnsetTimestamps ** 0 * 100 * thisAmplitude * pq.uA
            rateList = theseOnsetTimestamps ** 0 * stimRate * pq.Hz
            pwList = theseOnsetTimestamps ** 0 * 10 * stimPW * pq.us
            programList = theseOnsetTimestamps ** 0 * activeProgram * pq.dimensionless
            groupList = theseOnsetTimestamps ** 0 * activeGroup * pq.dimensionless
            tSegList = theseOnsetTimestamps ** 0 * thisTrialSegment
            usedExpTList = theseOnsetTimestamps ** 0 * usedExpectedT
            usedSlotList = theseOnsetTimestamps ** 0 * usedSlotToDetect
            #
            arrayAnn = {
                'amplitude': ampList, 'RateInHz': rateList,
                'pulseWidth': pwList,
                'trialSegment': tSegList,
                'endTime': theseOffsetTimestamps,
                'program': programList,
                'group': groupList,
                'offsetFromExpected': onsetDifferenceFromExpected,
                'offsetFromLogged': onsetDifferenceFromLogged,
                'usedExpectedT': usedExpTList,
                'usedSlotToDetect': usedSlotList}
            st = SpikeTrain(
                times=theseOnsetTimestamps, t_stop=spikeTStop,
                t_start=spikeTStart,
                name=thisUnit.name,
                array_annotations=arrayAnn,
                **arrayAnn)
            #  st.annotate(amplitude=thisAmplitude * 100 * pq.uA)
            #  st.annotate(rate=stimRate * pq.Hz)
            #  thisUnit.spiketrains.append(st)
            tempSpiketrainStorage[thisUnit.name].append(st)
    createRelationship = False
    for thisUnit in block.filter(objects=Unit):
        # print('getINSStimOnset packaging unit {}'.format(thisUnit.name))
        if len(tempSpiketrainStorage[thisUnit.name]) == 0:
            st = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=[], units='sec', t_stop=spikeTStop,
                t_start=spikeTStart)
            thisUnit.spiketrains.append(st)
            seg.spiketrains.append(st)
            st.unit = thisUnit
            st.segment = seg
        else:
            #  consolidate spiketrains
            consolidatedTimes = np.array([])
            consolidatedAnn = {
                'amplitude': np.array([]),
                'RateInHz': np.array([]),
                'program': np.array([]),
                'group': np.array([]),
                'pulseWidth': np.array([]),
                'endTime': np.array([]),
                'trialSegment': np.array([]),
                'offsetFromExpected': np.array([]),
                'offsetFromLogged': np.array([]),
                'usedExpectedT': np.array([]),
                'usedSlotToDetect': np.array([])
                }
            arrayAnnNames = {'arrayAnnNames': list(consolidatedAnn.keys())}
            for idx, st in enumerate(tempSpiketrainStorage[thisUnit.name]):
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
            thisUnit.spiketrains.append(newSt)
            print('thisUnit.st len = {}'.format(len(thisUnit.spiketrains)))
            newSt.unit = thisUnit
            if createRelationship:
                thisUnit.create_relationship()
            seg.spiketrains.append(newSt)
            newSt.segment = seg
    if createRelationship:
        for chanIdx in block.channel_indexes:
            chanIdx.create_relationship()
        seg.create_relationship()
        block.create_relationship()
    return block


def extractArtifactTimestampsNew(
        tdSeg,
        fs,
        gaussWid=200e-3,
        thresh=2,
        offsetFromPeak=0,
        stimRate=100,
        keepWhat='first',
        threshMethod='cross',
        enhanceEdges=True,
        expectedTimestamp=None,
        ROIBasis=None,
        ROIMask=None, keepSlice=0,
        name=None, plotting=None,
        plotAnomalies=None, anomalyOccured=None,
        plotDetection=False, plotKernel=False
        ):
    if plotDetection:
        sns.set()
        sns.set_style("whitegrid")
        sns.set_color_codes("dark")
        cPal = sns.color_palette(n_colors=tdSeg.columns.size)
        cLookup = {n: cPal[i] for i, n in enumerate(tdSeg.columns)}
        fig, ax = plt.subplots()
        for colName, tdCol in tdSeg.iteritems():
            ax.plot(
                tdSeg.index,
                tdCol.values,
                '-', c=cLookup[colName],
                label='original signal {}'.format(colName))
        ax.set_xlabel('Time (sec)')
    # convolve with a future facing kernel
    if enhanceEdges:
        edgeEnhancer = pd.Series(0, index=tdSeg.index)
        for colName, tdCol in tdSeg.iteritems():
            thisEdgeEnhancer = (
                hf.noisyTriggerCorrection(
                    tdCol, fs, gaussWid, order=2,
                    applyZScore=True, applyAbsVal=True,
                    applyScaler=None,
                    plotKernel=plotKernel))
            edgeEnhancer += thisEdgeEnhancer
            if plotDetection and False:
                ax.plot(
                    tdSeg.index,
                    thisEdgeEnhancer,
                    '--', c=cLookup[colName],
                    label='edge enhancer {}'.format(colName))
        edgeEnhancer.loc[:] = (
            MinMaxScaler(feature_range=(1e-2, 1))
            .fit_transform(edgeEnhancer.to_numpy().reshape(-1, 1))
            .squeeze()
            )
        if plotDetection:
            ax.plot(
                tdSeg.index,
                edgeEnhancer,
                'k--', label='final edge enhancer')
    else:
        edgeEnhancer = pd.Series(
            1, index=tdSeg.index)
    if ROIBasis is not None:
        expectedEnhancer = hf.gaussianSupport(
            tdSeg=tdSeg, gaussWid=6 * gaussWid, fs=fs, support=ROIBasis)
        if plotDetection:
            ax.plot(
                tdSeg.index,
                expectedEnhancer.values, 'c--', label='expected enhancer')
            #ax.axvline(expectedTimestamp, label='expected timestamp')
        correctionFactor = edgeEnhancer * expectedEnhancer
        correctionFactor = pd.Series(
            MinMaxScaler(feature_range=(1e-2, 1))
            .fit_transform(correctionFactor.to_numpy().reshape(-1, 1))
            .squeeze(),
            index=tdSeg.index)
    else:
        correctionFactor = edgeEnhancer
    #
    detectSignal = pd.Series(0, index=tdSeg.index)
    for colName, tdCol in tdSeg.iteritems():
        thisDetectSignal = hf.enhanceNoisyTriggers(
            tdCol, correctionFactor=correctionFactor,
            applyZScore=True, applyAbsVal=True,
            applyScaler=None)
        if plotDetection and False:
            ax.plot(
                tdSeg.index,
                thisDetectSignal,
                '-.', c=cLookup[colName],
                label='detect signal {}'.format(colName))
        detectSignal += thisDetectSignal
    detectSignal = detectSignal / tdSeg.shape[1]
    if ROIMask is not None:
        detectSignal.loc[~ROIMask] = np.nan
        detectSignal.interpolate(inplace=True)
        if plotDetection:
            ax.axvspan(
                detectSignal.index[ROIMask][1],
                detectSignal.index[ROIMask][-2],
                facecolor='g', alpha=0.5, zorder=-100)
    if plotDetection:
        ax.plot(
            detectSignal.index,
            detectSignal.values, 'y-', label='detect signal')
        ax.axhline(thresh, color='r')
    if threshMethod == 'peaks':
        idxLocal = peakutils.indexes(
            detectSignal.to_numpy(),
            thres=thresh,
            min_dist=int(1.05 * fs / stimRate), thres_abs=True,
            keep_what=keepWhat)
        idxLocal = np.atleast_1d(idxLocal)
    elif threshMethod == 'cross':
        keepMax = (keepWhat == 'max')
        crossIdx, crossMask = hf.getThresholdCrossings(
            detectSignal, thresh=thresh,
            iti=1.05 / stimRate, fs=fs,
            absVal=False, keep_max=keepMax)
        if crossMask.any():
            idxLocal = np.atleast_1d(np.flatnonzero(crossMask))
        else:
            idxLocal = []
    if len(idxLocal):
        idxLocal = idxLocal - int(fs * offsetFromPeak)
        idxLocal[idxLocal < 0] = 0
        usedExpectedTimestamp = False
        foundTimestamp = tdSeg.index[idxLocal]
        if plotDetection:
            ax.plot(
                foundTimestamp, foundTimestamp ** 0 - 1,
                'r*', markersize=10)
    else:
        print(
            'After peakutils.indexes, no peaks found! ' +
            'Using JSON times...')
        usedExpectedTimestamp = True
        foundTimestamp = np.atleast_1d(expectedTimestamp)
    if plotDetection:
        ax.legend()
        try:
            ax.set_xlim([
                max(detectSignal.index[0], foundTimestamp[0] - 3 * stimRate ** (-1)),
                min(detectSignal.index[-1], foundTimestamp[0] + 7 * stimRate ** (-1))
            ])
        except:
            pass
    return detectSignal, np.atleast_1d(foundTimestamp[keepSlice]), usedExpectedTimestamp

'''
def extractArtifactTimestamps(
        tdSeg, tdDF, ROIMask,
        fs, gaussWid,
        stimRate=100,
        closeThres=200e-3,
        enhanceEdges=True,
        expectedIdx=None,
        enhanceExpected=True,
        expandExpectedByStimRate=False,
        correctThresholdWithAmplitude=True,
        thisAmplitude=None,
        name=None, plotting=None,
        plotAnomalies=None, anomalyOccured=None,
        minDist=None, theseDetectOpts=None,
        maxSpikesPerGroup=None,
        fixedDelayIdx=None, delayByFreqIdx=None,
        keep_what=None, plotDetection=False, plotKernel=False
        ):
    if plotDetection:
        fig, ax = plt.subplots()
        for colName, tdCol in tdSeg.iteritems():
            ax.plot(
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
                ax.plot(
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
        ax.plot(
            edgeEnhancer.values,
            'k-', label='final edge enhancer')
    if enhanceExpected:
        assert expectedIdx is not None
        if expandExpectedByStimRate:
            print(' ')
            
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
            ax.plot(expectedEnhancer.values, 'k--', label='expected enhancer')
    #
    tdPow = tdSeg.abs().sum(axis=1)
    detectSignalFull = hf.enhanceNoisyTriggers(
        tdPow, correctionFactor)
    if plotDetection:
        ax.plot(detectSignalFull.values, 'g-', label='detect signal')
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
        ax.axhline(currentThresh, color='r')
        ax.legend()
        plt.show()
    
    detectSignal = detectSignalFull.loc[ROIMask]
    idxLocal = peakutils.indexes(
        detectSignal.values,
        thres=currentThresh,
        min_dist=int(minDist * fs), thres_abs=True,
        keep_what=keep_what)

    if not len(idxLocal):
        if plotAnomalies:
            anomalyOccured = True
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
'''

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
    if 'upsampleRate' in trialFilesStim:
        sampleRate *= trialFilesStim['upsampleRate']
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

    blockName = trialFilesStim['experimentName'] + '_ins'
    block = Block(name=blockName)
    block.annotate(jsonSessionNames=trialFilesStim['jsonSessionNames'])
    seg = Segment(name='seg0_' + blockName)
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


def unpackINSBlock(block, unpackAccel=True, convertStimToSinglePulses=False):
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
    allEvents = block.filter(
        objects=Event
        )
    events = [ev for ev in allEvents if 'ins' in ev.name]
    for idx, ev in enumerate(events):
        events[idx].name = ns5.childBaseName(ev.name, 'seg')
    stimStSer = ns5.eventsToDataFrame(
        events, idxT='t'
        )
    #  serialize stimStatus
    expandCols = [
        'RateInHz', 'therapyStatus', 'pulseWidth',
        'activeGroup', 'program', 'trialSegment']
    deriveCols = ['amplitudeRound', 'amplitude']
    progAmpNames = rcsa_helpers.progAmpNames
    stimStatus = stimStatusSerialtoLong(
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
