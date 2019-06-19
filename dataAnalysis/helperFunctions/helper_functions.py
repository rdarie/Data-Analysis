'''
try:
    import plotly
    import plotly.plotly as py
    import plotly.tools as tls
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
except:
    pass
'''
import psutil
import pdb, sys, itertools, os, pickle, gc, random, string,\
subprocess, collections, traceback, peakutils, math, argparse
from collections.abc import Iterable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.backends.backend_pdf
import random 
import numpy as np
import pandas as pd
import math as m
import tables as pt
import seaborn as sns
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
from neo import (
    Unit, AnalogSignal, Event, Epoch,
    Block, Segment, ChannelIndex, SpikeTrain)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, make_scorer, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import sklearn.pipeline
from sklearn.feature_selection import RFE, RFECV

import datetime
from datetime import datetime as dt
import json
import dataAnalysis.preproc.mdt_constants as mdt_constants
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
from scipy import stats, signal, ndimage
from copy import copy
from fractions import gcd
import h5py
LABELFONTSIZE = 5
from collections import OrderedDict


def animateDFSubset3D(
        unpackedFeatures, dataQuery, winWidth, nFrames, fps=100,
        xyzList=['PC1', 'PC2', 'PC3'], showNow=False, ax=None,
        colorCol='tdAmplitude', saveToFile='', pltKws=None, extraAni=False):

    colorOpts = sns.cubehelix_palette(128)

    def update_lines(idx, data, colorData, lines):
        #  print(idx)
        if idx >= winWidth:
            for ptIdx in range(winWidth):
                lines[ptIdx].set_data(
                    data[0:2, idx - ptIdx - 1:idx - ptIdx + 1])
                lines[ptIdx].set_3d_properties(
                    data[2, idx - ptIdx - 1:idx - ptIdx + 1])
                
                colorIdx = colorData.iloc[idx - ptIdx]
                rgbaColor = np.zeros((4))
                rgbaColor[:3] = colorOpts[colorIdx]
                rgbaColor[3] = (winWidth - ptIdx) / winWidth
                lines[ptIdx].set_color(rgbaColor)
        return lines
    
    featSubset = unpackedFeatures.query(dataQuery)
    data = featSubset.loc[:, xyzList].transpose().values

    _, colorBins = pd.cut(
        unpackedFeatures[colorCol], 128, labels=False, retbins=True)
    colorData = pd.cut(
        featSubset[colorCol], colorBins, labels=False)
    # Attaching 3D axis to the figure
    if ax is None:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
    else:
        fig = ax.figure
    # initialize the line
    lines = [None for i in range(winWidth)]
    for idx in range(winWidth):
        #print(idx)
        #pdb.set_trace()
        lines[idx] = ax.plot(
            data[0, idx: idx + 2],
            data[1, idx: idx + 2],
            data[2, idx: idx + 2],
            **pltKws)[0]
        lines[idx].set_color([0, 0, 0, 0])

    #  Setting the axes properties
    #  pdb.set_trace()
    
    nuMax = featSubset[xyzList[0]].quantile(0.99)
    nuMin = featSubset[xyzList[0]].quantile(0.01)
    sigSpread = nuMax - nuMin
    nuMax += sigSpread * 1e-2
    nuMin -= sigSpread * 1e-2
    ax.set_xlim3d([nuMin, nuMax])
    ax.set_xticklabels([])
    ax.set_xlabel(xyzList[0])

    nuMax = featSubset[xyzList[1]].quantile(0.99)
    nuMin = featSubset[xyzList[1]].quantile(0.01)
    sigSpread = nuMax - nuMin
    nuMax += sigSpread * 1e-2
    nuMin -= sigSpread * 1e-2
    ax.set_ylim3d([nuMin, nuMax])
    ax.set_yticklabels([])
    ax.set_ylabel(xyzList[1])

    nuMax = featSubset[xyzList[2]].quantile(0.99)
    nuMin = featSubset[xyzList[2]].quantile(0.01)
    sigSpread = nuMax - nuMin
    nuMax += sigSpread * 1e-2
    nuMin -= sigSpread * 1e-2
    ax.set_zlim3d([nuMin, nuMax])
    ax.set_zticklabels([])
    ax.set_zlabel(xyzList[2])

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, frames=nFrames,
        fargs=(data, colorData, lines),
        interval=int(1e3/fps), blit=False)
    
    if saveToFile:
        writer = FFMpegWriter(fps=int(fps), metadata=dict(artist='Me'), bitrate=3600)
        ani.save(saveToFile, writer=writer, extra_anim=extraAni)
    elif showNow:
        plt.show()
    return ani


def animateAngle3D(
        unpackedFeatures, dataQuery, winWidth, nFrames, showNow=False,
        saveToFile='', pltKws={}, extraAni=False, ax=None, fps=100):
    
    featSubset = unpackedFeatures.query(dataQuery)
    angleXYZ = pd.DataFrame(
        0, index=featSubset['position'].index,
        columns=['x', 'y', 'z'])
    #pdb.set_trace()
    angleXYZ['y'] = featSubset['position'].apply(np.deg2rad).apply(np.sin)
    angleXYZ['z'] = featSubset['position'].apply(np.deg2rad).apply(np.cos)
    data = np.zeros((3, 2))
    
    def update_lines(idx, angleXYZ, lines):
        #  print(idx)
        data = np.zeros((3, 2))
        data[:, 1] = angleXYZ.transpose().values[:, idx]
        #rint(data)
        lines[0].set_data(data[0:2, :])
        lines[0].set_3d_properties(data[2, :])
        return lines
    
    data[:, 1] = angleXYZ.transpose().values[:, 0]

    # Attaching 3D axis to the figure
    if ax is None:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
    else:
        fig = ax.figure
    # initialize the line
    lines = [None]
    lines[0] = ax.plot(
        data[0, 0:1], data[1, 0:1], data[2, 0:1], 'o-',
        **pltKws)[0]
    #  Setting the axes properties
    #  pdb.set_trace()
    
    ax.set_xlim3d([-.5, .5])
    ax.set_xticks([])
    
    ax.set_ylim3d([-1.1, 1.1])
    ax.set_yticks([])
    ax.set_ylabel('Pedal')

    ax.set_zlim3d([-1.1, 1.1])
    ax.set_zticks([])
    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, frames=nFrames,
        fargs=(angleXYZ, lines),
        interval=int(1e3/fps), blit=False)
    
    if saveToFile:
        writer = FFMpegWriter(fps=int(fps), metadata=dict(artist='Me'), bitrate=3600)
        ani.save(saveToFile, writer=writer, extra_anim=extraAni)
    elif showNow:
        plt.show()
    return ani


def animateDFSubset2D(
        unpackedFeatures, dataQuery, winWidth, nFrames,
        yCol={'feature': ['PC1']}, plotBlocking=True,
        xCol={'feature': None}, axList=None, showNow=False, fps=100, 
        colorCol='tdAmplitude', saveToFile='', pltKws={}, extraAni=None):
    assert len(xCol.keys()) == len(yCol.keys())

    featSubset = unpackedFeatures.query(dataQuery)
    yCol = OrderedDict(yCol)
    xCol = OrderedDict(xCol)
    #  pdb.set_trace()
    if axList is None:
        if len(yCol.keys()) == 1:
            fig, ax = plt.subplots()
            axList = [ax]
        else:
            fig, axList = plt.subplots(len(yCol.keys()), 1)
    else:
        fig = axList[0].figure

    nLines = np.sum([len(v) for k, v in yCol.items()])
    lineList = [None for i in range(nLines)]
    lineIdx = 0
    for axIdx, (key, value) in enumerate(yCol.items()):
        thisAx = axList[axIdx]
        for colName in value:
            if xCol[key] is None:
                lineList[lineIdx] = thisAx.plot(
                    featSubset[colName].iloc[:winWidth],
                    **pltKws)[0]
            else:
                xColName = xCol[key]
                lineList[lineIdx] = thisAx.plot(
                    featSubset[xColName].iloc[:winWidth],
                    featSubset[colName].iloc[:winWidth],
                    **pltKws)[0]
            lineIdx += 1
        thisAx.set_ylabel(key)
        # pdb.set_trace()
        nuMax = featSubset.loc[:, value].quantile(0.99).max().max()
        nuMin = featSubset.loc[:, value].quantile(0.01).min().min()
        sigSpread = nuMax - nuMin
        nuMax += sigSpread * 1e-2
        nuMin -= sigSpread * 1e-2
        thisAx.set_ylim(
            bottom=nuMin,
            top=nuMax)
        if xCol[key] is None:
            thisAx.set_xticklabels([])
        else:
            xColName = xCol[key]
            nuMax = featSubset[xColName].quantile(0.99).max()
            nuMin = featSubset[xColName].quantile(0.01).min()
            sigSpread = nuMax - nuMin
            nuMax += sigSpread * 1e-2
            nuMin -= sigSpread * 1e-2
            thisAx.set_xlim(
                left=nuMin,
                right=nuMax)
            thisAx.set_xlabel(xColName)
        #thisAx.legend()
    
    def init():  # only required for blitting to give a clean slate.
        lineIdx = 0
        for key, value in yCol.items():
            for colName in value:
                lineList[lineIdx].set_ydata([np.nan] * winWidth)
                if xCol[key] is not None:
                    lineList[lineIdx].set_xdata([np.nan] * winWidth)
                lineIdx += 1
        return lineList

    def animate(idx):
        if idx >= winWidth:
            lineIdx = 0
            for key, value in yCol.items():
                for colName in value:
                    lineData = (
                        featSubset[colName].iloc[idx-winWidth:idx].values)
                    lineList[lineIdx].set_ydata(lineData)
                    if xCol[key] is not None:
                        xColName = xCol[key]
                        lineXData = (
                            featSubset[xColName].iloc[idx-winWidth:idx].values)
                        lineList[lineIdx].set_xdata(lineXData)
                    lineIdx += 1
                    # update the data.
        return lineList
    
    ani = animation.FuncAnimation(
        fig, animate, frames=nFrames,
        init_func=init, interval=int(1e3/fps), blit=True)
    if saveToFile:
        writer = FFMpegWriter(fps=int(fps), metadata=dict(artist='Me'), bitrate=3600)
        ani.save(saveToFile, writer=writer, extra_anim=extraAni)
    elif showNow:
        plt.show(block=plotBlocking)
    return ani


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


def filterDF(
        df, fs,
        lowPass=None, lowOrder=2,
        highPass=None, highOrder=2,
        notch=False, filtFun='butter',
        columns=None):

    passedSeries = False
    if isinstance(df, pd.Series):
        passedSeries = True
        df = pd.DataFrame(df.values, index=df.index, columns=['temp'])
        columns = ['temp']
    elif columns is None:
            columns = df.columns

    if (lowPass is not None) and (highPass is not None):
        if notch:
            btype = 'bandstop'
        else:
            btype = 'bandpass'
        Wn = 2 * (highPass, lowPass) / fs
        filtOrder = max(lowOrder, highOrder)
    elif lowPass is not None:
        btype = 'low'
        Wn = 2 * lowPass / fs
        filtOrder = lowOrder
    elif highPass is not None:
        btype = 'high'
        Wn = 2 * highPass / fs
        filtOrder = lowOrder

    if filtFun == 'butter':
        b, a = signal.butter(
            filtOrder, Wn=Wn, btype=btype, analog=False)
    elif filtFun == 'bessel':
        b, a = signal.bessel(
            filtOrder, Wn=Wn, btype=btype, analog=False)
    elif filtFun == 'ellip':
        b, a = signal.ellip(
            filtOrder, rp=5, rs=10, Wn=Wn, btype=btype, analog=False)

    filteredDF = pd.DataFrame(df[columns])
    
    for column in filteredDF.columns:
        filteredDF.loc[:, column] = (
            signal.filtfilt(b, a, filteredDF[column]))
    if passedSeries:
        filteredDF = filteredDF['temp']
    return filteredDF


def closestSeries(takeFrom, compareTo, strictly='neither'):
    closest = pd.Series(
        np.nan, index=takeFrom.index)
    closestIdx = pd.Series(
        np.nan, index=takeFrom.index)
    for idx, value in enumerate(takeFrom.values):
        if strictly == 'greater':
            lookIn = compareTo[compareTo > value]
        if strictly == 'less':
            lookIn = compareTo[compareTo < value]
        else:
            lookIn = compareTo
        idxMin = np.abs(compareTo.values - value).argmin()
        closeValue = (
            lookIn
            .values
            .flat[idxMin])
        closest.iloc[idx] = closeValue
        closestIdx.iloc[idx] = lookIn.index[idxMin]
    return closest, closestIdx


def interpolateDF(
        df, newX, kind='linear', fill_value='extrapolate',
        x=None, columns=None):
    
    # print('interpolatingDF')
    if x is None:
        oldX = np.array(df.index)
        if columns is None:
            columns = df.columns
        outputDF = pd.DataFrame(index=newX, columns=columns)
    else:
        oldX = np.array(df[x])
        if columns is None:
            columns = list(
                df.columns[~df.columns.isin([x])])

        outputDF = pd.DataFrame(columns=columns+[x])
        outputDF[x] = newX
    
    for columnName in columns:
        if ('time' in columnName) or ('microseconds' in columnName):
            #  fix for issue where you can't interpolate time with zeros meaningfully.
            #  Probably better, in the future, to allow for fill value by column
            useFill = 'extrapolate'
        else:
            useFill = fill_value

        interpFun = interpolate.interp1d(
            oldX, df[columnName], kind=kind,
            fill_value=useFill, bounds_error=False)
        outputDF[columnName] = interpFun(newX)
    return outputDF


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
        forceRecalc=False, getInterpolated=True
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

            intersampleTickCount = (1/fs) / (100e-6)

            timeDomainMeta = rcsa_helpers.extract_td_meta_data(timeDomainJson)
            #  pdb.set_trace()
            # assume n channels is constant
            nChan = int(timeDomainMeta[0, 8])
                    
            timeDomainMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
                timeDomainMeta)

            timeDomainMeta, packetsNeededFixing =\
                rcsa_helpers.correct_meta_matrix_time_displacement(
                    timeDomainMeta, intersampleTickCount)
                    
            timeDomainMeta = rcsa_helpers.code_micro_and_macro_packet_loss(
                timeDomainMeta)
            #  pdb.set_trace()
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
                    
            #pdb.set_trace()
            tdData = tdData.drop_duplicates(
                ['t']
                ).sort_values('t').reset_index(drop=True)

            if getInterpolated:
                uniformT = np.arange(
                    tdData['t'].iloc[0],
                    tdData['t'].iloc[-1] + 1/fs,
                    1/fs)
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

                tdData = interpolateDF(
                    tdData, uniformT, x='t',
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
    #  pdb.set_trace()
    return td


def getINSTimeSyncFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        forceRecalc=False
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
                    #  pdb.set_trace()
            
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
        forceRecalc=False, getInterpolated=True
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


            intersampleTickCount = (1/fs) / (100e-6)
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
            #pdb.set_trace()
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
                accelData = interpolateDF(
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
    #  pdb.set_trace()
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
        #  pdb.set_trace()
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
        
        tdAx.legend()
        accAx.legend()

        tNSPMask = (channelData['t'] > tStartNSP) & (channelData['t'] < tStopNSP)
        triggerTrace = channelData['data'].loc[tNSPMask, 'ainp7']
        #  pdb.set_trace()
        ax[2].plot(
            channelData['t'].loc[tNSPMask],
            stats.zscore(triggerTrace),
            'c', label='Analog Sync', picker=line_picker)

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
            #  pdb.set_trace()
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
        
        accelPeakIdx = getTriggers(
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

        tdPeakIdx = getTriggers(
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

'''
def eventsToDataFrame(
        events, idxT='t'
        ):
    eventDict = {}
    
    for event in events:
        print(event.name)
        values = event.array_annotations['labels']
        if isinstance(values[0], bytes):
            #  event came from hdf, need to recover dtype
            originalDType = eval('np.' + event.annotations['originalDType'])
            values = np.array(values, dtype=originalDType)
        #  print(values.dtype)
        eventDict.update({
            event.name: pd.Series(values)})
    eventDict.update({idxT: pd.Series(event.times.magnitude)})
    return pd.concat(eventDict, axis=1)
'''

def synchronizeINStoNSP(
        tapTimestampsNSP, tapTimestampsINS,
        NSPTimeRanges=(None, None),
        td=None, accel=None, insBlock=None, trialSegment=None, degree=1
        ):
    # sanity check that the intervals match
    insDiff = tapTimestampsINS.diff().dropna().values
    nspDiff = tapTimestampsNSP.diff().dropna().values
    print('Trial Segment {}'.format(trialSegment))
    print('On the INS, the diff() between taps was\n{}'.format(insDiff))
    print('On the NSP, the diff() between taps was\n{}'.format(nspDiff))
    print('This amounts to a msec difference of\n{}'.format(
        (insDiff - nspDiff) * 1e3))
    if (insDiff - nspDiff > 20e-3).any():
        raise(Exception('Tap trains too different!'))
    #  pdb.set_trace()
    if degree > 0:
        synchPolyCoeffsINStoNSP = np.polyfit(
            x=tapTimestampsINS.values, y=tapTimestampsNSP.values, deg=degree)
    else:
        timeOffset = tapTimestampsNSP.values - tapTimestampsINS.values
        synchPolyCoeffsINStoNSP = np.array([1, np.mean(timeOffset)])
    timeInterpFunINStoNSP = np.poly1d(synchPolyCoeffsINStoNSP)
    if td is not None:
        td.loc[:, 'NSPTime'] = pd.Series(
            timeInterpFunINStoNSP(td['t']), index=td['t'].index)
    # accel['originalTime'] = accel['t']
    if accel is not None:
        accel.loc[:, 'NSPTime'] = pd.Series(
            timeInterpFunINStoNSP(accel['t']), index=accel['t'].index)
    if insBlock is not None:
        allUnits = [st.unit for st in insBlock.segments[0].spiketrains]
        for unit in allUnits:
                tStart = NSPTimeRanges[0]
                tStop = NSPTimeRanges[1]
                uniqueSt = []
                for st in unit.spiketrains:
                    if st not in uniqueSt:
                        uniqueSt.append(st)
                    else:
                        continue
                    if len(st.times):
                        segMask = np.array(
                            st.annotations['trialSegment'],
                            dtype=np.int) == trialSegment
                        st.magnitude[segMask] = (
                            timeInterpFunINStoNSP(st.times[segMask]))
                        #  kludgey fix for weirdness concerning t_start
                        st.t_start = min(tStart, st.times[0] * 0.999)
                        st.t_stop = min(tStop, st.times[-1] * 1.001)
                        validMask = st < st.t_stop
                        if ~validMask.all():
                            print('Deleted some spikes')
                            st = st[validMask]
                            if 'arrayAnnNames' in st.annotations.keys():
                                for key in st.annotations['arrayAnnNames']:
                                    st.annotations[key] = np.array(st.annotations[key])[validMask]          
                    else:
                        st.t_start = tStart
                        st.t_stop = tStop
        #  pdb.set_trace()
        allEvents = insBlock.filter(objects=Event)
        eventsDF = eventsToDataFrame(allEvents, idxT='t')
        segMask = getStimSerialTrialSegMask(eventsDF, trialSegment)
        for event in allEvents:
            event.magnitude[segMask] = (
                timeInterpFunINStoNSP(event.times[segMask]))
    return td, accel, insBlock, timeInterpFunINStoNSP


def timeOffsetBlock(block, timeOffset, masterTStart):
    for st in block.filter(objects=SpikeTrain):
        #  pdb.set_trace()
        if len(st.times):
            st.magnitude[:] = st.times.magnitude + timeOffset.magnitude
            st.t_start = min(masterTStart, st.times[0] * 0.999)
            st.t_stop = max(st.t_stop + timeOffset, st.times[-1] * 1.001)
        else:
            st.t_start += masterTStart
            st.t_stop += timeOffset
    for asig in block.filter(objects=AnalogSignal):
        asig.t_start += timeOffset
    for event in block.filter(objects=Event):
        event.magnitude[:] = event.magnitude + timeOffset.magnitude
    for epoch in block.filter(objects=Epoch):
        epoch.magnitude[:] = epoch.magnitude + timeOffset.magnitude
    return block


def blockUniqueUnits(block):

    return


def loadBlockProxyObjects(block):
    #  prune out proxy objects
    #  (will get added back)
    for metaIdx, chanIdx in enumerate(block.channel_indexes):
        if chanIdx.units:
            for unit in chanIdx.units:
                if unit.spiketrains:
                    unit.spiketrains = []
        if chanIdx.analogsignals:
            chanIdx.analogsignals = []
        if chanIdx.irregularlysampledsignals:
            chanIdx.irregularlysampledsignals = []

    for segIdx, seg in enumerate(block.segments):
        stProxyList = seg.spiketrains
        seg.spiketrains = []
        for stProxy in stProxyList:
            unit = stProxy.unit
            try:
                #  print('unit is {}'.format(stProxy.unit.name))
                #  print('spiketrain is {}'.format(stProxy.name))
                #  print('tstop is {}'.format(stProxy.t_stop))
                assert stProxy.shape[0] > 0, 'no times for this spike'
                st = stProxy.load(load_waveforms=True)
                #  st.left_sweep = None
                #  seems like writing ins data breaks the
                #  waveforms. oh well, turning it off for now
            except Exception:
                traceback.print_exc()
                #  pdb.set_trace()
                st = stProxy.load(load_waveforms=False)
                if st.waveforms is None:
                    st.waveforms = np.array([]).reshape((0, 0, 0))*pq.V
            #  link SpikeTrain and ID providing unit
            st.unit = unit
            # assign ownership to containers
            unit.spiketrains.append(st)
            seg.spiketrains.append(st)

        sigProxyList = seg.analogsignals
        seg.analogsignals = []
        for aSigProxy in sigProxyList:
            chanIdx = aSigProxy.channel_index
            asig = aSigProxy.load()
            #  link AnalogSignal and ID providing channel_index
            asig.channel_index = chanIdx
            # assign ownership to containers
            chanIdx.analogsignals.append(asig)
            seg.analogsignals.append(asig)
        
        seg.events = [i.load() for i in seg.events]
        seg.epochs = [i.load() for i in seg.epochs]
    block.create_relationship()
    return block


def extractSignalsFromBlock(
        block, keepSpikes=True,
        keepEvents=True, keepSignals=[]):
    newBlock = Block(name=block.name)
    newBlock.merge_annotations(block)
    for idx, seg in enumerate(block.segments):
        newSeg = Segment(name=seg.name)
        newSeg.merge_annotations(seg)
        newBlock.segments.append(newSeg)
        for asig in seg.analogsignals:
            if asig.name in keepSignals:
                newSeg.analogsignals.append(asig)
        if keepSpikes:
            newSeg.spiketrains = seg.spiketrains
        if keepEvents:
            newSeg.events = seg.events
            newSeg.epochs = seg.epochs

    newChanIdxList = []
    for idx, chanIdx in enumerate(block.channel_indexes):
        newChan = ChannelIndex(
            name=chanIdx.name, index=np.array([0]))
        newChan.merge_annotations(chanIdx)
        keepThisChan = False

        for asig in chanIdx.analogsignals:
            if asig.name in keepSignals:
                newChan.analogsignals.append(asig)
                asig.channel_index = newChan
                keepThisChan = True

        if keepSpikes:
            for unit in chanIdx.units:
                keepThisChan = True
                newChan.units.append(unit)
                unit.channel_index = newChan
        else:
            newChan.units = []

        if keepThisChan:
            newBlock.channel_indexes.append(newChan)
    newBlock.create_relationship()
    return newBlock


def getHUTtoINSSyncFun(
        timeSyncData,
        degree=1, plotting=False,
        trialSegments=None,
        syncTo='HostUnixTime'
        ):
    if trialSegments is None:
        trialSegments = pd.unique(timeSyncData['trialSegment'])
    timeInterpFunHUTtoINS = [None for i in trialSegments]

    for trialSegment in trialSegments:
        segmentMask = timeSyncData['trialSegment'] == trialSegment

        if degree > 0:
            synchPolyCoeffsHUTtoINS = np.polyfit(
                x=timeSyncData.loc[segmentMask, syncTo].values,
                y=timeSyncData.loc[segmentMask, 't'].values,
                deg=degree)
            #  print(
            #      'synchPolyCoeffsHUTtoINS = {}'.format(
            #          synchPolyCoeffsHUTtoINS))
        else:
            timeOffset = timeSyncData.loc[
                segmentMask, 't'].values - timeSyncData.loc[
                    segmentMask, syncTo].values * 1e-3
            synchPolyCoeffsHUTtoINS = np.array([1e-3, np.mean(timeOffset)])
        #  pdb.set_trace()
        timeInterpFunHUTtoINS[trialSegment] = np.poly1d(
            synchPolyCoeffsHUTtoINS)
        if plotting:
            plt.plot(
                timeSyncData.loc[segmentMask, syncTo].values * 1e-3,
                timeSyncData.loc[segmentMask, 't'].values, 'bo',
                label='original')
            plt.plot(
                timeSyncData.loc[segmentMask, syncTo].values * 1e-3,
                timeInterpFunHUTtoINS(
                    timeSyncData[syncTo].values), 'r-',
                label='first degree polynomial fit')
            plt.xlabel('Host Unix Time (msec)')
            plt.ylabel('INS Time (sec)')
            plt.legend()
            plt.show()
    return timeInterpFunHUTtoINS


def getStimSerialTrialSegMask(insDF, trialSegment):
    tsegMask = insDF['ins_property'] == 'trialSegment'
    tseg = pd.Series(np.nan, index=insDF.index)
    tseg.loc[tsegMask] = insDF.loc[tsegMask, 'ins_value']
    tseg.fillna(method='ffill', inplace=True)
    segmentMask = tseg == trialSegment
    return segmentMask

def synchronizeHUTtoINS(
        insDF, trialSegment, interpFun
        ):

    if 'INSTime' not in insDF.columns:
        insDF['INSTime'] = np.nan

    try:
        segmentMask = insDF['trialSegment'] == trialSegment
    except Exception:
        traceback.print_exc()
        #  event dataframes don't have an explicit trialSegment
        segmentMask = getStimSerialTrialSegMask(insDF, trialSegment)
        #  pdb.set_trace()
    insDF.loc[segmentMask, 'INSTime'] = interpFun(
        insDF.loc[segmentMask, 'HostUnixTime'])

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

        progAmpNames = rcsa_helpers.progAmpNames
        stripProgName = rcsa_helpers.strip_prog_name
        if logForm == 'serial':
            stimStatus = rcsa_helpers.extract_stim_meta_data_events(
                stimLog, trialSegment=idx)

        elif logForm == 'long':
            stimStatus = rcsa_helpers.extract_stim_meta_data(stimLog)
            stimStatus['activeProgram'] = stimStatus.loc[
                :, progAmpNames].idxmax(axis=1).apply(stripProgName)
            stimStatus['maxAmp'] = stimStatus.loc[:, progAmpNames].max(axis=1)
            stimStatus['trialSegment'] = idx
        
        allStimStatus.append(stimStatus)
    allStimStatusDF = pd.concat(allStimStatus, ignore_index=True)
    #  if all we're doing is turn something off, don't consider it a new round
    #  pdb.set_trace()
    if logForm == 'long':
        ampPositive = allStimStatusDF['maxAmp'].diff().fillna(0) >= 0
        allStimStatusDF['amplitudeIncrease'] = (
            allStimStatusDF['amplitudeChange'] & ampPositive)
        allStimStatusDF['amplitudeRound'] = allStimStatusDF[
            'amplitudeIncrease'].astype(np.float).cumsum()
    return allStimStatusDF


def stimStatusSerialtoLong(
        stimStSer, idxT='t', namePrefix='ins_', expandCols=[],
        deriveCols=[], progAmpNames=[], dropDuplicates=True, amplitudeCatBins=4):
    
    fullExpandCols = copy(expandCols)
    #  fixes issue with 'program' and 'amplitude' showing up unrequested
    if 'program' not in expandCols:
        fullExpandCols.append('program')
    if 'activeGroup' not in expandCols:
        fullExpandCols.append('activeGroup')

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

    debugPlot = False
    if debugPlot:
        stimCat = pd.concat((stimStLong, stimStSer), axis=1)

    for idx, pName in enumerate(progAmpNames):
        stimStLong[pName] = np.nan
        pMask = (stimStSer[namePrefix + 'property'] == 'amplitude') & (
            stimStLong['program'] == idx)
        stimStLong.loc[pMask, pName] = stimStSer.loc[pMask, namePrefix + 'value']
        stimStLong[pName].fillna(method='ffill', inplace=True)
        stimStLong[pName].fillna(method='bfill', inplace=True)
    #  pdb.set_trace()
    if dropDuplicates:
        stimStLong.drop_duplicates(subset=idxT, keep='last', inplace=True)
    
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        plt.show()

    ampIncrease = pd.Series(False, index=stimStLong.index)
    ampChange = pd.Series(False, index=stimStLong.index)
    for idx, pName in enumerate(progAmpNames):
        ampIncrease = ampIncrease | (stimStLong[pName].diff().fillna(0) > 0)
        ampChange = ampChange | (stimStLong[pName].diff().fillna(0) != 0)
        if debugPlot:
            plt.plot(stimStLong[pName].diff().fillna(0), label=pName)
    
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        ampIncrease.astype(float).plot(style='ro')
        ampChange.astype(float).plot(style='go')
        plt.legend()
        plt.show()

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
                ampsForSum[colName] = pd.cut(ampsForSum[colName], bins=1, labels=False)
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
    #  pdb.set_trace()
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


def gaussianSupport(tdSeg, peakIdx, gaussWid, fs):
    kernNSamp = min(int(gaussWid * fs), len(tdSeg.index)-1)
    
    gaussKern = signal.gaussian(
        kernNSamp, kernNSamp/6)

    support = pd.Series(0, index=tdSeg.index)
    support.loc[peakIdx] = 1
    support.iloc[:] = np.convolve(
        support.values,
        gaussKern, mode='same'
        )
    support = pd.Series(
        MinMaxScaler(feature_range=(1e-2, 1))
        .fit_transform(support.values.reshape(-1, 1))
        .squeeze(),
        index=support.index)
    return support


def noisyTriggerCorrection(
    tdSeries, fs, kernDur, order=1, invert=False, plotKernel=False):
    kernNSamp = int(kernDur * fs)
    if kernNSamp > len(tdSeries):
        kernNSamp = round(len(tdSeries) - 2)
    
    for orderIdx in range(order):
        gaussKern = signal.gaussian(
            kernNSamp, kernNSamp/(6 * 2**orderIdx))
        if orderIdx == 0:
            kern = np.diff(gaussKern)
        else:
            kern += np.diff(gaussKern)
    
    if invert: kern = -kern
    if plotKernel: plt.plot(kern); plt.show()

    correctionFactorUncorrected = pd.Series(
        np.convolve(kern, tdSeries, mode='same'),
        index=tdSeries.index)
    correctionFactor = np.abs(
        stats.zscore(
            correctionFactorUncorrected))
    return correctionFactor


def enhanceNoisyTriggers(tdSeries, correctionFactor):
    stimDetectSignal = pd.Series(
        ndimage.sobel(tdSeries, mode='reflect'),
        index=tdSeries.index)
    stimDetectSignal.iloc[:] = stimDetectSignal.values * correctionFactor.values
    stimDetectSignal = stimDetectSignal.fillna(0)
    stimDetectSignal.iloc[:] = stats.zscore(stimDetectSignal)
    stimDetectSignal = stimDetectSignal.abs()
    return stimDetectSignal

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
        min_dist = int(fs * iti * (1 - itiWiggle))
        peaks = np.array(range(len(crossIdx)))
        y = dataSeries.abs().values
        keep_max = True
        if peaks.size > 1 and min_dist > 1:
            print(len(peaks))
            if keep_max:
                highest = peaks[np.argsort(y[peaks])][::-1]
            else:
                highest = peaks
            rem = np.ones(y.size, dtype=bool)
            rem[peaks] = False

            for peak in highest:
                if not rem[peak]:
                    sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                    rem[sl] = True
                    rem[peak] = False

            peaks = np.arange(y.size)[~rem]
            crossIdx = crossIdx[peaks]
            crossMask = crossMask.iloc[peaks]
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
    elif edgeType == 'both':
        triggersPrime = triggersPrime.abs()
    # moments when camera capture occured (iloc style indexes!)
    if keep_max:
        keep_what='max'
    else:
        keep_what='first'
    
    triggersPrimeVals = triggersPrime.values.squeeze()
    peakIdx = peakutils.indexes(
        triggersPrimeVals, thres=thres,
        min_dist=width, thres_abs=True, keep_what=keep_what)
    
    if minAmp is not None:
        #  pdb.set_trace()
        triggersZScore = stats.zscore(dataSeries)
        triggersAtPeaks = triggersZScore[peakIdx]
        peakIdx = peakIdx[triggersAtPeaks > minAmp]

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

    return dataSeries.index[peakIdx]

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
    # NSP time of first camera trigger (TODO is it the first one after a pause?)
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
    show = False, prevFig = None, prevAx = None, zoomAxis = True, timeRange = (0,-1)):
    # Plot the data channel
    ch_idx  = whichChan

    #hdr_idx = channelData['ExtendedHeaderIndices'][channelData['elec_ids'].index(whichChan)]
    if not prevFig:
        #plt.figure()
        f, ax = plt.subplots()
    else:
        f = prevFig
        if prevAx is None:
            ax = prevFig.axes[0]
        else:
            ax = prevAx

    #  pdb.set_trace()
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

            if len(unitsOnThisChan):
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

def plotBinnedSpikes(spikeMat, show=True, normalizationType='linear',
    zAxis=None, ax=None, labelTxt='Spk/s'):

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
    cbAxPos = [axPos.x0 + axPos.width * 1.05, axPos.y0,
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
        labelpad = 3 * labelFontSize)
    ax.set_ylabel("Unit (#)", fontsize = labelFontSize,
        labelpad = 3 * labelFontSize)
    #plt.show()

    ax.tick_params(left=False, top=False, right=False, bottom=False,
        labelleft=False, labeltop=False, labelright=False,
        labelbottom=False)
    if show:
        plt.show()
    return fi, im

def plot_spikes(
        spikes, channel, ignoreUnits = [], showNow = False, ax = None,
        acrossArray = False, xcoords = None, ycoords = None,
        axesLabel = False, channelPlottingName = None, chanNameInLegend=True,
        legendTags=[],
        maskSpikes=None, maxSpikes=200, lineAlpha=0.025):
        
    if channelPlottingName is None:
        channelPlottingName = str(channel)

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = pd.unique(spikes['Classification'][ChanIdx])
    if 'ClassificationLabel' in spikes.keys():
        unitsLabelsOnThisChan = pd.unique(spikes['ClassificationLabel'][ChanIdx])
    else:
        unitsLabelsOnThisChan = None

    if acrossArray:
        # Check that we didn't ask to plot the spikes across channels into a single axis
        assert ax is None
        # Check that we have waveform data everywhere
        assert len(spikes['Waveforms'][ChanIdx].shape) == 3

    if ax is None and not acrossArray:
        fig, ax = plt.subplots()
        fig.set_tight_layout({'pad': 0.01})
    if ax is not None and not acrossArray:
        fig = ax.figure

    if unitsOnThisChan is not None:
        if acrossArray:
            xIdx, yIdx = coordsToIndices(xcoords, ycoords)
            fig, ax = plt.subplots(nrows = max(np.unique(xIdx)) + 1, ncols = max(np.unique(yIdx)) + 1)
            fig.set_tight_layout({'pad': 0.01})
        colorPalette = sns.color_palette(n_colors = 40)
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            #print('ignoreUnits are {}'.format([-1] + ignoreUnits))
            if unitName in [-1] + ignoreUnits:
                #  always ignore units marked -1
                continue
            unitMask = spikes['Classification'][ChanIdx] == unitName

            if 'ClassificationLabel' in spikes.keys():
                unitPlottingName = unitsLabelsOnThisChan[unitIdx]
            else:
                unitPlottingName = unitName
            if chanNameInLegend:
                labelName = 'chan %s, unit %s' % (channelPlottingName, unitPlottingName)
            else:
                labelName = 'unit %s' % (unitPlottingName)
            for legendTag in legendTags:
                if legendTag in spikes['basic_headers']:
                    labelName = '{} {}: {}'.format(
                        labelName,  legendTag,
                        spikes['basic_headers'][legendTag][unitName]
                    )
            #  plot all channels aligned to this spike?
            if acrossArray:
                waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, 0]
                if maxSpikes < waveForms.shape[0]:
                        selectIdx = random.sample(range(waveForms.shape[0]), maxSpikes)
                for idx, channel in enumerate(spikes['ChannelID']):
                    curAx = ax[xIdx[idx], yIdx[idx]]
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, idx]
                    if maskSpikes is not None:
                        waveForms = waveForms[maskSpikes, :]
                    if maxSpikes < waveForms.shape[0]:
                        waveForms = waveForms[selectIdx]
                    timeRange = np.arange(len(waveForms[0])) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                    for spIdx, thisSpike in enumerate(waveForms):
                        thisLabel = labelName if idx == 0 else None
                        curAx.plot(
                            timeRange, thisSpike, label=thisLabel,
                            linewidth=1, color=colorPalette[unitIdx], alpha=lineAlpha)
                sns.despine()
                for curAx in ax.flatten():
                    curAx.tick_params(left='off', top='off', right='off',
                    bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')
            else:
                #  plot only on main chan
                if len(spikes['Waveforms'][ChanIdx].shape) == 3:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, ChanIdx]
                else:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :]
                
                timeRange = np.arange(len(waveForms[0])) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                
                if maskSpikes is not None:
                    waveForms = waveForms[maskSpikes, :]
                if maxSpikes < waveForms.shape[0]:
                    waveForms = waveForms[random.sample(range(waveForms.shape[0]), maxSpikes), :]
                for spIdx, thisSpike in enumerate(waveForms):
                    thisLabel = labelName if spIdx == 0 else None
                    ax.plot(
                        timeRange, thisSpike,
                        linewidth=1, color=colorPalette[unitIdx], alpha=lineAlpha, label=thisLabel)
                ax.set_xlim(timeRange[0], timeRange[-1])
                if axesLabel:
                    ax.set_ylabel(spikes['Units'])
                    ax.set_xlabel('Time (msec)')
                    ax.set_title('Units on channel {}'.format(channelPlottingName))
                    ax.legend()
        if showNow:
            plt.show()
    return fig, ax

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


def getLastSpikeTime(spikes):
    lastTimestamp = -1
    for idx, timestamps in enumerate(spikes['TimeStamps']):
        lastTimestamp = max(lastTimestamp, timestamps[-1])
    return lastTimestamp


def getAllUnits(spikes):
    allUnitsList = []
    for idx, chanIdx in enumerate(spikes['ChannelID']):
        unitsOnThisChan = pd.unique(spikes['Classification'][idx])
        allUnitsList.append(pd.Series(unitsOnThisChan))
    allUnits = pd.unique(pd.concat(allUnitsList))
    return allUnits


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem
