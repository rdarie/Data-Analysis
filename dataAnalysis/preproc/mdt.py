import pandas as pd
import os, pdb
from neo import (
    AnalogSignal, Event, Block,
    Segment, ChannelIndex, SpikeTrain, Unit)
import neo
from collections.abc import Iterable
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import traceback
import itertools
import sys
#  import pickle
from copy import copy
import dataAnalysis.preproc.mdt_constants as mdt_constants
import quantities as pq
#  import argparse, linecache
from scipy import stats, signal, ndimage
import peakutils
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import datetime
from datetime import datetime as dt
import json
from copy import copy
sys.stderr = open(os.devnull, "w")  # silence stderr
import elephant as elph
sys.stderr = sys.__stderr__  # unsilence stderr

# INSReferenceTime = pd.Timestamp('2018-03-01')
INSReferenceTime = pd.Timestamp('2019-01-01')


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


def processMetaMatrixV2(
        inputJson, streamType,
        sampleRateLookupDict=None, frameDuration=None,
        tossPackets=True, fixTimeShifts=False,
        useApparentFS=False,
        verbose=False, makePlots=False):
    # count frame sizes
    if streamType == 'accel':
        metaMatrix, metaFig, metaAx, metaTwinAx = rcsa_helpers.extract_accel_meta_data(
            inputJson, sampleRateLookupDict=sampleRateLookupDict,
            plotting=makePlots)
    elif streamType == 'td':
        metaMatrix, metaFig, metaAx, metaTwinAx = rcsa_helpers.extract_td_meta_data(
            inputJson, sampleRateLookupDict=sampleRateLookupDict,
            frameDuration=frameDuration, plotting=makePlots)
    originalMetaDFColumns = rcsa_helpers.metaMatrixColumns
    metaDF = pd.DataFrame(
        metaMatrix, columns=originalMetaDFColumns)
    metaDF['microloss'] = metaDF['microloss'].astype(bool)
    metaDF['macroloss'] = metaDF['macroloss'].astype(bool)
    metaDF['bothloss'] = metaDF['bothloss'].astype(bool)
    # assume n channels is constant for all assembled sessions
    mostCommonSampleRateCode = metaDF['SampleRate'].value_counts().idxmax()
    fs = float(sampleRateLookupDict[mostCommonSampleRateCode])
    # system ticks are 100usec long
    intersampleTickCount = (fs ** -1) / (100e-6)
    if streamType == 'accel':
        # all accel packets are 8 samples long
        frameDuration = 8 * intersampleTickCount * 1e-1
    metaDF['dataPayloadDurSysTick'] = metaDF['lastSampleTick'].diff() * 1e-4
    #
    if useApparentFS:
        samplePeriodsSysTick = (
            metaDF['dataPayloadDurSysTick'].fillna(method='bfill') /
            metaDF['dataSizePerChannel'])
        # sns.distplot(samplePeriodsSysTick); plt.show()
        # by default, supportFraction = (samplePeriodsSysTick.shape[0] + 2) / (2 * samplePeriodsSysTick.shape[0])
        noDropMask = ~(metaDF['microloss'] | metaDF['macroloss']).to_numpy()
        try:
            cov = (
                MinCovDet(support_fraction=0.75)
                .fit(
                    samplePeriodsSysTick
                    .to_numpy()[noDropMask]
                    .reshape(-1, 1)))
            apparentSamplingPeriod = np.round(cov.location_[0], decimals=6)
        except Exception:
            traceback.print_exc()
            # apparentSamplingPeriod = samplePeriodsSysTick.median()
            apparentSamplingPeriod = np.round(
                samplePeriodsSysTick.loc[noDropMask].median(), decimals=6)
        apparentSamplingPeriodSEM = stats.sem(
            samplePeriodsSysTick.loc[noDropMask], nan_policy='omit')
        fs = apparentSamplingPeriod ** (-1)
        # intersampleTickCount = int(apparentSamplingPeriod / (100e-6))
        intersampleTickCount = apparentSamplingPeriod / (100e-6)
        if streamType == 'accel':
            # all accel packets are 8 samples long
            frameDuration = 8 * intersampleTickCount
        print(
            'Derived sampling interval is {:.5f} (SEM = {:.2e})'
            .format(apparentSamplingPeriod, apparentSamplingPeriodSEM))
        print(
            'Using derived sampling freq of {}'.format(fs))
        if streamType == 'accel':
            metaMatrix, metaFig, metaAx, metaTwinAx = rcsa_helpers.extract_accel_meta_data(
                inputJson, sampleRateLookupDict=sampleRateLookupDict,
                intersampleTickCount=intersampleTickCount,
                plotting=makePlots)
        elif streamType == 'td':
            metaMatrix, metaFig, metaAx, metaTwinAx = rcsa_helpers.extract_td_meta_data(
                inputJson, sampleRateLookupDict=sampleRateLookupDict,
                intersampleTickCount=intersampleTickCount,
                plotting=makePlots)
        metaDF = pd.DataFrame(
            metaMatrix, columns=originalMetaDFColumns)
        metaDF['microloss'] = metaDF['microloss'].astype(bool)
        metaDF['macroloss'] = metaDF['macroloss'].astype(bool)
        metaDF['bothloss'] = metaDF['bothloss'].astype(bool)
        metaDF['dataPayloadDurSysTick'] = metaDF['lastSampleTick'].diff() * 1e-4
    else:
        print('Using sampling freq of {}'.format(fs))
    # frameDuration is in msec
    nominalFrameDur = frameDuration * 1e-3
    nChan = int(metaDF['chanSamplesLen'].iloc[0])
    if fixTimeShifts:
        metaMatrix =\
            rcsa_helpers.correct_meta_matrix_time_displacement(
                metaMatrix, intersampleTickCount)
        metaMatrix = rcsa_helpers.code_micro_and_macro_packet_loss(
            metaMatrix)
        metaDF.loc[:, originalMetaDFColumns] = metaMatrix
        metaDF['microloss'] = metaDF['microloss'].astype(bool)
        metaDF['macroloss'] = metaDF['macroloss'].astype(bool)
        metaDF['bothloss'] = metaDF['bothloss'].astype(bool)
        packetsNeededFixing = metaDF.loc[metaDF['packetNeedsFixing'], :].index
    else:
        packetsNeededFixing = pd.Index([])
    if verbose:
        print('{} microlosses'.format(metaMatrix[:, 4].sum()))
        print('{} macrolosses'.format(metaMatrix[:, 5].sum()))
        print('{} coincident losses'.format(metaMatrix[:, 6].sum()))
    #
    metaDF['anyloss'] = (metaDF['microloss'] | metaDF['macroloss'])
    #
    metaDF['dataPayloadDur'] = metaDF['dataSizePerChannel'] * (fs ** -1)
    metaDF.loc[metaDF.index[0], 'dataPayloadDur'] = 0
    #
    metaDF['recordDur'] = metaDF['dataPayloadDur'].cumsum()
    metaDF['recordDurWhole'] = (
        nominalFrameDur *
        np.round(
            metaDF['recordDur'] / nominalFrameDur))
    metaDF['frameLenMismatch'] = metaDF['recordDurWhole'] - metaDF['recordDur']
    # will hold duration corrections based on sysTick
    metaDF['packetTimeCorrection'] = 0
    #
    metaDF['dataPayloadDurPGT'] = metaDF['PacketGenTime'].diff() * 1e-3
    metaDF.loc[metaDF.index[0], 'dataPayloadDurPGT'] = 0
    metaDF['recordDurPGT'] = metaDF['dataPayloadDurPGT'].cumsum()
    metaDF['recordDurWholePGT'] = (
        nominalFrameDur *
        np.round(
            metaDF['recordDurPGT'] / nominalFrameDur))
    metaDF['frameLenMismatchPGT'] = (
        metaDF['recordDurWholePGT'] -
        metaDF['recordDurPGT'])
    metaDF['recordDurMismatchPGT'] = (
        metaDF['recordDurPGT'] - metaDF['recordDur'])
    #
    metaDF['recordDurSysTick'] = metaDF['dataPayloadDurSysTick'].cumsum()
    metaDF.loc[metaDF.index[0], 'dataPayloadDurSysTick'] = 0
    metaDF['recordDurWholeSysTick'] = (
        nominalFrameDur *
        np.round(metaDF['recordDurSysTick'].fillna(1) / nominalFrameDur))
    metaDF.loc[metaDF.index[0], 'recordDurWholeSysTick'] = 0
    metaDF['frameLenMismatchSysTick'] = (
        metaDF['recordDurWholeSysTick'] -
        metaDF['recordDurSysTick'])
    metaDF['recordDurMismatch'] = metaDF['recordDurSysTick'] - metaDF['recordDur']
    ##
    metaDF.loc[:, 'packetIdx'] = metaDF.index
    ##
    if makePlots:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15, 15))
        # xAxisVariable = 'PacketRxUnixTime'
        # xAxisLabel = 'Packet Unix RX time (msec)'
        # plotExtent = 250000
        xAxisVariable = 'packetIdx'
        xAxisLabel = 'Packet number'
        plotExtent = 400
        zoomAxes = False
        if zoomAxes:
            if metaDF['anyloss'].any():
                newXMin = max(
                    metaDF.loc[
                        metaDF['anyloss'],
                        xAxisVariable].iloc[0] - int(plotExtent/4),
                    metaDF[xAxisVariable].iloc[0])
                newXMax = min(
                    newXMin + plotExtent,
                    metaDF[xAxisVariable].iloc[-1])
            else:
                newXMin = metaDF[xAxisVariable].iloc[0]
                newXMax = min(
                    metaDF[xAxisVariable].iloc[0] + plotExtent,
                    metaDF[xAxisVariable].iloc[-1])
            plotMask = (
                (metaDF[xAxisVariable] > newXMin) &
                (metaDF[xAxisVariable] < newXMax)
                ).to_numpy()
            # ax[0].set_xlim([newXMin, newXMax])
        else:
            plotMask = np.ones(metaDF.index.shape).astype(bool)
        ax[0].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'dataPayloadDurPGT'],
            label='frame length per packet (packet gen time)'
            )
        ax[0].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'dataPayloadDur'],
            label='frame length per packet'
            )
        ax[0].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'dataPayloadDurSysTick'],
            label='frame length per packet (systick)'
            )
        ax[1].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'frameLenMismatchPGT'],
            label='difference from nominal length per packet (packet gen time)',
            )
        ax[1].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'frameLenMismatch'],
            '.-',
            label='difference from nominal length per packet',
            )
        ax[1].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'frameLenMismatchSysTick'],
            label='difference from nominal length per packet (systick)',
            )
        ax[1].plot(
            metaDF.loc[metaDF['anyloss'] & plotMask, xAxisVariable],
            1e3 * metaDF.loc[metaDF['anyloss'] & plotMask, 'frameLenMismatch'],
            'y*', label='lost packet(s)'
            )
        ax[1].legend(loc='upper right')
        ax[1].set_ylabel('time interval (msec)')
        ax[2].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'recordDurMismatchPGT'],
            label='cumulative unresolved deviation from packetGen time            ',
            zorder=1)
        ax[2].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'recordDurMismatch'],
            label='cumulative unresolved deviation from systick time            ',
            zorder=1)
    else:
        fig, ax = None, None
    ###########################################################################
    for rowIdx in metaDF.loc[metaDF['anyloss'], :].index:
        thisMismatch = metaDF.loc[
            rowIdx, 'recordDurMismatch']
        thisCorrection = (
            (fs ** -1) *
            np.round(
                thisMismatch / (fs ** -1)))
        # print('{:.4f}'.format(thisCorrection))
        if thisCorrection > 0:
            metaDF.loc[
                rowIdx, 'packetTimeCorrection'] = thisCorrection
        metaDF['recordDur'] = (
            metaDF['dataPayloadDur'] + metaDF['packetTimeCorrection']).cumsum()
        metaDF['recordDurMismatch'] = metaDF['recordDurSysTick'] - metaDF['recordDur']
    metaDF['recordDurMismatch'] = (
        metaDF['recordDurSysTick'] - metaDF['recordDur'])
    metaDF['recordDurMismatchPGT'] = (
        metaDF['recordDurPGT'] - metaDF['recordDur'])
    if makePlots:
        ax[0].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'packetTimeCorrection'],
            label='length correction applied to each packet'
            )
        ax[0].plot(
            metaDF.loc[
                metaDF['anyloss'] & plotMask,
                xAxisVariable],
            1e3 * metaDF.loc[
                metaDF['anyloss'] & plotMask,
                'dataPayloadDur'],
            'y*', label='lost packet(s)'
            )
        ax[2].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'recordDurMismatchPGT'],
            label='cumulative unresolved deviation from packetGen time (after fix)',
            zorder=1)
        ax[2].plot(
            metaDF.loc[plotMask, xAxisVariable],
            1e3 * metaDF.loc[plotMask, 'recordDurMismatch'],
            label='cumulative unresolved deviation from systick time (after fix)',
            zorder=1)
        ax[2].text(
            0.1, 0.1,
            'min: {:.1f} max: {:.1f}'.format(
                1e3 * metaDF['recordDurMismatch'].min(),
                1e3 * metaDF['recordDurMismatch'].max()
            ),
            horizontalalignment='left', verticalalignment='baseline',
            transform=ax[2].transAxes)
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('time interval (msec)')
        ax[2].set_ylabel('time interval (msec)')
        ax[2].set_xlabel(xAxisLabel)
        ax[2].legend(loc='upper right')
    if tossPackets:
        tossMask = np.zeros((metaMatrix.shape[0])).astype(bool)
        dataTypeSeqIndex = np.flatnonzero(
            np.diff(metaMatrix[:, 1]) == 0)
        tossMask[dataTypeSeqIndex] = True
        sysTickIndex = np.flatnonzero(
            np.diff(metaMatrix[:, 2]) == 0)
        tossMask[sysTickIndex] = True
    else:
        tossMask = None
    metaDF.drop(columns=['packetIdx'], inplace=True)
    return (
        metaDF, nChan, fs, intersampleTickCount,
        nominalFrameDur, packetsNeededFixing,
        tossMask, fig, ax, metaFig, metaAx, metaTwinAx)


def getINSTDFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        frameDuration=50,
        assumeNoSamplesLost=True, verbose=True,
        forceRecalc=True, getInterpolated=True,
        upsampleRate=None, interpKind='linear',
        fixTimeShifts=False, tossPackets=True,
        assumeConsistentPacketSizes=True,
        makePlots=False, showPlots=False,
        figureOutputFolder=None, blockIdx=None
        ):

    if not isinstance(sessionNames, Iterable):
        sessionNames = [sessionNames]

    tdSessions = []
    for sessionIdx, sessionName in enumerate(sessionNames):
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
            tdSessions.append(tdData)
        except Exception:
            # traceback.print_exc()
            try:
                with open(os.path.join(jsonPath, 'RawDataTD.json'), 'r') as f:
                    timeDomainJson = json.load(f)
            except Exception:
                with open(os.path.join(jsonPath, 'RawDataTD.json'), 'r') as f:
                    timeDomainJsonText = f.read()
                    timeDomainJsonText = fixMalformedJson(timeDomainJsonText)
                    timeDomainJson = json.loads(timeDomainJsonText)
            #
            # warnings.filterwarnings("error")
            (
                tdMetaDF, nChan, fs,
                intersampleTickCount, nominalFrameDur,
                packetsNeededFixing, tossMask,
                fig, ax,
                metaFig, metaAx, metaTwinAx) = processMetaMatrixV2(
                    timeDomainJson, streamType='td', verbose=verbose,
                    frameDuration=frameDuration,
                    sampleRateLookupDict=mdt_constants.TdSampleRates,
                    tossPackets=tossPackets, fixTimeShifts=fixTimeShifts,
                    makePlots=makePlots)
            if makePlots:
                (
                    (tdMetaDF['dataSizePerChannel'] * (fs ** -1))
                    .value_counts()
                    .to_csv(
                        os.path.join(
                            figureOutputFolder,
                            'frame_size_counts_{:0>3}_TrialSeg{:0>1}_td.csv'.format(
                                blockIdx, sessionIdx)),
                        header=True))
                figSaveOpts = dict(
                    bbox_extra_artists=(thisAx.get_legend() for thisAx in ax),
                    bbox_inches='tight')
                ax[0].set_title(
                    'Block{:0>3}_TrialSeg{:0>1}_td'.format(
                        blockIdx, sessionIdx)
                    )
                #
                pdfPath = os.path.join(
                    figureOutputFolder,
                    'packet_jitter_{:0>3}_TrialSeg{:0>1}_td.pdf'.format(
                        blockIdx, sessionIdx))
                fig.savefig(pdfPath, **figSaveOpts)
                #
                pdfPath = os.path.join(
                    figureOutputFolder,
                    'meta_info_{:0>3}_TrialSeg{:0>1}_td.pdf'.format(
                        blockIdx, sessionIdx))
                metaFig.savefig(pdfPath, **figSaveOpts)
                if showPlots:
                    plt.show()
                else:
                    plt.close()
            #
            timeDomainValues = rcsa_helpers.unpacker_td(
                tdMetaDF.to_numpy(), timeDomainJson, intersampleTickCount)
            #  save the noninterpolated files to disk
            tdData = rcsa_helpers.save_to_disk(
                timeDomainValues, os.path.join(
                    jsonPath, 'RawDataTD.csv'),
                time_format='full', data_type='td', num_cols=nChan)
            if tossPackets:
                tossMaskFinal = tdData['packetIdx'].isin(
                    np.flatnonzero(tossMask))
                tdData = tdData.loc[~tossMaskFinal, :]
            # RD 2021-01-28
            # tdData['t'] = (
            #     tdData['actual_time'] - INSReferenceTime) / (
            #         datetime.timedelta(seconds=1))
            tdData['t'] = tdData['microseconds'] / pd.Timedelta('1S')
            tdData.drop(columns=['microseconds', 'coarseClock'], inplace=True)
            if sessionIdx == 0:
                tZero = tdData['t'].iloc[0]
            if not assumeNoSamplesLost:
                tdData = tdData.drop_duplicates(
                    ['t']
                    ).sort_values('t').reset_index(drop=True)
            else:
                # main workflow as of 12-15-2020
                tdData['equidistantT'] = np.nan
                lastPacketStartTime = tdData['t'].iloc[0]
                lastPacketT = lastPacketStartTime - fs ** (-1)
                for name, group in tdData.groupby('packetIdx'):
                    # metaRowIdx = tdMetaDF.loc[tdMetaDF['packetIdx'] == name].index
                    # assert metaRowIdx.size == 1
                    # metaRowIdx = tdMetaDF.loc[tdMetaDF['packetIdx'] == name].index[0]
                    # get time difference from sysTick
                    misalignTimeDiff = group['t'].iloc[0] - lastPacketT
                    startTimeOver = tdMetaDF.loc[name, 'anyloss']
                    if verbose and (name in packetsNeededFixing):
                        print('    fixed packet time shifted: t = {:+.4f}'.format(
                            group['t'].iloc[0] - tZero))
                    if startTimeOver:
                        if verbose:
                            print(
                                '    starting over; t = {:+.4f}; time skip = {:+.4f}'
                                .format(
                                    group['t'].iloc[0] - tZero,
                                    misalignTimeDiff))
                        if assumeConsistentPacketSizes:
                            # if using original version of processMetaMatrix
                            # packetTimeCorrection = (
                            #     tdMetaDF.loc[
                            #         name,
                            #         'frameLenMissingPacketCorrection'] +
                            #     nominalFrameDur *
                            #     np.round(misalignTimeDiff / nominalFrameDur))
                            # if using processMetaMatrixV2
                            packetTimeCorrection = (
                                tdMetaDF.loc[
                                    name,
                                    'packetTimeCorrection'])
                            if verbose:
                                print(
                                    '    applying {:+.4f} corection'
                                    .format(packetTimeCorrection))
                            lastPacketStartTime = (
                                lastPacketStartTime +
                                packetTimeCorrection)
                        else:
                            lastPacketStartTime = group['t'].iloc[0]
                    else:
                        if verbose and False:
                            print(
                                'not starting over; t = {:+.4f}; time skip = {:+.4f}'
                                .format(
                                    group['t'].iloc[0] - tZero,
                                    misalignTimeDiff))
                    tdData.loc[group.index, 'equidistantT'] = (
                        lastPacketStartTime +
                        fs ** (-1) * np.arange(0, group.shape[0]))
                    lastPacketStartTime = (
                        tdData.loc[group.index, 'equidistantT'].iloc[-1] +
                        fs ** (-1))
                    lastPacketT = group['t'].iloc[-1] + fs ** (-1)
                    # lastPacketStartCoarseTime = (
                    #     tdData.loc[group.index, 'coarseClock'].iloc[-1])
                tdData['t'] = tdData['equidistantT'].to_numpy()
                tdData.drop('equidistantT', inplace=True, axis='columns')
            if getInterpolated:
                if upsampleRate is not None:
                    fs = fs * upsampleRate
                uniformT = np.arange(
                    tdData['t'].iloc[0],
                    tdData['t'].iloc[-1] + float(fs) ** (-1),
                    float(fs) ** (-1))
                #  convert to floats before interpolating
                # tdData['microseconds'] = tdData['microseconds'] / (
                #     datetime.timedelta(microseconds=1))
                tdData['time_master'] = (
                    tdData['time_master'] - pd.Timestamp('2000-03-01')) / (
                    datetime.timedelta(seconds=1))
                HUTMapping = tdMetaDF['PacketGenTime']
                tdData['PacketGenTime'] = tdData['packetIdx'].map(HUTMapping)
                channelsPresent = [
                    i for i in tdData.columns if 'channel_' in i]
                channelsPresent += [
                    'time_master',
                    # 'microseconds',
                    'PacketGenTime']
                #
                tdDataInterp = hf.interpolateDF(
                    tdData, uniformT, x='t', kind=interpKind,
                    columns=channelsPresent, fill_value=(0, 0))
                #  interpolating converts to floats, recover
                tdDataInterp.loc[:, 'microseconds'] = pd.to_timedelta(
                    tdDataInterp['t'] * 1e6, unit='us')
                tdDataInterp.loc[:, 'time_master'] = pd.to_datetime(
                    tdDataInterp['time_master'], unit='s',
                    origin=pd.Timestamp('2000-03-01'))
                #  tdData['actual_time'] = tdData['time_master'] + (
                #      tdData['microseconds'])
                tdDataInterp.loc[:, 'trialSegment'] = sessionIdx
                tdDataInterp.to_csv(
                    os.path.join(jsonPath, 'RawDataTD_interpolated.csv'))
                tdSessions.append(tdDataInterp)
            else:
                tdData.loc[:, 'trialSegment'] = sessionIdx
                tdData.to_csv(
                    os.path.join(jsonPath, 'RawDataTD.csv'))
                tdSessions.append(tdData)
        print('Wrote insSession {}'.format(sessionIdx))
    '''
    td = {
        'data': pd.concat(tdSessions, ignore_index=True),
        't': None
        }
    td['t'] = td['data']['t']
    td['data']['INSTime'] = td['data']['t']
    td['INSTime'] = td['t']
    '''
    tdDF = pd.concat(tdSessions, ignore_index=True)
    tdDF.loc[:, 'INSTime'] = tdDF['t']
    return tdDF


def getINSTimeSyncFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        makePlots=False, showPlots=False,
        figureOutputFolder=None, 
        blockIdx=None, forceRecalc=True
        ):
    if not isinstance(sessionNames, Iterable):
        sessionNames = [sessionNames]
    tsSessions = []
    for sessionIdx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        try:
            if forceRecalc:
                raise(Exception('Debugging, always extract fresh'))
            #
            timeSyncData = pd.read_csv(os.path.join(jsonPath, 'TimeSync.csv'))
            #  loading from csv removes datetime formatting, recover it:
            timeSyncData['microseconds'] = pd.to_timedelta(
                timeSyncData['microseconds'], unit='us')
            timeSyncData['time_master'] = pd.to_datetime(
                timeSyncData['time_master'])
            timeSyncData['actual_time'] = pd.to_datetime(
                timeSyncData['actual_time'])
        except Exception:
            # traceback.print_exc()
            try:
                with open(os.path.join(jsonPath, 'TimeSync.json'), 'r') as f:
                    timeSync = json.load(f)[0]
            except Exception:
                with open(os.path.join(jsonPath, 'TimeSync.json'), 'r') as f:
                    timeSyncText = f.read()
                    timeSyncText = fixMalformedJson(timeSyncText)
                    timeSync = json.loads(timeSyncText)[0]
            timeSyncData = rcsa_helpers.extract_time_sync_meta_data(timeSync)
            #
            if makePlots:
                fig, ax = plt.subplots(2, 1, figsize=(15, 15))
                # ax[0] refers to the latency
                actualLatency = (
                    timeSyncData['PacketRxUnixTime'] - timeSyncData['PacketGenTime'])
                #
                ax[0].plot(
                    timeSyncData.index, timeSyncData['LatencyMilliseconds'],
                    '-o', label='LatencyMilliseconds')
                ax[0].plot(
                    timeSyncData.index, actualLatency,
                    '-o', label='PacketRxUnixTime - PacketGenTime')
                ax[0].set_ylabel('latency (msec)')
                ax[0].legend(loc='upper right')
                # ax[1] refers to the seconds counter "timestamps"
                p, = ax[1].plot(
                    timeSyncData.index, timeSyncData['timestamp'],
                    'r-o', label='timestamp')
                ax[1].legend(loc='upper right')
                ax[1].set_ylabel('timestamp (sec)')
                ax[1].set_xlabel('(Packet #)')
                ax[1].yaxis.get_label().set_color(p.get_color())
                tsDiffAx = ax[1].twinx()
                tsDiff = timeSyncData['timestamp'].diff()
                twinP, = tsDiffAx.plot(
                    tsDiff.index, tsDiff,
                    'm-o', label='timestamp diff')
                tsDiffAx.legend(loc='lower right')
                tsDiffAx.set_ylabel('timestamp diff (sec)')
                tsDiffAx.yaxis.get_label().set_color(twinP.get_color())
                figSaveOpts = dict(
                    bbox_extra_artists=(thisAx.get_legend() for thisAx in ax),
                    bbox_inches='tight')
                ax[0].set_title(
                    '{:0>3}_TrialSeg{:0>1}_time_sync'.format(
                        blockIdx, sessionIdx)
                    )
                pdfPath = os.path.join(
                    figureOutputFolder,
                    'time_sync_{:0>3}_TrialSeg{:0>1}.pdf'.format(
                        blockIdx, sessionIdx))
                plt.savefig(pdfPath, **figSaveOpts)
                if showPlots:
                    plt.show()
                else:
                    plt.close()
            #
            timeSyncData['trialSegment'] = sessionIdx
            timeSyncData.to_csv(os.path.join(jsonPath, 'TimeSync.csv'))
        # timeSyncData['t'] = (
        #     timeSyncData['actual_time'] - INSReferenceTime) / (
        #         datetime.timedelta(seconds=1))
        timeSyncData['t'] = timeSyncData['microseconds'] / pd.Timedelta('1S')
        timeSyncData.to_csv(os.path.join(jsonPath, 'TimeSync.csv'))
        tsSessions.append(timeSyncData)
    #
    allTimeSync = pd.concat(tsSessions, ignore_index=True)
    return allTimeSync


def getINSAccelFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        fs=65.104, forceRecalc=True, getInterpolated=True,
        assumeNoSamplesLost=True, verbose=False,
        fixTimeShifts=False, tossPackets=True,
        assumeConsistentPacketSizes=True,
        makePlots=False,
        showPlots=False,
        figureOutputFolder=None, blockIdx=None
        ):

    if not isinstance(sessionNames, Iterable):
        sessionNames = [sessionNames]

    accelSessions = []
    for sessionIdx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        print('getINSAccelFromJson( Loading {}'.format(jsonPath))
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
            # traceback.print_exc()
            try:
                with open(os.path.join(jsonPath, 'RawDataAccel.json'), 'r') as f:
                    accelJson = json.load(f)
            except Exception:
                with open(os.path.join(jsonPath, 'RawDataAccel.json'), 'r') as f:
                    accelJsonText = f.read()
                    accelJsonText = fixMalformedJson(accelJsonText)
                    accelJson = json.loads(accelJsonText)
            (
                accelMetaDF, nChan, fs,
                intersampleTickCount, nominalFrameDur,
                packetsNeededFixing, tossMask,
                fig, ax, metaFig, metaAx, metaTwinAx) = processMetaMatrixV2(
                    accelJson, streamType='accel', verbose=verbose,
                    sampleRateLookupDict=mdt_constants.AccelSampleRate,
                    tossPackets=tossPackets, fixTimeShifts=fixTimeShifts,
                    makePlots=makePlots)
            if makePlots:
                (
                    (accelMetaDF['dataSizePerChannel'] * (fs ** -1))
                    .value_counts()
                    .to_csv(
                        os.path.join(
                            figureOutputFolder,
                            'frame_size_counts_{:0>3}_TrialSeg{:0>1}_accel.csv'.format(
                                blockIdx, sessionIdx)),
                        header=True))
                figSaveOpts = dict(
                    bbox_extra_artists=(thisAx.get_legend() for thisAx in ax),
                    bbox_inches='tight')
                ax[0].set_title(
                    'Block{:0>3}_TrialSeg{:0>1}_accel'.format(
                        blockIdx, sessionIdx)
                    )
                pdfPath = os.path.join(
                    figureOutputFolder,
                    'packet_jitter_{:0>3}_TrialSeg{:0>1}_accel.pdf'.format(
                        blockIdx, sessionIdx))
                plt.savefig(pdfPath, **figSaveOpts)
                pdfPath = os.path.join(
                    figureOutputFolder,
                    'meta_info_{:0>3}_TrialSeg{:0>1}_accel.pdf'.format(
                        blockIdx, sessionIdx))
                metaFig.savefig(pdfPath, **figSaveOpts)
                if showPlots:
                    plt.show()
                else:
                    plt.close()
                if showPlots:
                    plt.show()
                else:
                    plt.close()
            accelDataValues = rcsa_helpers.unpacker_accel(
                accelMetaDF.to_numpy(), accelJson, intersampleTickCount)
            #  save the noninterpolated files to disk
            accelData = rcsa_helpers.save_to_disk(
                accelDataValues, os.path.join(
                    jsonPath, 'RawDataAccel.csv'),
                time_format='full', data_type='accel')
            if tossPackets:
                tossMaskFinal = accelData['packetIdx'].isin(
                    np.flatnonzero(tossMask))
                accelData = accelData.loc[~tossMaskFinal, :]
            #
            # accelData['t'] = (
            #     accelData['actual_time'] - INSReferenceTime) / (
            #         datetime.timedelta(seconds=1))
            accelData['t'] = accelData['microseconds'] / pd.Timedelta('1S')
            accelData.drop(columns=['microseconds'], inplace=True)
            if sessionIdx == 0:
                tZero = accelData['t'].iloc[0]
            if not assumeNoSamplesLost:
                accelData = accelData.drop_duplicates(
                    ['t']
                    ).sort_values('t').reset_index(drop=True)
            else:
                accelData['equidistantT'] = np.nan
                lastPacketStartTime = accelData['t'].iloc[0]
                lastPacketT = lastPacketStartTime - fs ** (-1)
                # lastPacketStartCoarseTime = accelData['coarseClock'].iloc[0]
                for name, group in accelData.groupby('packetIdx'):
                    # metaRowIdx = accelMetaDF.loc[accelMetaDF['packetIdx'] == name].index
                    # assert metaRowIdx.size == 1
                    # metaRowIdx = accelMetaDF.loc[accelMetaDF['packetIdx'] == name].index[0]
                    misalignTimeDiff = group['t'].iloc[0] - lastPacketT
                    startTimeOver = accelMetaDF.loc[name, 'anyloss']
                    if verbose and (name in packetsNeededFixing):
                        print('    fixed packet: t = {:+.4f}'.format(
                            group['t'].iloc[0] - tZero))
                    if startTimeOver:
                        if verbose:
                            print('    starting over; t = {:+.4f}; time skip = {:+.4f}'.format(
                                group['t'].iloc[0] - tZero, misalignTimeDiff))
                        if assumeConsistentPacketSizes:
                            # if using processMetaMatrix
                            # packetTimeCorrection = (
                            #     accelMetaDF.loc[
                            #         name,
                            #         'frameLenMissingPacketCorrection'] +
                            #     nominalFrameDur *
                            #     np.round(misalignTimeDiff / nominalFrameDur))
                            # elseif using processMetaMatrixV2
                            packetTimeCorrection = (
                                accelMetaDF.loc[
                                    name,
                                    'packetTimeCorrection'])
                            if verbose:
                                print(
                                    '    applying {:+.4f} corection'
                                    .format(packetTimeCorrection))
                            lastPacketStartTime = (
                                lastPacketStartTime + packetTimeCorrection)
                        else:
                            lastPacketStartTime = group['t'].iloc[0]
                    else:
                        if verbose and False:
                            print('not starting over; t = {:+.4f}; time skip = {:+.4f}'.format(
                                group['t'].iloc[0] - tZero, misalignTimeDiff))
                    accelData.loc[group.index, 'equidistantT'] = (
                        lastPacketStartTime +
                        fs ** (-1) * np.arange(0, group.shape[0]))
                    lastPacketStartTime = (
                        accelData.loc[group.index, 'equidistantT'].iloc[-1] +
                        fs ** (-1))
                    lastPacketT = group['t'].iloc[-1] + fs ** (-1)
                    # lastPacketStartCoarseTime = (
                    #     accelData.loc[group.index, 'coarseClock'].iloc[-1])
                accelData['t'] = accelData['equidistantT'].to_numpy()
                accelData.drop('equidistantT', inplace=True, axis='columns')
            if getInterpolated:
                uniformT = np.arange(
                    accelData['t'].iloc[0],
                    accelData['t'].iloc[-1] + 1/fs,
                    1/fs)
                #  convert to floats before interpolating
                # accelData['microseconds'] = accelData['microseconds'] / (
                #     datetime.timedelta(microseconds=1))
                accelData['time_master'] = (
                    accelData['time_master'] - pd.Timestamp('2000-03-01')) / (
                    datetime.timedelta(seconds=1))
                HUTMapping = accelMetaDF['PacketGenTime']
                accelData['PacketGenTime'] = accelData['packetIdx'].map(HUTMapping)
                channelsPresent = [
                    i for i in accelData.columns if 'accel_' in i]
                channelsPresent += [
                    'time_master',
                    # 'microseconds',
                    'PacketGenTime']
                accelData = hf.interpolateDF(
                    accelData, uniformT, x='t',
                    columns=channelsPresent, fill_value=(0, 0))
                #  interpolating converts to floats, recover
                accelData['microseconds'] = pd.to_timedelta(
                    1e6 * accelData['t'], unit='us')
                accelData['time_master'] = pd.to_datetime(
                    accelData['time_master'], unit='s',
                    origin=pd.Timestamp('2000-03-01'))
            inertia = accelData['accel_x']**2 +\
                accelData['accel_y']**2 +\
                accelData['accel_z']**2
            inertia = inertia.apply(np.sqrt)
            accelData['inertia'] = inertia

            accelData['trialSegment'] = sessionIdx

            if getInterpolated:
                accelData.to_csv(
                    os.path.join(jsonPath, 'RawDataAccel_interpolated.csv'))
            else:
                accelData.to_csv(
                    os.path.join(jsonPath, 'RawDataAccel.csv'))

        accelSessions.append(accelData)
    '''
    accel = {
        'data': pd.concat(accelSessions, ignore_index=True),
        't': None
        }
    accel['t'] = accelDF['t']
    accel['INSTime'] = accelDF['t']
    accelDF['INSTime'] = accel['t']
    '''
    accelDF = pd.concat(accelSessions, ignore_index=True)
    accelDF.loc[:, 'INSTime'] = accelDF['t']
    return accelDF


def realignINSTimestamps(
        dataStruct, trialSegment, alignmentFactor
        ):
    if isinstance(dataStruct, pd.DataFrame):
        segmentMask = (dataStruct['trialSegment'] == trialSegment)
        dataStruct.loc[segmentMask, 't'] = (
            dataStruct.loc[segmentMask, 'microseconds'] + alignmentFactor) / (
                pd.Timedelta(1, unit='s'))
        if 'INSTime' in dataStruct.columns:
            dataStruct.loc[
                segmentMask, 'INSTime'] = dataStruct.loc[segmentMask, 't']
    # elif isinstance(dataStruct, dict):
    #     segmentMask = dataStruct['data']['trialSegment'] == trialSegment
    #     dataStruct['data'].loc[segmentMask, 't'] = (
    #         dataStruct['data'].loc[
    #             segmentMask, 'microseconds'] + alignmentFactor) / (
    #                 pd.Timedelta(1, unit='s'))
    #     if 'INSTime' in dataStruct['data'].columns:
    #         dataStruct['data'].loc[
    #             segmentMask, 'INSTime'] = dataStruct[
    #                 'data'].loc[segmentMask, 't']
    #     #  ['INSTime'] is a reference to ['t']  for the dict
    #     dataStruct['t'].loc[segmentMask] = dataStruct['data'].loc[segmentMask, 't']
    #     if 'INSTime' in dataStruct.keys():
    #         dataStruct['INSTime'].loc[segmentMask] = dataStruct['t'].loc[segmentMask]
    return dataStruct


def plotHUTtoINS(
        tdDF, accelDF, plotStimStatus,
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

    plotMaskTD = (tdDF[tdX] > tStartTD) & (tdDF[tdX] < tStopTD)
    plotMaskAccel = (accelDF[tdX] > tStartTD) & (accelDF[tdX] < tStopTD)
    plotMaskStim = (plotStimStatus[stimX] * unitsCorrection > tStartStim) &\
        (plotStimStatus[stimX] * unitsCorrection < tStopStim)

    if plotEachPacket:
        tdIterator = tdDF.loc[plotMaskTD, :].groupby('packetIdx')
    else:
        tdIterator = enumerate([tdDF.loc[plotMaskTD, :]])

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
        accelIterator = accelDF.loc[plotMaskAccel, :].groupby('packetIdx')
    else:
        accelIterator = enumerate([accelDF.loc[plotMaskAccel, :]])

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
    return fig, ax


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
    # d = np.sqrt(
    #     (xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)
    d = np.abs(xdata - mouseevent.xdata)
    dmask = (d < maxd) & (d == min(d))
    ind, = np.nonzero(dmask)
    if len(ind):
        # print('ind = {}'.format(ind))
        pickx = xdata[ind]
        picky = ydata[ind]
        props = dict(ind=ind, pickx=pickx, picky=picky)
        return True, props
    else:
        return False, dict()


def peekAtTaps(
        tdDF, accelDF, tapDetectSignal,
        channelData, trialSegment,
        tapDetectOpts, sessionTapRangesNSP,
        nspChannelName='ainp7',
        onlyPlotDetectChan=True,
        insX='t', plotBlocking=True,
        allTapTimestampsINS=None,
        allTapTimestampsNSP=None,
        interTrigInterval=40e-3):
    sns.set_style('darkgrid')
    tempClick = {
        'ins': [],
        'nsp': []
        }

    def onpick(event):
        if 'INS' in event.artist.axes.get_title():
            tempClick['ins'].append(event.pickx[0])
            print('Clicked on ins {:.3f}'.format(event.pickx[0]))
        elif 'NSP' in event.artist.axes.get_title():
            tempClick['nsp'].append(event.pickx[0])
            print('Clicked on nsp {:.3f}'.format(event.pickx[0]))
        event.artist.axes.text(
            event.pickx[0], event.picky[0],
            '. {:.3f}'.format(event.pickx[0]),
            ha='left', va='baseline')
        event.artist.get_figure().canvas.draw_idle()

    #  NSP plotting Bounds
    tStartNSP = (
        sessionTapRangesNSP[trialSegment]['timeRanges'][0])
    tStopNSP = (
        sessionTapRangesNSP[trialSegment]['timeRanges'][1])
    tDiffNSP = tStopNSP - tStartNSP
    #  Set up figures
    fig = plt.figure(tight_layout=True)
    ax = [None for i in range(3)]
    ax[0] = fig.add_subplot(311)
    ax[1] = fig.add_subplot(312, sharex=ax[0])
    if insX == 't':
        # INS plotting bounds
        tStartINS = tdDF['t'].iloc[0]
        tStopINS = tdDF['t'].iloc[-1]
        for thisRange in tapDetectOpts[
                trialSegment]['timeRanges']:
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
    insDataAx = ax[0]
    insTapsAx = ax[1]
    twinAx = ax[2].twiny()
    # dataColMask = np.array(['ins_td' in i for i in tdDF.columns])
    # if 'accChan' in tapDetectOpts[trialIdx][trialSegment].keys():
    #     detectOn = 'acc'
    #     accAx = ax[1]
    #     tdAx = ax[0]
    #     accAx.get_shared_x_axes().join(accAx, twinAx)
    #     accAx.get_shared_y_axes().join(accAx, twinAx)
    #     accAxLineStyle = '.-'
    #     tdAxLineStyle = '-'
    #     if onlyPlotDetectChan:
    #         accColumnNames = [tapDetectOpts[trialIdx][trialSegment]['accChan']]
    #     else:
    #         accColumnNames = [
    #             'ins_accx', 'ins_accy',
    #             'ins_accz', 'ins_accinertia']
    #     tdColumnNames = tdDF.columns[dataColMask]
    # else:
    #     detectOn = 'td'
    #     accAx = ax[0]
    #     tdAx = ax[1]
    #     tdAx.get_shared_x_axes().join(tdAx, twinAx)
    #     tdAx.get_shared_y_axes().join(tdAx, twinAx)
    #     accAxLineStyle = '-'
    #     tdAxLineStyle = '.-'
    #     if onlyPlotDetectChan:
    #         tdColumnNames = [tapDetectOpts[trialIdx][trialSegment]['tdChan']]
    #     else:
    #         tdColumnNames = tdDF.columns[dataColMask]
    #     accColumnNames = [
    #         'ins_accx', 'ins_accy',
    #         'ins_accz', 'ins_accinertia']
    # 
    insTapsAx.get_shared_x_axes().join(insTapsAx, twinAx)
    insTapsAx.get_shared_y_axes().join(insTapsAx, twinAx)
    plotMask = tdDF.index.isin(tapDetectSignal.index)
    for columnName in accel.columns:
        if 'acc' in columnName:
            insDataAx.plot(
                accel.loc[plotMask, insX],
                stats.zscore(accel.loc[plotMask, columnName]),
                label=columnName)
    for columnName in tdDF.columns:
        if 'td' in columnName:
            insDataAx.plot(
                tdDF.loc[plotMask, insX],
                stats.zscore(tdDF.loc[plotMask, columnName]),
                label=columnName)
    insDataAx.set_title('INS Segment {}'.format(trialSegment))
    insDataAx.set_ylabel('Z Score (a.u.)')
    insTapsAx.plot(
        tdDF.loc[plotMask, insX],
        stats.zscore(tapDetectSignal),
        '.-', label='tap detect signal',
        picker=line_picker)
    insTapsAx.set_title('INS Segment {}'.format(trialSegment))
    insTapsAx.set_ylabel('Z Score (a.u.)')
    twinAx.plot(
        tdDF.loc[plotMask, insX],
        stats.zscore(tapDetectSignal),
        label='tap detect signal')

    if allTapTimestampsINS is not None:
        insTapsAx.plot(
            allTapTimestampsINS[trialSegment],
            allTapTimestampsINS[trialSegment] ** 0 - 1,
            'c*', label='tap peaks',
            # picker=line_picker
            )
    
    xmin, xmax = ax[1].get_xlim()
    xTicks = np.arange(xmin, xmax, interTrigInterval)
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
    insDataAx.legend()

    tNSPMask = (channelData['t'] > tStartNSP) & (channelData['t'] < tStopNSP)
    triggerTrace = channelData['data'].loc[tNSPMask, nspChannelName]
    decimateFactor = 1
    ax[2].plot(
        channelData['t'].loc[tNSPMask].iloc[::decimateFactor],
        stats.zscore(triggerTrace.iloc[::decimateFactor]),
        'c', label='Analog Sync',
        # picker=line_picker
        )
    xmin, xmax = ax[2].get_xlim()
    xTicks2 = np.arange(xmin, xmax, interTrigInterval)
    ax[2].set_xticks(xTicks2)
    ax[2].set_yticks([])
    ax[2].set_xticklabels([])
    ax[2].grid(which='major', color='c', alpha=0.75)
    if allTapTimestampsNSP is not None:
        ax[2].plot(
            allTapTimestampsNSP[trialSegment],
            allTapTimestampsNSP[trialSegment] ** 0 - 1,
            'm*', label='tap peaks',
            # picker=line_picker
            )
    ax[2].legend()
    ax[2].set_title('NSP Data')
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block=plotBlocking)
    # remove double clicks
    for key, value in tempClick.items():
        tempVal = pd.Series(value)
        tempClick[key] = tempVal.loc[tempVal.diff().fillna(1) > 10e-3]
    return tempClick


def peekAtTapsV2(
        nspDF, insDF, insAuxDataDF=None,
        plotMaskNSP=slice(None), plotMaskINS=slice(None),
        tapTimestampsINS=None, tapTimestampsNSP=None,
        tapDetectOptsNSP=None, tapDetectOptsINS=None,
        procFunINS=None, procFunNSP=None,
        plotGuideLines=False,
        ):
    #
    if procFunINS is None:
        procFunINS = lambda x: x
    if procFunNSP is None:
        procFunNSP = lambda x: x

    tempClick = {
        'ins': [],
        'nsp': []
        }

    def onpick(event):
        if 'INS' in event.artist.axes.get_title():
            tempClick['ins'].append(event.pickx[0])
            print(tempClick)
            print('Clicked on ins {:.3f}'.format(event.pickx[0]))
        elif 'NSP' in event.artist.axes.get_title():
            tempClick['nsp'].append(event.pickx[0])
            print('Clicked on nsp {:.3f}'.format(event.pickx[0]))
        event.artist.axes.text(
            event.pickx[0], event.picky[0],
            '. {:.3f}'.format(event.pickx[0]),
            ha='left', va='baseline')
        event.artist.get_figure().canvas.draw_idle()

    #  Set up figures
    fig, ax = plt.subplots(3, 1)
    insDataAx = ax[0]
    insTapsAx = ax[1]
    nspTapsAx = ax[2]
    twinAx = nspTapsAx.twiny()
    insTapsAx.get_shared_x_axes().join(insTapsAx, insDataAx)
    insTapsAx.get_shared_x_axes().join(insTapsAx, twinAx)
    insTapsAx.get_shared_y_axes().join(insTapsAx, twinAx)
    #
    for cN in insAuxDataDF.columns:
        insDataAx.plot(
            insDF.loc[plotMaskINS, 't'].to_numpy(),
            procFunINS(insAuxDataDF.loc[plotMaskINS, cN].to_numpy()),
            label=cN)
    insDataAx.set_title('INS data')
    insDataAx.set_ylabel('Z Score (a.u.)')
    insTapsAx.plot(
        insDF.loc[plotMaskINS, 't'].to_numpy(),
        procFunINS(insDF.loc[plotMaskINS, 'tapDetectSignal'].to_numpy()),
        c='b', label='tap detect signal',
        picker=line_picker
        )
    insTapsAx.set_title('INS tap detect signal')
    insTapsAx.set_ylabel('Z Score (a.u.)')
    twinAx.plot(
        insDF.loc[plotMaskINS, 't'].to_numpy(),
        procFunINS(insDF.loc[plotMaskINS, 'tapDetectSignal'].to_numpy()),
        label='tap detect signal')
    if tapTimestampsINS is not None:
        if tapTimestampsINS.size > 0:
            insTapsAx.plot(
                tapTimestampsINS,
                tapTimestampsINS ** 0 - 1,
                'c*', label='tap peaks')
    #
    nspTapsAx.plot(
        nspDF.loc[plotMaskNSP, 't'],
        procFunNSP(nspDF.loc[plotMaskNSP, 'tapDetectSignal'].to_numpy()),
        'c', label='NSP tap detect signal',
        # picker=line_picker
        )
    if tapTimestampsNSP is not None:
        nspTapsAx.plot(
            tapTimestampsNSP,
            tapTimestampsNSP ** 0 - 1,
            'm*', label='tap peaks')
    #
    xmin, xmax = insDF.loc[plotMaskINS, 't'].min(), insDF.loc[plotMaskINS, 't'].max()
    insDataAx.set_xlim(xmin, xmax)
    if plotGuideLines:
        xTicks = np.arange(xmin, xmax, tapDetectOptsINS['iti'])
        for linePos in xTicks:
            insDataAx.axvline(x=linePos, alpha=0.75)
            insTapsAx.axvline(x=linePos, alpha=0.75)
            insTapsAx.axvline(x=linePos, alpha=0.75)
    #
    xmin, xmax = nspDF.loc[plotMaskNSP, 't'].min(), nspDF.loc[plotMaskNSP, 't'].max()
    nspTapsAx.set_xlim(xmin, xmax)
    if plotGuideLines:
        xTicks2 = np.arange(xmin, xmax, tapDetectOptsINS['iti'])
        for linePos in xTicks2:
            nspTapsAx.axvline(x=linePos, alpha=0.75, c='c')
    nspTapsAx.legend()
    nspTapsAx.set_title('NSP Data')
    #
    fig.canvas.mpl_connect('pick_event', onpick)
    # remove double detections
    # for key, value in tempClick.items():
    #     tempVal = pd.Series(value)
    #     tempClick[key] = tempVal.loc[tempVal.diff().fillna(1) > 10e-3]
    return tempClick, fig, ax


def getINSTapTimestamp(
        tdDF=None, accelDF=None,
        tapDetectOpts={}, filterOpts=None,
        plotting=False
        ):
    #
    if 'timeRanges' in tapDetectOpts.keys():
        timeRanges = tapDetectOpts['timeRanges']
    else:
        timeRanges = None
    #
    if 'keepIndex' in tapDetectOpts.keys():
        keepIndex = tapDetectOpts['keepIndex']
    else:
        keepIndex = slice(None)
    #
    if 'iti' in tapDetectOpts.keys():
        iti = tapDetectOpts['iti']
    else:
        iti = 0.25
    #
    if timeRanges is None:
        tdMask = tdDF['t'] > 0
    else:
        for idx, timeSegment in enumerate(timeRanges):
            if idx == 0:
                tdMask = (tdDF['t'] > timeSegment[0]) & (
                    tdDF['t'] < timeSegment[1])
            else:
                tdMask = tdMask | (
                    (tdDF['t'] > timeSegment[0]) & (
                        tdDF['t'] < timeSegment[1]))
    #
    signalsToConcat = []
    if tdDF is not None:
        tdChans = [cn for cn in tapDetectOpts['chan'] if 'td' in cn]
        signalsToConcat.append(tdDF.loc[tdMask, tdChans])
        fs = (tdDF['t'].iloc[1] - tdDF['t'].iloc[0]) ** -1
    if accelDF is not None:
        accChans = [cn for cn in tapDetectOpts['chan'] if 'acc' in cn]
        signalsToConcat.append(accelDF.loc[tdMask, accChans])
        fs = (accelDF['t'].iloc[1] - accelDF['t'].iloc[0]) ** -1
    tapDetectSignal = pd.concat(signalsToConcat, axis='columns')
    if filterOpts is not None:
        filterCoeffs = hf.makeFilterCoeffsSOS(
            filterOpts, fs)
        tapDetectSignal.loc[:, :] = signal.sosfiltfilt(
            filterCoeffs,
            tapDetectSignal.to_numpy(), axis=0)
    try:
        cov = (
            MinCovDet(support_fraction=0.75)
            .fit(tapDetectSignal.to_numpy()))
    except Exception:
        traceback.print_exc()
        cov = (
            EmpiricalCovariance()
            .fit(tapDetectSignal.to_numpy()))
    tapDetectSignal = pd.Series(
        np.sqrt(cov.mahalanobis(tapDetectSignal.to_numpy())),
        index=tapDetectSignal.index)
    tdPeakIdx = hf.getTriggers(
        tapDetectSignal, iti=iti, fs=fs, thres=tapDetectOpts['thres'],
        edgeType='both', minAmp=None,
        expectedTime=None, keep_max=False, plotting=plotting)
    tdPeakIdx = tdPeakIdx[keepIndex]
    print('TD Timestamps \n{}'.format(tdDF['t'].loc[tdPeakIdx]))
    tapTimestamps = tdDF['t'].loc[tdPeakIdx]
    peakIdx = tdPeakIdx

    return tapTimestamps, peakIdx, tapDetectSignal


def getHUTtoINSSyncFun(
        timeSyncDF,
        degree=1, plotting=False,
        trialSegments=None,
        syncTo='HostUnixTime',
        chunkSize=None,
        ):
    # if trialSegments is None:
    #     trialSegments = pd.unique(timeSyncDF['trialSegment'])
    # timeInterpFunHUTtoINS = [[] for i in trialSegments]
    timeInterpFunHUTtoINS = {}  # will be indexed by timeChunkIdx
    timeSyncDF['timeChunksHUTSynch'] = 0
    currChunkIdx = 0
    for trialSegment, tsDataSegment in timeSyncDF.groupby('trialSegment'):
        if chunkSize is not None:
            nChunks = int(np.ceil(tsDataSegment.shape[0] / chunkSize))
            thisT = tsDataSegment['t'] - tsDataSegment['t'].min()
            for i in range(nChunks):
                if i < (nChunks - 1):
                    tMask = (thisT >= i * chunkSize) & (thisT < (i + 1) * chunkSize)
                else:
                    tMask = (thisT >= i * chunkSize)
                tsDataSegment.loc[tMask, 'timeChunksHUTSynch'] = currChunkIdx
                idxIntoFullTsData = tsDataSegment.loc[tMask, :].index
                timeSyncDF.loc[idxIntoFullTsData, 'timeChunksHUTSynch'] = currChunkIdx
                currChunkIdx += 1
        else:
            tsDataSegment.loc[:, 'timeChunksHUTSynch'] = 0
    # lastTimeChunk = tsDataSegment['timeChunksHUTSynch'].max()
    for timeChunkIdx, tsTimeChunk in timeSyncDF.groupby('timeChunksHUTSynch'):
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
        # if (timeChunk < lastTimeChunk):
        #     tStart = tsTimeChunk['t'].min()
        #     tStartHUT = tsTimeChunk[syncTo].min()
        #     tStop = tsDataSegment.loc[tsDataSegment['timeChunksHUTSynch'] == (timeChunk + 1), 't'].min()
        #     tStopHUT = tsDataSegment.loc[tsDataSegment['timeChunksHUTSynch'] == (timeChunk + 1), syncTo].min()
        # elif (timeChunk == 0) and (lastTimeChunk == 0):
        #     tStart = 0
        #     tStartHUT = 0
        #     tStop = tsTimeChunk['t'].max()
        #     tStopHUT = tsTimeChunk[syncTo].max()
        # elif (timeChunk == 0):
        #     tStart = 0
        #     tStartHUT = 0
        #     tStop = tsDataSegment.loc[tsDataSegment['timeChunksHUTSynch'] == (timeChunk + 1), 't'].min()
        #     tStopHUT = tsDataSegment.loc[tsDataSegment['timeChunksHUTSynch'] == (timeChunk + 1), syncTo].min()
        # else:
        #     tStart = tsTimeChunk['t'].min()
        #     tStartHUT = tsTimeChunk[syncTo].min()
        #     tStop = tsTimeChunk['t'].max()
        #     tStopHUT = tsTimeChunk[syncTo].max()
        thisInterpDict = {
            'fun': thisFun,
            # 'tStart': tStart,
            # 'tStop': tStop,
            # 'tStartHUT': tStartHUT,
            # 'tStopHUT': tStopHUT
            }
        ###########################################################
        timeInterpFunHUTtoINS[timeChunkIdx] = thisInterpDict
        ###########################################################
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
        insDF, timeSyncDF, interpFunDict,
        syncTo='HostUnixTime'
        ):
    #
    HUTChunks = hf.interpolateDF(
        timeSyncDF, insDF['HostUnixTime'],
        kind='previous', x='HostUnixTime',
        columns=['timeChunksHUTSynch']
        )
    insDF.loc[:, 'timeChunksHUTSynch'] = HUTChunks['timeChunksHUTSynch']
    if 'INSTime' not in insDF.columns:
        insDF.loc[:, 'INSTime'] = np.nan
    for timeChunkIdx, group in insDF.groupby('timeChunksHUTSynch'):
        interpFun = interpFunDict[timeChunkIdx]['fun']
        insDF.loc[group.index, 'INSTime'] = interpFun(
            insDF.loc[group.index, syncTo])
    insDF.drop(columns=['timeChunksHUTSynch'], inplace=True)
    return insDF


def getINSStimLogFromJson(
        folderPath, sessionNames,
        deviceName='DeviceNPC700373H',
        absoluteStartTime=None, logForm='serial'):
    allStimStatus = []
    for sessionIdx, sessionName in enumerate(sessionNames):
        jsonPath = os.path.join(folderPath, sessionName, deviceName)
        print('getINSStimLogFromJson(: Loading {}\n...'.format(jsonPath))
        try:
            with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
                stimLog = json.load(f)
        except Exception:
            try:
                with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
                    stimLogText = f.read()
                    stimLogText = fixMalformedJson(stimLogText)
                    stimLog = json.loads(stimLogText)
            except Exception:
                with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
                    stimLogText = f.read()
                    stimLogText = fixMalformedJson(stimLogText)
                    stimLog = json.loads(stimLogText)
        if logForm == 'serial':
            stimStatus = rcsa_helpers.extract_stim_meta_data_events(
                stimLog, trialSegment=sessionIdx)
        allStimStatus.append(stimStatus)
    allStimStatusDF = pd.concat(allStimStatus, ignore_index=True)
    
    return allStimStatusDF


def stimStatusSerialtoLong(
        stimStSer, idxT='t', namePrefix='ins_', expandCols=[],
        deriveCols=[], progAmpNames=[], dropDuplicates=True,
        amplitudeCatBins=4,
        dummyTherapySwitches=False, elecConfiguration=None):
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
    stimStLong.loc[:, idxT] = stimStSer[idxT]
    for pName in fullExpandCols:
        #  print(pName)
        stimStLong.loc[:, pName] = np.nan
        pMask = stimStSer[namePrefix + 'property'] == pName
        pValues = stimStSer.loc[pMask, namePrefix + 'value']
        stimStLong.loc[pMask, pName] = pValues
        if pName == 'movement':
            stimStLong.loc[:, pName].iloc[0] = 0
        stimStLong.loc[:, pName] = stimStLong[pName].fillna(
            method='ffill')
        stimStLong.loc[:, pName] = stimStLong[pName].fillna(
            method='bfill')
    #
    debugPlot = False
    if debugPlot:
        stimCat = pd.concat((stimStLong, stimStSer), axis=1)
    #
    for idx, pName in enumerate(progAmpNames):
        stimStLong.loc[:, pName] = np.nan
        pMask = (stimStSer[namePrefix + 'property'] == 'amplitude') & (
            stimStLong['program'] == idx)
        stimStLong.loc[pMask, pName] = stimStSer.loc[pMask, namePrefix + 'value']
        stimStLong.loc[:, pName] = stimStLong[pName].fillna(method='ffill')
        stimStLong.loc[:, pName] = stimStLong[pName].fillna(value=0)
    if dropDuplicates:
        stimStLong.drop_duplicates(subset=idxT, keep='last', inplace=True)
    #
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        plt.show()
    stimStLong = stimStLong.reset_index(drop=True)
    ###################
    # scan for rate changes and add dummy therapy increments
    if dummyTherapySwitches:
        rateChange = (
            (stimStLong['RateInHz'].diff().fillna(0) != 0) |
            (stimStLong['trialSegment'].diff().fillna(0) != 0)
            )
        stimStLong.loc[:, 'rateRound'] = rateChange.astype(float).cumsum()
        groupComponents = [group.copy() for name, group in stimStLong.groupby('rateRound')]
        for idx, (name, group) in enumerate(stimStLong.groupby('rateRound')):
            if idx < len(groupComponents) - 1:
                nextT = groupComponents[idx+1][idxT].iloc[0]
                lastRate = groupComponents[idx]['RateInHz'].iloc[-1]
                dummyEntry = group.iloc[-1, :].copy()
                dummyEntry.loc[:, idxT] = nextT - lastRate ** (-1)
                dummyEntry.loc[:, 'therapyStatus'] = 0
                groupComponents[idx] = group.append(dummyEntry)
        stimStLong = pd.concat(groupComponents).reset_index(drop=True)
    ###################
    ampIncrease = pd.Series(False, index=stimStLong.index)
    ampChange = pd.Series(False, index=stimStLong.index)
    for idx, pName in enumerate(progAmpNames):
        ampIncrease = ampIncrease | (stimStLong[pName].diff().fillna(0) > 0)
        ampChange = ampChange | (stimStLong[pName].diff().fillna(0) != 0)
        if debugPlot:
            plt.plot(stimStLong[pName].diff().fillna(0), label=pName)
    #####
    ampChange = ampChange | (stimStLong['trialSegment'].diff().fillna(0) != 0)
    ####
    stimStLong.loc[:, 'amplitudeRound'] = (
        ampIncrease.astype(float).cumsum())
    stimStLong.loc[:, 'amplitude'] = (
        stimStLong[progAmpNames].sum(axis=1))
    ################
    if elecConfiguration is not None:
        groupComponents = [group.copy() for name, group in stimStLong.groupby('amplitudeRound')]
        for idx, (name, group) in enumerate(stimStLong.groupby('amplitudeRound')):
            thisProgram = int(group['program'].iloc[0])
            thisAmplitude = group['program{}_amplitude'.format(thisProgram)].iloc[0]
            if thisAmplitude == 0:
                continue
            thisElecConfig = elecConfiguration[0][thisProgram]
            if thisElecConfig['cyclingEnabled']:
                thisCycleOnTime = (
                    thisElecConfig['cycleOnTime']['time'] *
                    mdt_constants.cycleUnits[thisElecConfig['cycleOnTime']['units']] *
                    pq.s).magnitude
                thisCycleOffTime = (
                    thisElecConfig['cycleOffTime']['time'] *
                    mdt_constants.cycleUnits[thisElecConfig['cycleOffTime']['units']] *
                    pq.s).magnitude
                firstT = (group[idxT].iloc[0])
                if idx < len(groupComponents) - 1:
                    lastT = (groupComponents[idx+1][idxT].iloc[0])
                else:
                    lastT = (group[idxT].iloc[-1]) - thisCycleOnTime
                onTimes = [
                    i
                    for i in np.arange(
                        firstT, lastT,
                        # (thisCycleOnTime + thisCycleOffTime)
                        (thisCycleOnTime + thisCycleOffTime) + 10e-3
                        )
                    if i > firstT]
                offTimes = [
                    i
                    for i in np.arange(
                        firstT + thisCycleOffTime,
                        lastT,
                        # (thisCycleOnTime + thisCycleOffTime)
                        (thisCycleOnTime + thisCycleOffTime) + 10e-3
                        )]
                dummyEntriesOn = pd.DataFrame(
                    np.nan, index=range(len(onTimes)), columns=group.columns)
                dummyEntriesOff = pd.DataFrame(
                    np.nan, index=range(len(offTimes)), columns=group.columns)
                for pName in progAmpNames:
                    dummyEntriesOn[pName] = 0
                    dummyEntriesOff[pName] = 0
                dummyEntriesOn[idxT] = onTimes
                dummyEntriesOff[idxT] = offTimes
                dummyEntriesOn['program{}_amplitude'.format(thisProgram)] = thisAmplitude
                dummyEntriesOff['program{}_amplitude'.format(thisProgram)] = 0
                dummyEntriesOn['amplitude'] = thisAmplitude
                dummyEntriesOff['amplitude'] = 0
                substituteGroup = pd.concat(
                    [group, dummyEntriesOn, dummyEntriesOff])
                substituteGroup.sort_values(idxT, inplace=True, kind='mergesort')
                substituteGroup.fillna(method='ffill', inplace=True)
                substituteGroup.fillna(method='bfill', inplace=True)
                groupComponents[idx] = substituteGroup
        stimStLong = pd.concat(groupComponents).reset_index(drop=True)
        ampIncrease = pd.Series(False, index=stimStLong.index)
        ampChange = pd.Series(False, index=stimStLong.index)
        for idx, pName in enumerate(progAmpNames):
            ampIncrease = ampIncrease | (stimStLong[pName].diff().fillna(0) > 0)
            ampChange = ampChange | (stimStLong[pName].diff().fillna(0) != 0)
            if debugPlot:
                plt.plot(stimStLong[pName].diff().fillna(0), label=pName)
        #####
        ampChange = ampChange | (stimStLong['trialSegment'].diff().fillna(0) != 0)
        ####
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        ampIncrease.astype(float).plot(style='ro')
        ampChange.astype(float).plot(style='go')
        plt.legend()
        plt.show()
    #
    stimStLong.loc[:, 'amplitudeRound'] = (
        ampIncrease.astype(float).cumsum())
    stimStLong.loc[:, 'amplitude'] = (
        stimStLong[progAmpNames].sum(axis=1))
    ###############
    if 'movementRound' in deriveCols:
        stimStLong.loc[:, 'movementRound'] = (
            stimStLong.loc[:, 'movement'].abs().cumsum())
    # if 'amplitude' in deriveCols:
    if 'amplitudeCat' in deriveCols:
        ampsForSum = copy(stimStLong[progAmpNames])
        for colName in ampsForSum.columns:
            if ampsForSum[colName].max() > 0:
                ampsForSum.loc[:, colName] = pd.cut(
                    ampsForSum[colName], bins=amplitudeCatBins, labels=False)
            else:
                ampsForSum.loc[:, colName] = pd.cut(
                    ampsForSum[colName], bins=1, labels=False)
        stimStLong.loc[:, 'amplitudeCat'] = (
            ampsForSum.sum(axis=1))
    if debugPlot:
        stimStLong.loc[:, ['program'] + progAmpNames].plot()
        (10 * stimStLong['amplitudeRound'] / (stimStLong['amplitudeRound'].max())).plot()
        plt.show()
    return stimStLong


def getINSDeviceConfig(
        folderPath, sessionName, deviceName='DeviceNPC700373H'):
    jsonPath = os.path.join(folderPath, sessionName, deviceName)

    try:
        with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
            deviceSettings = json.load(f)
    except Exception:
        try:
            with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
                deviceSettingsText = f.read()
                deviceSettingsText += ']'
                deviceSettings = json.loads(deviceSettingsText)
        except Exception:
            with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
                deviceSettingsText = f.read()
                deviceSettingsText = fixMalformedJson(deviceSettingsText)
                deviceSettings = json.loads(deviceSettingsText)
    # try:
    #     with open(os.path.join(jsonPath, 'StimLog.json'), 'r') as f:
    #         stimLog = json.load(f)
    # except Exception:
    #     with open(os.path.join(jsonPath, 'DeviceSettings.json'), 'r') as f:
    #         stimLogText = f.read()
    #         stimLogText = fixMalformedJson(stimLogText)
    #         stimLog = json.loads(stimLogText)

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
        lambda x: mdt_constants.TdSampleRates[x])
    senseInfo['minusInput'] = senseInfo['minusInput'].apply(
        lambda x: mdt_constants.muxIdx[x])
    senseInfo['plusInput'] = senseInfo['plusInput'].apply(
        lambda x: mdt_constants.muxIdx[x])
    senseInfo.loc[(2, 3), ('minusInput', 'plusInput')] += 8
    senseInfo.loc[:, ('minusInput', 'plusInput')].fillna(17, inplace=True)
    senseInfo = senseInfo.loc[senseInfo['sampleRate'].notnull(), :]
    senseInfo.reset_index(inplace=True)
    senseInfo.rename(columns={'index': 'senseChan'}, inplace=True)
    senseInfo.streamingFrameRate = deviceSettings[0]['SensingConfig']['miscSensing']['streamingRate'] * 10 # in msec
    return electrodeConfiguration, senseInfo


def preprocINS(
        trialFilesStim,
        insDataFilename,
        deviceName='DeviceNPC700373H',
        verbose=False, blockIdx=None,
        makePlots=False, showPlots=False,
        figureOutputFolder=None, plotBlocking=True):
    if verbose:
        print('Preprocessing...')
    jsonBaseFolder = trialFilesStim['folderPath']
    jsonSessionNames = trialFilesStim['jsonSessionNames']
    #
    stimStatusSerial = getINSStimLogFromJson(
        jsonBaseFolder, jsonSessionNames,
        deviceName=deviceName)
    #
    elecConfiguration, senseInfo = (
        getINSDeviceConfig(
            jsonBaseFolder, jsonSessionNames[0],
            deviceName=deviceName)
        )
    #############################################################
    #############################################################
    # maybe here is a good spot to intercept the stim configs and force them to be unique
    # change both stimStatusSerial and elecConfiguration
    sessionsToRename = [
        'Session1548343250517', 'Session1548430735314', 'Session1548611405556', 'Session1548869346906']
    if jsonSessionNames[0] in sessionsToRename:
        # hacky relabel group 0 as group 1 to avoind name overlap
        hackMask = (
            (stimStatusSerial['ins_property'] == 'activeGroup') &
            (stimStatusSerial['ins_value'] == 0)
            )
        stimStatusSerial.loc[hackMask, 'ins_value'] = 1
        # swap the config descriptions
        tempConfig = copy(elecConfiguration[1])
        elecConfiguration[1] = copy(elecConfiguration[0])
        elecConfiguration[0] = tempConfig
        #
    #############################################################
    #############################################################
    if 'upsampleRate' in trialFilesStim:
        upsampleRate = trialFilesStim['upsampleRate']
    else:
        upsampleRate = None
    if 'interpKind' in trialFilesStim:
        interpKind = trialFilesStim['interpKind']
    else:
        interpKind = 'linear'
    #############################################################
    try:
        accelDF = getINSAccelFromJson(
            jsonBaseFolder, jsonSessionNames,
            deviceName=deviceName,
            getInterpolated=True,
            forceRecalc=trialFilesStim['forceRecalc'],
            fixTimeShifts=True,
            verbose=verbose, blockIdx=blockIdx,
            makePlots=makePlots, showPlots=showPlots,
            figureOutputFolder=figureOutputFolder)
    except Exception:
        traceback.print_exc()
        accelDF = None
    ###########################################################
    # warnings.filterwarnings("error")
    timeSync = getINSTimeSyncFromJson(
        jsonBaseFolder, jsonSessionNames,
        deviceName=deviceName,
        makePlots=makePlots, showPlots=showPlots,
        figureOutputFolder=figureOutputFolder,
        blockIdx=blockIdx, forceRecalc=True)
    #############################################################
    tdDF = getINSTDFromJson(
        jsonBaseFolder, jsonSessionNames,
        deviceName=deviceName,
        frameDuration=senseInfo.streamingFrameRate,
        getInterpolated=True,
        verbose=verbose, blockIdx=blockIdx,
        fixTimeShifts=True,
        upsampleRate=upsampleRate, interpKind=interpKind,
        forceRecalc=trialFilesStim['forceRecalc'],
        makePlots=makePlots, showPlots=showPlots,
        figureOutputFolder=figureOutputFolder)
    renamer = {}
    tdDataCols = []
    for colName in tdDF.columns:
        if 'channel_' in colName:
            idx = int(colName[-1])
            updatedName = 'channel_{}'.format(senseInfo.loc[idx, 'senseChan'])
            tdDataCols.append(updatedName)
            renamer.update({colName: updatedName})
    tdDF.rename(columns=renamer, inplace=True)
    ##############################################################
    #  packets are aligned to INSReferenceTime, for convenience
    #  (otherwise the values in ['t'] would be huge)
    #  align them to the more reasonable minimum first timestamp across packets
    #  System Tick seconds before roll over
    rolloverSeconds = pd.to_timedelta(6.5535, unit='s')
    #
    for trialSegment in pd.unique(tdDF['trialSegment']):
        if accelDF is not None:
            accelSegmentMask = accelDF['trialSegment'] == trialSegment
            accelGroup = accelDF.loc[accelSegmentMask, :]
        tdSegmentMask = tdDF['trialSegment'] == trialSegment
        tdGroup = tdDF.loc[tdSegmentMask, :]
        timeSyncSegmentMask = timeSync['trialSegment'] == trialSegment
        timeSyncGroup = timeSync.loc[timeSyncSegmentMask, :]
        sessionStartTime = int(jsonSessionNames[trialSegment].split('Session')[-1])
        sessionMasterTime = datetime.datetime.utcfromtimestamp(sessionStartTime / 1000)
        streamInitTimestampsDict = {
            'td': tdGroup['time_master'].iloc[0],
            'timeSync': timeSyncGroup['time_master'].iloc[0],
            }
        if accelDF is not None:
            streamInitTimestampsDict['accel'] = accelGroup['time_master'].iloc[0]
        streamInitTimestamps = pd.Series(streamInitTimestampsDict)
        print('streamInitTimestamps\n{}'.format(streamInitTimestamps))
        streamInitHUTDict = {
            'td': datetime.datetime.utcfromtimestamp(tdGroup['PacketGenTime'].iloc[0] / 1e3),
            'timeSync': datetime.datetime.utcfromtimestamp(timeSyncGroup['PacketGenTime'].iloc[0] / 1e3),
            }
        if accelDF is not None:
            streamInitHUTDict['accel'] = datetime.datetime.utcfromtimestamp(accelGroup['PacketGenTime'].iloc[0] / 1e3)
        streamInitHUT = pd.Series(streamInitHUTDict) - sessionMasterTime
        print('streamInitHUT\n{}'.format(streamInitHUT))
        streamInitSysTicksDict = {
            'td': tdGroup['microseconds'].iloc[0],
            'timeSync': timeSyncGroup['microseconds'].iloc[0],
            }
        if accelDF is not None:
            streamInitSysTicksDict['accel'] = accelGroup['microseconds'].iloc[0]
        streamInitSysTicks = pd.Series(streamInitSysTicksDict)
        print('streamInitSysTicks\n{}'.format(streamInitSysTicks))
        rolloverCorrection = pd.Series(pd.Timedelta(seconds=0), index=streamInitSysTicks.index)
        if 'fractionForRollover' in trialFilesStim:
            fractionForRollover = trialFilesStim['fractionForRollover']
        else:
            fractionForRollover = 0.1
        containsRollover = (
                (streamInitSysTicks > rolloverSeconds * (1 - fractionForRollover)).any() &
                (streamInitSysTicks < rolloverSeconds * fractionForRollover).any())
        if containsRollover:
            rolloverCorrection.loc[(streamInitSysTicks < rolloverSeconds / 3)] = rolloverSeconds
        print('rolloverCorrection\n{}'.format(rolloverCorrection))
        firstStream = streamInitHUT.idxmin()
        if trialSegment == 0:
            absoluteRef = sessionMasterTime
            alignmentFactor = streamInitHUT[firstStream] - streamInitSysTicks[firstStream] + rolloverCorrection
        else:
            alignmentFactor = sessionMasterTime - absoluteRef + streamInitHUT[firstStream] - streamInitSysTicks[firstStream] + rolloverCorrection
        #  correct any roll-over
        #  alignmentFactor[rolledOver] += rolloverSeconds
        print('alignmentFactor\n\n{}'.format(alignmentFactor))
        print('alignment factor in seconds:')
        print([tS.total_seconds() for tS in alignmentFactor])
        print('\n')
        # warnings.filterwarnings("error")
        if accelDF is not None:
            accelDF = realignINSTimestamps(
                accelDF, trialSegment, alignmentFactor.loc['accel'])
        tdDF = realignINSTimestamps(
            tdDF, trialSegment, alignmentFactor.loc['td'])
        timeSync = realignINSTimestamps(
            timeSync, trialSegment, alignmentFactor.loc['timeSync'])
    #  Check timeSync
    progAmpNames = rcsa_helpers.progAmpNames
    expandCols = (
        ['RateInHz', 'therapyStatus', 'trialSegment'] +
        progAmpNames)
    deriveCols = ['amplitudeRound']
    HUTChunkSize = 25
    interpFunHUTtoINS = getHUTtoINSSyncFun(
        timeSync, degree=0,
        syncTo='PacketGenTime',
        chunkSize=HUTChunkSize)
    # # # #
    stimStatusSerial = synchronizeHUTtoINS(
        stimStatusSerial, timeSync, interpFunHUTtoINS,
        syncTo='HostUnixTime')
    # # # #
    block = insDataToBlock(
        tdDF, accelDF, stimStatusSerial,
        senseInfo, trialFilesStim,
        tdDataCols=tdDataCols)
    #  stim detection
    if trialFilesStim['detectStim']:
        block = getINSStimOnset(
            block, elecConfiguration, blockIdx=blockIdx,
            showPlots=showPlots, figureOutputFolder=figureOutputFolder,
            **trialFilesStim['getINSkwargs'])
        #  if we did stim detection, recalculate stimStatusSerial
        stimSpikes = block.filter(objects=SpikeTrain)
        stimSpikes = ns5.loadContainerArrayAnn(trainList=stimSpikes)
        stimSpikesDF = ns5.unitSpikeTrainArrayAnnToDF(stimSpikes)
        if stimSpikesDF.size > 0:
            if trialFilesStim['eventsFromFirstInTrain']:
                firstOfTrainMask = (stimSpikesDF['rankInTrain'] == 1).to_numpy()
                stimSpikesDF = stimSpikesDF.loc[firstOfTrainMask, :].reset_index()
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
                value_vars=[
                    'group', 'program', 'RateInHz',
                    'pulseWidth', 'amplitude'],
                var_name='ins_property', value_name='ins_value')
            onsetEvents.rename(columns={'t': 'INSTime'}, inplace=True)
            onsetEvents.loc[onsetEvents['ins_property'] == 'group', 'ins_property'] = 'activeGroup'
            # ampOnsets = onsetEvents.loc[onsetEvents['ins_property'] == 'amplitude', :]
            # stimSpikesDF.loc[:, ['t', 'endTime']]
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
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=newStimStatusSerial['INSTime'],
                compareTo=stimSpikesDF['t'])
            tSegAnnsDF = (
                stimSpikesDF
                .loc[closestIdx, 'trialSegment']
                .to_numpy())
            block.segments[0].events = newStimEvents
            for ev in newStimEvents:
                ev.array_annotations.update({
                    'trialSegment': tSegAnnsDF})
                ev.annotations.update({
                    'trialSegment': tSegAnnsDF,
                    'arrayAnnNames': ['trialSegment']})
                ev.segment = block.segments[0]
            stimStatusSerial = newStimStatusSerial
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
        name='seg0_ins_concatenatedEvents',
        times=statusLabels['INSTime'].to_numpy() * pq.s,
        labels=concatLabels
        )
    block.segments[0].events.append(concatEvents)
    concatEvents.segment = block.segments[0]
    createRelationship = False
    if createRelationship:
        block.create_relationship()
    '''
    stimStatus = stimStatusSerialtoLong(
        stimStatusSerial, idxT='INSTime', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames,
        dummyTherapySwitches=False, elecConfiguration=elecConfiguration
        )
    '''
    # also can make changes to events here before they get written out. e.g. annotate with session info?
    writer = neo.io.NixIO(filename=insDataFilename, mode='ow')
    writer.write_block(block, use_obj_names=True)
    writer.close()
    return block


def getINSStimOnset(
        block, elecConfiguration,  blockIdx=None,
        cyclePeriod=0, minDist=0, minDur=0,
        gaussWid=600e-3,
        timeInterpFunINStoNSP=None,
        offsetFromPeak=None,
        overrideStartTimes=None,
        maxSlotsStimRate=100,
        cyclePeriodCorrection=0,  # 18e-3,
        stimDetectOptsByChannel=None,
        plotAnomalies=False, artifactKeepWhat='max',
        predictSlots=True, snapToGrid=False,
        expectRateProportionalStimOnDelay=False,
        expectRateProportionalStimOffDelay=False,
        redetectTherapyStatus=False,
        treatAsSinglePulses=False,
        spikeWindow=[-32, 64],
        plotting=[], showPlots=False, figureOutputFolder=None):
    if figureOutputFolder is not None:
        if redetectTherapyStatus:
            therapyDetectionPDF = PdfPages(
                os.path.join(
                    figureOutputFolder,
                    'therapy_detection_{:0>3}.pdf'.format(blockIdx)))
        pulseDetectionPDF = PdfPages(
            os.path.join(
                figureOutputFolder,
                'pulses_detection_{:0>3}.pdf'.format(blockIdx)))
        pulseConfirmationPDF = PdfPages(
            os.path.join(
                figureOutputFolder,
                'pulses_confirmation_{:0>3}.pdf'.format(blockIdx)))
    segIdx = 0
    seg = block.segments[segIdx]
    fs = seg.analogsignals[0].sampling_rate
    # 
    # WIP: treat as single at the stimStatus level
    tdDF, accelDF, stimStatus = unpackINSBlock(
        block, convertStimToSinglePulses=False,
        dummyTherapySwitches=False, elecConfiguration=elecConfiguration)
    #  assume a fixed delay between onset and stim
    #  fixedDelayIdx = int(fixedDelay * fs)
    #  print('Using a fixed delay of {} samples'.format(fixedDelayIdx))
    defaultOptsDict = {
        'detectChannels': [i for i in tdDF.columns if 'ins_td' in i],
        'useForSlotDetection': True}
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
            thisUnit = Unit(name=electrodeCombo + '#0')
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
    if redetectTherapyStatus:
        # calculate therapy status starts and ends
        therapyDiff = tdDF['therapyStatus'].diff().fillna(0)
        therapyOnsetIdx = tdDF.index[therapyDiff == 1]
        therapyOnTimes = pd.DataFrame({
            'nominalOnIdx': therapyOnsetIdx})
        try:
            therapyOnTimes.loc[:, 'on'] = np.nan
        except Exception:
            pdb.set_trace()
        therapyOnTimes.loc[:, 'onIdx'] = np.nan
        for idx, row in therapyOnTimes.iterrows():
            print('Calculating therapy on times for segment {}'.format(idx))
            # figure out what the group rate is at therapy on
            winStartT = tdDF.loc[row['nominalOnIdx'], 't'] - gaussWid
            winStopT = tdDF.loc[row['nominalOnIdx'], 't'] + gaussWid
            tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
            groupRate = tdDF.loc[tMask, 'RateInHz'].value_counts().idxmax()
            # double search window...
            winStartT = tdDF.loc[row['nominalOnIdx'], 't'] - 2 * gaussWid
            winStopT = tdDF.loc[row['nominalOnIdx'], 't'] + 2 * gaussWid
            tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
            # and add group rate
            winStopT += groupRate ** (-1)
            tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
            tdSeg = tdDF.loc[tMask, allDataCol + ['t']]
            tdSegDetect = tdSeg.loc[:, allDataCol + ['t']].set_index('t')
            try:
                detectSignal, foundTimestamp, _, fig, ax, twinAx = extractArtifactTimestampsMahalanobis(
                    tdSegDetect, fs,
                    gaussWid=gaussWid,
                    thresh=200,
                    offsetFromPeak=offsetFromPeak,
                    enhanceEdges=True,
                    plotDetection=len(plotting) > 0
                    )
            except Exception:
                foundTimestamp = [None]
            if len(plotting) > 0:
                ax.legend(loc='upper left')
                twinAx.legend(loc='upper right')
                figSaveOpts = dict(
                    bbox_extra_artists=(ax.get_legend(), twinAx.get_legend()),
                    bbox_inches='tight')
                therapyDetectionPDF.savefig(**figSaveOpts)
                if showPlots:
                    plt.show()
                else:
                    plt.close()
            if foundTimestamp[0] is not None:
                therapyOnTimes.loc[idx, 'on'] = foundTimestamp
                localIdx = [tdSegDetect.index.get_loc(i) for i in foundTimestamp]
                therapyOnTimes.loc[idx, 'onIdx'] = tMask.index[tMask][localIdx]
            else:
                therapyOnTimes.loc[idx, 'on'] = tdDF.loc[row['nominalOnIdx'], 't']
                therapyOnTimes.loc[idx, 'onIdx'] = row['nominalOnIdx']
        # trueTherapyDiff = pd.Series(0, index=tdDF.index)
        trueTherapyDiff = tdDF['trialSegment'].diff().fillna(0)
        # 
        trueTherapyDiff.loc[therapyOnTimes['onIdx']] = 1
        tdDF.loc[:, 'therapyRound'] = trueTherapyDiff.cumsum()
        # find offsets
        therapyOffsetIdx = tdDF.index[therapyDiff == -1]
        therapyOffTimes = pd.DataFrame({
            'nominalOffIdx': therapyOffsetIdx})
        if therapyOffTimes.size > 0:
            therapyOffTimes.loc[:, 'off'] = np.nan
            therapyOffTimes.loc[:, 'offIdx'] = np.nan
            for idx, row in therapyOffTimes.iterrows():
                print('Calculating therapy off times for segment {}'.format(idx))
                # figure out what the group rate is at therapy on
                winStartT = tdDF.loc[row['nominalOffIdx'], 't'] - gaussWid
                winStopT = tdDF.loc[row['nominalOffIdx'], 't'] + gaussWid
                tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
                groupRate = tdDF.loc[tMask, 'RateInHz'].value_counts().idxmax()
                # double the search area...
                winStartT = tdDF.loc[row['nominalOffIdx'], 't'] - 2 * gaussWid
                winStopT = tdDF.loc[row['nominalOffIdx'], 't'] + 2 * gaussWid
                tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
                # and expand search window by group rate
                winStopT += 1/groupRate
                tMask = hf.getTimeMaskFromRanges(tdDF['t'], [(winStartT, winStopT)])
                tdSeg = tdDF.loc[tMask, allDataCol + ['t']]
                tdSegDetect = tdSeg.loc[:, allDataCol + ['t']].set_index('t')
                try:
                    detectSignal, foundTimestamp, _, fig, ax, twinAx = extractArtifactTimestampsMahalanobis(
                        tdSegDetect, fs,
                        gaussWid=gaussWid,
                        thresh=100,
                        offsetFromPeak=offsetFromPeak,
                        enhanceEdges=True,
                        plotDetection=len(plotting) > 0,
                        threshMethod='peaks',
                        keepWhat='last'
                        )
                except Exception:
                    foundTimestamp = [None]
                if len(plotting) > 0:
                    # ax = plt.gca()
                    ax.legend(loc='upper left')
                    twinAx.legend(loc='upper right')
                    figSaveOpts = dict(
                        bbox_extra_artists=(ax.get_legend(), twinAx.get_legend()),
                        bbox_inches='tight')
                    therapyDetectionPDF.savefig(**figSaveOpts)
                    if showPlots:
                        plt.show()
                    else:
                        plt.close()
                if foundTimestamp[0] is not None:
                    therapyOffTimes.loc[idx, 'off'] = foundTimestamp
                    localIdx = [tdSegDetect.index.get_loc(i) for i in foundTimestamp]
                    therapyOffTimes.loc[idx, 'offIdx'] = tMask.index[tMask][localIdx]
                else:
                    therapyOffTimes.loc[idx, 'off'] = tdDF.loc[row['nominalOffIdx'], 't']
                    therapyOffTimes.loc[idx, 'offIdx'] = row['nominalOffIdx']   
            trueTherapyDiff.loc[therapyOffTimes['offIdx']] = -1
        tdDF.loc[:, 'therapyStatus'] = trueTherapyDiff.cumsum()
        if figureOutputFolder is not None:
            therapyDetectionPDF.close()
    else:
        therapyDiff = tdDF['therapyStatus'].diff().fillna(0)
        therapyDiff.loc[therapyDiff != 1] = 0
        tdDF.loc[:, 'therapyRound'] = therapyDiff.cumsum()
    lastTherapyRound = 0
    # lastTrialSegment = 0
    # calculate slots
    tdDF.loc[:, 'slot'] = np.nan
    plottingSlots = False
    resolvedSlots = False
    lastRate = np.nan
    arrayAnnListOfNames = [
        'amplitude', 'RateInHz', 'program',
        'group', 'pulseWidth', 'endTime',
        'trialSegment', 'offsetFromExpected',
        'offsetFromLogged','usedExpectedT',
        'usedSlotToDetect',
        'rankInTrain', 'trainNPulses', 'trialSegment']
    # warnings.filterwarnings("error")
    for name, group in tdDF.groupby('amplitudeRound'):
        anomalyOccured = False
        # check that this round is worth analyzing
        groupAmpMask = (
            (group['amplitude'] > 0) &
            (group['therapyStatus'] > 0))
        if not groupAmpMask.any():
            print('Amplitude round {} stim never turned on!'.format(name))
            continue
        stimRate = (
            group
            .loc[groupAmpMask, 'RateInHz']
            .value_counts()
            .idxmax())
        groupTotalDuration = group.loc[groupAmpMask, 't'].max() - group.loc[groupAmpMask, 't'].min()
        if groupTotalDuration < 360e-3:
            print('Amplitude round {} is shorter than {:.3f} sec. Ignoring...'.format(name, 360e-3))
            continue
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
        # use the HUT derived stim onset to favor detection
        # minROIWid = 150e-3 # StimLog timestamps only reliable within 50 msec
        gaussWidIdx = int(gaussWid * fs)
        nominalStimOnIdx = group.loc[groupAmpMask, 't'].index[0]
        # 
        nominalStimOnT = group.loc[groupAmpMask, 't'].iloc[0]
        #  pad with paddingDuration msec to ensure robust z-score
        # paddingDuration = max(
        #     np.around(2 * stimPeriod, decimals=4),
        #     100e-3)
        # tStartPadded = max(
        #     tdDF['t'].iloc[0],
        #     group.loc[groupAmpMask, 't'].iloc[0] - 2 * paddingDuration)
        # tStopPadded = min(
        #     tdDF['t'].iloc[-1],
        #     group.loc[groupAmpMask, 't'].iloc[-1] + 0.25 * paddingDuration)
        # plotMaskTD = (tdDF['t'] > tStartPadded) & (tdDF['t'] < tStopPadded)
        paddingDuration = max(
            np.around(3.5 * stimPeriod, decimals=4),
            6 * gaussWid)
        plotMaskTD = hf.getTimeMaskFromRanges(
            tdDF['t'],
            [(
                nominalStimOnT - paddingDuration / 2,
                nominalStimOnT + paddingDuration / 2
            )]).to_numpy()
        #  load the appropriate detection options
        theseDetectOpts = stimDetectOptsByChannel[activeGroup][activeProgram]
        #  calculate signal used for stim artifact detection
        therapyOnMaskTD = (tdDF['therapyRound'] == thisTherapyRound).to_numpy()
        therSegDF = tdDF.loc[therapyOnMaskTD, :]
        tdSeg = (tdDF.loc[
            plotMaskTD & therapyOnMaskTD,
            theseDetectOpts['detectChannels'] + ['t']
            ])
        lastAmplitudeSer = tdDF.loc[
            plotMaskTD & therapyOnMaskTD,
            ampColName]
        if not len(lastAmplitudeSer):
            continue
        else:
            lastAmplitude = lastAmplitudeSer.iloc[0]
        if (lastRate != stimRate):
            # recalculate every time we increment amplitude from zero
            # (these should usually be very visible)
            # also recalculate on rate changes
            # in order to mitigate uncertainty about when Rate changes
            # are implemented
            print('resetting slots!')
            resolvedSlots = False
        lastRate = stimRate
        useThresh = theseDetectOpts['thres']
        if resolvedSlots:
            # this ROI is used for the detection of stim time based on artifact
            ROIWid = 3 * gaussWid
            tdSeg.loc[:, 'slot'] = tdDF.loc[tdSeg.index, 'slot']
            slotDiff = tdSeg['slot'].diff()
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
            # possibleSlotStartIdx = tdSeg.index[slotStartMask]
            possibleSlotStartIdx = (
                tdSeg.index[slotStartMask] +
                activeProgram * int(slotSize / 4))
            possibleSlotStartIdx = possibleSlotStartIdx[possibleSlotStartIdx.isin(tdSeg.index)]
            print('possibleSlotStartIdx is {}\n'.format(possibleSlotStartIdx))
            if len(possibleSlotStartIdx) > 1:
                stimOnUncertainty = pd.Series(
                    stats.norm.cdf(tdSeg['t'], nominalStimOnT, (gaussWid / 2)),
                    index=tdSeg.index)
                possibleSlotStartMask = tdSeg.index.isin(possibleSlotStartIdx)
                uncertaintyValsDF = stimOnUncertainty[possibleSlotStartMask].diff()
                uncertaintyValsDF.iloc[0] = 0
                uncertaintyVals = uncertaintyValsDF.to_numpy()
                # keepMask = uncertaintyVals > .1
                keepMask = uncertaintyVals > uncertaintyValsDF.quantile(0.9)
                keepMask[np.argmax(uncertaintyVals)] = True
                try:
                    possibleSlotStartIdx = possibleSlotStartIdx[keepMask]
                except:
                    pdb.set_trace()
                uncertaintyVals = uncertaintyVals[keepMask]
            else:
                uncertaintyVals = np.array([1])
            '''
            possibleOnsetIdx = (
                possibleSlotStartIdx +
                activeProgram * int(slotSize / 4)
                )
            '''
            possibleOnsetIdx = possibleSlotStartIdx
            allPossibleTimestamps = tdSeg.loc[possibleOnsetIdx, 't']
            print('allPossibleTimestamps\n{}\n'.format(allPossibleTimestamps))
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
            if overrideStartTimes is not None:
                ovrTimes = pd.Series(overrideStartTimes)
                ovrMask = (ovrTimes >= tdSeg['t'].iloc[0]) & (ovrTimes < tdSeg['t'].iloc[-1])
                ovrTimes = ovrTimes[ovrMask]
                if ovrTimes.any():
                    ROIWid = 3 * stimPeriod / 8
                    print('Using override time; replacing {}'.format(expectedTimestamp))
                    expectedTimestamp = ovrTimes.iloc[0]
                    tStartOnset = expectedTimestamp - ROIWid / 2
                    tStopOnset = expectedTimestamp + ROIWid / 2
            print("Expected timestamp is {}".format(expectedTimestamp))
            print("ROI = {}".format((tStartOnset, tStopOnset)))
            ROIMaskOnset = (
                (tdSeg['t'] >= tStartOnset) &
                (tdSeg['t'] <= tStopOnset))
            #
            slotMatchesMask = tdDF.loc[tdSeg.index, 'slot'].shift(-int(np.round(slotSize/16))) == activeProgram
            # ROIMaskOnset = ROIMaskOnset & slotMatchesMask
        else:
            # if not resolved slots
            ROIWid = 3 * gaussWid + stimPeriod
            #  on average, the StimLog update will land in the middle of the previous group cycle
            #  adjust the ROIWid to account for this extra variability
            if expectRateProportionalStimOnDelay:
                expectedOnsetIdx = (
                    nominalStimOnIdx +
                    # int(slotSize)
                    int(slotSize / 2) + activeProgram * int(slotSize / 4)
                    )
            else:
                expectedOnsetIdx = nominalStimOnIdx
            if not expectedOnsetIdx in tdSeg.index:
                expectedOnsetIdx = nominalStimOnIdx
            expectedTimestamp = tdSeg.loc[expectedOnsetIdx, 't']
            if overrideStartTimes is not None:
                print('Checking for overrides between {} and {}'.format(
                    tdSeg['t'].iloc[0], tdSeg['t'].iloc[-1]
                ))
                ovrTimes = pd.Series(overrideStartTimes)
                ovrMask = (ovrTimes >= tdSeg['t'].iloc[0]) & (ovrTimes < tdSeg['t'].iloc[-1])
                ovrTimes = ovrTimes[ovrMask]
                if ovrTimes.any():
                    ROIWid = 3 * stimPeriod / 8
                    print('Using override time: replacing {}'.format(expectedTimestamp))
                    expectedTimestamp = ovrTimes.iloc[0]
            print('Expected timestamp is {}'.format(expectedTimestamp))
            ROIBasis = pd.Series(0, index=tdSeg['t'])
            basisMask = (
                (ROIBasis.index >= (expectedTimestamp - stimPeriod / 2)) &
                (ROIBasis.index <= (expectedTimestamp + stimPeriod / 2)))
            ROIBasis[basisMask] = 1
            ROIMaskOnset = hf.getTimeMaskFromRanges(
                tdSeg['t'], [(expectedTimestamp - ROIWid / 2, expectedTimestamp + ROIWid / 2)]
            )
            #  where to look for the onset
            # tStartOnset = max(
            #     tdSeg['t'].iloc[0],
            #     expectedTimestamp - ROIWid / 2)
            # tStopOnset = min(
            #     tdSeg['t'].iloc[-1],
            #     expectedTimestamp + ROIWid / 2)
            # print("ROI = {}".format((tStartOnset, tStopOnset)))
            # ROIMaskOnset = (
            #     (tdSeg['t'] >= tStartOnset) &
            #     (tdSeg['t'] <= tStopOnset))
        # TODO: document why it's necessary to make the edges true (interpolation?)
        ROIMaskOnset.iloc[0] = True
        ROIMaskOnset.iloc[-1] = True
        tdSegDetect = tdSeg.loc[:, theseDetectOpts['detectChannels'] + ['t']].set_index('t')
        if plotting is not None:
            if isinstance(plotting, str):
                if (plotting == 'first_slot') and not resolvedSlots:
                    plottingEnabled = True
                else:
                    plottingEnabled = False
            elif isinstance(plotting, Iterable):
                plottingEnabled = (name in plotting) # and (not resolvedSlots)
        # detect stim onset
        print('Slot resolution status is: {}'.format(resolvedSlots))
        detectSignal, foundTimestamp, usedExpectedT, fig, ax, twinAx = extractArtifactTimestampsMahalanobis(
            tdSegDetect,
            fs,
            gaussWid=gaussWid,
            thresh=useThresh,
            stimRate=stimRate,
            threshMethod='peaks',
            keepWhat=artifactKeepWhat,
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
        # if we haven't resolved the slots yet, and we want them for future detections:
        if 'useForSlotDetection' not in theseDetectOpts:
            theseDetectOpts['useForSlotDetection'] = True
        if (not resolvedSlots) and predictSlots and (theseDetectOpts['useForSlotDetection']) and (stimRate <= maxSlotsStimRate):
            # have not resolved phase between slots and recording for this therapy segment
            # resolving now:
            rateDiff = therSegDF['RateInHz'].diff().fillna(method='bfill')
            rateChangeTimes = therSegDF.loc[rateDiff != 0, 't']
            print('Calculating slots for segment {}'.format(thisTherapyRound))
            try:
                groupRate = therSegDF.loc[foundIdx, 'RateInHz'].iloc[0]
            except:
                traceback.print_exc()
            groupPeriod = float(groupRate ** (-1))
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
                try:
                    tdDF.loc[startIdx:endIdx, 'slot'] = calculatedSlots.astype(np.int)
                except Exception:
                    traceback.print_exc()
                timeSeekIdx = min(
                    endIdx + 1,
                    therSegDF.index[-1])
                groupRate = therSegDF.loc[timeSeekIdx, 'RateInHz']
                # oldGroupPeriod = groupPeriod
                groupPeriod = float(groupRate ** (-1))
                startTime = therSegDF.loc[timeSeekIdx, 't']
                # startTime = (
                #     therSegDF.loc[timeSeekIdx, 't'] +
                #     (oldGroupPeriod - groupPeriod) / 8)
            group.loc[:, 'slot'] = tdDF.loc[group.index, 'slot']
            tdSeg.loc[:, 'slot'] = tdDF.loc[tdSeg.index, 'slot']
            slotDiff = tdSeg['slot'].diff()
            resolvedSlots = True
        # done resolving slots
        if plottingEnabled:
            # tdSegSlotDiff = tdDF.loc[plotMaskTD, 'slot'].diff()
            if not resolvedSlots:
                tdSeg.loc[:, 'slot'] = np.nan
                slotDiff = tdSeg['slot'].copy().diff()
            # slotDiff = tdSeg['slot'].diff()
            slotEdges = (
                tdSeg
                .loc[slotDiff.fillna(1) != 0, 't']
                .reset_index(drop=True))
            theseSlots = (
                # tdDF.loc[plotMaskTD, :]
                tdSeg
                .loc[slotDiff.fillna(1) != 0, 'slot']
                .reset_index(drop=True))
            # ax = plt.gca()
            ax.axvline(expectedTimestamp, color='b', linestyle='--', label='expected time')
            ax.axvline(group['t'].iloc[0], color='r', label='stimLog time')
            ax.set_title(
                'Grp {} Prog {} slots: {} (Rate {:.1f} Hz, Amp {:.1f} uA)'.format(
                    activeGroup, activeProgram,
                    usedSlotToDetect, stimRate, 100 * thisAmplitude))
            cPal = sns.color_palette(n_colors=4)
            for idx, slEdge in slotEdges.iloc[1:].iteritems():
                try:
                    ax.axvspan(
                        slotEdges[idx-1], slEdge,
                        alpha=0.5, facecolor=cPal[int(theseSlots[idx-1])])
                except Exception:
                    continue
        if (snapToGrid and resolvedSlots):
            slotMatchesMask = (tdSeg['slot'] == activeProgram) & (slotDiff.fillna(1) != 0)
            slotMatchesTime = tdSeg.loc[slotMatchesMask, 't']
            try:
                timeMatchesIdx = (slotMatchesTime - foundTimestamp[0]).abs().idxmin()
                theseOnsetTimestamps = np.atleast_1d(tdSeg.loc[timeMatchesIdx, 't']) * pq.s
            except Exception:
                traceback.print_exc()
                theseOnsetTimestamps = np.atleast_1d(foundTimestamp[0]) * pq.s
        else:
            theseOnsetTimestamps = np.atleast_1d(foundTimestamp[0]) * pq.s
        if plottingEnabled:
            twinAx.plot(
                theseOnsetTimestamps,
                theseOnsetTimestamps ** 0 - 1,
                'g*', label='final timestamps: {:.4f}'.format(theseOnsetTimestamps[0]))
            ax.legend(loc='upper left')
            twinAx.legend(loc='upper right')
            figSaveOpts = dict(
                bbox_extra_artists=(ax.get_legend(), twinAx.get_legend()),
                bbox_inches='tight')
            pulseDetectionPDF.savefig(**figSaveOpts)
            if showPlots:
                plt.show()
            else:
                plt.close()
        print('Found timestamps:\n {}'.format(theseOnsetTimestamps))
        onsetDifferenceFromExpected = (
            np.atleast_1d(tdSeg.loc[expectedOnsetIdx, 't']) -
            np.atleast_1d(foundTimestamp)) * pq.s
        onsetDifferenceFromLogged = (
            np.atleast_1d(group['t'].iloc[0]) -
            np.atleast_1d(foundTimestamp)) * pq.s
        #
        if expectRateProportionalStimOffDelay:
            stimOffIdx = min(
                group.index[-1],
                (
                    group.index[groupAmpMask][-1] +
                    # int(slotSize)
                    int(slotSize/2) + activeProgram * int(slotSize/4)
                ))
        else:
            stimOffIdx = min(
                group.index[-1],
                (
                    group.index[groupAmpMask][-1]
                ))
        theseOffsetTimestamps = np.atleast_1d(
            group
            .loc[stimOffIdx, 't']
            ) * pq.s
        # elecConfiguration
        electrodeCombo = 'g{:d}p{:d}'.format(activeGroup, activeProgram)
        if len(theseOnsetTimestamps):
            thisUnit = block.filter(
                objects=Unit,
                name=electrodeCombo + '#0'
                )[0]
            thisElecConfig = elecConfiguration[activeGroup][activeProgram]
            #
            rankInTrain = (theseOnsetTimestamps ** 0).magnitude * pq.dimensionless
            trainNPulses = (theseOnsetTimestamps ** 0).magnitude * pq.dimensionless
            theseTrialSegments = (theseOnsetTimestamps ** 0).magnitude * thisTrialSegment * pq.dimensionless
            ##############################################################################
            if treatAsSinglePulses:
                tempOnTimes = []
                tempRankInTrain = []
                tempTrainNPulses = []
                tempTrialSegs = []
                tempOffTimes = []
                tempOnDiffsE = []
                tempOnDiffsL = []
                for idx, onTime in enumerate(theseOnsetTimestamps):
                    offTime = theseOffsetTimestamps[idx]
                    interPulseInterval = 1 / (stimRate * pq.Hz)
                    pulseOnTimes = np.arange(
                        onTime, offTime,
                        interPulseInterval) * onTime.units
                    #
                    tempRankInTrain.append(np.cumsum(pulseOnTimes ** 0))
                    tempTrainNPulses.append(pulseOnTimes ** 0 * pulseOnTimes.size)
                    tempTrialSegs.append(
                        pulseOnTimes ** 0 * theseTrialSegments[idx])
                    #
                    pulseOffTimes = pulseOnTimes + 100 * stimPW * pq.us
                    try:
                        pulseOffTimes[0] = offTime
                    except Exception:
                        traceback.print_exc()
                        pdb.set_trace()
                    tempOnTimes.append(pulseOnTimes)
                    tempOffTimes.append(pulseOffTimes)
                    onDiffE = onsetDifferenceFromExpected[idx]
                    tempOnDiffsE.append(pulseOnTimes ** 0 * onDiffE)
                    onDiffL = onsetDifferenceFromLogged[idx]
                    tempOnDiffsL.append(pulseOnTimes ** 0 * onDiffL)
                theseOnsetTimestamps = np.concatenate(tempOnTimes) * onTime.units
                theseOffsetTimestamps = np.concatenate(tempOffTimes) * offTime.units
                onsetDifferenceFromExpected = np.concatenate(tempOnDiffsE) * onTime.units
                onsetDifferenceFromLogged = np.concatenate(tempOnDiffsL) * offTime.units
                rankInTrain = np.concatenate(tempRankInTrain) * pq.dimensionless
                trainNPulses = np.concatenate(tempTrainNPulses) * pq.dimensionless
                theseTrialSegments = np.concatenate(tempTrialSegs) * pq.dimensionless
            #
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
                'amplitude': ampList,
                'RateInHz': rateList,
                'pulseWidth': pwList,
                'trialSegment': tSegList,
                'endTime': theseOffsetTimestamps,
                'program': programList,
                'group': groupList,
                'offsetFromExpected': onsetDifferenceFromExpected,
                'offsetFromLogged': onsetDifferenceFromLogged,
                'rankInTrain': rankInTrain,
                'trainNPulses': trainNPulses,
                'trialSegment': theseTrialSegments,
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
    if figureOutputFolder is not None:
        pulseDetectionPDF.close()
    left_sweep_samples = spikeWindow[0] * (-1)
    left_sweep = left_sweep_samples / fs
    right_sweep_samples = spikeWindow[1] - 1
    right_sweep = right_sweep_samples / fs
    if figureOutputFolder is not None:
        stashStimTimesDict = {}
    for thisUnit in block.filter(objects=Unit):
        print('getINSStimOnset packaging unit {}'.format(thisUnit.name))
        if len(tempSpiketrainStorage[thisUnit.name]) == 0:
            placeHolderSt = SpikeTrain(
                name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                times=[], units='sec', t_stop=spikeTStop,
                t_start=spikeTStart, sampling_rate=fs, left_sweep=left_sweep,
                waveforms=np.asarray([]).reshape((0, 0, 0)) * pq.mV)
            placeHolderSt.annotations['unitAnnotations'] = json.dumps(thisUnit.annotations.copy())
            placeHolderArrayAnn = {
                annNm: np.array([])
                for annNm in arrayAnnListOfNames
                }
            placeHolderSt.annotations['arrayAnnNames'] = arrayAnnListOfNames
            placeHolderSt.annotations.update(placeHolderArrayAnn)
            placeHolderSt.array_annotations = placeHolderArrayAnn
            thisUnit.spiketrains.append(placeHolderSt)
            seg.spiketrains.append(placeHolderSt)
            placeHolderSt.unit = thisUnit
            placeHolderSt.segment = seg
        else:
            #  consolidate spiketrains
            consolidatedTimes = np.array([])
            consolidatedAnn = {
                annNm: np.array([])
                for annNm in arrayAnnListOfNames
            }
            arrayAnnNames = {'arrayAnnNames': list(consolidatedAnn.keys())}
            for idx, st in enumerate(tempSpiketrainStorage[thisUnit.name]):
                consolidatedTimes = np.concatenate((
                    consolidatedTimes,
                    st.times.magnitude
                ))
                for key, value in consolidatedAnn.items():
                    consolidatedAnn[key] = np.concatenate((
                        consolidatedAnn[key], st.annotations[key]
                        ))
            unitDetectedOn = thisUnit.annotations['detectChannels']
            consolidatedTimes, timesIndex = hf.closestSeries(
                takeFrom=pd.Series(consolidatedTimes),
                compareTo=tdDF['t'])
            # check cycling interval
            print('median inter pulse interval: {}'.format(
                consolidatedTimes.diff().median()
                ))
            #
            timesIndex = np.array(timesIndex.values, dtype=np.int)
            #  spike_duration = left_sweep + right_sweep
            spikeWaveforms = np.zeros(
                (
                    timesIndex.shape[0], len(unitDetectedOn),
                    left_sweep_samples + right_sweep_samples + 1),
                dtype=float)
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
                sampling_rate=fs, t_start=spikeTStart,
                **consolidatedAnn, **arrayAnnNames)
            #
            assert (consolidatedTimes.shape[0] == spikeWaveforms.shape[0])
            thisUnit.spiketrains.append(newSt)
            newSt.unit = thisUnit
            newSt.annotations['unitAnnotations'] = json.dumps(thisUnit.annotations.copy())
            if createRelationship:
                thisUnit.create_relationship()
            seg.spiketrains.append(newSt)
            newSt.segment = seg
            print('thisUnit.st len = {}'.format(len(thisUnit.spiketrains)))
            if figureOutputFolder is not None:
                stashStimTimesDict[thisUnit.name] = copy(consolidatedTimes)
    if createRelationship:
        for chanIdx in block.channel_indexes:
            chanIdx.create_relationship()
        seg.create_relationship()
        block.create_relationship()
    # [un.name for un in block.filter(objects=Unit)]
    # [len(un.spiketrains) for un in block.filter(objects=Unit)]
    ############## add plots
    if figureOutputFolder is not None:
        confPlotWinSize = 90.  # seconds
        plotRounds = tdDF['t'].apply(lambda x: np.floor(x / confPlotWinSize))
        plotCols = [i for i in tdDF.columns if 'ins_td' in i]
        for pr in plotRounds.unique():
            plotMask = (plotRounds == pr)
            fig, ax = plt.subplots(1, 1, figsize=(21, 3))
            for cN in plotCols:
                plotTrace = stats.zscore(tdDF.loc[plotMask, cN])
                ax.plot(tdDF.loc[plotMask, 't'], plotTrace, label=cN, alpha=0.3, rasterized=True)
            axSize = ax.get_ylim()[1] - ax.get_ylim()[0]
            rasterLevels = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], len(stashStimTimesDict) + 2)
            for uId, (uN, uT) in enumerate(stashStimTimesDict.items()):
                plotTMask = (uT >= tdDF.loc[plotMask, 't'].min()) & (uT < tdDF.loc[plotMask, 't'].max())
                ax.scatter(
                    uT[plotTMask], uT[plotTMask] ** 0 - 1 + rasterLevels[uId + 1],
                    label=uN, rasterized=True, marker='+')
            leg = ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('time domain data (a.u.)')
            fig.tight_layout()
            figSaveOpts = dict(
                bbox_extra_artists=(ax.get_legend(), ),
                bbox_inches='tight')
            pulseConfirmationPDF.savefig(**figSaveOpts)
            if showPlots:
                plt.show()
            else:
                plt.close()
        pulseConfirmationPDF.close()
    ######################
    return block


'''
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
            # if plotDetection and False:
            #     ax.plot(
            #         tdSeg.index,
            #         thisEdgeEnhancer,
            #         '--', c=cLookup[colName],
            #         label='edge enhancer {}'.format(colName))
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
                max(detectSignal.index[0], foundTimestamp[0] - 5 * stimRate ** (-1)),
                min(detectSignal.index[-1], foundTimestamp[0] + 15 * stimRate ** (-1))
            ])
        except:
            pass
    return detectSignal, np.atleast_1d(foundTimestamp[keepSlice]), usedExpectedTimestamp
'''


def extractArtifactTimestampsMahalanobis(
        tdSeg, fs,
        gaussWid=200e-3, mahalTrainingTMax=150e-3,
        thresh=2, offsetFromPeak=0,
        stimRate=100,
        keepWhat='first',
        threshMethod='cross',
        enhanceEdges=True,
        expectedTimestamp=None,
        ROIBasis=None,
        ROIMask=None, keepSlice=0, name=None, 
        plotAnomalies=None, anomalyOccured=None,
        plotDetection=False, plotKernel=False
        ):
    if plotDetection:
        # sns.set()
        # sns.set_style("whitegrid")
        # sns.set_color_codes("dark")
        cPal = sns.color_palette(n_colors=tdSeg.columns.size)
        cLookup = {n: cPal[i] for i, n in enumerate(tdSeg.columns)}
        fig, ax = plt.subplots(figsize=(20, 8))
        for colName, tdCol in tdSeg.iteritems():
            ax.plot(
                tdSeg.index,
                tdCol.values,
                '-', c=cLookup[colName],
                label='original signal {}'.format(colName))
        ax.set_xlabel('Time (sec)')
        twinAx = ax.twinx()
    else:
        fig, ax, twinAx = None, None, None
    #
    # fit on first part of tdSeg
    firstPartMask = tdSeg.index < tdSeg.index[0] + gaussWid
    empCov = EmpiricalCovariance().fit(tdSeg.loc[firstPartMask, :].to_numpy())
    mahalDist = pd.Series(
        np.sqrt(empCov.mahalanobis(tdSeg)),
        index=tdSeg.index)
    if enhanceEdges:
        # convolve with a future facing kernel
        edgeEnhancer = pd.Series(
            hf.noisyTriggerCorrection(
                mahalDist, fs, gaussWid, order=2,
                applyZScore=True, applyAbsVal=True,
                applyScaler=None,
                plotKernel=plotKernel), index=tdSeg.index)
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
    detectSignalPre = pd.Series(hf.enhanceNoisyTriggers(
        mahalDist, correctionFactor=correctionFactor,
        applyZScore=False, applyAbsVal=False,
        applyScaler=None), index=tdSeg.index).fillna(0)
    #
    detectEmpCov = EmpiricalCovariance().fit(
        detectSignalPre.loc[firstPartMask].to_numpy().reshape(-1, 1))
    #
    detectSignal = pd.Series(
        np.sqrt(detectEmpCov.mahalanobis(
            detectSignalPre.to_numpy().reshape(-1, 1))),
        index=tdSeg.index)
    #
    if plotDetection:
        twinAx.plot(
            detectSignal.index,
            detectSignal.values, 'm--', label='detect signal before slot mask')
        twinAx.axhline(thresh, color='r')
    #
    if ROIMask is not None:
        detectSignal.loc[~ROIMask] = np.nan
        detectSignal.interpolate(inplace=True)
        if plotDetection:
            ax.axvspan(
                detectSignal.index[ROIMask][1],
                detectSignal.index[ROIMask][-2],
                facecolor='g', alpha=0.5, zorder=-100)
    if plotDetection:
        twinAx.plot(
            detectSignal.index,
            detectSignal.values, 'y-', label='detect signal')
        twinAx.axhline(thresh, color='r')
    #
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
    #
    if len(idxLocal):
        idxLocal = idxLocal - int(fs * offsetFromPeak)
        idxLocal[idxLocal < 0] = 0
        usedExpectedTimestamp = False
        foundTimestamp = tdSeg.index[idxLocal]
        if plotDetection:
            twinAx.plot(
                foundTimestamp, foundTimestamp ** 0 - 1,
                'r*', markersize=12, label='t = {}'.format(foundTimestamp[0]))
    else:
        print(
            'After peakutils.indexes, no peaks found! ' +
            'Using JSON times...')
        usedExpectedTimestamp = True
        foundTimestamp = np.atleast_1d(expectedTimestamp)
    if plotDetection:
        try:
            ax.set_xlim([
                max(detectSignal.index[0], foundTimestamp[0] - 50e-3),
                min(detectSignal.index[-1], foundTimestamp[0] + 100e-3)
            ])
        except:
            pass
    return (
        detectSignal, np.atleast_1d(foundTimestamp[keepSlice]),
        usedExpectedTimestamp, fig, ax, twinAx)


def insDataToBlock(
        tdDF, accelDF, stimStatusSerial,
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
        #  tdDF['t'].iloc[0],
        tdDF['t'].iloc[-1] + 1/sampleRate.magnitude,
        1/sampleRate.magnitude
        )
    tdInterp = hf.interpolateDF(
        tdDF, fullX,
        kind='linear', fill_value=(0, 0),
        x='t', columns=tdDataCols)
    tdInterp.loc[:, 'validTD'] = False
    tdInterp.loc[:, 'trialSegment'] = np.nan
    for sessionIdx, group in tdDF.groupby('trialSegment'):
        validMask = (tdInterp['t'] >= group['t'].min()) & (tdInterp['t'] <= group['t'].max())
        tdInterp.loc[:, 'validTD'] = tdInterp.loc[:, 'validTD'] | validMask
        tdInterp.loc[validMask, 'trialSegment'] = sessionIdx
    tdInterp.loc[:, 'trialSegment'] = (
        tdInterp['trialSegment']
        .fillna(method='ffill')
        .fillna(method='bfill')
        )
    if accelDF is not None:
        accelInterp = hf.interpolateDF(
            accelDF, fullX,
            kind='linear', fill_value=(0, 0),
            x='t', columns=accelColumns)
    tStart = fullX[0]
    #
    blockName = trialFilesStim['experimentName'] + '_ins'
    block = Block(name=blockName)
    block.annotate(jsonSessionNames=trialFilesStim['jsonSessionNames'])
    seg = Segment(name='seg0_' + blockName)
    block.segments.append(seg)
    #
    for idx, colName in enumerate(tdDataCols):
        sigName = 'ins_td{}'.format(senseInfo.loc[idx, 'senseChan'])
        asig = AnalogSignal(
            tdInterp[colName].to_numpy()*pq.mV,
            name='seg0_' + sigName,
            sampling_rate=sampleRate,
            dtype=float,
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
    # non td data
    for idx, colName in enumerate(['validTD', 'trialSegment']):
        asig = AnalogSignal(
            tdInterp[colName].to_numpy()*pq.mV,
            name='seg0_' + colName,
            sampling_rate=sampleRate,
            dtype=float)
        asig.t_start = tStart*pq.s
        seg.analogsignals.append(asig)
        chanIdx = ChannelIndex(
            name=colName,
            index=np.array([0]),
            channel_names=np.array([colName]),
            channel_ids=np.array([0])
            )
        block.channel_indexes.append(chanIdx)
        chanIdx.analogsignals.append(asig)
        asig.channel_index = chanIdx
    if accelDF is not None:
        for idx, colName in enumerate(accelColumns):
            if colName == 'inertia':
                accUnits = (pq.N)
            else:
                accUnits = (pq.m/pq.s**2)
            sigName = 'ins_acc{}'.format(accelNixColNames[idx])
            asig = AnalogSignal(
                accelInterp[colName].values*accUnits,
                name='seg0_' + sigName,
                sampling_rate=sampleRate,
                dtype=float)
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
        annCol=['ins_property', 'ins_value'])
    evTimes = pd.Series(stimEvents[0].times.magnitude)
    closestTimes, closestIdx = hf.closestSeries(
        takeFrom=evTimes, compareTo=tdInterp['t'])
    evTSegs = tdInterp.loc[closestIdx, 'trialSegment'].to_numpy()
    for ev in stimEvents:
        ev.array_annotations.update({'trialSegment': evTSegs})
        ev.annotations.update({
            'trialSegment': evTSegs,
            'arrayAnnNames': ['trialSegment']})
    seg.events = stimEvents
    block.create_relationship()
    #
    return block


def unpackINSBlock(
    block, unpackAccel=True,
    dummyTherapySwitches=False, elecConfiguration=None,
    convertStimToSinglePulses=False):
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
        if len(accelAsig):
            accelDF = ns5.analogSignalsToDataFrame(accelAsig, useChanNames=True)
        else:
            accelDF = None
            unpackAccel = False
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
        deriveCols=deriveCols, progAmpNames=progAmpNames,
        dummyTherapySwitches=dummyTherapySwitches,
        elecConfiguration=elecConfiguration)
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
