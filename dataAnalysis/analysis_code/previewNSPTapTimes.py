"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --blockIdx=blockIdx                             which trial to analyze
    --exp=exp                                       which experimental day to analyze
    --lazy                                          whether to fully load data from blocks [default: True]
    --plotting                                      whether to display confirmation plots [default: False]
    --usedTENSPulses                                whether the sync was done using TENS pulses (as opposed to mechanical taps) [default: False]
"""
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'DISPLAY' in os.environ:
    matplotlib.use('QT5Agg')   # generate postscript output
else:
    matplotlib.use('PS')   # generate postscript output
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pdb, traceback
import os
import seaborn as sns
import quantities as pq
from neo.io import BlackrockIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
import neo
import elephant.pandas_bridge as elphpdb
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
'''

import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.motor_encoder_new as mea
import dataAnalysis.helperFunctions.estimateElectrodeImpedances as eti
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.mdt_constants as mdt_constants
import rcsanalysis.packetizer as rcsa
import rcsanalysis.packet_func as rcsa_helpers
import warnings
import h5py
import math as m
import json
import dill as pickle
from scipy import stats
import scipy.interpolate as intrp
from importlib import reload
import datetime
from datetime import datetime as dt
from datetime import timezone
import peakutils

'''
sns.set(
    context='paper', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#  Load NSP Data
############################################################
if 'rawBlockName' in spikeSortingOpts['utah']:
    BlackrockFileName = ns5FileName.replace(
        'Block', spikeSortingOpts['utah']['rawBlockName'])
else:
    BlackrockFileName = ns5FileName
nspPath = os.path.join(
    nspFolder, BlackrockFileName + '.ns5')
print('Loading NSP Block: {}'.format(nspPath))
reader = BlackrockIO(
    filename=nspPath, nsx_to_load=5)
reader.parse_header()
metadata = reader.header
nspBlock = ns5.readBlockFixNames(
    reader,
    block_index=0, lazy=True,
    signal_group_mode='split-all',
    reduceChannelIndexes=True,
    )

segIdx = 0
nspSeg = nspBlock.segments[segIdx]
dummyAsig = nspSeg.filter(objects=AnalogSignalProxy)[0]
recDateTime = pd.Timestamp(nspBlock.rec_datetime, tz='utc')
summaryText = '<h2>{}</h2>\n'.format(nspPath)
summaryText += '<h3>Block started {}</h3>\n'.format(
    recDateTime.tz_convert("America/New_York").strftime('%Y-%m-%d %H:%M:%S'))
recEndTime = recDateTime + pd.Timedelta(float(dummyAsig.t_stop), unit='s')
summaryText += '<h3>Block ended {}</h3>\n<br>\n'.format(
    recEndTime.tz_convert("America/New_York").strftime('%Y-%m-%d %H:%M:%S'))
#
orcaFolderPath = os.path.join(remoteBasePath, 'ORCA Logs')
listOfSummarizedPath = os.path.join(
    orcaFolderPath,
    subjectName + '_list_of_summarized.json'
    )
if os.path.exists(listOfSummarizedPath):
    summaryDF = pd.read_json(
        listOfSummarizedPath,
        orient='records',
        convert_dates=['tStart', 'tEnd'],
        dtype={
            'unixStartTime': int,
            'tStart': pd.DatetimeIndex,
            'tEnd': pd.DatetimeIndex,
            'hasTD': bool,
            'duration': float,
            'maxAmp': int,
            'minAmp': int,
        })
    sessionStarts = pd.DatetimeIndex(summaryDF['tStart']).tz_localize(tz="UTC")
    sessionEnds = pd.DatetimeIndex(summaryDF['tEnd']).tz_localize(tz="UTC")
    compatibleSessionsMask = (
        (sessionStarts > (recDateTime - pd.Timedelta('1M'))) &
        (sessionEnds < recEndTime) &
        (summaryDF['hasTD']) &
        (summaryDF['maxAmp'].notna()))
    insSessions = summaryDF.loc[compatibleSessionsMask, :].copy()
    insSessions.loc[:, 'tStart'] = sessionStarts[compatibleSessionsMask].tz_convert("America/New_York")
    insSessions.loc[:, 'tEnd'] = sessionEnds[compatibleSessionsMask].tz_convert("America/New_York")
    insSessions.loc[:, 'delayFromNSP'] = (sessionStarts[compatibleSessionsMask] - recDateTime).total_seconds()
    summaryText += '<h3>Companion INS sessions: </h3>'
    summaryText += insSessions.rename(
        columns={
            'tStart': 'Start Time', 'tEnd': 'End Time',
            'duration': 'Duration (sec)', 'delayFromNSP': 'delay after NSP start (sec)'
            }).to_html()
    summaryText += '<br> insSessions = [{}]'.format(
        ', '.join(
            [
                "'Session{}'".format(unT)
                for unT in insSessions['unixStartTime']
            ]
        )
    )
if 'tapSync' in eventInfo['inputIDs']:
    print('Detecting NSP Tap Timestamps...')
    nspChannelName = eventInfo['inputIDs']['tapSync']
    nspSyncAsig = nspSeg.filter(name='seg0_{}'.format(nspChannelName))[0].load()
    # try:
    #     tStart, tStop = synchInfo['nsp'][blockIdx]['timeRanges']
    # except Exception:
    #     traceback.print_exc()
    #     tStart = float(nspSyncAsig.times[0] + 2 * pq.s)
    #     tStop = float(nspSyncAsig.times[-1] - 2 * pq.s)
    # nspTimeMask = hf.getTimeMaskFromRanges(
    #     nspSyncAsig.times, [(tStart, tStop)])
    nspSrs = pd.Series(nspSyncAsig.magnitude.flatten())
    nspDF = nspSrs.to_frame(name=nspChannelName)
    nspDF['t'] = nspSyncAsig.times.magnitude
    channelData = {
        'data': nspDF,
        't': nspDF['t']
        }
    # nspLims = nspSrs.quantile([1e-6, 1-1e-6]).to_list()
    if arguments['usedTENSPulses']:
        interTriggerInterval = 39.7e-3  # 20 Hz
        minAnalogValue = 200  # mV (determined empirically)
        nspSrs.loc[nspSrs <= minAnalogValue] = 0
        nspPeakIdx = hf.getTriggers(
            nspSrs, iti=interTriggerInterval, itiWiggle=.5,
            fs=float(nspSyncAsig.sampling_rate), plotting=arguments['plotting'],
            thres=2.58, edgeType='rising')
    else:
        interTriggerInterval = .2
        nspLims = [min(nspSrs), max(nspSrs)]
        nspThresh = nspLims[0] + (nspLims[-1] - nspLims[0]) / 2
        nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
            nspSrs, thresh=nspThresh,
            iti=interTriggerInterval, fs=float(nspSyncAsig.sampling_rate),
            edgeType='both', itiWiggle=.2,
            absVal=False, plotting=arguments['plotting'], keep_max=False)

    allNSPTapTimes = nspDF.loc[nspPeakIdx, 't'].to_numpy()

    approxTapTimes = pd.DataFrame([allNSPTapTimes]).T
    approxTapTimes.columns = ['NSP']
    tapIntervals = approxTapTimes['NSP'].diff()
    approxTapTimes['tapGroup'] = (tapIntervals > 60).cumsum()
    autoTimeRanges = {
        'NSP': [],
        }
    for trialSegment, group in approxTapTimes.groupby('tapGroup'):
        firstNSPTap = (
            recDateTime +
            pd.Timedelta(int(group['NSP'].min() * 1e3), unit='milli'))
        summaryText += (
            '<h3>Segment {} started: '.format(trialSegment) +
            firstNSPTap.strftime('%Y-%m-%d %H:%M:%S') +
            ' (t = {:.3f} sec)</h3>\n'.format(group['NSP'].min()))
        if trialSegment == approxTapTimes['tapGroup'].max():
            lastNSPTime = (
                recDateTime +
                pd.Timedelta(int(nspDF['t'].max() * 1e3), unit='milli'))
        else:
            nextGroup = approxTapTimes.loc[(approxTapTimes['tapGroup'] == trialSegment + 1), :]
            lastNSPTime = recDateTime + pd.Timedelta(int(nextGroup['NSP'].min() * 1e3), unit='milli')
        segDur = lastNSPTime - firstNSPTap
        summaryText += (
            '<h3>             ended: '.format(trialSegment) +
            lastNSPTime.strftime('%Y-%m-%d %H:%M:%S') +
            ' (lasted up to {} sec)</h3>\n'.format(segDur.total_seconds()))
#### impedances
impedanceFilePath = os.path.join(
    remoteBasePath,
    '{}_blackrock_impedances.h5'.format(subjectName))
if os.path.exists(impedanceFilePath):
    impedances = prb_meta.getLatestImpedance(
        block=nspBlock, impedanceFilePath=impedanceFilePath)
    impedances.sort_values('impedance')
    summaryText += impedances.to_html()
#### problem channel id
try:
    segIdx = 0
    problemChannelsList = []
    problemThreshold = 4e3
    targetQuantile = 0.99
    summaryText += (
        '<h3>.95 voltage intervals:</h3>\n<p>\n'
        .format(2 * problemThreshold))
    asigList = []
    for asigP in nspBlock.segments[segIdx].analogsignals:
        chName = asigP.channel_index.name
        bankID = asigP.channel_index.annotations['connector_ID']
        # if 'ainp' not in chName:
        if True:
            print('    Loading {}'.format(chName))
            firstT = max(spikeSortingOpts['utah']['previewOffset'] * pq.s,  asigP.t_start)
            lastT = min(firstT + spikeSortingOpts['utah']['previewDuration'] * pq.s,  asigP.t_stop)
            # lastT = min(firstT + 30 * pq.s, asigP.t_stop)
            tempAsig = asigP.load(time_slice=[firstT, lastT])
            sigLims = np.quantile(
                tempAsig, [
                    (1 - targetQuantile) / 2,
                    (1 + targetQuantile) / 2
                    ])
            thisTextRow = (
                '{}: {:.1f} uV to {:.1f} uV <br>\n'
                .format(chName, sigLims[0], sigLims[1]))
            if (sigLims[0] < -1 * problemThreshold) and (sigLims[1] > problemThreshold):
                problemChannelsList.append(chName)
                thisTextRow = '<b>' + thisTextRow.replace('<br>', '</b><br>')
            summaryText += thisTextRow
            ##
            decimateFactor = 100
            asigDFIndex = pd.MultiIndex.from_tuples([(chName, bankID),], names=['feature', 'bankID'])
            asigDF = pd.DataFrame(
                tempAsig.magnitude[::decimateFactor].reshape(1, -1),
                index=asigDFIndex, columns=tempAsig.times.magnitude[::decimateFactor])
            asigDF.columns.name = 'time'
            asigList.append(asigDF)
            ##
    summaryText += ('</p>\n<h3>List view: </h3>\n')
    summaryText += '{}\n'.format(problemChannelsList)
except Exception:
    traceback.print_exc()
preprocDiagnosticsFolder = os.path.join(
    processedFolder, 'preprocDiagnostics'
    )
if not os.path.exists(preprocDiagnosticsFolder):
    os.makedirs(preprocDiagnosticsFolder, exist_ok=True)
approxTimesPath = os.path.join(
    preprocDiagnosticsFolder,
    '{}_{}_NS5_Preview.html'.format(
        experimentName, ns5FileName))
with open(approxTimesPath, 'w') as _file:
    _file.write(summaryText)
allAsigDF = pd.concat(asigList).stack('time').to_frame(name='signal').reset_index()
#
signalRangesFigPath = os.path.join(
    preprocDiagnosticsFolder,
    '{}_{}_NS5_Preview_ranges.pdf'.format(
        experimentName, ns5FileName))
nGroups = allAsigDF.groupby('bankID').ngroups
h = 18
w = 3
aspect = w / h
g = sns.catplot(
    col='bankID', x='signal', y='feature',
    data=allAsigDF, orient='h', kind='violin', ci='sd',
    linewidth=0.5, cut=0,
    sharex=False, sharey=False, height=h, aspect=aspect
    )
g.tight_layout()
g.fig.savefig(
    signalRangesFigPath, bbox_inches='tight')
plt.close()

