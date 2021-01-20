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
import matplotlib, pdb, traceback
# matplotlib.use('Qt5Agg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('PS')   # noninteract output
from matplotlib import pyplot as plt
import dill as pickle
from scipy import stats
from importlib import reload
import datetime
from datetime import datetime as dt
from datetime import timezone
import peakutils
import numpy as np
import pandas as pd
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.motor_encoder_new as mea
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.estimateElectrodeImpedances as eti
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.mdt_constants as mdt_constants
import warnings
import h5py
import os
import math as m
import seaborn as sns
import scipy.interpolate as intrp
import quantities as pq
import json
import rcsanalysis.packetizer as rcsa
import rcsanalysis.packet_func as rcsa_helpers
from neo.io import BlackrockIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
import neo
import elephant.pandas_bridge as elphpdb

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
print('Detecting NSP Timestamps...')
nspChannelName = eventInfo['inputIDs']['tapSync']

segIdx = 0
nspSeg = nspBlock.segments[segIdx]
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
#
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
summaryText = '<h2>{}</h2>\n'.format(nspPath)
for trialSegment, group in approxTapTimes.groupby('tapGroup'):
    recDateTime = nspBlock.rec_datetime.replace(tzinfo=timezone.utc).astimezone(tz=None)
    firstNSPTap = recDateTime + pd.Timedelta(int(group['NSP'].min() * 1e3), unit='milli')
    summaryText += (
        '<h3>Segment {} started: '.format(trialSegment) +
        firstNSPTap.strftime('%Y-%m-%d %H:%M:%S') +
        ' (t = {:.3f} sec)</h3>\n'.format(group['NSP'].min()))
    if trialSegment == approxTapTimes['tapGroup'].max():
        lastNSPTime = recDateTime + pd.Timedelta(int(nspDF['t'].max() * 1e3), unit='milli')
    else:
        nextGroup = approxTapTimes.loc[(approxTapTimes['tapGroup'] == trialSegment + 1), :]
        lastNSPTime = recDateTime + pd.Timedelta(int(nextGroup['NSP'].min() * 1e3), unit='milli')
    segDur = lastNSPTime - firstNSPTap
    summaryText += (
        '<h3>             ended: '.format(trialSegment) +
        lastNSPTime.strftime('%Y-%m-%d %H:%M:%S') +
        ' (lasted up to {} sec)</h3>\n'.format(segDur.total_seconds()))


segIdx = 0
problemChannelsList = []
problemThreshold = 4e3
targetQuantile = 0.99
summaryText += (
    '<h3>.95 voltage intervals:</h3>\n<p>\n'
    .format(2 * problemThreshold))
for asigP in nspBlock.segments[segIdx].analogsignals:
    chName = asigP.channel_index.name
    if 'ainp' not in chName:
        print('    Loading {}'.format(chName))
        lastT = min((spikeSortingOpts['utah']['previewOffset'] + spikeSortingOpts['utah']['previewDuration']) * pq.s,  asigP.t_stop)
        firstT = max(spikeSortingOpts['utah']['previewOffset'] * pq.s,  asigP.t_start)
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
summaryText += ('</p>\n<h3>List view: </h3>\n')
summaryText += '{}\n'.format(problemChannelsList)
approxTimesPath = os.path.join(
    scratchFolder,
    '{}_{}_NS5_Preview.html'.format(
        experimentName, ns5FileName))
with open(approxTimesPath, 'w') as _file:
    _file.write(summaryText)
