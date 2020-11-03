"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --blockIdx=blockIdx                             which trial to analyze
    --exp=exp                                       which experimental day to analyze
    --inputBlockSuffix=inputBlockSuffix             append a name to the resulting blocks?
    --inputInsBlockSuffix=inputInsBlockSuffix       append a name to the resulting blocks?
    --lazy                                          whether to fully load data from blocks [default: True]
    --curateManually                                whether to manually confirm synch [default: False]
    --preparationStage                              get aproximate timings and print to shell, to help identify time ranges? [default: False]
    --plotting                                      whether to display confirmation plots [default: False]
"""
import matplotlib, pdb, traceback
matplotlib.use('Qt5Agg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
#  matplotlib.use('PS')   # noninteract output
from matplotlib import pyplot as plt
import dill as pickle
from scipy import stats
from importlib import reload
from datetime import datetime as dt
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

import h5py
import os
import math as m
import seaborn as sns
import scipy.interpolate as intrp
import quantities as pq
import json
import rcsanalysis.packetizer as rcsa
import rcsanalysis.packet_func as rcsa_helpers
import datetime

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain)
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
if arguments['inputBlockSuffix'] is None:
    arguments['inputBlockSuffix'] = ''
if arguments['inputInsBlockSuffix'] is None:
    arguments['inputInsBlockSuffix'] = ''
synchFunPath = os.path.join(
    scratchFolder,
    '{}_{}_synchFun.pickle'.format(experimentName, ns5FileName))

#  Load NSP Data
############################################################
nspPath = os.path.join(
    scratchFolder,
    ns5FileName + arguments['inputBlockSuffix'] +
    '.nix')
nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=arguments['lazy'])
'''
startTime_s = None
dataLength_s = None
print(
    'Loading NSP Block {}'
    .format(ns5FileName + arguments['inputBlockSuffix']))
try:
    channelData, _ = ns5.getNIXData(
        fileName=ns5FileName + arguments['inputBlockSuffix'],
        folderPath=scratchFolder,
        elecIds=['ainp1'], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)
except Exception:
    traceback.print_exc()
'''
print('Detecting NSP Timestamps...')
interTriggerInterval = .2
nspChannelName = 'ainp1'
segIdx = 0
nspSeg = nspBlock.segments[segIdx]
nspSyncAsig = nspSeg.filter(name='seg0_{}'.format(nspChannelName))[0]
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
nspLims = [min(nspSrs), max(nspSrs)]
nspDiffUncertainty = nspSrs.diff().abs().quantile(1-1e-3) / 4
nspThresh = nspLims[0] + (nspLims[-1] - nspLims[0]) / 2
print(
    'On trial {}, detecting NSP threshold crossings.'
    .format(blockIdx))
nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
    nspSrs, thresh=nspThresh,
    iti=interTriggerInterval, fs=float(nspSyncAsig.sampling_rate),
    edgeType='both', itiWiggle=.2,
    absVal=False, plotting=arguments['plotting'], keep_max=False)
allNSPTapTimes = nspDF.loc[nspPeakIdx, 't'].to_numpy()
allTapTimestampsNSP = []
'''
for trialSegment in pd.unique(td['data']['trialSegment']):
    #  Where in NSP to look
    #  #)
    tStart = sessionTapRangesNSP[blockIdx][trialSegment]['timeRanges'][0]
    tStop = sessionTapRangesNSP[blockIdx][trialSegment]['timeRanges'][1]
    nspMask = (channelData['t'] > tStart) & (channelData['t'] < tStop)
    #
    tapIdxNSP = hf.getTriggers(
        channelData['data'].loc[nspMask, 'ainp7'],
        thres=2, iti=0.2, minAmp=1)
    tapTimestampsNSP = channelData['t'].loc[tapIdxNSP]
    keepIdx = sessionTapRangesNSP[blockIdx][trialSegment]['keepIndex']
    tapTimestampsNSP = tapTimestampsNSP.iloc[keepIdx]
    print('tSeg {}: NSP Taps:\n{}'.format(
        trialSegment, tapTimestampsNSP))
    print('diff:\n{}'.format(
        tapTimestampsNSP.diff() * 1e3))
    allTapTimestampsNSP.append(tapTimestampsNSP)
'''
#  load INS Data
############################################################
print('Loading INS Block...')
insPath = os.path.join(
    scratchFolder,
    ns5FileName + '_ins' +
    arguments['inputInsBlockSuffix'] +
    '.nix')
insReader, insBlock = ns5.blockFromPath(
    insPath, lazy=arguments['lazy'])
sessionStartTimes = [
    datetime.datetime.utcfromtimestamp(int(sN.split('Session')[-1]) / 1000)
    for sN in insBlock.annotations['jsonSessionNames']]
approxDeltaRecTime = (nspBlock.rec_datetime - sessionStartTimes[0]).total_seconds()
approxInsTapTimes = allNSPTapTimes + approxDeltaRecTime
approxTapTimes = pd.DataFrame([allNSPTapTimes, approxInsTapTimes]).T
approxTapTimes.columns = ['NSP', 'INS']
tapIntervals = approxTapTimes['NSP'].diff()
approxTapTimes['tapGroup'] = (tapIntervals > 60).cumsum()
autoTimeRanges = {
    'NSP': [],
    'INS': []
}
for trialSegment, group in approxTapTimes.groupby('tapGroup'):
    tapTrainDur = np.ceil(group['NSP'].max() - group['NSP'].min())
    firstNSPTap = group['NSP'].min()
    firstINSTap = group['INS'].min()
    approxTapTimes.loc[group.index, 'tStart_NSP'] = firstNSPTap - 1
    approxTapTimes.loc[group.index, 'tStop_NSP'] = firstNSPTap + tapTrainDur + 1
    approxTapTimes.loc[group.index, 'tStart_INS'] = firstINSTap - 1
    approxTapTimes.loc[group.index, 'tStop_INS'] = firstINSTap + tapTrainDur + 1
    autoTimeRanges['NSP'].append((firstNSPTap - 1, firstNSPTap + tapTrainDur + 1))
    autoTimeRanges['INS'].append((firstINSTap - 1, firstINSTap + tapTrainDur + 1))
    tapTimestampsNSP = group['NSP'].astype(np.float)
    allTapTimestampsNSP.append(tapTimestampsNSP)
    print('tSeg {}: NSP Taps:\n{}'.format(
        trialSegment, tapTimestampsNSP))
    print('diff:\n{}'.format(
        np.diff(tapTimestampsNSP) * 1e3))
approxTimesPath = os.path.join(
    scratchFolder,
    '{}_{}_approximateTimes.html'.format(experimentName, ns5FileName))
approxTapTimes.to_html(approxTimesPath)

defaultTapDetectOpts = {
    'iti': 0.2,
    'keepIndex': slice(None)
    }
tapDetectOpts = expOpts['synchInfo']['ins'][blockIdx]

for trialSegmentKey in tapDetectOpts.keys():
    if tapDetectOpts[trialSegmentKey]['timeRanges'] is None:
        tapDetectOpts[trialSegmentKey]['timeRanges'] = [autoTimeRanges['INS'][trialSegmentKey]]
    for key in defaultTapDetectOpts.keys():
        if key not in tapDetectOpts[trialSegmentKey].keys():
            tapDetectOpts[trialSegmentKey].update(
                {key: defaultTapDetectOpts[key]}
                )
defaultSessionTapRangesNSP = {
    'keepIndex': slice(None)
    }
sessionTapRangesNSP = expOpts['synchInfo']['nsp'][blockIdx]
for trialSegmentKey in sessionTapRangesNSP.keys():
    if sessionTapRangesNSP[trialSegmentKey]['timeRanges'] is None:
        sessionTapRangesNSP[trialSegmentKey]['timeRanges'] = autoTimeRanges['NSP'][trialSegmentKey]
    for key in defaultSessionTapRangesNSP.keys():
        if key not in sessionTapRangesNSP[trialSegmentKey].keys():
            sessionTapRangesNSP[trialSegmentKey].update(
                {key: defaultSessionTapRangesNSP[key]}
                )
#  make placeholders for interpolation functions
interpFunINStoNSP = {
    key: [None for i in value.keys()]
    for key, value in sessionTapRangesNSP.items()
    }
interpFunHUTtoINS = {
    key: [None for i in value.keys()]
    for key, value in sessionTapRangesNSP.items()
    }
'''
    reader = neo.io.NixIO(filename=insDataPath, mode='ro')
    insBlock = reader.read_block()
    insBlock.create_relationship()  # need this!
    reader.close()
    for st in insBlock.filter(objects=SpikeTrain):
        #  print('unit is {}'.format(st.unit.name))
        #  print('spiketrain is {}'.format(st.name))
        if 'arrayAnnNames' in st.annotations.keys():
            #  print(st.annotations['arrayAnnNames'])
            for key in st.annotations['arrayAnnNames']:
                st.array_annotations.update({key: st.annotations[key]})
    '''
#  insBlock = ns5.loadWithArrayAnn(insDataPath)
tdDF, accelDF, stimStatus = mdt.unpackINSBlock(insBlock)
td = {'data': tdDF, 't': tdDF['t']}
accel = {'data': accelDF, 't': accelDF['t']}
#  Detect INS taps
############################################################
td['data']['NSPTime'] = np.nan
accel['data']['NSPTime'] = np.nan
allTapTimestampsINSAligned = []
allTapTimestampsINS = []
if not os.path.exists(synchFunPath):
    print('Detecting INS Timestamps...')
    overrideSegments = overrideSegmentsForTapSync[blockIdx]
    clickDict = {
        i: {
            'ins': [],
            'nsp': []
            }
        for i in pd.unique(td['data']['trialSegment'])}
    for trialSegment in pd.unique(td['data']['trialSegment']).astype(int):
        print('detecting INS taps on trial segment {}\n'.format(trialSegment))
        accelGroupMask = accel['data']['trialSegment'] == trialSegment
        accelGroup = accel['data'].loc[accelGroupMask, :]
        tdGroupMask = td['data']['trialSegment'] == trialSegment
        tdGroup = td['data'].loc[tdGroupMask, :]
        tapTimestampsINS, peakIdx, tapDetectSignal = mdt.getINSTapTimestamp(
            tdGroup, accelGroup,
            tapDetectOpts[trialSegment]
            )
        print('tSeg {}, INS Taps:\n{}'.format(
            trialSegment, tapTimestampsINS))
        print('diff:\n{}'.format(
            tapTimestampsINS.diff() * 1e3))
        allTapTimestampsINS.append(tapTimestampsINS)
        
        keepIdx = sessionTapRangesNSP[trialSegment]['keepIndex']
        allTapTimestampsNSP[trialSegment] = allTapTimestampsNSP[trialSegment].iloc[keepIdx]
        #
        if arguments['curateManually']:
            try:
                clickDict[trialSegment] = mdt.peekAtTaps(
                    tdGroup, accelGroup, tapDetectSignal,
                    channelData, trialSegment,
                    tapDetectOpts, sessionTapRangesNSP,
                    insX='t', plotBlocking=plotBlocking,
                    nspChannelName=nspChannelName,
                    allTapTimestampsINS=allTapTimestampsINS,
                    allTapTimestampsNSP=allTapTimestampsNSP,
                    )
            except Exception:
                traceback.print_exc()
    # perform the sync
    ############################################################
    for trialSegment in pd.unique(td['data']['trialSegment']).astype(int):
        accelGroupMask = accel['data']['trialSegment'] == trialSegment
        accelGroup = accel['data'].loc[accelGroupMask, :]
        tdGroupMask = td['data']['trialSegment'] == trialSegment
        tdGroup = td['data'].loc[tdGroupMask, :]
        # if overriding with manually identified points
        if not (trialSegment in overrideSegments.keys()):
            if len(clickDict[trialSegment]['ins']):
                allTapTimestampsINS[trialSegment] = clickDict[trialSegment]['ins']
            theseTapTimestampsINS = allTapTimestampsINS[trialSegment]
            # if overriding with manually identified points
            if len(clickDict[trialSegment]['nsp']):
                allTapTimestampsNSP[trialSegment] = clickDict[trialSegment]['nsp']
            theseTapTimestampsNSP = allTapTimestampsNSP[trialSegment]
            #
        else:
            print('\t Overriding trialSegment {}'.format(trialSegment))
            theseTapTimestampsINS = allTapTimestampsINS[overrideSegments[trialSegment]]
            theseTapTimestampsNSP = allTapTimestampsNSP[overrideSegments[trialSegment]]
        #
        tdGroup, accelGroup, insBlock, thisINStoNSP = ns5.synchronizeINStoNSP(
            tapTimestampsNSP=theseTapTimestampsNSP,
            tapTimestampsINS=theseTapTimestampsINS,
            NSPTimeRanges=(
                channelData['t'].iloc[0], channelData['t'].iloc[-1]),
            td=tdGroup, accel=accelGroup, insBlock=insBlock,
            trialSegment=trialSegment, degree=0)
        td['data'].loc[tdGroupMask, 'NSPTime'] = tdGroup['NSPTime']
        accel['data'].loc[accelGroupMask, 'NSPTime'] = accelGroup['NSPTime']
        #
        interpFunINStoNSP[trialSegment] = thisINStoNSP
        allTapTimestampsINSAligned.append(thisINStoNSP(
            theseTapTimestampsINS))
    with open(synchFunPath, 'wb') as f:
        pickle.dump(interpFunINStoNSP, f)
else:
    with open(synchFunPath, 'rb') as f:
        interpFunINStoNSP = pickle.load(f)
    theseInterpFun = interpFunINStoNSP
    td['data']['NSPTime'] = np.nan
    accel['data']['NSPTime'] = np.nan
    for trialSegment in pd.unique(td['data']['trialSegment']).astype(int):
        accelGroupMask = accel['data']['trialSegment'] == trialSegment
        accelGroup = accel['data'].loc[accelGroupMask, :]
        tdGroupMask = td['data']['trialSegment'] == trialSegment
        tdGroup = td['data'].loc[tdGroupMask, :]
        #
        tdGroup, accelGroup, insBlock, thisINStoNSP = ns5.synchronizeINStoNSP(
            precalculatedFun=theseInterpFun[trialSegment],
            NSPTimeRanges=(
                channelData['t'].iloc[0], channelData['t'].iloc[-1]),
            td=tdGroup, accel=accelGroup, insBlock=insBlock,
            trialSegment=trialSegment, degree=0)
        #
        td['data'].loc[tdGroupMask, 'NSPTime'] = tdGroup['NSPTime']
        accel['data'].loc[accelGroupMask, 'NSPTime'] = accelGroup['NSPTime']
#
print(
    'Interpolation between INS and NSP: {}'
    .format(interpFunINStoNSP))
td['NSPTime'] = td['data']['NSPTime']
accel['NSPTime'] = accel['data']['NSPTime']
#
if plottingFigures:
    try:
        hf.peekAtTaps(
            td, accel,
            channelData,
            tapDetectOpts, sessionTapRangesNSP,
            insX='NSPTime',
            allTapTimestampsINS=allTapTimestampsINSAligned,
            allTapTimestampsNSP=allTapTimestampsNSP)
    except Exception:
        traceback.print_exc()
############################################################
addingToNix = True
if addingToNix:
    insBlockJustSpikes = hf.extractSignalsFromBlock(insBlock)
    insSpikeTrains = insBlockJustSpikes.filter(objects=SpikeTrain)
    #  reader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
    #  nspBlock = reader.read_block(lazy=True)
    #  nspStP = nspBlock.filter(objects=SpikeTrainProxy)
    #  nspSt = [i.load(load_waveforms=True) for i in nspStP]
    #  spikeReader = neo.io.nixio_fr.NixIO(filename=os.path.join(trialFilesFrom['utah']['folderPath'], 'tdc_' + trialFilesFrom['utah']['ns5FileName'], 'tdc_' + trialFilesFrom['utah']['ns5FileName'] + '.nix'))
    #  tdcBlock = spikeReader.read_block(lazy=True)
    #  tdcStP = tdcBlock.filter(objects=SpikeTrainProxy)
    #  tdcSt = [i.load(load_waveforms=True) for i in tdcStP]
    for st in insSpikeTrains:
        if st.waveforms is None:
            st.sampling_rate = 3e4*pq.Hz
            st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
    ns5.addBlockToNIX(
        insBlockJustSpikes, neoSegIdx=[0],
        writeAsigs=False, writeSpikes=True,
        fileName=ns5FileName + arguments['inputBlockSuffix'],
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
    tdColumns = [
        i for i in td['data'].columns
        if 'ins_' in i]
    tdInterp = hf.interpolateDF(
        td['data'], channelData['t'],
        kind='linear', fill_value=(0, 0),
        x='NSPTime', columns=tdColumns)
    tdBlock = ns5.dataFrameToAnalogSignals(
        tdInterp,
        idxT='NSPTime',
        probeName='insTD', samplingRate=3e4*pq.Hz,
        dataCol=tdColumns,
        forceColNames=tdColumns)
    ns5.addBlockToNIX(
        tdBlock, neoSegIdx=[0],
        fileName=ns5FileName + arguments['inputBlockSuffix'],
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
    accelColumns = [
        'ins_accx', 'ins_accy',
        'ins_accz', 'ins_accinertia']
    accelInterp = hf.interpolateDF(
        accel['data'], channelData['t'],
        kind='linear', fill_value=(0, 0),
        x='NSPTime', columns=accelColumns)
    accelBlock = ns5.dataFrameToAnalogSignals(
        accelInterp,
        idxT='NSPTime',
        probeName='insAccel', samplingRate=3e4*pq.Hz,
        dataCol=accelColumns,
        forceColNames=accelColumns)
    ns5.addBlockToNIX(
        accelBlock, neoSegIdx=[0],
        fileName=ns5FileName + arguments['inputBlockSuffix'],
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
