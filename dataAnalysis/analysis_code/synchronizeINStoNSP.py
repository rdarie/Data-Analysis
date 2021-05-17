"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --blockIdx=blockIdx                                   which trial to analyze
    --exp=exp                                             which experimental day to analyze
    --inputNSPBlockSuffix=inputNSPBlockSuffix             append a name to the input block?
    --inputINSBlockSuffix=inputINSBlockSuffix             append a name to the input block?
    --lazy                                                whether to fully load data from blocks [default: True]
    --addToBlockSuffix=addToBlockSuffix                   whether to also add stim info to the high pass filtered NSP blocks
    --addToNIX                                            whether to interpolate and add the ins data to the nix file [default: False]
    --curateManually                                      whether to manually confirm synch [default: False]
    --preparationStage                                    get aproximate timings and print to shell, to help identify time ranges? [default: False]
    --plotting                                            whether to display confirmation plots [default: False]
    --usedTENSPulses                                      whether the sync was done using TENS pulses (as opposed to mechanical taps) [default: False]
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
import datetime
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
#############################################################
if arguments['inputNSPBlockSuffix'] is None:
    inputNSPBlockSuffix = ''
else:
    inputNSPBlockSuffix = '_{}'.format(arguments['inputNSPBlockSuffix'])
if arguments['inputINSBlockSuffix'] is None:
    inputINSBlockSuffix = ''
else:
    inputINSBlockSuffix = "_{}".format(arguments['inputINSBlockSuffix'])
#  load INS Data
############################################################
insPath = os.path.join(
    scratchFolder,
    ns5FileName + '_ins' +
    inputINSBlockSuffix +
    '.nix')
print('Loading INS Block from: {}'.format(insPath))
insReader, insBlock = ns5.blockFromPath(
    insPath, lazy=arguments['lazy'],
    # lazy can be False, not worried about the ins data taking up too much
    reduceChannelIndexes=True)
# [un.name for un in insBlock.filter(objects=Unit)]
# [len(un.spiketrains) for un in insBlock.filter(objects=Unit)]
# [id(st) for st in insBlock.filter(objects=Unit)[0].spiketrains]
#  Load NSP Data
############################################################
if 'rawBlockName' in spikeSortingOpts['utah']:
    BlackrockFileName = ns5FileName.replace(
        'Block', spikeSortingOpts['utah']['rawBlockName'])
else:
    BlackrockFileName = ns5FileName
nspPath = os.path.join(
    scratchFolder,
    BlackrockFileName + inputNSPBlockSuffix +
    '.nix')
print('Loading NSP Block from: {}'.format(nspPath))
nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=arguments['lazy'],
    reduceChannelIndexes=True)
#
synchFunFolder = os.path.join(
    scratchFolder, 'synchFuns'
    )
if not os.path.exists(synchFunFolder):
    os.makedirs(synchFunFolder, exist_ok=True)
synchFunPath = os.path.join(
    synchFunFolder,
    '{}_{}_synchFun.pickle'.format(experimentName, ns5FileName))
if os.path.exists(synchFunPath):
    print('Synch already performed. Turning off plotting')
    arguments['plotting'] = False
print('Detecting NSP Timestamps...')

tapDetectOptsINS = expOpts['synchInfo']['ins'][blockIdx]
defaultSessionTapRangesNSP = {
    'keepIndex': slice(None)
    }
tapDetectOptsNSP = expOpts['synchInfo']['nsp'][blockIdx]
####
nspChannelName = tapDetectOptsNSP[0]['synchChanName']
interTriggerInterval = tapDetectOptsNSP[0]['iti']
nspThresh = tapDetectOptsNSP[0]['thres']
minAnalogValue = tapDetectOptsNSP[0]['minAnalogValue']  # mV (determined empirically)
segIdx = 0
nspSeg = nspBlock.segments[segIdx]
nspSyncAsig = nspSeg.filter(name='seg0_{}'.format(nspChannelName))[0]
if isinstance(nspSyncAsig, AnalogSignalProxy):
    nspSyncAsig = nspSyncAsig.load()
nspSrs = pd.Series(nspSyncAsig.magnitude.flatten())
####
nspTimeRestrictions = []
for insSessionIdx, sessionTapOpts in tapDetectOptsNSP.items():
    if sessionTapOpts['timeRanges'] is not None:
        nspTimeRestrictions.append(sessionTapOpts['timeRanges'])
if len(nspTimeRestrictions):
    restrictMask = hf.getTimeMaskFromRanges(nspSyncAsig.times.magnitude, nspTimeRestrictions)
    if restrictMask.any():
        nspSrs.loc[~restrictMask] = np.nan
        nspSrs = (
            nspSrs
            .interpolate(method='linear', limit_area='inside')
            .fillna(method='bfill').fillna(method='ffill'))
# pdb.set_trace()
nspDF = nspSrs.to_frame(name=nspChannelName)
nspDF['t'] = nspSyncAsig.times.magnitude
channelData = {
    'data': nspDF,
    't': nspDF['t']
    }
#
# nspLims = nspSrs.quantile([1e-6, 1-1e-6]).to_list()
print(
    'On trial {}, detecting NSP threshold crossings.'
    .format(blockIdx))
if arguments['usedTENSPulses']:
    if minAnalogValue is not None:
        nspSrs.loc[nspSrs <= minAnalogValue] = 0
    nspPeakIdx = hf.getTriggers(
        nspSrs, iti=interTriggerInterval, itiWiggle=.5,
        fs=float(nspSyncAsig.sampling_rate), plotting=arguments['plotting'],
        thres=nspThresh, edgeType='rising')
else:
    # older, used mechanical taps
    interTriggerInterval = .2
    nspLims = [min(nspSrs), max(nspSrs)]
    nspThresh = nspLims[0] + (nspLims[-1] - nspLims[0]) / 2
    nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
        nspSrs, thresh=nspThresh,
        iti=interTriggerInterval, fs=float(nspSyncAsig.sampling_rate),
        edgeType='both', itiWiggle=.2,
        absVal=False, plotting=arguments['plotting'], keep_max=False)
#
defaultTapDetectOpts = {
    'iti': interTriggerInterval,
    'keepIndex': slice(None)
    }
allNSPTapTimes = nspDF.loc[nspPeakIdx, 't'].to_numpy()
allTapTimestampsNSP = []
allTapTimestampsINS_coarse = []
#  Detect INS taps
############################################################
if isinstance(insBlock.annotations['jsonSessionNames'], str):
    insBlock.annotations['jsonSessionNames'] = [insBlock.annotations['jsonSessionNames']]
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
for insSession, group in approxTapTimes.groupby('tapGroup'):
    tapTrainDur = np.ceil(group['NSP'].max() - group['NSP'].min())
    firstNSPTap = group['NSP'].min()
    firstINSTap = group['INS'].min()
    lookAroundBack = 1.5
    lookAroundFwd = 1.5
    approxTapTimes.loc[group.index, 'tStart_NSP'] = firstNSPTap - lookAroundBack
    approxTapTimes.loc[group.index, 'tStop_NSP'] = firstNSPTap + tapTrainDur + lookAroundFwd
    approxTapTimes.loc[group.index, 'tStart_INS'] = firstINSTap - lookAroundBack
    approxTapTimes.loc[group.index, 'tStop_INS'] = firstINSTap + tapTrainDur + lookAroundFwd
    autoTimeRanges['NSP'].append((firstNSPTap - lookAroundBack, firstNSPTap + tapTrainDur + lookAroundFwd))
    autoTimeRanges['INS'].append((firstINSTap - lookAroundBack, firstINSTap + tapTrainDur + lookAroundFwd))
    tapTimestampsNSP = group['NSP'].astype(float)
    tapTimestampsINS_coarse = group['INS'].astype(float)
    allTapTimestampsNSP.append(tapTimestampsNSP)
    allTapTimestampsINS_coarse.append(tapTimestampsINS_coarse)
    print('tSeg {}: NSP Taps:\n{}'.format(
        insSession, tapTimestampsNSP))
    print('diff:\n{}'.format(
        np.diff(tapTimestampsNSP) * 1e3))
approxTimesPath = os.path.join(
    scratchFolder,
    '{}_{}_approximateTimes.html'.format(experimentName, ns5FileName))
approxTapTimes.to_html(approxTimesPath)
###########
#
for insSession in tapDetectOptsINS.keys():
    if tapDetectOptsINS[insSession]['timeRanges'] is None:
        tapDetectOptsINS[insSession]['timeRanges'] = [autoTimeRanges['INS'][insSession]]
    for key in defaultTapDetectOpts.keys():
        if key not in tapDetectOptsINS[insSession].keys():
            tapDetectOptsINS[insSession].update(
                {key: defaultTapDetectOpts[key]}
                )
###
for insSession in tapDetectOptsNSP.keys():
    if tapDetectOptsNSP[insSession]['timeRanges'] is None:
        tapDetectOptsNSP[insSession]['timeRanges'] = autoTimeRanges['NSP'][insSession]
    for key in defaultSessionTapRangesNSP.keys():
        if key not in tapDetectOptsNSP[insSession].keys():
            tapDetectOptsNSP[insSession].update(
                {key: defaultSessionTapRangesNSP[key]}
                )
#  make placeholders for interpolation functions
interpFunINStoNSP = {
    key: [None for i in value.keys()]
    for key, value in tapDetectOptsNSP.items()
    }
tdDF, accelDF, stimStatus = mdt.unpackINSBlock(insBlock)
td = {'data': tdDF, 't': tdDF['t']}
accel = {'data': accelDF, 't': accelDF['t']}
#  Detect INS taps
############################################################
# warnings.filterwarnings('error')
################################################################
td['data'].loc[:, 'NSPTime'] = np.nan
accel['data'].loc[:, 'NSPTime'] = np.nan
allTapTimestampsINSAligned = []
allTapTimestampsINS = []
if not os.path.exists(synchFunPath):
    print('Detecting INS Timestamps...')
    try:
        overrideSegments = overrideSegmentsForTapSync[blockIdx]
    except Exception:
        overrideSegments = {}
    clickDict = {
        i: {
            'ins': [],
            'nsp': []
            }
        for i in pd.unique(td['data']['trialSegment'])}
    for insSession in pd.unique(td['data']['trialSegment']).astype(int):
        print('detecting INS taps on trial segment {}\n'.format(insSession))
        accelGroupMask = accel['data']['trialSegment'] == insSession
        accelGroup = accel['data'].loc[accelGroupMask, :]
        tdGroupMask = td['data']['trialSegment'] == insSession
        tdGroup = td['data'].loc[tdGroupMask, :]
        tapTimestampsINS, peakIdx, tapDetectSignal = mdt.getINSTapTimestamp(
            tdGroup, accelGroup, tapDetectOptsINS[insSession],
            filterOpts=tapDetectFilterOpts,
            plotting=False
            )
        print('tSeg {}, INS Taps:\n{}'.format(
            insSession, tapTimestampsINS))
        print('diff:\n{}'.format(
            tapTimestampsINS.diff() * 1e3))
        allTapTimestampsINS.append(tapTimestampsINS)
        keepIdx = tapDetectOptsNSP[insSession]['keepIndex']
        allTapTimestampsNSP[insSession] = (
            allTapTimestampsNSP[insSession]
            .iloc[keepIdx])
        #
        if arguments['curateManually']:
            try:
                clickDict[insSession] = mdt.peekAtTaps(
                    tdGroup, accelGroup, tapDetectSignal,
                    channelData, insSession,
                    tapDetectOptsINS, tapDetectOptsNSP,
                    insX='t', plotBlocking=plotBlocking,
                    nspChannelName=nspChannelName,
                    allTapTimestampsINS=allTapTimestampsINS,
                    allTapTimestampsNSP=allTapTimestampsNSP,
                    interTrigInterval=interTriggerInterval
                    )
            except Exception:
                traceback.print_exc()
        # if overriding with manually identified points
        if len(clickDict[insSession]['ins']):
            allTapTimestampsINS[insSession] = clickDict[insSession]['ins']
        if len(clickDict[insSession]['nsp']):
            allTapTimestampsNSP[insSession] = clickDict[insSession]['nsp']
    # calculate the time interpolating functions and apply them
    ############################################################
    for insSession in pd.unique(td['data']['trialSegment']).astype(int):
        accelGroupMask = accel['data']['trialSegment'] == insSession
        accelGroup = accel['data'].loc[accelGroupMask, :].copy()
        tdGroupMask = td['data']['trialSegment'] == insSession
        tdGroup = td['data'].loc[tdGroupMask, :].copy()
        # if overriding with manually identified points
        if not (insSession in overrideSegments.keys()):
            theseTapTimestampsINS = allTapTimestampsINS[insSession]
            theseTapTimestampsNSP = allTapTimestampsNSP[insSession]
        else:
            print('\t Overriding insSession {}'.format(insSession))
            if overrideSegments[insSession] == 'coarse':
                theseTapTimestampsINS = allTapTimestampsINS_coarse[insSession]
                theseTapTimestampsNSP = allTapTimestampsNSP[insSession]
            else:
                theseTapTimestampsINS = allTapTimestampsINS[overrideSegments[insSession]]
                theseTapTimestampsNSP = allTapTimestampsNSP[overrideSegments[insSession]]
        # trim spikes that happened after end of NSP recording (last trial segment only)
        trimSpiketrains = (insSession == pd.unique(td['data']['trialSegment']).astype(int).max())
        tdGroup, accelGroup, insBlock, thisINStoNSP = ns5.synchronizeINStoNSP(
            tapTimestampsNSP=theseTapTimestampsNSP,
            tapTimestampsINS=theseTapTimestampsINS,
            NSPTimeRanges=(
                channelData['t'].iloc[0], channelData['t'].iloc[-1]),
            td=tdGroup, accel=accelGroup, insBlock=insBlock,
            trialSegment=insSession, degree=0, trimSpiketrains=trimSpiketrains)
        td['data'].loc[tdGroupMask, 'NSPTime'] = tdGroup['NSPTime']
        accel['data'].loc[accelGroupMask, 'NSPTime'] = accelGroup['NSPTime']
        #
        interpFunINStoNSP[insSession] = thisINStoNSP
        allTapTimestampsINSAligned.append(thisINStoNSP(
            theseTapTimestampsINS))
    with open(synchFunPath, 'wb') as f:
        pickle.dump(interpFunINStoNSP, f)
else:
    with open(synchFunPath, 'rb') as f:
        interpFunINStoNSP = pickle.load(f)
    theseInterpFun = interpFunINStoNSP
    td['data'].loc[:, 'NSPTime'] = np.nan
    accel['data'].loc[:, 'NSPTime'] = np.nan
    for insSession in pd.unique(td['data']['trialSegment']).astype(int):
        accelGroupMask = accel['data']['trialSegment'] == insSession
        accelGroup = accel['data'].loc[accelGroupMask, :].copy()
        tdGroupMask = td['data']['trialSegment'] == insSession
        tdGroup = td['data'].loc[tdGroupMask, :].copy()
        # trim spikes that happened after end of NSP recording (last ins session only)
        trimSpiketrains = (insSession == pd.unique(td['data']['trialSegment']).astype(int).max())
        tdGroup, accelGroup, insBlock, thisINStoNSP = ns5.synchronizeINStoNSP(
            precalculatedFun=theseInterpFun[insSession],
            NSPTimeRanges=(
                channelData['t'].iloc[0], channelData['t'].iloc[-1]),
            td=tdGroup, accel=accelGroup, insBlock=insBlock,
            trialSegment=insSession, degree=0, trimSpiketrains=trimSpiketrains)
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
############################################################
insBlockJustSpikes = hf.extractSignalsFromBlock(insBlock)
insSpikeTrains = insBlockJustSpikes.filter(objects=SpikeTrain)
for st in insSpikeTrains:
    if st.waveforms is None:
        st.sampling_rate = 3e4*pq.Hz
        st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
# check in case the nsp block already has INS blocks added
nspEvList = nspBlock.filter(objects=Event)
nspEvNames = [ev.name for ev in nspEvList]
if 'ins_property' in nspEvNames:
    raise Warning('INS events already in NSP block!\n\t\tWill not overwrite.')
    arguments['addToNIX'] = False
if arguments['addToNIX']:
    #  reader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
    #  nspBlock = reader.read_block(lazy=True)
    #  nspStP = nspBlock.filter(objects=SpikeTrainProxy)
    #  nspSt = [i.load(load_waveforms=True) for i in nspStP]
    #  spikeReader = neo.io.nixio_fr.NixIO(filename=os.path.join(trialFilesFrom['utah']['folderPath'], 'tdc_' + trialFilesFrom['utah']['ns5FileName'], 'tdc_' + trialFilesFrom['utah']['ns5FileName'] + '.nix'))
    #  tdcBlock = spikeReader.read_block(lazy=True)
    #  tdcStP = tdcBlock.filter(objects=SpikeTrainProxy)
    #  tdcSt = [i.load(load_waveforms=True) for i in tdcStP]
    ns5.addBlockToNIX(
        insBlockJustSpikes, neoSegIdx=[0],
        writeAsigs=False, writeSpikes=True,
        fileName=BlackrockFileName + inputNSPBlockSuffix,
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
        fileName=BlackrockFileName + inputNSPBlockSuffix,
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
        fileName=BlackrockFileName + inputNSPBlockSuffix,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
### if adding to any other blocks
if arguments['addToBlockSuffix'] is not None:
    ns5.addBlockToNIX(
        insBlockJustSpikes, neoSegIdx=[0],
        writeAsigs=False, writeSpikes=True,
        fileName=BlackrockFileName + '_' + arguments['addToBlockSuffix'],
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )