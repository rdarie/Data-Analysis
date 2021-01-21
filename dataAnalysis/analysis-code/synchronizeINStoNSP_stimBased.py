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
from datetime import datetime as dt
from datetime import timezone
import pytz
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
import neo
import elephant.pandas_bridge as elphpdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from elephant.conversion import binarize
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)
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
jsonSessionNames = jsonSessionNames[blockIdx]
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

tapDetectOptsNSP = expOpts['synchInfo']['nsp'][blockIdx]
tapDetectOptsINS = expOpts['synchInfo']['ins'][blockIdx]
#  ### INS Loading
insChanNames = []
for insSessIdx, tdo in tapDetectOptsINS.items():
    for scn in tdo['synchChanName']:
        if ('seg0_' + scn) not in insChanNames:
            insChanNames.append(('seg0_' + scn))
insDFList = {}
insBlockList = {}
for insSessIdx, jsonSessName in enumerate(jsonSessionNames):
    insPath = os.path.join(
        scratchFolder,
        jsonSessName +
        inputINSBlockSuffix +
        '.nix')
    print('Loading INS Block from: {}'.format(insPath))
    insReader, insBlock = ns5.blockFromPath(
        insPath, lazy=arguments['lazy'],
        reduceChannelIndexes=True)
    if arguments['lazy']:
        asigList = [
            asigP.load()
            for asigP in insBlock.filter(objects=AnalogSignalProxy)
            if asigP.name in insChanNames
            ]
    else:
        asigList = [
            asig
            for asig in insBlock.filter(objects=AnalogSignal)
            if asig.name in insChanNames
            ]
    insDFList[insSessIdx] = ns5.analogSignalsToDataFrame(asigList)
    insBlockList[insSessIdx] = insBlock
insSamplingRate = float(asigList[0].sampling_rate)
insDF = (
    pd.concat(insDFList, names=['trialSegment', 'index'])
    .reset_index().drop(columns=['index']))
sessionUnixTimeList = [
    int(sessionName.split('Session')[-1])
    for sessionName in jsonSessionNames
    ]
homeTZ = pytz.timezone("America/New_York")
sessionDatetimeList = [
    pd.Timestamp(sut, unit='ms', tz='America/New_York').astimezone('utc')
    for sut in sessionUnixTimeList
]
#  ##### NSP Loading
nspChanNames = []
for insSessIdx, tdo in tapDetectOptsNSP.items():
    for scn in tdo['synchChanName']:
        if ('seg0_' + scn) not in nspChanNames:
            nspChanNames.append(('seg0_' + scn))
if arguments['lazy']:
    asigList = [
        asigP.load()
        for asigP in nspBlock.filter(objects=AnalogSignalProxy)
        if asigP.name in nspChanNames
        ]
else:
    asigList = [
        asig
        for asig in nspBlock.filter(objects=AnalogSignal)
        if asig.name in nspChanNames
        ]
nspDF = ns5.analogSignalsToDataFrame(asigList)
nspSamplingRate = float(asigList[0].sampling_rate)
nspAbsStart = pd.Timestamp(nspBlock.annotations['recDatetimeStr'])
nspDF['unixTime'] = pd.TimedeltaIndex(nspDF['t'], unit='s') + nspAbsStart
manualAlignTimes = {
    insSessIdx: {
        'ins': [],
        'nsp': []
        }
    for insSessIdx, insGroup in insDF.groupby('trialSegment')}

for insSessIdx, insGroup in insDF.groupby('trialSegment'):
    insSessIdx = int(insSessIdx)
    sessTapOptsNSP = tapDetectOptsNSP[insSessIdx]
    sessTapOptsINS = tapDetectOptsINS[insSessIdx]
    #
    sessStartUnix = sessionDatetimeList[insSessIdx]
    insGroup.loc[:, 'unixTime'] = sessStartUnix + pd.TimedeltaIndex(insGroup['t'], unit='s')
    sessStopUnix = sessStartUnix + pd.Timedelta(insGroup['t'].max(), unit='s')
    nspSessMask = (nspDF['unixTime'] > sessStartUnix) & (nspDF['unixTime'] < sessStopUnix)
    #
    theseChanNamesNSP = [
        'seg0_' + scn
        for scn in sessTapOptsNSP['synchChanName']
        ]
    thisNspDF = (
        nspDF
        .loc[nspSessMask, ['t', 'unixTime']]
        .copy().reset_index(drop=True))
    nspVals = nspDF.loc[nspSessMask, theseChanNamesNSP].to_numpy()
    if sessTapOptsNSP['minAnalogValue'] is not None:
        nspVals[nspVals < sessTapOptsNSP['minAnalogValue']] = 0
    if True:
        empCov = EmpiricalCovariance().fit(nspVals)
        thisNspDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(nspVals)
    else:
        thisNspDF.loc[:, 'tapDetectSignal'] = np.mean(nspVals, axis=1)
    #
    if sessTapOptsNSP['timeRanges'] is not None:
        restrictMask = hf.getTimeMaskFromRanges(
            thisNspDF['t'], sessTapOptsNSP['timeRanges'])
        if restrictMask.any():
            thisNspDF.loc[~restrictMask, 'tapDetectSignal'] = np.nan
            thisNspDF.loc[:, 'tapDetectSignal'] = (
                thisNspDF.loc[:, 'tapDetectSignal']
                .interpolate(method='linear', limit_area='inside')
                .fillna(method='bfill').fillna(method='ffill'))
    #
    if True:
        nspPeakIdx = hf.getTriggers(
            thisNspDF['tapDetectSignal'], iti=sessTapOptsNSP['iti'], itiWiggle=.5,
            fs=nspSamplingRate, plotting=arguments['plotting'],
            thres=sessTapOptsNSP['thres'], edgeType='rising')
    else:
        nspPeakIdx, _ = hf.getThresholdCrossings(
            thisNspDF['tapDetectSignal'], thresh=sessTapOptsNSP['thres'],
            iti=sessTapOptsNSP['iti'], fs=nspSamplingRate,
            edgeType='both', itiWiggle=.2,
            absVal=False, plotting=arguments['plotting'], keep_max=False)
    nspTapTimes = thisNspDF.loc[nspPeakIdx, 't'].to_numpy()[sessTapOptsNSP['keepIndex']]
    searchRadius = 1.5
    nspSearchMask = (
        (thisNspDF['t'] >= nspTapTimes.min() - searchRadius) &
        (thisNspDF['t'] < nspTapTimes.max() + searchRadius)
        )
    nspSearchLims = thisNspDF.loc[nspSearchMask, 't'].quantile([0, 1])
    #
    thisInsDF = insGroup.loc[:, ['t', 'unixTime']].copy().reset_index(drop=True)
    #
    unixDerivedTimeDelta = thisNspDF['t'].iloc[0] - thisInsDF['t'].iloc[0]
    #
    theseChanNamesINS = [
        'seg0_' + scn
        for scn in sessTapOptsINS['synchChanName']
        ]
    insVals = insGroup.loc[:, theseChanNamesINS].to_numpy()
    if sessTapOptsINS['minAnalogValue'] is not None:
        insVals[insVals < sessTapOptsINS['minAnalogValue']] = 0
    if True:
        empCov = EmpiricalCovariance().fit(insVals)
        thisInsDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(insVals)
    else:
        thisInsDF.loc[:, 'tapDetectSignal'] = np.mean(insVals, axis=1)
    #
    if sessTapOptsINS['timeRanges'] is not None:
        insSearchMask = hf.getTimeMaskFromRanges(
            thisInsDF['t'], sessTapOptsINS['timeRanges'])
    else:
        nspSearchUnixTimes = thisNspDF.loc[nspSearchMask, 'unixTime']
        insSearchMask = (thisInsDF['unixTime'] >= nspSearchUnixTimes.min()) & (thisInsDF['unixTime'] < nspSearchUnixTimes.max())
    if insSearchMask.any():
        thisInsDF.loc[~insSearchMask, 'tapDetectSignal'] = np.nan
        thisInsDF.loc[:, 'tapDetectSignal'] = (
            thisInsDF.loc[:, 'tapDetectSignal']
            .interpolate(method='linear', limit_area='inside')
            .fillna(method='bfill').fillna(method='ffill'))
    insPeakIdx = hf.getTriggers(
        thisInsDF['tapDetectSignal'], iti=sessTapOptsINS['iti'], itiWiggle=.5,
        fs=insSamplingRate, plotting=arguments['plotting'], keep_max=True,
        thres=sessTapOptsINS['thres'], edgeType='rising')
    insTapTimes = thisInsDF.loc[insPeakIdx, 't'].to_numpy()[sessTapOptsINS['keepIndex']]
    if arguments['curateManually']:
        try:
            manualAlignTimes[insSessIdx], fig, ax = mdt.peekAtTapsV2(
                thisNspDF, thisInsDF,
                insAuxDataDF=insGroup.drop(columns=['t', 'unixTime', 'trialSegment']),
                plotMaskNSP=nspSearchMask, plotMaskINS=insSearchMask,
                tapTimestampsINS=insTapTimes, tapTimestampsNSP=nspTapTimes,
                tapDetectOptsNSP=sessTapOptsNSP, tapDetectOptsINS=sessTapOptsINS
                )
            plt.show()
        except Exception:
            traceback.print_exc()
    alignByXCorr = True
    if alignByXCorr:
        trigRasterSamplingRate = min(nspSamplingRate, insSamplingRate)
        trigSampleInterval = trigRasterSamplingRate ** (-1)
        #
        nspSearchDur = nspSearchLims[1] - nspSearchLims[0]
        trigRasterT = nspSearchLims[0] + np.arange(0, nspSearchDur, trigSampleInterval)
        trigRaster = pd.DataFrame({
            't': trigRasterT,
            'nspDiracDelta': np.zeros_like(trigRasterT),
            'insDiracDelta': np.zeros_like(trigRasterT),
            'nspTrigs': np.zeros_like(trigRasterT),
            'insTrigs': np.zeros_like(trigRasterT),
            })
        # closestTimes, closestIdx = hf.closestSeries(
        #     takeFrom=pd.Series(nspTapTimes), compareTo=trigRaster['t']
        #     )
        # trigRaster.loc[closestIdx, 'nspDiracDelta'] = 1
        nspDiracSt = SpikeTrain(
            times=nspTapTimes, units='s',
            t_start=trigRaster['t'].min() * pq.s,
            t_stop=trigRaster['t'].max() * pq.s)
        trigRaster['nspDiracDelta'] = binarize(
            nspDiracSt, sampling_rate=trigRasterSamplingRate * pq.Hz,
            t_start=trigRaster['t'].min() * pq.s, t_stop=trigRaster['t'].max() * pq.s
            )
        trigRaster.loc[:, 'nspTrigs'] = hf.gaussianSupport(
            support=trigRaster.set_index('t')['nspDiracDelta'], gaussWid=5e-3, fs=trigRasterSamplingRate).to_numpy()
        if True:
            insBlockJustSpikes = hf.extractSignalsFromBlock(insBlockList[insSessIdx], keepEvents=False)
            insBlockJustSpikes = hf.loadBlockProxyObjects(insBlockJustSpikes)
            spikeMatBlock = ns5.calcBinarizedArray(
                insBlockJustSpikes, trigRasterSamplingRate * pq.Hz,
                saveToFile=False)
            spikeMatDF = ns5.analogSignalsToDataFrame(spikeMatBlock.filter(objects=AnalogSignal))
            spikeMatDF.loc[:, 'nspT'] = spikeMatDF['t'] + unixDerivedTimeDelta
            # spikeMatDF is in ins time
            spikeMatMask = (spikeMatDF['nspT'] >= nspSearchLims[0]) & (spikeMatDF['nspT'] <= nspSearchLims[1])
            trigRaster.loc[:, 'insDiracDelta'] = (
                spikeMatDF.loc[spikeMatMask, :]
                .drop(columns=['t', 'nspT'])
                .any(axis='columns').to_numpy(dtype=np.float))
            # trigRaster['insDiracDelta']
        else:
            approxInsTapTimes = insTapTimes + unixDerivedTimeDelta
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=pd.Series(approxInsTapTimes), compareTo=trigRaster['t']
                )
            trigRaster.loc[closestIdx, 'insDiracDelta'] = 1
        trigRaster.loc[:, 'insTrigs'] = hf.gaussianSupport(
            support=trigRaster.set_index('t')['insDiracDelta'],
            gaussWid=5e-3, fs=trigRasterSamplingRate).to_numpy()
        xCorr = np.correlate(trigRaster['nspTrigs'], trigRaster['insTrigs'], mode='full')
        posLags = np.arange(0, nspSearchDur, trigSampleInterval)
        xCorrLags = np.concatenate([(-1) * np.flip(posLags[1:]), posLags])
        xCorrSrs = pd.Series(xCorr, index=xCorrLags)
        maxLag = xCorrSrs.idxmax()
        if True:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(trigRaster['t'], trigRaster['nspTrigs'], label='NSP trigs.')
            ax[0].plot(trigRaster['t'], trigRaster['insTrigs'], label='INS trigs.')
            #
            ax[1].plot(xCorrSrs, label='crossCorr')
        ax[0].legend()
        ax[1].legend()
        plt.show()
        pdb.set_trace()
        corrLags = 0
#
#
# get absolute timestamps of file extents (by INS session)
# INS session name -> absolute timestamp + time domain data last time for the end
#
# if using cross correlation approach, calculate binarized vector of 1's and zeros representing times of stim
# optionally, scale the 1's by the amplitude of the pulses
#
#
# nix Block rec_datetime -> absolute timestamp + td data last time for the end
#
# assign trial segments to the NSP data
# iterate through **trial segments**
# # # find coarse time offset based on absolute times
# # # apply coarse offset to INS data stream
# # # 
# # # if NSP data series has a handwritten time range, zero it outside that range
# # # detect threshold crossings from NSP data series (can be stim artifact, or TENS artifact or whatever)
# # #
# # #
