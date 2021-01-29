"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --blockIdx=blockIdx                                   which trial to analyze
    --exp=exp                                             which experimental day to analyze
    --inputNSPBlockSuffix=inputNSPBlockSuffix             append a name to the input block?
    --inputINSBlockSuffix=inputINSBlockSuffix             append a name to the input block?
    --lazy                                                whether to fully load data from blocks [default: True]
    --showFigures                                         whether to fully load data from blocks [default: False]
    --addToBlockSuffix=addToBlockSuffix                   whether to also add stim info to the high pass filtered NSP blocks
    --addToNIX                                            whether to interpolate and add the ins data to the nix file [default: False]
    --curateManually                                      whether to manually confirm synch [default: False]
    --preparationStage                                    get aproximate timings and print to shell, to help identify time ranges? [default: False]
    --plotting                                            whether to display confirmation plots [default: False]
    --forceRecalc                                         whether to overwrite any previous calculated synch [default: False]
    --usedTENSPulses                                      whether the sync was done using TENS pulses (as opposed to mechanical taps) [default: False]
"""
import matplotlib, pdb, traceback
matplotlib.use('Qt5Agg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
#  matplotlib.use('PS')   # noninteract output
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import dill as pickle
from scipy import signal
from importlib import reload
import peakutils
import shutil
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
from statsmodels import robust
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
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.7, color_codes=True)
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

searchRadius = [-5., 5.]
searchRadiusUnix = [
    pd.Timedelta(searchRadius[0], unit='s'),
    pd.Timedelta(searchRadius[1], unit='s')]
#
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
    '{}_{}_synchFun.json'.format(experimentName, ns5FileName))
#
tapDetectOptsNSP = expOpts['synchInfo']['nsp'][blockIdx]
tapDetectOptsINS = expOpts['synchInfo']['ins'][blockIdx]
#
insSignalsToSave = ([
        'seg0_ins_td{}'.format(tdIdx)
        for tdIdx in range(4)] +
    [
        'seg0_ins_acc{}'.format(accIdx)
        for accIdx in ['x', 'y', 'z', 'inertia']
        ]
    )
insEventsToSave = [
    'seg0_ins_property',
    'seg0_ins_value'
    ]
insSpikeTrainsToSave = []
for grpIdx in range(4):
    for prgIdx in range(4):
        insSpikeTrainsToSave.append('seg0_g{}p{}#0'.format(grpIdx, prgIdx))

interpFunINStoNSP = {}
interpFunNSPtoINS = {}

#  load INS Data
############################################################
insAsigNames = insSignalsToSave.copy()
for insSessIdx, tdo in tapDetectOptsINS.items():
    for scn in tdo['synchChanName']:
        if ('seg0_' + scn) not in insAsigNames:
            insAsigNames.append(('seg0_' + scn))
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
        insPath,
        lazy=arguments['lazy'],
        # lazy=False,
        reduceChannelIndexes=True,
        loadList={
            'asigs': insAsigNames,
            'events': insEventsToSave,
            'spiketrains': insSpikeTrainsToSave
        })
    asigList = [
        asig
        for asig in insBlock.filter(objects=AnalogSignal)
        if asig.name in insAsigNames
        ]
    evList = insBlock.filter(objects=Event)
    stList = insBlock.filter(objects=SpikeTrain)
    #
    evNames = [ev.name for ev in evList]
    insEventsToSave = [
        evN for evN in insEventsToSave if evN in evNames]
    stNames = [st.name for st in stList]
    insSpikeTrainsToSave = [
        stN for stN in insSpikeTrainsToSave if stN in stNames]
    #
    presentAsigNames = [asig.name for asig in asigList]
    insAsigNames = [cN for cN in insAsigNames if cN in presentAsigNames]
    insSignalsToSave = [cN for cN in insSignalsToSave if cN in presentAsigNames]
    #
    insDFList[insSessIdx] = ns5.analogSignalsToDataFrame(asigList)
    insBlockList[insSessIdx] = insBlock
insSamplingRate = float(asigList[0].sampling_rate)
insDF = (
    pd.concat(insDFList, names=['trialSegment', 'index'])
    .reset_index().drop(columns=['index']))
insDF.loc[:, 'trialSegment'] = insDF['trialSegment'].astype(int)
#

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

if os.path.exists(synchFunPath) and not arguments['forceRecalc']:
    with open(synchFunPath, 'r') as _f:
        interpFunLoaded = json.load(_f)
    interpFunINStoNSP = {
        int(key): np.poly1d(value)
        for key, value in interpFunLoaded['insToNsp'].items()
    }
    interpFunNSPtoINS = {
        int(key): np.poly1d(value)
        for key, value in interpFunLoaded['nspToIns'].items()
    }
else:
    sessionUnixTimeList = [
        int(sessionName.split('Session')[-1])
        for sessionName in jsonSessionNames
        ]
    homeTZ = pytz.timezone("America/New_York")
    sessionDatetimeList = [
        pd.Timestamp(sut, unit='ms', tz='America/New_York').astimezone('utc')
        for sut in sessionUnixTimeList
    ]
    nspAbsStart = pd.Timestamp(nspBlock.annotations['recDatetimeStr'])
    nspDF['unixTime'] = pd.TimedeltaIndex(nspDF['t'], unit='s') + nspAbsStart
    if tapDetectOptsNSP[0]['unixTimeAdjust'] is not None:
        nspDF.loc[:, 'unixTime'] = nspDF['unixTime'] + pd.Timedelta(tapDetectOptsNSP[0]['unixTimeAdjust'], unit='S')
    manualAlignTimes = {
        insSessIdx: {
            'ins': [],
            'nsp': []
            }
        for insSessIdx, insGroup in insDF.groupby('trialSegment')}
    #
    alignByXCorr = True
    alignByUnixTime = False
    getINSTrigTimes = False
    #
    plotSynchReport = True
    if plotSynchReport and alignByXCorr:
        insDiagnosticsFolder = os.path.join(figureFolder, 'insDiagnostics')
        if not os.path.exists(insDiagnosticsFolder):
            os.mkdirs(insDiagnosticsFolder)
        synchReportPDF = PdfPages(
            os.path.join(
                insDiagnosticsFolder,
                'ins_synch_report_Block{:0>3}.pdf'.format(blockIdx)))
    for insSessIdx, insGroup in insDF.groupby('trialSegment'):
        print('aligning session nb {}'.format(insSessIdx))
        sessTapOptsNSP = tapDetectOptsNSP[insSessIdx]
        sessTapOptsINS = tapDetectOptsINS[insSessIdx]
        theseChanNamesNSP = [
            'seg0_' + scn
            for scn in sessTapOptsNSP['synchChanName']
            ]
        #
        searchLimsUnixStartList = []
        searchLimsUnixStopList = []
        #
        sessStartUnix = sessionDatetimeList[insSessIdx]
        if sessTapOptsINS['unixTimeAdjust'] is not None:
            print('Adding manual adjustment to INS ({} sec)'.format(sessTapOptsINS['unixTimeAdjust']))
            sessStartUnix += pd.Timedelta(sessTapOptsINS['unixTimeAdjust'], unit='S')
        searchLimsUnixStartList.append(sessStartUnix)
        insGroup.loc[:, 'unixTime'] = sessStartUnix + pd.TimedeltaIndex(insGroup['t'], unit='s')
        sessStopUnix = sessStartUnix + pd.Timedelta(insGroup['t'].max(), unit='s')
        searchLimsUnixStopList.append(sessStopUnix)
        #
        if sessTapOptsINS['timeRanges'] is not None:
            insTimeRangesMask = hf.getTimeMaskFromRanges(
                insGroup['t'], sessTapOptsINS['timeRanges'])
            maskedInsUnixTimes = insGroup.loc[insTimeRangesMask, 'unixTime']
            searchLimsUnixStartList.append(maskedInsUnixTimes.min())
            searchLimsUnixStopList.append(maskedInsUnixTimes.max())
        if sessTapOptsNSP['timeRanges'] is not None:
            nspTimeRangesMask = hf.getTimeMaskFromRanges(
                nspDF['t'], sessTapOptsNSP['timeRanges'])
            maskedNspUnixTimes = insGroup.loc[insTimeRangesMask, 'unixTime']
            searchLimsUnixStartList.append(maskedNspUnixTimes.min())
            searchLimsUnixStopList.append(maskedNspUnixTimes.max())
        searchLimsUnix = {
            0: max(searchLimsUnixStartList),
            1: min(searchLimsUnixStopList)
            }
        '''
        else:
            nspSearchUnixTimes = thisNspDF.loc[nspSearchMask, 'unixTime']
            insSearchMask = (
                (thisInsDF['unixTime'] >= searchLimsUnix[0]) &
                (thisInsDF['unixTime'] < searchLimsUnix[1])
                )
        '''
        insSearchMask = (insGroup['unixTime'] >= searchLimsUnix[0]) & (insGroup['unixTime'] < searchLimsUnix[1])
        assert insSearchMask.any()
        #
        thisInsDF = (
            insGroup
            .loc[insSearchMask, ['t', 'unixTime']]
            .copy().reset_index(drop=True))
        # insSearchLims = thisInsDF.loc[:, 't'].quantile([0, 1])
        nspSearchMask = (nspDF['unixTime'] >= searchLimsUnix[0]) & (nspDF['unixTime'] < searchLimsUnix[1])
        assert nspSearchMask.any()
        nspGroup = (
            nspDF
            .loc[nspSearchMask, :])
        thisNspDF = (
            nspGroup
            .loc[:, ['t', 'unixTime']]
            .copy().reset_index(drop=True)
        )
        nspSearchLims = nspGroup.loc[:, 't'].quantile([0, 1])
        #
        unixDeltaT = thisNspDF['t'].iloc[0] - thisInsDF['t'].iloc[0]
        print('    delta T is approx {}'.format(unixDeltaT))
        #
        nspVals = nspGroup.loc[:, theseChanNamesNSP].to_numpy()
        filterOpts = None
        '''
        filterOpts = {
            'high': {
                'Wn': 1000,
                'N': 4,
                'btype': 'high',
                'ftype': 'bessel'
                }}
        '''
        if filterOpts is not None:
            print('Filtering NSP traces...')
            filterCoeffs = hf.makeFilterCoeffsSOS(
                filterOpts, nspSamplingRate)
            nspVals = signal.sosfiltfilt(
                filterCoeffs,
                nspVals, axis=0)
        if sessTapOptsNSP['minAnalogValue'] is not None:
            nspVals[nspVals < sessTapOptsNSP['minAnalogValue']] = 0
        if False:
            empCov = EmpiricalCovariance().fit(nspVals)
            thisNspDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(nspVals)
        else:
            thisNspDF.loc[:, 'tapDetectSignal'] = np.mean(nspVals, axis=1)
        '''
        if sessTapOptsNSP['timeRanges'] is not None:
            restrictMaskNSP = hf.getTimeMaskFromRanges(
                thisNspDF['t'], sessTapOptsNSP['timeRanges'])
            if restrictMaskNSP.any():
                thisNspDF.loc[~restrictMaskNSP, 'tapDetectSignal'] = np.nan
                thisNspDF.loc[:, 'tapDetectSignal'] = (
                    thisNspDF.loc[:, 'tapDetectSignal']
                    .interpolate(method='linear', limit_area='inside')
                    .fillna(method='bfill').fillna(method='ffill'))
        else:
            restrictMaskNSP = pd.Series(True, index=thisNspDF.index)
        '''
        nspTrigFinder = 'getThresholdCrossings'
        if nspTrigFinder == 'getTriggers':
            nspPeakIdx = hf.getTriggers(
                thisNspDF['tapDetectSignal'], iti=sessTapOptsNSP['iti'], itiWiggle=.2,
                fs=nspSamplingRate, plotting=arguments['plotting'], absVal=False,
                thres=sessTapOptsNSP['thres'], edgeType='rising', keep_max=True)
        elif nspTrigFinder == 'getThresholdCrossings':
            nspPeakIdx, _ = hf.getThresholdCrossings(
                thisNspDF['tapDetectSignal'], thresh=sessTapOptsNSP['thres'],
                iti=sessTapOptsNSP['iti'], fs=nspSamplingRate,
                edgeType='rising', itiWiggle=.2,
                absVal=False, plotting=arguments['plotting'])
        nspPeakIdx = nspPeakIdx[sessTapOptsNSP['keepIndex']]
        nspTapTimes = thisNspDF.loc[nspPeakIdx, 't'].to_numpy()
        print('nspTapTimes: {}'.format(nspTapTimes))
        # nspTapTimesUnix = thisNspDF.loc[nspPeakIdx, 'unixTime'].to_numpy()
        #
        # nspSearchMask = (
        #     (thisNspDF['unixTime'] >= searchLimsUnix[0]) &
        #     (thisNspDF['unixTime'] < searchLimsUnix[1])
        #     )
        # nspSearchLims = thisNspDF.loc[nspSearchMask, 't'].quantile([0, 1])
        #
        # thisInsDF = insGroup.loc[:, ['t', 'unixTime']].copy().reset_index(drop=True)
        #
        theseChanNamesINS = [
            'seg0_' + scn
            for scn in sessTapOptsINS['synchChanName']
            ]
        insVals = insGroup.loc[insSearchMask, theseChanNamesINS].to_numpy()
        if sessTapOptsINS['minAnalogValue'] is not None:
            insVals[insVals < sessTapOptsINS['minAnalogValue']] = 0
        if True:
            empCov = EmpiricalCovariance().fit(insVals)
            thisInsDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(insVals)
        else:
            thisInsDF.loc[:, 'tapDetectSignal'] = np.mean(insVals, axis=1)
        #
        if getINSTrigTimes:
            '''
            if insSearchMask.any():
                thisInsDF.loc[~insSearchMask, 'tapDetectSignal'] = np.nan
                thisInsDF.loc[:, 'tapDetectSignal'] = (
                    thisInsDF.loc[:, 'tapDetectSignal']
                    .interpolate(method='linear', limit_area='inside')
                    .fillna(method='bfill').fillna(method='ffill'))
            '''
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
                        # plotMaskNSP=nspSearchMask, plotMaskINS=insSearchMask,
                        tapTimestampsINS=insTapTimes, tapTimestampsNSP=nspTapTimes,
                        tapDetectOptsNSP=sessTapOptsNSP, tapDetectOptsINS=sessTapOptsINS
                        )
                    plt.show()
                except Exception:
                    traceback.print_exc()
            insTapTimesStList = SpikeTrain(
                insTapTimes, t_start=thisInsDF['t'].min() * pq.s,
                t_stop=thisInsDF['t'].max() * pq.s,
                units=pq.s, name='tap times')
        else:
            insTapTimes = None
            insTapTimesStList = [
                st.copy()
                for st in insBlockList[insSessIdx].filter(objects=SpikeTrain)
            ]
        #
        if alignByXCorr:
            if sessTapOptsINS['xCorrSamplingRate'] is not None:
                trigRasterSamplingRate = sessTapOptsINS['xCorrSamplingRate']
            else:
                trigRasterSamplingRate = min(nspSamplingRate, insSamplingRate)
            # trigRasterSamplingRate = 1000 #Hz
            trigSampleInterval = trigRasterSamplingRate ** (-1)
            if sessTapOptsINS['xCorrGaussWid'] is not None:
                gaussWid = sessTapOptsINS['xCorrGaussWid']
            else:
                gaussWid = 10e-3
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
            if sessTapOptsINS['synchStimUnitName'] is not None:
                # cross corr stim timestamps
                coarseSpikeMats = []
                for coarseSt in insTapTimesStList:
                    if len(coarseSt.times) > 0:
                        coarseSt.t_start = coarseSt.t_start + unixDeltaT * coarseSt.t_start.units
                        coarseSt.t_stop = coarseSt.t_stop + unixDeltaT * coarseSt.t_stop.units
                        newStT = coarseSt.times.magnitude + unixDeltaT
                        coarseSt.magnitude[:] = newStT
                        if 'amplitude' in coarseSt.array_annotations:
                            theseAmps = coarseSt.array_annotations['amplitude']
                            ampMask = theseAmps > sessTapOptsINS['minStimAmp']
                            spikesToBinarize = coarseSt.times[theseAmps > sessTapOptsINS['minStimAmp']]
                        else:
                            spikesToBinarize = coarseSt.times
                        thisSpikeMat = binarize(
                            spikesToBinarize,
                            sampling_rate=trigRasterSamplingRate * pq.Hz,
                            t_start=trigRaster['t'].min() * pq.s,
                            t_stop=trigRaster['t'].max() * pq.s)
                        coarseSpikeMats.append(thisSpikeMat[:, np.newaxis])
                        print('{}: coarseSt.times = {}'.format(coarseSt.name, coarseSt.times))
                        # print('st.times = {}'.format(st.times))
                if not len(coarseSpikeMats) > 0:
                    print('\n\n INS {} has no stim spikes!'.format(jsonSessionNames[insSessIdx]))
                    print('Defaulting to xcorr analog signals\n\n')
                    sessTapOptsINS['synchByXCorrTapDetectSignal'] = True
                    sessTapOptsNSP['synchByXCorrTapDetectSignal'] = True
                else:
                    trigRaster.loc[:, 'insDiracDelta'] = np.concatenate(
                        coarseSpikeMats, axis=1).any(axis=1)
                    debugDelta = trigRaster.loc[:, 'insDiracDelta'].unique()
                    print("trigRaster.loc[:, 'insDiracDelta'].unique() = {}".format(debugDelta))
                    #  if debugDelta.size > 2:
                    #      pdb.set_trace()
                    trigRaster.loc[:, 'insTrigs'] = hf.gaussianSupport(
                        support=trigRaster.set_index('t')['insDiracDelta'],
                        gaussWid=gaussWid, fs=trigRasterSamplingRate).to_numpy()
            elif sessTapOptsINS['synchByXCorrTapDetectSignal']:
                xcorrINS = thisInsDF.copy()
                xcorrINS.loc[:, 'coarseNspTime'] = maskedINS['t'] + unixDeltaT
                trigRaster.loc[:, 'insTrigs'] = hf.interpolateDF(
                    xcorrINS, trigRaster['t'], x='coarseNspTime',
                    columns=['tapDetectSignal'],
                    kind='linear', fill_value=(0, 0))
            else:
                approxInsTapTimes = insTapTimes + unixDeltaT
                closestTimes, closestIdx = hf.closestSeries(
                    takeFrom=pd.Series(approxInsTapTimes), compareTo=trigRaster['t']
                    )
                # TO DO: FINISH
                trigRaster.loc[closestIdx, 'insDiracDelta'] = 1
                trigRaster.loc[:, 'insTrigs'] = hf.gaussianSupport(
                    support=trigRaster.set_index('t')['insDiracDelta'],
                    gaussWid=gaussWid, fs=trigRasterSamplingRate).to_numpy()
            if sessTapOptsNSP['synchByXCorrTapDetectSignal']:
                trigRaster.loc[:, 'nspTrigs'] = hf.interpolateDF(
                    thisNspDF, trigRaster['t'], x='t',
                    columns=['tapDetectSignal'],
                    kind='linear', fill_value=(0, 0))
            else:
                nspDiracSt = SpikeTrain(
                    times=nspTapTimes, units='s',
                    t_start=trigRaster['t'].min() * pq.s,
                    t_stop=trigRaster['t'].max() * pq.s)
                trigRaster.loc[:, 'nspDiracDelta'] = binarize(
                    nspDiracSt, sampling_rate=trigRasterSamplingRate * pq.Hz,
                    t_start=trigRaster['t'].min() * pq.s, t_stop=trigRaster['t'].max() * pq.s
                    )
                trigRaster.loc[:, 'nspTrigs'] = hf.gaussianSupport(
                    support=trigRaster.set_index('t')['nspDiracDelta'],
                    gaussWid=gaussWid,
                    fs=trigRasterSamplingRate).to_numpy()
            #
            def corrAtLag(targetLag, xSrs=None, ySrs=None):
                return np.correlate(xSrs, ySrs.shift(targetLag).fillna(0))[0]
            #
            targetLags = np.arange(
                searchRadius[0] * trigRasterSamplingRate,
                searchRadius[1] * trigRasterSamplingRate + 1,
                # -0.8 * trigRasterSamplingRate,
                # 0.1 * trigRasterSamplingRate + 1,
                dtype=np.int)
            targetLagsSrs = pd.Series(
                targetLags, index=targetLags * trigSampleInterval)
            zeroMask = pd.Series(False, index=trigRaster.index)
            if (targetLags > 0).any():
                zeroMask = zeroMask | (trigRaster.index < targetLags.max())
            if (targetLags < 0).any():
                maxTRI = trigRaster.index.max()
                zeroMask = zeroMask | (trigRaster.index > maxTRI + targetLags.min())
            trigRaster.loc[zeroMask, 'insTrigs'] = 0
            print('Calculating cross corr')
            xCorrSrs = targetLagsSrs.apply(
                corrAtLag, xSrs=trigRaster['nspTrigs'],
                ySrs=trigRaster['insTrigs'])
            maxLag = xCorrSrs.idxmax()
            print('Optimal lag at ({:.3f}) + ({:.3f}) = ({:.3f})'.format(
                unixDeltaT, maxLag, unixDeltaT + maxLag
                ))
            if plotSynchReport:
                def plotTheReport():
                    fig, ax = plt.subplots(3, 1)
                    fig.set_size_inches(12, 8)
                    plotT0 = trigRaster.loc[
                        (trigRaster['insTrigs'] > 0.5) & (trigRaster['nspTrigs'] > 0.5), 't']
                    if plotT0.size > 0:
                        plotT0 = plotT0.iloc[0]
                    else:
                        plotT0 = trigRaster.loc[(trigRaster['insTrigs'] > 0.5), 't']
                        if plotT0.size > 0:
                            plotT0 = plotT0.iloc[0]
                        else:
                            plotT0 = trigRaster.loc[(trigRaster['insTrigs'] > 0.5) | (trigRaster['nspTrigs'] > 0.5), 't'].iloc[0]
                    plotMask = (trigRaster['t'] >= plotT0 + searchRadius[0]) & (trigRaster['t'] < plotT0 + searchRadius[1])
                    ax[0].plot(
                        trigRaster.loc[plotMask, 't'], trigRaster.loc[plotMask, 'nspTrigs'],
                        label='NSP trigs.')
                    ax[0].plot(
                        trigRaster.loc[plotMask, 't'], trigRaster.loc[plotMask, 'insTrigs'],
                        label='INS trigs.')
                    ax[0].set_xlim(plotT0 + searchRadius[0], plotT0 + searchRadius[1])
                    ax[0].set_xlabel('NSP time (sec)')
                    ax[0].set_ylabel('Triggers')
                    ax[0].legend(loc='upper right')
                    ax[0].set_title('Block {}, insSession {}: {}'.format(blockIdx, insSessIdx, jsonSessionNames[insSessIdx]))
                    # fake legend with annotations
                    customMessages = [
                        'INS session',
                        '    lasted {:.1f} sec'.format((sessStopUnix - sessStartUnix).total_seconds()),
                        '    approx delay {:.1f} sec'.format(unixDeltaT),
                        '# of NSP trigs = {}'.format(trigRaster.loc[:, 'nspDiracDelta'].sum()),
                        '# of INS trigs = {}'.format(trigRaster.loc[:, 'insDiracDelta'].sum())
                    ]
                    customLines = [
                        Line2D([0], [0], color='k', alpha=0)
                        for custMess in customMessages
                        ]
                    phantomAx = ax[0].twinx()
                    phantomAx.set_yticks([])
                    phantomAx.legend(customLines, customMessages, loc='upper left')
                    #
                    ax[1].plot(xCorrSrs, label='crossCorr')
                    ax[1].plot(
                        maxLag, xCorrSrs.loc[maxLag],
                        'y*', label='optimal lag = {:.3f}'.format(maxLag))
                    ax[1].set_xlabel('cross-corr lag (sec)')
                    ax[1].legend(loc='upper right')
                    ax[1].set_ylabel('cross-corr')
                    ax[1].set_xlim(searchRadius[0], searchRadius[1])
                    #
                    maxShift = targetLagsSrs[maxLag]
                    shiftedInsTrig = trigRaster['insTrigs'].shift(maxShift).fillna(0)
                    shiftedInsImpulse = trigRaster['insDiracDelta'].shift(maxShift).fillna(0)
                    shiftedProduct = (
                        trigRaster['nspTrigs'] * shiftedInsTrig)
                    lPFForSP = hf.makeFilterCoeffsSOS(
                        {
                            'low': {
                                'Wn': 1,
                                'N': 4,
                                'btype': 'low',
                                'ftype': 'bessel'
                                }},
                        trigRasterSamplingRate)
                    lowPassShiftedProduct = signal.sosfiltfilt(
                        lPFForSP, shiftedProduct.to_numpy())
                    ax[2].plot(
                        trigRaster['t'],
                        trigRaster['nspTrigs'], c='tab:blue')
                    nspTargetMask = trigRaster['nspDiracDelta'] > 0
                    if nspTargetMask.any():
                        ax[2].plot(
                            trigRaster.loc[nspTargetMask, 't'],
                            trigRaster.loc[nspTargetMask, 'nspTrigs'], 'o', c='tab:purple', label='NSP impulses')
                    ax[2].plot(
                        trigRaster['t'],
                        shiftedInsTrig, c='tab:orange')
                    insTargetMask = shiftedInsImpulse > 0
                    if insTargetMask.any():
                        defaultMarkerSize = matplotlib.rcParams['lines.markersize']
                        ax[2].plot(
                            trigRaster.loc[insTargetMask, 't'],
                            shiftedInsTrig.loc[insTargetMask], 'd', c='tab:red', label='INS impulses (shifted)',
                            markersize=defaultMarkerSize / 2)
                    ax[2].plot(trigRaster['t'], shiftedProduct, c='tab:green', label='elementwise product of trigs')
                    ax[2].set_xlabel('NSP time (sec)')
                    ax[2].set_ylabel('A.U.')
                    ax[2].legend(loc='upper right')
                    prodAx = ax[2].twinx()
                    prodAx.plot(trigRaster['t'], lowPassShiftedProduct, c='tab:olive', label='elementwise product of trigs (1 hz low pass)')
                    prodAx.set_yticks([])
                    prodAx.legend(loc='lower right')
                    figSaveOpts = dict(
                        bbox_extra_artists=(
                            theAx.get_legend()
                            for theAx in ax
                            if theAx.get_legend() is not None),
                        bbox_inches='tight')
                    synchReportPDF.savefig(**figSaveOpts)
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
                    return
                plotTheReport()
            funCoeffs = np.asarray([1, unixDeltaT + maxLag])
            invFunCoeffs = np.asarray([1, -1 * (unixDeltaT + maxLag)])
        elif alignByUnixTime:
            funCoeffs = np.asarray([1, unixDeltaT])
            invFunCoeffs = np.asarray([1, -1 * (unixDeltaT)])
        else:
            # align by regressing timestamps
            # funCoeffs = np.poly1d()
            pass
        interpFunINStoNSP[insSessIdx] = np.poly1d(funCoeffs)
        interpFunNSPtoINS[insSessIdx] = np.poly1d(invFunCoeffs)
        # end getting interp function
    interpFunExport = {
        'insToNsp': {
            key: list(value)
            for key, value in interpFunINStoNSP.items()
            },
        'nspToIns': {
            key: list(value)
            for key, value in interpFunNSPtoINS.items()
            }
        }
    with open(synchFunPath, 'w') as f:
        json.dump(interpFunExport, f)
    if plotSynchReport and alignByXCorr:
        synchReportPDF.close()

# synch fun calculated, now apply
allEvs = {}
allSts = {}
nspBoundaries = nspDF['t'].quantile([0, 1])
insDFListOut = {}
for insSessIdx, insGroup in insDF.groupby('trialSegment'):
    insBlock = insBlockList[insSessIdx]
    insBoundaries = np.polyval(interpFunNSPtoINS[insSessIdx], nspBoundaries)
    #
    asigList = [
        asig
        for asig in insBlock.filter(objects=AnalogSignal)
        if asig.name in insAsigNames
        ]
    evList = [
        ns5.loadObjArrayAnn(ev)
        for ev in insBlock.filter(objects=Event)
        if ev.name in insEventsToSave]
    stList = [
        ns5.loadObjArrayAnn(st)
        for st in insBlock.filter(objects=SpikeTrain)
        if st.name in insSpikeTrainsToSave]
    #
    for ev in evList:
        if ev.name in allEvs:
            allEvs[ev.name].append(ev)
        else:
            allEvs[ev.name] = [ev]
        newEvT = np.polyval(interpFunINStoNSP[insSessIdx], ev.magnitude)
        ev.times.magnitude[:] = newEvT
    #
    for st in stList:
        if st.name in allSts:
            allSts[st.name].append(st)
        else:
            allSts[st.name] = [st]
        # st = allSts[st.name][-1]
        st.t_start = min(
            np.polyval(
                interpFunINStoNSP[insSessIdx],
                st.t_start.magnitude) * st.t_start.units,
            nspBoundaries[0] * st.t_start.units)
        st.t_stop = max(
            np.polyval(
                interpFunINStoNSP[insSessIdx],
                st.t_stop.magnitude) * st.t_start.units,
            nspBoundaries[1] * st.t_start.units)
        if len(st.times) > 0:
            newStT = np.polyval(
                interpFunINStoNSP[insSessIdx],
                st.times.magnitude)
            ##### important to set st.magnitude vs. st.times.magnitude for this to stick
            st.magnitude[:] = newStT
    asigDF = ns5.analogSignalsToDataFrame(asigList)
    asigDF.loc[:, 't'] = np.polyval(interpFunINStoNSP[insSessIdx], asigDF['t'])
    insDFListOut[insSessIdx] = asigDF
#
eventsOut = {}
for evName, evList in allEvs.items():
    eventsOut[evName] = ns5.concatenateEventsContainer(evList)
spikesOut = {}
for stName, stList in allSts.items():
    spikesOut[stName] = ns5.concatenateEventsContainer(stList)
#
insDFOut = (
    pd.concat(insDFListOut, names=['trialSegment', 'index'])
    .reset_index().drop(columns=['index']))
insBoundaries = insDFOut['t'].quantile([0,1])
outTStart = min(nspBoundaries[0], insBoundaries[0])
outTStop = max(nspBoundaries[1], insBoundaries[1])
outT = np.arange(outTStart, outTStop + insSamplingRate ** -1, insSamplingRate ** -1)
insDFOutInterp = hf.interpolateDF(
    insDFOut, outT, x='t',
    kind='linear', fill_value=(0, 0))
insDFOutInterp.columns = [cN.replace('seg0_', '') for cN in insDFOutInterp.columns]
insBlockInterp = ns5.dataFrameToAnalogSignals(
    insDFOutInterp,
    idxT='t', useColNames=True, probeName='',
    dataCol=insDFOutInterp.drop(columns='t').columns,
    samplingRate=insSamplingRate * pq.Hz)
seg = insBlockInterp.segments[0]
for evName, ev in eventsOut.items():
    ev.segment = seg
    seg.events.append(ev)
#
for stNameOut, stOut in spikesOut.items():
    stOut.segment = seg
    seg.spiketrains.append(stOut)
    stOut.unit.spiketrains = [stOut]
    if stOut.unit.channel_index not in insBlockInterp.channel_indexes:
        insBlockInterp.channel_indexes.append(stOut.unit.channel_index)
insBlockInterp.name = 'ins_data'
insBlockInterp = ns5.purgeNixAnn(insBlockInterp)
insBlockInterp.create_relationship()
outPathName = os.path.join(
    scratchFolder,
    ns5FileName + '_ins.nix')
if os.path.exists(outPathName):
    os.remove(outPathName)
writer = neo.io.NixIO(filename=outPathName)
writer.write_block(insBlockInterp, use_obj_names=True)
writer.close()
'''
saveEventsToNSPBlock = True
if saveEventsToNSPBlock:
    # if nsp block already has the spike stuff, revert from the copy; else, save a copy and add them
    nspReader.file.close()
    backupNspPath = os.path.join(
        scratchFolder,
        BlackrockFileName + inputNSPBlockSuffix +
        '_backup.nix')
    shutil.copyfile(nspPath, backupNspPath)
    ns5.addBlockToNIX(
        insBlockInterp, neoSegIdx=[0],
        writeAsigs=False, writeSpikes=True, writeEvents=True,
        fileName=BlackrockFileName + inputNSPBlockSuffix,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
'''
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