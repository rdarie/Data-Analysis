"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --blockIdx=blockIdx                                   which trial to analyze
    --exp=exp                                             which experimental day to analyze
    --inputNSPBlockSuffix=inputNSPBlockSuffix             append a name to the input block?
    --inputINSBlockSuffix=inputINSBlockSuffix             append a name to the input block?
    --outputINSBlockSuffix=outputINSBlockSuffix           append a name to the output block? [default: ins]
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
import logging
logging.captureWarnings(True)
import os, sys
import matplotlib
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.lines import Line2D
# import dill as pickle
from scipy import signal
from scipy import stats
# from importlib import reload
import peakutils
import shutil
import numpy as np
import pandas as pd
# import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
# import dataAnalysis.helperFunctions.motor_encoder_new as mea
import dataAnalysis.helperFunctions.helper_functions_new as hf
# import dataAnalysis.helperFunctions.estimateElectrodeImpedances as eti
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.preproc.mdt as mdt
# import dataAnalysis.preproc.mdt_constants as mdt_constants
# import warnings
# import h5py
import traceback
# import math as m
import seaborn as sns
# import scipy.interpolate as intrp
import quantities as pq
import json
# from statsmodels import robust
# import rcsanalysis.packetizer as rcsa
# import rcsanalysis.packet_func as rcsa_helpers
from datetime import datetime as dt
# from datetime import timezone
import pytz
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
import neo
import pdb
# import elephant as elph
# import elephant.pandas_bridge as elphpdb
from elephant.conversion import binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }

mplRCParams = {
    'figure.titlesize': 7,
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1,  # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

#  load options
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'exp': 'exp201901271000', 'showFigures': True, 'outputINSBlockSuffix': 'ins', 'plotting': False,
        'usedTENSPulses': False, 'inputINSBlockSuffix': None, 'inputNSPBlockSuffix': 'analog_inputs',
        'addToNIX': True, 'forceRecalc': True, 'curateManually': True, 'blockIdx': '1', 'lazy': True,
        'preparationStage': False, 'addToBlockSuffix': None}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass

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
if arguments['outputINSBlockSuffix'] is None:
    outputINSBlockSuffix = ''
else:
    outputINSBlockSuffix = "_{}".format(arguments['outputINSBlockSuffix'])

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
    print('Loading session {}, {}'.format(insSessIdx, jsonSessName))
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
    ####
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

alignByXCorr = True
alignByUnixTime = False
getINSTrigTimes = True
plotSynchReport = True

if plotSynchReport and alignByXCorr:
    insDiagnosticsFolder = os.path.join(figureFolder, 'insDiagnostics')
    if not os.path.exists(insDiagnosticsFolder):
        os.mkdirs(insDiagnosticsFolder)
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
    if plotSynchReport and alignByXCorr:
        synchReportPDF = PdfPages(
            os.path.join(
                insDiagnosticsFolder,
                'ins_synch_report_Block{:0>3}{}.pdf'.format(blockIdx, inputINSBlockSuffix)))
    sessionUnixTimeList = [
        int(sessionName.split('Session')[-1])
        for sessionName in jsonSessionNames
        ]
    homeTZ = pytz.timezone("America/New_York")
    sessionDatetimeList = [
        pd.Timestamp(sut, unit='ms', tz='America/New_York')
        for sut in sessionUnixTimeList
    ]
    nspAbsStart = pd.Timestamp(nspBlock.annotations['recDatetimeStr']).astimezone(homeTZ)
    nspDF['unixTime'] = pd.TimedeltaIndex(nspDF['t'], unit='s') + nspAbsStart
    if tapDetectOptsNSP[0]['unixTimeAdjust'] is not None:
        nspDF.loc[:, 'unixTime'] = nspDF['unixTime'] + pd.Timedelta(tapDetectOptsNSP[0]['unixTimeAdjust'], unit='S')
    manualAlignTimes = {
        insSessIdx: {
            'ins': [],
            'nsp': []
            }
        for insSessIdx, insGroup in insDF.groupby('trialSegment')}
    for insSessIdx, insGroup in insDF.groupby('trialSegment'):
        print('aligning session nb {}'.format(insSessIdx))
        sessTapOptsNSP = tapDetectOptsNSP[insSessIdx]
        sessTapOptsINS = tapDetectOptsINS[insSessIdx]
        stimTrainEdgeProportion = sessTapOptsINS.pop('stimTrainEdgeProportion', None)
        minStimAmp = sessTapOptsINS.pop('minStimAmp', None)
        theseChanNamesNSP = [
            'seg0_' + scn
            for scn in sessTapOptsNSP['synchChanName']
            ]
        # searchRadius = [-1.5, 1.5]
        searchRadius = sessTapOptsINS.pop('searchRadius', [-1.5, 1.5])
        searchRadiusUnix = [
            pd.Timedelta(searchRadius[0], unit='s'),
            pd.Timedelta(searchRadius[1], unit='s')]
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
        else:
            insTimeRangesMask = pd.Series(True, index=insGroup.index)
        if sessTapOptsNSP['timeRanges'] is not None:
            nspTimeRangesMask = hf.getTimeMaskFromRanges(
                nspDF['t'], sessTapOptsNSP['timeRanges'])
            maskedNspUnixTimes = nspDF.loc[nspTimeRangesMask, 'unixTime']
            searchLimsUnixStartList.append(maskedNspUnixTimes.min())
            searchLimsUnixStopList.append(maskedNspUnixTimes.max())
        else:
            nspTimeRangesMask = pd.Series(True, index=nspDF.index)
        if (sessTapOptsNSP['timeRanges'] is not None) and (sessTapOptsINS['timeRanges'] is not None):
            timeRangeOffset = (maskedNspUnixTimes.mean() - maskedInsUnixTimes.mean()).total_seconds()
            print('Time ranges were specified for both NSP and INS.\nThey are offset by approx. {} seconds.'.format(timeRangeOffset))
            print('Consider offsetting INS time by that amount so that they are aligned (synchInfo[\'ins\'][{}][{}][\'unixTimeAdjust\'])'.format(blockIdx, insSessIdx))
        searchLimsUnix = {
            0: max(searchLimsUnixStartList),
            1: min(searchLimsUnixStopList)
            }
        searchWindowSize = (searchLimsUnix[1] - searchLimsUnix[0]).total_seconds()
        if searchWindowSize < 0:
            print('Warning! Session nb {}: search windows do not overlap.'.format(insSessIdx))
        insSearchMask = (insGroup['unixTime'] >= searchLimsUnix[0]) & (insGroup['unixTime'] < searchLimsUnix[1])
        assert insSearchMask.any()
        #
        thisInsDF = (
            insGroup
            .loc[insSearchMask, ['t', 'unixTime']]
            .copy().reset_index(drop=True))
        # pdb.set_trace()
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
        print('    delta T is approx {:.3f}'.format(unixDeltaT))
        #
        nspVals = nspGroup.loc[:, theseChanNamesNSP].to_numpy()
        filterOpts = None
        if filterOpts is not None:
            print('Filtering NSP traces...')
            filterCoeffs = hf.makeFilterCoeffsSOS(
                filterOpts, nspSamplingRate)
            nspVals = signal.sosfiltfilt(
                filterCoeffs,
                nspVals, axis=0)
        if sessTapOptsNSP['minAnalogValue'] is not None:
            nspVals[nspVals < sessTapOptsNSP['minAnalogValue']] = 0
        if sessTapOptsNSP['zScoreTapDetection']:
            empCov = EmpiricalCovariance().fit(nspVals)
            thisNspDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(nspVals)
        else:
            thisNspDF.loc[:, 'tapDetectSignal'] = np.mean(nspVals, axis=1)
        if sessTapOptsNSP['trigFinder'] == 'getTriggers':
            nspPeakIdx = hf.getTriggers(
                thisNspDF['tapDetectSignal'], iti=sessTapOptsNSP['iti'], itiWiggle=.2,
                fs=nspSamplingRate, plotting=arguments['plotting'], # absVal=False,
                thres=sessTapOptsNSP['thres'], edgeType='rising', keep_max=True)
        elif sessTapOptsNSP['trigFinder'] == 'getThresholdCrossings':
            nspPeakIdx, _ = hf.getThresholdCrossings(
                thisNspDF['tapDetectSignal'], thresh=sessTapOptsNSP['thres'],
                iti=sessTapOptsNSP['iti'], fs=nspSamplingRate,
                edgeType='rising', itiWiggle=.2,
                absVal=False, plotting=arguments['plotting'])
        elif sessTapOptsNSP['trigFinder'] == 'peaks':
            width = int(nspSamplingRate * sessTapOptsNSP['iti'] * 0.8)
            peakIdx = peakutils.indexes(
                thisNspDF['tapDetectSignal'].to_numpy(), thres=sessTapOptsNSP['thres'],
                min_dist=width, thres_abs=True, keep_what='max')
            nspPeakIdx = thisNspDF.index[peakIdx]
        nspPeakIdx = nspPeakIdx[sessTapOptsNSP['keepIndex']]
        nspTapTimes = thisNspDF.loc[nspPeakIdx, 't'].to_numpy()
        print(
            'nspTapTimes were detected between t = {:.3f} and {:.3f}'.format(
                nspTapTimes.min(), nspTapTimes.max()))
        insTapTimesStList = None
        if getINSTrigTimes and len(sessTapOptsINS['synchChanName']) > 0:
            theseChanNamesINS = [
                'seg0_' + scn
                for scn in sessTapOptsINS['synchChanName']
                ]
            insVals = insGroup.loc[insSearchMask, theseChanNamesINS].to_numpy()
            '''
            fig, ax = plt.subplots()
            for cN in theseChanNamesINS:
                # ax.plot(insGroup['unixTime'], insGroup[cN], label=cN)
                ax.plot(insGroup['t'], insGroup[cN], label=cN)
            ax.legend()
            plt.show()
            '''
            if sessTapOptsINS['minAnalogValue'] is not None:
                insVals[insVals < sessTapOptsINS['minAnalogValue']] = 0
            if True:
                empCov = EmpiricalCovariance().fit(insVals)
                thisInsDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(insVals)
            else:
                thisInsDF.loc[:, 'tapDetectSignal'] = np.mean(insVals, axis=1)
            insPeakIdx = hf.getTriggers(
                thisInsDF['tapDetectSignal'], iti=sessTapOptsINS['iti'], itiWiggle=.5,
                fs=insSamplingRate, plotting=arguments['plotting'], keep_max=True,
                thres=sessTapOptsINS['thres'], edgeType='rising')
            insTapTimes = thisInsDF.loc[insPeakIdx, 't'].to_numpy()[sessTapOptsINS['keepIndex']]
            if arguments['curateManually']:
                try:
                    manualAlignTimes[insSessIdx], fig, ax = mdt.peekAtTapsV2(
                        thisNspDF, thisInsDF,
                        insAuxDataDF=insGroup.loc[insSearchMask, :].drop(columns=['t', 'unixTime', 'trialSegment']).reset_index(drop=True),
                        tapTimestampsINS=insTapTimes, tapTimestampsNSP=nspTapTimes,
                        tapDetectOptsNSP=sessTapOptsNSP, tapDetectOptsINS=sessTapOptsINS,
                        procFunINS=stats.zscore,
                        # procFunNSP=stats.zscore
                        )
                    fig.suptitle('block {} sess # {} ({})'.format(blockIdx, insSessIdx, jsonSessionNames[insSessIdx]))
                    plt.show()
                    if len(manualAlignTimes[insSessIdx]['ins']) > 0:
                        insTapTimes = pd.Series(manualAlignTimes[insSessIdx]['ins'])
                        ## remove detections that are too close to each other
                        insTapTimes = insTapTimes.loc[insTapTimes.diff().fillna(1) > sessTapOptsINS['iti']].to_numpy()
                except Exception:
                    traceback.print_exc()
            insTapTimesStList = [SpikeTrain(
                insTapTimes, t_start=thisInsDF['t'].min() * pq.s,
                t_stop=(thisInsDF['t'].max()) * pq.s,
                units=pq.s, name='tap times', dtype=np.float64)]
            ###
            if (insTapTimes.size == nspTapTimes.size):
                unixDeltaT = (nspTapTimes - insTapTimes).mean()
                print('    Aligning preliminary tap times: new delta T = {:.3f}'.format(unixDeltaT))
        if sessTapOptsINS['synchStimUnitName'] is not None:
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
            if insTapTimesStList is not None:
                # cross corr stim timestamps
                coarseSpikeMats = []
                for coarseSt in insTapTimesStList:
                    if len(coarseSt.times) > 0:
                        coarseSt.t_start = coarseSt.t_start + unixDeltaT * coarseSt.t_start.units
                        coarseSt.t_stop = coarseSt.t_stop + unixDeltaT * coarseSt.t_stop.units
                        newStT = coarseSt.times.magnitude + unixDeltaT
                        coarseSt.magnitude[:] = newStT
                        selectionMask = (coarseSt.magnitude ** 0).astype(bool)
                        if 'amplitude' in coarseSt.array_annotations:
                            theseAmps = coarseSt.array_annotations['amplitude']
                            if minStimAmp is not None:
                                ampMask = theseAmps > minStimAmp
                                selectionMask = selectionMask & ampMask
                                # spikesToBinarize = coarseSt.times[ampMask]
                                # theseAmps = theseAmps[ampMask]
                        if 'rankInTrain' in coarseSt.array_annotations:
                            if stimTrainEdgeProportion is not None:
                                rankInTrain = coarseSt.array_annotations['rankInTrain']
                                assert 'trainNPulses' in coarseSt.array_annotations
                                trainNPulses = coarseSt.array_annotations['trainNPulses']
                                rankAsProportion = rankInTrain / trainNPulses
                                rankMask = (rankAsProportion < stimTrainEdgeProportion) | (rankAsProportion > (1 - stimTrainEdgeProportion))
                                selectionMask = selectionMask & rankMask
                        spikesToBinarize = coarseSt.times[selectionMask].astype(np.float64)
                        thisSpikeMat = binarize(
                            spikesToBinarize,
                            sampling_rate=trigRasterSamplingRate * pq.Hz,
                            t_start=trigRaster['t'].min() * pq.s,
                            t_stop=trigRaster['t'].max() * pq.s)
                        idxOfSpikes = np.flatnonzero(thisSpikeMat)
                        thisSpikeMat = thisSpikeMat.astype(float)
                        if 'amplitude' in coarseSt.array_annotations:
                            if len(idxOfSpikes) > selectionMask.sum():
                                excess = int(len(idxOfSpikes) - selectionMask.sum())
                                idxOfSpikes = idxOfSpikes[:(1-excess)]
                            elif len(idxOfSpikes) < selectionMask.sum():
                                excess = int(selectionMask.sum() - len(idxOfSpikes))
                                selectionMask[np.flatnonzero(selectionMask)[-excess:]] = False
                            thisSpikeMat[idxOfSpikes] = theseAmps[selectionMask]
                        coarseSpikeMats.append(thisSpikeMat[:, np.newaxis])
                        print(
                            '{}: coarseSt.times from {:.3f} to {:.3f}'.format(
                                coarseSt.name, min(coarseSt.times), max(coarseSt.times)))
                        # print('st.times = {}'.format(st.times))
                if not len(coarseSpikeMats) > 0:
                    print('\n\n INS {} has no stim spikes!'.format(jsonSessionNames[insSessIdx]))
                    print('Defaulting to xcorr analog signals\n\n')
                    sessTapOptsINS['synchByXCorrTapDetectSignal'] = True
                    sessTapOptsNSP['synchByXCorrTapDetectSignal'] = True
                else:
                    trigRaster.loc[:, 'insDiracDelta'] = np.concatenate(
                        coarseSpikeMats, axis=1).sum(axis=1)
                    trigRaster.loc[:, 'insTrigs'] = hf.gaussianSupport(
                        support=trigRaster.set_index('t')['insDiracDelta'],
                        gaussWid=gaussWid, fs=trigRasterSamplingRate).to_numpy()
                    # plt.plot(trigRaster['t'], trigRaster['insTrigs'])
            if sessTapOptsINS['synchByXCorrTapDetectSignal']:
                xcorrINS = thisInsDF.copy()
                xcorrINS.loc[:, 'coarseNspTime'] = xcorrINS['t'] + unixDeltaT
                trigRaster.loc[:, 'insTrigs'] = hf.interpolateDF(
                    xcorrINS, trigRaster['t'], x='coarseNspTime',
                    columns=['tapDetectSignal'],
                    kind='linear', fill_value=(0, 0))
            elif insTapTimes is not None:
                approxInsTapTimes = insTapTimes + unixDeltaT
                closestTimes, closestIdx = hf.closestSeries(
                    takeFrom=pd.Series(approxInsTapTimes), compareTo=trigRaster['t']
                    )
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
                    t_stop=(trigRaster['t'].max() + 1e-3) * pq.s, dtype=np.float64)
                nspDiracRaster = binarize(
                    nspDiracSt, sampling_rate=trigRasterSamplingRate * pq.Hz,
                    t_start=trigRaster['t'].min() * pq.s, t_stop=(trigRaster['t'].max()) * pq.s
                    )
                useValsAtTrigs = False
                if useValsAtTrigs:
                    indicesToUse = np.flatnonzero(nspDiracRaster)
                    nspDiracRaster = nspDiracRaster.astype(np.float64)
                    nspDiracRaster[indicesToUse] = thisNspDF.loc[nspPeakIdx, 'tapDetectSignal'].to_numpy()
                    trigRaster.loc[:, 'nspDiracDelta'] = nspDiracRaster
                else:
                    trigRaster.loc[:, 'nspDiracDelta'] = nspDiracRaster.astype(float)
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
                dtype=int)
            targetLagsSrs = pd.Series(
                targetLags, index=targetLags * trigSampleInterval)
            if sessTapOptsINS['zeroOutsideTargetLags']:
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
                customTitle = 'Block {}, insSession {}: {}'.format(blockIdx, insSessIdx, jsonSessionNames[insSessIdx])
                customMessages = [
                    'INS session',
                    '    lasted {:.1f} sec'.format((sessStopUnix - sessStartUnix).total_seconds()),
                    '    approx delay {:.1f} sec'.format(unixDeltaT),
                    '# of NSP trigs = {}'.format(trigRaster.loc[:, 'nspDiracDelta'].sum()),
                    '# of INS trigs = {}'.format(trigRaster.loc[:, 'insDiracDelta'].sum())
                ]
                fig, ax, figSaveOpts = hf.plotCorrSynchReport(
                    _trigRaster=trigRaster, _searchRadius=searchRadius,
                    _targetLagsSrs=targetLagsSrs, _maxLag=maxLag, _xCorrSrs=xCorrSrs,
                    customMessages=customMessages, customTitle=customTitle
                    )
                synchReportPDF.savefig(**figSaveOpts)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
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
        ev.magnitude[:] = newEvT
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
    ns5FileName + '{}.nix'.format(outputINSBlockSuffix))
if os.path.exists(outPathName):
    print('Found existing file and deleting {}...'.format(outPathName))
    os.remove(outPathName)
writer = neo.io.NixIO(filename=outPathName)
writer.write_block(insBlockInterp, use_obj_names=True)
writer.close()

print('Done writing synchronized .nix file.')

if plotSynchReport and alignByXCorr:
    print('Saving confirmation plots...')
    synchConfirmPDF = PdfPages(
        os.path.join(
            insDiagnosticsFolder,
            'ins_synch_illustration_Block{:0>3}{}.pdf'.format(blockIdx, inputINSBlockSuffix)))
    ###
    nspChanNames += ['seg0_position', 'seg0_utah_artifact_0']
    if arguments['lazy']:
        asigListNSP = [
            asigP.load()
            for asigP in nspBlock.filter(objects=AnalogSignalProxy)
            if asigP.name in nspChanNames
            ]
    else:
        asigListNSP = [
            asig
            for asig in nspBlock.filter(objects=AnalogSignal)
            if asig.name in nspChanNames
            ]
    nspChanNames = [asig.name for asig in asigListNSP]
    nspDF = ns5.analogSignalsToDataFrame(asigListNSP)
    insSignalsToPlot = insSignalsToSave
    asigListINS = [
        asig
        for asig in insBlockInterp.filter(objects=AnalogSignal)
        if asig.name in insSignalsToPlot
        ]
    insSignalsToPlot = [asig.name for asig in asigListINS]
    insDF = ns5.analogSignalsToDataFrame(asigListINS)
    ####
    confPlotWinSize = 50.  # seconds
    plotRounds = nspDF['t'].apply(lambda x: np.floor(x / confPlotWinSize))
    plotRoundsINS = insDF['t'].apply(lambda x: np.floor(x / confPlotWinSize))
    spikesToPlot = [st for st in insBlockInterp.filter(objects=SpikeTrain)]
    for pr in plotRounds.unique():
        plotMask = (plotRounds == pr)
        plotMaskINS = (plotRoundsINS == pr)
        fig, ax = plt.subplots(2, 1, figsize=(21, 3))
        for cNIdx, cN in enumerate(nspChanNames):
            try:
                # plotTrace = stats.zscore(nspDF.loc[plotMask, cN]) + cNIdx
                plotTrace = MinMaxScaler(feature_range=(1e-2, 1)).fit_transform(nspDF.loc[plotMask, [cN]].to_numpy()) + cNIdx
                ax[0].plot(nspDF.loc[plotMask, 't'], plotTrace.flatten(), label=cN, alpha=0.5, rasterized=True)
            except Exception:
                traceback.print_exc()
        for cNIdx, cN in enumerate(insSignalsToPlot):
            try:
                # plotTrace = stats.zscore(insDF.loc[plotMaskINS, cN]) + cNIdx
                plotTrace = MinMaxScaler(feature_range=(1e-2, 1)).fit_transform(insDF.loc[plotMaskINS, [cN]].to_numpy()) + cNIdx
                ax[1].plot(insDF.loc[plotMaskINS, 't'], plotTrace.flatten(), label=cN, alpha=0.5, rasterized=True)
            except Exception:
                traceback.print_exc()
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        rasterLevels0 = np.linspace(
            ax[0].get_ylim()[0], ax[0].get_ylim()[1],
            len(spikesToPlot) + 2)
        rasterLevels1 = np.linspace(
            ax[1].get_ylim()[0], ax[1].get_ylim()[1],
            len(spikesToPlot) + 2)
        for stId, st in enumerate(spikesToPlot):
            if len(st):
                plotTMask = (st.times >= nspDF.loc[plotMask, 't'].min()) & (st.times < nspDF.loc[plotMask, 't'].max())
                ax[0].scatter(
                    st.times[plotTMask], st.times[plotTMask] ** 0 - 1 + rasterLevels0[stId + 1],
                    label=st.name, rasterized=True, marker='+')
                ax[1].scatter(
                    st.times[plotTMask], st.times[plotTMask] ** 0 - 1 + rasterLevels1[stId + 1],
                    label=st.name, rasterized=True, marker='+')
        ax[0].legend(loc='lower left')
        ax[1].legend(loc='lower left')
        ax[1].set_xlabel('Time (s)')
        ax[0].set_xlim(
            [nspDF.loc[plotMask, 't'].min(),
             nspDF.loc[plotMask, 't'].max()])
        ax[1].set_xlim(
            [insDF.loc[plotMaskINS, 't'].min(),
             insDF.loc[plotMaskINS, 't'].max()])
        ax[0].set_ylabel('time domain data (a.u.)')
        ax[1].set_ylabel('time domain data (a.u.)')
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        figSaveOpts = dict(
            bbox_extra_artists=tuple([ta.get_legend() for ta in ax]),
            bbox_inches='tight')
        synchConfirmPDF.savefig(**figSaveOpts)
        plt.close()
    synchConfirmPDF.close()

print(
    '\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
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
