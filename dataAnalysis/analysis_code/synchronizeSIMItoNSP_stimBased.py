"""07: Combine Simi and NSP Data
Usage:
    synchronizeSimiToNSP [options]

Options:
    --blockIdx=blockIdx                                   which trial to analyze
    --exp=exp                                             which experimental day to analyze
    --inputBlockSuffix=inputBlockSuffix                   append a name to the input block?
    --inputBlockPrefix=inputBlockPrefix                   prepend a name to the input block?
    --lazy                                                whether to fully load data from blocks [default: False]
    --showFigures                                         whether to plot diagnostic figures [default: False]
    --curateManually                                      whether to manually confirm synch [default: False]
    --preparationStage                                    get aproximate timings and print to shell, to help identify time ranges? [default: False]
    --plotting                                            whether to display confirmation plots [default: False]
    --forceRecalc                                         whether to overwrite any previous calculated synch [default: False]
    --usedTENSPulses                                      whether the sync was done using TENS pulses (as opposed to mechanical taps) [default: False]
    --processAll                                          process entire experimental day? [default: False]
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
import json
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
import vg
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
import elephant as elph
import elephant.pandas_bridge as elphpdb
from elephant.conversion import binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
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

idxSl = pd.IndexSlice

#############################################################
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
searchRadius = [-30., 30.]
# searchRadius = [-.5, .5]
searchRadiusUnix = [
    pd.Timedelta(searchRadius[0], unit='s'),
    pd.Timedelta(searchRadius[1], unit='s')]
#
#  Load NSP Data
############################################################
nspPath = os.path.join(
    scratchFolder,
    blockBaseName + inputBlockSuffix +
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
    '{}_{}_Simi_to_NSP_synchFun.json'.format(experimentName, ns5FileName))
#
tapDetectOptsNSP = expOpts['synchInfo']['nspForSimi'][blockIdx]
tapDetectOptsSimi = expOpts['synchInfo']['simiToNsp'][blockIdx]

interpFunSimiToNSP = {}
interpFunNSPtoSimi = {}
#  load Simi Data
############################################################
projVideoFolder = '/gpfs/data/dborton/rdarie/Video_Tracking/dummy_projects/proprioInference2-Radu-2021-04-07/videos'
simiPath = os.path.join(
    projVideoFolder,
    '{}-{}-{}shuffle{}_{}_3D.h5'.format(
        experimentName, ns5FileName,
        tapDetectOptsSimi['scorerName'],
        tapDetectOptsSimi['shuffle'],
        tapDetectOptsSimi['snapshot'],
        ))
simiDF = pd.read_hdf(simiPath, 'led_status')
simiDF = simiDF.loc[:, tapDetectOptsSimi['synchChanName']]
simiDF.columns = pd.Series(simiDF.columns.to_numpy()).apply(lambda x: '_'.join(['{}'.format(comp) for comp in x]))
theseChanNamesSimi = simiDF.columns.to_list()
simiDF.index.name = 't'
simiDF.reset_index(inplace=True)
simiTimeMeta = pd.read_hdf(simiPath, 'file_metadata').iloc[0, :]
simiSamplingRate = 100
#  ##### NSP Loading
nspChanNames = []
for scn in tapDetectOptsNSP['synchChanName']:
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
    interpFunSimiToNSP = {
        int(key): np.poly1d(value)
        for key, value in interpFunLoaded['simiToNsp'].items()
    }
    interpFunNSPtoSimi = {
        int(key): np.poly1d(value)
        for key, value in interpFunLoaded['nspToSimi'].items()
    }
else:
    simiAbsStart = simiTimeMeta['systemTime'].tz_convert('UTC')
    print('simiAbsStart = {}\n'.format(simiAbsStart))
    simiDF.loc[:, 'unixTime'] = (
        pd.TimedeltaIndex(simiDF['t'], unit='s') +
        simiAbsStart)
    if tapDetectOptsNSP['unixTimeAdjust'] is not None:
        simiDF.loc[:, 'unixTime'] = (
            simiDF['unixTime'] +
            pd.Timedelta(tapDetectOptsSimi['unixTimeAdjust'], unit='S'))
    #
    nspAbsStart = pd.Timestamp(nspBlock.annotations['recDatetimeStr'])
    print('nspAbsStart = {}\n'.format(nspAbsStart))
    nspDF['unixTime'] = pd.TimedeltaIndex(nspDF['t'], unit='s') + nspAbsStart
    if tapDetectOptsNSP['unixTimeAdjust'] is not None:
        nspDF.loc[:, 'unixTime'] = (
            nspDF['unixTime'] +
            pd.Timedelta(tapDetectOptsNSP['unixTimeAdjust'], unit='S'))
    manualAlignTimes = {'simi': [], 'nsp': []}
    #
    alignByXCorr = True
    alignByUnixTime = False
    getSimiTrigTimes = True
    plotSynchReport = True
    plotTriggers = False
    #
    if plotSynchReport and alignByXCorr:
        simiDiagnosticsFolder = os.path.join(figureFolder, 'simiDiagnostics')
        if not os.path.exists(simiDiagnosticsFolder):
            os.mkdir(simiDiagnosticsFolder)
        synchReportPDF = PdfPages(
            os.path.join(
                simiDiagnosticsFolder,
                'simi_synch_report_Block{:0>3d}.pdf'.format(blockIdx)))
    #
    searchLimsUnixStartList = []
    searchLimsUnixStopList = []
    if tapDetectOptsSimi['timeRanges'] is not None:
        simiTimeRangesMask = hf.getTimeMaskFromRanges(
            simiDF['t'], tapDetectOptsSimi['timeRanges'])
        maskedSimiUnixTimes = simiDF.loc[simiTimeRangesMask, 'unixTime']
        searchLimsUnixStartList.append(maskedSimiUnixTimes.min())
        searchLimsUnixStopList.append(maskedSimiUnixTimes.max())
    else:
        searchLimsUnixStartList.append(simiDF['unixTime'].min())
        searchLimsUnixStopList.append(simiDF['unixTime'].max())
    #
    if tapDetectOptsNSP['timeRanges'] is not None:
        nspTimeRangesMask = hf.getTimeMaskFromRanges(
            nspDF['t'], tapDetectOptsNSP['timeRanges'])
        maskedNspUnixTimes = nspDF.loc[nspTimeRangesMask, 'unixTime']
        searchLimsUnixStartList.append(maskedNspUnixTimes.min())
        searchLimsUnixStopList.append(maskedNspUnixTimes.max())
    else:
        searchLimsUnixStartList.append(nspDF['unixTime'].min())
        searchLimsUnixStopList.append(nspDF['unixTime'].max())
    #
    searchLimsUnix = {
        0: max(searchLimsUnixStartList),
        1: min(searchLimsUnixStopList)
        }
    simiSearchMask = (
        (simiDF['unixTime'] >= searchLimsUnix[0]) &
        (simiDF['unixTime'] < searchLimsUnix[1])
        )
    assert simiSearchMask.any()
    simiGroup = (
        simiDF
        .loc[simiSearchMask, :])
    # preallocate DF to hold the filtered detect signal
    thisSimiDF = (
        simiGroup
        .loc[:, ['t', 'unixTime']]
        .copy().reset_index(drop=True))
    # simiSearchLims = thisSimiDF.loc[:, 't'].quantile([0, 1])
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
    unixDeltaT = thisNspDF['t'].iloc[0] - thisSimiDF['t'].iloc[0]
    print('    delta T is approx {}'.format(unixDeltaT))
    #
    nspVals = nspGroup.loc[:, nspChanNames].to_numpy()
    filterOpts = None
    if filterOpts is not None:
        print('Filtering NSP traces...')
        filterCoeffs = hf.makeFilterCoeffsSOS(
            filterOpts, nspSamplingRate)
        nspVals = signal.sosfiltfilt(
            filterCoeffs,
            nspVals, axis=0)
    if tapDetectOptsNSP['minAnalogValue'] is not None:
        nspVals[nspVals < tapDetectOptsNSP['minAnalogValue']] = 0
    if True:
        empCov = EmpiricalCovariance().fit(nspVals)
        thisNspDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(nspVals)
    else:
        thisNspDF.loc[:, 'tapDetectSignal'] = np.mean(nspVals, axis=1)
    #
    nspTrigFinder = 'getThresholdCrossings'
    #
    if nspTrigFinder == 'getTriggers':
        nspPeakIdx = hf.getTriggers(
            thisNspDF['tapDetectSignal'], iti=tapDetectOptsNSP['iti'], itiWiggle=.2,
            fs=nspSamplingRate, plotting=plotTriggers, absVal=False,
            thres=tapDetectOptsNSP['thres'], edgeType=tapDetectOptsNSP['edgeType'], keep_max=True)
    elif nspTrigFinder == 'getThresholdCrossings':
        nspPeakIdx, _ = hf.getThresholdCrossings(
            thisNspDF['tapDetectSignal'], thresh=tapDetectOptsNSP['thres'],
            iti=tapDetectOptsNSP['iti'], fs=nspSamplingRate,
            edgeType=tapDetectOptsNSP['edgeType'], itiWiggle=.2,
            absVal=False, plotting=plotTriggers)
    elif nspTrigFinder == 'peaks':
        width = int(nspSamplingRate * tapDetectOptsNSP['iti'] * 0.8)
        peakIdx = peakutils.indexes(
            thisNspDF['tapDetectSignal'].to_numpy(), thres=tapDetectOptsNSP['thres'],
            min_dist=width, thres_abs=True, keep_what='max')
        nspPeakIdx = thisNspDF.index[peakIdx]
    #
    nspPeakIdx = nspPeakIdx[tapDetectOptsNSP['keepIndex']]
    nspTapTimes = thisNspDF.loc[nspPeakIdx, 't'].to_numpy()
    print(
        'nspTapTimes from {:.3f} to {:.3f}'.format(
            nspTapTimes.min(), nspTapTimes.max()))
    simiVals = simiGroup.loc[:, theseChanNamesSimi].to_numpy()
    if tapDetectOptsSimi['minAnalogValue'] is not None:
        simiVals[simiVals < tapDetectOptsSimi['minAnalogValue']] = 0
    if True:
        empCov = EmpiricalCovariance().fit(simiVals)
        thisSimiDF.loc[:, 'tapDetectSignal'] = empCov.mahalanobis(simiVals)
    else:
        thisSimiDF.loc[:, 'tapDetectSignal'] = np.mean(simiVals, axis=1)
    #
    if getSimiTrigTimes:
        simiTrigFinder = 'getThresholdCrossings'
        #
        if simiTrigFinder == 'getTriggers':
            simiPeakIdx = hf.getTriggers(
                thisSimiDF['tapDetectSignal'], iti=tapDetectOptsSimi['iti'], itiWiggle=.2,
                fs=simiSamplingRate, plotting=plotTriggers, absVal=False,
                thres=tapDetectOptsSimi['thres'], edgeType=tapDetectOptsSimi['edgeType'], keep_max=True)
        elif simiTrigFinder == 'getThresholdCrossings':
            simiPeakIdx, _ = hf.getThresholdCrossings(
                thisSimiDF['tapDetectSignal'], thresh=tapDetectOptsSimi['thres'],
                iti=tapDetectOptsSimi['iti'], fs=simiSamplingRate,
                edgeType=tapDetectOptsSimi['edgeType'], itiWiggle=.2,
                absVal=False, plotting=plotTriggers)
        elif simiTrigFinder == 'peaks':
            width = int(simiSamplingRate * tapDetectOptsSimi['iti'] * 0.8)
            peakIdx = peakutils.indexes(
                thisSimiDF['tapDetectSignal'].to_numpy(), thres=tapDetectOptsSimi['thres'],
                min_dist=width, thres_abs=True, keep_what='max')
            simiPeakIdx = thisSimiDF.index[peakIdx]
        simiTapTimes = (
            thisSimiDF
            .loc[simiPeakIdx, 't']
            .to_numpy()[tapDetectOptsSimi['keepIndex']])
        if arguments['curateManually']:
            try:
                manualAlignTimes, fig, ax = mdt.peekAtTapsV2(
                    thisNspDF, thisSimiDF,
                    simiAuxDataDF=simiGroup.drop(
                        columns=['t', 'unixTime', 'trialSegment']),
                    # plotMaskNSP=nspSearchMask, plotMaskINS=simiSearchMask,
                    tapTimestampsINS=simiTapTimes,
                    tapTimestampsNSP=nspTapTimes,
                    tapDetectOptsNSP=tapDetectOptsNSP,
                    tapDetectOptsINS=tapDetectOptsSimi
                    )
                plt.show()
            except Exception:
                traceback.print_exc()
        # pdb.set_trace()
        simiTapTimesStList = [
            SpikeTrain(
                simiTapTimes, t_start=thisSimiDF['t'].min() * pq.s,
                t_stop=thisSimiDF['t'].max() * pq.s,
                units=pq.s, name='tap times')
                ]
    #
    if alignByXCorr:
        if tapDetectOptsSimi['xCorrSamplingRate'] is not None:
            trigRasterSamplingRate = tapDetectOptsSimi['xCorrSamplingRate']
        else:
            trigRasterSamplingRate = min(nspSamplingRate, simiSamplingRate)
        # trigRasterSamplingRate = 1000 #Hz
        trigSampleInterval = trigRasterSamplingRate ** (-1)
        if tapDetectOptsSimi['xCorrGaussWid'] is not None:
            gaussWid = tapDetectOptsSimi['xCorrGaussWid']
        else:
            gaussWid = 10e-3
        #
        nspSearchDur = nspSearchLims[1] - nspSearchLims[0]
        trigRasterT = (
            nspSearchLims[0] +
            np.arange(0, nspSearchDur, trigSampleInterval))
        trigRaster = pd.DataFrame({
            't': trigRasterT,
            'nspDiracDelta': np.zeros_like(trigRasterT),
            'insDiracDelta': np.zeros_like(trigRasterT),
            'nspTrigs': np.zeros_like(trigRasterT),
            'insTrigs': np.zeros_like(trigRasterT),
            })
        if getSimiTrigTimes:
            # cross corr stim timestamps
            coarseSpikeMats = []
            # pdb.set_trace()
            for coarseSt in simiTapTimesStList:
                if len(coarseSt.times) > 0:
                    coarseSt.t_start = coarseSt.t_start + unixDeltaT * coarseSt.t_start.units
                    coarseSt.t_stop = coarseSt.t_stop + unixDeltaT * coarseSt.t_stop.units
                    newStT = coarseSt.times.magnitude + unixDeltaT
                    coarseSt.magnitude[:] = newStT
                    selectionMask = (coarseSt.magnitude ** 0).astype(np.bool)
                    spikesToBinarize = coarseSt.times[selectionMask]
                    thisSpikeMat = binarize(
                        spikesToBinarize,
                        sampling_rate=trigRasterSamplingRate * pq.Hz,
                        t_start=trigRaster['t'].min() * pq.s,
                        t_stop=trigRaster['t'].max() * pq.s)
                    idxOfSpikes = np.flatnonzero(thisSpikeMat)
                    thisSpikeMat = thisSpikeMat.astype(np.float)
                    coarseSpikeMats.append(thisSpikeMat[:, np.newaxis])
                    # print('st.times = {}'.format(st.times))
            if not len(coarseSpikeMats) > 0:
                print('\n\n Simi has no stim spikes!')
                print('Defaulting to xcorr analog signals\n\n')
                tapDetectOptsSimi['synchByXCorrTapDetectSignal'] = True
                tapDetectOptsNSP['synchByXCorrTapDetectSignal'] = True
            else:
                trigRaster.loc[:, 'insDiracDelta'] = np.concatenate(
                    coarseSpikeMats, axis=1).sum(axis=1)
                trigRaster.loc[:, 'insTrigs'] = hf.gaussianSupport(
                    support=trigRaster.set_index('t')['insDiracDelta'],
                    gaussWid=gaussWid, fs=trigRasterSamplingRate).to_numpy()
        #
        if tapDetectOptsSimi['synchByXCorrTapDetectSignal']:
            xcorrSimi = thisSimiDF.copy()
            xcorrSimi.loc[:, 'coarseNspTime'] = xcorrSimi['t'] + unixDeltaT
            trigRaster.loc[:, 'insTrigs'] = hf.interpolateDF(
                xcorrSimi, trigRaster['t'], x='coarseNspTime',
                columns=['tapDetectSignal'],
                kind='linear', fill_value=(0, 0))
        #
        if tapDetectOptsNSP['synchByXCorrTapDetectSignal']:
            trigRaster.loc[:, 'nspTrigs'] = hf.interpolateDF(
                thisNspDF, trigRaster['t'], x='t',
                columns=['tapDetectSignal'],
                kind='linear', fill_value=(0, 0))
        else:
            nspDiracSt = SpikeTrain(
                times=nspTapTimes, units='s',
                t_start=trigRaster['t'].min() * pq.s,
                t_stop=trigRaster['t'].max() * pq.s)
            nspDiracRaster = binarize(
                nspDiracSt, sampling_rate=trigRasterSamplingRate * pq.Hz,
                t_start=trigRaster['t'].min() * pq.s, t_stop=trigRaster['t'].max() * pq.s
                )
            useValsAtTrigs = False
            if useValsAtTrigs:
                indicesToUse = np.flatnonzero(nspDiracRaster)
                nspDiracRaster = nspDiracRaster.astype(np.float)
                nspDiracRaster[indicesToUse] = thisNspDF.loc[nspPeakIdx, 'tapDetectSignal'].to_numpy()
                trigRaster.loc[:, 'nspDiracDelta'] = nspDiracRaster
            else:
                trigRaster.loc[:, 'nspDiracDelta'] = nspDiracRaster.astype(np.float)
            trigRaster.loc[:, 'nspTrigs'] = hf.gaussianSupport(
                support=trigRaster.set_index('t')['nspDiracDelta'],
                gaussWid=gaussWid,
                fs=trigRasterSamplingRate).to_numpy()
        if tapDetectOptsNSP['clipToThres']:
            trigRaster.loc[:, 'nspTrigs'] = trigRaster.loc[:, 'nspTrigs'].clip(upper=tapDetectOptsNSP['thres'])
        if tapDetectOptsSimi['clipToThres']:
            trigRaster.loc[:, 'insTrigs'] = trigRaster.loc[:, 'insTrigs'].clip(upper=tapDetectOptsSimi['thres'])
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
            customTitle = 'Block {}'.format(blockIdx)
            customMessages = [
                'Simi session',
                '    approx delay {:.1f} sec'.format(unixDeltaT),
                '# of NSP trigs = {}'.format(
                    trigRaster.loc[:, 'nspDiracDelta'].sum()),
                '# of Simi trigs = {}'.format(
                    trigRaster.loc[:, 'insDiracDelta'].sum())
            ]
            fig, ax, figSaveOpts = hf.plotCorrSynchReport(
                _trigRaster=trigRaster, _searchRadius=searchRadius,
                _targetLagsSrs=targetLagsSrs, _maxLag=maxLag,
                _xCorrSrs=xCorrSrs,
                customMessages=customMessages, customTitle=customTitle
                )
            synchReportPDF.savefig(**figSaveOpts)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        #########
        if True:
            from matplotlib.widgets import Slider, Button
            import matplotlib.dates as mdates
            with sns.plotting_context(rc={'ax.facecolor': 'w'}):
                mins_fmt = mdates.DateFormatter('%M:%S')
                # pdb.set_trace()
                maxShift = targetLagsSrs[maxLag]
                fig, ax = plt.subplots()
                twinAx = ax.twinx()
                plt.subplots_adjust(bottom=0.25)
                timeAsDate = pd.to_datetime(trigRaster['t'].to_numpy(), unit='s')
                simiL, = ax.plot(
                    timeAsDate, trigRaster['insTrigs'].shift(maxShift),
                    c='r', alpha=.7, label='simi triggers')
                ax.set_yticks([])
                ax.xaxis.set_major_formatter(mins_fmt)
                ax.margins(x=0)
                nspL, = twinAx.plot(
                    timeAsDate, trigRaster['nspTrigs'],
                    c='b', alpha=.7,
                    label='nsp triggers')
                twinAx.set_yticks([])
                twinAx.xaxis.set_major_formatter(mins_fmt)
                axShift = plt.axes([0.1, 0.1, 0.65, 0.06])
                sShift = Slider(
                    axShift, 'Shift',
                    targetLagsSrs.min(), targetLagsSrs.max(),
                    valinit=maxShift, valstep=1)
                def update(val):
                    newShift = sShift.val
                    simiL.set_ydata(trigRaster['insTrigs'].shift(newShift))
                    fig.canvas.draw_idle()
                sShift.on_changed(update)
                plt.show()
        #########
        funCoeffs = np.asarray([1, unixDeltaT + maxLag])
        invFunCoeffs = np.asarray([1, -1 * (unixDeltaT + maxLag)])
    elif alignByUnixTime:
        funCoeffs = np.asarray([1, unixDeltaT])
        invFunCoeffs = np.asarray([1, -1 * (unixDeltaT)])
    else:
        # align by regressing timestamps
        # funCoeffs = np.poly1d()
        pass
    interpFunSimiToNSP[0] = np.poly1d(funCoeffs)
    interpFunNSPtoSimi[0] = np.poly1d(invFunCoeffs)
    # end getting interp function
    interpFunExport = {
        'simiToNsp': {
            key: list(value)
            for key, value in interpFunSimiToNSP.items()
            },
        'nspToSimi': {
            key: list(value)
            for key, value in interpFunNSPtoSimi.items()
            }
        }
    with open(synchFunPath, 'w') as f:
        json.dump(interpFunExport, f)
    if plotSynchReport and alignByXCorr:
        synchReportPDF.close()
#
# synch fun calculated, now apply
nspBoundaries = nspDF['t'].quantile([0, 1])
simiDF.loc[:, 'nspT'] = np.polyval(interpFunSimiToNSP[0], simiDF['t'])
simiDF.drop(columns='t', inplace=True)
if 'unixTime' in simiDF.columns:
    simiDF.drop(columns='unixTime', inplace=True)
simiBoundaries = simiDF['nspT'].quantile([0,1])
outTStart = max(nspBoundaries[0], simiBoundaries[0])
outTStop = min(nspBoundaries[1], simiBoundaries[1])
outT = np.arange(outTStart, outTStop + simiSamplingRate ** -1, simiSamplingRate ** -1)

# pdb.set_trace()
simiPoses = pd.read_hdf(simiPath, 'df_with_missing')
simiPoses.columns = simiPoses.columns.droplevel('scorer')
lOfAngles = {}
anglesToCalc = {
    'right_hip': ['right_crest', 'right_hip', 'right_knee'],
    'right_knee': ['right_hip', 'right_knee', 'right_ankle'],
    'right_ankle': ['right_knee', 'right_ankle', 'right_knuckle']
    }
for jointName, listOfParts in anglesToCalc.items():
    p = [
        simiPoses.loc[:, idxSl['individual1', partName, ['x', 'y', 'z']]].to_numpy()
        for partName in listOfParts
        ]
    v = [
        p[1] - p[0],
        p[1] - p[2]
        ]
    simiPoses.loc[:, ('individual1', jointName, 'angle')] = vg.angle(v[0], v[1])

p = [
    simiPoses.loc[:, idxSl['single', partName, ['x', 'y', 'z']]].to_numpy()
    for partName in ['pedal_center', 'pedal_end']
    ]
v = [
    p[1] - p[0]
    ]
v.append(np.zeros_like(p[0]))
v[1][:, 0] = 1
simiPoses.loc[:, ('single', 'pedal', 'angle')] = vg.angle(v[0], v[1])
simiAngles = simiPoses.loc[:, idxSl[:, :, 'angle']].copy()
angleColumnNames = simiAngles.columns.to_frame().apply(lambda x: '{}_{}'.format(x[1], x[2]), axis=1)
simiAngles.columns = angleColumnNames.to_list()
simiAnglesInterp = (
    simiAngles
    .interpolate(method='linear')
    .fillna(method='bfill').fillna(method='ffill'))
simiDFOutInterp = hf.interpolateDF(
    pd.concat([simiDF, simiAnglesInterp], axis='columns'),
    outT, x='nspT',
    kind='linear', fill_value=(0, 0))
#
simiDFOutInterp.columns = [cN.replace('seg0_', '') for cN in simiDFOutInterp.columns]

videoMetaData = {
    'video_filenames': [
        os.path.join(
            projVideoFolder,
            '{}-{}-{}{}shuffle{}_{}_{}{}_bp_labeled.mp4'.format(
                experimentName, ns5FileName, cameraIdx,
                tapDetectOptsSimi['scorerName'],
                tapDetectOptsSimi['shuffle'],
                tapDetectOptsSimi['snapshot'],
                tapDetectOptsSimi['tracker'],
                tapDetectOptsSimi['filteredSuffix'],
            ))
        for cameraIdx in range(1, 6)],
    'video_times': [
        simiDFOutInterp['nspT'].to_list()
        for cameraIdx in range(1, 6)]
    }
videoMetaDataPath = os.path.join(
    scratchFolder,
    ns5FileName + '_simiTrigs_videoMetadata.json')
with open(videoMetaDataPath, 'w') as f:
    json.dump(videoMetaData, f)
#
simiBlockInterp = ns5.dataFrameToAnalogSignals(
    simiDFOutInterp,
    idxT='nspT', useColNames=True, probeName='',
    dataCol=simiDFOutInterp.drop(columns='nspT').columns,
    samplingRate=simiSamplingRate * pq.Hz)

seg = simiBlockInterp.segments[0]
simiBlockInterp.name = 'simi_data'
simiBlockInterp = ns5.purgeNixAnn(simiBlockInterp)
simiBlockInterp.create_relationship()

outPathName = os.path.join(
    scratchFolder,
    ns5FileName + '_simiTrigs.nix')
if os.path.exists(outPathName):
    os.remove(outPathName)
writer = neo.io.NixIO(filename=outPathName)
writer.write_block(simiBlockInterp, use_obj_names=True)
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
        simiBlockInterp, neoSegIdx=[0],
        writeAsigs=False, writeSpikes=True, writeEvents=True,
        fileName=BlackrockFileName + inputNSPBlockSuffix,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
'''
#
#
# get absolute timestamps of file extents (by Simi session)
# Simi session name -> absolute timestamp + time domain data last time for the end
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
# # # apply coarse offset to Simi data stream
# # # 
# # # if NSP data series has a handwritten time range, zero it outside that range
# # # detect threshold crossings from NSP data series (can be stim artifact, or TENS artifact or whatever)
# # #
# # #
