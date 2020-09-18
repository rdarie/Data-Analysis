"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze
    --exp=exp                              which experimental day to analyze
    --lazy                                 load from raw, or regular? [default: False]
    --trigRate=trigRate                    inter trigger interval [default: 60]
    --plotting                             whether to show diagnostic plots (must have display) [default: False]
    --inputBlockSuffix=inputBlockSuffix    append a name to the resulting blocks?
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
from namedQueries import namedQueries
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.mdt as preprocINS
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb
from importlib import reload

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['inputBlockSuffix'] is not None:
    ns5FileName = ns5FileName + arguments['inputBlockSuffix']
nspPath = os.path.join(
    scratchFolder,
    ns5FileName.replace('Block', 'utah') + '.nix')
#
oePath = os.path.join(
    scratchFolder,
    ns5FileName.replace('Block', 'nform') + '.nix')
oeReader, oeBlock = ns5.blockFromPath(
    oePath, lazy=arguments['lazy'])

nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=arguments['lazy'])
interTriggerInterval = float(arguments['trigRate']) ** (-1)
segIdx = 0
nspSeg = nspBlock.segments[segIdx]
oeSeg = oeBlock.segments[segIdx]
#
oeSyncAsig = oeSeg.filter(name='seg0_analog 1')[0]
tStart, tStop = synchInfo['nform'][blockIdx]['timeRanges']
#
oeTimeMask = hf.getTimeMaskFromRanges(
    oeSyncAsig.times, [(tStart, tStop)])
oeSrs = pd.Series(oeSyncAsig.magnitude[oeTimeMask].flatten())
print(
    'On block {}, detecting NForm threshold crossings.'
    .format(blockIdx))

oeLims = oeSrs.quantile([1e-6, 1-1e-6]).to_list()
oeDiffUncertainty = oeSrs.diff().abs().quantile(1-1e-6) / 4
oeThresh = (oeLims[-1] - oeLims[0]) / 2

nspSyncAsig = nspSeg.filter(name='seg0_ainp16')[0]
tStart, tStop = synchInfo['nsp'][blockIdx]['timeRanges']

nspTimeMask = hf.getTimeMaskFromRanges(
    nspSyncAsig.times, [(tStart, tStop)])
nspSrs = pd.Series(nspSyncAsig.magnitude[nspTimeMask].flatten())

nspThresh = 0
#
oePeakIdx, oeCrossMask = hf.getThresholdCrossings(
    oeSrs, thresh=oeThresh,
    iti=interTriggerInterval, fs=float(oeSyncAsig.sampling_rate),
    edgeType='both', itiWiggle=.1,
    absVal=False, plotting=arguments['plotting'], keep_max=False)
# oePeakIdx = hf.getTriggers(
#     oeSrs, iti=interTriggerInterval, fs=float(oeSyncAsig.sampling_rate),
#     thres=1.5, edgeType='falling', plotting=arguments['plotting'])
# oeCrossMask = oeSrs.index.isin(oePeakIdx)
print('Found {} triggers'.format(oePeakIdx.size))
#
print(
    'On trial {}, detecting NSP threshold crossings.'
    .format(blockIdx))
nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
    nspSrs, thresh=nspThresh,
    iti=interTriggerInterval, fs=float(nspSyncAsig.sampling_rate),
    edgeType='both', itiWiggle=.1,
    absVal=False, plotting=arguments['plotting'], keep_max=False)
# nspPeakIdx = hf.getTriggers(
#     nspSrs, iti=interTriggerInterval, itiWiggle=1,
#     fs=float(oeSyncAsig.sampling_rate), plotting=arguments['plotting'],
#     thres=2.58, edgeType='both')
# nspCrossMask = nspSrs.index.isin(nspPeakIdx)
print('Found {} triggers'.format(nspPeakIdx.size))
oeTimes = (
    oeSyncAsig.times[oeTimeMask][oeCrossMask][synchInfo['nform'][blockIdx]['chooseCrossings']])
nspTimes = (
    nspSyncAsig.times[nspTimeMask][nspCrossMask][synchInfo['nsp'][blockIdx]['chooseCrossings']])
###########
nMissingTriggers = nspTimes.size - oeTimes.size
sampleWiggle = 5 * oeSyncAsig.sampling_rate.magnitude ** (-1)
prelimOEMismatch = np.abs(np.diff(oeTimes) - interTriggerInterval * pq.s)
prelimNSPMismatch = np.abs(np.diff(nspTimes) - interTriggerInterval * pq.s)
listDiscontinuitiesNSP = np.flatnonzero(prelimNSPMismatch > sampleWiggle)
listDiscontinuitiesOE = np.flatnonzero(prelimOEMismatch > sampleWiggle)
# 
if nMissingTriggers > 0:
    # np.diff(oeTimes)[listDiscontinuitiesOE]
    # np.diff(nspTimes)[listDiscontinuitiesNSP]
    # nspTimes[listDiscontinuitiesNSP]
    # oeTimes[listDiscontinuitiesOE]
    # nspTimes[listDiscontinuitiesNSP] - nspTimes[0]
    # oeTimes[listDiscontinuitiesOE] - oeTimes[0]
    listDiscontinuities = listDiscontinuitiesOE
    nMissingTriggers = nspTimes.size - oeTimes.size
    print('Found {} discontinuities!'.format(len(listDiscontinuities)))
else:
    # 
    listDiscontinuities = np.flatnonzero(np.abs(prelimNSPMismatch - prelimOEMismatch) > sampleWiggle)
if len(listDiscontinuities):
    print(' On Delsys clock, discontinuities at:')
    for dIdx in listDiscontinuities:
        print(oeTimes[dIdx])
    oeDiscRound = np.zeros_like(oeTimes.magnitude)
    nspDiscRound = np.zeros_like(nspTimes.magnitude)
    for j, discIdx in enumerate(listDiscontinuities):
        oeDiscRound[discIdx+1:] += 1
        if nMissingTriggers > 0:
            nspDiscRound[discIdx+1+j] = 999  # use 999 as a discard marker
            nMissingTriggers -= 1
            print('Skipping NSP pulse at t={:.3f}'.format(nspTimes[discIdx+1+j]))
            nspDiscRound[discIdx+2+j:] += 1
        else:
            nspDiscRound[discIdx+1:] += 1
    if np.sum(nspDiscRound < 999) > np.sum(oeDiscRound < 999):
        # if there are more nsp triggers at the end, discard
        nspDiscRound[np.sum(oeDiscRound < 999):] = 999
    if np.sum(oeDiscRound < 999) > np.sum(nspDiscRound < 999):
        # if there are more nsp triggers at the end, discard
        oeDiscRound[np.sum(nspDiscRound < 999):] = 999
    pwSyncDict = {}  # piecewise sync parameters
    uniqueOeRounds = np.unique(oeDiscRound[oeDiscRound < 999])
    for roundIdx in uniqueOeRounds:
        thesePolyCoeffs = np.polyfit(
            x=oeTimes[oeDiscRound == roundIdx],
            y=nspTimes[nspDiscRound == roundIdx], deg=1)
        thisInterpFun = np.poly1d(thesePolyCoeffs)
        if roundIdx == 0:
            pwSyncDict[roundIdx] = {
                'inStart': 0,
                'inStop': np.max(oeTimes[oeDiscRound == roundIdx].magnitude),
                'tInterpFun': thisInterpFun}
        elif roundIdx == uniqueOeRounds[-1]:
            pwSyncDict[roundIdx] = {
                'inStart': np.max(oeTimes[oeDiscRound == roundIdx-1].magnitude),
                'inStop': 1e6,
                'tInterpFun': thisInterpFun}
        else:
            pwSyncDict[roundIdx] = {
                'inStart': np.max(oeTimes[oeDiscRound == roundIdx-1].magnitude),
                'inStop': np.max(oeTimes[oeDiscRound == roundIdx].magnitude),
                'tInterpFun': thisInterpFun}
    #
    def timeInterpFun(inputT):
        outputT = np.zeros_like(inputT)
        for k in sorted(pwSyncDict.keys()):
            inTimeMask = (
                (inputT >= pwSyncDict[k]['inStart']) &
                (inputT < pwSyncDict[k]['inStop']))
            outputT[inTimeMask] = pwSyncDict[k]['tInterpFun'](
                inputT[inTimeMask])
        plotting = False
        if plotting:
            import matplotlib.pyplot as plt
            for k in sorted(pwSyncDict.keys()):
                inTimeMask = (
                    (inputT >= pwSyncDict[k]['inStart']) &
                    (inputT < pwSyncDict[k]['inStop']))
                plt.plot(inputT[inTimeMask], outputT[inTimeMask])
            plt.show()
        return outputT
else:
    # assert np.max(np.abs((np.diff(oeTimes) - np.diff(nspTimes)))) < 1e-4
    # np.flatnonzero(np.abs(np.diff(oeTimes)) > 0.018)
    # np.sum(np.abs(np.diff(nspTimes)) < 0.017)
    # nspSynchDur = nspTimes[-1] - nspTimes[0]
    # oeSynchDur = oeTimes[-1] - oeTimes[0]
    ###########
    if oeTimes.size > nspTimes.size:
        # if there are more nsp triggers at the end, discard
        oeTimes = oeTimes[:nspTimes.size]
    if nspTimes.size > oeTimes.size:
        # if there are more nsp triggers at the end, discard
        nspTimes = nspTimes[:oeTimes.size]
    synchPolyCoeffs = np.polyfit(x=oeTimes, y=nspTimes, deg=1)
    # synchPolyCoeffs = np.array([1, np.mean(nspTimes - oeTimes)])
    # synchPolyCoeffs = np.array([1, np.mean(nspTimes[0] - oeTimes[0])])
    timeInterpFun = np.poly1d(synchPolyCoeffs)

dummyAsigNSP = nspSeg.filter(objects=AnalogSignal)[0]
newT = pd.Series(dummyAsigNSP.times.magnitude)
#

tStart = dummyAsigNSP.times[0]
tStop = dummyAsigNSP.times[-1]
uniqueSt = []
oeDF = ns5.analogSignalsToDataFrame(
    oeBlock.filter(objects=AnalogSignal),
    idxT='oeT', useChanNames=True)

for segIdx, seg in enumerate(oeBlock.segments):
    for stIdx, st in enumerate(seg.spiketrains):
        if seg.spiketrains[stIdx] not in uniqueSt:
            uniqueSt.append(seg.spiketrains[stIdx])
        else:
            continue
        if len(st.times):
            st.magnitude[:] = (
                timeInterpFun(st.times[:].magnitude))
            #  kludgey fix for weirdness concerning t_start
            st.t_start = max(tStart, st.times[0] * 0.999)
            st.t_stop = min(tStop, st.times[-1] * 1.001)
            validMask = (st < tStop) & (st > tStart)
            if (~validMask).any():
                print('Deleted some spikes')
                # invalidIndex = np.flatnonzero(~validMask)
                print('{}'.format(st[~validMask]))
                # st = np.delete(st, invalidIndex)
                st = st[validMask]
                # delete invalid spikes
                if 'arrayAnnNames' in st.annotations.keys():
                    for key in st.annotations['arrayAnnNames']:
                        st.annotations[key] = np.array(st.array_annotations[key])
        else:
            st.t_start = tStart
            st.t_stop = tStop
        if st.waveforms is None:
            st.sampling_rate = 3e4*pq.Hz
            st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
        # st becomes a copy somewhere in here, must reassign
        # find the unit and replace
        thisUnit = seg.spiketrains[stIdx].unit
        thisUnit.spiketrains[segIdx] = st
        seg.spiketrains[stIdx] = st

print('len(oeBlock.filter(objects=SpikeTrain))')
print(len(oeBlock.filter(objects=SpikeTrain)))
'''
print([st.times[-1] for st in oeBlock.filter(objects=SpikeTrain)])
print([st.unit for st in oeBlock.filter(objects=SpikeTrain)])
print([st.segment for st in oeBlock.filter(objects=SpikeTrain)])
print([len(un.spiketrains) for un in oeBlockJustSpikes.filter(objects=Unit)])

print([id(st) for st in oeBlockJustSpikes.filter(objects=SpikeTrain)])
oeBlockJustSpikes.filter(objects=SpikeTrain, name='seg0_nform_64#0')
'''
ns5.addBlockToNIX(
    oeBlock, neoSegIdx=[0],
    writeAsigs=False, writeSpikes=True, writeEvents=False,
    purgeNixNames=True,
    fileName=ns5FileName.replace('Block', 'utah'),
    folderPath=scratchFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )

oeDF.loc[:, 'nspT'] = timeInterpFun(oeDF['oeT'])
#
dummyAsigOe = oeSeg.filter(objects=AnalogSignal)[0]
#
interpCols = oeDF.columns.drop(['oeT', 'nspT'])
#
oeInterp = hf.interpolateDF(
    oeDF, newT,
    kind='pchip', fill_value=(0, 0),
    x='nspT', columns=interpCols, verbose=True)
#
del oeDF
#
oeInterpBlock = ns5.dataFrameToAnalogSignals(
    oeInterp,
    idxT='nspT',
    probeName='openEphys', samplingRate=dummyAsigNSP.sampling_rate,
    dataCol=interpCols, forceColNames=interpCols, verbose=True)
#
ns5.addBlockToNIX(
    oeInterpBlock, neoSegIdx=[0],
    writeAsigs=True, writeSpikes=False, writeEvents=False,
    purgeNixNames=True,
    fileName=ns5FileName.replace('Block', 'utah'),
    folderPath=scratchFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )
