"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --lazy                          load from raw, or regular? [default: False]
    --plotting                      whether to show diagnostic plots (must have display) [default: False]
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

nspPath = os.path.join(
    scratchFolder,
    ns5FileName + '.nix')
nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=arguments['lazy'])
#
oePath = os.path.join(
    scratchFolder,
    ns5FileName + '_delsys.nix')
oeReader, oeBlock = ns5.blockFromPath(
    oePath, lazy=arguments['lazy'])

segIdx = 0
nspSeg = nspBlock.segments[segIdx]
oeSeg = oeBlock.segments[segIdx]

chooseCrossings = slice(None, 59000)
# chooseCrossings = [0, -1]
# chooseCrossings = [i for i in range(1000)] + [i for i in range(-1000, -1)]
# chooseCrossings = slice(None)
#
oeSyncAsig = oeSeg.filter(name='seg0_AnalogInputAdapterAnalog')[0]
tStart, tStop = synchInfo['delsys'][blockIdx]['timeRanges']
#
oeThresh = 1.5
oeTimeMask = hf.getTimeMaskFromRanges(
    oeSyncAsig.times, [(tStart, tStop)])
oeSrs = pd.Series(oeSyncAsig.magnitude[oeTimeMask].flatten())
print('On block {}, detecting Delsys threshold crossings.'.format(blockIdx))
oePeakIdx, oeCrossMask = hf.getThresholdCrossings(
    oeSrs, thresh=oeThresh,
    iti=60 ** (-1), fs=float(oeSyncAsig.sampling_rate),
    edgeType='falling', itiWiggle=0.1,
    absVal=False, plotting=arguments['plotting'], keep_max=False)
#
nspSyncAsig = nspSeg.filter(name='seg0_analog 1')[0]
tStart, tStop = alignTimeBoundsLookup[blockIdx][0]
nspThresh = 1500
nspTimeMask = hf.getTimeMaskFromRanges(
    nspSyncAsig.times, [(tStart, tStop)])
nspSrs = pd.Series(nspSyncAsig.magnitude[nspTimeMask].flatten())
print('On trial {}, detecting NSP threshold crossings.'.format(blockIdx))
nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
    nspSrs, thresh=nspThresh,
    iti=60 ** (-1), fs=float(nspSyncAsig.sampling_rate),
    edgeType='falling', itiWiggle=0.1,
    absVal=False, plotting=arguments['plotting'], keep_max=False)
pdb.set_trace()
oeTimes = oeSyncAsig.times[oeTimeMask][oeCrossMask][chooseCrossings]
nspTimes = nspSyncAsig.times[nspTimeMask][nspCrossMask][chooseCrossings]
###########
# assert np.max(np.abs((np.diff(oeTimes) - np.diff(nspTimes)))) < 1e-4
# np.flatnonzero(np.abs(np.diff(oeTimes)) > 0.018)
# np.sum(np.abs(np.diff(nspTimes)) < 0.017)
# nspSynchDur = nspTimes[-1] - nspTimes[0]
# oeSynchDur = oeTimes[-1] - oeTimes[0]
###########
synchPolyCoeffs = np.polyfit(x=oeTimes, y=nspTimes, deg=1)
# synchPolyCoeffs = np.array([1, np.mean(nspTimes - oeTimes)])
timeInterpFun = np.poly1d(synchPolyCoeffs)
for event in oeBlock.filter(objects=Event):
    event.magnitude[:] = (
        timeInterpFun(event.times.magnitude))
for asig in oeBlock.filter(objects=AnalogSignal):
    asig.times.magnitude[:] = (
        timeInterpFun(asig.times.magnitude))
for st in oeBlock.filter(objects=SpikeTrain):
    st.times.magnitude[:] = (
        timeInterpFun(st.times.magnitude))
oeDF = ns5.analogSignalsToDataFrame(
    oeBlock.filter(objects=AnalogSignal),
    idxT='oeT', useChanNames=True)
oeDF['nspT'] = timeInterpFun(oeDF['oeT'])
dummyAsig = nspSeg.filter(objects=AnalogSignal)[0]
newT = pd.Series(dummyAsig.times.magnitude)
interpCols = oeDF.columns.drop(['oeT', 'nspT'])
#
oeInterp = hf.interpolateDF(
    oeDF, newT,
    kind='linear', fill_value=(0, 0),
    x='nspT', columns=interpCols)
oeInterpBlock = ns5.dataFrameToAnalogSignals(
    oeInterp,
    idxT='nspT',
    probeName='openEphys', samplingRate=dummyAsig.sampling_rate,
    dataCol=interpCols, forceColNames=interpCols)
# oeBlockJustEvents = hf.extractSignalsFromBlock(oeBlock)
# ns5.addBlockToNIX(
#     oeBlockJustEvents, neoSegIdx=[0],
#     writeAsigs=False, writeSpikes=False, writeEvents=True,
#     purgeNixNames=True,
#     fileName=ns5FileName,
#     folderPath=scratchFolder,
#     nixBlockIdx=0, nixSegIdx=[0],
#     )
ns5.addBlockToNIX(
    oeInterpBlock, neoSegIdx=[0],
    writeAsigs=True, writeSpikes=False, writeEvents=False,
    purgeNixNames=True,
    fileName=ns5FileName,
    folderPath=scratchFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )
