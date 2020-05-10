"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --lazy                          load from raw, or regular? [default: False]
    --showPlots                     whether to show diagnostic plots (must have display) [default: False]
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
plotting = arguments['showPlots']
nspPath = os.path.join(
    scratchFolder,
    ns5FileName + '.nix')
nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=False)
#
oePath = os.path.join(
    scratchFolder,
    ns5FileName + '_oe.nix')
oeReader, oeBlock = ns5.blockFromPath(
    oePath, lazy=False)

segIdx = 0
nspSeg = nspBlock.segments[segIdx]
oeSeg = oeBlock.segments[segIdx]

fs = 30000
nCrossings = 50000
#
oeSyncAsig = oeSeg.filter(name='seg0_ADC1')[0]
tStart = synchInfo['oe'][blockIdx][0]['timeRangesKinect'][0][0]
tStop = synchInfo['oe'][blockIdx][0]['timeRangesKinect'][0][1]
oeThresh = -2
oeTimeMask = hf.getTimeMaskFromRanges(
    oeSyncAsig.times, [(tStart, tStop)])
oeSrs = pd.Series(oeSyncAsig.magnitude[oeTimeMask].flatten())
print('On trial {}, detecting OE threshold crossings.'.format(blockIdx))
oePeakIdx, oeCrossMask = hf.getThresholdCrossings(
    oeSrs, thresh=oeThresh,
    iti=1e-4, fs=fs,
    absVal=False, plotting=plotting, keep_max=False)
oeTimes = oeSyncAsig.times[oeTimeMask][oeCrossMask][:nCrossings]
#
nspSyncAsig = nspSeg.filter(name='seg0_ainp16')[0]
tStart = synchInfo['nsp'][blockIdx][0]['timeRangesKinect'][0][0]
tStop = synchInfo['nsp'][blockIdx][0]['timeRangesKinect'][0][1]
nspThresh = 600
nspTimeMask = hf.getTimeMaskFromRanges(
    nspSyncAsig.times, [(tStart, tStop)])
nspSrs = pd.Series(nspSyncAsig.magnitude[nspTimeMask].flatten())
print('On trial {}, detecting NSP threshold crossings.'.format(blockIdx))
nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
    nspSrs, thresh=nspThresh,
    iti=1e-4, fs=fs,
    absVal=False, plotting=plotting, keep_max=False)
nspTimes = nspSyncAsig.times[nspTimeMask][nspCrossMask][:nCrossings]
assert np.max(np.abs((np.diff(oeTimes) - np.diff(nspTimes)))) < 1e-4

synchPolyCoeffs = np.polyfit(
    x=oeTimes,
    y=nspTimes,
    deg=1)
#  synchPolyCoeffs = np.array([1, np.mean(emgTensTimes - insTensTimes)])
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

oeInterp = hf.interpolateDF(
    oeDF, newT,
    kind='linear', fill_value=(0, 0),
    x='nspT', columns=interpCols)
oeInterpBlock = ns5.dataFrameToAnalogSignals(
    oeInterp,
    idxT='nspT',
    probeName='openEphys', samplingRate=dummyAsig.sampling_rate,
    dataCol=interpCols,
    forceColNames=interpCols)
oeBlockJustEvents = hf.extractSignalsFromBlock(oeBlock)
ns5.addBlockToNIX(
    oeBlockJustEvents, neoSegIdx=[0],
    writeAsigs=False, writeSpikes=False, writeEvents=True,
    purgeNixNames=True,
    fileName=ns5FileName,
    folderPath=scratchFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )
ns5.addBlockToNIX(
    oeInterpBlock, neoSegIdx=[0],
    writeAsigs=True, writeSpikes=False, writeEvents=False,
    purgeNixNames=True,
    fileName=ns5FileName,
    folderPath=scratchFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )
