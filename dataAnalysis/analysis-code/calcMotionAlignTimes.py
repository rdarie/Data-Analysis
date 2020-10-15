"""10a: Calculate align Times
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                  which trial to analyze [default: 1]
    --exp=exp                            which experimental day to analyze
    --analysisName=analysisName          append a name to the resulting blocks? [default: default]
    --processAll                         process entire experimental day? [default: False]
    --plotParamHistograms                plot pedal size, amplitude, duration distributions? [default: False]
    --lazy                               load from raw, or regular? [default: False]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output by default
matplotlib.use('Qt5Agg')   # generate postscript output by default
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")
import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import quantities as pq
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.helper_functions_new as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from collections import Iterable
import sys
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
# 
if arguments['processAll']:
    prefix = assembledName
    alignTimeBounds = [] # not working as of 12/31/19
    print('calcMotionStimAlignTimes does not support aggregate files')
    sys.exit()
# trick to allow joint processing of minirc and regular trials
if not (blockExperimentType == 'proprio'):
    print('skipping RC trial')
    sys.exit()
#
try:
    alignTimeBounds = alignTimeBoundsLookup[int(arguments['blockIdx'])]
except Exception:
    traceback.print_exc()
    alignTimeBounds = None
#
prefix = ns5FileName
dataBlockPath = os.path.join(
    analysisSubFolder,
    prefix + '_analyze.nix')
print('loading {}'.format(dataBlockPath))
dataReader, dataBlock = preproc.blockFromPath(
    dataBlockPath, lazy=arguments['lazy'])
#
dummyCateg = [
    'amplitude', 'amplitudeCat', 'program',
    'RateInHz', 'electrode', 'activeGroup',
    'program']
availableCateg = [
    'pedalVelocityCat', 'pedalMovementCat', 'pedalDirection',
    'pedalSizeCat', 'pedalSize', 'pedalMovementDuration']
signalsInAsig = [
    'velocityCat', 'position']

#  allocate block to contain events
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=dataBlock.annotations['neo_name'])

blockIdx = 0
checkReferences = False
for segIdx, dataSeg in enumerate(dataBlock.segments):
    print('Calculating motion align times for trial {}'.format(segIdx))
    #
    signalsInSegment = [
        'seg{}_'.format(segIdx) + i
        for i in signalsInAsig]
    asigProxysList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if asigP.annotations['neo_name'] in signalsInSegment]
    eventProxysList = dataSeg.events
    if checkReferences:
        for asigP in asigProxysList:
            da = asigP._rawio.da_list['blocks'][blockIdx]['segments'][segIdx]['data']
            print('segIdx {}, asigP.name {}'.format(
                segIdx, asigP.name))
            print('asigP._global_channel_indexes = {}'.format(
                asigP._global_channel_indexes))
            print('asigP references {}'.format(
                da[asigP._global_channel_indexes[0]]))
            try:
                assert asigP.name in da[asigP._global_channel_indexes[0]].name
            except Exception:
                traceback.print_exc()
        for evP in eventProxysList:
            print('segIdx {}, evP.name {}'.format(
                segIdx, evP.name))
            print('evP._event_channel_index = {}'.format(
                 evP._event_channel_index))
            evP_ch = evP._event_channel_index
            mts = evP._rawio.file.blocks[blockIdx].groups[segIdx].multi_tags
            try:
                assert evP.name in mts[evP_ch].name
            except Exception:
                traceback.print_exc()
    asigsList = [
        asigP.load()
        for asigP in asigProxysList]
    samplingRate = asigsList[0].sampling_rate
    # asigsDF = preproc.analogSignalsToDataFrame(asigsList, useChanNames=True)
    tdDF = preproc.analogSignalsToDataFrame(asigsList)
    tdDF.columns = [
        i.replace('seg{}_'.format(segIdx), '')
        for i in tdDF.columns
        ]
    #
    dataSegEvents = [evP.load() for evP in eventProxysList]
    eventDF = preproc.eventsToDataFrame(
        dataSegEvents, idxT='t',
        names=[
            'seg{}_property'.format(segIdx),
            'seg{}_value'.format(segIdx)]
        )
    eventDF.columns = [
        i.replace('seg{}_'.format(segIdx), '')
        for i in eventDF.columns
        ]
    tdDF.rename(
        columns={
            'velocityCat': 'pedalVelocityCat',
            'position': 'pedalPosition'},
        inplace=True)
    tdDF['pedalVelocityCat'] = tdDF['pedalVelocityCat'] - 1
    #  get alignment times
    moveMask = pd.Series(False, index=tdDF.index)
    stopMask = pd.Series(False, index=tdDF.index)

    if alignTimeBounds is not None:
        taskMask = pd.Series(False, index=tdDF.index)
        for idx, atb in enumerate(alignTimeBounds):
            taskMask = (
                (taskMask) | (
                    (tdDF['t'] > atb[0]) &
                    (tdDF['t'] < atb[1])
                    )
                )
    else:
        taskMask = (tdDF['t'] >= 0)
    ###################################################
    fs = asigsList[0].sampling_rate.magnitude
    pedalNeutralPoint = tdDF.loc[taskMask, 'pedalPosition'].iloc[0]
    centeredPedalPosition = tdDF['pedalPosition'] - pedalNeutralPoint
    pedalPosAbs = centeredPedalPosition.abs()
    pedalPosThresh = pedalPosAbs.max() / 1000
    minDist = 25e-3
    lowPassVelocity = hf.filterDF(
        tdDF.loc[:, 'pedalVelocityCat'],
        fs,
        lowPass=25, lowOrder=6)
    #
    _, startLowMask = hf.getThresholdCrossings(
        lowPassVelocity, thresh=-0.5, absVal=False,
        edgeType='falling', fs=fs, iti=minDist,
        plotting=False, keep_max=False, itiWiggle=0.05)
    _, startHighMask = hf.getThresholdCrossings(
        lowPassVelocity, thresh=0.5, absVal=False,
        edgeType='rising', fs=fs, iti=minDist,
        plotting=False, keep_max=False, itiWiggle=0.05)
    _, stopLowMask = hf.getThresholdCrossings(
        lowPassVelocity, thresh=-0.5, absVal=False,
        edgeType='rising', fs=fs, iti=minDist,
        plotting=False, keep_max=False, itiWiggle=0.05)
    _, stopHighMask = hf.getThresholdCrossings(
        lowPassVelocity, thresh=0.5, absVal=False,
        edgeType='falling', fs=fs, iti=minDist,
        plotting=False, keep_max=False, itiWiggle=0.05)
    #
    moveMaskForSeg = (startLowMask | startHighMask) & taskMask
    stopMaskForSeg = (stopLowMask | stopHighMask) & taskMask
    ################
    # movingAtAll = tdDF['pedalVelocityCat'].abs()
    # movementOnOff = movingAtAll.diff().fillna(0)
    # # guard against blips in the velocity rating
    # moveStartIndices = movementOnOff.index[movementOnOff.abs() > 0]
    # blipMask = tdDF.loc[moveStartIndices, 't'].diff() < 30e-3
    # movementOnOff.loc[blipMask.index[blipMask]] = 0
    # #
    checkMoveOnOff = False
    if checkMoveOnOff:
        plt.plot(tdDF['t'], tdDF['pedalPosition'].values, label='position')
        plt.plot(tdDF['t'], tdDF['pedalVelocityCat'].values, label='velocity (int)')
        plt.plot(tdDF['t'], lowPassVelocity, label='filtered velocity')
        plt.plot(tdDF['t'], startHighMask, label='start high')
        plt.plot(tdDF['t'], stopHighMask, label='stop high')
        plt.plot(tdDF['t'], startLowMask, label='start low')
        plt.plot(tdDF['t'], stopLowMask, label='stop low')
        plt.legend(); plt.show()
    # #
    # moveMaskForSeg = (movementOnOff == 1) & taskMask
    # stopMaskForSeg = (movementOnOff == -1) & taskMask
    ################
    print('Found {} movement starts'.format(moveMaskForSeg.sum()))
    print('Found {} movement stops'.format(stopMaskForSeg.sum()))
    # pdb.set_trace()
    assert stopMaskForSeg.sum() == moveMaskForSeg.sum(), 'unequal start and stop lengths'
    assert stopMaskForSeg.sum() % 2 == 0, 'number of movements not divisible by 2'
    moveMask.loc[moveMaskForSeg.index[moveMaskForSeg]] = True
    stopMask.loc[stopMaskForSeg.index[stopMaskForSeg]] = True
    moveTimes = tdDF.loc[
        moveMask, 't']
    stopTimes = tdDF.loc[
        stopMask, 't']
    tdDF['pedalMovementCat'] = np.nan
    for idx, dfMoveIdx in enumerate(moveMask.index[moveMask]):
        dfStopIndex = stopMask.index[stopMask][idx]
        assert dfStopIndex > dfMoveIdx
        if (idx % 2) == 0:
            tdDF.loc[dfMoveIdx, 'pedalMovementCat'] = 'outbound'
            tdDF.loc[dfStopIndex, 'pedalMovementCat'] = 'reachedPeak'
        else:
            tdDF.loc[dfMoveIdx, 'pedalMovementCat'] = 'return'
            tdDF.loc[dfStopIndex, 'pedalMovementCat'] = 'reachedBase'
    outboundMask = tdDF['pedalMovementCat'] == 'outbound'
    reachedPeakMask = tdDF['pedalMovementCat'] == 'reachedPeak'
    returnMask = tdDF['pedalMovementCat'] == 'return'
    reachedBaseMask = tdDF['pedalMovementCat'] == 'reachedBase'
    #  check that the trials are intact
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedBase').sum())
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedPeak').sum())
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'return').sum())
    #  calculate movement sizes (relative to starting point)
    midPeakIdx = ((
        returnMask[returnMask].index +
        reachedPeakMask[reachedPeakMask].index) / 2).astype('int64')
    tdDF['pedalSize'] = np.nan
    tdDF.loc[midPeakIdx, 'pedalSize'] = tdDF.loc[midPeakIdx, 'pedalPosition']
    tdDF['pedalSize'].interpolate(method='nearest', inplace=True)
    tdDF['pedalSize'].fillna(method='ffill', inplace=True)
    tdDF['pedalSize'].fillna(method='bfill', inplace=True)
    #
    tdDF.loc[:, 'pedalSize'] = tdDF['pedalSize'] - pedalNeutralPoint
    #  plt.plot(tdDF['t'], tdDF['pedalSize'])
    #  plt.plot(tdDF['t'], tdDF['pedalPosition']); plt.show()
    if (segIdx == 0) and arguments['plotParamHistograms']:
        ax = sns.distplot(
            tdDF.loc[midPeakIdx, 'pedalSize'].abs(),
            bins=200, kde=False)
        plt.savefig(
            os.path.join(
                figureFolder, 'pedalSizeDistribution.pdf'))
        # plt.show()
        plt.close()
    #  determine size category

    tdDF['pedalSizeCat'] = pd.cut(
        tdDF['pedalSize'].abs(), movementSizeBins,
        labels=movementSizeBinLabels)
    #  determine CW or CCW
    tdDF['pedalDirection'] = np.nan
    tdDF.loc[tdDF['pedalSize'] > 0, 'pedalDirection'] = 'CW'
    tdDF.loc[tdDF['pedalSize'] <= 0, 'pedalDirection'] = 'CCW'
    #  calculate movement durations
    tdDF['pedalMovementDuration'] = np.nan
    outboundTimes = tdDF.loc[
        outboundMask,
        't']
    reachedBaseTimes = tdDF.loc[
        reachedBaseMask,
        't']
    pedalMovementDurations = (
        reachedBaseTimes.values -
        outboundTimes.values
        )
    tdDF.loc[outboundTimes.index, 'pedalMovementDuration'] = (
        pedalMovementDurations
        )
    tdDF.loc[reachedBaseTimes.index, 'pedalMovementDuration'] = (
        pedalMovementDurations
        )
    # import seaborn as sns
    # sns.distplot(tdDF['pedalMovementDuration'].dropna())
    tdDF['pedalMovementDuration'].interpolate(method='nearest', inplace=True)
    tdDF['pedalMovementDuration'].fillna(method='ffill', inplace=True)
    tdDF['pedalMovementDuration'].fillna(method='bfill', inplace=True)

if True:
    peakTimes = tdDF.loc[midPeakIdx, 't']
    #  get intervals halfway between move stop and move start
    pauseLens = moveTimes.shift(-1).values - stopTimes
    maskForLen = pauseLens > 1.5
    halfOffsets = (
        samplingRate.magnitude * (pauseLens / 2)).fillna(0).astype(int)
    otherTimesIdx = (stopTimes.index + halfOffsets.values)[maskForLen]
    otherTimes = tdDF.loc[otherTimesIdx, 't']

    moveCategories = tdDF.loc[
        tdDF['t'].isin(moveTimes), availableCateg
        ].reset_index(drop=True)
    moveCategories['pedalMetaCat'] = 'onset'
    
    stopCategories = tdDF.loc[
        tdDF['t'].isin(stopTimes), availableCateg
        ].reset_index(drop=True)
    stopCategories['pedalMetaCat'] = 'offset'
    
    otherCategories = tdDF.loc[
        tdDF['t'].isin(otherTimes), availableCateg
        ].reset_index(drop=True)
    otherCategories['pedalMetaCat'] = 'control'
    
    peakCategories = tdDF.loc[
        tdDF['t'].isin(peakTimes), availableCateg
        ].reset_index(drop=True)
    peakCategories['pedalMetaCat'] = 'midPeak'
    peakCategories['pedalMovementCat'] = 'midPeak'
    # pdb.set_trace()
    alignTimes = pd.concat((
        moveTimes, stopTimes, otherTimes, peakTimes),
        axis=0, ignore_index=True)
    #  sort align times
    alignTimes.sort_values(inplace=True, kind='mergesort')
    categories = pd.concat((
        moveCategories, stopCategories, otherCategories, peakCategories),
        axis=0, ignore_index=True)
    categories['program'] = 999
    categories['RateInHz'] = 0
    categories['amplitude'] = 0
    categories['electrode'] = 'NA'
    categories['amplitudeCat'] = 0
    #  sort categories by align times
    #  (needed to propagate values forward)
    # pdb.set_trace()
    categories = categories.loc[alignTimes.index, :]
    alignTimes.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    #
    alignEventsDF = pd.concat([alignTimes, categories], axis=1)
    alignEvents = preproc.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_motionAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #
    concatLabelColumns = [
        'pedalMovementCat', 'pedalMetaCat',
        'pedalSizeCat', 'pedalDirection',
        'pedalMovementDuration']
    concatLabelsDF = alignEventsDF[concatLabelColumns]
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatLabelsDF.iterrows()])
    concatEvents = Event(
        name='seg{}_motionAlignTimesConcatenated'.format(segIdx),
        times=alignEvents.times,
        labels=concatLabels
        )
    newSeg = Segment(name=dataSeg.annotations['neo_name'])
    newSeg.annotate(nix_name=dataSeg.annotations['neo_name'])
    newSeg.events.append(alignEvents)
    newSeg.events.append(concatEvents)
    alignEvents.segment = newSeg
    concatEvents.segment = newSeg
    masterBlock.segments.append(newSeg)

dataReader.file.close()
masterBlock.create_relationship()
allSegs = list(range(len(masterBlock.segments)))
if arguments['processAll']:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeAsigs=False, writeSpikes=False, writeEvents=True,
        fileName=experimentName + '_analyze',
        folderPath=analysisSubFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
else:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeAsigs=False, writeSpikes=False, writeEvents=True,
        fileName=ns5FileName + '_analyze',
        folderPath=analysisSubFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
