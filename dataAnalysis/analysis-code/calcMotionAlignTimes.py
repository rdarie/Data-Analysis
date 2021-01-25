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
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
# 
if arguments['processAll']:
    prefix = assembledName
    alignTimeBounds = []  # not working as of 12/31/19
    print('calcMotionStimAlignTimes does not support aggregate files')
    sys.exit()
# trick to allow joint processing of minirc and regular trials
if not ((blockExperimentType == 'proprio') or (blockExperimentType == 'proprio-motionOnly')):
    print('skipping trial with no movement')
    sys.exit()
#
prefix = ns5FileName
dataBlockPath = os.path.join(
    analysisSubFolder,
    prefix + '_analyze.nix')
print('loading {}'.format(dataBlockPath))
dataReader, dataBlock = preproc.blockFromPath(
    dataBlockPath, lazy=arguments['lazy'])
####
try:
    alignTimeBounds = alignTimeBoundsLookup[int(arguments['blockIdx'])]
except Exception:
    traceback.print_exc()
    fallbackTStart = float(
        dataBlock.segments[0]
        .filter(objects=AnalogSignalProxy)[0].t_start)
    fallbackTStop = float(
        dataBlock.segments[-1]
        .filter(objects=AnalogSignalProxy)[0].t_stop)
    alignTimeBounds = [
        [fallbackTStart, fallbackTStop]
    ]
    print(
        '\n Setting alignTimeBounds to {} -> {}'
        .format(fallbackTStart, fallbackTStop))
###
dummyCateg = [
    'amplitude', 'amplitudeCat', 'program',
    'RateInHz', 'electrode', 'activeGroup',
    'program']
availableCateg = [
    'pedalDirection', 'pedalSizeCat',
    'pedalSize', 'pedalMovementDuration']
signalsInAsig = [
    'velocity', 'position']

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
    if arguments['lazy']:
        asigProxysList = [
            asigP
            for asigP in dataSeg.filter(objects=AnalogSignalProxy)
            if asigP.name in signalsInSegment]
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
        dataSegEvents = [evP.load() for evP in eventProxysList]
        dummyAsig = asigsList[0]
    samplingRate = dummyAsig.sampling_rate
    #
    tdDF = preproc.analogSignalsToDataFrame(asigsList)
    tdDF.columns = [
        i.replace('seg{}_'.format(segIdx), '')
        for i in tdDF.columns
        ]
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
            'velocity': 'pedalVelocity',
            'position': 'pedalPosition'},
        inplace=True)
    #
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
    pedalPosQuantiles = tdDF.loc[taskMask, 'pedalPosition'].quantile([0, 1])
    pedalPosTolerance = (
        pedalPosQuantiles.iloc[1] -
        pedalPosQuantiles.iloc[0]) / 100
    tdDF.loc[:, 'pedalVelocityAbs'] = tdDF['pedalVelocity'].abs()
    pedalVelQuantiles = tdDF.loc[taskMask, 'pedalVelocityAbs'].quantile([0, 1])
    pedalVelTolerance = (
        pedalVelQuantiles.iloc[1] -
        pedalVelQuantiles.iloc[0]) / 100
    # pedalRestingMask = tdDF['pedalVelocityAbs'] < pedalVelTolerance
    pedalRestingMask = tdDF['pedalVelocityAbs'] == 0
    pedalNeutralPoint = float(
        tdDF.loc[taskMask & pedalRestingMask, 'pedalPosition'].value_counts().idxmax())
    #
    tdDF.loc[:, 'pedalPosition'] = tdDF['pedalPosition'] - pedalNeutralPoint
    tdDF.loc[:, 'pedalPositionAbs'] = tdDF['pedalPosition'].abs()
    crossIdx, crossMask = hf.getThresholdCrossings(
        tdDF.loc[taskMask, 'pedalPositionAbs'], thresh=pedalPosTolerance,
        edgeType='rising', fs=samplingRate,
        iti=1,  # at least 1 sec between triggers
        )
    movementOnOff = pd.Series(0, index=tdDF.index)
    movementOnOff.loc[crossIdx] = 1
    tdDF.loc[:, 'movementRound'] = movementOnOff.cumsum()
    tdDF.loc[:, 'movementRound'] -= 1  # 0 is all the stuff before the first movement
    movementCatTypes = ['outbound', 'reachedPeak', 'midPeak', 'return', 'reachedBase']
    trialsDict = {
        key: tdDF.loc[crossIdx, ['t', 'movementRound']].copy()
        for key in movementCatTypes
        }
    for mvCat in movementCatTypes:
        trialsDict[mvCat].index.name = 'outboundTdIdx'
        trialsDict[mvCat].set_index('movementRound', inplace=True)
        for catName in availableCateg + ['tdIndex']:
            trialsDict[mvCat].loc[:, catName] = np.nan
    for idx, (mvRound, group) in enumerate(tdDF.groupby('movementRound')):
        if mvRound >= 0:
            trialsDict['outbound'].loc[mvRound, 'tdIndex'] = group.index[0]
            #  back to start
            crossIdxFall, crossMaskFall = hf.getThresholdCrossings(
                group['pedalPositionAbs'], thresh=pedalPosTolerance,
                edgeType='falling', fs=samplingRate,
                iti=1,  # at least 1 sec between triggers
                )
            try:
                assert crossIdxFall.size == 1
                crossIdxFall = crossIdxFall[0]
            except Exception:
                print('\n\n{}\n Error at t = {} sec\n\n'.format(
                    dataBlockPath, group.loc[crossIdxFall, 't']))
                traceback.print_exc()
                pdb.set_trace()
            trialsDict['reachedBase'].loc[mvRound, 'tdIndex'] = crossIdxFall
            trialsDict['reachedBase'].loc[mvRound, 't'] = group.loc[crossIdxFall, 't']
            pedalSizeAbs = group['pedalPositionAbs'].quantile(1)
            #  reach peak
            crossIdxReachPeak, _ = hf.getThresholdCrossings(
                group.loc[:crossIdxFall, 'pedalPositionAbs'],
                thresh=pedalSizeAbs - pedalPosTolerance,
                edgeType='rising', fs=samplingRate,
                iti=1,  # at least 1 sec between triggers
                )
            try:
                assert crossIdxReachPeak.size == 1
                crossIdxReachPeak = crossIdxReachPeak[0]
            except Exception:
                print('\n\n{}\n Error at t = {} sec\n\n'.format(
                    dataBlockPath, group.loc[crossIdxReachPeak, 't']))
                traceback.print_exc()
                pdb.set_trace()
                # plt.plot(group['t'], group['pedalPosition'])
                # plt.show()
                # sns.distplot(tdDF.loc[taskMask & pedalRestingMask, 'pedalPosition'])
            trialsDict['reachedPeak'].loc[mvRound, 'tdIndex'] = crossIdxReachPeak
            trialsDict['reachedPeak'].loc[mvRound, 't'] = group.loc[crossIdxReachPeak, 't']
            #  return
            crossIdxReturn, _ = hf.getThresholdCrossings(
                group.loc[crossIdxReachPeak:crossIdxFall, 'pedalPositionAbs'],
                thresh=pedalSizeAbs - pedalPosTolerance,
                edgeType='falling', fs=samplingRate,
                iti=1,  # at least 1 sec between triggers
                )
            try:
                assert crossIdxReturn.size == 1
                crossIdxReturn = crossIdxReturn[0]
            except Exception:
                traceback.print_exc()
                pdb.set_trace()
            trialsDict['return'].loc[mvRound, 'tdIndex'] = crossIdxReturn
            trialsDict['return'].loc[mvRound, 't'] = (
                group.loc[crossIdxReturn, 't'])
            midPeakIdx = int(np.mean([crossIdxReachPeak, crossIdxReturn]))
            #  mid peak
            trialsDict['midPeak'].loc[mvRound, 'tdIndex'] = midPeakIdx
            trialsDict['midPeak'].loc[mvRound, 't'] = (
                group.loc[midPeakIdx, 't'])
            #
            movementDuration = float(
                trialsDict['reachedBase'].loc[mvRound, 't'] -
                trialsDict['outbound'].loc[mvRound, 't'])
            #
            for mvCat in movementCatTypes:
                trialsDict[mvCat].loc[mvRound, 'pedalSize'] = (
                    group.loc[midPeakIdx, 'pedalPosition'])
                if group.loc[midPeakIdx, 'pedalPosition'] > 0:
                    trialsDict[mvCat].loc[mvRound, 'pedalDirection'] = 'CW'
                else:
                    trialsDict[mvCat].loc[mvRound, 'pedalDirection'] = 'CCW'
                trialsDict[mvCat].loc[mvRound, 'pedalMovementDuration'] = (
                    movementDuration)
    #
    if (segIdx == 0) and arguments['plotParamHistograms']:
        ax = sns.distplot(
            trialsDict['midPeak'].loc[:, 'pedalSize'].abs(),
            bins=200, kde=False)
        plt.savefig(
            os.path.join(
                figureFolder, 'pedalSizeDistribution.pdf'))
        # plt.show()
        plt.close()
    #  determine size category
    pedalSizeCat = pd.cut(
        trialsDict['outbound']['pedalSize'].abs(), movementSizeBins,
        labels=movementSizeBinLabels)
    for mvCat in movementCatTypes:
        trialsDict[mvCat].loc[:, 'pedalSizeCat'] = pedalSizeCat
    # pdb.set_trace()
    alignEventsDF = (
        pd.concat(
            trialsDict, axis=0,
            names=['pedalMovementCat', 'movementRound'])
        .reset_index())
    #  sort align times
    alignEventsDF.sort_values(by='t', inplace=True, kind='mergesort')
    alignEvents = preproc.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_motionAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #
    concatLabelColumns = [
        'pedalMovementCat',
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
preproc.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeAsigs=False, writeSpikes=False, writeEvents=True,
    fileName=ns5FileName + '_analyze',
    folderPath=analysisSubFolder,
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )
