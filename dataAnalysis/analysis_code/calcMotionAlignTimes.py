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
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('PS')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
#
import matplotlib.pyplot as plt
import seaborn as sns
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
import dataAnalysis.preproc.ns5 as ns5
# import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from collections import Iterable
import sys
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt

sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
###
#!! applyDelay: Move align time by this factor, to account for build-up to the threshold crossing
#               applyDelay should be positive, we'll resolve the sign later
applyDelay = 10e-3

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
figureOutputFolder = os.path.join(
    scratchFolder, 'preprocDiagnostics'
    )
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
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
dataReader, dataBlock = ns5.blockFromPath(
    dataBlockPath, lazy=arguments['lazy'])
####
try:
    alignTimeBounds = alignTimeBoundsLookup[int(arguments['blockIdx'])]
except Exception:
    print('No alignTimeBounds read...')
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
    'trialAmplitude', 'amplitudeCat', 'program',
    'trialRateInHz', 'electrode', 'activeGroup',
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
        asigsList = [
            asigP.load()
            for asigP in asigProxysList]
        dataSegEvents = [evP.load() for evP in eventProxysList]
        dummyAsig = asigsList[0]
    samplingRate = dummyAsig.sampling_rate
    #
    tdDF = ns5.analogSignalsToDataFrame(asigsList)
    tdDF.columns = [
        i.replace('seg{}_'.format(segIdx), '')
        for i in tdDF.columns
        ]
    eventDF = ns5.eventsToDataFrame(
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
    tdDF.loc[:, 'isInTask'] = taskMask
    tdDF.loc[:, 'pedalVelocityAbs'] = tdDF['pedalVelocity'].abs()
    try:
        zeroEpochs = pd.Series(pedalPositionZeroEpochs[blockIdx])
        closestTimes, closestIdx  = hf.closestSeries(takeFrom=zeroEpochs, compareTo=tdDF['t'])
        epochChanges = pd.Series(0, index=tdDF.index)
        epochChanges.loc[closestIdx] = 1
        tdDF.loc[:, 'positionEpoch'] = epochChanges.cumsum()
    except Exception:
        tdDF.loc[:, 'positionEpoch'] = 0
    #
    for name, group in tdDF.groupby('positionEpoch'):
        pedalRestingMask = group['pedalVelocityAbs'] < 1e-12
        thisTaskMask = group['isInTask']
        # pdb.set_trace()
        thisNeutralPoint = float(group.loc[thisTaskMask & pedalRestingMask, 'pedalPosition'].mode().iloc[0])
        assert np.abs(thisNeutralPoint) < 0.1 # if the neutral point > 10 deg, there might be a problem
        tdDF.loc[group.index, 'pedalPosition'] = group['pedalPosition'] - thisNeutralPoint
    #
    tdDF.loc[:, 'pedalPositionAbs'] = tdDF['pedalPosition'].abs()
    pedalPosQuantiles = tdDF.loc[taskMask, 'pedalPosition'].quantile([0, 1])
    pedalPosTolerance = (
        pedalPosQuantiles.iloc[1] -
        pedalPosQuantiles.iloc[0]) / 100
    pedalVelQuantiles = tdDF.loc[taskMask, 'pedalVelocityAbs'].quantile([0, 1])
    pedalVelTolerance = (
        pedalVelQuantiles.iloc[1] -
        pedalVelQuantiles.iloc[0]) / 100
    # pedalRestingMask = tdDF['pedalVelocityAbs'] < pedalVelTolerance
    # pedalRestingMask = tdDF['pedalVelocityAbs'] == 0
    # pedalNeutralPoint = float(
    #     tdDF.loc[taskMask & pedalRestingMask, 'pedalPosition'].value_counts().idxmax())
    # pedalNeutralPoint = float(tdDF.loc[taskMask & pedalRestingMask, 'pedalPosition'].mode())
    # assert np.abs(pedalNeutralPoint) < 0.1 # if the neutral point > 10 deg, there might be a problem
    #
    # tdDF.loc[:, 'pedalPosition'] = tdDF['pedalPosition'] - pedalNeutralPoint
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
        for catName in availableCateg + ['tdIndex', 'markForDeletion']:
            trialsDict[mvCat].loc[:, catName] = np.nan
    markForDeletionMaster = pd.Series(False, index=trialsDict['outbound'].index)
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
                print('\n\n{}\n Error finding edge between t = {} sec and {} sec\n\n'.format(
                    dataBlockPath, group['t'].min(), group['t'].max()))
                markForDeletionMaster.loc[mvRound] = True
                traceback.print_exc()
                continue
                # pdb.set_trace()
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
                print('\n\n{}\n Error finding reach peak between t = {} sec and {} sec\n\n'.format(
                    dataBlockPath, group['t'].min(), group['t'].max()))
                markForDeletionMaster.loc[mvRound] = True
                traceback.print_exc()
                continue
                # pdb.set_trace()
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
                print('\n\n{}\n Error finding return between t = {} sec and {} sec\n\n'.format(
                    dataBlockPath, group['t'].min(), group['t'].max()))
                markForDeletionMaster.loc[mvRound] = True
                traceback.print_exc()
                continue
                # pdb.set_trace()
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
            markForDeletionMaster.loc[mvRound] = (movementDuration > 10.)
            #
    #  determine size category
    observedPedalSizes = trialsDict['outbound'].loc[~markForDeletionMaster, 'pedalSize']
    pedalSizeCat = pd.cut(
        observedPedalSizes.abs(), movementSizeBins,
        labels=movementSizeBinLabels)
    assert not pedalSizeCat.isna().any()
    for mvCat in movementCatTypes:
        trialsDict[mvCat].loc[markForDeletionMaster, 'pedalSize'] = 0.
        trialsDict[mvCat].loc[markForDeletionMaster, 'pedalSizeCat'] = 'NA'
        trialsDict[mvCat].loc[~markForDeletionMaster, 'pedalSizeCat'] = pedalSizeCat
        trialsDict[mvCat].loc[:, 'markForDeletion'] = markForDeletionMaster
    #
    if (segIdx == 0) and arguments['plotParamHistograms']:
        ax = sns.distplot(
            trialsDict['midPeak'].loc[~markForDeletionMaster, 'pedalSize'].abs(),
            bins=100, kde=False)
        plt.savefig(
            os.path.join(
                figureOutputFolder, '{}_pedalSizeDistribution.pdf'.format(prefix)))
        plt.close()
        ax = sns.distplot(
            trialsDict['midPeak'].loc[~markForDeletionMaster, 'pedalMovementDuration'],
            bins=100, kde=False)
        plt.savefig(
            os.path.join(
                figureOutputFolder, '{}_pedalMovementDurationDistribution.pdf'.format(prefix)))
        # plt.show()
        plt.close()
    #
    alignEventsDF = (
        pd.concat(
            trialsDict, axis=0,
            names=['pedalMovementCat', 'movementRound'])
        .reset_index())
    deleteTheseIdx = alignEventsDF.loc[alignEventsDF['markForDeletion'].astype(bool), :].index
    if deleteTheseIdx.any():
        print('calcMotionAlignTimes, dropping idx = {} (automatically marked for deletion)'.format(deleteTheseIdx))
        alignEventsDF.drop(index=deleteTheseIdx, inplace=True)
    alignEventsDF.drop(columns=['markForDeletion'], inplace=True)
    try:
        manuallyRejectedRounds = dropMotionRounds[blockIdx]
        rejectMask = alignEventsDF['movementRound'].isin(manuallyRejectedRounds)
        print('calcMotionAlignTimes, dropping idx = {} (manually rejected)'.format(deleteTheseIdx))
        alignEventsDF.drop(index=rejectMask.loc[rejectMask].index, inplace=True)
    except Exception:
        traceback.print_exc()
    # print('Found events at {}'.format(alignEventsDF['t'].tolist()))
    if arguments['plotParamHistograms']:
        fig, ax = plt.subplots()
        # pdb.set_trace()
        fig.set_size_inches(4 * np.ceil(tdDF['t'].max() / 500), 4)
        pPosDeg = 100 * tdDF['pedalPosition']
        ax.plot(tdDF['t'], pPosDeg, label='pedal position')
        ax.set_ylabel('deg.')
        ax.plot(
            trialsDict['outbound'].loc[~markForDeletionMaster, 't'],
            pPosDeg.loc[trialsDict['outbound'].loc[~markForDeletionMaster, 'tdIndex']],
            'ro', label='trial outbound movement')
        ax.legend()
        plt.savefig(
            os.path.join(
                figureOutputFolder, '{}_pedalMovementTrials.pdf'.format(prefix)))
        # plt.show()
        plt.close()
    #  add metaCat
    alignEventsDF.loc[:, 'pedalMetaCat'] = 'NA'
    alignEventsDF.loc[(alignEventsDF['pedalMovementCat'] == 'outbound') | (alignEventsDF['pedalMovementCat'] == 'return'), 'pedalMetaCat'] = 'starting'
    alignEventsDF.loc[(alignEventsDF['pedalMovementCat'] == 'reachedPeak') | (alignEventsDF['pedalMovementCat'] == 'reachedBase'), 'pedalMetaCat'] = 'stopping'
    #  sort align times
    alignEventsDF.sort_values(by='t', inplace=True, kind='mergesort')
    if applyDelay is not None:
        # rising thresholds probably resolve earlier
        alignEventsDF.loc[alignEventsDF['pedalMetaCat'] == 'starting', 't'] -= applyDelay
        # falling thresholds probably resolve later
        alignEventsDF.loc[alignEventsDF['pedalMetaCat'] == 'stopping', 't'] += applyDelay
    # pdb.set_trace()
    alignEventsDF.loc[:, 'expName'] = arguments['exp']
    htmlOutPath = os.path.join(figureOutputFolder, '{}_motionAlignTimes.html'.format(prefix))
    alignEventsDF.to_html(htmlOutPath)
    alignEvents = ns5.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_motionAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #
    concatLabelColumns = [
        'pedalMovementCat',
        'pedalSizeCat', 'pedalDirection',
        'movementRound', 'expName',
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

outputPath = os.path.join(
    scratchFolder,
    ns5FileName + '_epochs'
    )
if not os.path.exists(outputPath + '.nix'):
    writer = ns5.NixIO(filename=outputPath + '.nix', mode='ow')
    writer.write_block(masterBlock, use_obj_names=True)
    writer.close()
else:
    preReader, preBlock = ns5.blockFromPath(
        outputPath + '.nix', lazy=arguments['lazy'])
    eventExists = alignEvents.name in [ev.name for ev in preBlock.filter(objects=[EventProxy, Event])]
    preReader.file.close()
    # if events already exist...
    if eventExists:
        print('motion times already calculated! Deleting block and starting over')
        os.remove(outputPath + '.nix')
        writer = ns5.NixIO(filename=outputPath + '.nix', mode='ow')
        writer.write_block(masterBlock, use_obj_names=True)
        writer.close()
    else:
        ns5.addBlockToNIX(
            masterBlock, neoSegIdx=allSegs,
            writeAsigs=False, writeSpikes=False, writeEvents=True,
            fileName=ns5FileName + '_epochs',
            folderPath=scratchFolder,
            purgeNixNames=False,
            nixBlockIdx=0, nixSegIdx=allSegs,
            )
