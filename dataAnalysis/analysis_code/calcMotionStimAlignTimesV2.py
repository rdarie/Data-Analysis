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
if not (blockExperimentType == 'proprio'):
    print('skipping blocks without movement *and* stim')
    sys.exit()
#
prefix = ns5FileName
dataBlockPath = os.path.join(
    analysisSubFolder,
    prefix + '_analyze.nix')
eventBlockPath = os.path.join(
    # analysisSubFolder,
    scratchFolder,
    prefix + '_epochs.nix')
#
figureOutputFolder = os.path.join(
    processedFolder, 'preprocDiagnostics'
    )
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
print('loading {}'.format(dataBlockPath))
dataReader, dataBlock = preproc.blockFromPath(
    dataBlockPath, lazy=arguments['lazy'])
print('loading {}'.format(eventBlockPath))
eventReader, eventBlock = preproc.blockFromPath(
    eventBlockPath, lazy=arguments['lazy'])
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
availableCateg = [
    'pedalDirection', 'pedalSizeCat', 'pedalMovementCat',
    'pedalSize', 'pedalMovementDuration']
motionAnnNamesForStim = [
    'pedalDirection', 'pedalSizeCat', 'pedalMovementCat',
    'pedalSize', 'pedalMetaCat', 'pedalMovementDuration', 'movementRound']
#  allocate block to contain events
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=dataBlock.annotations['neo_name'])

extraAnnNames = ['stimDelay']
blockIdx = 0
checkReferences = False
searchRadius = .7  # sec
for segIdx, dataSeg in enumerate(dataBlock.segments):
    eventSeg = eventBlock.segments[segIdx]
    print('Calculating motion stim align times for trial {}'.format(segIdx))
    #
    if arguments['lazy']:
        eventProxysList = eventSeg.events
        eventSegEvents = [evP.load() for evP in eventProxysList]
    # samplingRate = dummyAsig.sampling_rate
    motionEvent = preproc.loadObjArrayAnn([ev for ev in eventSegEvents if ev.name == 'seg0_motionAlignTimes'][0])
    motionEvDict = motionEvent.array_annotations.copy()
    motionEvDict['t'] = motionEvent.times.magnitude
    motionEvDF = pd.DataFrame(motionEvDict)
    #
    stimEvent = preproc.loadObjArrayAnn([ev for ev in eventSegEvents if ev.name == 'seg0_stimAlignTimes'][0])
    stimEvDict = stimEvent.array_annotations.copy()
    stimAnnNames = [nm for nm in sorted(list(stimEvDict.keys())) if nm not in ['stimCat']]
    stimEvDict['t'] = stimEvent.times.magnitude
    stimEvDF = pd.DataFrame(stimEvDict)
    noStimFiller = preproc.metaFillerLookup.copy()
    noStimFiller['expName'] = arguments['exp']
    # pdb.set_trace()
    for annName in stimAnnNames + extraAnnNames:
        motionEvDF.loc[:, annName] = np.nan
    for annName in motionAnnNamesForStim + extraAnnNames:
        stimEvDF.loc[:, annName] = np.nan
    stimEvDF.loc[:, 'assignedTo'] = np.nan
    movementCatTypes = list(motionEvDF['pedalMovementCat'].unique())
    #
    allStimOffEvents = stimEvDF.loc[stimEvDF['stimCat'] == 'stimOff', :]
    allStimOnEvents = stimEvDF.loc[stimEvDF['stimCat'] == 'stimOn', :]
    #
    oldReachBaseT = None
    oldReachBaseIndex = None
    for mvRound, group in motionEvDF.groupby('movementRound'):
        print('Analyzing round {}'.format(mvRound))
        outboundT = group.loc[group['pedalMovementCat'] == 'outbound', 't']
        outboundIdx = outboundT.index
        reachPeakT = group.loc[group['pedalMovementCat'] == 'reachedPeak', 't']
        reachPeakIdx = reachPeakT.index
        returnT = group.loc[group['pedalMovementCat'] == 'return', 't']
        returnIdx = returnT.index
        reachBaseT = group.loc[group['pedalMovementCat'] == 'reachedBase', 't']
        reachBaseIdx = reachBaseT.index
        ################################################################################
        # first, check entire round, in case it only has one set of stimOn stim Off
        if mvRound > 0:
            tSearchStart = max(
                (float(outboundT) - searchRadius),
                float(oldReachBaseT)  # from prev trial
                )
        else:
            tSearchStart = max(
                (float(outboundT) - searchRadius),
                stimEvDF['t'].min() - 1e-3
                )
        #
        tSearchStop = min(
            (float(reachBaseT) + searchRadius),
            stimEvDF['t'].max() + 1e-3
            )
        #
        stimInSearchRadius = (
            (stimEvDF['t'] >= tSearchStart) &
            (stimEvDF['t'] < tSearchStop))
        roundStimOnEvents = stimEvDF.loc[
            (
                (stimEvDF['stimCat'] == 'stimOn') &
                (stimEvDF['assignedTo'].isna()) &
                stimInSearchRadius),
            :]
        roundStimOffEvents = stimEvDF.loc[
            (
                (stimEvDF['stimCat'] == 'stimOff') &
                (stimEvDF['assignedTo'].isna()) &
                stimInSearchRadius),
            :]
        try:
            # what goes up, must come down?
            assert (roundStimOnEvents.shape[0] == roundStimOffEvents.shape[0])
        except Exception:
            traceback.print_exc()
            raise(Exception('Move round has different number of stim ons and stim offs (t = {:.3f})!'.format(float(outboundT))))
        singleOnOffAllRound = (roundStimOnEvents.shape[0] == 1)
        specialCaseSingleStim = False
        if singleOnOffAllRound:
            singleStimOnT = roundStimOnEvents['t'].iloc[0]
            singleStimOffT = roundStimOffEvents['t'].iloc[0]
            if (singleStimOffT > returnT.iloc[0]):
                specialCaseSingleStim = True
        else:
            singleStimOnT = None
            singleStimOffT = None
        ################################################################################
        # define search bounds to find stim train corresponding to outbound movement
        tSearchStop = min(
            (float(returnT) - searchRadius),
            stimEvDF['t'].max() + 1e-3
            )
        stimInSearchRadius = (
            (stimEvDF['t'] >= tSearchStart) &
            (stimEvDF['t'] < tSearchStop))
        stimOnTs = stimEvDF.loc[
            (
                (stimEvDF['stimCat'] == 'stimOn') &
                (stimEvDF['assignedTo'].isna()) &
                stimInSearchRadius),
            't']
        if stimOnTs.size > 0:
            roundHasOutBStim = True
            print('Found outbound stim for round {} at t = {}'.format(mvRound, stimOnTs.iloc[0]))
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=outboundT, compareTo=stimOnTs,
                strictly='neither')
            stimDelay = float(closestTimes - outboundT)
            theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            outBStimOn = theseStimAnn['t']
            outBStimOnIdx = closestIdx
            for annName in stimAnnNames:
                motionEvDF.loc[outboundIdx, annName] = theseStimAnn[annName]
                motionEvDF.loc[reachPeakIdx, annName] = theseStimAnn[annName]
            motionEvDF.loc[outboundIdx, 'stimDelay'] = stimDelay
            # assign this stim event to the corresponding pedal movement
            stimEvDF.loc[closestIdx, 'assignedTo'] = outboundIdx
            theseMotAnn = motionEvDF.loc[outboundIdx, :].iloc[0]
            stimEvDF.loc[closestIdx, 'stimDelay'] = stimDelay
            for annName in motionAnnNamesForStim:
                stimEvDF.loc[closestIdx, annName] = theseMotAnn[annName]
        else:
            # if no onsets detected
            roundHasOutBStim = False
            # RD 09/10/21 taking this out, check if it breaks anything
            # closestTimes, closestIdx = hf.closestSeries(
            #     takeFrom=outboundT, compareTo=stimEvDF['t'],
            #     strictly='less')
            # theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            for annName in stimAnnNames:
                motionEvDF.loc[outboundIdx, annName] = noStimFiller[annName]
                motionEvDF.loc[reachPeakIdx, annName] = noStimFiller[annName]
            motionEvDF.loc[outboundIdx, 'stimDelay'] = noStimFiller['stimDelay']
            motionEvDF.loc[reachPeakIdx, 'stimDelay'] = noStimFiller['stimDelay']
        ################################################################################
        #### find stims associated with the return segment
        tSearchStart = max(
            (float(returnT) - searchRadius),
            float(reachPeakT)
        )
        if mvRound == motionEvDF['movementRound'].max():
            tSearchStop = min(
                (float(reachBaseT) + searchRadius),
                motionEvDF['t'].max() + 1e-3
            )
        else:
            tSearchStop = min(
                (float(reachBaseT) + searchRadius),
                motionEvDF.loc[motionEvDF['movementRound'] == mvRound + 1, 't'].max() +  1e-3
            )
        stimInSearchRadius = (
            (stimEvDF['t'] >= tSearchStart) &
            (stimEvDF['t'] < tSearchStop))
        stimOnTs = stimEvDF.loc[
            (
                (stimEvDF['stimCat'] == 'stimOn') &
                (stimEvDF['assignedTo'].isna()) &
                stimInSearchRadius),
            't']
        if stimOnTs.size > 0:
            roundHasRetStim = True
            print('Found return stim for round {} at t = {:.3f}'.format(mvRound, stimOnTs.iloc[0]))
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=returnT, compareTo=stimOnTs,
                strictly='neither')
            theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            retStimOn = theseStimAnn['t']
            stimDelay = float(closestTimes - returnT)
            for annName in stimAnnNames:
                motionEvDF.loc[returnIdx, annName] = theseStimAnn[annName]
                motionEvDF.loc[reachBaseIdx, annName] = theseStimAnn[annName]
            motionEvDF.loc[returnIdx, 'stimDelay'] = stimDelay
            # assign this stim event to the corresponding pedal movement
            stimEvDF.loc[closestIdx, 'assignedTo'] = returnIdx
            theseMotAnn = motionEvDF.loc[returnIdx, :].iloc[0]
            stimEvDF.loc[closestIdx, 'stimDelay'] = stimDelay
            for annName in motionAnnNamesForStim:
                stimEvDF.loc[closestIdx, annName] = theseMotAnn[annName]
        elif specialCaseSingleStim:
            roundHasRetStim = True
            # special case where there is one stim per movement
            theseStimAnn = roundStimOnEvents.iloc[0, :]
            retStimOn = theseStimAnn['t']
            for annName in stimAnnNames:
                motionEvDF.loc[returnIdx, annName] = theseStimAnn[annName]
                motionEvDF.loc[reachBaseIdx, annName] = theseStimAnn[annName]
        else:
            roundHasRetStim = False
            # closestTimes, closestIdx = hf.closestSeries(
            #     takeFrom=returnT, compareTo=stimEvDF['t'],
            #     strictly='less')
            # theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            for annName in stimAnnNames:
                motionEvDF.loc[returnIdx, annName] = noStimFiller[annName]
                motionEvDF.loc[reachBaseIdx, annName] = noStimFiller[annName]
            motionEvDF.loc[returnIdx, 'stimDelay'] = noStimFiller['stimDelay']
            motionEvDF.loc[reachBaseIdx, 'stimDelay'] = noStimFiller['stimDelay']
        #  annotate reachPeak and reachBase
        if roundHasOutBStim:
            if roundHasRetStim:
                stimInSearchRadius = (
                    (stimEvDF['t'] > outBStimOn) &
                    (stimEvDF['t'] < retStimOn))
            else:
                stimInSearchRadius = (
                    (stimEvDF['t'] > outBStimOn) &
                    (stimEvDF['t'] < float(returnT) + searchRadius))
            stimOffTs = stimEvDF.loc[
                (
                    (stimEvDF['stimCat'] == 'stimOff') &
                    (stimEvDF['assignedTo'].isna()) &
                    stimInSearchRadius),
                't']
            if stimOffTs.size > 0:
                print('Found outbound stim OFF for round {} at t = {:.3f}'.format(mvRound, stimOffTs.iloc[0]))
                closestTimes, closestIdx = hf.closestSeries(
                    takeFrom=reachPeakT, compareTo=stimOffTs,
                    strictly='neither')
                theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
                stimDelay = float(closestTimes - reachPeakT)
                motionEvDF.loc[reachPeakIdx, 'stimDelay'] = stimDelay
                # assign this stim event to the corresponding pedal movement
                stimEvDF.loc[closestIdx, 'assignedTo'] = reachPeakIdx
                theseMotAnn = motionEvDF.loc[reachPeakIdx, :].iloc[0]
                stimEvDF.loc[closestIdx, 'stimDelay'] = stimDelay
                for annName in motionAnnNamesForStim:
                    stimEvDF.loc[closestIdx, annName] = theseMotAnn[annName]
            elif specialCaseSingleStim:
                # special case where there is one stim per movement
                theseStimAnn = roundStimOffEvents.iloc[0, :]
                pass
            else:
                closestTimes, closestIdx = hf.closestSeries(
                    takeFrom=reachPeakT, compareTo=allStimOffEvents.loc[allStimOffEvents['t'] > outBStimOn, 't'],
                    strictly='neither')
                if closestTimes.empty:
                    raise(Exception('No off time corresponding to stim on for this move round (t = {:.3f})!'.format(float(reachPeakT))))
                elif (closestTimes.min() - outBStimOn) > 10:
                    raise(Exception('No off time corresponding to stim on for this move round (t = {:.3f})!'.format(float(reachPeakT))))
        if roundHasRetStim:
            stimInSearchRadius = (
                (stimEvDF['t'] > retStimOn) &
                (stimEvDF['t'] < float(reachBaseT) + searchRadius))
            stimOffTs = stimEvDF.loc[
                (
                    (stimEvDF['stimCat'] == 'stimOff') &
                    (stimEvDF['assignedTo'].isna()) &
                    stimInSearchRadius),
                't']
            if stimOffTs.size > 0:
                print('Found return stim OFF for round {} at t = {:.3f}'.format(mvRound, stimOffTs.iloc[0]))
                closestTimes, closestIdx = hf.closestSeries(
                    takeFrom=reachBaseT, compareTo=stimOffTs,
                    strictly='neither')
                theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
                stimDelay = float(closestTimes - reachBaseT)
                motionEvDF.loc[reachBaseIdx, 'stimDelay'] = stimDelay
                # assign this stim event to the corresponding pedal movement
                stimEvDF.loc[closestIdx, 'assignedTo'] = reachBaseIdx
                theseMotAnn = motionEvDF.loc[reachBaseIdx, :].iloc[0]
                stimEvDF.loc[closestIdx, 'stimDelay'] = stimDelay
                for annName in motionAnnNamesForStim:
                    stimEvDF.loc[closestIdx, annName] = theseMotAnn[annName]
            else:
                stimOffTs = reachBaseT + searchRadius
                print('No off time corresponding to stim on for this move round (t = {:.3f})!'.format(float(reachBaseT)))
                # raise(Exception('No off time corresponding to stim on for this move round (t = {:.3f})!'.format(float(reachBaseT))))
        oldReachBaseT = reachBaseT
    # pdb.set_trace()
    for cN in stimAnnNames:
        motionEvDF.loc[motionEvDF[cN].isna(), cN] = noStimFiller[cN]
    '''motionEvDF.loc[motionEvDF[cN].isna(), stimAnnNames] = (
        motionEvDF
        .loc[:, stimAnnNames]
        .fillna(method='ffill')
        .fillna(method='bfill'))'''
    if (segIdx == 0) and arguments['plotParamHistograms']:
        fig, ax = plt.subplots()
        for cN in movementCatTypes:
            theseEvents = (
                motionEvDF
                .loc[(motionEvDF['pedalMovementCat'] == cN) & ~motionEvDF['stimDelay'].isna(), :])
            if theseEvents.size > 0:
                thisLabel = '\n'.join([
                    'epoch: {}'.format(cN),
                    'median: {:.1f} msec; std: {:.1f} msec'.format(
                        1000 * theseEvents['stimDelay'].median(),
                        1000 * theseEvents['stimDelay'].std())
                    ])
                sns.distplot(
                    theseEvents['stimDelay'],
                    # bins=200,
                    kde=True, ax=ax, label=thisLabel)
                ax.legend()
                print(
                    theseEvents
                    .sort_values('stimDelay', ascending=False)
                    .loc[:, ['t', 'stimDelay']]
                    .head(10)
                    )
        fig.savefig(
            os.path.join(
                figureOutputFolder, '{}_stimDelayDistribution.pdf'.format(prefix)))
        # plt.show()
        plt.close()
    ###
    # pdb.set_trace()
    htmlOutPath = os.path.join(figureOutputFolder, '{}_motionPeriStim.html'.format(prefix))
    motionEvDF.to_html(htmlOutPath)
    alignEventsMotion = preproc.eventDataFrameToEvents(
        motionEvDF, idxT='t',
        annCol=None,
        eventName='seg{}_motionStimAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEventsMotion.annotate(nix_name=alignEventsMotion.name)
    #
    concatLabelColumns = availableCateg + stimAnnNames + extraAnnNames
    concatLabelsDF = motionEvDF[concatLabelColumns]
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatLabelsDF.iterrows()])
    concatEvents = Event(
        name='seg{}_motionStimAlignTimesConcatenated'.format(segIdx),
        times=alignEventsMotion.times,
        labels=concatLabels
        )
    newSeg = Segment(name=dataSeg.annotations['neo_name'])
    newSeg.annotate(nix_name=dataSeg.annotations['neo_name'])
    newSeg.events.append(alignEventsMotion)
    newSeg.events.append(concatEvents)
    alignEventsMotion.segment = newSeg
    concatEvents.segment = newSeg
    ####
    stimEvDF.dropna(inplace=True)
    htmlOutPath = os.path.join(figureOutputFolder, '{}_stimPeriMotion.html'.format(prefix))
    stimEvDF.to_html(htmlOutPath)
    alignEventsStim = preproc.eventDataFrameToEvents(
        stimEvDF, idxT='t',
        annCol=None,
        eventName='seg{}_stimPerimotionAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEventsStim.annotate(nix_name=alignEventsStim.name)
    #
    concatLabelColumns = motionAnnNamesForStim + stimAnnNames + extraAnnNames
    concatStimLabelsDF = stimEvDF[concatLabelColumns]
    concatStimLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatStimLabelsDF.iterrows()])
    concatStimEvents = Event(
        name='seg{}_stimPerimotionAlignTimesConcatenated'.format(segIdx),
        times=alignEventsStim.times,
        labels=concatStimLabels
        )
    newSeg.events.append(alignEventsStim)
    newSeg.events.append(concatStimEvents)
    alignEventsStim.segment = newSeg
    concatStimEvents.segment = newSeg
    #
    masterBlock.segments.append(newSeg)
    newSeg.block = masterBlock
dataReader.file.close()
eventReader.file.close()
masterBlock.create_relationship()
allSegs = list(range(len(masterBlock.segments)))
#
outputPath = os.path.join(
    scratchFolder,
    ns5FileName + '_epochs'
    )
preReader, preBlock = preproc.blockFromPath(
    outputPath + '.nix', lazy=arguments['lazy'])
existingEvNames = [ev.name for ev in preBlock.filter(objects=[EventProxy, Event])]
eventExists = (alignEventsStim.name in existingEvNames) or (alignEventsMotion.name in existingEvNames)
# pdb.set_trace()
preReader.file.close()
if eventExists:
    raise(Exception('CalcMotionStimAlignTimes: calculated events, but they already exist in the events block'))
preproc.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeAsigs=False, writeSpikes=False, writeEvents=True,
    fileName=ns5FileName + '_epochs',
    folderPath=scratchFolder,
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )
