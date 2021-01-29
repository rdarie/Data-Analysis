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
if not (blockExperimentType == 'proprio'):
    print('skipping blocks without movement *and* stim')
    sys.exit()
#
prefix = ns5FileName
dataBlockPath = os.path.join(
    analysisSubFolder,
    prefix + '_analyze.nix')
eventBlockPath = os.path.join(
    analysisSubFolder,
    prefix + '_epochs.nix')
print('loading {}'.format(dataBlockPath))
dataReader, dataBlock = preproc.blockFromPath(
    dataBlockPath, lazy=arguments['lazy'])
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
#  allocate block to contain events
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=dataBlock.annotations['neo_name'])

extraAnnNames = ['stimDelay']
blockIdx = 0
checkReferences = False
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
    noStimFiller = pd.Series({
        'amplitude': 0,
        'RateInHz': 0,
        'program': 999,
        'activeGroup': 0,
        'electrode': 'NA',
        'stimDelay': np.nan
    })
    for annName in stimAnnNames + extraAnnNames:
        motionEvDF.loc[:, annName] = np.nan
    searchRadius = .7  # sec
    stimEvDF.loc[:, 'alreadyAssigned'] = False
    movementCatTypes = list(motionEvDF['pedalMovementCat'].unique())
    for mvRound, group in motionEvDF.groupby('movementRound'):
        outboundT = group.loc[group['pedalMovementCat'] == 'outbound', 't']
        outboundIdx = outboundT.index
        reachT = group.loc[group['pedalMovementCat'] == 'reachedPeak', 't']
        reachIdx = reachT.index
        returnT = group.loc[group['pedalMovementCat'] == 'return', 't']
        returnIdx = returnT.index
        ####
        if mvRound > 0:
            tSearchStart = max(
                (float(outboundT) - searchRadius),
                float(reachBaseT)  # from prev trial
                )
        else:
            tSearchStart = max(
                (float(outboundT) - searchRadius),
                stimEvDF['t'].min()
                )
        tSearchStop = min(
            (float(reachT) + searchRadius),
            float(returnT))
        stimInSearchRadius = (
            (stimEvDF['t'] >= tSearchStart) &
            (stimEvDF['t'] < tSearchStop))
        stimOnTs = stimEvDF.loc[
            (
                (stimEvDF['stimCat'] == 'stimOn') &
                (~stimEvDF['alreadyAssigned']) &
                stimInSearchRadius),
            't']
        if stimOnTs.size > 0:
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=outboundT, compareTo=stimOnTs,
                strictly='neither')
            stimDelay = float(closestTimes - outboundT)
            theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            for annName in stimAnnNames:
                motionEvDF.loc[outboundIdx, annName] = theseStimAnn[annName]
                motionEvDF.loc[reachIdx, annName] = theseStimAnn[annName]
            stimEvDF.loc[closestIdx, 'alreadyAssigned'] = True
            motionEvDF.loc[outboundIdx, 'stimDelay'] = stimDelay
            motionEvDF.loc[reachIdx, 'stimDelay'] = stimDelay
        else:
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=outboundT, compareTo=stimEvDF['t'],
                strictly='less')
            theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            try:
                assert theseStimAnn['stimCat'] == 'stimOff'
            except Exception:
                print('\n\n{}\n Error at t= {} sec\n\n'.format(dataBlockPath, outboundT.iloc[0]))
                traceback.print_exc()
                pdb.set_trace()
            for annName in stimAnnNames:
                motionEvDF.loc[outboundIdx, annName] = noStimFiller[annName]
                motionEvDF.loc[reachIdx, annName] = noStimFiller[annName]
            motionEvDF.loc[outboundIdx, 'stimDelay'] = noStimFiller['stimDelay']
            motionEvDF.loc[reachIdx, 'stimDelay'] = noStimFiller['stimDelay']
        ####
        reachBaseT = group.loc[group['pedalMovementCat'] == 'reachedBase', 't']
        reachBaseIdx = reachBaseT.index
        tSearchStart = max(
            (float(returnT) - searchRadius),
            float(reachT)
        )
        if mvRound == motionEvDF['movementRound'].max():
            tSearchStop = min(
                (float(reachBaseT) + searchRadius),
                motionEvDF['t'].max()
            )
        else:
            tSearchStop = min(
                (float(reachBaseT) + searchRadius),
                motionEvDF.loc[motionEvDF['movementRound'] == mvRound + 1, 't'].max()
            )
        stimInSearchRadius = (
            (stimEvDF['t'] >= tSearchStart) &
            (stimEvDF['t'] < tSearchStop))
        stimOnTs = stimEvDF.loc[
            (
                (stimEvDF['stimCat'] == 'stimOn') &
                (~stimEvDF['alreadyAssigned']) &
                stimInSearchRadius),
            't']
        if stimOnTs.size > 0:
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=returnT, compareTo=stimOnTs,
                strictly='neither')
            theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            stimDelay = float(closestTimes - returnT)
            for annName in stimAnnNames:
                motionEvDF.loc[returnIdx, annName] = theseStimAnn[annName]
                motionEvDF.loc[reachBaseIdx, annName] = theseStimAnn[annName]
            motionEvDF.loc[returnIdx, 'stimDelay'] = stimDelay
            motionEvDF.loc[reachBaseIdx, 'stimDelay'] = stimDelay
        else:
            closestTimes, closestIdx = hf.closestSeries(
                takeFrom=returnT, compareTo=stimEvDF['t'],
                strictly='less')
            theseStimAnn = stimEvDF.loc[closestIdx, :].iloc[0]
            try:
                assert theseStimAnn['stimCat'] == 'stimOff'
            except Exception:
                print('\n\n{}\nError at t= {} sec\n\n'.format(dataBlockPath, returnT.iloc[0]))
                traceback.print_exc()
                # pdb.set_trace()
            for annName in stimAnnNames:
                motionEvDF.loc[returnIdx, annName] = noStimFiller[annName]
                motionEvDF.loc[reachBaseIdx, annName] = noStimFiller[annName]
            motionEvDF.loc[returnIdx, 'stimDelay'] = noStimFiller['stimDelay']
            motionEvDF.loc[reachBaseIdx, 'stimDelay'] = noStimFiller['stimDelay']
    motionEvDF.loc[:, stimAnnNames] = (
        motionEvDF
        .loc[:, stimAnnNames]
        .fillna(method='ffill')
        .fillna(method='bfill'))
    if (segIdx == 0) and arguments['plotParamHistograms']:
        fig, ax = plt.subplots()
        for cN in ['outbound', 'return']:
            theseEvents = (
                motionEvDF
                .loc[motionEvDF['pedalMovementCat'] == cN, :])
            # pdb.set_trace()
            sns.distplot(
                theseEvents.loc[~theseEvents['stimDelay'].isna(), 'stimDelay'],
                bins=200, kde=False, ax=ax)
            print(
                theseEvents
                .sort_values('stimDelay', ascending=False)
                .loc[:, ['t', 'stimDelay']]
                .head(10)
                )
        fig.savefig(
            os.path.join(
                figureFolder, 'stimDelayDistribution.pdf'))
        # plt.show()
        plt.close()
    alignEvents = preproc.eventDataFrameToEvents(
        motionEvDF, idxT='t',
        annCol=None,
        eventName='seg{}_motionStimAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #
    concatLabelColumns = availableCateg + stimAnnNames + extraAnnNames
    concatLabelsDF = motionEvDF[concatLabelColumns]
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatLabelsDF.iterrows()])
    concatEvents = Event(
        name='seg{}_motionStimAlignTimesConcatenated'.format(segIdx),
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
eventReader.file.close()
masterBlock.create_relationship()
allSegs = list(range(len(masterBlock.segments)))

preproc.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeAsigs=False, writeSpikes=False, writeEvents=True,
    fileName=ns5FileName + '_epochs',
    folderPath=analysisSubFolder,
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )
