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
import logging
logging.captureWarnings(True)
import os, sys

from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from dataAnalysis.analysis_code.namedQueries import namedQueries
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
import seaborn as sns
import pdb, traceback
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
#  load options
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
        'lines.markersize': 2.4,
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
    'mathtext.default': 'regular',
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV
from pandas import IndexSlice as idxSl
from datetime import datetime as dt
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
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
    processedFolder, 'figures', 'preprocDiagnostics'
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
        taskMask = hf.getTimeMaskFromRanges(
            tdDF['t'], alignTimeBounds)
    else:
        taskMask = (tdDF['t'] >= 0)
    ###################################################
    tdDF.loc[:, 'isInTask'] = taskMask
    tdDF.loc[:, 'pedalVelocityAbs'] = tdDF['pedalVelocity'].abs()
    try:
        zeroEpochs = pd.Series(pedalPositionZeroEpochs[blockIdx])
        closestTimes, closestIdx = hf.closestSeries(takeFrom=zeroEpochs, compareTo=tdDF['t'])
        epochChanges = pd.Series(0, index=tdDF.index)
        epochChanges.loc[closestIdx] = 1
        tdDF.loc[:, 'positionEpoch'] = epochChanges.cumsum()
    except Exception:
        tdDF.loc[:, 'positionEpoch'] = 0
    #
    eventMask = (eventDF['t'].diff() > 0) & (eventDF['t'].diff() < 60)
    eventBasedEpochs = [
        [eventDF.loc[tid, 't'] - 0.5, eventDF.loc[tid, 't']]
        for tid in eventDF.index[eventMask]
        ]
    periEventMask = hf.getTimeMaskFromRanges(
        tdDF['t'], eventBasedEpochs)
    tdDF.loc[:, 'isPeriEvent'] = periEventMask
    #
    for name, group in tdDF.groupby('positionEpoch'):
        pedalRestingMask = group['pedalVelocityAbs'] < 1e-12
        thisTaskMask = group['isInTask']
        thisPeriEventMask = group['isPeriEvent']
        thisNeutralPoint = float(group.loc[thisTaskMask & pedalRestingMask & thisPeriEventMask, 'pedalPosition'].mode().iloc[0])
        print('positionEpoch {}, np.abs(thisNeutralPoint) = {}'.format(name, np.abs(thisNeutralPoint)))
        # assert np.abs(thisNeutralPoint) < 0.1  # if the neutral point > 10 deg, there might be a problem
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
    alignEventsDF.reset_index(drop=True, inplace=True)
    newRoundMap = {oldIJ: newIJ for newIJ, oldIJ in enumerate(alignEventsDF['movementRound'].unique())}
    alignEventsDF.loc[:, 'originalMovementRound'] = alignEventsDF['movementRound'].copy()
    alignEventsDF.loc[:, 'movementRound'] = alignEventsDF['movementRound'].map(newRoundMap)
    # print('Found events at {}'.format(alignEventsDF['t'].tolist()))
    if arguments['plotParamHistograms']:
        print('plotting movement trial summary')
        pdfPath = os.path.join(figureOutputFolder, '{}_pedalMovementTrials.pdf'.format(prefix))
        with PdfPages(pdfPath) as pdf:
            confPlotWinSize = 200.  # seconds
            plotRounds = tdDF['t'].apply(lambda x: np.floor(x / confPlotWinSize))
            plotChanNames = ['pedalPosition']
            for pr in plotRounds.unique():
                plotMask = (plotRounds == pr)
                fig, ax = plt.subplots(1, 1, figsize=(10, 1))
                for cNIdx, cN in enumerate(plotChanNames):
                    try:
                        plotTrace = 100 * tdDF.loc[plotMask, cN].to_numpy()
                        ax.plot(tdDF.loc[plotMask, 't'], plotTrace, label=cN, alpha=0.5, rasterized=True)
                    except Exception:
                        traceback.print_exc()
                goodTs = trialsDict['outbound'].loc[~markForDeletionMaster, 't']
                goodTMask = (goodTs >= tdDF.loc[plotMask, 't'].min()) & (goodTs < tdDF.loc[plotMask, 't'].max())
                axMiddle = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
                ax.scatter(
                    trialsDict['outbound'].loc[~markForDeletionMaster, :].loc[goodTMask, 't'].to_numpy(),
                    trialsDict['outbound'].loc[~markForDeletionMaster, :].loc[goodTMask, 't'].to_numpy() ** 0 - 1 + axMiddle,
                    c='g', marker='+', label='set of movement trials')
                badTs = trialsDict['outbound'].loc[markForDeletionMaster, 't']
                badTMask = (badTs >= tdDF.loc[plotMask, 't'].min()) & (badTs < tdDF.loc[plotMask, 't'].max())
                ax.scatter(
                    trialsDict['outbound'].loc[markForDeletionMaster, :].loc[badTMask, 't'].to_numpy(),
                    trialsDict['outbound'].loc[markForDeletionMaster, :].loc[badTMask, 't'].to_numpy() ** 0 - 1 + axMiddle,
                    c='r', marker='o', label='rejected trials')
                ax.set_ylabel('deg.')
                ax.set_xlabel('time (sec)')
                ax.legend(loc='lower left')
                fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                figSaveOpts = dict(
                    # bbox_extra_artists=tuple([ta.get_legend() for ta in ax]),
                    bbox_extra_artists=(ax.get_legend(),),
                    bbox_inches='tight')
                pdf.savefig(**figSaveOpts)
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
#############
print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')