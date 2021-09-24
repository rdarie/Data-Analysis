"""10a: Calculate align Times ##WIP
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                                   which trial to analyze [default: 1]
    --exp=exp                                             which experimental day to analyze
    --processAll                                          process entire experimental day? [default: False]
    --plotParamHistograms                                 plot pedal size, amplitude, duration distributions? [default: False]
    --analysisName=analysisName                           append a name to the resulting blocks? [default: default]
    --makeControl                                         make control align times? [default: False]
    --lazy                                                load from raw, or regular? [default: False]
    --inputNSPBlockSuffix=inputNSPBlockSuffix             append a name to the input block?
    --inputINSBlockSuffix=inputINSBlockSuffix             append a name to the input block?
    --plotting                                            display diagnostic plots? [default: False]
    --removeLabels=removeLabels                           remove certain labels, e.g. stimOff (comma separated)
    --disableRefinement                                   by default, use xcorr to refine stim times [default: False]
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
import pdb, traceback, sys
from importlib import reload
from matplotlib.lines import Line2D
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import quantities as pq
import dataAnalysis.helperFunctions.helper_functions_new as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.preproc.mdt as mdt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Iterable
from elephant.conversion import binarize
#  load options
from docopt import docopt
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
from pandas import IndexSlice as idxSl
from datetime import datetime as dt
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
if arguments['inputNSPBlockSuffix'] is None:
    inputNSPBlockSuffix = ''
else:
    inputNSPBlockSuffix = '_{}'.format(arguments['inputNSPBlockSuffix'])
if arguments['inputINSBlockSuffix'] is None:
    inputINSBlockSuffix = ''
else:
    inputINSBlockSuffix = "_{}".format(arguments['inputINSBlockSuffix'])

print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
# trick to allow joint processing of minirc and regular trials
if blockExperimentType == 'proprio-motionOnly':
    print('skipping blocks without stim')
    sys.exit()
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
experimentDataPath = experimentDataPath.format(arguments['analysisName'])
analysisDataPath = analysisDataPath.format(arguments['analysisName'])
#  fetch stim details
insPath = os.path.join(
    scratchFolder,
    ns5FileName + '_ins' + inputINSBlockSuffix + '.nix')
insSpikeTrainsToLoad = []
for grpIdx in range(4):
    for prgIdx in range(4):
        insSpikeTrainsToLoad.append('seg0_g{}p{}#0'.format(grpIdx, prgIdx))
insLoadList = {
    # 'asigs': insAsigNames,
    'events': [
        'seg0_ins_property',
        'seg0_ins_value'
        ],
    'spiketrains': insSpikeTrainsToLoad
    }
print('Loading INS Block from: {}'.format(insPath))
insReader, insBlock = ns5.blockFromPath(
    insPath,
    lazy=arguments['lazy'],
    # lazy=False,
    reduceChannelIndexes=True,
    loadList=insLoadList)
#
if 'rawBlockName' in spikeSortingOpts['utah']:
    BlackrockFileName = ns5FileName.replace(
        'Block', spikeSortingOpts['utah']['rawBlockName'])
else:
    BlackrockFileName = ns5FileName
nspPath = os.path.join(
    scratchFolder,
    BlackrockFileName + inputNSPBlockSuffix +
    '.nix')

insDiagnosticsFolder = os.path.join(figureFolder, 'insDiagnostics')
if not os.path.exists(insDiagnosticsFolder):
    os.mkdirs(insDiagnosticsFolder)
if arguments['plotting']:
    synchReportPDF = PdfPages(
        os.path.join(
            insDiagnosticsFolder,
            'ins_stim_refinement_Block{:0>3}{}.pdf'.format(blockIdx, inputINSBlockSuffix)))
sessTapOptsNSP = expOpts['synchInfo']['nsp'][blockIdx][0]
sessTapOptsINS = expOpts['synchInfo']['ins'][blockIdx][0]
print('Loading NSP Block from: {}'.format(nspPath))
nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=arguments['lazy'],
    reduceChannelIndexes=True, loadList={
        'asigs': ['seg0_' + nM for nM in sessTapOptsNSP['synchChanName']]
    })
#
trigRasterSamplingRate = 2000
trigSampleInterval = trigRasterSamplingRate ** (-1)
if sessTapOptsINS['xCorrGaussWid'] is not None:
    gaussWid = sessTapOptsINS['xCorrGaussWid']
else:
    gaussWid = 10e-3

for asig in nspBlock.filter(objects=AnalogSignal):
    if len(asig.times):
        nspSamplingRate = float(asig.sampling_rate)
        dummyAsig = asig.copy()
try:
    alignTimeBounds = alignTimeBoundsLookup[int(arguments['blockIdx'])]
except Exception:
    alignTimeBounds = [
        [
            float(nspBlock.segments[0].filter(objects=AnalogSignalProxy)[0].t_start),
            float(nspBlock.segments[-1].filter(objects=AnalogSignalProxy)[0].t_stop)
        ]
    ]
#
availableCateg = [
    'amplitude', 'amplitudeRound', 'program', 'activeGroup', 'RateInHz']
progAmpNames = rcsa_helpers.progAmpNames
expandCols = [
    'RateInHz', 'movement', 'program', 'trialSegment']
deriveCols = ['amplitude', 'amplitudeRound']
columnsToBeAdded = (
    expandCols + deriveCols + progAmpNames)

#  allocate block to contain events
masterBlock = Block()
masterBlock.name = nspBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=nspBlock.annotations['neo_name'])
#
def corrAtLag(targetLag, xSrs=None, ySrs=None):
    return np.correlate(xSrs, ySrs.shift(targetLag).fillna(0))[0]
#

for segIdx, nspSeg in enumerate(nspBlock.segments):
    insSeg = insBlock.segments[segIdx]
    eventDF = ns5.eventsToDataFrame(
        insSeg.events, idxT='t',
        names=['seg0_ins_property', 'seg0_ins_value']
        )
    eventDF.rename(columns={'seg0_ins_property': 'property', 'seg0_ins_value': 'value'}, inplace=True)
    stimStatus = mdt.stimStatusSerialtoLong(
        eventDF, idxT='t', namePrefix='', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    #
    # stimStatus = stimStatus.loc[tMask, :].reset_index(drop=True)
    #
    asigList = [asig for asig in nspSeg.analogsignals if isinstance(asig, AnalogSignal)]
    thisNspDF = ns5.analogSignalsToDataFrame(asigList)
    thisNspDF.rename(columns={'seg0_' + sessTapOptsNSP['synchChanName'][0]: 'tapDetectSignal'}, inplace=True)
    if alignTimeBounds is not None:
        tMask = hf.getTimeMaskFromRanges(
            stimStatus['t'], alignTimeBounds)
        nspMask = hf.getTimeMaskFromRanges(
            thisNspDF['t'], alignTimeBounds)
        evMask = hf.getTimeMaskFromRanges(
            eventDF['t'], alignTimeBounds)
    else:
        tMask = pd.Series(True, index=stimStatus.index)
        nspMask = pd.Series(True, index=thisNspDF.index)
        evMask = pd.Series(True, index=eventDF.index)
    if (~nspMask).any():
        thisNspDF.loc[~nspMask, 'tapDetectSignal'] = np.nan
        thisNspDF.loc[:, 'tapDetectSignal'] = (
            thisNspDF.loc[:, 'tapDetectSignal']
            .interpolate(method='linear', limit_area='inside')
            .fillna(method='bfill').fillna(method='ffill'))
    nspTrigFinder = 'getThresholdCrossings'
    if nspTrigFinder == 'getTriggers':
        nspPeakIdx = hf.getTriggers(
            thisNspDF['tapDetectSignal'], iti=sessTapOptsNSP['iti'], itiWiggle=.2,
            fs=nspSamplingRate, plotting=False, absVal=False,
            thres=sessTapOptsNSP['thres'], edgeType='rising', keep_max=True)
    elif nspTrigFinder == 'getThresholdCrossings':
        nspPeakIdx, _ = hf.getThresholdCrossings(
            thisNspDF['tapDetectSignal'], thresh=sessTapOptsNSP['thres'],
            iti=sessTapOptsNSP['iti'], fs=nspSamplingRate,
            edgeType='rising', itiWiggle=.2,
            absVal=False, plotting=False)
    nspTapTimes = thisNspDF.loc[nspPeakIdx, 't'].to_numpy()
    nspDiracSt = SpikeTrain(
        times=nspTapTimes, units='s',
        t_start=thisNspDF['t'].min() * pq.s,
        t_stop=thisNspDF['t'].max() * pq.s)
    #
    ampUpdateMask = (
        (eventDF['property'] == 'amplitude') &
        (eventDF['value'] > 0) &
        (evMask)
        )
    ampUpdateMask = (eventDF['property'] == 'amplitude')
    ampMask = stimStatus['t'].isin(eventDF.loc[ampUpdateMask, 't'])
    #
    categories = stimStatus.loc[ampMask, availableCateg + ['t']].reset_index(drop=True)
    categories.loc[categories['amplitude'] > 0, 'stimCat'] = 'stimOn'
    categories.loc[categories['amplitude'] == 0, 'stimCat'] = 'stimOff'
    # check for alternation
    onOffCheck = pd.Series(0, categories.index)
    onOffCheck.loc[categories['stimCat'] == 'stimOn'] = 1
    onOffCheck.loc[categories['stimCat'] == 'stimOff'] = -1
    simultOn = onOffCheck.cumsum()
    '''
    try:
        assert simultOn.max() == 1
    except Exception:
        traceback.print_exc()
        pdb.set_trace()
    '''
    #
    globalStart = stimStatus['t'].min()
    globalStop = stimStatus['t'].max()
    #
    categories.loc[:, 'detectionDelay'] = 999
    print('getting stimulation times')
    if not arguments['disableRefinement']:
        for name, group in tqdm(categories.groupby(['amplitudeRound'])):
            prgIdx = int(group['program'].iloc[0])
            grpIdx = int(group['activeGroup'].iloc[0])
            stimRate = group['RateInHz'].iloc[0]
            searchRadius = [-2 * stimRate ** (-1), 2 * stimRate ** (-1)]
            # searchRadius = [-25e-3, 25e-3]
            targetLags = np.arange(
                searchRadius[0] * trigRasterSamplingRate,
                searchRadius[1] * trigRasterSamplingRate + 1,
                # -0.8 * trigRasterSamplingRate,
                # 0.1 * trigRasterSamplingRate + 1,
                dtype=int)
            targetLagsSrs = pd.Series(
                targetLags, index=targetLags * trigSampleInterval)
            pulsesSt = insSeg.filter(objects=SpikeTrain, name='seg0_g{}p{}#0'.format(grpIdx, prgIdx))
            assert len(pulsesSt) == 1
            #
            groupStart = group['t'].min()
            groupStop = group['t'].max()
            # expand window to make sure we don't clip edges
            windowStart = max(
                groupStart + 5 * searchRadius[0],
                globalStart
                )
            windowStop = min(
                groupStop + 5 * searchRadius[1],
                globalStop
            )
            pulseMask = (pulsesSt[0] >= windowStart * pq.s) & (pulsesSt[0] <= windowStop * pq.s)
            theseInsPulses = pulsesSt[0][np.where(pulseMask)].copy()
            # pulsesSt[0].magnitude[]
            theseInsRaster = binarize(
                theseInsPulses,
                sampling_rate=trigRasterSamplingRate * pq.Hz,
                t_start=windowStart * pq.s,
                t_stop=windowStop * pq.s)
            rasterT = windowStart + np.arange(theseInsRaster.shape[0]) * trigSampleInterval
            theseInsRasterSrs = pd.Series(
                theseInsRaster, index=rasterT
                )
            theseInsTrigs = hf.gaussianSupport(
                support=theseInsRasterSrs,
                gaussWid=gaussWid, fs=trigRasterSamplingRate, returnCopy=True)
            #
            nspPulseMask = (nspDiracSt >= windowStart * pq.s) & (nspDiracSt <= windowStop * pq.s)
            if nspPulseMask.any():
                theseNspPulses = nspDiracSt[np.where(nspPulseMask)].copy()
                theseNspRaster = binarize(
                    theseNspPulses,
                    sampling_rate=trigRasterSamplingRate * pq.Hz,
                    t_start=windowStart * pq.s,
                    t_stop=windowStop * pq.s)
                theseNspRasterSrs = pd.Series(
                    theseNspRaster, index=rasterT
                    )
                theseNspTrigs = hf.gaussianSupport(
                    support=theseNspRasterSrs,
                    gaussWid=gaussWid, fs=trigRasterSamplingRate, returnCopy=True)
                trigRaster = pd.DataFrame({
                    't': rasterT,
                    'nspDiracDelta': theseNspRasterSrs.to_numpy(),
                    'nspTrigs': theseNspTrigs.to_numpy(),
                    'insDiracDelta': theseInsRasterSrs.to_numpy(),
                    'insTrigs': theseInsTrigs.to_numpy(),
                })
                zeroMask = pd.Series(False, index=trigRaster.index)
                if (targetLags > 0).any():
                    zeroMask = zeroMask | (trigRaster.index < targetLags.max())
                if (targetLags < 0).any():
                    maxTRI = trigRaster.index.max()
                    zeroMask = zeroMask | (trigRaster.index > maxTRI + targetLags.min())
                # trigRaster.loc[zeroMask, 'insTrigs'] = 0
                # print('Calculating cross corr')
                xCorrSrs = targetLagsSrs.apply(
                    corrAtLag, xSrs=trigRaster['nspTrigs'],
                    ySrs=trigRaster['insTrigs'])
                maxLag = xCorrSrs.idxmax()
                # print('maxLag = {:.1f} millisec'.format(1000 * maxLag))
                if arguments['plotting']:
                    fig, ax, figSaveOpts = hf.plotCorrSynchReport(
                        _trigRaster=trigRaster, _searchRadius=searchRadius,
                        _targetLagsSrs=targetLagsSrs, _maxLag=maxLag, _xCorrSrs=xCorrSrs,
                        )
                    synchReportPDF.savefig(**figSaveOpts)
                    # plt.show()
                    plt.close()
                categories.loc[group.loc[group['stimCat'] == 'stimOn', :].index, 'detectionDelay'] = maxLag
                categories.loc[group.loc[group['stimCat'] == 'stimOn', :].index, 't'] += maxLag
    # pdb.set_trace()
    if (segIdx == 0) and arguments['plotParamHistograms']:
        fig, ax = plt.subplots()
        # phantomAx = ax.twinx()
        theseEvents = (
            categories
            .loc[categories['detectionDelay'] < 999, :])
        if theseEvents.size > 0:
            # pdb.set_trace()
            # customMessages = []
            listOfLegends = []
            for evN, evG in theseEvents.groupby('RateInHz'):
                thisLegend = 'RateInHz = {} median = {:.3f} msec, std = {:.3f} msec'.format(evN, 1000 * evG['detectionDelay'].median(), 1000 * evG['detectionDelay'].std())
                sns.distplot(
                    evG.loc[~evG['detectionDelay'].isna(), 'detectionDelay'],
                    # bins=200, kde=False,
                    ax=ax, label=thisLegend)
                # customMessages.append('\n'.join([
                #
                #     ]))
            '''
            customLines = [
                Line2D([0], [0], color='k', alpha=0)
                for custMess in customMessages
                ]
            phantomAx.set_yticks([])
            listOfLegends.append(phantomAx.legend(customLines, customMessages, loc='upper left'))
            '''
            listOfLegends.append(ax.legend())
            print(
                theseEvents
                .sort_values('detectionDelay', ascending=False)
                .loc[:, ['t', 'detectionDelay']]
                .head(10)
                )
            fig.savefig(
                os.path.join(
                    figureFolder, 'stimDetectionDelayDistribution{}.pdf'.format(inputINSBlockSuffix)))
    #
    if arguments['makeControl']:
        midTimes = []
        for name, group in stimStatus.groupby('amplitudeRound'):
            if name > 0:
                ampOn = group.query('amplitude>0')
                if len(ampOn):
                    tStart = ampOn['t'].iloc[0]
                    prevIdx = max(ampOn.index[0] - 1, stimStatus.index[0])
                    tPrev = stimStatus.loc[prevIdx, 't']
                    midTimes.append((tStart + tPrev) / 2)
        #
        midCategories = pd.DataFrame(midTimes, columns=['t'])
        midCategories['stimCat'] = 'NA'
        midCategories['amplitude'] = ns5.metaFillerLookup['amplitude']
        midCategories['program'] = ns5.metaFillerLookup['program']
        midCategories['RateInHz'] = ns5.metaFillerLookup['RateInHz']
        #
        alignEventsDF = pd.concat((
            categories, midCategories),
            axis=0, ignore_index=True, sort=True)
    else:
        alignEventsDF = categories
    alignEventsDF.sort_values('t', inplace=True, kind='mergesort')
    uniqProgs = pd.unique(alignEventsDF['program'])
    #  pull actual electrode names
    alignEventsDF['electrode'] = np.nan
    for name, group in alignEventsDF.groupby(['activeGroup', 'program']):
        gName = int(name[0])
        pName = int(name[1])
        if pName == 999:
            alignEventsDF.loc[group.index, 'electrode'] = 'NA'
        else:
            unitName = 'g{}p{}#0'.format(gName, pName)
            unitCandidates = insBlock.filter(objects=Unit, name=unitName)
            #
            if len(unitCandidates) == 1:
                thisUnit = unitCandidates[0]
                cathodes = thisUnit.annotations['cathodes']
                anodes = thisUnit.annotations['anodes']
                elecName = ''
                if isinstance(cathodes, Iterable):
                    elecName += '-' + ''.join(['E{:02d}'.format(i) for i in cathodes])
                else:
                    elecName += '-E{:02d}'.format(cathodes)
                elecName += ''
                if isinstance(anodes, Iterable):
                    elecName += '+' + ''.join(['E{:02d}'.format(i) for i in anodes])
                else:
                    elecName += '+E{:02d}'.format(anodes)
                alignEventsDF.loc[group.index, 'electrode'] = elecName
    #
    # TODO: fix synch code so that all units are present, to avoid this hack:
    alignEventsDF.loc[:, 'electrode'] = alignEventsDF['electrode'].fillna('NA')
    #
    if arguments['removeLabels'] is not None:
        labelsToRemove = arguments['removeLabels'].split(', ')
        idxToRemove = categories.index[categories['stimCat'].isin(labelsToRemove)]
        categories.drop(index=idxToRemove, inplace=True)
        categories.reset_index(drop=True)
    alignEventsDF.loc[:, 'expName'] = arguments['exp']
    # rename amplitude to avoid ambiguity with time domain amplitude signals
    alignEventsDF.rename(columns={
        'amplitude': 'trialAmplitude',
        'RateInHz': 'trialRateInHz'}, inplace=True)
    htmlOutPath = os.path.join(insDiagnosticsFolder, '{}_stimAlignTimes.html'.format(ns5FileName))
    alignEventsDF.to_html(htmlOutPath)
    alignEvents = ns5.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_stimAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #
    concatLabelsDF = alignEventsDF
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatLabelsDF.iterrows()])
    concatEvents = Event(
        name='seg{}_stimAlignTimesConcatenated'.format(segIdx),
        times=alignEvents.times,
        labels=concatLabels
        )
    newSeg = Segment(name=nspSeg.annotations['neo_name'])
    newSeg.annotate(nix_name=nspSeg.annotations['neo_name'])
    newSeg.events.append(alignEvents)
    newSeg.events.append(concatEvents)
    alignEvents.segment = newSeg
    concatEvents.segment = newSeg
    masterBlock.segments.append(newSeg)
    print('Saving events {}'.format(alignEvents.name))

nspReader.file.close()
if arguments['plotting']:
    synchReportPDF.close()

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
        print('stim times already calculated! Deleting block and starting over')
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
