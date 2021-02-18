"""10a: Calculate align Times ##WIP
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                  which trial to analyze [default: 1]
    --exp=exp                            which experimental day to analyze
    --processAll                         process entire experimental day? [default: False]
    --plotParamHistograms                plot pedal size, amplitude, duration distributions? [default: False]
    --analysisName=analysisName          append a name to the resulting blocks? [default: default]
    --makeControl                        make control align times? [default: False]
    --lazy                               load from raw, or regular? [default: False]
    --removeLabels=removeLabels          remove certain labels, e.g. stimOff (comma separated)
"""
import os, pdb, traceback, sys
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import quantities as pq
#  import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.helper_functions_new as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.preproc.mdt as mdt
#  import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Iterable
from tqdm import tqdm
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
insReader = neo.NixIO(
    filename=insDataPath)
insBlock = insReader.read_block(0)
#  all experimental days?
#  if arguments['processAll']:
#      alignTimeBounds = []
#      dataReader = neo.io.nixio_fr.NixIO(
#          filename=experimentDataPath)
#  else:
#      # alignTimeBounds = alignTimeBoundsLookup[experimentName][int(arguments['blockIdx'])]

dataReader = neo.io.nixio_fr.NixIO(
    filename=analysisDataPath)

dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in dataBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])
try:
    alignTimeBounds = [
    alignTimeBoundsLookup[int(arguments['blockIdx'])]
    ]
except Exception:
    alignTimeBounds = [[
        [
            float(dataBlock.segments[0].filter(objects=AnalogSignalProxy)[0].t_start),
            float(dataBlock.segments[-1].filter(objects=AnalogSignalProxy)[0].t_stop)
        ]
    ]]
availableCateg = [
    'amplitude', 'program', 'activeGroup', 'RateInHz']
progAmpNames = rcsa_helpers.progAmpNames
expandCols = [
    'RateInHz', 'movement', 'program', 'trialSegment']
deriveCols = ['amplitude', 'amplitudeRound']
columnsToBeAdded = (
    expandCols + deriveCols + progAmpNames)

#  allocate block to contain events
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=dataBlock.annotations['neo_name'])

checkReferences = False
for segIdx, dataSeg in enumerate(dataBlock.segments):
    eventProxysList = dataSeg.events
    if checkReferences:
        for evP in eventProxysList:
            print('segIdx {}, evP.name {}'.format(
                segIdx, evP.name))
            print('evP._event_channel_index = {}'.format(
                 evP._event_channel_index))
            evP_ch = evP._event_channel_index
            mts = evP._rawio.file.blocks[0].groups[segIdx].multi_tags
            try:
                assert evP.name in mts[evP_ch].name
            except Exception:
                traceback.print_exc()
    dataSegEvents = [evP.load() for evP in eventProxysList]
    eventDF = ns5.eventsToDataFrame(
        dataSegEvents, idxT='t',
        names=['property', 'value']
        )
    stimStatus = mdt.stimStatusSerialtoLong(
        eventDF, idxT='t', namePrefix='', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    print('Usign alignTimeBounds {}'.format(alignTimeBounds[segIdx]))
    #  #)
    tMask = (
        (stimStatus['t'] > alignTimeBounds[segIdx][0][0]) &
        (stimStatus['t'] < alignTimeBounds[segIdx][-1][1])
        )
    stimStatus = stimStatus.loc[tMask, :].reset_index(drop=True)
    #
    ampUpdateMask = (
        (eventDF['property'] == 'amplitude') &
        (eventDF['value'] > 0)
        )
    ampUpdateMask = (eventDF['property'] == 'amplitude')
    ampMask = stimStatus['t'].isin(eventDF.loc[ampUpdateMask, 't'])
    #
    categories = stimStatus.loc[ampMask, availableCateg + ['t']]
    categories.loc[categories['amplitude'] > 0, 'stimCat'] = 'stimOn'
    categories.loc[categories['amplitude'] == 0, 'stimCat'] = 'stimOff'
    # pdb.set_trace()
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
        midCategories['stimCat'] = 'control'
        midCategories['amplitude'] = 0
        midCategories['program'] = 999
        midCategories['RateInHz'] = 0
        #
        alignEventsDF = pd.concat((
            categories, midCategories),
            axis=0, ignore_index=True, sort=True)
    else:
        alignEventsDF = categories
    alignEventsDF.sort_values('t', inplace=True, kind='mergesort')
    #
    uniqProgs = pd.unique(alignEventsDF['program'])
    #  pull actual electrode names
    alignEventsDF['electrode'] = np.nan
    for name, group in alignEventsDF.groupby(['activeGroup', 'program']):
        gName = int(name[0])
        pName = int(name[1])
        if pName == 999:
            alignEventsDF.loc[group.index, 'electrode'] = 'control'
        else:
            unitName = 'g{}p{}#0'.format(gName, pName)
            unitCandidates = insBlock.filter(objects=Unit, name=unitName)
            #
            if len(unitCandidates) == 1:
                thisUnit = unitCandidates[0]
                cathodes = thisUnit.annotations['cathodes']
                anodes = thisUnit.annotations['anodes']
                elecName = ''
                if isinstance(anodes, Iterable):
                    elecName += '+ ' + ', '.join(['E{}'.format(i) for i in anodes])
                else:
                    elecName += '+ E{}'.format(anodes)
                elecName += ' '
                if isinstance(cathodes, Iterable):
                    elecName += '- ' + ', '.join(['E{}'.format(i) for i in cathodes])
                else:
                    elecName += '- E{}'.format(cathodes)
                alignEventsDF.loc[group.index, 'electrode'] = elecName
    #
    # TODO: fix synch code so that all units are present, to avoid this hack:
    alignEventsDF.loc[:, 'electrode'] = alignEventsDF['electrode'].fillna('NA')
    if arguments['removeLabels'] is not None:
        pdb.set_trace()
        labelsToRemove = ', '.split(arguments['removeLabels'])
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
    newSeg = Segment(name=dataSeg.annotations['neo_name'])
    newSeg.annotate(nix_name=dataSeg.annotations['neo_name'])
    newSeg.events.append(alignEvents)
    newSeg.events.append(concatEvents)
    alignEvents.segment = newSeg
    concatEvents.segment = newSeg
    masterBlock.segments.append(newSeg)
    print('Saving events {}'.format(alignEvents.name))

dataReader.file.close()

masterBlock.create_relationship()
allSegs = list(range(len(masterBlock.segments)))

outputPath = os.path.join(
    scratchFolder,
    ns5FileName + '_epochs'
    )
if not os.path.exists(outputPath + '.nix'):
    writer = ns5.NixIO(filename=outputPath + '.nix')
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
