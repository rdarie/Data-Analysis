"""10a: Calculate align Times ##WIP
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                  which trial to analyze [default: 1]
    --exp=exp                            which experimental day to analyze
    --processAll                         process entire experimental day? [default: False]
    --plotParamHistograms                plot pedal size, amplitude, duration distributions? [default: False]
"""
import os, pdb, traceback
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
#  import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Iterable

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

#  fetch stim details
insReader = neo.NixIO(
    filename=insDataPath)
insBlock = insReader.read_block(0)

#  all experimental days?
if arguments['processAll']:
    dataReader = neo.io.nixio_fr.NixIO(
        filename=experimentDataPath)
else:
    alignTimeBounds = alignTimeBoundsLookup[int(arguments['trialIdx'])]    
    dataReader = neo.io.nixio_fr.NixIO(
        filename=analysisDataPath)

dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in dataBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])

availableCateg = [
    'amplitude', 'program', 'RateInHz']

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

blockIdx = 0
checkReferences = False
for segIdx, dataSeg in enumerate(dataBlock.segments):
    print('Calculating stim align times for trial {}'.format(segIdx))
    eventProxysList = dataSeg.events
    if checkReferences:
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
    dataSegEvents = [evP.load() for evP in eventProxysList]
    eventDF = ns5.eventsToDataFrame(
        dataSegEvents, idxT='t',
        names=['property', 'value']
        )
    stimStatus = hf.stimStatusSerialtoLong(
        eventDF, idxT='t', namePrefix='', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    
    tMask = stimStatus['t'] > 0
    stimStatus = stimStatus.loc[tMask, :].reset_index(drop=True)

    ampMask = stimStatus['amplitude'] > 0
    categories = stimStatus.loc[
        ampMask,
        availableCateg + ['t']]
    categories['stimCat'] = 'stimOn'
    
    offIdx = []
    midTimes = []
    for name, group in stimStatus.groupby('amplitudeRound'):
        ampOff = group.query('amplitude==0')
        if len(ampOff):
            offIdx.append(ampOff.index[0])
        if name > 0:
            ampOn = group.query('amplitude>0')
            if len(ampOn):
                tStart = ampOn['t'].iloc[0]
                tPrev = stimStatus.loc[ampOn.index[0] - 1, 't']
                midTimes.append((tStart + tPrev) / 2)
    offCategories = stimStatus.loc[
        offIdx,
        availableCateg + ['t']]
    offCategories['stimCat'] = 'stimOff'
    
    midCategories = pd.DataFrame(midTimes, columns=['t'])
    midCategories['stimCat'] = 'control'
    midCategories['amplitude'] = 0
    midCategories['program'] = 999
    midCategories['RateInHz'] = 0

    alignEventsDF = pd.concat((
        categories, offCategories, midCategories),
        axis=0, ignore_index=True, sort=True)
    alignEventsDF.sort_values('t', inplace=True, kind='mergesort')
    #  pdb.set_trace()
    uniqProgs = pd.unique(alignEventsDF['program'])
    #  pull actual electrode names
    alignEventsDF['electrode'] = np.nan
    for pName in uniqProgs:
        pMask = alignEventsDF['program'] == pName
        if pName == 999:
            alignEventsDF.loc[pMask, 'electrode'] = 'control'
        else:
            unitName = 'g0p{}'.format(int(pName))
            thisUnit = insBlock.filter(objects=Unit, name=unitName)[0]
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
            alignEventsDF.loc[pMask, 'electrode'] = elecName

    alignEvents = ns5.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_stimAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #  pdb.set_trace()
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

dataReader.file.close()

masterBlock.create_relationship()
allSegs = list(range(len(masterBlock.segments)))
if arguments['processAll']:
    ns5.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeAsigs=False, writeSpikes=False, writeEvents=True,
        fileName=experimentName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
else:
    ns5.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeAsigs=False, writeSpikes=False, writeEvents=True,
        fileName=ns5FileName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
