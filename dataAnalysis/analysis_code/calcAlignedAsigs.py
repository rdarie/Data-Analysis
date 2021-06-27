"""  12: Calculate Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                                  which trial to analyze [default: 1]
    --exp=exp                                            which experimental day to analyze
    --processAll                                         process entire experimental day? [default: False]
    --window=window                                      process with short window? [default: short]
    --lazy                                               load from raw, or regular? [default: False]
    --chanQuery=chanQuery                                how to restrict channels? [default: raster]
    --outputBlockSuffix=outputBlockSuffix                name for new block [default: raster]
    --signalSubfolder=signalSubfolder                    name of folder where the signal block is [default: default]
    --signalBlockSuffix=signalBlockSuffix                name of signal block
    --signalBlockPrefix=signalBlockPrefix                name of signal block
    --eventBlockPrefix=eventBlockPrefix                  name of event block
    --eventBlockSuffix=eventBlockSuffix                  name of events object to align to [default: analyze]
    --eventSubfolder=eventSubfolder                      name of folder where the event block is [default: None]
    --eventName=eventName                                name of events object to align to [default: motionStimAlignTimes]
    --analysisName=analysisName                          append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName                    append a name to the resulting blocks? [default: motion]
    --verbose                                            print diagnostics? [default: False]
"""

import os, pdb, traceback, sys
from importlib import reload
# import neo
from copy import copy, deepcopy
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pandas as pd
import quantities as pq
import json
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {
    arg.lstrip('-'): value
    for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if (overrideChanNames is not None) and (arguments['chanQuery'] in ['fr', 'fr_sqrt', 'raster']):
    arguments['chanNames'] = [i + '_{}'.format(arguments['chanQuery']) for i in overrideChanNames]
    arguments['chanQuery'] = None
else:
    arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
        namedQueries, scratchFolder, **arguments)

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder)
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder)
###
if arguments['processAll']:
    prefix = ''
    #  paths relevant to the entire experimental day
else:
    prefix = ns5FileName
    if arguments['signalBlockPrefix'] is not None:
        signalPrefix = '{}{:0>3}'.format(arguments['signalBlockPrefix'], blockIdx)
    else:
        signalPrefix = ns5FileName
    if arguments['eventBlockPrefix'] is not None:
        eventPrefix = '{}{:0>3}'.format(arguments['eventBlockPrefix'], blockIdx)
    else:
        eventPrefix = ns5FileName

if arguments['eventBlockSuffix'] is not None:
    eventBlockSuffix = '_{}'.format(arguments['eventBlockSuffix'])
else:
    eventBlockSuffix = ''
if arguments['signalBlockSuffix'] is not None:
    signalBlockSuffix = '_{}'.format(arguments['signalBlockSuffix'])
else:
    signalBlockSuffix = ''

if arguments['eventSubfolder'] != 'None':
    eventPath = os.path.join(
        scratchFolder, arguments['eventSubfolder'],
        eventPrefix + '{}.nix'.format(eventBlockSuffix))
else:
    eventPath = os.path.join(
        scratchFolder,
        eventPrefix + '{}.nix'.format(eventBlockSuffix))
if arguments['signalSubfolder'] != 'None':
    signalPath = os.path.join(
        scratchFolder, arguments['signalSubfolder'],
        signalPrefix + '{}.nix'.format(signalBlockSuffix))
    chunkingInfoPath = os.path.join(
        scratchFolder, arguments['signalSubfolder'],
        signalPrefix + signalBlockSuffix + '_chunkingInfo.json'
        )
else:
    signalPath = os.path.join(
        scratchFolder,
        signalPrefix + '{}.nix'.format(signalBlockSuffix))
    chunkingInfoPath = os.path.join(
        scratchFolder,
        signalPrefix + signalBlockSuffix + '_chunkingInfo.json'
        )
#
#
if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC'):
    # has stim but no motion
    if arguments['eventName'] == 'motion':
        print('Block does not have motion!')
        sys.exit()
    if arguments['eventName'] == 'stim':
        eventName = 'stimAlignTimes'
    minNConditionRepetitions['categories'] = stimConditionNames
elif blockExperimentType == 'proprio-motionOnly':
    # has motion but no stim
    if arguments['eventName'] == 'motion':
        eventName = 'motionAlignTimes'
    if arguments['eventName'] == 'stim':
        print('Block does not have stim!')
        sys.exit()
    minNConditionRepetitions['categories'] = motionConditionNames
elif blockExperimentType == 'proprio':
    if arguments['eventName'] == 'stim':
        eventName = 'stimPerimotionAlignTimes'
    elif arguments['eventName'] == 'motion':
        eventName = 'motionStimAlignTimes'
    minNConditionRepetitions['categories'] = motionConditionNames + stimConditionNames
elif blockExperimentType == 'isi':
    if arguments['eventName'] == 'stim':
        eventName = 'stimAlignTimes'
    minNConditionRepetitions['categories'] = stimConditionNames

print('Loading events from {}'.format(eventPath))
print('Loading signal from {}'.format(signalPath))

eventReader, eventBlock = ns5.blockFromPath(
    eventPath, lazy=arguments['lazy'],
    loadList={'events': ['seg0_{}'.format(eventName)]},
    purgeNixNames=True)
#
if eventPath == signalPath:
    signalReader = eventReader
    signalBlock = eventBlock
else:
    signalReader, signalBlock = ns5.blockFromPath(
        signalPath, lazy=arguments['lazy'],
        chunkingInfoPath=chunkingInfoPath, purgeNixNames=True)
#
if len(signalBlock.segments) != len(eventBlock.segments):
    # assume eventBlock is not chunked, while signalBlock is chunked
    evSegNames = [evSeg.name for evSeg in eventBlock.segments]
    targetEvent = eventBlock.filter(objects=Event)[0]
    if chunkingInfoPath is not None:
        if os.path.exists(chunkingInfoPath):
            with open(chunkingInfoPath, 'r') as f:
                chunkingMetadata = json.load(f)
            for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
                if signalBlock.segments[idx].name not in evSegNames:
                    newSeg = Segment(name=signalBlock.segments[idx].name)
                    newSeg.block = eventBlock
                    eventBlock.segments.append(newSeg)
                else:
                    newSeg = eventBlock.segments[0]
                    newSeg.events = []
                tMask = (targetEvent.times.magnitude >= chunkMeta['chunkTStart']) & (targetEvent.times.magnitude < chunkMeta['chunkTStop'])
                newEvent = targetEvent[tMask].copy()
                newEvent.name = 'seg{}_{}'.format(idx, ns5.childBaseName(newEvent.name, 'seg'))
                newEvent.segment = newSeg
                newSeg.events.append(newEvent)
# [ev.name for ev in eventBlock.filter(objects=Event)]
windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['window']]]
#   arguments['chanNames'] = [
#       'elec85#0_raster', 'elec85#1_raster',
#       'elec71#1_raster', 'elec71#2_raster',
#       'elec75#0_raster', 'elec75#1_raster',
#       'elec77#0_raster', 'elec77#1_raster', 'elec77#2_raster',
#       ]

ns5.getAsigsAlignedToEvents(
    eventBlock=eventBlock, signalBlock=signalBlock,
    chansToTrigger=arguments['chanNames'],
    chanQuery=arguments['chanQuery'],
    eventName=eventName,
    windowSize=windowSize,
    appendToExisting=False,
    minNReps=minNConditionRepetitions,
    checkReferences=False,
    verbose=arguments['verbose'],
    fileName=prefix + '_{}_{}'.format(
        arguments['outputBlockSuffix'], arguments['window']),
    folderPath=alignSubFolder, chunkSize=alignedAsigsChunkSize)
print('Done calcAlignedAsigs')
