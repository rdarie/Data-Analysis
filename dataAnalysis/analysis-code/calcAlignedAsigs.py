"""  12: Calculate Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --exp=exp                                    which experimental day to analyze
    --processAll                                 process entire experimental day? [default: False]
    --window=window                              process with short window? [default: short]
    --lazy                                       load from raw, or regular? [default: False]
    --chanQuery=chanQuery                        how to restrict channels? [default: raster]
    --outputBlockName=outputBlockName            name for new block [default: raster]
    --eventBlockName=eventBlockName              name of events object to align to [default: analyze]
    --signalBlockName=signalBlockName            name of signal block [default: analyze]
    --eventName=eventName                        name of events object to align to [default: motionStimAlignTimes]
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --amplitudeFieldName=amplitudeFieldName      what is the amplitude named? [default: nominalCurrent]
    --verbose                                    print diagnostics? [default: False]
"""

import os, pdb, traceback, sys
from importlib import reload
#import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pandas as pd
import quantities as pq
#
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
    os.makedirs(analysisSubFolder, exist_ok=True)
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)

assembledName = ''
if arguments['processAll']:
    prefix = assembledName
    #  paths relevant to the entire experimental day
    '''
    eventPath = os.path.join(
        scratchFolder, '{}',
        assembledName + '_{}.nix'.format(arguments['eventBlockName'])).format(arguments['analysisName'])
    '''
    eventPath = os.path.join(
        scratchFolder,
        assembledName + '_{}.nix'.format(arguments['eventBlockName']))
    signalPath = os.path.join(
        scratchFolder, '{}',
        assembledName + '_{}.nix'.format(arguments['signalBlockName'])).format(arguments['analysisName'])
else:
    prefix = ns5FileName
    '''
    eventPath = os.path.join(
        scratchFolder, '{}',
        ns5FileName + '_{}.nix'.format(arguments['eventBlockName'])).format(arguments['analysisName'])
    '''
    eventPath = os.path.join(
        scratchFolder,
        ns5FileName + '_{}.nix'.format(arguments['eventBlockName']))
    signalPath = os.path.join(
        scratchFolder, '{}',
        ns5FileName + '_{}.nix'.format(arguments['signalBlockName'])).format(arguments['analysisName'])

print('Loading events from {}'.format(eventPath))
print('Loading signal from {}'.format(signalPath))

eventReader, eventBlock = ns5.blockFromPath(
    eventPath, lazy=arguments['lazy'])
#
if arguments['eventBlockName'] == arguments['signalBlockName']:
    signalReader = eventReader
    signalBlock = eventBlock
else:
    signalReader, signalBlock = ns5.blockFromPath(
        signalPath, lazy=arguments['lazy'])
# 
windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['window']]]
#   arguments['chanNames'] = [
#       'elec85#0_raster', 'elec85#1_raster',
#       'elec71#1_raster', 'elec71#2_raster',
#       'elec75#0_raster', 'elec75#1_raster',
#       'elec77#0_raster', 'elec77#1_raster', 'elec77#2_raster',
#       ]
stimConditionNames = [
    'electrode', arguments['amplitudeFieldName'], 'RateInHz']
motionConditionNames = [
    'pedalMovementCat', 'pedalSizeCat', 'pedalDirection']
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
        sys.exit()
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
        arguments['outputBlockName'], arguments['window']),
    folderPath=alignSubFolder, chunkSize=alignedAsigsChunkSize)
