"""  12: Calculate Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                 which trial to analyze [default: 1]
    --exp=exp                           which experimental day to analyze
    --processAll                        process entire experimental day? [default: False]
    --window=window                     process with short window? [default: short]
    --lazy                              load from raw, or regular? [default: False]
    --analysisName=analysisName         append a name to the resulting blocks? [default: default]
    --chanQuery=chanQuery               how to restrict channels? [default: raster]
    --blockName=blockName               name for new block [default: raster]
    --eventName=eventName               name of events object to align to [default: motionStimAlignTimes]
"""

import os, pdb, traceback
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

import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {
    arg.lstrip('-'): value
    for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
    namedQueries, scratchFolder, **arguments)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
experimentBinnedSpikePath = experimentBinnedSpikePath.format(arguments['analysisName'])
binnedSpikePath = binnedSpikePath.format(arguments['analysisName'])
experimentDataPath = experimentDataPath.format(arguments['analysisName'])
analysisDataPath = analysisDataPath.format(arguments['analysisName'])
    
verbose = True
#  source of events
if arguments['processAll']:
    eventPath = experimentDataPath
else:
    eventPath = analysisDataPath
eventReader, eventBlock = ns5.blockFromPath(
    eventPath, lazy=arguments['lazy'])

#  source of analogsignals
if arguments['processAll']:
    signalPath = experimentBinnedSpikePath
else:
    signalPath = binnedSpikePath

signalReader, signalBlock = ns5.blockFromPath(
    signalPath, lazy=arguments['lazy'])

windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['window']]]
#   arguments['chanNames'] = [
#       'elec85#0_raster', 'elec85#1_raster',
#       'elec71#1_raster', 'elec71#2_raster',
#       'elec75#0_raster', 'elec75#1_raster',
#       'elec77#0_raster', 'elec77#1_raster', 'elec77#2_raster',
#       ]

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName

ns5.getAsigsAlignedToEvents(
    eventBlock=eventBlock, signalBlock=signalBlock,
    chansToTrigger=arguments['chanNames'],
    chanQuery=arguments['chanQuery'],
    eventName=arguments['eventName'],
    windowSize=windowSize,
    appendToExisting=False,
    checkReferences=False,
    verbose=verbose,
    fileName=prefix + '_{}_{}'.format(
        arguments['blockName'], arguments['window']),
    folderPath=analysisSubFolder, chunkSize=alignedAsigsChunkSize)
