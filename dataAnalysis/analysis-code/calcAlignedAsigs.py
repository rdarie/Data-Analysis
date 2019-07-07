"""  11: Calculate Firing Rates aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: short]
    --chanQuery=chanQuery           how to restrict channels if not providing a list? [default: fr]
    --blockName=blockName           name for new block [default: fr]
    --eventName=eventName           name of events object to align to [default: motionStimAlignTimes]
"""
import os, pdb, traceback
from importlib import reload
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pandas as pd
import quantities as pq

import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
    namedQueries, scratchFolder, **arguments)
verbose = False
#  source of events
if arguments['processAll']:
    eventReader = ns5.nixio_fr.NixIO(
        filename=experimentDataPath)
else:
    eventReader = ns5.nixio_fr.NixIO(
        filename=analysisDataPath)

eventBlock = eventReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in eventBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])

#  source of analogsignals
signalBlock = eventBlock

windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['window']]]

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
    fileName='{}_{}_{}'.format(
        prefix, arguments['blockName'], arguments['window']),
    folderPath=scratchFolder, chunkSize=alignedAsigsChunkSize)
