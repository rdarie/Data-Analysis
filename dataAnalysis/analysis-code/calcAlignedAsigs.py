"""  11: Calculate Firing Rates aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: short]
    --unitQuery=unitQuery           how to restrict channels if not providing a list? [default: (chanName.str.endswith(\'fr\'))]
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

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

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

chansToTrigger = None
# chansToTrigger = ns5.listChanNames(signalBlock, chanQuery)

windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['window']]]

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName

ns5.getAsigsAlignedToEvents(
    eventBlock=eventBlock, signalBlock=signalBlock,
    chansToTrigger=chansToTrigger,
    chanQuery=arguments['unitQuery'],
    eventName=arguments['eventName'],
    windowSize=windowSize,
    appendToExisting=False,
    checkReferences=False,
    verbose=verbose,
    fileName=prefix + '_{}_{}'.format(
        arguments['blockName'], arguments['window']),
    folderPath=scratchFolder, chunkSize=alignedAsigsChunkSize)
