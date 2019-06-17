"""  11: Calculate Firing Rates aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: shortWindow]
"""
import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
import quantities as pq

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

#  source of events
if arguments['--processAll']:
    eventReader = neo.io.nixio_fr.NixIO(
        filename=experimentDataPath)
else:
    eventReader = neo.io.nixio_fr.NixIO(
        filename=analysisDataPath)

eventBlock = eventReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in eventBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])

#  source of analogsignals
signalBlock = eventBlock

chansToTrigger = np.unique([
    i.name
    for i in signalBlock.filter(objects=AnalogSignalProxy)])
eventName = 'stimAlignTimes'

#  chansToTrigger = [
#      'ins_td3', 'position', 'amplitude',
#      'elec75#0_fr', 'elec75#1_fr']

if arguments['--processAll']:
    preproc.analogSignalsAlignedToEvents(
        eventBlock=eventBlock, signalBlock=signalBlock,
        chansToTrigger=chansToTrigger, eventName=eventName,
        windowSize=[i * pq.s for i in rasterOpts[arguments['--window']]],
        appendToExisting=False,
        checkReferences=False,
        fileName=experimentName + '_triggered_{}'.format(arguments['--window']),
        folderPath=scratchFolder)
else:
    preproc.analogSignalsAlignedToEvents(
        eventBlock=eventBlock, signalBlock=signalBlock,
        chansToTrigger=chansToTrigger, eventName=eventName,
        windowSize=[i * pq.s for i in rasterOpts[arguments['--window']]],
        appendToExisting=False,
        checkReferences=False,
        fileName=ns5FileName + '_triggered_{}'.format(arguments['--window']),
        folderPath=scratchFolder)
