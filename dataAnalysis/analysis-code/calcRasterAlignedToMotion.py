"""  12: Calculate Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --processShort                  process with short window? [default: False]
"""

import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, EventProxy)
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
if arguments['--processAll']:
    signalReader = neo.io.nixio_fr.NixIO(
        filename=experimentBinnedSpikePath)
else:
    signalReader = neo.io.nixio_fr.NixIO(
        filename=binnedSpikePath)

signalBlock = signalReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToTrigger = np.unique([
    i.name
    for i in signalBlock.filter(objects=AnalogSignalProxy)])

#  chansToTrigger = ['elec75#0_raster', 'elec75#1_raster']
eventName = 'motionStimAlignTimes'

if arguments['--processAll']:
    if arguments['--processShort']:
        preproc.analogSignalsAlignedToEvents(
            eventBlock=eventBlock, signalBlock=signalBlock,
            chansToTrigger=chansToTrigger, eventName=eventName,
            windowSize=[i * pq.s for i in rasterOpts['shortWindowSize']],
            appendToExisting=True,
            checkReferences=False,
            fileName=experimentName + '_triggered_short',
            folderPath=scratchFolder)
    else:
        preproc.analogSignalsAlignedToEvents(
            eventBlock=eventBlock, signalBlock=signalBlock,
            chansToTrigger=chansToTrigger, eventName=eventName,
            windowSize=[i * pq.s for i in rasterOpts['longWindowSize']],
            appendToExisting=True,
            checkReferences=False,
            fileName=experimentName + '_triggered_long',
            folderPath=scratchFolder)
else:
    if arguments['--processShort']:
        preproc.analogSignalsAlignedToEvents(
            eventBlock=eventBlock, signalBlock=signalBlock,
            chansToTrigger=chansToTrigger, eventName=eventName,
            windowSize=[i * pq.s for i in rasterOpts['shortWindowSize']],
            appendToExisting=True,
            checkReferences=False,
            fileName=ns5FileName + '_triggered_short',
            folderPath=scratchFolder)
    else:
        preproc.analogSignalsAlignedToEvents(
            eventBlock=eventBlock, signalBlock=signalBlock,
            chansToTrigger=chansToTrigger, eventName=eventName,
            windowSize=[i * pq.s for i in rasterOpts['longWindowSize']],
            appendToExisting=True,
            checkReferences=False,
            fileName=ns5FileName + '_triggered_long',
            folderPath=scratchFolder)
