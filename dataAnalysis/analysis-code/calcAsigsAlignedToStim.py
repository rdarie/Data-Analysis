"""  11: Calculate Firing Rates aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
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
eventReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
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
eventName = 'alignTimes'
windowSize = [i * pq.s for i in rasterOpts['windowSize']]
#  chansToTrigger = ['PC1', 'PC2']
preproc.analogSignalsAlignedToEvents(
    eventBlock=eventBlock, signalBlock=signalBlock,
    chansToTrigger=chansToTrigger, eventName=eventName,
    windowSize=windowSize, appendToExisting=False,
    checkReferences=False,
    fileName=experimentName + '_triggered',
    folderPath=scratchFolder)