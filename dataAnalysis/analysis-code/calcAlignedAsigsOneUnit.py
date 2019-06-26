"""  11: Calculate Firing Rates aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: short]
    --chanQuery=chanQuery           how to restrict channels? [default: (chanName.str.endswith(\'fr\'))]
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

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']),
    arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

verbose = False
#  source of events
if arguments['--processAll']:
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

chansToAlignNames = None
# chansToTrigger = [
#     'elec75#0_fr_sqrt', 'elec75#1_fr_sqrt',
#     'elec83#0_fr_sqrt', 'elec78#0_fr_sqrt',
#     'elec78#1_fr_sqrt']
if chansToAlignNames is None:
    chansToAlignNames = ns5.listChanNames(signalBlock, arguments['--chanQuery'])
chansToAlign = []
for n in chansToAlignNames:
    chansToAlign.append(signalBlock.filter(objects=ChannelIndex, name=n))

windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['--window']]]

pdb.set_trace()
