"""  12: Calculate Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: short]
    --chanQuery=chanQuery           how to restrict channels? [default: (chanName.str.endswith(\'raster\'))]
    --blockName=blockName           name for new block [default: raster]
    --eventName=eventName           name of events object to align to [default: motionStimAlignTimes]
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

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(int(arguments['--trialIdx']), arguments['--exp'])
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
if arguments['--processAll']:
    signalReader = ns5.nixio_fr.NixIO(
        filename=experimentBinnedSpikePath)
else:
    signalReader = ns5.nixio_fr.NixIO(
        filename=binnedSpikePath)

signalBlock = signalReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToTrigger = None
windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['--window']]]
#  chansToTrigger = ['elec75#0_raster', 'elec75#1_raster']

if arguments['--processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
ns5.getAsigsAlignedToEvents(
    eventBlock=eventBlock, signalBlock=signalBlock,
    chansToTrigger=chansToTrigger,
    chanQuery=arguments['--chanQuery'],
    eventName=arguments['--eventName'],
    windowSize=windowSize,
    appendToExisting=True,
    checkReferences=False,
    verbose=verbose,
    fileName=prefix + '_{}_{}'.format(
        arguments['--blockName'], arguments['--window']),
    folderPath=scratchFolder, chunkSize=alignedAsigsChunkSize)
