"""  11: Calculate Firing Rates aligned to Stim
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --chanQuery=chanQuery                  how to restrict channels if not providing a list? [default: fr]
    --outputBlockName=outputBlockName      name for new block [default: fr]
    --eventName=eventName                  name of events object to align to [default: motionStimAlignTimes]
    --verbose                              print diagnostics? [default: False]
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
    int(arguments['blockIdx']),
    arguments['exp'])
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
#pdb.set_trace()
experimentDataPath = experimentDataPath.format(arguments['analysisName'])
analysisDataPath = analysisDataPath.format(arguments['analysisName'])

@profile
def calcAlignedAsigsWrapped():
    #  source of events
    if arguments['processAll']:
        eventPath = experimentDataPath
    else:
        eventPath = analysisDataPath

    print('Loading events from {}'.format(eventPath))
    eventReader, eventBlock = ns5.blockFromPath(
        eventPath, lazy=arguments['lazy'])
    #  eventBlock = eventReader.read_block(
    #      block_index=0, lazy=True,
    #      signal_group_mode='split-all')
    #  for ev in eventBlock.filter(objects=EventProxy):
    #      ev.name = '_'.join(ev.name.split('_')[1:])

    #  source of analogsignals
    signalBlock = eventBlock

    windowSize = [
        i * pq.s
        for i in rasterOpts['windowSizes'][arguments['window']]]

    if arguments['processAll']:
        prefix = assembledName
    else:
        prefix = ns5FileName

    ns5.getAsigsAlignedToEvents(
        eventBlock=eventBlock, signalBlock=signalBlock,
        chansToTrigger=arguments['chanNames'],
        chanQuery=arguments['chanQuery'],
        eventName=arguments['eventName'],
        windowSize=windowSize,
        minNReps=minNConditionRepetitions,
        appendToExisting=False,
        checkReferences=False,
        verbose=arguments['verbose'],
        fileName='{}_{}_{}'.format(
            prefix, arguments['outputBlockName'], arguments['window']),
        folderPath=alignSubFolder, chunkSize=alignedAsigsChunkSize)

    print('Finished CalcAlignedAsigs')
    return

if __name__ == "__main__":
    runProfiler = True
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=calcAlignedAsigsWrapped,
            modulesToProfile=[ash, ns5],
            #outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        calcAlignedAsigsWrapped()