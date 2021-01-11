"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze
    --exp=exp                              which experimental day to analyze
    --verbose                              print out messages? [default: False]
    --lazy                                 load as neo proxy objects or no? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --sourceFilePrefix=sourceFilePrefix    Does the block have an unusual prefix
    --sourceFileSuffix=sourceFileSuffix    append a name to the resulting blocks?
    --chanQuery=chanQuery                  how to restrict channels if not providing a list? [default: fr]
    --samplingRate=samplingRate            resample the result??
    --rigOnly                              is there no INS block? [default: False]
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from copy import copy
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.ns5 as ns5
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb
import traceback
from importlib import reload
import json
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
binOpts = rasterOpts['binOpts'][arguments['analysisName']]


def calcBlockAnalysisWrapper():
    arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
        namedQueries, scratchFolder, **arguments)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
        )
    if not os.path.exists(analysisSubFolder):
        os.makedirs(analysisSubFolder, exist_ok=True)
    if arguments['samplingRate'] is not None:
        samplingRate = float(arguments['samplingRate']) * pq.Hz
    else:
        samplingRate = float(1 / binOpts['binInterval']) * pq.Hz
    #
    if arguments['sourceFileSuffix'] is not None:
        sourceFileSuffix = '_' + arguments['sourceFileSuffix']
    else:
        sourceFileSuffix = ''
    #  electrode array name (changes the prefix of the file)
    arrayName = arguments['sourceFilePrefix']
    if arguments['sourceFilePrefix'] is not None:
        blockBaseName = ns5FileName.replace(
            'Block', arguments['sourceFilePrefix'])
    else:
        blockBaseName = copy(ns5FileName)
    #
    chunkingInfoPath = os.path.join(
        scratchFolder,
        blockBaseName + sourceFileSuffix + '_chunkingInfo.json'
        )
    #
    if os.path.exists(chunkingInfoPath):
        with open(chunkingInfoPath, 'r') as f:
            chunkingMetadata = json.load(f)
    else:
        chunkingMetadata = {
            '0': {
                'filename': os.path.join(
                    scratchFolder, blockBaseName + sourceFileSuffix + '.nix'
                    ),
                'partNameSuffix': '',
                'chunkTStart': 0,
                'chunkTStop': 'NaN'
            }}
    #  ########################################
    spikeSource = 'tdc'
    #  ########################################
    # Scan ahead through all files and ensure that
    # spikeTrains and units are present across all assembled files
    masterChanDF = pd.DataFrame([], columns=[
        'index', 'channel_names', 'channel_ids',
        'hasUnits', 'hasAsigs'
        ])
    masterUnitDF = pd.DataFrame([], columns=['parentChanName'])
    channelIndexCache = {}
    unitCache = {}
    dataBlockCache = {}
    spikeBlockCache = {}
    # get list of channels and units
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        # chunkIdx = int(chunkIdxStr)
        nameSuffix = sourceFileSuffix + chunkMeta['partNameSuffix']
        nspPath = os.path.join(
            scratchFolder,
            blockBaseName + nameSuffix + '.nix')
        #####
        print('Loading {}'.format(nspPath))
        dataReader, dataBlock = ns5.blockFromPath(
            nspPath, lazy=arguments['lazy'],
            reduceChannelIndexes=True)
        if spikeSource == 'tdc':
            tdcPath = os.path.join(
                scratchFolder,
                'tdc_' + blockBaseName + nameSuffix,
                'tdc_' + blockBaseName + nameSuffix + '.nix'
                )
            print('Loading {}'.format(tdcPath))
            spikeReader, spikeBlock = ns5.blockFromPath(
                tdcPath, lazy=arguments['lazy'],
                reduceChannelIndexes=True)
        else:
            spikeBlock = dataBlock
        if idx == 0:
            outputBlock = Block(
                name=dataBlock.name,
                **dataBlock.annotations
            )
            seg = dataBlock.segments[0]
            newSeg = Segment(
                index=0, name=seg.name,
                description=seg.description,
                file_origin=seg.file_origin,
                file_datetime=seg.file_datetime,
                rec_datetime=seg.rec_datetime,
                **seg.annotations
            )
        for chIdx in dataBlock.filter(objects=ChannelIndex):
            chAlreadyThere = (chIdx.name in channelIndexCache.keys())
            if not chAlreadyThere:
                newChIdx = copy(chIdx)
                newChIdx.analogsignals = []
                newChIdx.annotations['hasAsigs'] = False
                newChIdx.units = []
                newChIdx.annotations['hasUnits'] = False
                channelIndexCache[chIdx.name] = newChIdx
            if len(chIdx.analogsignals):
                channelIndexCache[chIdx.name].annotations['hasAsigs'] = True
        for unit in (spikeBlock.filter(objects=Unit)):
            uAlreadyThere = (unit.name in unitCache.keys())
            if not uAlreadyThere:
                newUnit = copy(unit)
                newUnit.spiketrains = []
                newUnit.annotations['hasSpiketrains'] = False
                newUnit.annotations['parentChanName'] = unit.channel_index.name
                unitCache[unit.name] = newUnit
            if len(unit.spiketrains):
                unitCache[unit.name].annotations['hasSpiketrains'] = True
        spikeBlockCache[chunkIdxStr] = spikeBlock
        dataBlockCache[chunkIdxStr] = dataBlock
        print('Finished chunk {}'.format(chunkIdxStr))
    # link chIdxes and Units
    for uName, unit in unitCache.items():
        if unit.annotations['parentChanName'] in channelIndexCache:
            channelIndexCache[unit.annotations['parentChanName']].annotations['hasUnits'] = True
            if unit not in channelIndexCache[unit.annotations['parentChanName']]:
                channelIndexCache[unit.annotations['parentChanName']].units.append(unit)
                unit.channel_index = channelIndexCache[unit.annotations['parentChanName']]
        else:
            newChIdx = ChannelIndex(
                name=unit.annotations['parentChanName'], index=0)
            newChIdx.annotations['hasUnits'] = True
            channelIndexCache[unit.annotations['parentChanName']] = newChIdx
            if unit not in newChIdx.units:
                newChIdx.units.append(unit)
                unit.channel_index = newChIdx
    #
    outputBlock.channel_indexes = [
        v
        for k, v in channelIndexCache.items()
        if v.annotations['hasAsigs'] or v.annotations['hasUnits']]
    tdChanNames = ns5.listChanNames(
        outputBlock, arguments['chanQuery'], objType=ChannelIndex)
    outputBlock.channel_indexes = [
        v
        for k, v in channelIndexCache.items()
        if v.name in tdChanNames or v.annotations['hasUnits']
        ]
    pdb.set_trace()
    # preallocate dataframe for requested asigs
    # preallocate dict for requested spikes
    # preallocate dict for requested events
    # iterate through again
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        # get dataframe of requested analog signals
        # get requested spiketrains
        pdb.set_trace()
    return


if __name__ == "__main__":
    runProfiler = False
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        if arguments['lazy']:
            nameSuffix = 'lazy'
        else:
            nameSuffix = 'not_lazy'
        prf.profileFunction(
            topFun=calcBlockAnalysisWrapper,
            modulesToProfile=[ash, ns5, hf],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        calcBlockAnalysisWrapper()