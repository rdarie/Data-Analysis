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
    --insFilePrefix=insFilePrefix          Does the INS block have an unusual prefix
    --sourceFileSuffix=sourceFileSuffix    append a name to the resulting blocks?
    --rigFileSuffix=rigFileSuffix          append a name to the resulting blocks?
    --spikeFileSuffix=spikeFileSuffix      append a name to the resulting blocks?
    --insFileSuffix=insFileSuffix          append a name to the resulting blocks? [default: ins]
    --spikeSource=spikeSource              append a name to the resulting blocks?
    --chanQuery=chanQuery                  how to restrict channels if not providing a list? [default: fr]
    --samplingRate=samplingRate            resample the result??
    --rigOnly                              is there no INS block? [default: False]
"""
from neo.io import NixIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from copy import copy
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.profiling as prf
from namedQueries import namedQueries
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.ns5 as ns5
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb, gc
import traceback
from scipy import signal
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

trackMemory = True


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
    if arguments['spikeFileSuffix'] is not None:
        spikeFileSuffix = '_' + arguments['spikeFileSuffix']
    else:
        spikeFileSuffix = ''
    if arguments['rigFileSuffix'] is not None:
        rigFileSuffix = '_' + arguments['rigFileSuffix']
    else:
        rigFileSuffix = ''
    if arguments['insFileSuffix'] is not None:
        insFileSuffix = '_' + arguments['insFileSuffix']
    else:
        insFileSuffix = ''
    #  electrode array name (changes the prefix of the file)
    arrayName = arguments['sourceFilePrefix']
    if arguments['sourceFilePrefix'] is not None:
        blockBaseName = ns5FileName.replace(
            'Block', arguments['sourceFilePrefix'])
    else:
        blockBaseName = copy(ns5FileName)
    if arguments['insFilePrefix'] is not None:
        insBlockBaseName = ns5FileName.replace(
            'Block', arguments['insFilePrefix'])
    else:
        insBlockBaseName = copy(ns5FileName)
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
    asigBlocks = {}
    spikeBlocks = {}
    eventBlocks = {}
    #
    asigReaders = {}
    spikeReaders = {}
    eventReaders = {}
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        chunkIdx = int(chunkIdxStr)
        nameSuffix = sourceFileSuffix + chunkMeta['partNameSuffix']
        nspPath = os.path.join(
            scratchFolder,
            blockBaseName + nameSuffix + '.nix')
        #####
        print('Loading {}'.format(nspPath))
        asigReader, asigBlock = ns5.blockFromPath(
            nspPath, lazy=arguments['lazy'],
            reduceChannelIndexes=True)
        #################
        #  chunked asigBlocks have built-in timestamps
        asigBlock.annotations['chunkTStart'] = 0
        asigBlocks[chunkIdx] = asigBlock
        asigReaders[chunkIdx] = asigReader
        #########################################
        if arguments['spikeSource'] == 'tdc':
            tdcPath = os.path.join(
                scratchFolder,
                'tdc_' + blockBaseName + spikeFileSuffix + chunkMeta['partNameSuffix'],
                'tdc_' + blockBaseName + spikeFileSuffix + chunkMeta['partNameSuffix'] + '.nix'
                )
            print('Loading {}'.format(tdcPath))
            spikeReader, spikeBlock = ns5.blockFromPath(
                tdcPath, lazy=arguments['lazy'],
                reduceChannelIndexes=True)
            # tdc blocks lose their chunking information
            spikeBlock.annotations['chunkTStart'] = chunkMeta['chunkTStart']
            spikeReaders[chunkIdx] = spikeReader
        else:
            spikeBlock = asigBlock
        spikeBlocks[chunkIdx] = spikeBlock
        #####
        eventBlocks[chunkIdx] = asigBlock
        if trackMemory:
            print('Pre-loading chunk {} memory usage: {:.1f} MB'.format(
                idx, prf.memory_usage_psutil()))
    chanQuery = arguments['chanQuery']
    ##############################################################################
    outputBlock = ns5.concatenateBlocks(
        asigBlocks, spikeBlocks, eventBlocks,
        chunkingMetadata, samplingRate, chanQuery,
        arguments['lazy'], trackMemory, arguments['verbose'])
    ##############################################################################
    # close open readers, etc
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        chunkIdx = int(chunkIdxStr)
        if arguments['lazy']:
            asigReaders[chunkIdx].file.close()
            if arguments['spikeSource'] == 'tdc':
                spikeReaders[chunkIdx].file.close()
        del asigBlocks[chunkIdx]
        if arguments['spikeSource'] == 'tdc':
            spikeBlocks[chunkIdx]
        gc.collect()
        if trackMemory:
            print('Deleting blocks from chunk {} memory usage: {:.1f} MB'.format(
                idx, prf.memory_usage_psutil()))
    del spikeBlocks, asigBlocks
    outputFilePath = os.path.join(
        analysisSubFolder,
        ns5FileName + '_analyze.nix'
        )
    writer = NixIO(
        filename=outputFilePath, mode='ow')
    writer.write_block(outputBlock, use_obj_names=True)
    writer.close()
    # pdb.set_trace()
    for asig in outputBlock.filter(objects=AnalogSignal):
        if asig.size > 0:
            dummyOutputAsig = asig
            break
    outputBlockT = pd.Series(dummyOutputAsig.times)
    if len(outputBlock.filter(objects=SpikeTrain)):
        binnedSpikePath = os.path.join(
            analysisSubFolder,
            ns5FileName + '_binarized.nix'
            )
        _ = ns5.calcBinarizedArray(
            outputBlock, samplingRate,
            binnedSpikePath,
            saveToFile=True, matchT=outputBlockT)
    #  ###### load analog inputs
    print('Loading {}'.format(nspPath))
    nspPath = os.path.join(
        scratchFolder,
        blockBaseName + rigFileSuffix + '.nix')
    nspLoadList = {'events': ['seg0_rig_property', 'seg0_rig_value']}
    nspReader, nspBlock = ns5.blockFromPath(
        nspPath, lazy=arguments['lazy'],
        reduceChannelIndexes=True, loadList=nspLoadList)
    nspSeg = nspBlock.segments[0]
    #
    insPath = os.path.join(
        scratchFolder,
        insBlockBaseName + insFileSuffix + '.nix')
    print('Loading {}'.format(insPath))
    insSignalsToLoad = ([
        'seg0_ins_td{}'.format(tdIdx)
        for tdIdx in range(4)] +
        [
        'seg0_ins_acc{}'.format(accIdx)
        for accIdx in ['x', 'y', 'z', 'inertia']
        ])
    insEventsToLoad = [
        'seg0_ins_property',
        'seg0_ins_value'
        ]
    insSpikesToLoad = ['seg0_g0p0#0']
    insLoadList = {
        'asigs': insSignalsToLoad,
        'events': insEventsToLoad,
        'spiketrains': insSpikesToLoad
        }
    insReader, insBlock = ns5.blockFromPath(
        insPath, lazy=arguments['lazy'],
        reduceChannelIndexes=True,
        loadList=insLoadList)
    insSeg = insBlock.segments[0]
    # convert stim updates to time series
    if not arguments['rigOnly']:
        ins_events = [
            ev for ev in insBlock.filter(objects=Event)
            if ev.name in ['seg0_ins_property', 'seg0_ins_value']]
        if len(ins_events):
            expandCols = [
                    'RateInHz', 'therapyStatus',
                    'activeGroup', 'program', 'trialSegment']
            deriveCols = ['amplitudeRound', 'amplitude']
            progAmpNames = rcsa_helpers.progAmpNames
            #
            stimStSer = ns5.eventsToDataFrame(
                ins_events, idxT='t')
            stimStatus = mdt.stimStatusSerialtoLong(
                stimStSer, idxT='t',  namePrefix='seg0_ins_',
                expandCols=expandCols,
                deriveCols=deriveCols, progAmpNames=progAmpNames)
            columnsToBeAdded = ['amplitude', 'program', 'RateInHz'] + progAmpNames
            # pdb.set_trace()
            # stimSt
            infoFromStimStatus = hf.interpolateDF(
                stimStatus, outputBlockT,
                x='t', columns=columnsToBeAdded, kind='previous')
            infoFromStimStatus.set_index('t', inplace=True)
        else:
            infoFromStimStatus = None
    # insBlock = nspBlock
    ######
    #  synchronize INS
    ######
    evList = []
    for key in ['property', 'value']:
        if not arguments['rigOnly']:
            insPropList = insBlock.filter(
                objects=Event,
                name='seg0_ins_' + key
                )
            if len(insPropList):
                insProp = insPropList[0]
            else:
                print(
                    'INS properties not found! analyzing rig events only.')
                arguments['rigOnly'] = True
        else:
            insPropList = []
            insProp = None
        rigPropList = nspBlock.filter(
            objects=Event,
            name='seg0_rig_' + key
            )
        if len(rigPropList):
            rigProp = rigPropList[0]
            if arguments['rigOnly']:
                allProp = rigProp
                allProp.name = 'seg0_' + key
                evList.append(allProp)
            else:
                allProp = insProp.merge(rigProp)
                allProp.name = 'seg0_' + key
                evSortIdx = np.argsort(allProp.times, kind='mergesort')
                allProp = allProp[evSortIdx]
                evList.append(allProp)
        else:
            #  RC's don't have rig_events
            if insProp is not None:
                allProp = insProp
                allProp.name = 'seg0_' + key
                evList.append(allProp)
    if len(evList):
        #  make concatenated event, for viewing
        concatLabels = np.array([
            (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
                elphpdb._convert_value_safe(evList[1].labels[i])) for
            i in range(len(evList[0]))
            ])
        concatEvent = Event(
            name='seg0_concatenated_updates',
            times=allProp.times,
            labels=concatLabels
            )
        concatEvent.merge_annotations(allProp)
        evList.append(concatEvent)
    rigChanQuery = '(chanName.notna())'
    alreadyThereNames = [asi.name for asi in outputBlock.filter(objects=AnalogSignal)]
    if arguments['lazy']:
        rigChanNames = ns5.listChanNames(
            nspBlock, rigChanQuery, objType=AnalogSignalProxy)
        rigChanNames = [rcn for rcn in rigChanNames if rcn not in alreadyThereNames]
        # pdb.set_trace()
        asigList = []
        for asigP in nspSeg.analogsignals:
            if asigP.name in rigChanNames:
                asig = asigP.load()
                asig.channel_index = asigP.channel_index
                asigList.append(asig)
                if trackMemory:
                    print('loading {} from proxy object. memory usage: {:.1f} MB'.format(
                        asigP.name, prf.memory_usage_psutil()))
    else:
        rigChanNames = ns5.listChanNames(
            asigBlock, rigChanQuery, objType=AnalogSignal)
        rigChanNames = [rcn for rcn in rigChanNames if rcn not in alreadyThereNames]
        asigList = [
            asig
            for asig in nspSeg.analogsignals
            if asig.name in rigChanNames
            ]
    for asig in asigList:
        if asig.size > 0:
            dummyRigAsig = asig
            break
    tdDF = ns5.analogSignalsToDataFrame(asigList)
    del asigList
    tdDF.loc[:, 't'] += asigBlock.annotations['chunkTStart']
    origTimeStep = tdDF['t'].iloc[1] - tdDF['t'].iloc[0]
    tdDF.set_index('t', inplace=True)
    # interpolate rig analog signals
    if samplingRate != dummyRigAsig.sampling_rate:
        if samplingRate < dummyRigAsig.sampling_rate:
            lowPassOpts = {
                'low': {
                    'Wn': float(samplingRate),
                    'N': 2,
                    'btype': 'low',
                    'ftype': 'bessel'
                }
            }
            filterCoeffs = hf.makeFilterCoeffsSOS(
                lowPassOpts, float(dummyRigAsig.sampling_rate))
            if trackMemory:
                print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                    prf.memory_usage_psutil()))
            # tdDF.loc[:, tdChanNames] = signal.sosfiltfilt(
            filteredAsigs = signal.sosfiltfilt(
                filterCoeffs, tdDF.to_numpy(),
                axis=0)
            tdDF = pd.DataFrame(
                filteredAsigs,
                index=tdDF.index,
                columns=tdDF.columns)
            if trackMemory:
                print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                    prf.memory_usage_psutil()))
        tdInterp = hf.interpolateDF(
            tdDF, outputBlockT,
            kind='linear', fill_value=(0, 0),
            verbose=arguments['verbose'])
        # free up memory used by full resolution asigs
        del tdDF
    else:
        tdInterp = tdDF
    # pdb.set_trace()
    insAsigList = [
        asig
        for asig in insSeg.analogsignals
        if asig.name in insLoadList['asigs']
        ]
    for asig in insAsigList:
        if asig.size > 0:
            dummyInsAsig = asig
            break
    insDF = ns5.analogSignalsToDataFrame(insAsigList)
    origInsTimeStep = insDF['t'].iloc[1] - insDF['t'].iloc[0]
    insDF.set_index('t', inplace=True)
    # interpolate rig analog signals
    if samplingRate != dummyInsAsig.sampling_rate:
        if samplingRate < dummyInsAsig.sampling_rate:
            lowPassOpts = {
                'low': {
                    'Wn': float(samplingRate),
                    'N': 2,
                    'btype': 'low',
                    'ftype': 'bessel'
                }
            }
            filterCoeffs = hf.makeFilterCoeffsSOS(
                lowPassOpts, float(dummyInsAsig.sampling_rate))
            if trackMemory:
                print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                    prf.memory_usage_psutil()))
            # insDF.loc[:, tdChanNames] = signal.sosfiltfilt(
            filteredAsigs = signal.sosfiltfilt(
                filterCoeffs, insDF.to_numpy(),
                axis=0)
            insDF = pd.DataFrame(
                filteredAsigs,
                index=insDF.index,
                columns=insDF.columns)
            if trackMemory:
                print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                    prf.memory_usage_psutil()))
        insInterp = hf.interpolateDF(
            insDF, outputBlockT,
            kind='linear', fill_value=(0, 0),
            verbose=arguments['verbose'])
        # free up memory used by full resolution asigs
        del insDF
    else:
        insInterp = insDF
    # add analog traces derived from position
    # if 'seg0_position' in tdInterp.columns:
    #     tdInterp.loc[:, 'seg0_position_x'] = ((
    #         np.cos(
    #             tdInterp.loc[:, 'seg0_position'] *
    #             100 * 2 * np.pi / 360))
    #         .to_numpy())
    #     tdInterp.sort_index(axis='columns', inplace=True)
    #     tdInterp.loc[:, 'seg0_position_y'] = ((
    #         np.sin(
    #             tdInterp.loc[:, 'seg0_position'] *
    #             100 * 2 * np.pi / 360))
    #         .to_numpy())
    #     tdInterp.loc[:, 'seg0_velocity_x'] = ((
    #         tdInterp.loc[:, 'seg0_position_y'] *
    #         (-1) *
    #         (tdInterp.loc[:, 'seg0_velocity'] * 3e2))
    #         .to_numpy())
    #     tdInterp.loc[:, 'seg0_velocity_y'] = ((
    #         tdInterp.loc[:, 'seg0_position_x'] *
    #         (tdInterp.loc[:, 'seg0_velocity'] * 3e2))
    #         .to_numpy())
    #     rigChanNames += [
    #         'seg0_position_x', 'seg0_position_y',
    #         'seg0_velocity_x', 'seg0_velocity_y']
    #
    concatList = [tdInterp, insInterp]
    if not arguments['rigOnly']:
        concatList.append(infoFromStimStatus)
    if len(concatList) > 1:
        tdInterp = pd.concat(
            concatList,
            axis=1)
    # smooth by simi fps
    # simiFps = 100
    # smoothWindowStd = int(1 / (origTimeStep * simiFps * 2))
    # if not arguments['rigOnly']:
    #     tdInterp.loc[:, 'RateInHz'] = (
    #         tdInterp.loc[:, 'RateInHz'] *
    #         (tdInterp.loc[:, 'amplitude'].abs() > 0))
    #     for pName in progAmpNames:
    #         if pName in tdInterp.columns:
    #             tdInterp.loc[:, pName.replace('amplitude', 'ACR')] = (
    #                 tdInterp.loc[:, pName] *
    #                 tdInterp.loc[:, 'RateInHz'])
    #             tdInterp.loc[:, pName.replace('amplitude', 'dAmpDt')] = (
    #                 tdInterp.loc[:, pName].diff()
    #                 .rolling(6 * smoothWindowStd, center=True, win_type='gaussian')
    #                 .mean(std=smoothWindowStd).fillna(0) / origTimeStep)
    tdInterp.sort_index(axis='columns', inplace=True)
    tdInterp.columns = [cN.replace('seg0_', '') for cN in tdInterp.columns]
    tdBlockInterp = ns5.dataFrameToAnalogSignals(
        tdInterp,
        idxT=None, useColNames=True,
        dataCol=tdInterp.columns,
        samplingRate=samplingRate)
    #
    # pdb.set_trace()
    tdBlockInterp.segments[0].events = evList
    for ev in evList:
        ev.segment = tdBlockInterp.segments[0]
    # [ev.name for ev in evList]
    ns5.addBlockToNIX(
        tdBlockInterp, neoSegIdx=[0],
        writeSpikes=False, writeEvents=True,
        fileName=ns5FileName + '_analyze',
        folderPath=analysisSubFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
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