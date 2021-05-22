"""   09: Assemble binarized array and relevant analogsignals
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                        which trial to analyze [default: 1]
    --exp=exp                                  which experimental day to analyze
    --processAll                               process entire experimental day? [default: False]
    --analysisName=analysisName                append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName          append a name to the resulting blocks? [default: default]
    --window=window                            process with short window? [default: long]
    --lazy                                     load from raw, or regular? [default: False]
    --inputBlockSuffix=inputBlockSuffix        which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix        which trig_ block to pull [default: Block]
    --commitResults                            whether to additionally save to the processed data folder [default: False]
"""
# import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb, traceback, shutil
# from importlib import reload
import neo
from neo import (
    Unit, AnalogSignal, Event, Epoch,
    Block, Segment, ChannelIndex, SpikeTrain)
from copy import copy
import dataAnalysis.helperFunctions.helper_functions_new as hf
import numpy as np
import pandas as pd
import quantities as pq
import dataAnalysis.helperFunctions.profiling as prf
import gc
import dataAnalysis.preproc.ns5 as ns5
# from matplotlib import pyplot as plt
# import seaborn as sns

#  load options
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import sys
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

trackMemory = True

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)

experimentsToAssemble = expOpts['experimentsToAssemble']
trialsToAssemble = []
for key in sorted(experimentsToAssemble.keys()):
    val = experimentsToAssemble[key]
    for tIdx in val:
        trialsToAssemble.append(
            os.path.join(
                scratchPath, key, '{}', '{}',
                '{{}}{:0>3}{{}}_{{}}.nix'.format(int(tIdx))
            )
        )
outputFileName = blockBaseName + inputBlockSuffix + '_{}.nix'.format(arguments['window'])
outputPath = os.path.join(
    alignSubFolder, outputFileName)
# Scan ahead through all files and ensure that
# spikeTrains and units are present across all assembled files
channelIndexCache = {}
unitCache = {}
asigCache = {}
asigAnnCache = {}
spiketrainCache = {}
eventCache = {}
segIdx = 0

stTimeUnits = pq.s
wvfUnits = pq.uV
for idx, trialBasePath in enumerate(trialsToAssemble):
    gc.collect()
    segIdx = idx
    trialDataPath = (
        trialBasePath
        .format(
            arguments['analysisName'], arguments['alignFolderName'],
            blockBaseName, inputBlockSuffix, arguments['window'])
        )
    if os.path.exists(trialDataPath):
        print('Loading {}'.format(trialDataPath))
        dataReader, dataBlock = ns5.blockFromPath(
            trialDataPath, lazy=arguments['lazy'],
            reduceChannelIndexes=True)
        if arguments['lazy']:
            ns5.loadSpikeTrainList(dataBlock, replaceInParents=True)
            ns5.loadEventList(dataBlock, replaceInParents=True)
            ns5.loadAsigList(dataBlock, replaceInParents=True)
    else:
        raise (Exception('\n{}\nDoes not exist!\n'.format(trialDataPath)))
    # pdb.set_trace()
    dataSeg = dataBlock.segments[0]
    if idx == 0:
        outputBlock = Block(
            name=dataBlock.name,
            file_origin=dataBlock.file_origin,
            file_datetime=dataBlock.file_datetime,
            rec_datetime=dataBlock.rec_datetime,
            **dataBlock.annotations
        )
        outputBlock.segments = []
    newSeg = Segment(
        index=segIdx, name='seg{}_{}'.format(segIdx, outputBlock.name),
        description=dataSeg.description,
        file_origin=dataSeg.file_origin,
        file_datetime=dataSeg.file_datetime,
        rec_datetime=dataSeg.rec_datetime,
        **dataSeg.annotations
    )
    outputBlock.segments.append(newSeg)
    newSeg.block = outputBlock
    #
    for chIdx in dataBlock.filter(objects=ChannelIndex):
        print('Examining ChannelIndex {}'.format(chIdx.name))
        chAlreadyThere = (chIdx.name in channelIndexCache.keys())
        if not chAlreadyThere:
            print('        Creating ChannelIndex {}'.format(chIdx.name))
            newChIdx = copy(chIdx)
            #
            if not len(newChIdx.channel_ids):
                newChIdx.channel_ids = [int(chIdx.index)]
            chIdxNames = chIdx.channel_names
            if not len(newChIdx.channel_names):
                newChIdx.channel_names = [chIdx.name]
            #
            newChIdx.analogsignals = []
            newChIdx.units = []
            #
            channelIndexCache[chIdx.name] = newChIdx
            newChIdx.block = outputBlock
            outputBlock.channel_indexes.append(newChIdx)
            asigCache[chIdx.name] = {}
        if len(chIdx.analogsignals) > 0:
            assert len(chIdx.analogsignals) == 1
            asig = chIdx.analogsignals[0]
            if asig.size > 0:
                asigBaseName = ns5.childBaseName(asig.name, 'seg')
                asig.name = 'seg{}_{}'.format(segIdx, asigBaseName)
                print('Renamed asig {}'.format(asig.name))
                asigCache[chIdx.name][segIdx] = asig
    for unit in (dataBlock.filter(objects=Unit)):
        print('Examining Unit {}'.format(unit.name))
        uAlreadyThere = (unit.name in unitCache.keys())
        if not uAlreadyThere:
            print('        Creating Unit {}'.format(unit.name))
            newUnit = copy(unit)
            newUnit.spiketrains = []
            pcn = unit.channel_index.name
            assert pcn in channelIndexCache
            channelIndexCache[pcn].units.append(newUnit)
            newUnit.channel_index = channelIndexCache[pcn]
            unitCache[unit.name] = newUnit
            spiketrainCache[unit.name] = {}
        if len(unit.spiketrains) > 0:
            try:
                assert len(unit.spiketrains) == 1
            except Exception:
                print('\n\n')
                print('        Error while processing unit {}'.format(unit.name))
                traceback.print_exc()
                print('\n\n')
            st = unit.spiketrains[0]
            if len(st.times):
                wvfUnits = st.waveforms.units
                stTimeUnits = st.units
                if 'tStop' not in newSeg.annotations:
                    newSeg.annotations['tStop'] = st.t_stop
                else:
                    newSeg.annotations['tStop'] = max(newSeg.annotations['tStop'], st.t_stop)
                stBaseName = ns5.childBaseName(st.name, 'seg')
                st.name = 'seg{}_{}'.format(segIdx, stBaseName)
                spiketrainCache[unit.name][segIdx] = st
                print('Renamed {}'.format(st.name))
    #
    newSeg.events = []
    for ev in dataSeg.events:
        if len(ev.times):
            evBaseName = ns5.childBaseName(ev.name, 'seg')
            ev.name = 'seg{}_{}'.format(segIdx, evBaseName)
            eventCache[segIdx] = ev
            print('Assigning Event {}'.format(ev.name))
            newSeg.events.append(ev)
            ev.segment = dataSeg
#
for segIdx, outputSeg in enumerate(outputBlock.segments):
    for chIdx in outputBlock.filter(objects=ChannelIndex):
        if segIdx in asigCache[chIdx.name]:
            asig = asigCache[chIdx.name][segIdx]
            asig.channel_index = chIdx
            chIdx.analogsignals.append(asig)
            outputSeg.analogsignals.append(asig)
            asig.segment = outputSeg
        elif len(asigCache[chIdx.name]):
            dummyAsig = None
            for _, asigDict in asigCache.items():
                if len(asigDict):
                    for _, asig in asigDict.items():
                        if asig.size > 0:
                            dummyAsig = asig.copy()
                            break
                if dummyAsig is not None:
                    break
            dummyAsig.name = 'seg{}_'.format(segIdx) + chIdx.name
            dummyAsig.annotations['neo_name'] = dummyAsig.name
            dummyAsig.magnitude[:] = 0
            dummyAsig.channel_index = chIdx
            chIdx.analogsignals.append(dummyAsig)
            outputSeg.analogsignals.append(dummyAsig)
            dummyAsig.segment = outputSeg
    #
    for unit in outputBlock.filter(objects=Unit):
        if segIdx in spiketrainCache[unit.name]:
            st = spiketrainCache[unit.name][segIdx]
            unit.spiketrains.append(st)
            st.unit = unit
            st.segment = outputSeg
            outputSeg.spiketrains.append(st)
        elif len(spiketrainCache[unit.name]):
            dummySt = SpikeTrain(
                times=[], units=stTimeUnits,
                t_stop=outputSeg.annotations['tStop'],
                waveforms=np.array([]).reshape((0, 0, 0)) * wvfUnits,
                name='seg{}_'.format(segIdx) + newUnit.name)
            dummySt.annotations['neo_name'] = dummySt.name
            #
            dummySt.unit = unit
            unit.spiketrains.append(dummySt)
            outputSeg.spiketrains.append(dummySt)
            dummySt.segment = outputSeg

outputBlock = ns5.purgeNixAnn(outputBlock)
createRelationship = True
if createRelationship:
    outputBlock.create_relationship()
if os.path.exists(outputPath):
    os.remove(outputPath)
writer = neo.io.NixIO(filename=outputPath, mode='ow')
print('writing {} ...'.format(outputPath))
writer.write_block(outputBlock, use_obj_names=True)
writer.close()
if arguments['commitResults']:
    processedSubFolder = os.path.join(
        processedFolder, arguments['analysisName'], arguments['alignFolderName'])
    processedOutputPath = os.path.join(processedSubFolder, outputFileName)
    if not os.path.exists(processedSubFolder):
        os.makedirs(processedSubFolder, exist_ok=True)
    print('copying from:\n{}\ninto\n{}'.format(outputPath, processedOutputPath))
    shutil.copyfile(outputPath, processedOutputPath)
