"""   09: Assemble binarized array and relevant analogsignals
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx              which trial to analyze [default: 1]
    --exp=exp                        which experimental day to analyze
    --analysisName=analysisName      append a name to the resulting blocks? [default: default]
    --processAsigs                   whether to process the analog signals [default: False]
    --processRasters                 whether to process the rasters [default: False]
    --commitResults                  whether to additionally save to the processed data folder [default: False]
"""
# import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb, traceback, shutil
# from importlib import reload
import neo
from neo import (
    Unit, AnalogSignal, Event, Epoch,
    Block, Segment, ChannelIndex, SpikeTrain)
from copy import copy
# import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions_new as hf
# import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
# import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import quantities as pq
# from matplotlib import pyplot as plt
# import seaborn as sns

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

applyTimeOffset = False
suffixList = []
if arguments['processAsigs']:
    suffixList.append('_analyze')
if arguments['processRasters']:
    suffixList.append('_binarized')
suffixList.append('_fr')

for suffix in suffixList:
    print('assembling {}'.format(suffix))
    experimentDataPath = os.path.join(
        scratchFolder, arguments['analysisName'],
        assembledName +
        suffix + '.nix')
    # Scan ahead through all files and ensure that
    # spikeTrains and units are present across all assembled files
    masterChanDF = pd.DataFrame([], columns=[
        'index', 'channel_names', 'channel_ids',
        'hasUnits', 'hasAsigs'
        ])
    masterUnitDF = pd.DataFrame([], columns=['parentChanName'])
    blocksCache = {}
    for idx, trialBasePath in enumerate(trialsToAssemble):
        trialDataPath = (
            trialBasePath
            .format(arguments['analysisName'])
            .replace('.nix', '{}.nix'.format(suffix))
            )
        # dataReader, dataBlock = preproc.blockFromPath(
        #     trialDataPath, lazy=True, reduceChannelIndexes=True)
        dataBlock = preproc.loadWithArrayAnn(
            trialDataPath, fromRaw=False,
            reduceChannelIndexes=True)
        # [cI.name for cI in dataBlock.channel_indexes]
        pdb.set_trace()
        #
        blocksCache[trialDataPath] = dataBlock
        if idx == 0:
            masterDataPath = trialDataPath
        for chIdx in dataBlock.filter(objects=ChannelIndex):
            chAlreadyThere = masterChanDF.index == chIdx.name
            if not chAlreadyThere.any():
                masterChanDF.loc[chIdx.name, 'hasUnits'] = len(chIdx.units) > 0
                masterChanDF.loc[chIdx.name, 'hasAsigs'] = len(chIdx.analogsignals) > 0
                try:
                    chIdxNames = chIdx.channel_names
                    chIdxIDS = chIdx.channel_ids
                    print('chIdx index = {}'.format(chIdx.index))
                    if not len(chIdxIDS):
                        chIdxIDS = [int(chIdx.index)]
                    if not len(chIdxNames):
                        chIdxNames = [chIdx.name]
                    masterChanDF.loc[chIdx.name, 'index'] = int(chIdx.index)
                    masterChanDF.loc[chIdx.name, 'channel_names'] = chIdxNames
                    masterChanDF.loc[chIdx.name, 'channel_ids'] = chIdxIDS
                except Exception:
                    traceback.print_exc()
                for annName,  annVal in chIdx.annotations.items():
                    masterChanDF.loc[chIdx.name, annName] = annVal
            else:
                if len(chIdx.units) > 0:
                    masterChanDF.loc[chAlreadyThere, 'hasUnits'] = True
                if len(chIdx.analogsignals):
                    masterChanDF.loc[chAlreadyThere, 'hasAsigs'] = True
        for unit in (dataBlock.filter(objects=Unit)):
            uAlreadyThere = masterUnitDF.index == unit.name
            if not uAlreadyThere.any():
                for annName, annVal in unit.annotations.items():
                    masterUnitDF.loc[unit.name, annName] = annVal
                unitParentChanName = unit.channel_index.name
                masterUnitDF.loc[unit.name, 'parentChanName'] = unitParentChanName
                # chAlreadyThere = masterChanDF.index == unitParentChanName
        # dataReader.file.close()
    # now merge the blocks
    for idx, trialBasePath in enumerate(trialsToAssemble):
        trialDataPath = (
            trialBasePath
            .format(arguments['analysisName'])
            .replace('.nix', '{}.nix'.format(suffix))
            )
        print('loading trial {}'.format(trialDataPath))
        if idx == 0:
            blocksCache[trialDataPath].name = experimentName + suffix
            if applyTimeOffset:
                masterTStart = blocksCache[trialDataPath].filter(objects=AnalogSignal)[0].t_start
                oldTStop = blocksCache[trialDataPath].filter(objects=AnalogSignal)[0].t_stop
        else:
            blocksCache[trialDataPath].name = blocksCache[masterDataPath].name
            if applyTimeOffset:
                tStart = blocksCache[trialDataPath].filter(objects=AnalogSignal)[0].t_start
                timeOffset = oldTStop - tStart
                blocksCache[trialDataPath] = hf.timeOffsetBlock(
                    blocksCache[trialDataPath], timeOffset, masterTStart)
                #  [i.times for i in dataBlock.filter(objects=SpikeTrain)]
                #  [i.unit.channel_index.name for i in masterBlock.filter(objects=SpikeTrain)]
                tStop = dataBlock.filter(objects=AnalogSignal)[0].t_stop
        # if suffix == '_binarized':
        #     for seg in blocksCache[trialDataPath].segments:
        #         seg.spiketrains = []
        for rowIdx, row in masterChanDF.iterrows():
            matchingCh = blocksCache[trialDataPath].filter(
                objects=ChannelIndex, name=rowIdx)
            if not len(matchingCh):
                '''
                    # [ch.index for ch in blocksCache[trialDataPath].filter(objects=ChannelIndex)]
                    # if row['index'] is None:
                    #     pdb.set_trace()
                    #     chIdx = ChannelIndex(
                    #         name=rowIdx,
                    #         index=np.asarray([0]),
                    #         channel_ids=np.asarray([0]),
                    #         channel_names=np.asarray([rowIdx]),
                    #         file_origin=blocksCache[trialDataPath].channel_indexes[-1].file_origin
                    #         )
                    # else:
                '''
                # create it
                print('ch {} not found; creating now'.format(rowIdx))
                chIdx = ChannelIndex(
                    name=rowIdx,
                    index=np.asarray([row['index']]).flatten(),
                    channel_ids=np.asarray([row['channel_ids']]).flatten(),
                    channel_names=np.asarray([row['channel_names']]).flatten(),
                    file_origin=blocksCache[trialDataPath].channel_indexes[-1].file_origin
                    )
                for aN in row.drop(['index', 'channel_names', 'channel_ids']).index:
                    chIdx.annotations[aN] = row[aN]
                blocksCache[trialDataPath].channel_indexes.append(chIdx)
                chIdx.block = blocksCache[trialDataPath]
                # create blank asigs
                if row['hasAsigs']:
                    dummyAsig = blocksCache[trialDataPath].filter(objects=AnalogSignal)[0].copy()
                    dummyAsig.name = 'seg0_' + chIdx.name
                    dummyAsig.annotations['neo_name'] = dummyAsig.name
                    dummyAsig.magnitude[:] = 0
                    dummyAsig.channel_index = chIdx
                    chIdx.analogsignals.append(dummyAsig)
                    blocksCache[trialDataPath].segments[0].analogsignals.append(dummyAsig)
                    dummyAsig.segment = blocksCache[trialDataPath].segments[0]
                    # pdb.set_trace()
        anySpikeTrains = blocksCache[trialDataPath].filter(objects=SpikeTrain)
        if len(anySpikeTrains):
            wvfUnits = anySpikeTrains[0].waveforms.units
            stTimeUnits = anySpikeTrains[0].units
        else:
            stTimeUnits = pq.s
            wvfUnits = pq.uV
        for rowIdx, row in masterUnitDF.iterrows():
            matchingUnit = blocksCache[trialDataPath].filter(
                objects=Unit, name=rowIdx)
            if not len(matchingUnit):
                parentChanName = row['parentChanName']
                # parentChanName = rowIdx
                # if parentChanName.endswith('_stim#0'):
                #     parentChanName.replace('_stim#0', '')
                # if parentChanName.endswith('#0'):
                #     parentChanName.replace('#0', '')
                matchingCh = blocksCache[trialDataPath].filter(
                    objects=ChannelIndex, name=parentChanName)
                '''
                    if not len(matchingCh):
                        masterListEntry = masterChanDF.loc[parentChanName, :]
                        parentChIdx = ChannelIndex(
                            name=parentChanName,
                            index=masterListEntry['index'],
                            channel_ids=masterListEntry['channel_ids'],
                            channel_names=masterListEntry['channel_names'],
                            file_origin=blocksCache[trialDataPath].channel_indexes[-1].file_origin
                            )
                        blocksCache[trialDataPath].channel_indexes.append(parentChIdx)
                        parentChIdx.block = blocksCache[trialDataPath]
                    else:
                        parentChIdx = matchingCh[0]
                '''
                parentChIdx = matchingCh[0]
                print('unit {} not found; creating now'.format(rowIdx))
                newUnit = Unit(name=rowIdx)
                for annName in row.index:
                    newUnit.annotations[annName] = row[annName]
                newUnit.channel_index = parentChIdx
                parentChIdx.units.append(newUnit)
                for seg in blocksCache[trialDataPath].segments:
                    dummyST = SpikeTrain(
                        times=[], units=stTimeUnits,
                        t_stop=seg.filter(objects=AnalogSignal)[0].t_stop,
                        waveforms=np.array([]).reshape((0, 0, 0)) * wvfUnits,
                        name=seg.name + newUnit.name)
                    dummyST.unit = newUnit
                    dummyST.segment = seg
                    newUnit.spiketrains.append(dummyST)
                    seg.spiketrains.append(dummyST)
        typesNeedRenaming = [SpikeTrain, AnalogSignal, Event]
        blocksCache[trialDataPath].segments[0].name = 'seg{}_{}'.format(
            idx, blocksCache[trialDataPath].name)
        for objType in typesNeedRenaming:
            listOfChildren = blocksCache[trialDataPath].filter(objects=objType)
            print('{}\n{} objects of type {}'.format(
                trialDataPath, len(listOfChildren), objType
            ))
            for child in listOfChildren:
                childBaseName = preproc.childBaseName(child.name, 'seg')
                child.name = 'seg{}_{}'.format(idx, childBaseName)
        blocksCache[trialDataPath].create_relationship()
        blocksCache[trialDataPath] = preproc.purgeNixAnn(blocksCache[trialDataPath])
        ########
        sanityCheck = False
        if sanityCheck and idx == 2:
            doublePath = trialDataPath.replace(suffix, suffix + '_backup')
            if os.path.exists(doublePath):
                os.remove(doublePath)
            print('writing {} ...'.format(doublePath))
            for idx, chIdx in enumerate(blocksCache[trialDataPath].channel_indexes):
                print('{}: {}, chan_id = {}'.format(
                    chIdx.name, chIdx.index, chIdx.channel_ids))
            writer = neo.io.NixIO(filename=doublePath)
            writer.write_block(blocksCache[trialDataPath], use_obj_names=True)
            writer.close()
        ############
        if idx > 0:
            blocksCache[masterDataPath].merge(blocksCache[trialDataPath])
            if applyTimeOffset:
                oldTStop = tStop
    '''
        print([evSeg.events[0].name for evSeg in masterBlock.segments])
        print([asig.name for asig in masterBlock.filter(objects=AnalogSignal)])
        print([st.name for st in masterBlock.filter(objects=SpikeTrain)])
        print([ev.name for ev in masterBlock.filter(objects=Event)])
        print([chIdx.name for chIdx in blocksCache[trialDataPath].filter(objects=ChannelIndex)])
        print([un.name for un in masterBlock.filter(objects=Unit)])
    '''
    # blocksCache[masterDataPath].create_relationship()
    if os.path.exists(experimentDataPath):
        os.remove(experimentDataPath)
    writer = neo.io.NixIO(filename=experimentDataPath)
    print('writing {} ...'.format(experimentDataPath))
    writer.write_block(blocksCache[masterDataPath], use_obj_names=True)
    writer.close()
    if arguments['commitResults']:
        analysisProcessedSubFolder = os.path.join(
            processedFolder, arguments['analysisName']
            )
        if not os.path.exists(analysisProcessedSubFolder):
            os.makedirs(analysisProcessedSubFolder, exist_ok=True)
        for suffix in suffixList:
            experimentDataPath = os.path.join(
                scratchFolder, arguments['analysisName'],
                assembledName +
                suffix + '.nix')
            processedOutPath = os.path.join(
                analysisProcessedSubFolder, arguments['analysisName'],
                assembledName +
                suffix + '.nix')
            print('copying from:\n{}\ninto\n{}'.format(experimentDataPath, processedOutPath))
            shutil.copyfile(experimentDataPath, processedOutPath)
