"""   09: Assemble binarized array and relevant analogsignals
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --analysisName=analysisName     append a name to the resulting blocks? [default: default]
    --processAsigs                  whether to process the analog signals [default: False]
    --processRasters                whether to process the rasters [default: False]
"""
import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, Epoch, AnalogSignal, SpikeTrain)
from copy import copy
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import quantities as pq
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import fit_grid_point
from sklearn.metrics import mean_squared_error, r2_score
from neo import (
    Unit, AnalogSignal, Event, Epoch,
    Block, Segment, ChannelIndex, SpikeTrain)

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

# Scan ahead through all files and ensure that
# spikeTrains and units are present across all assembled files
masterChanDF = pd.DataFrame([], columns=[
    'index', 'channel_names', 'channel_ids',
    # 'hasUnits', 'hasAsigs'
    ])
masterUnitDF = pd.DataFrame([])

for idx, trialBasePath in enumerate(trialsToAssemble):
    filePath = (
        trialBasePath
        .format(arguments['analysisName'])
        .replace('.nix', '_analyze.nix')
        )
    print('loading {}'.format(filePath))
    dataReader, dataBlock = preproc.blockFromPath(
        filePath, lazy=True, reduceChannelIndexes=True)
    for chIdx in (dataBlock.filter(objects=ChannelIndex)):
        chAlreadyThere = masterChanDF.index == chIdx.name
        if not chAlreadyThere.any():
            # masterChanDF.loc[chIdx.name, 'hasUnits'] = len(chIdx.units) > 0
            # masterChanDF.loc[chIdx.name, 'hasAsigs'] = len(chIdx.analogsignals) > 0
            masterChanDF.loc[chIdx.name, 'index'] = chIdx.index
            masterChanDF.loc[chIdx.name, 'channel_names'] = chIdx.channel_names
            masterChanDF.loc[chIdx.name, 'channel_ids'] = chIdx.channel_ids
            for annName,  annVal in chIdx.annotations.items():
                masterChanDF.loc[chIdx.name, annName] = annVal
        # else:
        #     if len(chIdx.units) > 0:
        #         masterChanDF.loc[chAlreadyThere, 'hasUnits'] = True
        #     if len(chIdx.analogsignals):
        #         masterChanDF.loc[chAlreadyThere, 'hasAsigs'] = True
    for unit in (dataBlock.filter(objects=Unit)):
        uAlreadyThere = masterUnitDF.index == unit.name
        if not uAlreadyThere.any():
            for annName, annVal in unit.annotations.items():
                masterUnitDF.loc[unit.name, annName] = annVal
    dataReader.file.close()

# masterChanDF[masterChanDF['hasUnits']]
for suffix in suffixList:
    print('assembling {}'.format(suffix))
    experimentDataPath = os.path.join(
        scratchFolder, arguments['analysisName'],
        assembledName +
        suffix + '.nix')
    for idx, trialBasePath in enumerate(trialsToAssemble):
        print('loading trial {}'.format(trialBasePath))
        trialDataPath = (
            trialBasePath
            .format(arguments['analysisName'])
            .replace('.nix', suffix + '.nix')
            )
        if idx == 0:
            masterBlock = preproc.loadWithArrayAnn(
                trialDataPath, fromRaw=False)
            masterBlock.name = experimentName + suffix
            if suffix == '_binarized':
                for seg in masterBlock.segments:
                    seg.spiketrains = []
            if applyTimeOffset:
                masterTStart = masterBlock.filter(objects=AnalogSignal)[0].t_start
                oldTStop = masterBlock.filter(objects=AnalogSignal)[0].t_stop
            if suffix == '_analyze':
                for rowIdx, row in masterChanDF.iterrows():
                    matchingCh = masterBlock.filter(objects=ChannelIndex, name=rowIdx)
                    if not len(matchingCh):
                        # create it
                        print('ch {} not found; creating now'.format(rowIdx))
                        # [ch.index for ch in masterBlock.filter(objects=ChannelIndex)]
                        chIdx = ChannelIndex(
                            index=row['index'],
                            channel_ids=row['channel_ids'],
                            channel_names=row['channel_names'],
                            file_origin=dataBlock.channel_indexes[-1].file_origin
                            )
                        masterBlock.channel_indexes.append(chIdx)
                        chIdx.block = masterBlock
                        # TODO: create blank asigs
                anySpikeTrains = masterBlock.filter(objects=SpikeTrain)
                if len(anySpikeTrains):
                    wvfUnits = anySpikeTrains[0].waveforms.units
                    stTimeUnits = anySpikeTrains[0].units
                else:
                    stTimeUnits = pq.s
                    wvfUnits = pq.uV
                for rowIdx, row in masterUnitDF.iterrows():
                    matchingUnit = masterBlock.filter(
                        objects=Unit, name=rowIdx)
                    if not len(matchingUnit):
                        parentChanName = (
                            rowIdx.replace('_stim#0', '')
                            .replace('#0', ''))
                        matchingCh = masterBlock.filter(
                            objects=ChannelIndex, name=parentChanName)
                        parentChIdx = matchingCh[0]
                        newUnit = Unit(name=rowIdx)
                        for annName in row.index:
                            newUnit.annotations[annName] = row[annName]
                        newUnit.channel_index = parentChIdx
                        parentChIdx.units.append(newUnit)
                        for seg in masterBlock.segments:
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
            masterBlock.segments[0].name = 'seg{}_{}'.format(idx, masterBlock.name)
            for objType in typesNeedRenaming:
                for child in masterBlock.filter(objects=objType):
                    childBaseName = preproc.childBaseName(child.name, 'seg')
                    child.name = 'seg{}_{}'.format(idx, childBaseName)
        else:
            dataBlock = preproc.loadWithArrayAnn(
                trialDataPath, fromRaw=False)
            dataBlock.name = masterBlock.name
            if suffix == '_binarized':
                for seg in dataBlock.segments:
                    seg.spiketrains = []
            if applyTimeOffset:
                tStart = dataBlock.filter(objects=AnalogSignal)[0].t_start
                timeOffset = oldTStop - tStart
                dataBlock = hf.timeOffsetBlock(dataBlock, timeOffset, masterTStart)
                #  [i.times for i in dataBlock.filter(objects=SpikeTrain)]
                #  [i.unit.channel_index.name for i in masterBlock.filter(objects=SpikeTrain)]
                tStop = dataBlock.filter(objects=AnalogSignal)[0].t_stop
            if suffix == '_analyze':
                for rowIdx, row in masterChanDF.iterrows():
                    matchingCh = dataBlock.filter(objects=ChannelIndex, name=rowIdx)
                    if not len(matchingCh):
                        # create it
                        print('ch {} not found; creating now'.format(rowIdx))
                        # [ch.index for ch in dataBlock.filter(objects=ChannelIndex)]
                        chIdx = ChannelIndex(
                            index=row['index'],
                            channel_ids=row['channel_ids'],
                            channel_names=row['channel_names'],
                            file_origin=dataBlock.channel_indexes[-1].file_origin
                            )
                        dataBlock.channel_indexes.append(chIdx)
                        chIdx.block = dataBlock
                        # TODO: create blank asigs
                anySpikeTrains = dataBlock.filter(objects=SpikeTrain)
                if len(anySpikeTrains):
                    wvfUnits = anySpikeTrains[0].waveforms.units
                    stTimeUnits = anySpikeTrains[0].units
                else:
                    stTimeUnits = pq.s
                    wvfUnits = pq.uV
                for rowIdx, row in masterUnitDF.iterrows():
                    matchingUnit = dataBlock.filter(
                        objects=Unit, name=rowIdx)
                    if not len(matchingUnit):
                        parentChanName = (
                            rowIdx.replace('_stim#0', '')
                            .replace('#0', ''))
                        matchingCh = dataBlock.filter(
                            objects=ChannelIndex, name=parentChanName)
                        parentChIdx = matchingCh[0]
                        newUnit = Unit(name=rowIdx)
                        for annName in row.index:
                            newUnit.annotations[annName] = row[annName]
                        newUnit.channel_index = parentChIdx
                        parentChIdx.units.append(newUnit)
                        for seg in dataBlock.segments:
                            dummyST = SpikeTrain(
                                times=[], units=stTimeUnits,
                                t_stop=seg.filter(objects=AnalogSignal)[0].t_stop,
                                waveforms=np.array([]).reshape((0, 0, 0)) * wvfUnits,
                                name=seg.name + newUnit.name)
                            dummyST.unit = newUnit
                            dummyST.segment = seg
                            newUnit.spiketrains.append(dummyST)
                            seg.spiketrains.append(dummyST)
            dataBlock.segments[0].name = 'seg{}_{}'.format(idx, dataBlock.name)
            for objType in typesNeedRenaming:
                for child in dataBlock.filter(objects=objType):
                    childBaseName = preproc.childBaseName(child.name, 'seg')
                    child.name = 'seg{}_{}'.format(idx, childBaseName)
            # print([asig.name for asig in dataBlock.filter(objects=AnalogSignal)])
            # print([st.name for st in dataBlock.filter(objects=SpikeTrain)])
            # print([ev.name for ev in dataBlock.filter(objects=Event)])
            # print([chIdx.name for chIdx in dataBlock.filter(objects=ChannelIndex)])
            # #)
            masterBlock.merge(dataBlock)
            if applyTimeOffset:
                oldTStop = tStop
    # #)
    # print([evSeg.events[0].name for evSeg in masterBlock.segments])
    # print([asig.name for asig in masterBlock.filter(objects=AnalogSignal)])
    # print([st.name for st in masterBlock.filter(objects=SpikeTrain)])
    # print([ev.name for ev in masterBlock.filter(objects=Event)])
    # print([chIdx.name for chIdx in masterBlock.filter(objects=ChannelIndex)])
    masterBlock = preproc.purgeNixAnn(masterBlock)
    writer = neo.io.NixIO(filename=experimentDataPath)
    writer.write_block(masterBlock, use_obj_names=True)
    writer.close()
