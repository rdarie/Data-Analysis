"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcTrialAnalysisNix.py [options]

Options:
    --trialIdx=trialIdx               which trial to analyze
    --exp=exp                         which experimental day to analyze
    --analysisName=analysisName       append a name to the resulting blocks? [default: default]
    --chanQuery=chanQuery             how to restrict channels if not providing a list? [default: fr]
    --samplingRate=samplingRate       subsample the result??
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
from namedQueries import namedQueries
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb
from importlib import reload


#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
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
    samplingRate = float(1 / rasterOpts['binInterval']) * pq.Hz
#pdb.set_trace()
trialBasePath = os.path.join(
    scratchFolder,
    ns5FileName + '.nix'
)
nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
nspBlock = preproc.readBlockFixNames(nspReader, block_index=0)

spikesBlock = hf.extractSignalsFromBlock(
    nspBlock, keepSpikes=True)
spikesBlock = hf.loadBlockProxyObjects(spikesBlock)
#  save ins time series
tdChanNames = ns5.listChanNames(nspBlock, arguments['chanQuery'], objType=AnalogSignalProxy)
if not (miniRCTrial or RCTrial):
    tdChanNames += ['seg0_position', 'seg0_velocityCat']

allSpikeTrains = [
    i
    for i in spikesBlock.filter(objects=SpikeTrain)
    if '#' in i.name]
if len(allSpikeTrains):
    for segIdx, dataSeg in enumerate(spikesBlock.segments):
        spikeList = dataSeg.filter(objects=SpikeTrain)
        spikeList = preproc.loadContainerArrayAnn(trainList=spikeList)
#  merge events
evList = []
for key in ['property', 'value']:
    #  key = 'property'
    insProp = spikesBlock.filter(
        objects=Event,
        name='seg0_ins_' + key
        )[0]
    rigProp = spikesBlock.filter(
        objects=Event,
        name='seg0_rig_' + key
        )
    if len(rigProp):
        rigProp = rigProp[0]
        allProp = insProp.merge(rigProp)
        allProp.name = 'seg0_' + key

        evSortIdx = np.argsort(allProp.times, kind='mergesort')
        allProp = allProp[evSortIdx]
        evList.append(allProp)
    else:
        #  mini RC's don't have rig_events
        allProp = insProp
        allProp.name = 'seg0_' + key
        evList.append(insProp)
#  make concatenated event, for viewing
concatLabels = np.array([
    (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
        elphpdb._convert_value_safe(evList[1].labels[i])) for
    i in range(len(evList[0]))
    ])
concatEvent = Event(
    name='seg0_' + 'concatenated_updates',
    times=allProp.times,
    labels=concatLabels
    )
concatEvent.merge_annotations(allProp)
evList.append(concatEvent)
spikesBlock.segments[0].events = evList
for ev in evList:
    ev.segment = spikesBlock.segments[0]
# print([asig.name for asig in spikesBlock.filter(objects=AnalogSignal)])
# print([st.name for st in spikesBlock.filter(objects=SpikeTrain)])
# print([ev.name for ev in spikesBlock.filter(objects=Event)])
spikesBlock = preproc.purgeNixAnn(spikesBlock)
writer = neo.io.NixIO(
    filename=analysisDataPath.format(arguments['analysisName']))
writer.write_block(spikesBlock, use_obj_names=True)
writer.close()
#
if len(allSpikeTrains):
    spikeMatBlock = preproc.calcBinarizedArray(
        spikesBlock, samplingRate,
        binnedSpikePath.format(arguments['analysisName']),
        saveToFile=True)
    newT = pd.Series(
        spikeMatBlock.filter(objects=AnalogSignal)[0].times.magnitude)
else:
    dummyT = nspBlock.filter(objects=AnalogSignalProxy)[0]
    newT = pd.Series(
        np.arange(
            dummyT.t_start, dummyT.t_stop + 1/samplingRate, 1/samplingRate))
#
tdBlock = hf.extractSignalsFromBlock(
    nspBlock, keepSpikes=False, keepSignals=tdChanNames)
tdBlock = hf.loadBlockProxyObjects(tdBlock)

ins_events = [
    i for i in tdBlock.filter(objects=Event)
    if 'ins_' in i.name]
tdDF = preproc.analogSignalsToDataFrame(
    tdBlock.filter(objects=AnalogSignal))
if samplingRate != tdBlock.filter(objects=AnalogSignal)[0].sampling_rate:
    tdInterp = hf.interpolateDF(
        tdDF, newT,
        kind='linear', fill_value=(0, 0),
        x='t', columns=tdChanNames)
else:
    tdInterp = tdDF
expandCols = [
        'RateInHz', 'therapyStatus',
        'activeGroup', 'program', 'trialSegment']
deriveCols = ['amplitudeRound', 'amplitude']
progAmpNames = rcsa_helpers.progAmpNames
#
stimStSer = preproc.eventsToDataFrame(
    ins_events, idxT='t'
    )
stimStatus = hf.stimStatusSerialtoLong(
    stimStSer, idxT='t',  namePrefix='seg0_ins_',
    expandCols=expandCols,
    deriveCols=deriveCols, progAmpNames=progAmpNames)
columnsToBeAdded = ['amplitude', 'program']
infoFromStimStatus = hf.interpolateDF(
    stimStatus, tdInterp['t'],
    x='t', columns=columnsToBeAdded, kind='previous')
infoFromStimStatus.rename(
    columns={
        'amplitude': 'seg0_amplitude',
        'program': 'seg0_program'
    }, inplace=True)
tdInterp = pd.concat((
    tdInterp,
    infoFromStimStatus.drop(columns='t')),
    axis=1)
tdBlockInterp = preproc.dataFrameToAnalogSignals(
    tdInterp,
    idxT='t', useColNames=True,
    dataCol=tdInterp.drop(columns='t').columns,
    samplingRate=samplingRate)
# 
# print([asig.name for asig in tdBlockInterp.filter(objects=AnalogSignal)])
# print([st.name for st in tdBlockInterp.filter(objects=SpikeTrain)])
# print([ev.name for ev in tdBlockInterp.filter(objects=Event)])
# print([chIdx.name for chIdx in tdBlockInterp.filter(objects=ChannelIndex)])
typesNeedRenaming = [ChannelIndex, AnalogSignal]
for objType in typesNeedRenaming:
    for child in tdBlockInterp.filter(objects=objType):
        child.name = preproc.childBaseName(child.name, 'seg')
# pdb.set_trace()

preproc.addBlockToNIX(
    tdBlockInterp, neoSegIdx=[0],
    writeSpikes=False, writeEvents=False,
    purgeNixNames=False,
    fileName=ns5FileName + '_analyze',
    folderPath=analysisSubFolder,
    nixBlockIdx=0, nixSegIdx=[0],
    )
