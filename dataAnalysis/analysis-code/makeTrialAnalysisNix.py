"""
Usage:
    makeTrialAnalysisNix.py [--trialIdx=trialIdx]

Arguments:
    trialIdx            which trial to analyze
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain)
import neo
from currentExperiment import *
import dataAnalysis.helperFunctions.helper_functions as hf
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os
from importlib import reload
from docopt import docopt


arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    trialBasePath = os.path.join(
        nspFolder,
        ns5FileName + '.nix')
    analysisDataPath = os.path.join(
        insFolder,
        experimentName,
        ns5FileName + '_analyze.nix')
    binnedSpikePath = os.path.join(
        insFolder,
        experimentName,
        ns5FileName + '_binarized.nix')


samplingRate = 1 / rasterOpts['binInterval'] * pq.Hz

nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
nspBlock = nspReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

dataBlock = hf.extractSignalsFromBlock(
    nspBlock)
dataBlock = hf.loadBlockProxyObjects(dataBlock)
allSpikeTrains = dataBlock.filter(objects=SpikeTrain)
for st in allSpikeTrains:
    if 'arrayAnnNames' in st.annotations.keys():
        for key in st.annotations['arrayAnnNames']:
            #  fromRaw, the ann come back as tuple, need to recast
            st.array_annotations.update(
                {key: np.array(st.annotations[key])})
    st.sampling_rate = samplingRate
    if st.waveforms is None:
        st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV

#  tests...
#  [i.unit.channel_index.name for i in insBlockJustSpikes.filter(objects=SpikeTrain)]
#  [i.channel_index.name for i in nspBlock.filter(objects=AnalogSignalProxy)]
#  [i.name for i in nspBlock.filter(objects=AnalogSignalProxy)]
#  [i.channel_index.name for i in dataBlock.filter(objects=AnalogSignal)]
#  [i.channel_index.name for i in dataBlock.filter(objects=AnalogSignal)]

#  dataBlock already has the stim times if we wrote them to that file
#  if not, add them here
dataBlock.segments[0].name = 'analysis seg'
#  merge events
evList = []
for key in ['property', 'value']:
    #  key = 'property'
    insProp = dataBlock.filter(
        objects=Event,
        name='ins_' + key
        )[0]
    rigProp = dataBlock.filter(
        objects=Event,
        name='rig_' + key
        )
    if len(rigProp):
        rigProp = rigProp[0]
        allProp = insProp.merge(rigProp)
        allProp.name = key

        evSortIdx = np.argsort(allProp.times, kind='mergesort')
        allProp = allProp[evSortIdx]
        evList.append(allProp)
    else:
        #  mini RC's don't have rig_ events
        allProp = insProp
        allProp.name = key
        evList.append(insProp)

#  make concatenated event, for viewing
concatLabels = np.array([
    (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
        elphpdb._convert_value_safe(evList[1].labels[i])) for
    i in range(len(evList[0]))
    ])
concatEvent = Event(
    name='concatenated_updates',
    times=allProp.times,
    labels=concatLabels
    )
concatEvent.merge_annotations(allProp)
evList.append(concatEvent)
dataBlock.segments[0].events = evList

testEventMerge = False
if testEventMerge:
    insProp = dataBlock.filter(
        objects=Event,
        name='ins_property'
        )[0]
    allDF = preproc.eventsToDataFrame(
        [insProp], idxT='t'
        )
    allDF[allDF['ins_property'] == 'movement']
    rigDF = preproc.eventsToDataFrame(
        [rigProp], idxT='t'
        )

dataBlock = preproc.purgeNixAnn(dataBlock)
writer = neo.io.NixIO(filename=analysisDataPath)
writer.write_block(dataBlock)
writer.close()

spikeMatBlock = preproc.calcBinarizedArray(
    dataBlock, samplingRate,
    binnedSpikePath, saveToFile=True)

#  save ins time series
tdChanNames = [
    i.name for i in nspBlock.filter(objects=AnalogSignalProxy)
    if 'ins_td' in i.name] + ['position', 'velocityCat']
tdBlock = hf.extractSignalsFromBlock(
    nspBlock, keepSpikes=False, keepSignals=tdChanNames)
tdBlock = hf.loadBlockProxyObjects(tdBlock)

ins_events = [
    i for i in tdBlock.filter(objects=Event)
    if 'ins_' in i.name]
tdDF = preproc.analogSignalsToDataFrame(
    tdBlock.filter(objects=AnalogSignal))
newT = pd.Series(
    spikeMatBlock.filter(objects=AnalogSignal)[0].times.magnitude)
tdInterp = hf.interpolateDF(
    tdDF, newT,
    kind='linear', fill_value=(0, 0),
    x='t', columns=tdChanNames)

expandCols = [
        'RateInHz', 'therapyStatus',
        'activeGroup', 'program', 'trialSegment']
deriveCols = ['amplitudeRound', 'amplitude']
progAmpNames = rcsa_helpers.progAmpNames

stimStSer = preproc.eventsToDataFrame(
    ins_events, idxT='t'
    )
stimStatus = hf.stimStatusSerialtoLong(
    stimStSer, idxT='t', expandCols=expandCols,
    deriveCols=deriveCols, progAmpNames=progAmpNames)
columnsToBeAdded = ['amplitude', 'program']
infoFromStimStatus = hf.interpolateDF(
    stimStatus, tdInterp['t'],
    x='t', columns=columnsToBeAdded, kind='previous')
tdInterp = pd.concat((
    tdInterp,
    infoFromStimStatus.drop(columns='t')),
    axis=1)
tdBlockInterp = preproc.dataFrameToAnalogSignals(
    tdInterp,
    idxT='t', useColNames=True,
    dataCol=tdInterp.drop(columns='t').columns,
    samplingRate=samplingRate)

preproc.addBlockToNIX(
    tdBlockInterp, segIdx=0,
    writeSpikes=False, writeEvents=False,
    fileName=ns5FileName + '_analyze',
    folderPath=os.path.join(
        insFolder,
        experimentName),
    nixBlockIdx=0, nixSegIdx=0,
    )

testSaveability = True
#  pdb.set_trace()
#  for st in dataBlock.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
#  for st in insBlockJustSpikes.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
#  for st in insBlock.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))

