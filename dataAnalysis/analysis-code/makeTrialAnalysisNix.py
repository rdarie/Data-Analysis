from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain)
import neo
from currentExperiment import *
import dataAnalysis.helperFunctions.helper_functions as hf
import numpy as np
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import quantities as pq
nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)

nspBlock = nspReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
dataBlock = hf.extractSignalsFromBlock(
    nspBlock)
dataBlock = hf.loadBlockProxyObjects(dataBlock)
allSpikeTrains = dataBlock.filter(objects=SpikeTrain)
for st in allSpikeTrains:
        if st.waveforms is None:
            st.sampling_rate = 3e4*pq.Hz
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
stimStSer = preproc.eventsToDataFrame(
    ins_events, idxT='t'
    )
tdDF = preproc.analogSignalsToDataFrame(
    tdBlock.filter(objects=AnalogSignal))
newT = np.arange()
testSaveability = True
#  pdb.set_trace()
#  for st in dataBlock.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
#  for st in insBlockJustSpikes.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
#  for st in insBlock.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
dataBlock = preproc.purgeNixAnn(dataBlock)
writer = neo.io.NixIO(filename=analysisDataPath)
writer.write_block(dataBlock)
writer.close()
############################################################
confirmNixAddition = False
if confirmNixAddition:
    for idx, oUnit in enumerate(insBlock.list_units):
        if len(oUnit.spiketrains[0]):
            st = oUnit.spiketrains[0]
            break

    trialBasePath = os.path.join(
        trialFilesFrom['utah']['folderPath'],
        trialFilesFrom['utah']['ns5FileName'])
    loadedReader = neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix')
    loadedBlock = loadedReader.read_block(
        block_index=0,
        lazy=True)
    from neo.io.proxyobjects import SpikeTrainProxy
    lStPrx = loadedBlock.filter(objects=SpikeTrainProxy, name=st.name)[0]
    lSt = lStPrx.load()
    plt.eventplot(st.times, label='original', lw=5)
    plt.eventplot(lSt.times, label='loaded', colors='r')
    plt.legend()
    plt.show()
    
############################################################