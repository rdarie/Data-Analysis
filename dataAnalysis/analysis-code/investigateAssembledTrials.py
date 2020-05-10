import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from currentExperiment import *
import quantities as pq

dataReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
binnedReader = neo.io.nixio_fr.NixIO(
    filename=experimentBinnedSpikePath)

dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chanNames = ['elec44#0_fr', 'elec91#0_fr']
masterSpikeMats, _ = preproc.loadSpikeMats(
    experimentBinnedSpikePath, rasterOpts,
    chans=chanNames,
    loadAll=True)

for segIdx, segSpikeMat in masterSpikeMats.items():
    plt.plot(segSpikeMat.iloc[:, 1])
plt.show()

binnedBlock = binnedReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for segIdx, dataSeg in enumerate(binnedBlock.segments):
    asigProxyList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if asigP.name in chanNames]
    asigP = asigProxyList[0]
    plt.plot(asigP.load().magnitude)
plt.show()

checkTimes = True
for segIdx in range(dataReader.header['nb_segment'][0]):
    print('t_start = {}'.format(dataReader.get_signal_t_start(0, segIdx)))
trialReader = neo.io.nixio_fr.NixIO(
    filename=analysisDataPath)
for segIdx in range(trialReader.header['nb_segment'][0]):
    print('t_start = {}'.format(trialReader.get_signal_t_start(0, segIdx)))
insReader = neo.io.nixio_fr.NixIO(
    filename=insDataPath)
for segIdx in range(insReader.header['nb_segment'][0]):
    print('t_start = {}'.format(insReader.get_signal_t_start(0, segIdx)))
nspReader = neo.io.nixio_fr.NixIO(
    filename=trialBasePath)
for segIdx in range(nspReader.header['nb_segment'][0]):
    print('t_start = {}'.format(nspReader.get_signal_t_start(0, segIdx)))
tdcReader = neo.io.nixio_fr.NixIO(
    filename=spikePath)
tdcBlock = tdcReader.read_block(
    block_index=0, lazy=True, signal_group_mode='split-all')
for st in tdcBlock.segments[0].filter(objects=SpikeTrainProxy):
    print('{}.t_start={}'.format(st.name, st.t_start))
brmReader = neo.io.BlackrockIO(
    filename=trialBasePath.replace('nix', 'ns5'))
brmBlock = brmReader.read_block(
    block_index=0, lazy=True, signal_group_mode='split-all')
for st in brmBlock.segments[0].filter(objects=SpikeTrainProxy):
    print('{}.t_start={}'.format(st.name, st.t_start))

checkAccessMethods = False
if checkAccessMethods:
    segIdx = 3
    byIndex = reader.get_analogsignal_chunk(
        block_index=0, seg_index=segIdx,
        i_start=1000000, i_stop=1100000,
        channel_ids=chanIdx)
    byName = reader.get_analogsignal_chunk(
        block_index=0, seg_index=segIdx,
        i_start=1000000, i_stop=1100000,
        channel_names=chanNames)
    plt.plot(byName, '', label='byName', lw=5)
    plt.plot(byIndex, label='byIndex')
    plt.legend()
    plt.show()

blockIdx = 0
checkReferences = True
for segIdx, dataSeg in enumerate(dataBlock.segments):
    asigProxyList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if '_fr' in asigP.name]    
    if checkReferences:
        for asigP in asigProxyList:
            da = asigP._rawio.da_list['blocks'][blockIdx]['segments'][segIdx]['data']
            print('segIdx {}, asigP.name {}'.format(
                segIdx, asigP.name))
            print('asigP._global_channel_indexes = {}'.format(
                asigP._global_channel_indexes))
            print('asigP references {}'.format(
                da[asigP._global_channel_indexes[0]]))
            try:
                assert asigP.name in da[asigP._global_channel_indexes[0]].name
            except Exception:
                traceback.print_exc()

checkReferences = True
for segIdx, dataSeg in enumerate(dataBlock.segments):
    stProxyList = [
        stP
        for stP in dataSeg.filter(objects=SpikeTrainProxy)]
    if checkReferences:
        for stP in stProxyList[:2]:
            mts = stP._rawio.unit_list['blocks'][blockIdx]['segments'][segIdx]['spiketrains']
            print('segIdx {}, stP.name {}'.format(
                segIdx, stP.name))
            print('stP._unit_index = {}'.format(
                stP._unit_index))
            print('stP references {}'.format(
                mts[stP._unit_index]))
            try:
                assert stP.name in mts[stP._unit_index].name
            except Exception:
                traceback.print_exc()
