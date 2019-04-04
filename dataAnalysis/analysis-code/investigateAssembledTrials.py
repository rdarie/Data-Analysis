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

masterSpikeMats, _ = preproc.loadSpikeMats(
    experimentBinnedSpikePath, rasterOpts,
    chans=['elec44#0', 'elec91#0'],
    loadAll=True)

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
    plt.plot(byName, '--', label='byName', lw=5)
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
