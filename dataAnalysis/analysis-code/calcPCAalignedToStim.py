import dataAnalysis.ephyviewer.scripts as vis_scripts
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
from collections import Iterable
#  load options
from currentExperiment import *
from joblib import dump, load
import quantities as pq
#  all experimental days
dataReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in dataBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])

chansToTrigger = np.unique(
    [
        i.name
        for i in dataBlock.filter(objects=AnalogSignalProxy)])
eventName = 'alignTimes'
blockIdx = 0
windowSize = [i * pq.s for i in rasterOpts['windowSize']]
#  def eventTriggeredSpikes():
#  dataBlock
#  
#  chansToTrigger = ['PC1', 'PC2']
#  
#  
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
#  make channels and units for triggered time series
for chanName in chansToTrigger:
    chanIdx = ChannelIndex(name=chanName + '#0', index=[0])
    chanIdx.annotate(nix_name=chanIdx.name)
    thisUnit = Unit(name=chanIdx.name)
    thisUnit.annotate(nix_name=chanIdx.name)
    chanIdx.units.append(thisUnit)
    thisUnit.channel_index = chanIdx
    masterBlock.channel_indexes.append(chanIdx)

for segIdx, dataSeg in enumerate(dataBlock.segments):
    newSeg = Segment(name=dataSeg.annotations['neo_name'])
    masterBlock.segments.append(newSeg)

    alignEvents = [
        i.load()
        for i in dataSeg.filter(
            objects=EventProxy, name=eventName)]
    alignEvents = preproc.loadContainerArrayAnn(trainList=alignEvents)[0]
    
    '''
    headLen = 3
    alignEventsAnn = pd.concat(
        [
            pd.DataFrame(alignEvents[:headLen].array_annotations),
            pd.Series(alignEvents[:headLen].times, name='t')
            ], axis=1)
    originalEvents = dataSeg.filter(
        objects=EventProxy, name='concatenated_updates')[0].load()
    tMask = (
        (originalEvents >= (alignEventsAnn['t'].iloc[0] - 1)) &
        (originalEvents <= (alignEventsAnn['t'].iloc[-1] + 1)))
    originalEvents[tMask]
    alignEventsAnn.transpose()
    '''

    for chanName in chansToTrigger:
        asigP = dataSeg.filter(objects=AnalogSignalProxy, name=chanName)[0]
        checkReferences = True
        if checkReferences:
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
        rawWaveforms = [
            asigP.load(time_slice=(t + windowSize[0], t + windowSize[1]))
            for t in alignEvents]

        samplingRate = asigP.sampling_rate
        waveformUnits = rawWaveforms[0].units
        #  fix length
        minLen = min([rW.shape[0] for rW in rawWaveforms])
        rawWaveforms = [rW[:minLen] for rW in rawWaveforms]

        spikeWaveforms = (
            np.hstack([rW.magnitude for rW in rawWaveforms])
            .transpose()[:, np.newaxis, :] * waveformUnits
            )

        thisUnit = masterBlock.filter(objects=Unit, name=chanName + '#0')[0]
        stAnn = {
            k: v
            for k, v in alignEvents.annotations.items()
            if k not in ['nix_name', 'neo_name']
            }
        st = SpikeTrain(
            name='seg{}_{}'.format(int(segIdx), thisUnit.name),
            times=alignEvents.times,
            waveforms=spikeWaveforms,
            t_start=asigP.t_start, t_stop=asigP.t_stop,
            left_sweep=windowSize[0] * (-1),
            sampling_rate=samplingRate,
            array_annotations=alignEvents.array_annotations,
            **stAnn
            )
        st.annotate(nix_name=st.name)
        thisUnit.spiketrains.append(st)
        newSeg.spiketrains.append(st)
        st.unit = thisUnit

dataReader.file.close()

masterBlock = preproc.purgeNixAnn(masterBlock)
writer = neo.io.NixIO(filename=experimentDataPath)
writer.write_block(masterBlock, use_obj_names=True)
writer.close()