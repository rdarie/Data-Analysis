import dataAnalysis.plotting.aligned_signal_plots as asp
import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#  load options
from exp201901271000 import *
from joblib import dump, load
import quantities as pq
from statsmodels.stats.weightstats import ttest_ind


def getConditionAverages(
        dataBlock, unitNames, dataQueryTemplate,
        collapseSizes=True, verbose=False):
    allUnits = [
        i
        for i in dataBlock.filter(objects=Unit)
        if i.name in unitNames]
    allWaveformsList = []
    for thisUnit in allUnits:
        if verbose:
            print('Extracting {}'.format(thisUnit.name))
        waveformsDict = {}
        uniqueSpiketrains = []
        for i, st in enumerate(thisUnit.spiketrains):
            if st not in uniqueSpiketrains:
                uniqueSpiketrains.append(st)
                waveformsDict.update({
                    i: preproc.spikeTrainWaveformsToDF([st])})
        unitWaveforms = pd.concat(
            waveformsDict,
            names=['segment'] + waveformsDict[0].index.names)
        allWaveformsList.append(unitWaveforms.query(dataQueryTemplate))
    allWaveforms = pd.concat(allWaveformsList)
    saveIndexNames = allWaveforms.index.names
    allWaveforms.reset_index(inplace=True)
    
    if collapseSizes:
        try:
            allWaveforms.loc[allWaveforms['pedalSizeCat'] == 'XL', 'pedalSizeCat'] = 'L'
            allWaveforms.loc[allWaveforms['pedalSizeCat'] == 'XS', 'pedalSizeCat'] = 'S'
        except Exception:
            traceback.print_exc()
    try:
        for idx, (name, group) in enumerate(allWaveforms.groupby(['segment', 'index'])):
            if group['amplitudeFuzzy'].sum() == 0:
                allWaveforms.loc[group.index, 'programFuzzy'] = idx % 3
    except Exception:
        traceback.print_exc()
    
    allWaveforms.set_index(
        saveIndexNames,
        inplace=True)
    allWaveforms.columns.name = 'bin'
    return allWaveforms
