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
    dataBlock, unitName, dataQueryTemplate,
    collapseSizes=True):
    thisUnit = dataBlock.filter(
        objects=Unit, name=unitName)[0]
    waveformsDict = {}
    uniqueSpiketrains = []
    for i, st in enumerate(thisUnit.spiketrains):
        if st not in uniqueSpiketrains:
            uniqueSpiketrains.append(st)
            waveformsDict.update({
                i: preproc.spikeTrainWaveformsToDF([st])})
    allWaveforms = pd.concat(
        waveformsDict,
        names=['segment'] + waveformsDict[0].index.names)
    allWaveforms.columns.name = 'bin'
    allWaveformsSkinny = allWaveforms.stack().reset_index(name='signal')
    dataQuery = dataQueryTemplate.format(unitName)
    aveDF = allWaveformsSkinny.query(dataQuery)
    
    if collapseSizes:
        try:
            aveDF.loc[aveDF['pedalSizeCat'] == 'XL', 'pedalSizeCat'] = 'L'
            aveDF.loc[aveDF['pedalSizeCat'] == 'XS', 'pedalSizeCat'] = 'S'
        except Exception:
            pass
    try:  
        for idx, (name, group) in enumerate(aveDF.groupby(['segment', 'index'])):
            if group['amplitudeFuzzy'].sum() == 0:
                aveDF.loc[group.index, 'programFuzzy'] = idx % 3
    except Exception:
        pass
    
    aveDF.set_index(aveDF.columns.drop(['bin', 'signal']).tolist(), inplace=True)
    returnDF = aveDF.pivot(columns='bin')['signal']
    return returnDF, dataQuery
