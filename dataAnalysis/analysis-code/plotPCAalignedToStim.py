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
from currentExperiment import *
from joblib import dump, load
import quantities as pq

sns.set_style("whitegrid")
sns.set_context("talk")

dataBlock = preproc.loadWithArrayAnn(
    experimentTriggeredPath)

unitNames = np.unique([
    i.name
    for i in dataBlock.filter(objects=Unit)])
unitsToPlot = [
    i
    for i in unitNames
    if 'elec' not in i]

dataQueryTemplate = '&'.join([
        '(RateInHz >= 100)',
        '(feature==\'{}\')',
        '((pedalSizeCat == \'M\') | (pedalSizeCat == \'Control\'))',
        ])

for unitName in unitsToPlot:
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
    
    allWaveformsPlot = allWaveforms.stack().reset_index(name='signal')
    dataQuery = dataQueryTemplate.format(unitName)
    plotDF = allWaveformsPlot.query(dataQuery)
    g = sns.relplot(
        x='bin', y='signal',
        hue='amplitudeFuzzy',
        col='pedalMovementCat', row='program', ci='sem',
        height=5, aspect=1.5, kind='line', data=plotDF)
    plt.suptitle(dataQuery)
    #  plt.show()
    #  break
    #  block=False
    #  plt.pause(3)
    g.fig.savefig(
        os.path.join(
            alignedFeaturesFolder, '{}.pdf'.format(unitName)))
    plt.close(g.fig)
