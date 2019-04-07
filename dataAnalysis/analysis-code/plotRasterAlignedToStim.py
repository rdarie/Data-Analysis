import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.plotting.aligned_signal_plots as asp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#  load options
from currentExperiment import *
from joblib import dump, load
import quantities as pq

dataBlock = preproc.loadWithArrayAnn(
    experimentTriggeredPath)

unitNames = np.unique([
    i.name
    for i in dataBlock.filter(objects=Unit)])
unitsToPlot = [
    i
    for i in unitNames
    if (('_fr' in i) and ('_fr_fr' not in i))]
unitsToPlot = [unitsToPlot[5]]
#  unitsToPlot = ['PC{}#0'.format(i+1) for i in range(10)]
#  unitsToPlot = ['PC3#0']

dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '((RateInHz==100) | (RateInHz==999))',
    '(signal==1000)',
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
    
    asp.getRasterFacetIdx(
        plotDF, 'index',
        col='pedalMovementCat', row='program')
    g = sns.relplot(
        x='bin', y='index_facetIdx',
        hue='amplitudeFuzzy',
        col='pedalMovementCat', row='program',
        height=5, aspect=1.5, kind='scatter', data=plotDF,
        facet_kws={'sharey': False}, marker='|')
    plt.suptitle(unitName)
    #  plt.show()
    #  block=False
    #  plt.pause(3)
    plt.savefig(
        os.path.join(
            figureFolder, 'alignedRasters', '{}.pdf'.format(chanName)))
    plt.close()
