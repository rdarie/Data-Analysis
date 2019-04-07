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

rasterToPlot = [
    i
    for i in unitNames
    if (('_fr' in i) and ('_fr_fr' not in i))]
continuousToPlot = [
    i
    for i in unitNames
    if '_fr_fr' in i]

dataQueryTemplate = '&'.join([
    '(RateInHz >= 100)',
    '(feature==\'{}\')',
    '((pedalSizeCat == \'M\') | (pedalSizeCat == \'Control\'))',
    ])

#  rasterToPlot = [rasterToPlot[5]]
#  continuousToPlot = [continuousToPlot[5]]
for rasterName, continuousName in zip(rasterToPlot, continuousToPlot):
    thisRaster = dataBlock.filter(
        objects=Unit, name=rasterName)[0]
    
    waveformsDict = {}
    uniqueSpiketrains = []
    for i, st in enumerate(thisRaster.spiketrains):
        if st not in uniqueSpiketrains:
            uniqueSpiketrains.append(st)
            waveformsDict.update({
                i: preproc.spikeTrainWaveformsToDF([st])})
    
    allWaveforms = pd.concat(
        waveformsDict,
        names=['segment'] + waveformsDict[0].index.names)
    allWaveforms.columns.name = 'bin'
    allWaveformsPlot = allWaveforms.stack().reset_index(name='signal')
    
    dataQuery = dataQueryTemplate.format(rasterName)
    plotDF = allWaveformsPlot.query(dataQuery)
    
    thisContinuous = dataBlock.filter(
        objects=Unit, name=continuousName)[0]
    
    waveformsDict = {}
    uniqueSpiketrains = []
    for i, st in enumerate(thisContinuous.spiketrains):
        if st not in uniqueSpiketrains:
            uniqueSpiketrains.append(st)
            waveformsDict.update({
                i: preproc.spikeTrainWaveformsToDF([st])})
    
    allWaveformsContinuous = pd.concat(
        waveformsDict,
        names=['segment'] + waveformsDict[0].index.names)
    allWaveformsContinuous.columns.name = 'bin'
    allWaveformsContinuousPlot = (
        allWaveformsContinuous
        .stack()
        .reset_index(name='signal'))
    
    dataQuery = dataQueryTemplate.format(continuousName)
    plotDFContinuous = allWaveformsContinuousPlot.query(dataQuery)
    plotDF.loc[:, 'signal_twin'] = plotDFContinuous.loc[:, 'signal']
    
    asp.getRasterFacetIdx(
        plotDF, 'index',
        col='pedalMovementCat', row='program')
    
    g = asp.twin_relplot(
        x='bin', y1='index_facetIdx', y2='signal_twin',
        hue='amplitudeFuzzy',
        col='pedalMovementCat', row='program',
        func1_kws={'marker': '|'}, func2_kws={'ci': 'sd'},
        facet1_kws={'sharey': False}, facet2_kws={},
        query1='(signal == 1000)', query2=None,
        height=5, aspect=1.5, kind1='scatter', kind2='line', data=plotDF)
    
    plt.suptitle(dataQuery)
    #  plt.show()
    #  block=False
    #  plt.pause(3)
    plt.savefig(
        os.path.join(
            figureFolder, 'alignedSignals', '{}.pdf'.format(rasterName)))
    plt.close()
