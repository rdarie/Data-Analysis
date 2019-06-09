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
import getConditionAverages as tempgca

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("dark")
experimentTriggeredPath = (
    experimentTriggeredPath
    .replace('triggered', 'triggered_long'))
dataBlock = preproc.loadWithArrayAnn(
    experimentTriggeredPath)

unitNames = np.unique([
    i.name
    for i in dataBlock.filter(objects=Unit)])

#  positional averages
unitsToPlot = ['position#0']
dataQueryTemplate = '&'.join([
        '(feature==\'{}\')',
        '(pedalMovementCat==\'midPeak\')',
        '(bin>2750)', '(bin<7250)',
        ])
colorPal = "ch:0,-.2,dark=.3,light=0.7,reverse=1" #  for positions
for unitName in unitsToPlot:
    plotDFWide, dataQuery = tempgca.getConditionAverages(
        dataBlock, unitName, dataQueryTemplate, collapseSizes=False)
    plotDF = plotDFWide.stack().reset_index(name='signal')
    plotDF['signal'] = plotDF['signal'] * (-100)
    plotDF['signal'] = plotDF['signal'] - plotDF['signal'].min()
    g = sns.relplot(
        x='bin', y='signal',
        hue='pedalSizeCat', ci='sd',
        hue_order=['XS', 'S', 'M', 'L', 'XL'],
        palette=colorPal,
        height=5, aspect=1.5, kind='line', data=plotDF)
    plt.suptitle(dataQuery)
    g.fig.savefig(
        os.path.join(
            alignedFeaturesFolder, '{}.pdf'.format(unitName)))
    plt.close(g.fig)

#  amplitude averages
unitsToPlot = ['amplitude#0']
colorPal = "ch:1.2,-.2,dark=.3,light=0.7,reverse=1"  # for positions
dataQueryTemplate = '&'.join([
        '(feature==\'{}\')',
        '(pedalMovementCat==\'midPeak\')',
        '(pedalSizeCat==\'M\')',
        '(bin>2750)', '(bin<7250)',
        ])
for unitName in unitsToPlot:
    plotDFWide, dataQuery = tempgca.getConditionAverages(dataBlock, unitName, dataQueryTemplate)
    plotDF = plotDFWide.stack().reset_index(name='signal')
    plotDF['signal'] = plotDF['signal'] / 10
    g = sns.relplot(
        x='bin', y='signal',
        hue='amplitudeFuzzy',
        col='programFuzzy', ci='sd',
        height=5, aspect=1.5, kind='line', data=plotDF)
    plt.suptitle(dataQuery)
    g.fig.savefig(
        os.path.join(
            alignedFeaturesFolder, '{}.pdf'.format(unitName)))
    plt.close(g.fig)