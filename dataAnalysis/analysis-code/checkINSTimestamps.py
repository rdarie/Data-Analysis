"""
Usage:
    checkINSTimestamps [--trialIdx=trialIdx]

Options:
    --trialIdx=trialIdx   which trial to analyze
"""

from docopt import docopt
import os
import pdb
import dataAnalysis.preproc.ns5 as preproc
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, AnalogSignal, SpikeTrain)
import neo
from currentExperiment import *
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from importlib import reload

sns.set_style('whitegrid')
sns.set_context('poster', font_scale=.5)

annotationsList = []
for trialIdx in [1, 2, 3, 4]:
    ns5FileName = 'Trial00{}'.format(trialIdx)
    insDataPath = os.path.join(
        remoteBasePath, 'processed', experimentName,
        ns5FileName + '_ins.nix'
    )

    insBlock = preproc.loadWithArrayAnn(
        insDataPath, fromRaw=False)

    allUnits = []
    for thisUnit in insBlock.filter(objects=Unit):
        spiketrains = []
        keepUnit = False
        waveformsDict = {}
        for st in thisUnit.spiketrains:
            if (st not in spiketrains) and len(st):
                spiketrains.append(st)
                keepUnit = True
        if keepUnit:
            thisUnit.spiketrains = spiketrains
            allUnits.append(thisUnit)
    
    for unitIdx, thisUnit in enumerate(allUnits):
        annotationsList.append(preproc.spikeTrainArrayAnnToDF(
            thisUnit.spiketrains))

annotationsDF = pd.concat(annotationsList)
annotationsDF.reset_index(inplace=True)

for name, group in annotationsDF.groupby('name'):
    print('{}, {}'.format(name, pd.unique(group['amplitudes'])))
dataQuery = ' & '.join([
     '(amplitudes >= 700)'
    ])
g = sns.catplot(
    x='amplitudes', y='offsetFromLogged',
    hue='rates', col='name', kind='box',
    data=annotationsDF)
plt.show()
g = sns.lmplot(
    x='rates', y='offsetFromLogged',
    hue='amplitudes', col='name',
    data=annotationsDF.query(dataQuery))
plt.show()
g = sns.lmplot(
    x='rates', y='offsetFromExpected',
    hue='amplitudes', col='name',
    data=annotationsDF.query(dataQuery))
plt.show()
g = sns.regplot(
    x='rates', y='offsetFromLogged',
    data=annotationsDF)
plt.show()
g = sns.regplot(
    x='rates', y='offsetFromExpected',
    data=annotationsDF)
plt.show()

plotDF = pd.melt(
    annotationsDF,
    id_vars=['name', 'amplitudes', 'rates'],
    value_vars=['offsetFromExpected', 'offsetFromLogged'],
    var_name='measuredFrom',
    value_name='offset')

for name, group in plotDF.groupby('name'):
    print('{}, {}'.format(name, pd.unique(group['amplitudes'])))

dataQuery = ' & '.join([
     '(amplitudes > 600)',
     'measuredFrom == \'offsetFromExpected\''
    ])


def annotated_distplot(x, **kwargs):
    xBar = x.mean()
    xStd = x.std()
    if 'label' in kwargs.keys():
        kwargs['label'] = '{} : {:.2} pm {:.2}'.format(
            kwargs['label'], xBar, xStd)
    else:
        legendLabel = '{:.2} pm {:.2}'.format(
            xBar, xStd)
        kwargs.update({'label': legendLabel})
    print(kwargs['label'])
    sns.distplot(x, **kwargs)

g = sns.FacetGrid(
    plotDF.query(dataQuery),
    row='measuredFrom', col='name', hue='rates', aspect=1.5)
g.map(annotated_distplot, 'offset')
for ax in g.axes.flat:
    ax.legend()
plt.suptitle(ns5FileName)
plt.show()

g = sns.FacetGrid(
    plotDF.query(dataQuery),
    row='measuredFrom', hue='rates', aspect=1.5)
g.map(annotated_distplot, 'offset')
for ax in g.axes.flat:
    ax.legend()
plt.suptitle(ns5FileName)
plt.show()

g = sns.FacetGrid(
    plotDF,
    row='measuredFrom', hue='amplitudes', aspect=1.5)
g.map(annotated_distplot, 'offset')
for ax in g.axes.flat:
    ax.legend()
plt.suptitle(ns5FileName)
plt.show()