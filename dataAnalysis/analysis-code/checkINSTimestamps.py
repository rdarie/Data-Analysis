"""
Usage:
    checkINSTimestamps [options]

Options:
    --trialIdx=trialIdx        which trial to analyze [default: 1]
    --exp=exp                  which experimental day to analyze
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate postscript output by default
import os
import pdb
import dataAnalysis.preproc.ns5 as ns5
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, AnalogSignal, SpikeTrain)
import neo
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from importlib import reload
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=.5)

annotationsList = []
for trialIdx in [1]:
    ns5FileName = 'Trial00{}'.format(trialIdx)
    insDataPath = os.path.join(
        scratchFolder,
        ns5FileName + '_ins.nix'
    )
    insBlock = ns5.loadWithArrayAnn(
        insDataPath, fromRaw=False)
    theseAnnDF = ns5.concatenateUnitSpikeTrainWaveformsDF(
        insBlock.filter(objects=Unit), verbose=True)
    annotationsList.append(theseAnnDF.index.to_frame().reset_index(drop=True))
    
annotationsDF = pd.concat(annotationsList)
annotationsDF.sort_values(by='t', inplace=True)
annotationsDF.reset_index(inplace=True)
annotationsDF['offsetFromExpectedModulus'] = annotationsDF['offsetFromExpected'] % (annotationsDF['RateInHz'] ** (-1) / 2)
for name, group in annotationsDF.groupby('feature'):
    print('{}, {}'.format(name, pd.unique(group['amplitude'])))


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


plotDF = pd.melt(
    annotationsDF,
    id_vars=['feature', 'amplitude', 'RateInHz', 'usedSlotToDetect', 'usedExpectedT'],
    value_vars=['offsetFromExpected', 'offsetFromLogged', 'offsetFromExpectedModulus'],
    var_name='measuredFrom',
    value_name='offset')
dataQuery = ' & '.join([
    '(amplitude > 0)',
    '(usedSlotToDetect == True)'
    ])

g = sns.catplot(
    x='amplitude', y='offsetFromLogged',
    hue='RateInHz', col='feature', kind='box',
    data=annotationsDF.query(dataQuery))
plt.suptitle('offsetFromLogged'); plt.show()
g = sns.catplot(
    x='amplitude', y='offsetFromExpected',
    hue='RateInHz', col='feature', kind='box',
    data=annotationsDF.query(dataQuery))
plt.suptitle('offsetFromExpected'); plt.show()
g = sns.catplot(
    x='amplitude', y='offsetFromExpected',
    hue='RateInHz', kind='box',
    data=annotationsDF.query(dataQuery))
plt.suptitle('offsetFromExpected'); plt.show()
#
g = sns.lmplot(
    x='RateInHz', y='offsetFromLogged',
    hue='amplitude', col='feature',
    data=annotationsDF.query(dataQuery))
plt.suptitle('offsetFromLogged'); plt.show()
g = sns.lmplot(
    x='RateInHz', y='offsetFromExpected',
    hue='amplitude', col='feature',
    data=annotationsDF.query(dataQuery), x_jitter=5)
plt.suptitle('offsetFromExpected'); plt.show()
g = sns.regplot(
    x='RateInHz', y='offsetFromLogged',
    data=annotationsDF.query(dataQuery), x_jitter=5)
plt.suptitle('offsetFromLogged'); plt.show()
g = sns.regplot(
    x='RateInHz', y='offsetFromExpected',
    data=annotationsDF.query(dataQuery), x_jitter=5)
plt.suptitle('offsetFromExpected'); plt.show()
#
g = sns.regplot(
    x='RateInHz', y='offsetFromExpectedModulus',
    data=annotationsDF.query(dataQuery), x_jitter=5)
plt.suptitle('offsetFromExpectedModulus'); plt.show()

for name, group in plotDF.groupby('feature'):
    print('{}, {}'.format(name, pd.unique(group['amplitude'])))

dataQuery = ' & '.join([
     '(amplitude >= 0)',
     "measuredFrom == 'offsetFromExpected'",
     '(usedSlotToDetect == True)'
    ])

g = sns.FacetGrid(
    plotDF.query(dataQuery),
    row='measuredFrom', col='feature', hue='RateInHz', aspect=1.5)
g.map(annotated_distplot, 'offset')
for ax in g.axes.flat:
    ax.legend()
plt.suptitle(ns5FileName)
plt.show()

g = sns.FacetGrid(
    plotDF.query(dataQuery),
    row='measuredFrom', hue='RateInHz', aspect=1.5)
g.map(annotated_distplot, 'offset')
for ax in g.axes.flat:
    ax.legend()
plt.suptitle('With slot detection')
g.axes.flat[-1].set_xlabel('Time difference (sec)')
plt.show()

g = sns.FacetGrid(
    plotDF,
    row='measuredFrom', hue='amplitude', aspect=1.5)
g.map(annotated_distplot, 'offset')
for ax in g.axes.flat:
    ax.legend()
plt.suptitle(ns5FileName)
plt.show()

g = sns.FacetGrid(
    plotDF,
    row='measuredFrom', aspect=1.5)
g.map(annotated_distplot, 'offset')
g.axes.flat[-1].set_xlabel('Time difference (sec)')

for ax in g.axes.flat:
    ax.legend()

plt.suptitle('With slot detection')
plt.show()