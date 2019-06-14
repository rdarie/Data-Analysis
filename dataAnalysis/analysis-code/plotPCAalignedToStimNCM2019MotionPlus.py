"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
"""
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
from joblib import dump, load
import quantities as pq
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import multipletests as mt
import getConditionAverages as tempgca
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("dark")

if arguments['--processAll']:
    dataBlock = preproc.loadWithArrayAnn(
        experimentTriggeredPath)
else:
    dataBlock = preproc.loadWithArrayAnn(
        trialTriggeredPath)

unitNames = np.unique([
    i.name
    for i in dataBlock.filter(objects=Unit)])
rasterToPlot = [
    i
    for i in unitNames
    if '_raster' in i]

continuousToPlot = [
    i
    for i in unitNames
    if '_fr' in i]
nUnits = len(continuousToPlot)
# during movement and stim
pedalSizeQuery = '(' + '|'.join([
    '(pedalSizeCat == \'{}\')'.format(i)
    for i in ['M', 'L', 'XL']
    ]) + ')'
dataQuery = '&'.join([
    #  '(feature==\'{}\')',
    '(RateInHzFuzzy==100)|(RateInHzFuzzy==0)',
    '(pedalMovementCat==\'outbound\')',
    pedalSizeQuery,
    #  '(pedalDirection == \'CW\')'
    ])

enablePlots = True
allPvals = {}
colorPal = "ch:0.6,-.2,dark=.3,light=0.7,reverse=1"  #  for firing rates
rasterToPlot = ['elec75#0_raster#0', 'elec75#1_raster#0']
continuousToPlot = ['elec75#0_fr#0', 'elec75#1_fr#0']

pdfName = '{}_motionPlusStim.pdf'.format(experimentName)
uniqProgs = np.array([0, 1, 2, 999])
stimProgs = np.array([0, 1, 2])

countingQuery = '&'.join([
    '(pedalMovementCat==\'outbound\')',
    '(pedalSizeCat != \'NA\')'
    ])

dummyAsig = tempgca.getConditionAverages(
    dataBlock, continuousToPlot[0], collapseSizes=False,
    dataQuery=countingQuery)
dummyAsig.reset_index(inplace=True)
countedNames = [
    'segment', 'pedalSizeCat',
    'RateInHzFuzzy', 'pedalMetaCat', 'pedalDirection']
for countedName in countedNames:
    print(dummyAsig[countedName].value_counts())
    print('\n')

import scipy
def triggeredAsigCompareMeans(
        asigWide, groupBy, testVar,
        tStart=None, tStop=None,
        testWidth=100, testStride=20,
        correctMultiple=True):
    
    if tStart is None:
        tStart = asigWide.columns[0]
    if tStop is None:
        tStop = asigWide.columns[-1]
    testBins = np.arange(
        tStart, tStop, testStride)
    
    if isinstance(groupBy, str):
        pValIndex = pd.Index(
            asigWide.groupby(groupBy).groups.keys())
        pValIndex.name = groupBy
    else:
        pValIndex = pd.MultiIndex.from_tuples(
            asigWide.groupby(groupBy).groups.keys(),
            names=groupBy)

    pVals = pd.DataFrame(
        np.nan,
        index=pValIndex,
        columns=testBins)
    pVals.columns.name = 'bin'
    for testBin in testBins:
        tMask = (
            (asigWide.columns > testBin - testWidth / 2) &
            (asigWide.columns < testBin + testWidth / 2)
            )
        testAsig = asigWide.loc[:, tMask]
        for name, group in testAsig.groupby(groupBy):
            testGroups = [
                np.ravel(i)
                for _, i in group.groupby(testVar)]
            if len(testGroups) > 1:
                stat, p = scipy.stats.kruskal(*testGroups)
                pVals.loc[name, testBin] = p

    if correctMultiple:
        flatPvals = pVals.stack()
        _, fixedPvals, _, _ = mt(flatPvals.values, method='holm')
        flatPvals.loc[:] = fixedPvals
        flatPvals = flatPvals.unstack('bin')
        pVals.loc[flatPvals.index, flatPvals.columns] = flatPvals

    return pVals


with PdfPages(os.path.join(figureFolder, pdfName)) as pdf:
    for idx, (rasterName, continuousName) in enumerate(zip(rasterToPlot, continuousToPlot)):
        try:
            rasterWide = tempgca.getConditionAverages(
                dataBlock, rasterName, dataQuery,
                makeControlProgram=True, duplicateControlsByProgram=True)
            asigWide = tempgca.getConditionAverages(
                dataBlock, continuousName, dataQuery,
                makeControlProgram=True, duplicateControlsByProgram=True)
            raster = rasterWide.stack().reset_index(name='raster')
            asig = asigWide.stack().reset_index(name='fr')
            
            testStride = 20
            elecPvals = triggeredAsigCompareMeans(
                asigWide.query('programFuzzy != 999'),
                groupBy=['pedalDirection', 'programFuzzy'], testVar='amplitudeCatFuzzy',
                tStart=500, tStop=None,
                testWidth=100, testStride=testStride)
            #  correct for comparisons across units
            elecPvals = elecPvals * len(rasterToPlot)

            allPvals.update({idx: elecPvals})
            if enablePlots:
                raster.loc[:, 'fr'] = asig.loc[:, 'fr']
                raster = asp.getRasterFacetIdx(
                    raster, 't',
                    col='programFuzzy', row='pedalDirection', hue='amplitudeCatFuzzy')
                g = asp.twin_relplot(
                    x='bin',
                    y2='fr', y1='t_facetIdx',
                    query2=None, query1='(raster == 1000)',
                    hue='amplitudeCatFuzzy',
                    col='programFuzzy', row='pedalDirection',
                    palette=colorPal,
                    func1_kws={'marker': '|', 'alpha': 0.6}, func2_kws={'ci': 'sem'},
                    facet1_kws={'sharey': False}, facet2_kws={'sharey': True},
                    height=5, aspect=1.5, kind1='scatter', kind2='line', data=raster)
                for (ro, co, hu), dataSubset in g.facet_data():
                    progIdx = g.col_names[co]
                    rowIdx = g.row_names[ro]
                    try:
                        pQuery = '&'.join([
                            '(programFuzzy=={})'.format(progIdx),
                            '(pedalDirection==\'{}\')'.format(rowIdx),
                        ])
                        thesePvals = (
                            elecPvals
                            .query(pQuery))
                        significantBins = (
                            thesePvals
                            .columns[np.ravel(thesePvals < pThresh)])
                        if not significantBins.empty:
                            ymin, ymax = g.axes[ro, co].get_ylim()
                            g.axes[ro, co].plot(
                                significantBins,
                                significantBins ** 0 * ymax * 0.99,
                                'm*')
                    except Exception:
                        traceback.print_exc()
                        pdb.set_trace()
                plt.suptitle(rasterName)
                for ax in g.axes.flat:
                    ax.axvline(500, color='m')
                pdf.savefig()
                plt.close()
        except Exception:
            traceback.print_exc()
            #  pdb.set_trace()

allPvalsWide = pd.concat(allPvals, names=['unit'] + elecPvals.index.names)
allPvalsDF = pd.DataFrame(allPvalsWide.stack(), columns=['pvalue'])
allPvalsDF['significant'] = allPvalsDF['pvalue'] < pThresh

gPH = sns.catplot(
    y='significant', x='bin', row='pedalDirection', col='programFuzzy',
    kind='bar', ci=None, data=allPvalsDF.reset_index(),
    linewidth=0, color='m', dodge=False
    )
for ax in gPH.axes.flat:
    ax.set_ylim((0, 1))
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if (i%200 != 0): labels[i] = ''  # skip every nth label
    ax.set_xticklabels(labels, rotation=30)
    newwidth = testStride
    for bar in ax.patches:
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width/2.
        bar.set_x(centre - newwidth/2.)
        bar.set_width(newwidth)
gPH.savefig(os.path.join(figureFolder, 'motionPlusStim_pCount.pdf'))
plt.close()