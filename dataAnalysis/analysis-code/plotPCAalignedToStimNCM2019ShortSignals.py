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
sns.set_context("poster")
sns.set_style("whitegrid")

dataBlock = preproc.loadWithArrayAnn(
    experimentTriggeredPath)

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
    
dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '(pedalMovementCat==\'outbound\')',
    '(amplitudeCatFuzzy==\'0\')',
    '(pedalSizeCat == \'M\')',
    '(bin>300)',
    ])

#  rasterToPlot = [rasterToPlot[5]]
#  continuousToPlot = [continuousToPlot[5]]
enablePlots = True
allPvals = {}
colorPal = "ch:0.6,-.2,dark=.3,light=0.7,reverse=1" #  for firing rates
for idx, (rasterName, continuousName) in enumerate(zip(rasterToPlot, continuousToPlot)):
    try:
        rasterWide, dataQuery = tempgca.getConditionAverages(dataBlock, rasterName, dataQueryTemplate)
        asigWide, dataQuery = tempgca.getConditionAverages(dataBlock, continuousName, dataQueryTemplate)
        raster = rasterWide.stack().reset_index(name='raster')
        asig = asigWide.stack().reset_index(name='fr')
        refBin = 310
        elecPvals = pd.DataFrame(
            np.nan,
            index=asigWide.columns,
            columns=['pvalue'])
        tTestStride = 10
        testBins = np.arange(500, 1000, tTestStride)
        pThresh = 1e-3
        xref = asigWide.loc[:, refBin:refBin + 2 * tTestStride].stack()
        for testBin in testBins:
            x1 = asigWide.loc[:, testBin:testBin + 2 * tTestStride].stack()
            tstat, pvalue, df = ttest_ind(x1, xref)
            elecPvals.loc[testBin, 'pvalue'] = pvalue
        elecPvals.loc[elecPvals.index < 300, 'pvalue'] = 1
        elecPvals.interpolate(method='ffill', inplace=True)
        elecPvals.fillna(1, inplace=True)
        elecPvals = elecPvals * len(testBins)
        elecPvals.loc[:, 'significant'] = elecPvals['pvalue'] < pThresh
        allPvals.update({idx: elecPvals})
        if enablePlots:
            raster.loc[:, 'fr'] = asig.loc[:, 'fr']
            asp.getRasterFacetIdx(
                raster, 't',
                col='pedalMovementCat')
            g = asp.twin_relplot(
                x='bin',
                y2='fr', y1='t_facetIdx',
                query2=None, query1='(raster == 1000)',
                col='pedalMovementCat',
                func1_kws={'marker': '|', 'alpha': 0.6}, func2_kws={'ci': 'sem'},
                facet1_kws={'sharey': False}, facet2_kws={'sharey': True},
                height=5, aspect=1.5, kind1='scatter', kind2='line', data=raster)
            for (ro, co, hu), dataSubset in g.facet_data():
                progIdx = g.col_names[co]
                if elecPvals['significant'].any():
                    ymin, ymax = g.axes[ro, co].get_ylim()
                    g.axes[ro, co].plot(
                        elecPvals.loc[elecPvals['significant'], 'bin'],
                        elecPvals.loc[elecPvals['significant'], 'bin'] ** 0 * ymax * 0.99,
                        'm*')
            plt.suptitle(dataQuery)
            g.savefig(
                os.path.join(
                    alignedRastersFolder, '{}_motion.pdf'.format(rasterName)))
            plt.close()
            if idx > 3: break
    except Exception:
        traceback.print_exc()
    
allPvalsDF = pd.concat(allPvals).reset_index()
gPH = sns.catplot(
    y='significant', x='bin',
    kind='bar', ci=None, data=allPvalsDF,
    linewidth=0, color='m', dodge=False
    )
for ax in gPH.axes.flat:
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if (i%5 != 0): labels[i] = '' # skip every nth label
    ax.set_xticklabels(labels, rotation=30)
    newwidth = tTestStride
    for bar in ax.patches:
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width/2.
        bar.set_x(centre - newwidth/2.)
        bar.set_width(newwidth)
gPH.savefig(os.path.join(figureFolder, 'motion_pCount.pdf'))
plt.close()

# during movement and stim
dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '(pedalMovementCat==\'outbound\')',
    '(pedalSizeCat == \'M\')',
    '(bin>300)',
    ])
#  rasterToPlot = [rasterToPlot[5]]
#  continuousToPlot = [continuousToPlot[5]]
enablePlots = True
allPvals = {}
colorPal = "ch:0.6,-.2,dark=.3,light=0.7,reverse=1" #  for firing rates
for idx, (rasterName, continuousName) in enumerate(zip(rasterToPlot, continuousToPlot)):
    try:
        rasterWide, dataQuery = tempgca.getConditionAverages(dataBlock, rasterName, dataQueryTemplate)
        asigWide, dataQuery = tempgca.getConditionAverages(dataBlock, continuousName, dataQueryTemplate)
        raster = rasterWide.stack().reset_index(name='raster')
        asig = asigWide.stack().reset_index(name='fr')
        refBin = 310
        uniqProgs = np.array([0, 1, 2])
        elecPvals = pd.DataFrame(
            np.nan,
            index=uniqProgs,
            columns=asigWide.columns)
        tTestStride = 10
        testBins = np.arange(500, 1000, tTestStride)
        pThresh = 1e-3
        for progIdx in uniqProgs:
            thisAsig = asigWide.query('(programFuzzy=={})'.format(progIdx))
            uniqueAmps = thisAsig.index.get_level_values('amplitude').unique()
            maxAmp = uniqueAmps.max()
            minAmp = uniqueAmps.min()
            #  testAmp = uniqueAmps[uniqueAmps < maxAmp * .8].max()
            for testBin in testBins:
                x1 = thisAsig.query('(amplitude=={})'.format(maxAmp)).loc[:, testBin:testBin + 2 * tTestStride].stack()
                x2 = thisAsig.query('(amplitude=={})'.format(minAmp)).loc[:, testBin:testBin + 2 * tTestStride].stack()
                tstat, pvalue, df = ttest_ind(x1, x2)
                elecPvals.loc[progIdx, testBin] = pvalue
        elecPvals.loc[elecPvals.index < 300, 'pvalue'] = 1
        elecPvals.interpolate(method='ffill', axis=1, inplace=True)
        elecPvals.fillna(1, inplace=True)
        elecPvals = elecPvals * len(uniqProgs) * len(testBins)
        elecPvals = elecPvals.stack().reset_index(name='pvalue')
        elecPvals['significant'] = elecPvals['pvalue'] < pThresh
        allPvals.update({idx: elecPvals})
        if enablePlots:
            raster.loc[:, 'fr'] = asig.loc[:, 'fr']
            asp.getRasterFacetIdx(
                raster, 't',
                col='pedalMovementCat')
            g = asp.twin_relplot(
                x='bin',
                y2='fr', y1='t_facetIdx',
                query2=None, query1='(raster == 1000)',
                hue='amplitudeCatFuzzy',
                col='programFuzzy',
                palette=colorPal,
                func1_kws={'marker': '|', 'alpha': 0.6}, func2_kws={'ci': 'sem'},
                facet1_kws={'sharey': False}, facet2_kws={'sharey': True},
                height=5, aspect=1.5, kind1='scatter', kind2='line', data=raster)
            for (ro, co, hu), dataSubset in g.facet_data():
                progIdx = g.col_names[co]
                if elecPvals['significant'].any():
                    ymin, ymax = g.axes[ro, co].get_ylim()
                    g.axes[ro, co].plot(
                        elecPvals.loc[elecPvals['significant'], 'bin'],
                        elecPvals.loc[elecPvals['significant'], 'bin'] ** 0 * ymax * 0.99,
                        'm*')
            plt.suptitle(dataQuery)
            g.savefig(
                os.path.join(
                    alignedRastersFolder, '{}_motionPlusStim.pdf'.format(rasterName)))
            plt.close()
            if idx > 3: break
    except Exception:
        traceback.print_exc()
    
allPvalsDF = pd.concat(allPvals).reset_index()
gPH = sns.catplot(
    y='significant', x='bin',
    kind='bar', ci=None, data=allPvalsDF,
    linewidth=0, color='m', dodge=False
    )
for ax in gPH.axes.flat:
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if (i%5 != 0): labels[i] = '' # skip every nth label
    ax.set_xticklabels(labels, rotation=30)
    newwidth = tTestStride
    for bar in ax.patches:
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width/2.
        bar.set_x(centre - newwidth/2.)
        bar.set_width(newwidth)
gPH.savefig(os.path.join(figureFolder, 'motionPlusStim_pCount.pdf'))
plt.close()
