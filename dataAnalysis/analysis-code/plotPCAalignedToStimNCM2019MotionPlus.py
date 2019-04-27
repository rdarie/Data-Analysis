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
from statsmodels.stats.multitest import multipletests as mt
import getConditionAverages as tempgca
from scipy import stats

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("dark")

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
nUnits = len(continuousToPlot)
# during movement and stim
dataQueryTemplate = '&'.join([
    '(RateInHz==100)',
    '(feature==\'{}\')',
    '(pedalMovementCat==\'outbound\')',
    '(pedalSizeCat == \'M\')',
    '(bin>300)',
    ])
#  rasterToPlot = [rasterToPlot[0]]
#  continuousToPlot = [continuousToPlot[0]]
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
        elecPvals.index.name='programFuzzy'
        tTestWidth = 10
        tTestStride = 10
        testBins = np.arange(500, 1000, tTestStride)
        pThresh = 1e-3
        for progIdx in uniqProgs:
            thisAsig = asigWide.query('(programFuzzy=={})'.format(progIdx))
            uniqueAmps = thisAsig.index.get_level_values('amplitudeCatFuzzy').unique()
            maxAmp = uniqueAmps.max()
            minAmp = uniqueAmps.min()
            #  testAmp = uniqueAmps[uniqueAmps < maxAmp * .8].max()
            for testBin in testBins:
                x1 = thisAsig.query('(amplitudeCatFuzzy=={})'.format(maxAmp)).loc[:, testBin:testBin + tTestWidth].stack()
                x2 = thisAsig.query('(amplitudeCatFuzzy=={})'.format(minAmp)).loc[:, testBin:testBin + tTestWidth].stack()
                #  tstat, pvalue, df = ttest_ind(x1, x2)
                tstat, pvalue = stats.ttest_ind(x1, x2, equal_var=False)
                elecPvals.loc[progIdx, testBin] = pvalue

        flatPvals = elecPvals.stack()
        _, fixedPvals, _, _ = mt(flatPvals.values, method='holm')
        flatPvals.loc[:] = fixedPvals
        flatPvals = flatPvals.unstack('bin')
        elecPvals.loc[flatPvals.index, flatPvals.columns] = flatPvals

        elecPvals.interpolate(method='ffill', axis=1, inplace=True)
        elecPvals.fillna(1, inplace=True)
        elecPvals.loc[:, elecPvals.columns < 300] = 1
        #elecPvals = elecPvals * nUnits
        elecPvals = elecPvals.stack().reset_index(name='pvalue')
        elecPvals['significant'] = elecPvals['pvalue'] < pThresh
        """
        for progIdx, pvals in elecPvals.groupby('programFuzzy'):
            ax = sns.distplot(np.log10(pvals['pvalue']), kde=False)
        plt.savefig(os.path.join(figureFolder, 'miniRC_ttestvis.pdf'))
        plt.close()
        """
        allPvals.update({idx: elecPvals})
        if enablePlots:
            raster.loc[:, 'fr'] = asig.loc[:, 'fr']
            raster = asp.getRasterFacetIdx(
                raster, 't',
                col='programFuzzy', hue='amplitudeCatFuzzy')
            g = asp.twin_relplot(
                x='bin',
                y2='fr', y1='t_facetIdx',
                query2=None, query1='(raster == 1000)',
                hue='amplitudeCatFuzzy',
                col='programFuzzy',
                palette=colorPal,
                func1_kws={'marker': '|', 'alpha': 0.6}, func2_kws={'ci': 'sem'},
                facet1_kws={'sharey': False}, facet2_kws={'sharey': True},
                height=5, aspect=1.5, kind1='scatter', kind2='line', data=raster.query('(amplitudeCatFuzzy==0)|(amplitudeCatFuzzy==3)'))
            for (ro, co, hu), dataSubset in g.facet_data():
                progIdx = g.col_names[co]
                thesePvals = elecPvals.query('programFuzzy=={}'.format(progIdx))
                if thesePvals['significant'].any():
                    ymin, ymax = g.axes[ro, co].get_ylim()
                    g.axes[ro, co].plot(
                        thesePvals.loc[thesePvals['significant'], 'bin'],
                        thesePvals.loc[thesePvals['significant'], 'bin'] ** 0 * ymax * 0.99,
                        'm*')
            plt.suptitle(dataQuery)
            for ax in g.axes.flat:
                ax.axvline(500, color='m')
            g.savefig(
                os.path.join(
                    alignedRastersFolder, '{}_motionPlusStim.pdf'.format(rasterName)))
            plt.close()
    except Exception:
        traceback.print_exc()
        pdb.set_trace()

allPvalsDF = pd.concat(allPvals).reset_index()
gPH = sns.catplot(
    y='significant', x='bin', col='programFuzzy',
    kind='bar', ci=None, data=allPvalsDF,
    linewidth=0, color='m', dodge=False
    )
for ax in gPH.axes.flat:
    ax.set_ylim((0,1))
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if (i%200 != 0): labels[i] = '' # skip every nth label
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
pdb.set_trace()
# raster.query('(programFuzzy==0)&(amplitudeCatFuzzy==0)')