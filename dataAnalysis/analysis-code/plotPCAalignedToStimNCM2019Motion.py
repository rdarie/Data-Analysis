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
#  load options
from joblib import dump, load
import quantities as pq
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import multipletests as mt
import getConditionAverages as tempgca
from scipy import stats

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
dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '(pedalMovementCat==\'outbound\')',
    '(amplitudeCatFuzzy==\'0\')',
    '(pedalSizeCat == \'M\')',
    '(bin>300)',
    ])
enablePlots = True
allPvals = {}
colorPal = "ch:0,-.2,dark=.3,light=0.7,reverse=1"  # for firing rates
rasterToPlot = [rasterToPlot[87]]
continuousToPlot = [continuousToPlot[87]]

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
        tTestWidth = 10
        testBins = np.arange(500, 1000, tTestStride)
        pThresh = 1e-3
        xref = asigWide.loc[:, refBin:refBin + tTestWidth].stack()
        for testBin in testBins:
            x1 = asigWide.loc[:, testBin:testBin + tTestWidth].stack()
            # tstat, pvalue, df = ttest_ind(x1, xref)
            tstat, pvalue = stats.ttest_ind(x1, xref, equal_var=False)
            elecPvals.loc[testBin, 'pvalue'] = pvalue
        
        flatPvals = elecPvals.loc[elecPvals['pvalue'].notna(), 'pvalue']
        _, fixedPvals, _, _ = mt(flatPvals.values, method='holm')
        flatPvals.loc[:] = fixedPvals
        elecPvals.loc[flatPvals.index, 'pvalue'] = flatPvals
        elecPvals.interpolate(method='ffill', inplace=True)
        elecPvals.fillna(1, inplace=True)
        elecPvals = elecPvals * nUnits
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
                        elecPvals.loc[elecPvals['significant'], :].index,
                        elecPvals.loc[elecPvals['significant'], :].index ** 0 * ymax * 0.99,
                        'b*')
            plt.suptitle(dataQuery)
            for ax in g.axes.flat:
                ax.axvline(500, color='b')
            g.savefig(
                os.path.join(
                    alignedRastersFolder, '{}_motion.pdf'.format(rasterName)))
            plt.close()
    except Exception:
        pdb.set_trace()
        traceback.print_exc()

allPvalsDF = pd.concat(allPvals).reset_index()
gPH = sns.catplot(
    y='significant', x='bin',
    kind='bar', ci=None, data=allPvalsDF,
    linewidth=0, color='b', dodge=False
    )

for ax in gPH.axes.flat:
    ax.set_ylim((0,1))
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