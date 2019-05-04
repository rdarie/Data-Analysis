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

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("dark")

dataBlock = preproc.loadWithArrayAnn(
    experimentTriggeredPath)

unitNames = np.unique([
    i.name
    for i in dataBlock.filter(objects=Unit)])

continuousToPlot = [
    i
    for i in unitNames
    if '_fr' in i]
#  continuousToPlot = continuousToPlot[:5]
#  done extracting data
pdb.set_trace()
    
allAsig = []
dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '(pedalMovementCat==\'reachedPeak\')',
    '((amplitudeCatFuzzy==0) | (amplitudeCatFuzzy==3))',
    ])

for idx, unitName in enumerate(continuousToPlot):
    print('Getting rasters for unit {}'.format(idx))
    asigWide, dataQuery = tempgca.getConditionAverages(dataBlock, unitName, dataQueryTemplate)
    allAsig.append(asigWide)
#  done extracting data
pdb.set_trace()
    
allFeatures = np.hstack([
    i.values for i in allAsig])
allTargets = pd.Series(asigWide.index.get_level_values('pedalSizeCat'))

noStimMask = asigWide.index.get_level_values('amplitudeCatFuzzy') == 0

trainTargets = allTargets[noStimMask].reset_index(drop=True)
trainFeatures = allFeatures[noStimMask]

nExamples = trainTargets.value_counts().min()
subsetList = []
for name in pd.unique(trainTargets):
    theseIndexes = trainTargets[trainTargets == name].index
    subsetList += (random.sample(theseIndexes.tolist(), nExamples))

trainFeatures = trainFeatures[subsetList, :]
trainTargets = trainTargets[subsetList].values

estimator = Pipeline([('pca', PCA(n_components=int(len(trainTargets)/5))), ('lda', LDA())])
estimator.fit(trainFeatures, trainTargets)
#  done fitting estimator
pdb.set_trace()

nExamples = allTargets.value_counts().min()
subsetList = []
for name in pd.unique(allTargets):
    theseIndexes = allTargets[allTargets == name].index
    subsetList += (random.sample(theseIndexes.tolist(), nExamples))

projectedFeatures = estimator.transform(allFeatures)
featNames = ['LD{}'.format(i) for i in range(projectedFeatures.shape[1])]
featuresDF = pd.DataFrame(
    projectedFeatures, index=asigWide.index, columns = featNames)
featuresDF = featuresDF.iloc[subsetList, :]
featuresDF.reset_index(inplace=True)
featuresDF.loc[featuresDF['amplitudeCatFuzzy']==0, 'programFuzzy'] = -1

cmapLookup = {
    -1: sns.cubehelix_palette(
    start=0, rot=-0.2, light=0.7, dark=0.3, reverse=False, as_cmap=True),
    0: sns.cubehelix_palette(
    start=0.6, rot=-0.2, light=0.7, dark=0.3, reverse=False, as_cmap=True),
    1: sns.cubehelix_palette(
    start=0.6, rot=0.2, light=0.7, dark=0.3, reverse=False, as_cmap=True),
    2: sns.cubehelix_palette(
    start=0.1, rot=0.2, light=0.7, dark=0.3, reverse=False, as_cmap=True)}

palLookup = {
    -1: sns.cubehelix_palette(
    start=0, rot=-0.2, light=0.7, dark=0.3, reverse=False),
    0: sns.cubehelix_palette(
    start=0.6, rot=-0.2, light=0.7, dark=0.3, reverse=False),
    1: sns.cubehelix_palette(
    start=0.6, rot=0.2, light=0.7, dark=0.3, reverse=False),
    2: sns.cubehelix_palette(
    start=0.1, rot=0.2, light=0.7, dark=0.3, reverse=False)}

posDF = featuresDF.query('amplitudeCatFuzzy==0')
densityAlpha = 0.5
markerAlpha = 0.75
nContourLevels = 5
ax = sns.scatterplot(
    x='LD0', y='LD1', data=posDF,
    linewidth=0, c=np.atleast_2d(palLookup[-1][4]), alpha=markerAlpha)

ax = sns.kdeplot(
    posDF['LD0'], posDF['LD1'],
    n_levels=nContourLevels,
    cmap=cmapLookup[-1], alpha=densityAlpha,
    shade=True, shade_lowest=False)

for name, group in posDF.groupby('pedalSizeCat'):
    plt.scatter(
        [group['LD0'].mean()], [group['LD1'].mean()],
        s=30, c=np.atleast_2d(palLookup[-1][4]),
        marker='+', label='size {}'.format(name))

plt.legend()
plt.savefig(os.path.join(figureFolder, 'motion_LDA.pdf'))
plt.close()
markerLookup = {'S': 's', 'M': 'o', 'L': 'X'}

for name, group in featuresDF.groupby('programFuzzy'):
    #baselineDF = featuresDF.query(
    #    '((programFuzzy==-1) & (amplitudeCatFuzzy==0) & (pedalSizeCat==\'{}\'))'.format(name[1]))
    ax = sns.scatterplot(
        s=15, x='LD0', y='LD1', style='pedalSizeCat', data=posDF,
        markers=markerLookup,
        linewidth=0, c=np.atleast_2d(palLookup[-1][4]), alpha=markerAlpha, legend=False)
    ax = sns.kdeplot(
        posDF['LD0'], posDF['LD1'],
        cmap=cmapLookup[-1],
        n_levels=nContourLevels, alpha=densityAlpha,
        shade=True, shade_lowest=False, label='baseline')
    for subName, subGroup in posDF.groupby('pedalSizeCat'):
        plt.scatter(
            [subGroup['LD0'].mean()], [subGroup['LD1'].mean()],
            s=30, c=np.atleast_2d(palLookup[-1][4]),
            marker=markerLookup[subName], label='size {}'.format(subName))
    # plot current observations
    testDF = group.query(
        '(amplitudeCatFuzzy==3)|(amplitudeCatFuzzy==2)')
    ax = sns.scatterplot(
        x='LD0', y='LD1', style='pedalSizeCat', data=testDF,
        markers=markerLookup,
        s=15, linewidth=0, c=np.atleast_2d(palLookup[name][4]),
        alpha=markerAlpha, legend=False)
    ax = sns.kdeplot(
        testDF['LD0'], testDF['LD1'],
        cmap=cmapLookup[name],
        n_levels=nContourLevels, alpha=densityAlpha,
        shade=True, shade_lowest=False, label='stimulation')
    for subName, subGroup in testDF.groupby('pedalSizeCat'):
        plt.scatter(
            [subGroup['LD0'].mean()], [subGroup['LD1'].mean()],
            s=30, c=np.atleast_2d(palLookup[name][4]),
            marker=markerLookup[subName])
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(
        os.path.join(figureFolder, 'motionStim_LDA_{}.pdf'.format(name)),
        bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()