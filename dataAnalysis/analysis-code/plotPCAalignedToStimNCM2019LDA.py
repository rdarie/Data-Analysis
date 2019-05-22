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
from statsmodels.multivariate.manova import MANOVA as manova
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

dataQueryTemplate = '&'.join([
    '(pedalMovementCat==\'reachedPeak\')',
    '(pedalSizeCat!=\'Control\')',
    ])
asigWide = tempgca.getConditionAverages(
    dataBlock, continuousToPlot, dataQueryTemplate, verbose=True)
#  done extracting data

allFeatures = asigWide.unstack(level='feature')
allTargets = pd.Series(allFeatures.index.get_level_values('pedalSizeCat'))

noStimMask = allFeatures.index.get_level_values('amplitudeFuzzy') == 0

trainTargets = allTargets.loc[noStimMask].reset_index(drop=True)
trainFeatures = allFeatures.loc[noStimMask, :].reset_index(drop=True)

nExamples = trainTargets.value_counts().min()
nReplacement = 2  # sample with replacement
subsetList = []

for name in pd.unique(trainTargets):
    theseIndexes = trainTargets.loc[trainTargets == name].index
    for i in range(nReplacement):
        subsetList += (random.sample(theseIndexes.tolist(), nExamples))
trainFeatures = trainFeatures.loc[subsetList, :]
trainTargets = trainTargets.loc[subsetList]

estimator = Pipeline([
    ('pca', PCA(n_components=int(len(trainTargets)/3))),
    ('lda', LDA())])
estimator.fit(trainFeatures, trainTargets)
#  done fitting estimator

'''
nExamples = allTargets.value_counts().min()
subsetList = []
for name in pd.unique(allTargets):
    theseIndexes = allTargets[allTargets == name].index
    subsetList += (random.sample(theseIndexes.tolist(), nExamples))
'''
projectedFeatures = estimator.transform(allFeatures)
featNames = ['LD{}'.format(i) for i in range(projectedFeatures.shape[1])]
featuresDF = pd.DataFrame(
    projectedFeatures, index=allFeatures.index, columns=featNames)
#  featuresDF = featuresDF.iloc[subsetList, :]
featuresDF.reset_index(inplace=True)
featuresDF.loc[featuresDF['amplitudeFuzzy'] == 0, 'programFuzzy'] = -1

posDF = featuresDF.query('amplitudeFuzzy==0')

pdb.set_trace()

meanMarkerAlpha = 1
meanMarkerSize = 60
markerSize = 30
nContourLevels = 5
markerColorIdx = int(3 * nContourLevels * 0.7)
markerEdgeColorIdx = int(3 * nContourLevels * 0.1)
arrowColorIdx = int(3 * nContourLevels * 0.3)
kernelBandwidth = 'scott'
LDBounds = [-9.5, 9.5]
markerLookup = {'S': 's', 'M': 'o', 'L': 'X'}
sns.set_style("dark", {
    'font.family': 'Nimbus Sans L',
    'axes.spines.bottom': True, 'axes.edgecolor': 'black',
    'axes.spines.left': True, 'axes.edgecolor': 'black',
    'xtick.color': 'black', 'xtick.bottom': True,
    'ytick.color': 'black', 'ytick.left': True})
maxLight = 0.9
minDark = 0.6
colorRotation = 0.1
colorWheel = [i + 0.25 for i in [0, 0.75, 1.5, 2.25]]
cmapLookup = {
    -1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[3], rot=colorRotation, light=maxLight, dark=minDark, reverse=True, as_cmap=True),
    0: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[0], rot=colorRotation, light=maxLight, dark=minDark, reverse=True, as_cmap=True),
    1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[1], rot=colorRotation, light=maxLight, dark=minDark, reverse=True, as_cmap=True),
    2: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[2], rot=colorRotation, light=maxLight, dark=minDark, reverse=True, as_cmap=True)}

palLookup = {
    -1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[3], rot=colorRotation, light=maxLight, dark=minDark, reverse=True),
    0: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[0], rot=colorRotation, light=maxLight, dark=minDark, reverse=True),
    1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[1], rot=colorRotation, light=maxLight, dark=minDark, reverse=True),
    2: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[2], rot=colorRotation, light=maxLight, dark=minDark, reverse=True)}

'''
electrode configuration was [
    {'cathodes': [10], 'anodes': [16]},
    {'cathodes': [14], 'anodes': [16]},
    {'cathodes': [8], 'anodes': [16]},
    {'cathodes': [12], 'anodes': [16]}]
'''
prgLookup = {
    0: 'Caudal',
    1: 'Rostral',
    2: 'Midline'
    }

densityAlpha = 0.2
baseDensityAlpha = 0.3
densityAlphaMarg = 0.5
baseDensityAlphaMarg = 0.7
markerAlpha = 0.7
for name, group in featuresDF.groupby('programFuzzy'):
    sns.palplot(palLookup[name])
    plt.savefig(
        os.path.join(figureFolder, 'program_{}_colorref.pdf'.format(name)),
        bbox_inches='tight')
    plt.close()
    if name == -1:
        # skip prg -1
        continue
    g = sns.JointGrid(x='LD0', y='LD1', data=posDF)
    for subName in ['S', 'M', 'L']:
        subGroup = posDF.query('pedalSizeCat==\'{}\''.format(subName))
        sns.kdeplot(
            subGroup['LD0'], subGroup['LD1'],
            ax=g.ax_joint,
            cmap=cmapLookup[-1], bw=kernelBandwidth,
            n_levels=nContourLevels, alpha=baseDensityAlpha,
            shade=True, shade_lowest=False, label=' No Stim', legend=False)
        g.ax_joint.scatter(
            subGroup['LD0'], subGroup['LD1'],
            marker=markerLookup[subName],
            s=markerSize,
            linewidth=0, c=np.atleast_2d(palLookup[-1][markerColorIdx]),
            alpha=markerAlpha)
        g.ax_joint.scatter(
            [subGroup['LD0'].mean()], [subGroup['LD1'].mean()], zorder=100,
            s=meanMarkerSize, c=np.atleast_2d(palLookup[-1][markerColorIdx]),
            linewidth=0.1, edgecolor=np.atleast_2d(palLookup[-1][markerEdgeColorIdx]),
            marker=markerLookup[subName])
        sns.kdeplot(
            subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[-1][markerColorIdx],
            linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
            legend=False)
        sns.kdeplot(
            subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[-1][markerColorIdx],
            linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
            vertical=True, legend=False)
    # plot current observations
    testDF = group.query('(amplitudeCatFuzzy>=2)')
    for subName in ['S', 'M', 'L']:
        subGroup = testDF.query('pedalSizeCat==\'{}\''.format(subName))
        statsQuery = '&'.join([
            '((programFuzzy=={})|(programFuzzy==-1))'.format(name),
            '(pedalSizeCat==\'{}\')'.format(subName)
            ])
        statsDF = featuresDF.query(statsQuery)
        baselineDF = posDF.loc[posDF['pedalSizeCat'] == subName, :]
        baseMean = [baselineDF['LD0'].mean(),  baselineDF['LD1'].mean()]
        groupMean = [subGroup['LD0'].mean(),  subGroup['LD1'].mean()]
        displacement = np.sqrt(
            np.sum(
                (np.array(baseMean) - np.array(groupMean))**2
                ))
        statsResults = manova(
            statsDF.loc[:, ['LD0', 'LD1']],
            statsDF.loc[:, 'amplitudeFuzzy']).mv_test()
        pvalue = statsResults.results['x0']['stat'].loc['Hotelling-Lawley trace', 'Pr > F']
        statsSummary = 'size {}: p={:.2e}, d={:.1f}'.format(
            subName, pvalue, displacement)
        print(statsSummary)
        sns.kdeplot(
            subGroup['LD0'], subGroup['LD1'],
            ax=g.ax_joint,
            cmap=cmapLookup[name], bw=kernelBandwidth,
            n_levels=nContourLevels, alpha=densityAlpha,
            shade=True, shade_lowest=False, label=' Stim (0.75% of motor threshold)', legend=False)
        g.ax_joint.scatter(
            subGroup['LD0'], subGroup['LD1'],
            marker=markerLookup[subName],
            s=markerSize, linewidth=0, c=np.atleast_2d(palLookup[name][markerColorIdx]),
            alpha=markerAlpha)
        g.ax_joint.scatter(
            [groupMean[0]], [groupMean[1]], zorder=100,
            s=meanMarkerSize, c=np.atleast_2d(palLookup[name][markerColorIdx]),
            linewidth=0.1, edgecolor=np.atleast_2d(palLookup[name][markerEdgeColorIdx]),
            marker=markerLookup[subName], label=statsSummary)
        sns.kdeplot(
            subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[name][markerColorIdx],
            linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
            legend=False)
        sns.kdeplot(
            subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[name][markerColorIdx],
            linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
            vertical=True, legend=False)
        # draw arrows
        g.ax_joint.annotate(
            "", xy=(groupMean[0], groupMean[1]),
            xytext=(baseMean[0], baseMean[1]),
            arrowprops={
                'edgecolor': palLookup[name][arrowColorIdx],
                'facecolor': palLookup[name][arrowColorIdx],
                'width': .25,
                'headlength': 5,
                'shrink': 0,
                'headwidth': 5}, zorder=101)
    g.ax_joint.set(yticks=[0, 2])
    g.ax_joint.set_xlim([i - .5 for i in LDBounds])
    g.ax_joint.set_ylim(LDBounds)
    g.ax_joint.set_xlabel('Linear Discriminant Axis (a.u.)')
    g.ax_joint.set_ylabel('Linear Discriminant Axis (a.u.)')
    g.ax_joint.set(xticks=[0, 2])
    g.ax_joint.tick_params(axis='both', which='both', length=5)
    g.ax_marg_x.tick_params(axis='both', which='both', color='w')
    g.ax_marg_y.tick_params(axis='both', which='both', color='w')
    sns.despine(trim=True)
    # Improve the legend 
    handles, labels = g.ax_joint.get_legend_handles_labels()
    keepIdx = [0, 6, 8, 11, 14]
    keepHandles = [handles[i] for i in keepIdx]
    keepLabels = [labels[i] for i in keepIdx]
    lgd = g.ax_joint.legend(
        keepHandles, keepLabels, title="LDA projection",
        handletextpad=0, bbox_to_anchor=(1.25, 0.5), loc="center left",
        borderaxespad=0)
    fig = plt.gcf()
    fig.suptitle(prgLookup[name])
    plt.savefig(
        os.path.join(figureFolder, 'motionStim_LDA_{}.pdf'.format(name)),
        bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

######################
######################

densityAlpha = 0.2
baseDensityAlpha = 0.4
g = sns.JointGrid(x='LD0', y='LD1', data=posDF)
for subName in ['S', 'M', 'L']:
    subGroup = posDF.query('pedalSizeCat==\'{}\''.format(subName))
    sns.kdeplot(
        subGroup['LD0'], subGroup['LD1'],
        ax=g.ax_joint,
        cmap=cmapLookup[-1], bw=kernelBandwidth,
        n_levels=nContourLevels, alpha=baseDensityAlpha,
        shade=True, shade_lowest=False, label=' No Stim', legend=False)
    '''
    g.ax_joint.scatter(
        subGroup['LD0'], subGroup['LD1'],
        marker=markerLookup[subName],
        s=15,
        linewidth=0, c=np.atleast_2d(palLookup[-1][markerColorIdx]), alpha=markerAlpha)
    '''
    g.ax_joint.scatter(
        [subGroup['LD0'].mean()], [subGroup['LD1'].mean()], zorder=100,
        s=meanMarkerSize, c=np.atleast_2d(palLookup[-1][markerColorIdx]),
        linewidth=0.1, edgecolor=np.atleast_2d(palLookup[-1][markerEdgeColorIdx]),
        marker=markerLookup[subName])
    sns.kdeplot(
        subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[-1][markerColorIdx],
        linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
        legend=False)
    sns.kdeplot(
        subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[-1][markerColorIdx],
        linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
        vertical=True, legend=False)

for name, group in featuresDF.groupby('programFuzzy'):
    if name == -1:
        # skip prg -1
        continue
    # plot current observations
    testDF = group.query('(amplitudeCatFuzzy>=2)')
    for subName in ['S', 'M', 'L']:
        subGroup = testDF.query('pedalSizeCat==\'{}\''.format(subName))
        baselineDF = posDF.loc[posDF['pedalSizeCat'] == subName, :]
        baseMean = [baselineDF['LD0'].mean(),  baselineDF['LD1'].mean()]
        groupMean = [subGroup['LD0'].mean(),  subGroup['LD1'].mean()]
        sns.kdeplot(
            subGroup['LD0'], subGroup['LD1'],
            ax=g.ax_joint,
            cmap=cmapLookup[name], bw=kernelBandwidth,
            n_levels=nContourLevels, alpha=densityAlpha,
            shade=True, shade_lowest=False, legend=False, label=' ' + prgLookup[name])
        g.ax_joint.scatter(
            [groupMean[0]], [groupMean[1]], zorder=100,
            s=meanMarkerSize, c=np.atleast_2d(palLookup[name][markerColorIdx]),
            linewidth=0.1, edgecolor=np.atleast_2d(palLookup[name][markerEdgeColorIdx]),
            marker=markerLookup[subName], label=statsSummary)
        sns.kdeplot(
            subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[name][markerColorIdx],
            linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
            legend=False)
        sns.kdeplot(
            subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[name][markerColorIdx],
            linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
            vertical=True, legend=False)
        # draw arrows
        g.ax_joint.annotate(
            "", xy=(groupMean[0], groupMean[1]),
            xytext=(baseMean[0], baseMean[1]),
            arrowprops={
                'edgecolor': palLookup[name][arrowColorIdx],
                'facecolor': palLookup[name][arrowColorIdx],
                'width': .25,
                'headlength': 3,
                'shrink': 0,
                'headwidth': 3}, zorder=101)

g.ax_joint.set(yticks=[0, 2])
g.ax_joint.set_xlim([i - 0.5 for i in LDBounds])
g.ax_joint.set_ylim(LDBounds)
g.ax_joint.set_xlabel('Linear Discriminant Axis (a.u.)')
g.ax_joint.set_ylabel('Linear Discriminant Axis (a.u.)')
g.ax_joint.set(xticks=[0, 2])
g.ax_joint.tick_params(axis='both', which='both', length=5)
g.ax_marg_x.tick_params(axis='both', which='both', color='w')
g.ax_marg_y.tick_params(axis='both', which='both', color='w')
sns.despine(trim=True)
# Improve the legend 
handles, labels = g.ax_joint.get_legend_handles_labels()
keepIdx = [0, 3, 9, 15]
keepHandles = [handles[i] for i in keepIdx]
keepLabels = [labels[i] for i in keepIdx]
lgd = g.ax_joint.legend(
    keepHandles, keepLabels, title="LDA projection",
    handletextpad=0, bbox_to_anchor=(1.25, 0.5), loc="center left",
    borderaxespad=0)
plt.savefig(
    os.path.join(figureFolder, 'motionStim_LDA.pdf'.format(name)),
    bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
