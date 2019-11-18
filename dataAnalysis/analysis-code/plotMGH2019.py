"""
Usage:
    plotMGH2019.py [options]

Options:
    --trialIdx=trialIdx                          which trial to analyze [default: 1]
    --exp=exp                                    which experimental day to analyze [default: exp201901211000]
    --processAll                                 process entire experimental day? [default: False]
    --lazy                                       load from raw, or regular? [default: False]
    --window=window                              process with short window? [default: long]
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --unitQuery=unitQuery                        how to select channels if not supplying a list? [default: fr_sqrt]
    --inputBlockName=inputBlockName              filename for input block [default: fr_sqrt]
    --alignQuery=alignQuery                      query what the units will be aligned to? [default: midPeak]
    --selector=selector                          filename if using a unit selector
    --estimatorName=estimatorName                filename for resulting estimator [default: pca]
    --verbose                                    print diagnostics? [default: False]
    --plotting                                   plot out the correlation matrix? [default: True]
"""
#  The text block above is used by the docopt package to parse command line arguments
#  e.g. you can call <python3 calcTrialSimilarityMatrix.py> to run with default arguments
#  but you can also specify, for instance <python3 calcTrialSimilarityMatrix.py --trialIdx=2>
#  to load trial 002 instead of trial001 which is the default
#
#  regular package imports
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')  # generate postscript output 
matplotlib.use('QT5Agg')  # generate postscript output
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")
import matplotlib.ticker as ticker
import os, pdb, traceback
from importlib import reload
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pandas as pd
import quantities as pq
from statsmodels.multivariate.manova import MANOVA as manova
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import elephant as elph
import dill as pickle
import joblib as jb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#   you can specify options related to the analysis via the command line arguments,
#   or by saving variables in the currentExperiment.py file, or the individual exp2019xxxxxxxx.py files
#
#   these lines process the command line arguments
#   they produces a python dictionary called arguments
from namedQueries import namedQueries
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
#  this stuff imports variables from the currentExperiment.py and exp2019xxxxxxxx.py files
from currentExperiment import parseAnalysisOptions
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#   once these lines run, your workspace should contain a bunch of variables that were imported
#   you can check them by calling the python functions globals() for global variables and locals() for local variables
#   both of these functions return python dictionaries
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
#
triggeredPath = os.path.join(
    analysisSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
fullEstimatorName = '{}_{}_{}_{}'.format(
    prefix,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])
estimatorPath = os.path.join(
    analysisSubFolder,
    fullEstimatorName + '.joblib')
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, analysisSubFolder, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=False,
    getMetaData=[
        'RateInHz', 'activeGroup', 'amplitude', 'amplitudeCat',
        'bin', 'electrode', 'pedalDirection', 'pedalMetaCat',
        'pedalMovementCat', 'pedalMovementDuration',
        'pedalSize', 'pedalSizeCat', 'pedalVelocityCat',
        'program', 'segment', 't'],
    decimate=10,
    metaDataToCategories=False,
    verbose=False, procFun=None))
#
reloadFeatures = True
reloadEstimator = True
reducerOpts = {
    'reducerClass': PCA,
    'reducerKWargs': {
        'n_components': None},
    'secondReducerClass': PCA,
    'secondReducerKWargs': {
        'n_components': None},
    'visReducerClass': PCA,
    'visReducerKWargs': {
        'n_components': 2},
    'classifierClass': LDA,
    'classifierKWargs': {}
        }
metaDataPath = triggeredPath.replace(
    '.nix', '_{}_projection_meta.pickle'.format(arguments['estimatorName']))
projectionH5Path = triggeredPath.replace(
    '.nix', '_{}_projection.h5'.format(arguments['estimatorName']))
#
if os.path.exists(metaDataPath):
    with open(metaDataPath, 'rb') as f:
        metaData = pickle.load(f)
    sameFeatures = (metaData['alignedAsigsKWargs'] == alignedAsigsKWargs)
    sameRedOpts = (metaData['reducerOpts'] == reducerOpts)
    if sameFeatures:
        print('Reusing features matrix')
        reloadFeatures = False
        frDF = pd.read_hdf(projectionH5Path, 'rawFeatures')
    if sameRedOpts:
        print('Reusing estimator')
        # reloadEstimator = False
        estimator = jb.load(estimatorPath)
#
if reloadFeatures or reloadEstimator:
    globals().update(reducerOpts)

if reloadFeatures:
    if arguments['verbose']:
        print('Loading dataBlock: {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    if arguments['verbose']:
        print('Loading alignedAsigs: {}'.format(triggeredPath))
    frDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    # reject outliers
    from scipy.stats import zscore
    for name, group in frDF.apply(zscore).groupby(['segment', 'originalIndex']):
        if (group.abs() > 10).any().sum() > 0:
            frDF.drop(index=group.index, inplace=True)
            print('Dropping trial {}'.format(name[1]))
    frDF.to_hdf(projectionH5Path, 'rawFeatures', mode='w')

metaData = {
    'alignedAsigsKWargs': alignedAsigsKWargs,
    'reducerOpts': reducerOpts
}
with open(metaDataPath, 'wb') as f:
    pickle.dump(metaData, f)

estimator = reducerOpts['reducerClass'](**reducerOpts['reducerKWargs'])
noStimMask = frDF.index.get_level_values('amplitude') == 0
noStimDF = frDF.loc[noStimMask, :]
estimator.fit(noStimDF)
jb.dump(estimator, estimatorPath)

if 'explained_variance_ratio_' in dir(estimator):
    plt.plot(np.cumsum(estimator.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('Proportion of explained variance')
    plt.title('PCA')
    plt.show()

featDF = pd.DataFrame(
    estimator.transform(frDF),
    index=frDF.index).unstack(level='bin')
noStimFeatMask = featDF.index.get_level_values('amplitude') == 0
#
noStimFeatDF = featDF.loc[noStimFeatMask, :]
targetDF = featDF.index.to_frame().reset_index(drop=True)
targetSer = (
    targetDF[['pedalSizeCat', 'pedalDirection']]
    .apply(lambda x: ' '.join(x), axis=1))
if 'n_components' in reducerOpts['secondReducerKWargs']:
    reducerOpts['secondReducerKWargs']['n_components'] = int(noStimFeatMask.sum()/2)
if True:
    # simplify size names
    targetSer = (
        targetSer
        .apply(lambda x: x.replace('XL', 'L'))
        .apply(lambda x: x.replace('XS', 'S'))
        )
    classes = np.asarray([
        'L CCW', 'M CCW', 'S CCW',
        'S CW', 'M CW', 'L CW'])
else:
    classes = np.asarray([
        'XL CCW', 'L CCW', 'M CCW', 'S CCW', 'XS CCW',
        'XS CW', 'S CW', 'M CW', 'L CW', 'XL CW'])
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
targetLB = pd.DataFrame(lb.fit_transform(targetSer), index=targetSer.index)

from sklearn.model_selection import cross_val_score, cross_val_predict
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

RANDOM_STATE = 42
classifier = make_pipeline(
    RandomOverSampler(random_state=RANDOM_STATE),  
    reducerOpts['secondReducerClass'](**reducerOpts['secondReducerKWargs']),
    reducerOpts['classifierClass'](**reducerOpts['classifierKWargs'])
    )
#  movement classification
scores = cross_val_score(
    classifier, featDF.loc[noStimFeatMask, :],
    y=targetSer.loc[noStimFeatMask],
    cv=3, scoring='balanced_accuracy')
ypred = cross_val_predict(
    classifier, featDF.loc[noStimFeatMask, :],
    y=targetSer.loc[noStimFeatMask], cv=3)
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

classifier.fit(
    featDF.loc[noStimFeatMask, :],
    y=targetSer.loc[noStimFeatMask])
visRed = reducerOpts['visReducerClass'](**reducerOpts['visReducerKWargs'])
if 'transform' in dir(classifier):
    visDF = pd.DataFrame(
        visRed.fit_transform(classifier.transform(featDF)),
        index=featDF.index)
else:
    pass
visDF.columns = ['LD{}'.format(i) for i in visDF.columns]
visDF.reset_index(inplace=True)
visDF['target'] = targetSer.to_numpy()
visDF.loc[visDF['amplitude'] == 0, 'program'] = -1
pDataQuery = '&'.join([
    '(amplitude==0)'
])
visDFNoStim = visDF.query('(amplitude==0)')
#  pltData = visDF.reset_index()#.query(pDataQuery)
#  pltData['pedalSizeAbs'] = pltData['pedalSize'].abs() * 100 
#  _, colorBins = pd.cut(
#      pltDataPC['pedalSize'], 128, labels=False, retbins=True)
#  colorData = pd.cut(
#      pltDataPC['pedalSize'], colorBins, labels=False)
#  cubeHelix = sns.cubehelix_palette(128)
#  plotColors = np.asarray([cubeHelix[i] for i in colorData])
#  styleLookup = {
#      'CW': 'o',
#      'CCW': '+'
#  }
#  fig = plt.figure(figsize=(8, 6))
#  ax = fig.add_subplot(111, projection='3d')
#  for k, v in styleLookup.items():
#      dataMask = pltDataPC['pedalDirection'] == k
#      ax.scatter(
#          pltDataPC.loc[dataMask, 0], pltDataPC.loc[dataMask, 1],
#          pltDataPC.loc[dataMask, 2],
#          s=pltDataPC.loc[dataMask, 'pedalSize'].abs() * 50, c=plotColors[dataMask],
#          marker=v,
#          alpha=0.6, edgecolors='w')
#  plt.show()

# sns.scatterplot(
#     x='LD0', y='LD1', hue='pedalSizeAbs', size='pedalSizeAbs',
#     style='pedalDirection',
#     data=pltData)
# plt.show()

meanMarkerAlpha = 1
meanMarkerSize = 60
markerSize = 30
nContourLevels = 5
markerColorIdx = int(3 * nContourLevels * 0.7)
markerEdgeColorIdx = int(3 * nContourLevels * 0.1)
arrowColorIdx = int(3 * nContourLevels * 0.3)
kernelBandwidth = 'scott'
LDBounds = [-9.5, 9.5]
markerLookup = {
    'XS CW': 'P', 'S CW': 'P', 'M CW': 'o', 'L CW': 'D', 'XL CW': 'D',
    'XS CCW': 'X', 'S CCW': 'X', 'M CCW': '*', 'L CCW': 's', 'XL CCW': 's'}
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
reverseColorMap = True
cmapLookup = {
    -1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[3], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap, as_cmap=True),
    0: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[0], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap, as_cmap=True),
    1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[1], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap, as_cmap=True),
    2: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[2], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap, as_cmap=True)}
cmapLookupConfMat = {
    -1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[3], rot=colorRotation, light=maxLight, dark=minDark, reverse=False, as_cmap=True),
    0: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[0], rot=colorRotation, light=maxLight, dark=minDark, reverse=False, as_cmap=True),
    1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[1], rot=colorRotation, light=maxLight, dark=minDark, reverse=False, as_cmap=True),
    2: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[2], rot=colorRotation, light=maxLight, dark=minDark, reverse=False, as_cmap=True)}
palLookup = {
    -1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[3], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap),
    0: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[0], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap),
    1: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[1], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap),
    2: sns.cubehelix_palette(
    n_colors=3 * nContourLevels, start=colorWheel[2], rot=colorRotation, light=maxLight, dark=minDark, reverse=reverseColorMap)}
#
prgLookup = {
    0: 'Caudal',
    1: 'Rostral',
    2: 'Midline'
    }
#
densityAlpha = 0.2
baseDensityAlpha = 0.3
densityAlphaMarg = 0.5
baseDensityAlphaMarg = 0.7
markerAlpha = 0.7
# plot motion confusion matrix
np.set_printoptions(precision=1)
cm = confusion_matrix(
    targetSer.loc[noStimFeatMask], ypred, labels=classes)
#  classes = unique_labels(targetSer.loc[noStimFeatMask], ypred)
hf.plotConfusionMatrix(
    cm, classes,
    normalize=True,
    title='Confusion matrix',
    cmap=cmapLookupConfMat[-1])
plt.savefig(
        os.path.join(figureFolder, 'confmat_LDA.pdf'))
plt.close()
# plt.show()
### Plot motion LDA
g = sns.JointGrid(x='LD0', y='LD1', data=visDFNoStim)
# plot the no stim
for subName in classes:
    subGroup = visDFNoStim.query('target==\'{}\''.format(subName))
    try:
        sns.kdeplot(
            subGroup['LD0'], subGroup['LD1'],
            ax=g.ax_joint,
            cmap=cmapLookup[-1], bw=kernelBandwidth,
            n_levels=nContourLevels, alpha=baseDensityAlpha,
            shade=True, shade_lowest=False, label=' No Stim', legend=False)
    except Exception:
        pass
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
    try:
        sns.kdeplot(
            subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[-1][markerColorIdx],
            linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
            legend=False)
        sns.kdeplot(
            subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[-1][markerColorIdx],
            linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
            vertical=True, legend=False)
    except Exception:
        pass

g.ax_joint.set(yticks=[0, 2])
# g.ax_joint.set_xlim([i - .5 for i in LDBounds])
# g.ax_joint.set_ylim(LDBounds)
g.ax_joint.set_xlabel('Linear Discriminant Axis (a.u.)')
g.ax_joint.set_ylabel('Linear Discriminant Axis (a.u.)')
g.ax_joint.set(xticks=[0, 2])
g.ax_joint.tick_params(axis='both', which='both', length=5)
g.ax_marg_x.tick_params(axis='both', which='both', color='w')
g.ax_marg_y.tick_params(axis='both', which='both', color='w')
sns.despine(trim=True)
# Improve the legend 
handles, labels = g.ax_joint.get_legend_handles_labels()
# keepIdx = [0, 6, 8, 11, 14]
keepIdx = [i for i in range(len(handles))]
keepHandles = [handles[i] for i in keepIdx]
keepLabels = [labels[i] for i in keepIdx]
lgd = g.ax_joint.legend(
    keepHandles, keepLabels, title="LDA projection",
    handletextpad=0, bbox_to_anchor=(1.25, 0.5), loc="center left",
    borderaxespad=0)
fig = plt.gcf()
fig.suptitle('Movement Linear Discriminant Projection')
plt.savefig(
    os.path.join(figureFolder, 'motion_LDA.pdf'),
    bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()
plt.close()

### Plot stim overlays
for name, group in visDF.groupby('program'):
    sns.palplot(palLookup[name])
    plt.savefig(
        os.path.join(figureFolder, 'program_{}_colorref.pdf'.format(name)),
        bbox_inches='tight')
    plt.close()
    if name == -1:
        # skip prg -1
        continue
    cm = confusion_matrix(
        targetSer.loc[group.index],
        classifier.predict(featDF.iloc[group.index, :]),
        labels=classes)
    hf.plotConfusionMatrix(
        cm, classes,
        normalize=True,
        title='Confusion matrix',
        cmap=cmapLookupConfMat[name])
    plt.savefig(
        os.path.join(figureFolder, 'confmat_LDA_{}.pdf'.format(name)))
    plt.close()
    g = sns.JointGrid(x='LD0', y='LD1', data=visDFNoStim)
    # plot the no stim
    for subName in classes:
        subGroup = visDFNoStim.query('target==\'{}\''.format(subName))
        try:
            sns.kdeplot(
                subGroup['LD0'], subGroup['LD1'],
                ax=g.ax_joint,
                cmap=cmapLookup[-1], bw=kernelBandwidth,
                n_levels=nContourLevels, alpha=baseDensityAlpha,
                shade=True, shade_lowest=False, label=' No Stim', legend=False)
        except Exception:
            pass
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
        try:
            sns.kdeplot(
                subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[-1][markerColorIdx],
                linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
                legend=False)
            sns.kdeplot(
                subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[-1][markerColorIdx],
                linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
                vertical=True, legend=False)
        except Exception:
            pass
    # plot current observations
    testDF = group.query('(amplitudeCat>=2)')
    for subName in classes:
        subGroup = testDF.query('target==\'{}\''.format(subName))
        statsQuery = '&'.join([
            '((program=={})|(program==-1))'.format(name),
            '(target==\'{}\')'.format(subName)
            ])
        statsDF = visDF.query(statsQuery)
        baselineDF = visDFNoStim.loc[visDFNoStim['target'] == subName, :]
        baseMean = [baselineDF['LD0'].mean(),  baselineDF['LD1'].mean()]
        groupMean = [subGroup['LD0'].mean(),  subGroup['LD1'].mean()]
        displacement = np.sqrt(
            np.sum(
                (np.array(baseMean) - np.array(groupMean))**2
                ))
        statsResults = manova(
            statsDF.loc[:, ['LD0', 'LD1']],
            statsDF.loc[:, 'amplitude']).mv_test()
        pvalue = statsResults.results['x0']['stat'].loc['Hotelling-Lawley trace', 'Pr > F']
        statsSummary = 'size {}: p={:.2e}, d={:.1f}'.format(
            subName, pvalue, displacement)
        print(statsSummary)
        try:
            sns.kdeplot(
                subGroup['LD0'], subGroup['LD1'],
                ax=g.ax_joint,
                cmap=cmapLookup[name], bw=kernelBandwidth,
                n_levels=nContourLevels, alpha=densityAlpha,
                shade=True, shade_lowest=False, label=' Stim (0.75% of motor threshold)', legend=False)
        except Exception:
            pass
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
        try:
            sns.kdeplot(
                subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[name][markerColorIdx],
                linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
                legend=False)
            sns.kdeplot(
                subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[name][markerColorIdx],
                linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
                vertical=True, legend=False)
        except Exception:
            pass
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
    # g.ax_joint.set_xlim([i - .5 for i in LDBounds])
    # g.ax_joint.set_ylim(LDBounds)
    g.ax_joint.set_xlabel('Linear Discriminant Axis (a.u.)')
    g.ax_joint.set_ylabel('Linear Discriminant Axis (a.u.)')
    g.ax_joint.set(xticks=[0, 2])
    g.ax_joint.tick_params(axis='both', which='both', length=5)
    g.ax_marg_x.tick_params(axis='both', which='both', color='w')
    g.ax_marg_y.tick_params(axis='both', which='both', color='w')
    sns.despine(trim=True)
    # Improve the legend 
    handles, labels = g.ax_joint.get_legend_handles_labels()
    # keepIdx = [0, 6, 8, 11, 14]
    keepIdx = [i for i in range(len(handles))]
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
    # plt.show()
    plt.close()

######################
######################

densityAlpha = 0.2
baseDensityAlpha = 0.4
g = sns.JointGrid(x='LD0', y='LD1', data=visDFNoStim)
for subName in classes:
    subGroup = visDFNoStim.query('target==\'{}\''.format(subName))
    try:
        sns.kdeplot(
            subGroup['LD0'], subGroup['LD1'],
            ax=g.ax_joint,
            cmap=cmapLookup[-1], bw=kernelBandwidth,
            n_levels=nContourLevels, alpha=baseDensityAlpha,
            shade=True, shade_lowest=False, label=' No Stim', legend=False)
    except Exception:
        pass
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
    try:
        sns.kdeplot(
            subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[-1][markerColorIdx],
            linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
            legend=False)
        sns.kdeplot(
            subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[-1][markerColorIdx],
            linewidth=0, alpha=baseDensityAlphaMarg, shade=True, bw=kernelBandwidth,
            vertical=True, legend=False)
    except Exception:
        pass

for name, group in visDF.groupby('program'):
    if name == -1:
        # skip prg -1
        continue
    # plot current observations
    testDF = group.query('(amplitudeCat>=2)')
    for subName in classes:
        subGroup = testDF.query('target==\'{}\''.format(subName))
        baselineDF = visDFNoStim.loc[visDFNoStim['target'] == subName, :]
        baseMean = [baselineDF['LD0'].mean(),  baselineDF['LD1'].mean()]
        groupMean = [subGroup['LD0'].mean(),  subGroup['LD1'].mean()]
        try:
            sns.kdeplot(
                subGroup['LD0'], subGroup['LD1'],
                ax=g.ax_joint,
                cmap=cmapLookup[name], bw=kernelBandwidth,
                n_levels=nContourLevels, alpha=densityAlpha,
                shade=True, shade_lowest=False, legend=False, label=' ' + prgLookup[name])
        except Exception:
            pass
        g.ax_joint.scatter(
            [groupMean[0]], [groupMean[1]], zorder=100,
            s=meanMarkerSize, c=np.atleast_2d(palLookup[name][markerColorIdx]),
            linewidth=0.1, edgecolor=np.atleast_2d(palLookup[name][markerEdgeColorIdx]),
            marker=markerLookup[subName], label=statsSummary)
        try:
            sns.kdeplot(
                subGroup['LD0'], ax=g.ax_marg_x, color=palLookup[name][markerColorIdx],
                linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
                legend=False)
            sns.kdeplot(
                subGroup['LD1'], ax=g.ax_marg_y, color=palLookup[name][markerColorIdx],
                linewidth=0, alpha=densityAlphaMarg, shade=True, bw=kernelBandwidth,
                vertical=True, legend=False)
        except Exception:
            pass
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
# g.ax_joint.set_xlim([i - 0.5 for i in LDBounds])
# g.ax_joint.set_ylim(LDBounds)
g.ax_joint.set_xlabel('Linear Discriminant Axis (a.u.)')
g.ax_joint.set_ylabel('Linear Discriminant Axis (a.u.)')
g.ax_joint.set(xticks=[0, 2])
g.ax_joint.tick_params(axis='both', which='both', length=5)
g.ax_marg_x.tick_params(axis='both', which='both', color='w')
g.ax_marg_y.tick_params(axis='both', which='both', color='w')
sns.despine(trim=True)
# Improve the legend 
handles, labels = g.ax_joint.get_legend_handles_labels()
# keepIdx = [0, 3, 9, 15]
keepIdx = [i for i in range(len(handles))]
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