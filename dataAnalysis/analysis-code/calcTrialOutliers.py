"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --saveResults                          load from raw, or regular? [default: False]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --verbose                              print diagnostics? [default: False]
    --plotting                             plot results?
    --window=window                        process with short window? [default: long]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: all]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: all]
    --selector=selector                    filename if using a unit selector
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
"""

import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
import pingouin as pg
import pandas as pd
import numpy as np
from copy import deepcopy
from docopt import docopt
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_outliers.h5'.format(
        arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, removeFuzzyName=False,
    decimate=1, windowSize=(-100e-3, 400e-3),
    metaDataToCategories=False,
    transposeToColumns='feature', concatOn='columns',
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)

dataDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
if 'outlierDetectColumns' in locals():
    dataDF.drop(
        columns=[
            cn[0]
            for cn in dataDF.columns
            if cn[0] not in outlierDetectColumns],
        level='feature', inplace=True)
# fix order of magnitude
ordMag = np.floor(np.log10(dataDF.abs().mean().mean()))
if ordMag < 0:
    dataDF = dataDF * 10 ** (-ordMag)
dataDF = dataDF.apply(lambda x: x - x.mean())
outlierLogPath = os.path.join(
    figureFolder,
    prefix + '_{}_outlierTrials.txt'.format(arguments['window']))
if os.path.exists(outlierLogPath):
    os.remove(outlierLogPath)

def findOutliers(
        mahalDistDF, qThresh=None, sdThresh=None,
        nDim=1, multiplier=1):
    if sdThresh is None:
        if qThresh is None:
            qThresh = 1 - 1e-6
        sdThresh = multiplier * chi2.interval(qThresh, nDim)[1]
    seg = (
        mahalDistDF
        .index
        .get_level_values('segment')
        .unique())
    t = (
        mahalDistDF
        .index
        .get_level_values('t')
        .unique())
    # print('Outlier thresh is {}'.format(sdThresh))
    # deviation = (mahalDistDF).quantile(q=0.9)[0]
    deviation = (mahalDistDF).max()[0]
    tooMuch = (deviation >= sdThresh)
    if tooMuch:
        try:
            summaryMessage = [
                'segment {} time {:.2f}\n'.format(seg[0], t[0]),
                'average deviation {:.2f} > {:.2f}\n'.format(
                    deviation, sdThresh)]
            print(summaryMessage)
            with open(outlierLogPath, 'a') as f:
                f.writelines(summaryMessage)
        except Exception:
            pass
    # else:
    #     print(
    #         'segment {} time {} average deviation {} < {}'.format(
    #             seg[0], t[0], deviation, sdThresh))
    return deviation, tooMuch, seg[0], t[0]


def applyMad(ser):
    if np.median(ser) == 0:
        return np.abs(zscore(ser))
    else:
        return np.abs(ser - np.median(ser)) / pg.mad(ser)

testVar = None
groupBy = ['segment', 'originalIndex', 't']
resultNames = [
    'deviation', 'rejectBlock', 'seg', 't']

print('working with {} samples'.format(dataDF.shape[0]))
randSample = slice(None, None, None)
# tBoundsCovCalc = [0, 150e-3]
allEpochs = dataDF.index.get_level_values('bin')
epochs = pd.cut(allEpochs, bins=5)
epochs.name = 'epoch'

dataDF.set_index(pd.Index(epochs, name='epoch'), append=True, inplace=True)
mahalDist = pd.DataFrame(
    np.nan,
    index=dataDF.index, columns=['mahalDist'])
#
groupNames = ['electrode', 'nominalCurrent', 'epoch']
# groupNames = ['electrode']
# groupNames = ['nominalCurrent']
# groupNames = None
if groupNames is not None:
    grouper = dataDF.groupby(groupNames)
else:
    grouper = [('all', dataDF)]

if arguments['verbose']:
    print('Calculating covariance matrix...')
useEmpiricalCovariance = False
for name, group in tqdm(grouper):
    # tBins = group.index.get_level_values('bin')
    # tMask = (tBins >= tBoundsCovCalc[0]) & (tBins <=tBoundsCovCalc[1])
    # subData = group.to_numpy()[tMask, :]
    subData = group.to_numpy()[randSample, :]
    defaultSupport = (
        (subData.shape[0] + subData.shape[1] + 1) /
        (2 * subData.shape[0])
        )
    if not useEmpiricalCovariance:
        supportFraction = None
        # supportFraction = .9
        covMat = MinCovDet(support_fraction=supportFraction).fit(subData)
    else:
        covMat = EmpiricalCovariance().fit(subData)
    mahalDist.loc[group.index, 'mahalDist'] = covMat.mahalanobis(
        group.to_numpy())

outlierTrials = ash.applyFunGrouped(
    mahalDist,
    groupBy, testVar,
    fun=findOutliers, funArgs=[],
    funKWargs=dict(multiplier=4, nDim=len(dataDF.columns)),
    # funKWargs=dict(sdThresh=300),
    resultNames=resultNames,
    plotting=False)

print(outlierTrials['deviation'].sort_values('all').tail())
print('Outlier proportion was:')
print(outlierTrials['rejectBlock']['all'].sum() / outlierTrials['rejectBlock']['all'].size)

if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.plot(mahalDist.to_numpy())
    plt.show()
if arguments['plotting']:
    fig, ax = plt.subplots()
    hist, binEdges = np.histogram(
        outlierTrials['deviation'],
        bins=np.arange(
            0,
            outlierTrials['deviation']['all'].max() + 10,
            10)
        )
    ax.plot(
        binEdges[:-1],
        np.cumsum(hist) / hist.sum())
    ax.set_xlabel('mahalanobis distance')
    plt.show(block=False)
if arguments['plotting']:
    fig, ax = plt.subplots(2, 1)
    sns.boxplot(
        outlierTrials['deviation'],
        ax=ax[0])
    ax[0].set_xlabel(arguments['unitQuery'])
    plt.show(block=False)
if arguments['plotting']:
    theseOutliers = (
        outlierTrials['deviation']
        .loc[
            outlierTrials['rejectBlock']['all'].astype(np.bool),
            'all']).sort_values()
    nRowCol = int(np.ceil(np.sqrt(theseOutliers.size)))
    fig, ax = plt.subplots(
        nRowCol, nRowCol, sharex=True, sharey=True)
    fig.set_size_inches(2 * nRowCol, 3 * nRowCol)
    # for idx, (name, group) in enumerate(dataDF.loc[fullOutMask, :].groupby(theseOutliers.index.names)):
    for idx, (name, row) in enumerate(theseOutliers.items()):
        outlierDataMasks = []
        for lvlIdx, levelName in enumerate(theseOutliers.index.names):
            outlierDataMasks.append(dataDF.index.get_level_values(levelName) == name[lvlIdx])
        fullOutMask = np.logical_and.reduce(outlierDataMasks)
        for cN in dataDF.columns:
            ax.flat[idx].plot(
                dataDF.loc[fullOutMask, :].index.get_level_values('bin'),
                dataDF.loc[fullOutMask, cN], label=cN[0])
            ax.flat[idx].text(
                1, 1, 'dev = {:.2f}'.format(row),
                va='top', ha='right',
                transform=ax.flat[idx].transAxes)
    leg = ax.flat[0].legend(
        bbox_to_anchor=(1.01, 1.01),
        loc='upper left',
        bbox_transform=ax[0, -1].transAxes)
    ax.flat[0].set_ylim([-25, 50])
    pdfName = os.path.join(figureOutputFolder, 'outlier_trials_by_condition_and_epoch_robust.pdf')
    fig.savefig(pdfName, bbox_inches='tight', pad_inches=0, bbox_extra_artists=[leg])
    plt.show()
    outlierDataMasks = []
    for lvlIdx, levelName in enumerate(theseOutliers.index.names):
        outlierDataMasks.append(dataDF.index.get_level_values(levelName).isin(theseOutliers.index.get_level_values(levelName)))
    indexInfo = dataDF.index.to_frame().reset_index(drop=True)
    fullOutMask = np.logical_and.reduce(outlierDataMasks)
    firstBinMask = indexInfo['bin'] == indexInfo['bin'].unique()[0]
    indexInfo.loc[fullOutMask & firstBinMask, :].groupby(['electrode', 'nominalCurrent'])['RateInHz'].value_counts()

if arguments['plotting']:
    fig, ax = plt.subplots(2, 1, sharex=True)
    bla = (mahalDist.xs(992, level='originalIndex').xs(3, level='segment'))
    ax[0].plot(
        bla.index.get_level_values('bin').to_numpy(),
        bla.to_numpy())
    bla = (dataDF.xs(992, level='originalIndex').xs(3, level='segment'))
    ax[1].plot(
        bla.index.get_level_values('bin').to_numpy(),
        bla.to_numpy())
    plt.show()
# 
if arguments['saveResults']:
    if os.path.exists(resultPath):
        os.remove(resultPath)
    outlierTrials['deviation'].to_hdf(
        resultPath, 'deviation', format='fixed')
    outlierTrials['rejectBlock'].to_hdf(
        resultPath, 'rejectBlock', format='fixed')