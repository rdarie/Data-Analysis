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

import pdb, traceback
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
    getMetaData=[
        'RateInHz', 'feature', 'electrode', 'nominalCurrent', 'stimPeriod',
        'stimCat', 'originalIndex', 'segment', 't'],
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
# dataDF = dataDF.apply(lambda x: x - x.mean())

# TODO: cut dataDF into epochs based on RateInHz
# epochs = pd.cut(
#     dataDF.index.get_level_values('bin'),
#     bins=50)
trialInfo = dataDF.index.to_frame().reset_index(drop=True)
epochs = pd.Series(index=trialInfo.index)
# nominal epoch size
targetEpochSize = 2e-3
#  delay to account for transmission between event
#  at t=0 and the signal being recorded
transmissionDelay = 0
for stimPeriod, group in trialInfo.groupby('stimPeriod'):
    # adjust epoch size down from nominal, to capture
    # integer number of stim periods
    if stimPeriod > targetEpochSize:
        epochSize = stimPeriod / np.floor(stimPeriod / targetEpochSize)
    else:
        epochSize = stimPeriod * np.ceil(targetEpochSize / stimPeriod)
    print('stimPeriod = {}, epochSize = {}'.format(stimPeriod, epochSize))
    theseTBins = group['bin'].to_numpy()
    epochBins = np.arange(
        theseTBins.min(), theseTBins.max(), epochSize)
    # align epoch bins to window
    epochOffset = np.max(epochBins[epochBins <= 0])
    epochBins = epochBins - epochOffset + transmissionDelay
    # stretch first and last epoch bin to cover entire window
    epochBins[0] = theseTBins.min() - 1
    epochBins[-1] = theseTBins.max() + 1
    theseEpochs = pd.cut(theseTBins, bins=epochBins, labels=False)
    epochs.loc[group.index] = theseEpochs
# pdb.set_trace()
dataDF.set_index(
    pd.Index(epochs, name='epoch'),
    append=True, inplace=True)

outlierLogPath = os.path.join(
    figureFolder,
    prefix + '_{}_outlierTrials.txt'.format(arguments['window']))
if os.path.exists(outlierLogPath):
    os.remove(outlierLogPath)

def findOutliers(
        mahalDistDF, qThresh=None, sdThresh=None,
        devQuantile=None,
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
    if devQuantile is not None:
        deviation = (mahalDistDF).quantile(q=devQuantile)[0]
    else:
        deviation = (mahalDistDF).max()[0]
    tooMuch = (deviation >= sdThresh)
    if tooMuch:
        try:
            summaryMessage = [
                'segment {} time {:.2f}\n'.format(seg[0], t[0]),
                ' deviation {:.2f} > {:.2f}\n'.format(
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
groupBy = ['segment', 't']
resultNames = [
    'deviation', 'rejectBlock', 'seg', 't']

print('working with {} samples'.format(dataDF.shape[0]))
randSample = slice(None, None, None)
# tBoundsCovCalc = [0, 150e-3]
#
groupNames = ['electrode', 'nominalCurrent', 'RateInHz', 'epoch']
# groupNames = None
if groupNames is not None:
    grouper = dataDF.groupby(groupNames)
else:
    grouper = [('all', dataDF)]

useCachedMahalDist = True
if useCachedMahalDist and os.path.exists(resultPath):
    mahalDist = pd.read_hdf(
        resultPath, 'mahalDist')
    mahalDistLoaded = True
else:
    mahalDistLoaded = False

useEmpiricalCovariance = True
# def calcCovMat(partition):
# 
if not mahalDistLoaded:
    mahalDist = pd.DataFrame(
        np.nan,
        index=dataDF.index, columns=['mahalDist'])
    if arguments['verbose']:
        print('Calculating covariance matrix...')
    
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
            # supportFraction = None
            supportFraction = .9
            try:
                covMat = MinCovDet(support_fraction=supportFraction).fit(subData)
            except Exception:
                traceback.print_exc()
                pdb.set_trace()
                covMat = EmpiricalCovariance().fit(subData)
        else:
            covMat = EmpiricalCovariance().fit(subData)
        mahalDist.loc[group.index, 'mahalDist'] = covMat.mahalanobis(
            group.to_numpy())
#
print('#######################################################')
refInterval = chi2.interval(1 - 1e-2, len(dataDF.columns))
print('Data is {} dimensional'.format(len(dataDF.columns)))
print('The mahalanobis distance should lie within {}'.format(refInterval))
print('#######################################################')

outlierTrials = ash.applyFunGrouped(
    mahalDist,
    groupBy, testVar,
    fun=findOutliers, funArgs=[],
    funKWargs=dict(
        multiplier=1, qThresh=1-1e-2,
        nDim=len(dataDF.columns), devQuantile=0.95),
    # funKWargs=dict(sdThresh=100),
    resultNames=resultNames,
    plotting=False)

print(outlierTrials['deviation'].sort_values('all').tail())
print('Outlier proportion was:')
print(outlierTrials['rejectBlock']['all'].sum() / outlierTrials['rejectBlock']['all'].size)

# if arguments['plotting']:
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     plt.plot(mahalDist.to_numpy())
#     plt.show()
if arguments['plotting']:
    binSize = 1
    hist, binEdges = np.histogram(
        outlierTrials['deviation'],
        bins=np.arange(
            0,
            outlierTrials['deviation']['all'].max() + binSize,
            binSize)
        )
    cumFrac = np.cumsum(hist) / hist.sum()
    mask = (cumFrac > 0.5) & (cumFrac < 0.99)
    rv = chi2(len(dataDF.columns))
    theoreticalPMF = rv.pdf(binEdges[:-1] * binSize)
    # plt.plot(binEdges[:-1][mask], theoreticalPMF[mask]); plt.show()
    #
    fig, ax = plt.subplots(1, 2, sharex=True)
    ax[0].plot(
        binEdges[:-1][mask],
        cumFrac[mask]
        )
    ax[0].set_xlabel('mahalanobis distance')
    ax[0].set_ylabel('cummulative fraction')
    # sns.distplot(
    #     outlierTrials['deviation'],
    #     bins=binEdges, kde=False, ax=ax[1])
    ax[1].plot(
        binEdges[:-1][mask],
        hist[mask] / hist.sum()
        )
    ax[1].plot(binEdges[:-1][mask], theoreticalPMF[mask])
    ax[1].set_xlabel('mahalanobis distance')
    ax[1].set_ylabel('fraction')
    pdfName = os.path.join(
        figureOutputFolder,
        'mh_dist_histogram_by_condition_and_epoch_robust.pdf')
    plt.savefig(
        pdfName, bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()
#
# if arguments['plotting']:
#     fig, ax = plt.subplots(2, 1)
#     sns.boxplot(
#         outlierTrials['deviation'],
#         ax=ax[0])
#     ax[0].set_xlabel(arguments['unitQuery'])
#     plt.show(block=False)
# #
theseOutliers = (
    outlierTrials['deviation']
    .loc[
        outlierTrials['rejectBlock']['all'].astype(np.bool),
        'all']).sort_values()
maxDroppedTrials = pd.Series(
    index=np.concatenate(
        [
            [theseOutliers.min()],
            np.linspace(
                theseOutliers.min() / 3,
                3 * theseOutliers.min(), 10)]
        ))
firstBinMask = trialInfo['bin'] == trialInfo['bin'].unique()[0]
for ix, devThreshold in enumerate(maxDroppedTrials.index):
    print(ix)
    if (theseOutliers >= devThreshold).any():
        outlierDataMasks = []
        for lvlIdx, levelName in enumerate(theseOutliers.index.names):
            outlierDataMasks.append(trialInfo[levelName].isin(theseOutliers.loc[theseOutliers >= devThreshold].index.get_level_values(levelName)))
        fullOutMask = np.logical_and.reduce(outlierDataMasks)
        nOutliersPerCondition = (
            trialInfo
            .loc[fullOutMask & firstBinMask, :]
            .groupby(['electrode', 'nominalCurrent'])['RateInHz']
            .value_counts())
        if ix == 0:
            saveNOutliers = nOutliersPerCondition
        maxDroppedTrials[devThreshold] = nOutliersPerCondition.max()

print(maxDroppedTrials)
print(saveNOutliers.sort_values())

if arguments['plotting']:
    nRowCol = int(np.ceil(np.sqrt(theseOutliers.size)))
    emgFig, emgAx = plt.subplots(
        nRowCol, nRowCol, sharex=True)
    emgFig.set_size_inches(5 * nRowCol, 3 * nRowCol)
    mhFig, mhAx = plt.subplots(
        nRowCol, nRowCol, sharex=True)
    mhFig.set_size_inches(5 * nRowCol, 3 * nRowCol)
    # for idx, (name, group) in enumerate(dataDF.loc[fullOutMask, :].groupby(theseOutliers.index.names)):
    for idx, (name, row) in enumerate(theseOutliers.items()):
        outlierDataMasks = []
        for lvlIdx, levelName in enumerate(theseOutliers.index.names):
            outlierDataMasks.append(dataDF.index.get_level_values(levelName) == name[lvlIdx])
        fullOutMask = np.logical_and.reduce(outlierDataMasks)
        for cN in dataDF.columns:
            emgAx.flat[idx].plot(
                dataDF.loc[fullOutMask, :].index.get_level_values('bin'),
                dataDF.loc[fullOutMask, cN], alpha=0.8, label=cN[0])
            emgAx.flat[idx].text(
                1, 1, 'dev = {:.2f}'.format(row),
                va='top', ha='right',
                transform=emgAx.flat[idx].transAxes)
        mhAx.flat[idx].plot(
            mahalDist.loc[fullOutMask, :].index.get_level_values('bin'),
            mahalDist.loc[fullOutMask, 'mahalDist'], label='mahalDist')
        mhAx.flat[idx].text(
            1, 1, 'dev = {:.2f}'.format(row),
            va='top', ha='right',
            transform=mhAx.flat[idx].transAxes)
    emgLeg = emgAx.flat[0].legend(
        bbox_to_anchor=(1.01, 1.01),
        loc='upper left',
        bbox_transform=emgAx[0, -1].transAxes)
    mhLeg = mhAx.flat[0].legend(
        bbox_to_anchor=(1.01, 1.01),
        loc='upper left',
        bbox_transform=mhAx[0, -1].transAxes)
    # emgAx.flat[0].set_ylim([-25, 50])
    pdfName = os.path.join(
        figureOutputFolder,
        'outlier_trials_by_condition_and_epoch_robust.pdf')
    emgFig.savefig(
        pdfName,
        bbox_inches='tight', pad_inches=0, bbox_extra_artists=[emgLeg])
    pdfName = os.path.join(
        figureOutputFolder,
        'mh_dist_by_condition_and_epoch_robust.pdf')
    mhFig.savefig(
        pdfName,
        bbox_inches='tight',
        pad_inches=0, bbox_extra_artists=[mhLeg])
    plt.close()
    # plt.show()
#
# if arguments['plotting']:
#     fig, ax = plt.subplots(2, 1, sharex=True)
#     bla = (mahalDist.xs(992, level='originalIndex').xs(3, level='segment'))
#     ax[0].plot(
#         bla.index.get_level_values('bin').to_numpy(),
#         bla.to_numpy())
#     bla = (dataDF.xs(992, level='originalIndex').xs(3, level='segment'))
#     ax[1].plot(
#         bla.index.get_level_values('bin').to_numpy(),
#         bla.to_numpy())
#     plt.show()
# 
if arguments['saveResults']:
    if os.path.exists(resultPath):
        os.remove(resultPath)
    outlierTrials['deviation'].to_hdf(
        resultPath, 'deviation', format='fixed')
    outlierTrials['rejectBlock'].to_hdf(
        resultPath, 'rejectBlock', format='fixed')
    mahalDist.to_hdf(
        resultPath, 'mahalDist', format='fixed')

# pdb.set_trace()
minNObservations = 5
firstBinTrialInfo = trialInfo.loc[firstBinMask, :]
goodTrialInfo = firstBinTrialInfo.loc[~outlierTrials['rejectBlock'].to_numpy().flatten().astype(np.bool), :]
goodTrialCount = goodTrialInfo.groupby(['electrode', 'nominalCurrent'])['RateInHz'].value_counts().to_frame(name='count').reset_index()
goodTrialCount = goodTrialCount.loc[goodTrialCount['count'] > minNObservations, :]
goodTrialCount.to_csv(os.path.join(figureOutputFolder, 'good_trial_breakdown.csv'))
goodTrialCount.groupby(['electrode', 'RateInHz', 'nominalCurrent']).ngroups
badTrialInfo = firstBinTrialInfo.loc[outlierTrials['rejectBlock'].to_numpy().flatten().astype(np.bool), :]
badTrialCount = badTrialInfo.groupby(['electrode', 'nominalCurrent'])['RateInHz'].value_counts().sort_values().to_frame(name='count').reset_index()
outlierTrials['deviation'].reset_index().sort_values(['segment', 'all']).to_csv(os.path.join(figureOutputFolder, 'trial_deviation_breakdown.csv'))
print('Bad trial count:\n{}'.format(badTrialCount))

# .to_csv(os.path.join(figureOutputFolder, 'bad_trial_breakdown.csv'))
