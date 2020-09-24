"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                    which experimental day to analyze
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --processAll                                 process entire experimental day? [default: False]
    --lazy                                       load from raw, or regular? [default: False]
    --saveResults                                load from raw, or regular? [default: False]
    --inputBlockName=inputBlockName              which trig_ block to pull [default: pca]
    --verbose                                    print diagnostics? [default: False]
    --plotting                                   plot results?
    --window=window                              process with short window? [default: long]
    --unitQuery=unitQuery                        how to restrict channels if not supplying a list? [default: all]
    --alignQuery=alignQuery                      query what the units will be aligned to? [default: all]
    --selector=selector                          filename if using a unit selector
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --amplitudeFieldName=amplitudeFieldName      what is the amplitude named? [default: nominalCurrent]
    --sqrtTransform                              for firing rates, whether to take the sqrt to stabilize variance [default: False]
"""

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("notebook")
    sns.set_style("darkgrid")
import pdb, traceback
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
# import pingouin as pg
import pandas as pd
import numpy as np
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from copy import deepcopy
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['plotting']:
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

calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
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
    calcSubFolder,
    prefix + '_{}_outliers.h5'.format(
        arguments['window']))

outlierLogPath = os.path.join(
    figureFolder,
    prefix + '_{}_outlierTrials.txt'.format(arguments['window']))
if os.path.exists(outlierLogPath):
    os.remove(outlierLogPath)

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, removeFuzzyName=False,
    decimate=1,
    metaDataToCategories=False,
    getMetaData=[
        'RateInHz', 'feature', 'electrode',
        arguments['amplitudeFieldName'], 'stimPeriod',
        'pedalMovementCat', 'pedalSizeCat', 'pedalDirection',
        'stimCat', 'originalIndex', 'segment', 't'],
    transposeToColumns='feature', concatOn='columns',
    verbose=False, procFun=None))
#
if 'outlierDetectOptions' in locals():
    targetEpochSize = outlierDetectOptions['targetEpochSize']
    stimulusConditionNames = outlierDetectOptions['conditionNames']
    twoTailed = outlierDetectOptions['twoTailed']
    alignedAsigsKWargs['windowSize'] = outlierDetectOptions['windowSize']
else:
    targetEpochSize = 1e-3
    stimulusConditionNames = [
        'electrode', arguments['amplitudeFieldName'], 'RateInHz']
    twoTailed = False
    alignedAsigsKWargs['windowSize'] = (-100e-3, 400e-3)

def takeSqrt(waveDF, spkTrain):
    return np.sqrt(waveDF)

if arguments['sqrtTransform']:
    alignedAsigsKWargs['procFun'] = takeSqrt

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)


def findOutliers(
        mahalDistDF, groupBy=None,
        qThresh=None, sdThresh=None, sdThreshInner=None,
        devQuantile=None, nDim=1, multiplier=1, twoTailed=False):
    #
    if sdThresh is None:
        if qThresh is None:
            qThresh = 1 - 1e-6
        chi2Bounds = chi2.interval(qThresh, nDim)
        sdThresh = multiplier * chi2Bounds[1]
    #
    chiProba = pd.Series(
        -np.log(np.squeeze(chi2.pdf(mahalDistDF, nDim))),
        index=mahalDistDF.index)
    chiProbaLim = -np.log(chi2.pdf(sdThresh, nDim))
    if devQuantile is not None:
        deviation = chiProba.groupby(groupBy).quantile(q=devQuantile)
        # maxMhDist = mahalDistDF.groupby(groupBy).quantile(q=devQuantile).iloc[:, -1] - sdThresh
        # minMhDist = sdThreshInner - mahalDistDF.groupby(groupBy).quantile(q=1-devQuantile).iloc[:, -1]
    else:
        deviation = chiProba.groupby(groupBy).max()
        # maxMhDist = mahalDistDF.groupby(groupBy).max().iloc[:, -1] - sdThresh
        # minMhDist = sdThreshInner - mahalDistDF.groupby(groupBy).min().iloc[:, -1]
    #
    # if twoTailed:
    #     deviation = pd.concat([maxMhDist, minMhDist], axis='columns').max(axis='columns')
    #     pdb.set_trace()
    # else:
    #     deviation = maxMhDist
    if isinstance(deviation, pd.Series):
        deviationDF = deviation.to_frame(name='deviation')
    else:
        deviationDF = deviation
    deviationDF['rejectBlock'] = (deviationDF['deviation'] > chiProbaLim)
    return deviationDF


def calcCovMat(
        partition, dataColNames=None,
        useEmpiricalCovariance=True,
        supportFraction=None, verbose=False):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    # print('partition shape = {}'.format(partitionData.shape))
    if not useEmpiricalCovariance:
        try:
            est = MinCovDet(support_fraction=supportFraction)
            est.fit(partitionData.values)
        except Exception:
            traceback.print_exc()
            print('\npartition shape = {}\n'.format(partitionData.shape))
            est = EmpiricalCovariance()
            est.fit(partitionData.values)
    else:
        est = EmpiricalCovariance()
        est.fit(partitionData.values)
    result = pd.DataFrame(
        est.mahalanobis(partitionData.values),
        index=partition.index, columns=['mahalDist'])
    # print('result shape is {}'.format(result.shape))
    result = pd.concat([result, partition.loc[:, ~dataColMask]], axis=1)
    result.name = 'mahalanobisDistance'
    # pdb.set_trace()
    # if result['electrode'].iloc[0] == 'foo':
    #     pdb.set_trace()
    # print('result type is {}'.format(type(result)))
    # print(result.T)
    # print('partition shape = {}'.format(partitionData.shape))
    return result


if __name__ == "__main__":
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

    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    trialInfo['epoch'] = np.nan
    firstBinMask = trialInfo['bin'] == trialInfo['bin'].unique()[0]
    groupNames = stimulusConditionNames + ['epoch']
    #  delay to account for transmission between event
    #  at t=0 and the signal being recorded
    transmissionDelay = 0
    if 'stimPeriod' not in trialInfo.columns:
        trialInfo['stimPeriod'] = trialInfo['RateInHz'] ** (-1)
        trialInfo.loc[np.isinf(trialInfo['stimPeriod']), 'stimPeriod'] = 10
    #
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
        validBins = (epochBins > theseTBins.min()) & (epochBins < theseTBins.max())
        epochBins = epochBins[validBins]
        # stretch first and last epoch bin to cover entire window
        epochBins[0] = theseTBins.min() - 1
        epochBins[-1] = theseTBins.max() + 1
        theseEpochs = pd.cut(theseTBins, bins=epochBins, labels=False)
        trialInfo.loc[group.index, 'epoch'] = theseEpochs
    #
    dataDF.set_index(
        pd.Index(trialInfo['epoch'], name='epoch'),
        append=True, inplace=True)

    testVar = None
    groupBy = ['segment', 't']
    resultNames = [
        'deviation', 'rejectBlock']

    print('working with {} samples'.format(dataDF.shape[0]))
    randSample = slice(None, None, None)

    useCachedMahalDist = True
    if useCachedMahalDist and os.path.exists(resultPath):
        with pd.HDFStore(resultPath,  mode='r') as store:
            mahalDist = pd.read_hdf(
                store, 'mahalDist')
            mahalDistLoaded = True
    else:
        mahalDistLoaded = False

    covOpts = dict(
        useEmpiricalCovariance=False,
        supportFraction=None)
    daskComputeOpts = dict(
        scheduler='processes'
        # scheduler='single-threaded'
        )
    if not mahalDistLoaded:
        if arguments['verbose']:
            print('Calculating covariance matrix...')
        daskClient = Client()
        mahalDist = ash.splitApplyCombine(
            dataDF, fun=calcCovMat, resultPath=resultPath,
            funKWArgs=covOpts,
            rowKeys=groupNames, colKeys=['lag'],
            useDask=True, reindexFromInput=False,
            daskComputeOpts=daskComputeOpts
            )
        mahalDist.columns = ['mahalDist']
        if arguments['saveResults']:
            if os.path.exists(resultPath):
                os.remove(resultPath)
            mahalDist.to_hdf(
                resultPath, 'mahalDist')

    print('#######################################################')
    refInterval = chi2.interval(1 - 1e-6, len(dataDF.columns))
    print('Data is {} dimensional'.format(len(dataDF.columns)))
    print('The mahalanobis distance should lie within {}'.format(refInterval))
    print('#######################################################')

    outlierTrials = findOutliers(
        mahalDist, groupBy=groupBy, multiplier=1, qThresh=1-1e-6,
        nDim=len(dataDF.columns), devQuantile=0.95, twoTailed=twoTailed)
    # outlierTrials = ash.applyFunGrouped(
    #     mahalDist,
    #     groupBy, testVar,
    #     fun=findOutliers, funArgs=[],
    #     funKWargs=dict(
    #         multiplier=1, qThresh=1-1e-2,
    #         nDim=len(dataDF.columns), devQuantile=0.95),
    #     # funKWargs=dict(sdThresh=100),
    #     resultNames=resultNames,
    #     plotting=False)
    print('\nHighest observed deviations were:')
    print(outlierTrials['deviation'].sort_values().tail())
    print('\nOutlier proportion was:')
    print(outlierTrials['rejectBlock'].sum() / outlierTrials['rejectBlock'].size)

    if arguments['plotting'] and outlierTrials['rejectBlock'].astype(np.bool).any():
        binSize = 1
        hist, binEdges = np.histogram(
            outlierTrials['deviation'],
            bins=np.arange(
                0,
                outlierTrials['deviation'].max() + binSize,
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
            prefix + '_mh_dist_histogram_by_condition_and_epoch_robust.pdf')
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
        outlierTrials
        .loc[outlierTrials['rejectBlock'].astype(np.bool), 'deviation'].sort_values()
        )
    # maxDroppedTrials = pd.Series(
    #     index=np.concatenate(
    #         [
    #             [theseOutliers.min()],
    #             np.linspace(
    #                 theseOutliers.min() / 3,
    #                 3 * theseOutliers.min(), 10)]
    #         ))
    # 
    # for ix, devThreshold in enumerate(maxDroppedTrials.index):
    #     # print(ix)
    #     if (theseOutliers >= devThreshold).any():
    #         outlierDataMasks = []
    #         for lvlIdx, levelName in enumerate(theseOutliers.index.names):
    #             outlierDataMasks.append(trialInfo[levelName].isin(theseOutliers.loc[theseOutliers >= devThreshold].index.get_level_values(levelName)))
    #         fullOutMask = np.logical_and.reduce(outlierDataMasks)
    #         nOutliersPerCondition = (
    #             trialInfo
    #             .loc[fullOutMask & firstBinMask, :]
    #             .groupby(['electrode', arguments['amplitudeFieldName']])['RateInHz']
    #             .value_counts())
    #         if ix == 0:
    #             saveNOutliers = nOutliersPerCondition
    #         maxDroppedTrials[devThreshold] = nOutliersPerCondition.max()

    # print('\nMaximum number of dropped trials, as a function of deviation threshold:')
    # print(maxDroppedTrials)
    # print(saveNOutliers.sort_values())
    if arguments['plotting'] and outlierTrials['rejectBlock'].astype(np.bool).any():
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
            prefix + '_outlier_trials_by_condition_and_epoch_robust.pdf')
        emgFig.savefig(
            pdfName,
            bbox_inches='tight', pad_inches=0, bbox_extra_artists=[emgLeg])
        pdfName = os.path.join(
            figureOutputFolder,
            prefix + '_mh_dist_by_condition_and_epoch_robust.pdf')
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
        outlierTrials['deviation'].to_hdf(
            resultPath, 'deviation')
        outlierTrials['rejectBlock'].to_hdf(
            resultPath, 'rejectBlock')

    # 
    minNObservations = 5
    firstBinTrialInfo = trialInfo.loc[firstBinMask, :]
    goodTrialInfo = firstBinTrialInfo.loc[~outlierTrials['rejectBlock'].to_numpy().flatten().astype(np.bool), :]
    goodTrialCount = goodTrialInfo.groupby(['electrode', arguments['amplitudeFieldName']])['RateInHz'].value_counts().to_frame(name='count').reset_index()
    goodTrialCount = goodTrialCount.loc[goodTrialCount['count'] > minNObservations, :]
    goodTrialCount.to_csv(os.path.join(figureOutputFolder, prefix + '_good_trial_breakdown.csv'))
    goodTrialCount.groupby(['electrode', 'RateInHz', arguments['amplitudeFieldName']]).ngroups
    badTrialInfo = firstBinTrialInfo.loc[outlierTrials['rejectBlock'].to_numpy().flatten().astype(np.bool), :]
    badTrialCount = badTrialInfo.groupby(['electrode', arguments['amplitudeFieldName']])['RateInHz'].value_counts().sort_values().to_frame(name='count').reset_index()
    outlierTrials['deviation'].reset_index().sort_values(['segment', 'deviation']).to_csv(os.path.join(figureOutputFolder, prefix + '_trial_deviation_breakdown.csv'))
    print('Bad trial count:\n{}'.format(badTrialCount))

    # .to_csv(os.path.join(figureOutputFolder, 'bad_trial_breakdown.csv'))
