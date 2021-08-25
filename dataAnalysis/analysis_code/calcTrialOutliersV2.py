"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                    which experimental day to analyze
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --processAll                                 process entire experimental day? [default: False]
    --lazy                                       load from raw, or regular? [default: False]
    --saveResults                                load from raw, or regular? [default: False]
    --useCachedMahalanobis                       load previous covariance matrix? [default: False]
    --inputBlockSuffix=inputBlockSuffix          which trig_ block to pull [default: pca]
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
    --arrayName=arrayName                        name of electrode array? (for map file) [default: utah]
"""
import logging
logging.captureWarnings(True)
import matplotlib, os, sys
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
import matplotlib.pyplot as plt
import seaborn as sns
import pdb, traceback
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
# import pingouin as pg
import quantities as pq
import pandas as pd
import numpy as np
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from copy import deepcopy
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
from sklearn.preprocessing import StandardScaler, RobustScaler
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'outlierTrials')
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

calcSubFolder = os.path.join(
    scratchFolder, 'outlierTrials', arguments['alignFolderName'])
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = 'Block'
else:
    prefix = ns5FileName

triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockSuffix'], arguments['window']))

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
    getMetaData=essentialMetadataFields,
    transposeToColumns='feature', concatOn='columns',
    verbose=False, procFun=None))
#
print(
    "'outlierDetectOptions' in locals(): {}"
    .format('outlierDetectOptions' in locals()))
#
if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC') or (blockExperimentType == 'isi'):
    # has stim but no motion
    stimulusConditionNames = stimConditionNames
elif blockExperimentType == 'proprio-motionOnly':
    # has motion but no stim
    stimulusConditionNames = motionConditionNames
else:
    stimulusConditionNames = stimConditionNames + motionConditionNames
print('Block type {}; using the following stimulus condition breakdown:'.format(blockExperimentType))
print('\n'.join(['    {}'.format(scn) for scn in stimulusConditionNames]))
if 'outlierDetectOptions' in locals():
    targetEpochSize = outlierDetectOptions['targetEpochSize']
    twoTailed = outlierDetectOptions['twoTailed']
    alignedAsigsKWargs['windowSize'] = outlierDetectOptions['windowSize']
    devQuantile = outlierDetectOptions.pop('devQuantile', 0.95)
    qThresh = outlierDetectOptions.pop('qThresh', 1e-6)
else:
    targetEpochSize = 1e-3
    twoTailed = False
    alignedAsigsKWargs['windowSize'] = (-100e-3, 400e-3)
    devQuantile = 0.95
    qThresh = 1e-6


alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)


def findOutliers(
        mahalDistDF, groupBy=None,
        qThresh=None, sdThresh=None, sdThreshInner=None,
        manualOutlierOverride=None,
        devQuantile=None, nDim=1, multiplier=1, twoTailed=False):
    #
    if sdThresh is None:
        if qThresh is None:
            qThresh = 1e-6
        chi2Bounds = chi2.isf(qThresh, nDim)
        sdThresh = multiplier * chi2Bounds
    #
    if twoTailed:
        chiProba = pd.Series(
            -(np.squeeze(chi2.logpdf(mahalDistDF, nDim))),
            index=mahalDistDF.index)
        chiProbaLim = -(chi2.logpdf(sdThresh, nDim))
        if devQuantile is not None:
            deviation = chiProba.groupby(groupBy).quantile(q=devQuantile)
        else:
            deviation = chiProba.groupby(groupBy).max()
    else:
        if devQuantile is not None:
            deviation = mahalDistDF['mahalDist'].groupby(groupBy).quantile(q=devQuantile)
        else:
            deviation = mahalDistDF['mahalDist'].groupby(groupBy).max()
    deviationDF = deviation.to_frame(name='deviation')
    #
    if twoTailed:
        deviationDF.loc[np.isinf(deviationDF['deviation']), 'deviation'] = 100 * chiProbaLim
        deviationDF['rejectBlock'] = (deviationDF['deviation'] > chiProbaLim)
        deviationDF['nearRejectBlock'] = (deviationDF['deviation'] > chiProbaLim / 2)
        deviationDF.deviationThreshold = chiProbaLim
    else:
        deviationDF.loc[np.isinf(deviationDF['deviation']), 'deviation'] = 100 * sdThresh
        deviationDF['rejectBlock'] = (deviationDF['deviation'] > sdThresh)
        deviationDF['nearRejectBlock'] = (deviationDF['deviation'] > sdThresh / 2)
        deviationDF.deviationThreshold = sdThresh
    if manualOutlierOverride is not None:
        print('Identifying these indices as outliers based on manualOutlierOverride\n{}'.format(deviationDF.index[manualOutlierOverride]))
        deviationDF.loc[deviationDF.index[manualOutlierOverride], ['nearRejectBlock', 'rejectBlock']] = True
    return deviationDF


def calcCovMat(
        partition, dataColNames=None,
        useMinCovDet=False,
        supportFraction=None, verbose=False):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    # print('partition shape = {}'.format(partitionData.shape))
    est = RobustScaler()
    result = pd.DataFrame(
        (
            est
            .fit_transform(partitionData.to_numpy().reshape(-1, 1))
            .reshape(partitionData.shape)),
        index=partition.index).abs().max(axis='columns').to_frame(name='mahalDist')
    result = pd.concat(
        [partition.loc[:, ~dataColMask], result],
        axis='columns')
    result.name = 'mahalanobisDistance'
    result.columns.name = partition.columns.name
    return result


if __name__ == "__main__":
    if 'mapDF' not in locals():
        electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
        mapExt = electrodeMapPath.split('.')[-1]
        if mapExt == 'cmp':
            mapDF = prb_meta.cmpToDF(electrodeMapPath)
        elif mapExt == 'map':
            mapDF = prb_meta.mapToDF(electrodeMapPath)
    print('loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
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
    dataDF = dataDF.astype(float)
    ordMag = np.floor(np.log10(dataDF.abs().mean().mean()))
    if ordMag < 0:
        dataDF = dataDF * 10 ** (-ordMag)
    # dataDF = dataDF.apply(lambda x: x - x.mean())
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    trialInfo['epoch'] = 0.
    firstBinMask = trialInfo['bin'] == trialInfo['bin'].unique()[0]
    groupNames = stimulusConditionNames + ['epoch']
    #  delay to account for transmission between event
    #  at t=0 and the signal being recorded
    transmissionDelay = 0
    #
    dataDF.set_index(
        pd.Index(trialInfo['epoch'], name='epoch'),
        append=True, inplace=True)
    testVar = None
    groupBy = ['segment', 't']
    resultNames = [
        'deviation', 'rejectBlock', 'nearRejectBlock']

    print('working with {} samples'.format(dataDF.shape[0]))
    randSample = slice(None, None, None)

    if arguments['useCachedMahalanobis'] and os.path.exists(resultPath):
        with pd.HDFStore(resultPath,  mode='r') as store:
            mahalDist = pd.read_hdf(
                store, 'mahalDist')
            mahalDistLoaded = True
    else:
        mahalDistLoaded = False

    covOpts = dict(
        useMinCovDet=False,
        supportFraction=0.95)
    daskComputeOpts = dict(
        # scheduler='processes'
        scheduler='single-threaded'
        )
    if not mahalDistLoaded:
        if arguments['verbose']:
            print('Calculating covariance matrix...')
        mahalDist = ash.splitApplyCombine(
            dataDF, fun=calcCovMat, resultPath=resultPath,
            funKWArgs=covOpts, daskResultMeta=None,
            rowKeys=groupNames, colKeys=['lag'],
            daskPersist=True, useDask=True, daskComputeOpts=daskComputeOpts)
        mahalDist.columns = ['mahalDist']
        if arguments['saveResults']:
            if os.path.exists(resultPath):
                os.remove(resultPath)
            mahalDist.to_hdf(
                resultPath, 'mahalDist')
    print('#######################################################')
    nDoF = 1  #  len(dataDF.columns)
    refValue = chi2.isf(qThresh, nDoF)
    print('Data is {} dimensional'.format(nDoF))
    print('The mahalanobis distance should be less than {}'.format(refValue))
    refLogProba = -np.log(chi2.pdf(refValue, nDoF))
    print('If the test is two-tailed, the log probability limit is {}'.format(refLogProba))
    print('#######################################################')
    #
    try:
        manualOutlierOverride = manualOutlierOverrideDict[int(arguments['blockIdx'])]
    except:
        manualOutlierOverride = None
    outlierTrials = findOutliers(
        mahalDist, groupBy=groupBy, multiplier=1, qThresh=qThresh,
        manualOutlierOverride=manualOutlierOverride,
        nDim=nDoF, devQuantile=devQuantile, twoTailed=twoTailed)
    print('\nHighest observed deviations were:')
    print(outlierTrials['deviation'].sort_values().tail())
    outlierCount = outlierTrials['rejectBlock'].sum()
    outlierProportion = outlierCount / outlierTrials.shape[0]
    print('\nOutlier proportion was\n{:.2f} ({} out of {})'.format(outlierProportion, outlierCount, outlierTrials.shape[0]))

    if arguments['saveResults']:
        outlierTrials['deviation'].to_hdf(
            resultPath, 'deviation')
        outlierTrials['rejectBlock'].to_hdf(
            resultPath, 'rejectBlock')
        print('#######################################################')
        print('Done saving data')
        print('#######################################################')
    if arguments['plotting'] and outlierTrials['rejectBlock'].astype(bool).any():
        pdfPath = os.path.join(
            figureOutputFolder,
            prefix + '_mahalanobis_dist_histogram.pdf')
        print('Plotting deviation histogram')
        with PdfPages(pdfPath) as pdf:
            '''binSize = 1
            theseBins = np.arange(
                0,
                outlierTrials['deviation'].max() + binSize,
                binSize)'''
            theseBins = np.linspace(0, outlierTrials['deviation'].max() * 1.01, 200)
            hist, binEdges = np.histogram(
                outlierTrials['deviation'],
                bins=theseBins
                )
            cumFrac = np.cumsum(hist) / hist.sum()
            mask = (cumFrac > 0.75) & (cumFrac < 0.999)
            fig, ax = plt.subplots(1, 2, sharex=True)
            fig.set_size_inches((6, 4))
            ax[0].plot(
                binEdges[:-1][mask],
                cumFrac[mask]
                )
            ax[0].axvline(outlierTrials.deviationThreshold, color='r')
            ax[0].set_xlabel('signal deviation')
            ax[0].set_ylabel('cummulative fraction')
            ax[1].plot(
                binEdges[:-1][mask],
                hist[mask] / hist.sum()
                )
            # ax[1].plot(binEdges[:-1][mask], theoreticalPMF[mask])
            ax[1].axvline(outlierTrials.deviationThreshold, color='r')
            ax[1].set_xlabel('signal deviation')
            ax[1].set_ylabel('fraction')
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()
            fig, ax = plt.subplots()
            plotDF = outlierTrials.reset_index()
            ax.plot(plotDF['t'], plotDF['deviation'], 'b')
            ax.axhline(outlierTrials.deviationThreshold, color='r')
            ax.set_xlabel('trial time (sec)')
            ax.set_ylabel('trial deviation')
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()
    #
    featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
    dataChanNames = featureInfo['feature'].apply(lambda x: x.replace('#0', ''))
    banksLookup = mapDF.loc[:, ['label', 'bank']].set_index(['label'])['bank']
    featureInfo.loc[:, 'bank'] = dataChanNames.map(banksLookup)
    trialInfo.loc[:, 'wasKept'] = ~outlierTrials.loc[pd.MultiIndex.from_frame(trialInfo[['segment', 't']]), 'rejectBlock'].to_numpy()
    dataDF.set_index(pd.MultiIndex.from_frame(trialInfo), inplace=True)
    #
    signalRangesFigPath = os.path.join(
        figureOutputFolder,
        prefix + '_outlier_channels.pdf')
    print('plotting signal ranges')
    with PdfPages(signalRangesFigPath) as pdf:
        # pdb.set_trace()
        keepLevels = ['wasKept']
        dropLevels = [lN for lN in dataDF.index.names if lN not in keepLevels]
        plotDF = dataDF.T.reset_index(drop=True).T.reset_index(dropLevels, drop=True)
        plotDF.columns.name = 'flatFeature'
        plotDF = plotDF.stack().reset_index(name='signal')
        plotDF.loc[:, 'bank'] = plotDF['flatFeature'].map(featureInfo['bank'])
        plotDF.loc[:, 'feature'] = plotDF['flatFeature'].map(featureInfo['feature'])
        h = 18
        w = 3
        aspect = w / h
        g = sns.catplot(
            col='bank', x='signal', y='feature',
            data=plotDF, orient='h', kind='violin', ci='sd',
            linewidth=0.5, cut=0,
            sharex=False, sharey=False, height=h, aspect=aspect)
        g.suptitle('original')
        g.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        try:
            g = sns.catplot(
                col='bank', x='signal', y='feature',
                data=plotDF.loc[plotDF['wasKept'], :], orient='h', kind='violin', ci='sd',
                linewidth=0.5, cut=0,
                sharex=False, sharey=False, height=h, aspect=aspect
                )
            g.suptitle('triaged')
            g.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()
        except Exception:
            traceback.print_exc()

    theseOutliers = (
        outlierTrials
        .loc[outlierTrials['nearRejectBlock'].astype(bool), 'deviation'].sort_values()
        ).iloc[-100:]

    if arguments['plotting'] and outlierTrials['nearRejectBlock'].astype(bool).any():
        print('Plotting rejected trials')
        # nRowCol = int(np.ceil(np.sqrt(theseOutliers.size)))
        pdfPath = os.path.join(
            figureOutputFolder,
            prefix + '_outlier_trials.pdf')
        with PdfPages(pdfPath) as pdf:
            ##############################################
            nRowCol = max(int(np.ceil(np.sqrt(theseOutliers.size))), 2)
            mhFig, mhAx = plt.subplots(
                nRowCol, nRowCol, sharex=True)
            mhFig.set_size_inches(5 * nRowCol, 3 * nRowCol)
            # for idx, (name, group) in enumerate(dataDF.loc[fullOutMask, :].groupby(theseOutliers.index.names)):
            for idx, (name, row) in enumerate(theseOutliers.items()):
                thisDeviation = row
                deviationThreshold = outlierTrials.deviationThreshold
                wasThisRejected = 'rejected' if outlierTrials.loc[name, 'rejectBlock'] else 'not rejected'
                annotationColor = 'r' if outlierTrials.loc[name, 'rejectBlock'] else 'g'
                outlierDataMasks = []
                for lvlIdx, levelName in enumerate(theseOutliers.index.names):
                    outlierDataMasks.append(dataDF.index.get_level_values(levelName) == name[lvlIdx])
                fullOutMask = np.logical_and.reduce(outlierDataMasks)
                mhp = - chi2.logpdf(mahalDist.loc[fullOutMask, 'mahalDist'], nDoF)
                mhAx.flat[idx].plot(
                    mahalDist.loc[fullOutMask, :].index.get_level_values('bin'),
                    mhp,
                    c=(0, 0, 0, .6),
                    label='mahalDist', rasterized=True)
                mhAx.flat[idx].text(
                    1, 1, 'dev={:.2f}\nthresh={:.2f}\n({})'.format(
                        thisDeviation, deviationThreshold, wasThisRejected),
                    va='top', ha='right', color=annotationColor,
                    transform=mhAx.flat[idx].transAxes)
            mhAx.flat[0].set_ylabel('- log probability')
            mhFig.suptitle('Outlier proportion {:.2f} ({} out of {})'.format(outlierProportion, outlierCount,
                                                                             outlierTrials.shape[0]))
            pdf.savefig(
                bbox_inches='tight',
                pad_inches=0,
                # bbox_extra_artists=[mhLeg]
            )
            plt.close()
            mhFig, mhAx = plt.subplots(
                nRowCol, nRowCol, sharex=True)
            mhFig.set_size_inches(5 * nRowCol, 3 * nRowCol)
            # for idx, (name, group) in enumerate(dataDF.loc[fullOutMask, :].groupby(theseOutliers.index.names)):
            for idx, (name, row) in enumerate(theseOutliers.items()):
                thisDeviation = row
                deviationThreshold = outlierTrials.deviationThreshold
                wasThisRejected = 'rejected' if outlierTrials.loc[name, 'rejectBlock'] else 'not rejected'
                annotationColor = 'r' if outlierTrials.loc[name, 'rejectBlock'] else 'g'
                outlierDataMasks = []
                for lvlIdx, levelName in enumerate(theseOutliers.index.names):
                    outlierDataMasks.append(dataDF.index.get_level_values(levelName) == name[lvlIdx])
                fullOutMask = np.logical_and.reduce(outlierDataMasks)
                mhAx.flat[idx].plot(
                    mahalDist.loc[fullOutMask, :].index.get_level_values('bin'),
                    mahalDist.loc[fullOutMask, 'mahalDist'],
                    c=(0, 0, 0, .6),
                    label='mahalDist', rasterized=True)
                mhAx.flat[idx].text(
                    1, 1, 'dev={:.2f}\nthresh={:.2f}\n({})'.format(
                        thisDeviation, deviationThreshold, wasThisRejected),
                    va='top', ha='right', color=annotationColor,
                    transform=mhAx.flat[idx].transAxes)
            mhAx.flat[0].set_ylabel('Mahalanobis distance')
            mhFig.suptitle('Outlier proportion {:.2f} ({} out of {})'.format(outlierProportion, outlierCount,
                                                                             outlierTrials.shape[0]))
            pdf.savefig(
                bbox_inches='tight',
                pad_inches=0,
                # bbox_extra_artists=[mhLeg]
            )
            plt.close()
            emgFig, emgAx = plt.subplots(
                nRowCol, nRowCol, sharex=True)
            emgFig.set_size_inches(5 * nRowCol, 3 * nRowCol)
            for idx, (name, row) in enumerate(theseOutliers.items()):
                thisDeviation = row
                deviationThreshold = outlierTrials.deviationThreshold
                wasThisRejected = 'rejected' if outlierTrials.loc[name, 'rejectBlock'] else 'not rejected'
                annotationColor = 'r' if outlierTrials.loc[name, 'rejectBlock'] else 'g'
                outlierDataMasks = []
                for lvlIdx, levelName in enumerate(theseOutliers.index.names):
                    outlierDataMasks.append(dataDF.index.get_level_values(levelName) == name[lvlIdx])
                fullOutMask = np.logical_and.reduce(outlierDataMasks)
                for cN in dataDF.columns:
                    emgAx.flat[idx].plot(
                        dataDF.loc[fullOutMask, :].index.get_level_values('bin'),
                        dataDF.loc[fullOutMask, cN],
                        c=(0, 0, 0, .3),
                        alpha=0.8, label=cN[0], rasterized=True)
                    if not hasattr(emgAx.flat[idx], 'wasAnnotated'):
                        emgAx.flat[idx].text(
                            1, 1, 'dev={:.2f}\nthresh={:.2f}\n({})'.format(
                                thisDeviation, deviationThreshold, wasThisRejected),
                            va='top', ha='right', color=annotationColor,
                            transform=emgAx.flat[idx].transAxes)
                        emgAx.flat[idx].wasAnnotated = True
            emgAx.flat[0].set_ylabel('signal units (uV)')
            emgFig.suptitle(
                'Outlier proportion {:.2f} ({} out of {})'.format(outlierProportion, outlierCount, outlierTrials.shape[0]))
            pdf.savefig(
                bbox_inches='tight', pad_inches=0,
                # bbox_extra_artists=[emgLeg]
                )
            plt.close()
    if arguments['saveResults']:
        exportDF = (
            outlierTrials
            .apply(
                lambda x: 'deviation: {:.3f}, rejected: {}'.format(x['deviation'], x['rejectBlock']),
                axis='columns')
            .reset_index()
            .drop(columns=['segment'])
                )
        event = ns5.Event(
            name='seg0_outlierTrials',
            times=exportDF['t'].to_numpy() * pq.sec,
            labels=exportDF[0].to_numpy()
            )
        eventBlock = ns5.Block(name='outlierTrials')
        seg = ns5.Segment(name='seg0_outlierTrials')
        eventBlock.segments.append(seg)
        seg.block = eventBlock
        seg.events.append(event)
        event.segment = seg
        eventExportPath = os.path.join(
            calcSubFolder,
            prefix + '_{}_outliers.nix'.format(
                arguments['window']))
        writer = ns5.NixIO(filename=eventExportPath)
        writer.write_block(eventBlock, use_obj_names=True)
        writer.close()
    # 
    minNObservations = 0
    firstBinTrialInfo = trialInfo.loc[firstBinMask, :]
    goodTrialInfo = firstBinTrialInfo.loc[~outlierTrials['rejectBlock'].to_numpy().flatten().astype(bool), :]
    goodTrialCount = goodTrialInfo.groupby(stimulusConditionNames).count().iloc[:, 0].to_frame(name='count').reset_index()
    goodTrialCount = goodTrialCount.loc[goodTrialCount['count'] > minNObservations, :]
    goodTrialCount.to_html(os.path.join(figureOutputFolder, prefix + '_good_trial_breakdown.html'))
    # goodTrialCount.groupby(stimulusConditionNames).ngroups
    badTrialInfo = firstBinTrialInfo.loc[outlierTrials['rejectBlock'].to_numpy().flatten().astype(bool), :]
    badTrialCount = badTrialInfo.groupby(stimulusConditionNames).count().iloc[:, 0].sort_values().to_frame(name='count').reset_index()
    outlierTrials['deviation'].reset_index().to_html(os.path.join(figureOutputFolder, prefix + '_trial_deviation_breakdown.html'))
    outlierTrials['deviation'].reset_index().sort_values(['segment', 'deviation']).to_html(os.path.join(figureOutputFolder, prefix + '_trial_deviation_breakdown_sorted.html'))
    print('Bad trial count:\n{}'.format(badTrialCount))