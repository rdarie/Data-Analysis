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
import os, sys
import matplotlib
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
##
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
from scipy.stats import zscore, chi2, norm
# import pingouin as pg
import quantities as pq
import pandas as pd
import numpy as np
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from copy import deepcopy
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, LedoitWolf
from sklearn.decomposition import PCA
import dataAnalysis.custom_transformers.tdr as tdr
from sklearn.utils.random import sample_without_replacement as swr
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
        'lines.markersize': 2.4,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 7,
    'mathtext.default': 'regular',
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV
#######
from pandas import IndexSlice as idxSl
from datetime import datetime as dt

def findOutliers(
        normalizedDF, groupBy=None,
        qThresh=None, sdThresh=None,
        manualOutlierOverride=None, multiplier=1,
        devQuantile=None, twoTailed=False):
    #
    if sdThresh is None:
        if qThresh is None:
            qThresh = 1e-6
        sdThresh = multiplier * norm.isf(qThresh)
    #
    if devQuantile is not None:
        maxDeviation = normalizedDF.groupby(groupBy).quantile(q=devQuantile)
    else:
        maxDeviation = normalizedDF.groupby(groupBy).max()
    if twoTailed:
        if devQuantile is not None:
            minDeviation = normalizedDF.groupby(groupBy).quantile(q=1-devQuantile)
        else:
            minDeviation = normalizedDF.groupby(groupBy).min()
        mostDevMask = (minDeviation.abs() > maxDeviation.abs())
        maxDeviation.mask(mostDevMask, minDeviation.abs(), inplace=True)
    deviationDF = maxDeviation.to_frame(name='deviation')
    #
    deviationDF.loc[np.isinf(deviationDF['deviation']), 'deviation'] = sdThresh + 1
    deviationDF['rejectBlock'] = (deviationDF['deviation'] > sdThresh)
    deviationDF['nearRejectBlock'] = (deviationDF['deviation'] > sdThresh / 2)
    deviationDF.deviationThreshold = sdThresh
    if manualOutlierOverride is not None:
        print('Identifying these indices as outliers based on manualOutlierOverride\n{}'.format(deviationDF.index[manualOutlierOverride]))
        deviationDF.loc[deviationDF.index[manualOutlierOverride], ['nearRejectBlock', 'rejectBlock']] = True
    return deviationDF


def calcAverageZScore(
        partition, dataColNames=None,
        fitWindow=None, reduction='absmax',
        useMinCovDet=False, applyScaler=False,
        supportFraction=None, verbose=False):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    if fitWindow is not None:
        fitMask = (partition['bin'] >= fitWindow[0]) & (partition['bin'] < fitWindow[1])
        if not fitMask.any():
            fitMask = partition['bin'].notna()
    else:
        fitMask = partition['bin'].notna()
    # print('partition shape = {}'.format(partitionData.shape))
    est = StandardScaler()
    est.fit(partitionData.loc[fitMask, :].to_numpy())
    result = pd.DataFrame(
        est.transform(partitionData.to_numpy()),
        index=partition.index)
    if reduction == 'absmax':
        result = result.abs().max(axis='columns').to_frame(name='zScore')
    result = pd.concat(
        [partition.loc[:, ~dataColMask], result],
        axis='columns')
    result.name = 'zScore'
    result.columns.name = partition.columns.name
    return result

def calcCovMat(
        partition, dataColNames=None,
        fitWindow=None,
        useMinCovDet=False, applyScaler=False,
        supportFraction=None, verbose=False):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    if fitWindow is not None:
        fitMask = (partition['bin'] >= fitWindow[0]) & (partition['bin'] < fitWindow[1])
        if not fitMask.any():
            fitMask = partition['bin'].notna()
    else:
        fitMask = partition['bin'].notna()
    if verbose:
        print('partition shape = {}'.format(partitionData.shape))
    if useMinCovDet:
        try:
            est = MinCovDet(support_fraction=supportFraction)
            est.fit(partitionData.loc[fitMask, :])
        except Exception:
            traceback.print_exc()
            print('\npartition shape = {}\n'.format(partitionData.shape))
            # est = LedoitWolf(store_precision=False)
            est = EmpiricalCovariance()
            est.fit(partitionData.loc[fitMask, :])
    else:
        # est = LedoitWolf(store_precision=False)
        est = EmpiricalCovariance()
        est.fit(partitionData.loc[fitMask, :])
    resNP = est.mahalanobis(partitionData.to_numpy())
    if applyScaler:
        resNP = PowerTransformer().fit_transform(resNP.reshape(-1, 1))
        resNP = np.abs(resNP)
    result = pd.DataFrame(
        resNP,
        index=partition.index, columns=['mahalDist'])
    if verbose:
        print('result shape is {}'.format(result.shape))
    result = pd.concat(
        [partition.loc[:, ~dataColMask], result],
        axis=1)
    result.name = 'mahalanobisDistance'
    result.columns.name = partition.columns.name
    #
    # if result['electrode'].iloc[0] == 'foo':
    #     pdb.set_trace()
    # print('result type is {}'.format(type(result)))
    # print(result.T)
    # print('partition shape = {}'.format(partitionData.shape))
    return result

def calcSVD(
        partition, dataColNames=None,
        fitWindow=None,
        useMinCovDet=False, applyScaler=False,
        supportFraction=None, verbose=False):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    if fitWindow is not None:
        fitMask = (partition['bin'] >= fitWindow[0]) & (partition['bin'] < fitWindow[1])
        if not fitMask.any():
            fitMask = partition['bin'].notna()
    else:
        fitMask = partition['bin'].notna()
    # print('partition shape = {}'.format(partitionData.shape))
    est = PCA(whiten=True, svd_solver='full')
    est.fit(partitionData)
    svdThresh = tdr.optimalSVDThreshold(est.get_covariance()) * np.median(est.singular_values_)
    svdMask = est.singular_values_ >= svdThresh
    newNDim = svdMask.sum()
    # plt.plot(est.singular_values_)
    # plt.gca().axhline(svdThresh, c='r'); plt.show()
    components = est.transform(partitionData.to_numpy())
    result = pd.DataFrame(
        np.sum(components[:, svdMask] ** 2, axis=1).reshape(-1, 1),
        index=partition.index, columns=['nDim={}'.format(newNDim)])
    # print('result shape is {}'.format(result.shape))
    result = pd.concat(
        [partition.loc[:, ~dataColMask], result],
        axis=1)
    result.columns.name = partition.columns.name
    result.columns.names = partition.columns.names
    # if result['electrode'].iloc[0] == 'foo':
    #     pdb.set_trace()
    # print('result type is {}'.format(type(result)))
    # print(result.T)
    # print('partition shape = {}'.format(partitionData.shape))
    return result

if __name__ == "__main__":
    #
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    #
    if arguments['alignFolderName'] == 'stim':
        if blockExperimentType == 'proprio-motionOnly':
            print('skipping block {} (no stim)'.format(arguments['blockIdx']))
            sys.exit()
    if arguments['alignFolderName'] == 'motion':
        if blockExperimentType == 'proprio-miniRC':
            print('skipping block {} (no movement)'.format(arguments['blockIdx']))
            sys.exit()
    #
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
    #
    calcSubFolder = os.path.join(
        scratchFolder, 'outlierTrials', arguments['alignFolderName'])
    if not os.path.exists(calcSubFolder):
        os.makedirs(calcSubFolder, exist_ok=True)
    #
    if arguments['processAll']:
        prefix = 'Block'
    else:
        prefix = ns5FileName
    #
    triggeredPath = os.path.join(
        alignSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    #
    resultPath = os.path.join(
        calcSubFolder,
        prefix + '_{}_outliers.h5'.format(
            arguments['window']))
    resultPathCSV = os.path.join(
        calcSubFolder,
        prefix + '_{}_outliers.csv'.format(
            arguments['window']))
    #
    outlierLogPath = os.path.join(
        figureFolder,
        prefix + '_{}_outlierTrials.txt'.format(arguments['window']))
    if os.path.exists(outlierLogPath):
        os.remove(outlierLogPath)
    #
    alignedAsigsKWargs.update(dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False, removeFuzzyName=False,
        decimate=3, rollingWindow=3,
        metaDataToCategories=False,
        getMetaData=essentialMetadataFields,
        transposeToColumns='feature', concatOn='columns',
        verbose=False,
        procFun=None
        ))
    print(
        "'outlierDetectOptions' in locals(): {}"
            .format('outlierDetectOptions' in locals()))
    #
    '''if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC') or (
            blockExperimentType == 'isi'):
        # has stim but no motion
        stimulusConditionNames = stimConditionNames
    elif blockExperimentType == 'proprio-motionOnly':
        # has motion but no stim
        stimulusConditionNames = motionConditionNames
    else:
        stimulusConditionNames = stimConditionNames + motionConditionNames'''
    print('Block type {}; using the following stimulus condition breakdown:'.format(blockExperimentType))
    print('\n'.join(['    {}'.format(scn) for scn in stimulusConditionNames]))
    if 'outlierDetectOptions' in locals():
        twoTailed = outlierDetectOptions['twoTailed']
        alignedAsigsKWargs['windowSize'] = outlierDetectOptions['windowSize']
        devQuantile = outlierDetectOptions.pop('devQuantile', 0.95)
        qThresh = outlierDetectOptions.pop('qThresh', 1e-6)
        fitWindow = outlierDetectOptions.pop('fitWindow', None)
    else:
        twoTailed = False
        alignedAsigsKWargs['windowSize'] = (-100e-3, 400e-3)
        devQuantile = 0.95
        qThresh = 1e-6
        fitWindow = (-100e-3, 400e-3)
    #
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, alignSubFolder, **arguments)
    #
    try:
        print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
    except:
        pass
    for arg in sys.argv:
        print(arg)
    if 'mapDF' not in locals():
        electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
        mapExt = electrodeMapPath.split('.')[-1]
        if mapExt == 'cmp':
            mapDF = prb_meta.cmpToDF(electrodeMapPath)
        elif mapExt == 'map':
            mapDF = prb_meta.mapToDF(electrodeMapPath)
    print('loading {}'.format(triggeredPath))
    pdb.set_trace()
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
    # dataDF = dataDF.astype(float)
    # ordMag = np.floor(np.log10(dataDF.abs().mean().mean()))
    # if ordMag < 0:
    #     dataDF = dataDF * 10 ** (-ordMag)
    # dataDF = dataDF.apply(lambda x: x - x.mean())
    dataDF = pd.DataFrame(
        StandardScaler().fit_transform(dataDF),
        index=dataDF.index, columns=dataDF.columns
        )
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    firstBinMask = trialInfo['bin'] == trialInfo['bin'].unique()[0]
    groupNames = ['expName']
    testVar = None
    groupBy = ['segment', 't']
    resultNames = [
        'deviation', 'rejectBlock', 'nearRejectBlock']

    print('working with {} samples'.format(dataDF.shape[0]))
    randSample = slice(None, None, None)
    mahalDistLoaded = False
    #
    if arguments['useCachedMahalanobis'] and os.path.exists(resultPath):
        with pd.HDFStore(resultPath,  mode='r') as store:
            mahalDist = pd.read_hdf(
                store, 'mahalDist')
            mahalDistLoaded = True
    covOpts = dict(
        useMinCovDet=False,
        applyScaler=False,
        supportFraction=0.95,
        fitWindow=fitWindow)
    zScoreOpts = dict(
        fitWindow=fitWindow,
        reduction='absmax'
        )
    daskComputeOpts = dict(
        # scheduler='processes'
        scheduler='single-threaded'
        )
    if not mahalDistLoaded:
        if arguments['verbose']:
            print('Calculating covariance matrix...')
        mahalDist = ash.splitApplyCombine(
            dataDF,
            fun=calcCovMat,
            resultPath=resultPath,
            funKWArgs=covOpts, daskResultMeta=None,
            daskProgBar = True,
            rowKeys=groupNames, colKeys=['lag'],
            daskPersist=True, useDask=True,
            daskComputeOpts=daskComputeOpts)
        columnsIndex = mahalDist.columns.to_frame().reset_index(drop=True)
        if columnsIndex['feature'].str.contains('nDim=').any():
            nDoF = int(columnsIndex['feature'].iloc[0].split('nDim=')[-1])
        if covOpts['applyScaler']:
            nDoF = 1
        else:
            nDoF = len(dataDF.columns)  # 1
        mahalDist.columns = ['mahalDist']
        mahalDistNormalized = pd.Series(
            norm.isf(chi2.sf(mahalDist['mahalDist'], df=nDoF)),
            index=dataDF.index)
        zScoreDF = ash.splitApplyCombine(
            dataDF,
            fun=calcAverageZScore,
            resultPath=resultPath,
            funKWArgs=zScoreOpts, daskResultMeta=None,
            daskProgBar = True,
            rowKeys=groupNames, colKeys=['lag'],
            daskPersist=True, useDask=True,
            daskComputeOpts=daskComputeOpts)
        zScoreDF.columns = ['zScore']
        mostDeviantMask = (mahalDistNormalized.abs() < zScoreDF['zScore'].abs())
        overallDeviation = mahalDistNormalized.mask(mostDeviantMask, zScoreDF['zScore'])
        if arguments['saveResults']:
            if os.path.exists(resultPath):
                os.remove(resultPath)
            mahalDist.to_hdf(
                resultPath, 'mahalDist')
            if os.path.exists(resultPathCSV):
                os.remove(resultPathCSV)
    print('#######################################################')
    print('Data is {} dimensional'.format(nDoF))
    #
    normValueMax = np.sqrt(chi2.isf(qThresh, 1))
    normValueMin = np.sqrt(chi2.ppf(qThresh, 1))
    #
    chi2ValueMax = chi2.isf(qThresh, nDoF)
    chi2ValueMin = chi2.ppf(qThresh, nDoF)
    print('The normalized mahalanobis distance should lie between {} and {}'.format(normValueMin, normValueMax))
    refLogProba = -np.log(norm.pdf(normValueMax))
    print('If the test is two-tailed, the log probability limit is {}'.format(refLogProba))
    print('#######################################################')
    try:
        manualOutlierOverride = manualOutlierOverrideDict[arguments['alignFolderName']][int(arguments['blockIdx'])]
    except:
        manualOutlierOverride = None
    outlierTrials = findOutliers(
        overallDeviation, groupBy=groupBy, qThresh=qThresh,
        manualOutlierOverride=manualOutlierOverride,
        devQuantile=devQuantile, twoTailed=twoTailed)
    print('\nHighest observed deviations were:')
    print(outlierTrials['deviation'].sort_values().tail())
    outlierCount = outlierTrials['rejectBlock'].sum()
    outlierProportion = outlierCount / outlierTrials.shape[0]
    print('\nOutlier proportion was\n{:.2f} ({} out of {})'.format(
        outlierProportion, outlierCount, outlierTrials.shape[0]))
    #
    if arguments['saveResults']:
        outlierTrials['deviation'].to_hdf(
            resultPath, 'deviation')
        outlierTrials['rejectBlock'].to_hdf(
            resultPath, 'rejectBlock')
        outlierTrials.reset_index().to_csv(resultPathCSV, float_format='%.9f')
        # pd.read_csv(resultPathCSV)
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
            # ax.plot(plotDF['t'], plotDF['deviation'], 'b')
            ax.stem(
                plotDF['t'], plotDF['deviation'],
                basefmt='k-'
                )
            ax.axhline(outlierTrials.deviationThreshold, color='r', label='curr. threshold')
            ax.set_xlabel('trial time (sec)')
            ax.set_ylabel('trial deviation')
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()
            devOpts = dict(nDim=nDoF, twoTailed=twoTailed)
            devLookup = pd.Series({
                q: norm.isf(q)
                for q in np.logspace(-3, -48, 100)})
            devLookup.name = 'deviation'
            devLookup.index.name = 'quantile'
            fig, ax = plt.subplots()
            ax.semilogx(devLookup.index, devLookup, label='deviation')
            ax.set_ylabel('deviation')
            ax.set_xlabel('quantile')
            ax.axhline(outlierTrials.deviationThreshold, color='r', label='curr. threshold')
            ax.legend()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()
            fig, ax = plt.subplots()
            zoomFactor = (chi2ValueMax - chi2ValueMin)
            x = np.linspace(chi2ValueMin - zoomFactor / 4, chi2ValueMax + zoomFactor, 200)
            ax.plot(
                x, chi2.pdf(x, nDoF),
                'r-', lw=1, alpha=0.6,
                label='chi2 pdf ({} df)'.format(nDoF))
            sns.histplot(
                x='mahalDist', data=mahalDist,
                bins=x,
                ax=ax, color='b', stat='density', label='all mahal dist')
            sns.histplot(
                x='mahalDist', data=mahalDist.groupby(groupBy).max(),
                bins=x,
                ax=ax, color='y', stat='density', label='max mahal. dist')
            ax.axvline(chi2ValueMin, c='g', label='min thresh')
            ax.axvline(chi2ValueMax, c='y', label='max thresh')
            ax.legend()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()
            fig, ax = plt.subplots()
            zoomFactorNorm = (normValueMax - normValueMin)
            x = np.linspace(
                normValueMin - zoomFactorNorm / 8, normValueMax + zoomFactorNorm / 2,
                200)
            '''ax.plot(
                x, chi2.pdf(x, 1),
                'r-', lw=1, alpha=0.6,
                label='chi2 pdf (1 df)')'''
            # zScoreAsMahal = zScoreDF ** 2
            sns.histplot(
                x='zScore', data=zScoreDF,
                bins=x,
                ax=ax, color='b', stat='density', label='all max zscore squared')
            sns.histplot(
                x='zScore', data=zScoreDF.groupby(groupBy).min(),
                bins=x,
                ax=ax, color='g', stat='density', label='min zscore squared')
            sns.histplot(
                x='zScore', data=zScoreDF.groupby(groupBy).max(),
                bins=x,
                ax=ax, color='y', stat='density', label='max zscore scquared')
            ax.axvline(normValueMin, c='g', ls='--', label='min thresh')
            ax.axvline(normValueMax, c='y', ls='--', label='max thresh')
            ax.legend()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            plt.close()
    featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
    dataChanNames = featureInfo['feature'].apply(lambda x: x.replace('#0', ''))
    banksLookup = mapDF.loc[:, ['label', 'bank']].set_index(['label'])['bank']
    featureInfo.loc[:, 'bank'] = dataChanNames.map(banksLookup).fillna(mapDF['bank'].unique()[0])
    trialInfo.loc[:, 'wasKept'] = ~outlierTrials.loc[pd.MultiIndex.from_frame(trialInfo[['segment', 't']]), 'rejectBlock'].to_numpy()
    dataDF.set_index(pd.MultiIndex.from_frame(trialInfo), inplace=True)
    #
    if False:
        signalRangesFigPath = os.path.join(
            figureOutputFolder,
            prefix + '_outlier_channels.pdf')
        print('plotting signal ranges')
        print('dataDF.columns = {}'.format(dataDF.columns.to_frame().reset_index(drop=True)))
        with PdfPages(signalRangesFigPath) as pdf:
            keepLevels = ['wasKept']
            dropLevels = [lN for lN in dataDF.index.names if lN not in keepLevels]
            plotDF = dataDF.T.reset_index(drop=True).T.reset_index(dropLevels, drop=True)
            plotDF.columns.name = 'flatFeature'
            plotDF = plotDF.stack().reset_index(name='signal')
            plotDF.loc[:, 'bank'] = plotDF['flatFeature'].map(featureInfo['bank'])
            plotDF.loc[:, 'feature'] = plotDF['flatFeature'].map(featureInfo['feature'])
            plotDF.sort_values(by=['bank', 'feature'], inplace=True)
            h = 18
            w = 3
            aspect = w / h
            seekIdx = slice(None, None, max(1, int(np.ceil(plotDF.shape[0] // int(1e7)))))
            g = sns.catplot(
                col='bank', x='signal', y='feature',
                data=plotDF.iloc[seekIdx, :], orient='h', kind='violin', ci='sd',
                linewidth=0.5, cut=0,
                sharex=False, sharey=False, height=h, aspect=aspect)
            g.suptitle('original')
            g.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()
            try:
                seekIdx = slice(
                    None, None,
                    max(1, plotDF['wasKept'].sum() // int(1e7)))
                g = sns.catplot(
                    col='bank', x='signal', y='feature',
                    data=plotDF.loc[plotDF['wasKept'], :].iloc[seekIdx, :], orient='h', kind='violin', ci='sd',
                    linewidth=0.5, cut=0,
                    sharex=False, sharey=False, height=h, aspect=aspect
                    )
                g.suptitle('triaged')
                g.tight_layout()
                pdf.savefig(bbox_inches='tight')
                plt.close()
            except Exception:
                traceback.print_exc()
    #
    if arguments['plotting'] and outlierTrials['nearRejectBlock'].astype(bool).any():
        print('Plotting rejected trials')
        theseOutliers = (
            outlierTrials
            .loc[outlierTrials['nearRejectBlock'].astype(bool), 'deviation'].sort_values()
            ).iloc[-1 * int(8**2):]
        # nRowCol = int(np.ceil(np.sqrt(theseOutliers.size)))
        pdfPath = os.path.join(
            figureOutputFolder,
            prefix + '_outlier_trials.pdf')
        with PdfPages(pdfPath) as pdf:
            ##############################################
            nRowCol = max(int(np.ceil(np.sqrt(theseOutliers.size))), 2)
            if False:
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
                mhFig.suptitle('Outlier proportion {:.2f} ({} out of {})'.format(
                    outlierProportion, outlierCount,
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
                    c=(.8, .8, .8),
                    label='mahalDist', rasterized=True)
                mhAx.flat[idx].axhline(chi2ValueMin, color='g', label='min thresh')
                mhAx.flat[idx].axhline(chi2ValueMax, color='y', label='max thresh')
                mhAx.flat[idx].text(
                    1, 1, 'dev={:.2f}\nthresh={:.2f}\n({})'.format(
                        thisDeviation, deviationThreshold, wasThisRejected),
                    va='top', ha='right', color=annotationColor,
                    transform=mhAx.flat[idx].transAxes)
                mhTwinAx = mhAx.flat[idx].twinx()
                mhTwinAx.plot(
                    zScoreDF.loc[fullOutMask, :].index.get_level_values('bin'),
                    zScoreDF.loc[fullOutMask, 'zScore'],
                    c=(.6, .6, .6), label='zScore', rasterized=True)
                mhTwinAx.axhline(normValueMin, color='g', ls='--', label='min thresh')
                mhTwinAx.axhline(normValueMax, color='y', ls='--', label='max thresh')
                #
                mhAx.flat[idx].set_ylabel('Mahalanobis distance')
                mhAx.flat[idx].legend(loc='upper left')
                mhTwinAx.legend(loc='lower left')
                mhTwinAx.set_ylabel('Max z-score')
            mhFig.suptitle('Outlier proportion {:.2f} ({} out of {})'.format(
                outlierProportion, outlierCount,
                outlierTrials.shape[0]))
            pdf.savefig(
                bbox_inches='tight',
                pad_inches=0,
                # bbox_extra_artists=[mhLeg]
                )
            plt.close()
            if True:
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
                            # alpha=0.5,
                            label=cN[0], rasterized=True)
                        if not hasattr(emgAx.flat[idx], 'wasAnnotated'):
                            emgAx.flat[idx].text(
                                1, 1, 'dev={:.2f}\nthresh={:.2f}\n({})'.format(
                                    thisDeviation, deviationThreshold, wasThisRejected),
                                va='top', ha='right', color=annotationColor,
                                transform=emgAx.flat[idx].transAxes)
                            emgAx.flat[idx].wasAnnotated = True
                emgAx.flat[0].set_ylabel('signal units (uV)')
                emgFig.suptitle(
                    'Outlier proportion {:.2f} ({} out of {})'.format(
                        outlierProportion, outlierCount, outlierTrials.shape[0]))
                pdf.savefig(bbox_inches='tight', pad_inches=0)
                plt.close()
    if arguments['saveResults']:
        exportDF = (
            outlierTrials
            .apply(
                lambda x: 'deviation: {:.3f}, rejected: {}'.format(
                    x['deviation'], x['rejectBlock']),
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
    print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
