"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --blockIdx=blockIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --plotting                                plot out the correlation matrix? [default: True]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --inputBlockName=inputBlockName           filename for inputs [default: fr]
    --maskOutlierBlocks                       delete outlier trials? [default: False]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                   query what the units will be aligned to? [default: midPeak]
    --alignFolderName=alignFolderName         append a name to the resulting blocks? [default: motion]
    --selector=selector                       filename if using a unit selector
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, PredefinedSplit
from sklearn.utils import shuffle
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler, StandardScaler
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
import seaborn as sns
from math import factorial
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("dark")
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=5, windowSize=(0, 300e-3),
    metaDataToCategories=False,
    transposeToColumns='bin', concatOn='index',
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    calcSubFolder, prefix, **arguments)
from sklearn.preprocessing import scale, robust_scale

def noiseCeil(
        group, dataColNames=None,
        tBounds=None,
        plotting=False, iterMethod='loo',
        corrMethod='pearson',
        maxIter=1e6):
    # print('Group shape is {}'.format(group.shape))
    dataColMask = group.columns.isin(dataColNames)
    groupData = group.loc[:, dataColMask]
    indexColMask = ~group.columns.isin(dataColNames)
    indexCols = group.columns[indexColMask]
    keepIndexCols = indexCols[~indexCols.isin(['segment', 'originalIndex', 't'])]
    # 
    if tBounds is not None:
        maskX = (groupData.columns > tBounds[0]) & (groupData.columns < tBounds[1])
    else:
        maskX = np.ones_like(groupData.columns).astype(np.bool)
    #
    nSamp = groupData.shape[0]
    if iterMethod == 'loo':
        loo = LeaveOneOut()
        allCorr = pd.Series(index=groupData.index)
        iterator = loo.split(groupData)
        for idx, (train_index, test_index) in enumerate(iterator):
            refX = groupData.iloc[train_index, maskX].mean()
            testX = pd.Series(
                groupData.iloc[test_index, maskX].to_numpy().squeeze(),
                index=refX.index)
            allCorr.iloc[test_index] = refX.corr(
                testX, method=corrMethod)
            if idx > maxIter:
                break
        allCorr.dropna(inplace=True)
    elif iterMethod == 'half':
        nChoose = int(groupData.shape[0] / 2)
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        #
        allCorr = pd.Series(index=range(maxIter))
        allCov = pd.Series(index=range(maxIter))
        allMSE = pd.Series(index=range(maxIter))
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.mean()
            refX = groupData.loc[~groupData.index.isin(testX.index), :]
            refXBar = refX.mean()
            # if refX.mean().isna().any() or testX.mean().isna().any():
            #
            allCorr.iloc[idx] = refX.mean().corr(
                testXBar, method=corrMethod)
            allCov.iloc[idx] = refXBar.cov(testXBar)
            allMSE.iloc[idx] = (
                ((refXBar - testXBar) ** 2)
                .mean())
    #
    # if allCorr.mean() < 0:
    #     pdb.set_trace()
    #     plt.plot(testXBar); plt.plot(refXBar); plt.show()
    #     plt.plot(testX.transpose(), 'b'); plt.plot(refX.transpose(), 'r'); plt.show()
    resultDF = pd.DataFrame(
        {
            'noiseCeil': allCorr.mean(),
            'noiseCeilStd': allCorr.std(),
            'covariance': allCov.mean(),
            'covarianceStd': allCov.std(),
            'mse': allMSE.mean(),
            'mseStd': allMSE.std()}, index=[group.index[0]])
    for cN in keepIndexCols:
        resultDF.loc[group.index[0], cN] = group.loc[group.index[0], cN]
    return resultDF


if __name__ == "__main__":
    testVar = None
    groupBy = [
        'feature', 'electrode',
        'RateInHz', 'nominalCurrent']
    # resultMeta = {
    #     'noiseCeil': np.float,
    #     'noiseCeilStd': np.float,
    #     'covariance': np.float,
    #     'covarianceStd': np.float,
    #     'mse': np.float,
    #     'mseStd': np.float
    #     }
    resultMeta = None
    recalc = True
    if recalc:
        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = preproc.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = preproc.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
        funKWArgs = dict(
                tBounds=None,
                plotting=False, iterMethod='half',
                corrMethod='pearson', maxIter=500)
        # daskClient = Client()
        # daskComputeOpts = {}
        daskComputeOpts = dict(
            scheduler='processes'
            # scheduler='single-threaded'
            )
        resDF = ash.splitApplyCombine(
            dataDF, fun=noiseCeil, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, useDask=True,
            daskPersist=True, daskProgBar=True, daskResultMeta=resultMeta,
            daskComputeOpts=daskComputeOpts,
            reindexFromInput=False)
        #
        # nObs = dataDF.reset_index().groupby(groupBy)['feature'].value_counts()
        # resDF = ash.applyFunGrouped(
        #     dataDF,
        #     groupBy, testVar,
        #     fun=noiseCeil, funArgs=[],
        #     funKWargs=dict(
        #         plotting=arguments['plotting'],
        #         iterMethod='half', maxIter=1e2),
        #     resultNames=resultNames, plotting=False)
        for rN in resDF.columns:
            resDF[rN].to_hdf(resultPath, rN, format='fixed')
        noiseCeil = resDF['noiseCeil']
        covar = resDF['covariance']
        mse = resDF['mse']
    else:
        noiseCeil = pd.read_hdf(resultPath, 'noiseCeil')
        covar = pd.read_hdf(resultPath, 'covariance')
        mse = pd.read_hdf(resultPath, 'mse')
        resDF = pd.concat({
            'noiseCeil': noiseCeil,
            'covariance': covar,
            'mse': mse,
            }, axis='columns')
    for cN in ['covariance', 'mse']:
        robScaler = RobustScaler(quantile_range=(5, 95))
        # inputDF = resDF[cN].unstack(level='feature')
        #
        robScaler.fit(resDF.loc[resDF[cN].notna(), cN].to_numpy().reshape(-1, 1))
        preScaled = (robScaler.transform(resDF[cN].to_numpy().reshape(-1, 1)))
        resDF[cN + '_q_scale'] = pd.Series(
            preScaled.squeeze(),
            index=resDF[cN].index)
        scaledMask = np.abs(preScaled.squeeze()) < 2
        # scaledMask = pd.Series(
        #     np.abs(preScaled.squeeze()) < 2,
        #     index=resDF[cN].index)
        mmScaler = MinMaxScaler()
        mmScaler.fit(resDF.loc[scaledMask, cN].to_numpy().reshape(-1, 1))
        resDF[cN + '_scaled'] = mmScaler.transform(resDF[cN].to_numpy().reshape(-1, 1))
    exportToDeepSpine = True
    if exportToDeepSpine:
        deepSpineExportPath = os.path.join(
            alignSubFolder,
            prefix + '_{}_{}_export.h5'.format(
                arguments['inputBlockName'], arguments['window']))
        for cN in ['noiseCeil', 'covariance', 'covariance_q_scale']:
            resDF[cN].to_hdf(deepSpineExportPath, cN)
    pdb.set_trace()
    mask = pd.DataFrame(
        False,
        index=resDF['noiseCeil'].index)
    trialInfo = resDF['noiseCeil'].index.to_frame().reset_index(drop=True)
    dropColumns = []
    for cN in dropColumns:
        mask.loc[trialInfo['feature'] == cN] = True
    dropElectrodes = []
    dropIndex = trialInfo['electrode'].isin(dropElectrodes)
    mask.loc[dropIndex] = True
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'])
        pdfName = os.path.join(figureOutputFolder, 'noise_ceil_halves.pdf')
        with PdfPages(pdfName) as pdf:
            for cN in ['noiseCeil', 'covariance_q_scale', 'mse_q_scale']:
                plotIndex = pd.Index(
                    trialInfo.loc[:, [
                        'electrode',
                        'RateInHz', 'nominalCurrent']])
                plotDF = resDF[cN].reset_index(drop=True).set_index(plotIndex)
                fig, ax = plt.subplots(figsize=(12, 12))
                sns.heatmap(
                    plotDF, mask=mask.to_numpy(),
                    ax=ax, center=0, vmin=-1, vmax=1)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
                ax.set_title(cN)
                plt.tight_layout()
                pdf.savefig()
                plt.show()
            # for pageIdx, (pageName, pageGroup) in enumerate(tqdm(noiseCeil.groupby('electrode'))):

    if arguments['plotting']:
        plotNoiseCeil = (
            resDF['noiseCeil']
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='columns')
            .stack())
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.distplot(plotNoiseCeil, ax=ax, kde=False)
        ax.set_title('Noise ceiling histogram')
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Count')
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_histogram.pdf'))
        # 
        plotCovar = (
            resDF['covariance_q_scale']
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='columns')
            .stack())
        plotDF = pd.concat(
            {'noiseCeil': plotNoiseCeil,
            'covar': plotCovar}, axis='columns').reset_index()
        plotDF['nominalCurrent'] *= -1
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            x='covar', y='noiseCeil',
            hue='electrode', size='nominalCurrent', style='feature',
            data=plotDF, ax=ax, alpha=0.75, sizes=(50, 500))
        ax.set_xlabel('Scaled Covariance')
        ax.set_ylabel('Reliability')
        ax.set_xlim([-.2, 1])
        # ax.set_ylim([-1, 1])
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_scatterplot.pdf'))
        plt.show()
    keepMask = ((plotNoiseCeil > 0.4) & (plotCovar > 0.1))
    keepFeats = plotNoiseCeil[keepMask].index.to_frame().reset_index(drop=True)
