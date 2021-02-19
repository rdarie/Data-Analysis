"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                   which experimental day to analyze
    --blockIdx=blockIdx                         which trial to analyze [default: 1]
    --processAll                                process entire experimental day? [default: False]
    --lazy                                      load from raw, or regular? [default: False]
    --verbose                                   print diagnostics? [default: False]
    --exportToDeepSpine                         look for a deepspine exported h5 and save these there [default: False]
    --plotting                                  plot out the correlation matrix? [default: True]
    --showPlots                                 show the plots? [default: False]
    --analysisName=analysisName                 append a name to the resulting blocks? [default: default]
    --inputBlockSuffix=inputBlockSuffix         filename for inputs [default: fr]
    --maskOutlierBlocks                         delete outlier trials? [default: False]
    --window=window                             process with short window? [default: long]
    --unitQuery=unitQuery                       how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                     query what the units will be aligned to? [default: midPeak]
    --alignFolderName=alignFolderName           append a name to the resulting blocks? [default: motion]
    --selector=selector                         filename if using a unit selector
    --amplitudeFieldName=amplitudeFieldName     what is the amplitude named? [default: nominalCurrent]
"""
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)
# from tqdm import tqdm
import pdb
import os, random
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, PredefinedSplit
from sklearn.utils import shuffle
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler, StandardScaler

from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from math import factorial
from sklearn.preprocessing import scale, robust_scale
from dask.distributed import Client

#
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
        arguments['inputBlockSuffix'], arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockSuffix'], arguments['window']))
# e.g. resultPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202006171300-Peep/emgLoRes/stim/_emg_XS.nix'
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(0, 9e-3),
    metaDataToCategories=False,
    # getMetaData=[
    #     'RateInHz', 'feature', 'electrode',
    #     arguments['amplitudeFieldName'], 'stimPeriod',
    #     'pedalMovementCat', 'pedalSizeCat', 'pedalDirection',
    #     'stimCat', 'originalIndex', 'segment', 't'],
    transposeToColumns='bin', concatOn='index',
    verbose=False, procFun=None))
#
alignedAsigsKWargs['procFun'] = ash.genDetrender(
    timeWindow=(alignedAsigsKWargs['windowSize'][-1] - 1e-3, alignedAsigsKWargs['windowSize'][-1]))
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    calcSubFolder, prefix, **arguments)

from lmfit.models import ExponentialModel, GaussianModel, ConstantModel

np.random.seed(0)
exp_mod = ExponentialModel(prefix='exp_')
c_mod = ConstantModel(prefix='const_')
gauss1 = GaussianModel(prefix='g1_')
gauss2 = GaussianModel(prefix='g2_')
gauss3 = GaussianModel(prefix='g3_')
gauss4 = GaussianModel(prefix='g4_')
mod = gauss1 + gauss2 + gauss3 + gauss4 + exp_mod + c_mod

pars = exp_mod.make_params()
pars.update(gauss1.make_params())
pars.update(gauss2.make_params())
pars.update(gauss3.make_params())
pars.update(gauss4.make_params())
pars.update(c_mod.make_params())
#
pars['exp_decay'].set(min=.1, max=5)
pars['g1_center'].set(min=.5, max=10)
pars['g2_center'].set(expr='g1_center + 2 * g1_sigma + 2 * g2_sigma')
pars['g3_center'].set(expr='g1_center + 2 * g1_sigma + 4 * g2_sigma + 2 * g3_sigma')
pars['g4_center'].set(expr='g1_center + 2 * g1_sigma + 4 * g2_sigma + 4 * g3_sigma + 2 * g4_sigma')
for idx in range(4):
    pars['g{}_sigma'.format(idx+1)].set(value=.3, min=.2, max=.5)


def applyModel(dataSrs):
    x = dataSrs.index.to_numpy(dtype=np.float) * 1e3
    y = dataSrs.to_numpy(dtype=np.float)
    thesePars = pars.copy()
    #
    prelim_stats = np.percentile(y, q=[25, 75])
    iqr = prelim_stats[1] - prelim_stats[0]
    if iqr == 0:
        return pd.Series(0, index=dataSrs.index)
    thesePars['const_c'].set(value=0, min=-3*iqr, max=3*iqr)
    thesePars['exp_amplitude'].set(
        value=np.median(y[x <= 1.4]),
        min=-3*iqr, max=3*iqr)
    exp_guess = exp_mod.guess(y, x=x)
    thesePars.update(exp_guess)
    #
    init_exp = exp_mod.eval(exp_guess, x=x)
    init_resid = y - init_exp
    resid_stats = np.percentile(init_resid, q=[25, 75])
    iqr = resid_stats[1] - resid_stats[0]
    thesePars['g1_center'].set(value=np.random.uniform(0, 2))
    #
    thesePars['g1_amplitude'].set(value=iqr, min=0, max=3 * iqr)
    thesePars['g2_amplitude'].set(value=(-1) * iqr, min=-3 * iqr, max=0)
    thesePars['g3_amplitude'].set(value=iqr, min=0, max=3 * iqr)
    thesePars['g4_amplitude'].set(value=(-1) * iqr, min=-3 * iqr, max=0)
    #
    init = mod.eval(thesePars, x=x)
    out = mod.fit(y, thesePars, x=x)
    #
    fig, ax = plotLmFit(x, y, init, out)
    return pd.Series(out.best_fit, index=dataSrs.index)


def plotLmFit(x, y, init, out):
    print(out.fit_report())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, 'b')
    axes[0].plot(x, init, 'k--', label='initial fit')
    axes[0].plot(x, out.best_fit, 'r-', label='best fit')
    axes[0].legend(loc='best')
    comps = out.eval_components(x=x)
    axes[1].plot(x, y, 'b')
    axes[1].plot(x, comps['g1_'], 'c--', lw=2, label='Gaussian component 1')
    axes[1].plot(x, comps['g2_'], 'm--', lw=2, label='Gaussian component 2')
    axes[1].plot(x, comps['g3_'], 'y--', lw=2, label='Gaussian component 3')
    axes[1].plot(x, comps['g4_'], 'g--', lw=2, label='Gaussian component 4')
    axes[1].plot(x, comps['exp_'] + comps['const_'], 'k--', lw=2, label='Offset exponential component')
    # axes[1].axhline(comps['const_'], c='r', lw=2, label='Constant component')
    axes[1].legend(loc='best')
    return fig, axes


def shapeFit(
        group, dataColNames=None,
        tBounds=None,
        plotting=False, iterMethod='loo',
        modelFun=None, corrMethod='pearson',
        maxIter=1e6):
    # print('os.getpid() = {}'.format(os.getpid()))
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
    fig, ax = plt.subplots()
    ax.plot(groupData.T)
    ax.text(
        0.95, 0.95,
        '{}'.format(group.iloc[0, :].loc[keepIndexCols]),
        horizontalalignment='right', verticalalignment='top',
        transform=ax.transAxes)
    plt.show()
    if iterMethod == 'loo':
        loo = LeaveOneOut()
        allCorr = pd.Series(index=groupData.index)
        iterator = loo.split(groupData)
        for idx, (train_index, test_index) in enumerate(iterator):
            refX = groupData.iloc[train_index, maskX].mean()
            testX = modelFun(refX)
            '''
            testX = pd.Series(
                groupData.iloc[test_index, maskX].to_numpy().squeeze(),
                index=refX.index)
            '''
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
        allCorr = pd.Series(index=range(maxIter))
        allCov = pd.Series(index=range(maxIter))
        allMSE = pd.Series(index=range(maxIter))
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.iloc[:, maskX].mean()
            refX = groupData.loc[~groupData.index.isin(testX.index), :]
            refXBar = refX.iloc[:, maskX].mean()
            '''
            allCorr.iloc[idx] = refX.mean().corr(
                testXBar, method=corrMethod)
            allCov.iloc[idx] = refXBar.cov(testXBar)
            allMSE.iloc[idx] = (
                ((refXBar - testXBar) ** 2)
                .mean())
            '''
    elif iterMethod == 'chooseN':
        nChoose = max(groupData.shape[0] - 2, int(groupData.shape[0] / 2), 2)
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        allCorr = pd.Series(index=range(maxIter))
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.iloc[:, maskX].mean()
            testX = modelFun(testXBar)
    # if allCorr.mean() < 0:
    #     pdb.set_trace()
    #     plt.plot(testXBar); plt.plot(refXBar); plt.show()
    #     plt.plot(testX.transpose(), 'b'); plt.plot(refX.transpose(), 'r'); plt.show()
    plt.show()
    resultDF = pd.DataFrame(
        {
            'noiseCeil': allCorr.mean(),
            'noiseCeilStd': allCorr.std(),
            # 'covariance': allCov.mean(),
            # 'covarianceStd': allCov.std(),
            # 'mse': allMSE.mean(),
            # 'mseStd': allMSE.std()
            },
        index=[group.index[0]])
    for cN in keepIndexCols:
        resultDF.loc[group.index[0], cN] = group.loc[group.index[0], cN]
    # print(os.getpid())
    return resultDF


if __name__ == "__main__":
    testVar = None
    conditionNames = [
        'electrode',
        # 'RateInHz',
        'nominalCurrent']
    groupBy = ['feature'] + conditionNames
    # resultMeta = {
    #     'noiseCeil': np.float,
    #     'noiseCeilStd': np.float,
    #     'covariance': np.float,
    #     'covarianceStd': np.float,
    #     'mse': np.float,
    #     'mseStd': np.float
    #     }
    alignedAsigsKWargs['getMetaData'] = conditionNames
    for nM in ['RateInHz', 'stimCat', 'originalIndex', 'segment', 't']:
        if nM not in alignedAsigsKWargs['getMetaData']:
            alignedAsigsKWargs['getMetaData'].append(nM)
    useCachedResult = True
    if not (useCachedResult and os.path.exists(resultPath)):
        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = preproc.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = preproc.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
        egFeatName = dataDF.index.get_level_values('feature').unique()[0]
        breakDownData, breakDownText, fig, ax = asp.printBreakdown(
            dataDF.xs(egFeatName, level='feature', drop_level=False),
            'RateInHz', 'electrode', 'nominalCurrent')
        funKWArgs = dict(
                tBounds=[1.3e-3, 8e-3],
                modelFun=applyModel,
                plotting=False, iterMethod='chooseN',
                maxIter=3)
        # daskComputeOpts = {}
        daskComputeOpts = dict(
            # scheduler='threads'
            # scheduler='processes'
            scheduler='single-threaded'
            )
        exampleOutput = pd.DataFrame(
        {
            'noiseCeil': float(1),
            'noiseCeilStd': float(1),
            'covariance': float(1),
            'covarianceStd': float(1),
            'mse': float(1),
            'mseStd': float(1)}, index=[0])
        daskClient = Client()
        resDF = ash.splitApplyCombine(
            dataDF, fun=shapeFit, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, useDask=True,
            daskPersist=True, 
            daskProgBar=False,
            daskResultMeta=None,
            daskComputeOpts=daskComputeOpts, nPartitionMultiplier=2,
            reindexFromInput=False)
        #
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
    if arguments['exportToDeepSpine']:
        deepSpineExportPath = os.path.join(
            alignSubFolder,
            prefix + '_{}_{}_export.h5'.format(
                arguments['inputBlockSuffix'], arguments['window']))
        for cN in ['noiseCeil', 'covariance', 'covariance_q_scale']:
            resDF[cN].to_hdf(deepSpineExportPath, cN)
    #
    # pdb.set_trace()
    trialInfo = resDF['noiseCeil'].index.to_frame().reset_index(drop=True)
    dropColumns = []
    dropElectrodes = []
    
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'])
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
        pdfName = os.path.join(figureOutputFolder, 'noise_ceil_halves.pdf')
        with PdfPages(pdfName) as pdf:
            plotIndex = pd.MultiIndex.from_frame(
                trialInfo.loc[:, [
                    'electrode',
                    'RateInHz', 'nominalCurrent', 'feature']])
            for cN in ['noiseCeil', 'covariance_q_scale', 'mse_q_scale']:
                plotDF = (
                    resDF[cN]
                    .unstack(level='feature')
                    .drop(dropElectrodes, axis='index', level='electrode')
                    .drop(dropColumns, axis='columns'))
                plotDF.index = pd.MultiIndex.from_frame(
                    plotDF.index.to_frame(index=False).loc[
                        :, ['electrode', 'RateInHz', 'nominalCurrent']])
                fig, ax = plt.subplots(figsize=(12, 12))
                sns.heatmap(
                    plotDF, ax=ax,
                    center=0, vmin=-1, vmax=1)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
                ax.set_title(cN)
                plt.tight_layout()
                pdf.savefig()
                if arguments['showPlots']:
                    plt.show()
                else:
                    plt.close()
    keepMask = ((resDF['noiseCeil'] > 0.4) & (resDF['covariance_q_scale'] > 0.1))
    keepFeats = (resDF.loc[keepMask, 'noiseCeil'].index.to_frame().reset_index(drop=True).groupby(['RateInHz', 'nominalCurrent', 'electrode'])['feature'])
    keepFeats.name = 'numFeatures'
    #
    minReliabilityPerEMG = (
        resDF['noiseCeil']
        .unstack(level='feature')
        .drop(dropElectrodes, axis='index', level='electrode')
        .drop(dropColumns, axis='columns').quantile(0.75))
    minReliabilityPerElectrode = (
        resDF['noiseCeil']
        .unstack(level='electrode')
        .drop(dropColumns, axis='index', level='feature')
        .drop(dropElectrodes, axis='columns').quantile(0.75))
    if arguments['plotting']:
        plotCovar = (
            resDF.loc[keepMask, 'covariance_q_scale']
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='index', level='feature'))
        plotNoiseCeil = (
            resDF.loc[keepMask, 'noiseCeil']
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='index', level='feature')
            )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.distplot(plotNoiseCeil, ax=ax, kde=False)
        ax.set_title('Noise ceiling histogram')
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Count')
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_histogram.pdf'))
        if arguments['showPlots']:
            plt.show()
        else:
            plt.close()
        # 
    if arguments['plotting']:
        plotDF = pd.concat(
            {
                'noiseCeil': plotNoiseCeil,
                'covar': plotCovar}, axis='columns').reset_index()
        plotDF['nominalCurrent'] *= -1
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            x='covar', y='noiseCeil',
            hue='electrode', size='nominalCurrent', style='feature',
            markers=EMGStyleMarkers,
            data=plotDF, ax=ax, alpha=0.75, sizes=(10, 100))
        ax.set_xlabel('Scaled Covariance')
        ax.set_ylabel('Reliability')
        ax.set_xlim([-.2, 1])
        # ax.set_ylim([-1, 1])
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_scatterplot.pdf'))
        if arguments['showPlots']:
            plt.show()
        else:
            plt.close()
    if arguments['plotting']:
        plotDF = (
            resDF.loc[keepMask, :]
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='index', level='feature')
            .reset_index()
            )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(x='feature', y='noiseCeil', data=plotDF, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-60)
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_per_emg.pdf'))
        if arguments['showPlots']:
            plt.show()
        else:
            plt.close()
