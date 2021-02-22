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

useCachedResult = False
if not (useCachedResult and os.path.exists(resultPath)):
    if os.path.exists(resultPath):
        os.remove(resultPath)
# e.g. resultPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202006171300-Peep/emgLoRes/stim/_emg_XS.nix'
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(-100e3, 20e-3),
    metaDataToCategories=False,
    # getMetaData=[
    #     'RateInHz', 'feature', 'electrode',
    #     arguments['amplitudeFieldName'], 'stimPeriod',
    #     'pedalMovementCat', 'pedalSizeCat', 'pedalDirection',
    #     'stimCat', 'originalIndex', 'segment', 't'],
    transposeToColumns='bin', concatOn='index',
    verbose=False, procFun=None))
#
'''
alignedAsigsKWargs['procFun'] = ash.genDetrender(
    timeWindow=(alignedAsigsKWargs['windowSize'][-1] - 1e-3, alignedAsigsKWargs['windowSize'][-1]))
'''
alignedAsigsKWargs['procFun'] = ash.genDetrender(
    timeWindow=(alignedAsigsKWargs['windowSize'][0], -1e-3))

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    calcSubFolder, prefix, **arguments)

from lmfit.models import ExponentialModel, GaussianModel, ConstantModel

np.random.seed(0)
exp1 = ExponentialModel(prefix='exp1_')
exp2 = ExponentialModel(prefix='exp2_')
# const = ConstantModel(prefix='const_')
gauss1 = GaussianModel(prefix='g1_')
gauss2 = GaussianModel(prefix='g2_')
gauss3 = GaussianModel(prefix='g3_')
gauss4 = GaussianModel(prefix='g4_')
exp_mod = exp1 + exp2  # + const
gauss_mod = gauss1 + gauss2 + gauss3 + gauss4
full_mod = exp_mod + gauss_mod
#
expPars = exp1.make_params()
expPars.update(exp2.make_params())
# expPars.update(const.make_params())
expPars['exp1_decay'].set(value=100, min=10, max=1000)
expPars['exp2_decay'].set(value=.4, min=.2, max=.6)
expPars.add(name='exp2_ratio', value=-1, min=-1.1, max=-0.9, vary=False)
expPars['exp2_amplitude'].set(expr='exp2_ratio * exp1_amplitude')
#
gaussPars = gauss1.make_params()
gaussPars.update(gauss2.make_params())
gaussPars.update(gauss3.make_params())
gaussPars.update(gauss4.make_params())
#
gaussPars.add(name='g1_offset', min=0.1, max=3)
gaussPars.add(name='g2_offset', value=2, min=0, max=10)
gaussPars['g1_center'].set(expr='2 * g1_sigma + g1_offset')
gaussPars['g2_center'].set(expr='g1_center + 2 * g1_sigma + 2 * g2_sigma + g2_offset')
gaussPars['g3_center'].set(expr='g2_center + 2 * g2_sigma + 2 * g3_sigma')
gaussPars['g4_center'].set(expr='g3_center + 2 * g3_sigma + 2 * g4_sigma')
gaussPars.add(name='g4_ratio', value=.1, min=0, max=2)
gaussPars['g4_amplitude'].set(expr='g4_ratio * g2_amplitude')

dependentParamNames = [
    'g1_center', 'g2_center',
    'g3_center', 'g4_center',
    'g4_amplitude',
    'exp2_amplitude']
fullPars = expPars.copy()
fullPars.update(gaussPars)

modelColumnNames = [
    'g4_amplitude', 'g4_center', 'g4_sigma', 'g3_amplitude', 'g3_center',
    'g3_sigma', 'g2_amplitude', 'g2_center', 'g2_sigma', 'g1_amplitude',
    'g1_center', 'g1_sigma', 'exp2_amplitude', 'exp2_decay',
    'exp1_amplitude', 'exp1_decay', 'chisqr', 'r2']


def applyModel(
        x, y,
        method='nelder', scoreBounds=None,
        verbose=True, plotting=False):
    ##############################################
    prelim_stats = np.percentile(y, q=[1, 99])
    iqr = prelim_stats[1] - prelim_stats[0]
    dummy = pd.Series(0, index=x)
    dummyAnns = pd.Series({key: 0 for key in modelColumnNames})
    if iqr == 0:
        return pd.concat([dummy, dummyAnns])
    #
    guessTauFast = 0.5  # msec
    guessTauSlow = 200  # msec
    ampGuess = np.mean(
        y / (np.exp(-x/guessTauSlow) - np.exp(-x/guessTauFast)),
        axis=None)
    if ampGuess == 0:
        return pd.concat([dummy, dummyAnns])
    # expPars['const_c'].set(value=0, min=-iqr, max=iqr, vary=False)
    #
    expPars['exp1_amplitude'].set(
        value=ampGuess,
        )
    if ampGuess > 0:
        expPars['exp1_amplitude'].set(max=2*ampGuess, min=0)
    else:
        expPars['exp1_amplitude'].set(max=0, min=2*ampGuess)
    #
    exp_out = exp_mod.fit(y, expPars, x=x, method=method)
    intermed_y = y - exp_out.best_fit
    intermed_stats = np.percentile(intermed_y, q=[1, 99])
    intermed_iqr = intermed_stats[1] - intermed_stats[0]
    if verbose:
        print(exp_out.fit_report())
    #
    for idx in range(4):
        gaussPars['g{}_sigma'.format(idx+1)].set(
            value=np.random.uniform(.1, .3),
            min=.1, max=.75)
    gaussPars['g1_offset'].set(value=np.random.uniform(.5, 1.))
    # positives
    gaussPars['g1_amplitude'].set(
        value=0,  # vary=False,
        min=0, max=2 * intermed_iqr)
    # negatives
    gaussPars['g2_amplitude'].set(
        value=0,  # vary=False,
        min=0, max=2 * intermed_iqr)
    gaussPars['g3_amplitude'].set(
        value=0,  # vary=False,
        max=0, min=-2 * intermed_iqr)
    freezeGaussians = False
    if freezeGaussians:
        gaussPars['g1_amplitude'].set(value=0, vary=False)
        gaussPars['g2_amplitude'].set(value=0, vary=False)
        gaussPars['g4_amplitude'].set(value=0, vary=False)
        for idx in range(4):
            gaussPars['g{}_sigma'.format(idx+1)].set(value=.2, vary=False)
    #
    gauss_out = gauss_mod.fit(intermed_y, gaussPars, x=x, method=method)
    if verbose:
        print(gauss_out.fit_report())
    #
    for nM, value in exp_out.best_values.items():
        if nM not in dependentParamNames:
            fullPars[nM].set(value=value)
    for nM, value in gauss_out.best_values.items():
        if nM not in dependentParamNames:
            fullPars[nM].set(value=value)
    # print(out.fit_report())
    init = full_mod.eval(fullPars, x=x)
    out = full_mod.fit(y, fullPars, x=x, method=method)
    outSrs = pd.Series(out.best_fit, index=x)
    outParams = pd.Series(out.best_values)
    if scoreBounds is not None:
        maskX = (x >= scoreBounds[0]) & (x < scoreBounds[1])
    else:
        maskX = np.ones_like(x).astype(np.bool)
    # pdb.set_trace()
    chisqr = ((y[maskX] - out.best_fit[maskX]) ** 2).sum()
    r2 = 1 - (chisqr / (y[maskX] ** 2).sum())
    outStats = pd.Series(
        {
            'chisqr': chisqr,
            'r2': r2})
    # pdb.set_trace()
    if plotting:
        fig, ax = plotLmFit(x, y, init, out, verbose=verbose)
        ax[1].set_title('R^2 = {}'.format(r2))
    return pd.concat([outSrs, outParams, outStats])


def plotLmFit(x, y, init, out, verbose=False):
    if verbose:
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
    axes[1].plot(
        x, comps['exp1_'] + comps['exp2_'],  # + comps['const_'],
        'k--', lw=2, label='Offset exponential component')
    axes[1].legend(loc='best')
    return fig, axes


def shapeFit(
        group, dataColNames=None,
        tBounds=None, verbose=False,
        scoreBounds=None,
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
        maskX = (groupData.columns >= tBounds[0]) & (groupData.columns < tBounds[1])
    else:
        maskX = np.ones_like(groupData.columns).astype(np.bool)
    #
    nSamp = groupData.shape[0]
    if (not (groupData == 1).all(axis=None)) and plotting:
        fig, ax = plt.subplots()
        ax.plot(groupData.T)
        ax.text(
            0.95, 0.95,
            '{}'.format(group.iloc[0, :].loc[keepIndexCols]),
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes)
        # plt.show()
    groupT = groupData.columns[maskX].to_numpy(dtype=np.float)
    if scoreBounds is not None:
        scBnds = [1e3 * (sb - groupT[0]) for sb in scoreBounds]
    groupT = 1e3 * (groupT - groupT[0])
    groupT = groupT - groupT[0]
    # groupT = groupT - .666
    outList = []
    if iterMethod == 'loo':
        loo = LeaveOneOut()
        iterator = loo.split(groupData)
        for idx, (train_index, test_index) in enumerate(iterator):
            refX = groupData.iloc[train_index, maskX].mean()
            testX = modelFun(groupT, refX.to_numpy())
            if idx > maxIter:
                break
    elif iterMethod == 'half':
        nChoose = int(groupData.shape[0] / 2)
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.iloc[:, maskX].mean()
            refX = groupData.loc[~groupData.index.isin(testX.index), :]
            refXBar = refX.iloc[:, maskX].mean()
    elif iterMethod == 'chooseN':
        nChoose = max(groupData.shape[0] - 2, int(groupData.shape[0] / 2), 2)
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.iloc[:, maskX].mean()
            resultsSrs = modelFun(
                groupT, testXBar.to_numpy(),
                scoreBounds=scBnds,
                verbose=verbose, plotting=plotting)
            # TODO: return best model too
            outList.append(resultsSrs)
    # if allCorr.mean() < 0:
    #     pdb.set_trace()
    #     plt.plot(testXBar); plt.plot(refXBar); plt.show()
    #     plt.plot(testX.transpose(), 'b'); plt.plot(refX.transpose(), 'r'); plt.show()
    prelimDF = pd.DataFrame(outList)
    resultDF = prelimDF.loc[prelimDF['r2'].argmax(), :].to_frame().T
    resultDF.index = [group.index[0]]
    if (not (groupData == 1).all(axis=None)) and plotting:
        plt.show()
    for cN in keepIndexCols:
        resultDF.loc[group.index[0], cN] = group.loc[group.index[0], cN]
    # print(os.getpid())
    return resultDF


if __name__ == "__main__":
    testVar = None
    conditionNames = [
        'electrode',
        # 'RateInHz',
        arguments['amplitudeFieldName']]
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
    if not (useCachedResult and os.path.exists(resultPath)):
        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = preproc.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = preproc.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
        featNames = dataDF.index.get_level_values('feature')
        egFeatName = featNames.unique()[0]
        # breakDownData, breakDownText, fig, ax = asp.printBreakdown(
        #     dataDF.xs(egFeatName, level='feature', drop_level=False),
        #     'RateInHz', 'electrode', arguments['amplitudeFieldName'])
        # daskComputeOpts = {}
        daskComputeOpts = dict(
            # scheduler='threads'
            scheduler='processes'
            # scheduler='single-threaded'
            )
        '''
        exampleOutput = pd.DataFrame(
            {
                'noiseCeil': float(1),
                'noiseCeilStd': float(1),
                'covariance': float(1),
                'covarianceStd': float(1),
                'mse': float(1),
                'mseStd': float(1)},
            index=[0])
        '''
        #############################
        #############################
        featNames = dataDF.index.get_level_values('feature')
        elecNames = dataDF.index.get_level_values('electrode')
        rates = dataDF.index.get_level_values('RateInHz')
        amps = dataDF.index.get_level_values(arguments['amplitudeFieldName'])
        print('Available rates are {}'.format(np.unique(rates)))
        '''
        dbMask = (
            featNames.str.contains('rostralY_e12') &
            elecNames.str.contains('caudalY_e11') &
            (rates < 60) &
            (amps < -500)
            )
        '''
        dbMask = (rates < 60)
        dataDF = dataDF.loc[dbMask, :]
        #############################
        #############################
        daskClient = Client()
        funKWArgs = dict(
                # tBounds=[1.3e-3, 9.9e-3],
                tBounds=[1.2e-3, 19.7e-3],
                scoreBounds=[1.2e-3, 9.7e-3],
                modelFun=applyModel,
                iterMethod='chooseN',
                plotting=False, verbose=False,
                maxIter=100)
        resDF = ash.splitApplyCombine(
            dataDF,
            fun=shapeFit, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, useDask=True,
            daskPersist=True,
            daskProgBar=True,
            daskResultMeta=None,
            daskComputeOpts=daskComputeOpts,
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
        resDF.set_index(modelColumnNames, inplace=True, append=True)
        resDF.to_hdf(resultPath, 'lmfit_lfp')
        # pdb.set_trace()
        presentNames = [cn for cn in resDF.index.names if cn in dataDF.index.names]
        meansDF = dataDF.groupby(presentNames).mean()
        meansDF.to_hdf(resultPath, 'flp')
    else:
        resDF = pd.read_hdf(resultPath, 'lmfit_lfp')
        meansDF = pd.read_hdf(resultPath, 'flp')
    try:
        meansDF = dataDF.groupby(groupBy).mean()
    except Exception:
        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = preproc.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = preproc.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
    allIdxNames = resDF.index.names
    resDF = resDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    pdb.set_trace()
    resDF.columns = (resDF.columns + 1.2) / 1e3
    allIdxNames = meansDF.index.names
    meansDF = meansDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    #
    plotDF = pd.concat(
        {'fit': resDF, 'target': meansDF},
        names=['regrID'] + list(resDF.index.names))
    plotDF.dropna(axis='columns', inplace=True)
    plotDF.columns.name = 'bin'
    plotDF = plotDF.stack().to_frame(name='signal')
    sns.relplot(
        data=plotDF.reset_index(),
        x='bin', y='signal', style='regrID', kind='line', row='feature', col='nominalCurrent')
    plt.show()
    pdb.set_trace()

    '''
    if arguments['exportToDeepSpine']:
        deepSpineExportPath = os.path.join(
            alignSubFolder,
            prefix + '_{}_{}_export.h5'.format(
                arguments['inputBlockSuffix'], arguments['window']))
    #
    # pdb.set_trace()
    trialInfo = resDF.index.to_frame().reset_index(drop=True)
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
    '''