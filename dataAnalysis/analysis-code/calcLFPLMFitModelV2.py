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
import os, sys
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
from tqdm import tqdm
import pdb, traceback
import random
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
    decimate=1, windowSize=(-100e3, 100e-3),
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
from lmfit import Model, CompositeModel

# np.random.seed(0)


def offsetExponential(x, amplitude=1, decay=1, offset=0):
    res = amplitude * np.exp(-(x - offset)/decay)
    if isinstance(x, np.ndarray):
        res[x < offset] = 0
        return res
    else:
        if x >= offset:
            return res
        else:
            return 0


exp1 = Model(offsetExponential, prefix='exp1_')
exp2 = Model(offsetExponential, prefix='exp2_')
p1 = GaussianModel(prefix='p1_')
n1 = GaussianModel(prefix='n1_')
p2 = GaussianModel(prefix='p2_')
p3 = GaussianModel(prefix='p3_')
n2 = GaussianModel(prefix='n2_')
p4 = GaussianModel(prefix='p4_')
exp_mod = exp1 + exp2  # + const
gauss_mod = p1 + n1 + p2 + p3 + n2 + p4
full_mod = exp_mod + gauss_mod
#
expPars = exp1.make_params()
expPars.update(exp2.make_params())
# expPars.update(const.make_params())
expPars['exp1_amplitude'].set(value=1)
expPars['exp1_decay'].set(value=200, min=10, max=500)
expPars['exp2_decay'].set(value=2., min=.1, max=2.)
'''
expPars.add(name='exp2_ratio', value=-1, vary=False)
expPars['exp1_offset'].set(value=0.7, vary=False)
expPars['exp2_offset'].set(value=1.2, vary=False)
'''
#
expPars.add(name='exp2_ratio', value=-1., min=-2., max=-0.5)
expPars['exp1_offset'].set(value=0, vary=False)
expPars['exp2_offset'].set(value=0, vary=False)
#
expPars['exp2_amplitude'].set(expr='exp2_ratio * exp1_amplitude')
#
gaussPars = p1.make_params()
gaussPars.update(n1.make_params())
gaussPars.update(p2.make_params())
gaussPars.update(p3.make_params())
gaussPars.update(n2.make_params())
gaussPars.update(p4.make_params())
#
gaussPars.add(name='n1_offset', min=1.1, max=1.3)
gaussPars.add(name='p3_offset', value=0, min=0, max=.25)
gaussPars['n1_center'].set(expr='n1_offset + 2 * n1_sigma')
gaussPars['p1_center'].set(expr='n1_offset - 2 * p1_sigma')
gaussPars['p2_center'].set(expr='n1_center + 2 * n1_sigma + 2 * p2_sigma')
gaussPars['p3_center'].set(expr='p2_center + 2 * p2_sigma + p3_offset + 2 * p3_sigma')
gaussPars['n2_center'].set(expr='p3_center + 2 * p3_sigma + 2 * n2_sigma')
gaussPars['p4_center'].set(expr='n2_center + 2 * n2_sigma + 2 * p4_sigma')
# gaussPars.add(name='p3_ratio', value=1e-3, min=0, max=2)
# gaussPars['p3_amplitude'].set(expr='p3_ratio * p2_amplitude')
gaussPars['n1_sigma'].set(value=100e-3, min=75e-3, max=150e-3)
gaussPars['p1_sigma'].set(value=100e-3, min=75e-3, max=150e-3)
gaussPars['p2_sigma'].set(value=200e-3, min=150e-3, max=300e-3)
gaussPars['p3_sigma'].set(value=200e-3, min=150e-3, max=300e-3)
gaussPars['n2_sigma'].set(value=400e-3, min=300e-3, max=600e-3)
gaussPars['p4_sigma'].set(value=400e-3, min=300e-3, max=1200e-3)

pars = expPars.copy()
pars.update(gaussPars)

dependentParamNames = [
    'n1_center', 'p1_center',
    'p2_center', 'p3_center',
    'n2_center', 'p4_center'
    'exp2_amplitude']

modelColumnNames = [
    'p4_amplitude', 'p4_center', 'p4_sigma', 'n2_amplitude', 'n2_center',
    'n2_sigma', 'p3_amplitude', 'p3_center', 'p3_sigma', 'p2_amplitude',
    'p2_center', 'p2_sigma', 'n1_amplitude', 'n1_center', 'n1_sigma',
    'p1_amplitude', 'p1_center', 'p1_sigma', 'exp2_amplitude', 'exp2_decay',
    'exp2_offset', 'exp1_amplitude', 'exp1_decay', 'exp1_offset', 'chisqr', 'r2']


def applyModel(
        x, y,
        method='nelder', scoreBounds=None,
        verbose=True, plotting=False):
    #
    fullPars = pars.copy()
    dummy = pd.Series(0, index=x)
    dummyAnns = pd.Series({key: 0 for key in modelColumnNames})
    #
    prelim_stats = np.percentile(y, q=[1, 99])
    iqr = prelim_stats[1] - prelim_stats[0]
    if iqr == 0:
        return dummy, dummyAnns
    #
    signalGuess = exp_mod.eval(fullPars, x=x)
    ampGuess = np.nanmedian(y / signalGuess, axis=None)
    if ampGuess == 0:
        return dummy, dummyAnns
    #
    try:
        fullPars['exp1_amplitude'].set(
            value=ampGuess,
            )
        if ampGuess > 0:
            fullPars['exp1_amplitude'].set(max=2*ampGuess, min=1e-3 * ampGuess)
        else:
            fullPars['exp1_amplitude'].set(max=1e-3 * ampGuess, min=2*ampGuess)
        #
        exp_out = exp_mod.fit(y, fullPars, x=x, method=method)
        intermed_y = y - exp_out.best_fit
        intermed_stats = np.percentile(intermed_y, q=[1, 99])
        intermed_iqr = intermed_stats[1] - intermed_stats[0]
        if verbose:
            print(exp_out.fit_report())
        #
        for pref in ['n1', 'p1', 'p2', 'p3', 'n2', 'p4']:
            pName = '{}_sigma'.format(pref)
            fullPars[pName].set(
                value=np.random.uniform(
                    fullPars[pName].min,
                    fullPars[pName].max))
        '''
        fullPars['n1_sigma'].set(value=np.random.uniform(75e-3, 150e-3))
        fullPars['p1_sigma'].set(value=np.random.uniform(75e-3, 150e-3))
        fullPars['p2_sigma'].set(value=np.random.uniform(75e-3, 150e-3))
        fullPars['p3_sigma'].set(value=np.random.uniform(150e-3, 300e-3))
        '''
        #
        fullPars['n1_offset'].set(value=np.random.uniform(1.1, 1.3))
        # positives
        for pref in ['p1', 'p2', 'p3', 'p4']:
            pName = '{}_amplitude'.format(pref)
            fullPars[pName].set(
                value=1e-3 * intermed_iqr,  # vary=False,
                min=1e-3 * intermed_iqr, max=2 * intermed_iqr)
        # negatives
        for pref in ['n1', 'n2']:
            pName = '{}_amplitude'.format(pref)
            fullPars[pName].set(
                value=-1e-3 * intermed_iqr,  # vary=False,
                max=-1e-3 * intermed_iqr, min=-2 * intermed_iqr)
        # freeze any?
        freezeGaussians = ['p1']
        # freezeGaussians = ['p1', 'n1', 'p2', 'p3']
        if len(freezeGaussians):
            for pref in freezeGaussians:
                pName = '{}_amplitude'.format(pref)
                fullPars[pName].set(value=0, vary=False)
                pName = '{}_sigma'.format(pref)
                fullPars[pName].set(
                    value=(fullPars[pName].max - fullPars[pName].min) / 2,
                    vary=False)
        #
        gauss_out = gauss_mod.fit(intermed_y, fullPars, x=x, method=method)
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
        comps = out.eval_components(x=x)
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
            fig, ax = plotLmFit(x, y, init, out, comps, verbose=verbose)
            ax[1].set_title('R^2 = {}'.format(r2))
        # pdb.set_trace()
        return outSrs, pd.concat([outParams, outStats])
    except Exception:
        traceback.print_exc()
        return dummy, dummyAnns


def plotLmFit(x, y, init, out, comps, verbose=False):
    if verbose:
        print(out.fit_report())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, 'b')
    axes[0].plot(x, init, 'c--', label='initial fit')
    axes[0].plot(x, out.best_fit, 'r-', label='best fit')
    axes[0].legend(loc='best')
    axes[1].plot(x, y, 'b')
    expComp = comps['exp1_'] + comps['exp2_']
    axes[1].plot(x, y - expComp, 'b--', label='Residual after exponent.')
    axes[1].plot(
        x, expComp,  # + comps['const_'],
        'k--', lw=2, label='Offset exponential component')
    axes[1].plot(x, comps['p1_'], 'm--', lw=2, label='P1')
    axes[1].plot(x, comps['n1_'], 'c--', lw=2, label='N1')
    axes[1].plot(x, comps['p2_'], 'y--', lw=2, label='P2')
    axes[1].plot(x, comps['p3_'], 'r--', lw=2, label='P3')
    axes[1].plot(x, comps['n2_'], 'b--', lw=2, label='N2')
    axes[1].plot(x, comps['p4_'], 'g--', lw=2, label='P4')
    axes[1].legend(loc='best')
    return fig, axes


def shapeFit(
        group, dataColNames=None,
        tBounds=None, verbose=False,
        scoreBounds=None, tOffset=0,
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
    # convert to msec
    if scoreBounds is not None:
        scBnds = [1e3 * (sb - tOffset) for sb in scoreBounds]
    groupT = 1e3 * (groupT - tOffset)
    # groupT = groupT - .666
    outList = {'model': [], 'params': []}
    if iterMethod == 'chooseN':
        nChoose = max(groupData.shape[0] - 2, int(groupData.shape[0] / 2), 2)
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.iloc[:, maskX].mean()
            resultsSrs, resultsParams = modelFun(
                groupT, testXBar.to_numpy(),
                scoreBounds=scBnds,
                verbose=verbose, plotting=plotting)
            # TODO: return best model too
            outList['model'].append(resultsSrs)
            outList['params'].append(resultsParams)
    prelimParams = pd.DataFrame(outList['params'])
    prelimDF = pd.DataFrame(outList['model'])
    # pdb.set_trace()
    resultDF = pd.concat([prelimDF, prelimParams], axis='columns').loc[prelimParams['r2'].argmax(), :].to_frame().T
    resultDF.index = [group.index[0]]
    if (not (groupData == 1).all(axis=None)) and plotting:
        plt.show()
    print('shapeFit (proc {}) finished: '.format(os.getpid()))
    for cN in keepIndexCols:
        print('        {} = {}'.format(cN, group.loc[group.index[0], cN]))
        resultDF.loc[group.index[0], cN] = group.loc[group.index[0], cN]
    # print(os.getpid())
    return resultDF

funKWArgs = dict(
    tBounds=[1.2e-3, 99e-3],
    # tBounds=[3e-3, 39.6e-3],
    # tOffset=.7e-3,
    # tOffset=1.2e-3,
    scoreBounds=[1.2e-3, 8e-3],
    modelFun=applyModel,
    iterMethod='chooseN',
    plotting=False, verbose=False,
    # plotting=True, verbose=True,
    maxIter=10
    )

if __name__ == "__main__":
    testVar = None
    conditionNames = [
        'electrode',
        # 'RateInHz',
        arguments['amplitudeFieldName']
        ]
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
    for nM in ['RateInHz', arguments['amplitudeFieldName'], 'stimCat', 'originalIndex', 'segment', 't']:
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
            # featNames.str.contains('rostralY_e12') &
            # elecNames.str.contains('caudalZ_e23') &
            # (featNames.str.contains('rostralY_e15') | featNames.str.contains('caudalY_e11')) &
            elecNames.str.contains('rostralY_e11') &
            (rates < 50) &
            (amps < -1000)
            )
        '''
        dbMask = (rates < 20)
        #
        dataDF = dataDF.loc[dbMask, :]
        #############################
        #############################
        daskClient = Client()
        resDF = ash.splitApplyCombine(
            dataDF,
            fun=shapeFit, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, useDask=True,
            daskPersist=True, daskProgBar=True,
            daskResultMeta=None, daskComputeOpts=daskComputeOpts,
            reindexFromInput=False)
        resDF.set_index(modelColumnNames, inplace=True, append=True)
        resDF.to_hdf(resultPath, 'lmfit_lfp')
        # pdb.set_trace()
        presentNames = [cn for cn in resDF.index.names if cn in dataDF.index.names]
        meansDF = dataDF.groupby(presentNames).mean()
        meansDF.to_hdf(resultPath, 'lfp')
    else:
        resDF = pd.read_hdf(resultPath, 'lmfit_lfp')
        meansDF = pd.read_hdf(resultPath, 'lfp')
    allIdxNames = resDF.index.names
    resDF = resDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    # pdb.set_trace()
    resDF.columns = resDF.columns.astype(np.float) / 1e3 + funKWArgs.pop('tOffset', 0)
    allIdxNames = meansDF.index.names
    meansDF = meansDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    #
    plotDF = pd.concat(
        {'fit': resDF, 'target': meansDF},
        names=['regrID'] + list(resDF.index.names))
    plotDF.dropna(axis='columns', inplace=True)
    plotDF.columns.name = 'bin'
    plotDF = plotDF.stack().to_frame(name='signal').reset_index()
    alignedFeaturesFolder = os.path.join(
        figureFolder, arguments['analysisName'],
        'alignedFeatures')
    if not os.path.exists(alignedFeaturesFolder):
        os.makedirs(alignedFeaturesFolder, exist_ok=True)
    pdfPath = os.path.join(
        alignedFeaturesFolder,
        prefix + '_{}_{}_lmfit.pdf'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    with PdfPages(pdfPath) as pdf:
        for name, group in tqdm(plotDF.groupby('feature')):
            # pdb.set_trace()
            g = sns.relplot(
                data=group.query('bin < 10e-3'),
                x='bin', y='signal',
                hue='regrID', style='regrID',
                row='nominalCurrent', col='electrode',
                facet_kws=dict(sharey=False),
                **relplotKWArgs)
            g.fig.suptitle(name)
            pdf.savefig()
            plt.close()
            # plt.show()

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