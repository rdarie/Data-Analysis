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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
#
sns.set(
    context='poster',
    # context='notebook',
    style='darkgrid',
    palette='pastel', font='sans-serif',
    font_scale=.5,
    color_codes=True)
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
resultFolder = os.path.join(
    calcSubFolder, 'lmfit')
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)
resultPath = os.path.join(
    resultFolder,
    prefix + '_{}_{}_lmfit.h5'.format(
        arguments['inputBlockSuffix'], arguments['window']))

# e.g. resultPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202006171300-Peep/emgLoRes/stim/_emg_XS.nix'
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(-10e3, 100e-3),
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
    timeWindow=(alignedAsigsKWargs['windowSize'][0], -1e-3))
'''
alignedAsigsKWargs['procFun'] = ash.genDetrender(
    timeWindow=(alignedAsigsKWargs['windowSize'][-1] - 10e-3, alignedAsigsKWargs['windowSize'][-1]))
'''

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    prefix, **arguments)

from lmfit.models import ExponentialModel, GaussianModel, ConstantModel
from lmfit import Model, CompositeModel, Parameters
from lmfit.model import save_modelresult, load_modelresult
# np.random.seed(0)

funKWArgs = lmfitFunKWArgs
daskComputeOpts = dict(
    # scheduler='threads'
    scheduler='processes'
    # scheduler='single-threaded'
    )
daskOpts = dict(
    useDask=True,
    daskPersist=True, daskProgBar=True,
    daskResultMeta=None, daskComputeOpts=daskComputeOpts,
    reindexFromInput=False)

useCachedResult = False

DEBUGGING = False
if DEBUGGING:
    daskOpts['daskProgBar'] = False
    daskOpts['daskComputeOpts']['scheduler'] = 'single-threaded'
    funKWArgs.update({
        'maxIter': 1,
        # 'plotting': False, 'verbose': False
        'plotting': True, 'verbose': True
        })


def offsetExponential(x, amplitude=1, decay=1, offset=0):
    res = amplitude * np.exp(-(x - offset) / decay)
    if isinstance(x, np.ndarray):
        res[x < offset] = 0
        return res
    else:
        if x >= offset:
            return res
        else:
            return 0


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x-center)**2 / (2 * sigma ** 2))


# exp1 = Model(offsetExponential, prefix='exp1_')
# exp2 = Model(offsetExponential, prefix='exp2_')
exp1 = ExponentialModel(prefix='exp1_')
exp2 = ExponentialModel(prefix='exp2_')
p1 = Model(gaussian, prefix='p1_')
n1 = Model(gaussian, prefix='n1_')
p2 = Model(gaussian, prefix='p2_')
p3 = Model(gaussian, prefix='p3_')
n2 = Model(gaussian, prefix='n2_')
p4 = Model(gaussian, prefix='p4_')
exp_mod = exp1 + exp2  # + const
gauss_mod = p1 + n1 + p2 + p3 + n2 + p4
full_mod = exp_mod + gauss_mod
#
# pdb.set_trace()
expPars = exp1.make_params()
expPars.update(exp2.make_params())
# expPars.update(const.make_params())
expPars['exp1_amplitude'].set(value=1)
expPars['exp1_decay'].set(value=250, min=5, max=500)
# expPars['exp1_offset'].set(value=0, vary=False)
#
# expPars['exp2_offset'].set(value=0, vary=False)
expPars['exp2_decay'].set(value=.1, min=.01, max=2.)
expPars.add(name='exp2_delay', value=0., min=0., max=.5)
expPars.add(name='exp2_ratio', expr='-1 * exp(-1 * exp2_delay / exp1_decay)')
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
gaussPars.add(
    name='n1_offset',
    min=max(0.5, funKWArgs['tBounds'][0] * 1e3),
    max=1.5)
gaussPars.add(name='p3_offset', value=0, min=0, max=.25)
gaussPars['n1_center'].set(expr='n1_offset + 2 * n1_sigma')
gaussPars['p1_center'].set(expr='n1_offset - 2 * p1_sigma')
gaussPars['p2_center'].set(expr='n1_center + 2 * n1_sigma + 2 * p2_sigma')
gaussPars['p3_center'].set(expr='p2_center + 2 * p2_sigma + p3_offset + 2 * p3_sigma')
gaussPars['n2_center'].set(expr='p3_center + 2 * p3_sigma + 2 * n2_sigma')
gaussPars['p4_center'].set(expr='n2_center + 2 * n2_sigma + 2 * p4_sigma')
# gaussPars.add(name='p3_ratio', value=1e-3, min=0, max=2)
# gaussPars['p3_amplitude'].set(expr='p3_ratio * p2_amplitude')
gaussPars['n1_sigma'].set(value=100e-3, min=50e-3, max=150e-3)
gaussPars['p1_sigma'].set(value=100e-3, min=50e-3, max=150e-3)
gaussPars['p2_sigma'].set(value=200e-3, min=100e-3, max=300e-3)
gaussPars['p3_sigma'].set(value=200e-3, min=100e-3, max=300e-3)
gaussPars['n2_sigma'].set(value=400e-3, min=200e-3, max=600e-3)
gaussPars['p4_sigma'].set(value=400e-3, min=200e-3, max=1200e-3)
absMaxPotential = 1e3 # uV
pars = expPars.copy()
pars.update(gaussPars)

dependentParamNames = [
    'n1_center', 'p1_center',
    'p2_center', 'p3_center',
    'n2_center', 'p4_center'
    'exp2_delay', 'exp2_amplitude']

modelColumnNames = [
    'p4_amplitude', 'p4_center', 'p4_sigma', 'n2_amplitude', 'n2_center',
    'n2_sigma', 'p3_amplitude', 'p3_center', 'p3_sigma', 'p2_amplitude',
    'p2_center', 'p2_sigma', 'n1_amplitude', 'n1_center', 'n1_sigma',
    'p1_amplitude', 'p1_center', 'p1_sigma', 'exp2_amplitude', 'exp2_decay',
    'exp1_amplitude', 'exp1_decay', 'chisqr', 'r2', 'model']


def applyModel(
        x, y,
        method='nelder', scoreBounds=None,
        slowExpTBounds=None,
        verbose=True, plotting=False):
    #
    fullPars = pars.copy()
    dummy = pd.Series(0, index=x)
    dummyAnns = pd.Series({key: 0 for key in modelColumnNames})
    #
    prelim_stats = np.percentile(y, q=[1, 99])
    iqr = prelim_stats[1] - prelim_stats[0]
    if iqr == 0:
        return dummy, dummyAnns, None
    #
    #  Slow exp fitting
    #
    if slowExpTBounds is not None:
        exp_tStart = slowExpTBounds[0] if slowExpTBounds[0] is not None else x[0]
        exp_tStop = slowExpTBounds[1] if slowExpTBounds[1] is not None else x[-1] + 1
        # pdb.set_trace()
        exp_xMask = (x >= (1e3 * exp_tStart)) & (x < (1e3 * exp_tStop))
        exp_x = x[exp_xMask]
        exp_y = y[exp_xMask]
    else:
        exp_x = x
        exp_y = y
    # pdb.set_trace()
    '''
    signalGuess = exp1.eval(fullPars, x=exp_x)
    ampGuess = np.nanmean(exp_y / signalGuess, axis=None)
    print('amp guess = {}'.format(ampGuess))
    '''
    expGuess = exp1.guess(exp_y, x=exp_x)
    signGuess = np.sign(np.nanmean(exp_y, axis=None))
    ampGuess = signGuess * expGuess['exp1_amplitude'].value
    # pdb.set_trace()
    if ampGuess == 0:
        return dummy, dummyAnns, None
    #
    try:
        tempPars = exp1.make_params()
        if ampGuess > 0:
            tempPars['exp1_amplitude'].set(value=ampGuess, max=3 * ampGuess, min=1e-3 * ampGuess)
        else:
            tempPars['exp1_amplitude'].set(value=ampGuess, max=1e-3 * ampGuess, min=3 * ampGuess)
        #
        tempPars['exp1_decay'].set(value=expGuess['exp1_decay'].value)
        ###
        exp_out = exp1.fit(exp_y, tempPars, x=exp_x, method=method)
        #
        fullPars['exp1_amplitude'].set(value=exp_out.best_values['exp1_amplitude'], vary=False)
        fullPars['exp1_decay'].set(value=exp_out.best_values['exp1_decay'], vary=False)
        #
        print('####')
        if verbose:
            print(fullPars['exp1_amplitude'])
            print(fullPars['exp1_decay'])
        print('####')
        #
        intermed_y = exp_y - exp_out.best_fit
        if verbose:
            print(exp_out.fit_report())
        intermed_stats = np.percentile(intermed_y, q=[1, 99])
        intermed_iqr = intermed_stats[1] - intermed_stats[0]
        #
        for pref in ['n1', 'p1', 'p2', 'p3', 'n2', 'p4']:
            pName = '{}_sigma'.format(pref)
            fullPars[pName].set(
                value=np.random.uniform(
                    fullPars[pName].min,
                    fullPars[pName].max))
        #
        fullPars['n1_offset'].set(
            value=np.random.uniform(
                fullPars['n1_offset'].min, fullPars['n1_offset'].max
                ))
        fullPars['exp2_decay'].set(
            value=np.random.uniform(
                fullPars['exp2_decay'].min, fullPars['exp2_decay'].max
                ))
        # positives
        for pref in ['p1', 'p2', 'p3', 'p4']:
            pName = '{}_amplitude'.format(pref)
            maxAmp = min(intermed_iqr, absMaxPotential)
            fullPars[pName].set(
                value=.1 * maxAmp,  # vary=False,
                min=1e-3 * maxAmp, max=maxAmp
                )
        # negatives
        for pref in ['n1', 'n2']:
            pName = '{}_amplitude'.format(pref)
            maxAmp = min(intermed_iqr, absMaxPotential)
            fullPars[pName].set(
                value=-.1 * maxAmp,
                max=-1e-3 * maxAmp,
                min=-maxAmp
                )
        # freeze any?
        # freezeGaussians = ['p1']
        freezeGaussians = []
        # freezeGaussians = ['p1', 'n1', 'p2', 'p3']
        if len(freezeGaussians):
            for pref in freezeGaussians:
                pName = '{}_amplitude'.format(pref)
                fullPars[pName].set(value=0, vary=False)
                pName = '{}_sigma'.format(pref)
                fullPars[pName].set(
                    value=(fullPars[pName].max - fullPars[pName].min) / 2,
                    vary=False)
        init = full_mod.eval(fullPars, x=x)
        out = full_mod.fit(y, fullPars, x=x, method=method)
        outSrs = pd.Series(out.best_fit, index=x)
        outParams = pd.Series(out.best_values)
        if scoreBounds is not None:
            maskX = (x >= scoreBounds[0]) & (x < scoreBounds[1])
        else:
            maskX = np.ones_like(x).astype(np.bool)
        chisqr = ((y[maskX] - out.best_fit[maskX]) ** 2).sum()
        r2 = 1 - (chisqr / (y[maskX] ** 2).sum())
        outStats = pd.Series(
            {
                'chisqr': chisqr,
                'r2': r2})
        if plotting:
            comps = out.eval_components(x=x)
            fig, ax = plotLmFit(x, y, init, out, comps, verbose=verbose)
            ax[1].set_title('R^2 = {}'.format(r2))
        return outSrs, pd.concat([outParams, outStats]), out
    except Exception:
        traceback.print_exc()
        return dummy, dummyAnns, None


def plotLmFit(x, y, init, out, comps, verbose=False):
    if verbose:
        print(out.fit_report())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, c='k', label='data')
    axes[0].plot(x, init, '--', c='dimgray', label='initial fit')
    axes[0].plot(x, out.best_fit, '-', c='silver', label='best fit')
    axes[0].legend(loc='best')
    axes[1].plot(x, y, c='k', label='data')
    expComp = comps['exp1_'] + comps['exp2_']
    axes[1].plot(x, y - expComp, '--', c='dimgray', label='Residual after exponent.')
    axes[1].plot(
        x, expComp,  # + comps['const_'],
        '--', c='silver', lw=2, label='Offset exponential component')
    axes[1].plot(x, comps['p1_'], 'c--', lw=2, label='P1')
    axes[1].plot(x, comps['n1_'], 'm--', lw=2, label='N1')
    axes[1].plot(x, comps['p2_'], 'y--', lw=2, label='P2')
    axes[1].plot(x, comps['p3_'], 'b--', lw=2, label='P3')
    axes[1].plot(x, comps['n2_'], 'r--', lw=2, label='N2')
    axes[1].plot(x, comps['p4_'], 'g--', lw=2, label='P4')
    axes[1].legend(loc='best')
    return fig, axes


def shapeFit(
        group, dataColNames=None,
        tBounds=None, verbose=False, slowExpTBounds=None,
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
        ax.plot(groupData.T, '.-')
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
    outList = {'model': [], 'best_fit': [], 'params': []}
    if iterMethod == 'chooseN':
        # nChoose = max(groupData.shape[0], int(groupData.shape[0] / 2), 2)
        nChoose = 1
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=nChoose)
            testXBar = testX.iloc[:, maskX].mean()
            resultsSrs, resultsParams, mod = modelFun(
                groupT, testXBar.to_numpy(),
                scoreBounds=scBnds, slowExpTBounds=slowExpTBounds,
                verbose=verbose, plotting=plotting)
            # TODO: return best model too
            outList['best_fit'].append(resultsSrs)
            outList['params'].append(resultsParams)
            outList['model'].append(mod)
    elif iterMethod == 'sampleOneManyTimes':
        if maxIter is None:
            maxIter = 5
        else:
            maxIter = int(maxIter)
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=1)
            resultsSrs, resultsParams, mod = modelFun(
                groupT, testX.iloc[:, maskX].to_numpy().flatten(),
                scoreBounds=scBnds, slowExpTBounds=slowExpTBounds,
                verbose=verbose, plotting=plotting)
            # TODO: return best model too
            outList['best_fit'].append(resultsSrs)
            outList['params'].append(resultsParams)
            outList['model'].append(mod)
    prelimParams = pd.DataFrame(outList['params'])
    prelimDF = pd.DataFrame(outList['best_fit'])
    # pdb.set_trace()
    bestIdx = prelimParams['r2'].argmax()
    resultDF = pd.concat([prelimDF, prelimParams], axis='columns').loc[bestIdx, :].to_frame().T
    resultDF.index = [group.index[0]]
    resultDF.loc[group.index[0], 'model'] = outList['model'][bestIdx]
    if (not (groupData == 1).all(axis=None)) and plotting:
        plt.show()
    print('\n\n#######################')
    print('shapeFit (proc {}) finished: '.format(os.getpid()))
    for cN in keepIndexCols:
        print('        {} = {}'.format(cN, group.loc[group.index[0], cN]))
        resultDF.loc[group.index[0], cN] = group.loc[group.index[0], cN]
    # print(os.getpid())
    return resultDF


funKWArgs.update({'modelFun': applyModel})
# pdb.set_trace()

if __name__ == "__main__":
    testVar = None
    conditionNames = [
        'electrode',
        # 'RateInHz',
        arguments['amplitudeFieldName'],
        'originalIndex', 'segment', 't'
        ]
    groupBy = conditionNames + ['feature']
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
        # pdb.set_trace()
        if DEBUGGING:
            dbMask = (
                # featNames.str.contains('rostralY_e12') &
                # elecNames.str.contains('caudalZ_e23') &
                (featNames.str.contains('caudalY_e11') | featNames.str.contains('caudalY_e09')) &
                elecNames.str.contains('caudalY_e11') &
                (rates < funKWArgs['tBounds'][-1] ** (-1)) &
                (amps < -480)
                )
        else:
            dbMask = (rates < funKWArgs['tBounds'][-1] ** (-1))
        #
        # pdb.set_trace()
        dataDF = dataDF.loc[dbMask, :]
        #############################
        #############################
        daskClient = Client()
        resDF = ash.splitApplyCombine(
            dataDF,
            fun=shapeFit, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, **daskOpts)
        # pdb.set_trace()
        if not (useCachedResult and os.path.exists(resultPath)):
            if os.path.exists(resultPath):
                os.remove(resultPath)
        resDF.index = dataDF.index
        modelsSrs = resDF['model'].copy()
        resDF.drop(columns='model', inplace=True)
        modelColumnNames.remove('model')
        modelColumnNames.append('model_index')
        resDF.loc[:, 'model_index'] = range(resDF.shape[0])
        # pdb.set_trace()
        for idx, metaIdx in enumerate(modelsSrs.index):
            modelResult = modelsSrs[metaIdx]
            thisPath = os.path.join(
                resultFolder,
                prefix + '_{}_{}_lmfit_{}.sav'.format(
                    arguments['inputBlockSuffix'], arguments['window'], idx))
            save_modelresult(modelResult, thisPath)
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
    modelIndex = pd.Series(
        resDF.index.get_level_values('model_index'),
        index=resDF.index)
    resDF = resDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    modelIndex = modelIndex.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    resDF.columns = resDF.columns.astype(np.float) / 1e3 + funKWArgs.pop('tOffset', 0)
    allIdxNames = meansDF.index.names
    meansDF = meansDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    # pdb.set_trace()
    compsDict = {}
    for rowIdx, row in resDF.iterrows():
        thisMIdx = modelIndex.loc[rowIdx]
        modelPath = thisPath = os.path.join(
            resultFolder,
            prefix + '_{}_{}_lmfit_{}.sav'.format(
                arguments['inputBlockSuffix'], arguments['window'], thisMIdx))
        model = load_modelresult(modelPath)
        comps = model.eval_components(x=(resDF.columns * 1e3).to_numpy(dtype=np.float))
        for key, value in comps.items():
            if key not in compsDict:
                compsDict[key] = [value]
            else:
                compsDict[key].append(value)
    concatComps = {}
    for key, value in compsDict.items():
        concatComps[key] = pd.DataFrame(
            np.concatenate([v[:, np.newaxis] for v in value], axis=1).T,
            index=resDF.index, columns=resDF.columns)
    concatComps['exp_'] = concatComps['exp1_'] + concatComps['exp2_']
    concatComps.update({'model': resDF, 'target': meansDF.groupby(meansDF.index.names).mean()})
    concatComps['exp_resid'] = concatComps['target'] - concatComps['exp_']
    plotDF = pd.concat(
        concatComps,
        names=['regrID'] + list(resDF.index.names))
    plotDF.dropna(axis='columns', inplace=True)
    plotDF.columns.name = 'bin'
    plotDF = plotDF.stack().to_frame(name='signal').reset_index()
    plotDF.loc[:, 'columnLabel'] = 'NA'
    plotDF.loc[plotDF['regrID'].isin(['model', 'target', 'exp_']), 'columnLabel'] = 'targets'
    plotDF.loc[plotDF['regrID'].isin(['p1_', 'n1_', 'p2_', 'p3_', 'n2_', 'p4_', 'exp_resid']), 'columnLabel'] = 'components'
    #
    plotDF.loc[:, 'rowLabel'] = (
        plotDF['electrode'].astype(np.str) +
        ': ' +
        plotDF[arguments['amplitudeFieldName']].astype(np.str))
    
    alignedFeaturesFolder = os.path.join(
        figureFolder, arguments['analysisName'],
        'alignedFeatures')
    if not os.path.exists(alignedFeaturesFolder):
        os.makedirs(alignedFeaturesFolder, exist_ok=True)
    pdfPath = os.path.join(
        alignedFeaturesFolder,
        prefix + '_{}_{}_lmfit.pdf'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    relplotKWArgs.pop('palette', None)
    # pdb.set_trace()
    plotDF.drop(index=plotDF.index[plotDF['columnLabel'] == 'NA'], inplace=True)
    with PdfPages(pdfPath) as pdf:
        for name, group in tqdm(plotDF.groupby('feature')):
            g = sns.relplot(
                data=group,
                # data=group.query('bin < 10e-3'),
                x='bin', y='signal',
                hue='regrID',
                row='rowLabel', col='columnLabel',
                facet_kws=dict(sharey=False),
                **relplotKWArgs)
            g.fig.suptitle(name)
            pdf.savefig()
            plt.close()
            # plt.show()