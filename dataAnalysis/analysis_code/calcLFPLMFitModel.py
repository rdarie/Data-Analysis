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
    --debugging                                 plot out the correlation matrix? [default: True]
    --smallDataset                              plot out the correlation matrix? [default: True]
    --interactive                               plot out the correlation matrix? [default: True]
    --showFigures                               show the plots? [default: False]
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
##########################################################
##########################################################
useCachedResult = True
##########################################################
##########################################################
import os, sys, re
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if arguments['interactive'] or arguments['debugging']:
    matplotlib.use('QT5Agg')   # generate interactive output
else:
    matplotlib.use('PS')   # generate postscript output
from tqdm import tqdm
import pdb, traceback, shutil
import random
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
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
import dataAnalysis.plotting.spike_sorting_plots as ssplt
from copy import copy

sns.set(
    context='poster',
    # context='notebook',
    style='darkgrid',
    palette='muted', font='sans-serif',
    font_scale=1,
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
    prefix = 'Block'
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
    timeWindow=(alignedAsigsKWargs['windowSize'][0], -2e-3))

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchFolder, prefix, **arguments)

from lmfit.models import ExponentialModel, GaussianModel, ConstantModel
from lmfit import Model, CompositeModel, Parameters
from lmfit.model import save_modelresult, load_modelresult, ModelResult
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


if arguments['debugging']:
    sns.set(font_scale=.5)
    daskOpts['daskProgBar'] = False
    daskOpts['daskComputeOpts']['scheduler'] = 'single-threaded'
    funKWArgs.update({
        'maxIter': 1,
        # 'plotting': False, 'verbose': False
        'plotting': True, 'verbose': True
        })


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x-center)**2 / (2 * sigma ** 2))


expPrefixes = ['exp1_', 'exp2_', 'exp3_']
gaussPrefixes = ['p1_', 'n1_', 'p2_', 'p3_', 'n2_', 'p4_']
modelPrefixes = expPrefixes + gaussPrefixes
#
modelParamNames = [
    'p4_amplitude', 'p4_center', 'p4_sigma', 'n2_amplitude', 'n2_center',
    'n2_sigma', 'p3_amplitude', 'p3_center', 'p3_sigma', 'p2_amplitude',
    'p2_center', 'p2_sigma', 'n1_amplitude', 'n1_center', 'n1_sigma',
    'p1_amplitude', 'p1_center', 'p1_sigma', 'exp3_amplitude', 'exp3_decay',
    'exp2_amplitude', 'exp2_decay', 'exp1_amplitude', 'exp1_decay']
modelStatsNames = ['chisqr', 'r2', 'model']

expMs = {pref: ExponentialModel(prefix=pref) for pref in expPrefixes}
gaussMs = {pref: Model(gaussian, prefix=pref) for pref in gaussPrefixes}
exp_mod = expMs['exp1_'] + expMs['exp2_'] + expMs['exp3_']
gauss_mod = gaussMs['p1_'] + gaussMs['n1_'] + gaussMs['p2_'] + gaussMs['p3_'] + gaussMs['n2_'] + gaussMs['p4_']
full_mod = exp_mod + gauss_mod
#
expPars = Parameters()
for pref, mod in expMs.items():
    expPars.update(mod.make_params())
gaussPars = Parameters()
for pref, mod in gaussMs.items():
    gaussPars.update(mod.make_params())
# Decays in **milliseconds**
expPars['exp1_amplitude'].set(value=0)
expPars['exp1_decay'].set(min=.2, value=1000., max=1000.)
#
expPars['exp2_amplitude'].set(value=0)
expPars['exp2_decay'].set(min=.1, value=20., max=20.)
#
expPars['exp3_amplitude'].set(value=0)
expPars['exp3_decay'].set(min=.05, value=.5, max=2.)
#
gaussPars.add(
    name='n1_offset',
    min=max(0.5, funKWArgs['tBounds'][0] * 1e3),
    max=1.5)
gaussPars.add(name='p3_offset', value=0, min=0, max=1.)
gaussPars['n1_center'].set(expr='n1_offset + n1_sigma')
gaussPars['p1_center'].set(expr='n1_offset - 2 * p1_sigma')
gaussPars['p2_center'].set(expr='n1_center + 2 * n1_sigma + 2 * p2_sigma')
gaussPars['p3_center'].set(expr='p2_center + 2 * p2_sigma + p3_offset + 2 * p3_sigma')
gaussPars['n2_center'].set(expr='p3_center + 2 * p3_sigma + 2 * n2_sigma')
gaussPars['p4_center'].set(expr='n2_center + 2 * n2_sigma + 2 * p4_sigma')
#
gaussPars['n1_sigma'].set(min=75e-3, max=500e-3)
gaussPars['p1_sigma'].set(min=75e-3, max=500e-3)
gaussPars['p2_sigma'].set(min=75e-3, max=500e-3)
gaussPars['p3_sigma'].set(min=150e-3, max=1000e-3)
gaussPars['n2_sigma'].set(min=150e-3, max=1000e-3)
gaussPars['p4_sigma'].set(min=150e-3, max=1000e-3)
#
absMaxPotential = 1e3 # uV
pars = expPars.copy()
pars.update(gaussPars)


def applyModel(
        x, y,
        method='nelder', fit_kws={},
        max_nfev=None, tBounds=None,
        scoreBounds=None, expOpts=None,
        verbose=True, plotting=False):
    #
    if tBounds is not None:
        tStart = tBounds[0] if tBounds[0] is not None else x[0]
        tStop = tBounds[1] if tBounds[1] is not None else x[-1] + 1
        xMask = (x >= (1e3 * tStart)) & (x < (1e3 * tStop))
        fit_x = x[xMask]
        fit_y = y[xMask]
    else:
        fit_x = x
        fit_y = y
    if scoreBounds is not None:
        scoreMaskX = (fit_x >= scoreBounds[0]) & (fit_x < scoreBounds[1])
    else:
        scoreMaskX = np.ones_like(fit_x).astype(np.bool)
    fullPars = pars.copy()
    dummy = pd.Series(0, index=x)
    dummyAnns = pd.Series({key: 0 for key in modelParamNames + modelStatsNames})
    #
    prelim_stats = np.percentile(fit_y, q=[1, 99])
    iqr = prelim_stats[1] - prelim_stats[0]
    if iqr == 0:
        return dummy, dummyAnns, None
    ################################################
    #
    #  Exp fitting
    #
    ################################################
    y_resid = fit_y
    maxDecay = None
    for pref, thisExpMod in expMs.items():
        theseExpOpts = expOpts.copy().pop(pref, dict())
        expTBounds = theseExpOpts.copy().pop('tBounds', None)
        if expTBounds is not None:
            exp_tStart = expTBounds[0] if expTBounds[0] is not None else x[0]
            exp_tStop = expTBounds[1] if expTBounds[1] is not None else x[-1] + 1
            exp_xMask = (fit_x >= (1e3 * exp_tStart)) & (fit_x < (1e3 * exp_tStop))
            exp_x = fit_x[exp_xMask]
            exp_y = y_resid[exp_xMask]
        else:
            exp_x = fit_x
            exp_y = y_resid
        if maxDecay is None:
            maxDecay = fullPars['{}decay'.format(pref)].max
        expGuess = thisExpMod.guess(exp_y, x=exp_x)
        # signGuess = np.sign(np.nanmean(exp_y, axis=None))
        positivePred = thisExpMod.eval(expGuess, x=exp_x)
        positiveResid = np.sum((exp_y - positivePred) ** 2)
        negativeResid = np.sum((exp_y + positivePred) ** 2)
        if positiveResid < negativeResid:
            signGuess = 1.
        else:
            signGuess = -1.
        ampGuess = signGuess * expGuess['{}amplitude'.format(pref)].value
        decayGuess = np.clip(
            expGuess['{}decay'.format(pref)].value,
            fullPars['{}decay'.format(pref)].min,
            maxDecay)
        #
        if ampGuess == 0:
            return dummy, dummyAnns, None
        #
        try:
            tempPars = thisExpMod.make_params()
            if ampGuess > 0:
                ampGuess = max(ampGuess, 1e-6)
                tempPars['{}amplitude'.format(pref)].set(
                    value=ampGuess, max=3 * ampGuess, min=1e-3 * ampGuess)
            else:
                ampGuess = min(ampGuess, -1e-6)
                tempPars['{}amplitude'.format(pref)].set(
                    value=ampGuess, max=1e-3 * ampGuess, min=3 * ampGuess)
            if verbose:
                print('{} ampGuess is {}'.format(pref, ampGuess))
            thisMaxDecay = max(
                min(maxDecay, fullPars['{}decay'.format(pref)].max),
                decayGuess)
            tempPars['{}decay'.format(pref)].set(
                value=decayGuess,
                min=fullPars['{}decay'.format(pref)].min,
                max=thisMaxDecay
                )
            ###
            exp_out = thisExpMod.fit(
                exp_y, tempPars, x=exp_x, method=method,
                max_nfev=max_nfev, fit_kws=fit_kws)
            assessThisModel = theseExpOpts.copy().pop('assessModel', False)
            if assessThisModel:
                modelFitsWellEnough = True
                try:
                    for pName, thisP in exp_out.params.items():
                        assert thisP.stderr is not None
                        assert thisP.value is not None
                        relativeError = 2 * thisP.stderr / thisP.value
                        if np.abs(relativeError) > 1:
                            print('Removing component {} (fit not adequate)'.format(pref))
                            modelFitsWellEnough = False
                except Exception:
                    print('WARNING!')
                    traceback.print_exc()
                    origEnergy = np.sum(exp_y ** 2)
                    residEnergy = np.sum(exp_out.residual ** 2)
                    print('best_values = {}'.format(exp_out.best_values))
                    if origEnergy > residEnergy:
                        print('Keeping component {} (error assesing fit, but fit ok)'.format(pref))
                        modelFitsWellEnough = True
                    else:
                        print('Removing component {} (error assesing fit)'.format(pref))
                        modelFitsWellEnough = False
                    print('origEnergy / residEnergy = {}'.format(origEnergy / residEnergy))
            else:
                if verbose:
                    print('using component {} without assesing'.format(pref))
                modelFitsWellEnough = True
            #
            '''if True:
                plt.show()
                fig, ax = plt.subplots()
                ax.plot(exp_x, exp_y, label='{} original'.format(pref))
                ax.plot(exp_x, exp_out.best_fit, label='{} fit'.format(pref))
                ax.legend()
                plt.show()'''
            if modelFitsWellEnough:
                if verbose:
                    print('using component {}'.format(pref))
                fullPars['{}amplitude'.format(pref)].set(
                    value=exp_out.best_values['{}amplitude'.format(pref)],
                    vary=False)
                foundDecay = exp_out.best_values['{}decay'.format(pref)]
                fullPars['{}decay'.format(pref)].set(
                    value=foundDecay, vary=False)
                maxDecay = min(foundDecay, maxDecay)
            else:
                if verbose:
                    for pName, thisP in exp_out.params.items():
                        print(thisP)
                fullPars['{}amplitude'.format(pref)].set(value=0, vary=False)
                fullPars['{}decay'.format(pref)].set(
                    value=fullPars['{}decay'.format(pref)].max, vary=False)
            if verbose:
                print('####')
                print(fullPars['{}amplitude'.format(pref)])
                print(fullPars['{}decay'.format(pref)])
                print('####')
            y_resid = y_resid - exp_out.eval(fullPars, x=fit_x)
            '''if verbose:
                print(exp_out.fit_report())'''
            intermed_stats = np.percentile(y_resid, q=[1, 99])
            intermed_iqr = intermed_stats[1] - intermed_stats[0]
        except Exception:
            traceback.print_exc()
            return dummy, dummyAnns, None
    ################################################
    #
    # gaussian fitting
    #
    ################################################
    try:
        # Randomize std dev
        for pref in ['n1_', 'p1_', 'p2_', 'p3_', 'n2_', 'p4_']:
            pName = '{}sigma'.format(pref)
            fullPars[pName].set(
                value=np.random.uniform(
                    fullPars[pName].min,
                    2 * fullPars[pName].min))
        # Randomize n1_offset
        fullPars['n1_offset'].set(
            value=np.random.uniform(
                fullPars['n1_offset'].min, fullPars['n1_offset'].max
                ))
        # positives
        for pref in ['p1_', 'p2_', 'p3_', 'p4_']:
            pName = '{}amplitude'.format(pref)
            maxAmp = min(intermed_iqr, absMaxPotential)
            fullPars[pName].set(
                value=0, min=1e-6 * maxAmp, max=maxAmp)
        # negatives
        for pref in ['n1_', 'n2_']:
            pName = '{}amplitude'.format(pref)
            maxAmp = min(intermed_iqr, absMaxPotential)
            fullPars[pName].set(
                value=0, max=-1e-6 * maxAmp, min=-maxAmp)
        # freeze any?
        freezeGaussians = []
        if len(freezeGaussians):
            for pref in freezeGaussians:
                pName = '{}amplitude'.format(pref)
                fullPars[pName].set(value=0, vary=False)
                pName = '{}sigma'.format(pref)
                fullPars[pName].set(
                    value=fullPars[pName].min, vary=False)
        out = full_mod.fit(
            fit_y, fullPars, x=fit_x, method=method,
            max_nfev=max_nfev, fit_kws=fit_kws)
        outSrs = pd.Series(full_mod.eval(out.params, x=x), index=x)
        outParams = pd.Series(out.best_values)
        #
        scoreComps = out.eval_components(x=fit_x[scoreMaskX])
        bestFit = y_resid[scoreMaskX] ** 0 - 1
        for cName in gaussPrefixes:
            bestFit += scoreComps[cName]
        residuals = (y_resid[scoreMaskX] - bestFit)
        chisqr = (residuals ** 2).sum()
        r2 = 1 - (chisqr / (y_resid[scoreMaskX] ** 2).sum())
        outStats = pd.Series(
            {
                'chisqr': chisqr,
                'r2': r2})
        if plotting:
            comps = out.eval_components(x=fit_x)
            init = full_mod.eval(fullPars, x=fit_x)
            fig, ax = plotLmFit(
                fit_x, fit_y, init, out, comps,
                verbose=False)
            ax[0].set_title('t_0 = {:.3f} msec'.format(1e3 * tBounds[0]))
            ax[1].set_title('R^2 = {:.3f}'.format(r2))
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
    expComp = comps['exp1_'] + comps['exp2_'] + comps['exp3_']
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
        group, dataColNames=None, fit_kws={},
        method='nelder', max_nfev=None,
        tBounds=None, verbose=False, expOpts=None,
        scoreBounds=None, tOffset=0, startJitter=.3e-3,
        plotting=False, iterMethod='loo',
        modelFun=None, corrMethod='pearson',
        maxIter=10):
    # print('os.getpid() = {}'.format(os.getpid()))
    # print('Group shape is {}'.format(group.shape))
    dataColMask = group.columns.isin(dataColNames)
    groupData = group.loc[:, dataColMask]
    indexColMask = ~group.columns.isin(dataColNames)
    indexCols = group.columns[indexColMask]
    keepIndexCols = indexCols[~indexCols.isin(['segment', 'originalIndex', 't'])]
    groupT = groupData.columns.to_numpy(dtype=float)
    if tBounds is None:
        tBounds = [groupT[0], groupT[-1] + 1]
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
    # convert to msec
    if scoreBounds is not None:
        scBnds = [1e3 * (sb - tOffset) for sb in scoreBounds]
    groupT = 1e3 * (groupT - tOffset)
    outList = {'model': [], 'best_fit': [], 'params': []}
    if iterMethod == 'sampleOneManyTimes':
        if maxIter is None:
            maxIter = 5
        else:
            maxIter = int(maxIter)
        for idx in range(maxIter):
            testX = shuffle(groupData, n_samples=1)
            theseTBounds = copy(tBounds)
            theseTBounds[0] += np.random.uniform(0, startJitter)
            resultsSrs, resultsParams, mod = modelFun(
                groupT, testX.to_numpy().flatten(),
                tBounds=theseTBounds,
                scoreBounds=scBnds, fit_kws=fit_kws,
                expOpts=expOpts, method=method,
                max_nfev=max_nfev,
                verbose=verbose, plotting=plotting)
            #
            outList['best_fit'].append(resultsSrs)
            outList['params'].append(resultsParams)
            outList['model'].append(mod)
    prelimParams = pd.DataFrame(outList['params'])
    prelimDF = pd.DataFrame(outList['best_fit'])
    prelimDF.columns = groupData.columns
    bestIdx = prelimParams['r2'].argmax()
    resultDF = pd.concat([prelimDF, prelimParams], axis='columns').loc[bestIdx, :].to_frame().T
    resultDF.index = [group.index[0]]
    bestModel = outList['model'][bestIdx]
    if isinstance(bestModel, ModelResult):
        resultDF.loc[group.index[0], 'model'] = bestModel
    else:
        resultDF.loc[group.index[0], 'model'] = np.nan
    if (not (groupData == 1).all(axis=None)) and plotting:
        plt.show()
    if verbose:
        print('\n\n#######################')
        print('shapeFit (proc {}) finished: '.format(os.getpid()))
    for cN in indexCols:
        if verbose:
            print('        {} = {}'.format(cN, group.loc[group.index[0], cN]))
        resultDF.loc[group.index[0], cN] = group.loc[group.index[0], cN]
    # print(os.getpid())
    return resultDF


funKWArgs.update({'modelFun': applyModel})

if __name__ == "__main__":
    daskClient = Client()
    testVar = None
    conditionNames = [
        'electrode',
        'RateInHz',
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
        dataDF.columns = dataDF.columns.astype(float)
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
        featNames = dataDF.index.get_level_values('feature')
        elecNames = dataDF.index.get_level_values('electrode')
        rates = dataDF.index.get_level_values('RateInHz')
        print('Available rates are {}'.format(np.unique(rates)))
        amps = dataDF.index.get_level_values(arguments['amplitudeFieldName'])
        print('Available amps are {}'.format(np.unique(amps)))
        if arguments['smallDataset']:
            dbIndexMask = (
                # elecNames.str.contains('caudalZ_e23') &
                # (featNames.str.contains('caudalY_e11') | featNames.str.contains('rostralY_e11')) &
                featNames.str.contains('rostralY') &
                elecNames.str.contains('caudalY') &
                (rates < funKWArgs['tBounds'][-1] ** (-1))
                # (amps == -900)
                )
        else:
            dbIndexMask = (
                    (featNames.str.contains('rostral')) &
                    (rates < funKWArgs['tBounds'][-1] ** (-1))
                    # (elecNames.str.contains('caudal'))
                )
        dbColMask = (
            (dataDF.columns.astype(float) >= funKWArgs['tBounds'][0]) &
            (dataDF.columns.astype(float) < funKWArgs['tBounds'][-1]))
        dataDF = dataDF.loc[dbIndexMask, dbColMask]
        resDF = ash.splitApplyCombine(
            dataDF,
            fun=shapeFit, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, **daskOpts)
        if not (useCachedResult and os.path.exists(resultPath)):
            if os.path.exists(resultPath):
                shutil.rmtree(resultFolder)
                os.makedirs(resultFolder)
        modelsSrs = pd.Series(resDF.index.get_level_values('model').copy())
        resDF.index = resDF.index.droplevel('model')
        # modelStatsNames.remove('model')
        # modelStatsNames.append('model_index')
        resDF.loc[:, 'model_index'] = range(resDF.shape[0])
        resDF.set_index('model_index', append=True, inplace=True)
        resDF.to_hdf(resultPath, 'lmfit_lfp')
        presentNames = [cn for cn in resDF.index.names if cn in dataDF.index.names]
        meansDF = dataDF.groupby(presentNames).mean()
        meansDF.to_hdf(resultPath, 'lfp')
        for idx, metaIdx in enumerate(modelsSrs.index):
            modRes = modelsSrs[metaIdx]
            if isinstance(modRes, ModelResult):
                thisPath = os.path.join(
                    resultFolder,
                    prefix + '_{}_{}_lmfit_{}.sav'.format(
                        arguments['inputBlockSuffix'], arguments['window'], idx))
                try:
                    save_modelresult(modRes, thisPath)
                except Exception:
                    traceback.print_exc()
                    pdb.set_trace()
    else:
        modelStatsNames.remove('model')
        modelStatsNames.append('model_index')
        resDF = pd.read_hdf(resultPath, 'lmfit_lfp')
        meansDF = pd.read_hdf(resultPath, 'lfp')
    #########################################################################
    modelParams = resDF.index.to_frame()
    allIdxNames = resDF.index.names
    modelIndex = pd.Series(
        resDF.index.get_level_values('model_index'),
        index=resDF.index)
    resDF.index = resDF.index.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    modelParams = modelParams.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    modelIndex = modelIndex.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    allIdxNames = meansDF.index.names
    meansDF = meansDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    # resDF.columns = resDF.columns.astype(np.float) / 1e3 + funKWArgs.pop('tOffset', 0)
    compsDict = {}
    paramsCI = pd.DataFrame(np.nan, index=modelParams.index, columns=modelParamNames)
    for rowIdx, row in resDF.iterrows():
        prf.print_memory_usage(prefix='Loading result {}'.format(rowIdx))
        thisMIdx = modelIndex.loc[rowIdx]
        modelPath = thisPath = os.path.join(
            resultFolder,
            prefix + '_{}_{}_lmfit_{}.sav'.format(
                arguments['inputBlockSuffix'], arguments['window'], thisMIdx))
        try:
            model = load_modelresult(modelPath)
            for pName in modelParamNames:
                paramsCI.loc[rowIdx, pName] = model.params[pName].stderr
            comps = model.eval_components(x=(resDF.columns * 1e3).to_numpy(dtype=np.float))
            for key, value in comps.items():
                if key not in compsDict:
                    compsDict[key] = [value]
                else:
                    compsDict[key].append(value)
        except Exception:
            traceback.print_exc()
            filler = resDF.columns.to_numpy() * 0
            for key in modelPrefixes:
                if key not in compsDict:
                    compsDict[key] = [filler]
                else:
                    compsDict[key].append(filler)
    relativeCI = (4 * paramsCI / modelParams).loc[:, modelParamNames]
    modelParams = modelParams.loc[:, modelParamNames]
    meanParams = modelParams.groupby(['electrode', arguments['amplitudeFieldName'], 'feature']).mean() * np.nan
    for name, group in modelParams.groupby(['electrode', arguments['amplitudeFieldName'], 'feature']):
        for pref in ['n1_', 'n2_', 'p1_', 'p2_', 'p3_', 'p4_']:
            prefMask = group.columns.str.startswith(pref)
            subGroup = group.loc[:, prefMask].reset_index(drop=True)
            subGroupCI = relativeCI.loc[group.index, prefMask].reset_index(drop=True)
            goodFitMask = (subGroupCI[pref + 'amplitude'].abs() < 1).to_numpy()
            subGroupMean = subGroup.loc[goodFitMask, :].median()
            if np.isnan(subGroupMean[pref + 'amplitude']):
                subGroupMean.loc[pref + 'amplitude'] = 0
            meanParams.loc[name, subGroupMean.index] = subGroupMean
    meanParams.loc[:, 'N1P2'] = meanParams['p2_amplitude'] - meanParams['n1_amplitude']
    for pref in ['n1_', 'p2_', 'n2_']:
        meanParams.loc[:, '{}latency'.format(pref)] = (
            meanParams['{}center'.format(pref)] +
            3 * meanParams['{}sigma'.format(pref)])
    meanParams.loc[meanParams['N1P2'] == 0, 'N1P2'] = np.nan
    meanParams = meanParams.reset_index()
    # resDF = resDF.reset_index(drop=True)
    #
    concatComps = {}
    for key, value in compsDict.items():
        concatComps[key] = pd.DataFrame(
            np.concatenate([v[:, np.newaxis] for v in value], axis=1).T,
            index=resDF.index, columns=resDF.columns)
    concatComps['exp_'] = concatComps['exp1_'] + concatComps['exp2_'] + concatComps['exp3_']
    concatComps.update({'model': resDF, 'target': meansDF.groupby(meansDF.index.names).mean()})
    concatComps['exp_resid'] = concatComps['target'] - concatComps['exp_']
    compsAndTargetDF = pd.concat(
        concatComps,
        names=['regrID'] + list(resDF.index.names))
    compsAndTargetDF.dropna(axis='columns', inplace=True)
    compsAndTargetDF.columns = compsAndTargetDF.columns.astype(float)
    compsAndTargetDF.columns.name = 'bin'
    # plotDF = compsAndTargetDF.stack().to_frame(name='signal').reset_index()
    plotDF = compsAndTargetDF.reset_index()
    # plotDF = compsAndTargetDF.copy()
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
    relplotKWArgs.pop('palette', None)
    plotDF.drop(index=plotDF.index[plotDF['columnLabel'] == 'NA'], inplace=True)
    #
    stimConfigLookup, elecChanNames = prb_meta.parseElecComboName(meanParams['electrode'].unique())
    rippleMapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
    rippleMapDF.loc[
        rippleMapDF['label'].str.contains('caudal'),
        'ycoords'] += 500
    #
    meanParams.loc[:, 'xCoords'] = meanParams['feature'].str.replace('#0', '').map(pd.Series(rippleMapDF['xcoords'].to_numpy(), index=rippleMapDF['label']))
    meanParams.loc[:, 'yCoords'] = meanParams['feature'].str.replace('#0', '').map(pd.Series(rippleMapDF['ycoords'].to_numpy(), index=rippleMapDF['label']))
    #
    coordLookup = pd.DataFrame(rippleMapDF.loc[:, ['xcoords', 'ycoords', 'label']])
    coordLookup.loc[:, 'label'] = coordLookup.loc[:, 'label'].str.replace('_a', '').str.replace('_b', '')
    coordLookup.drop_duplicates(inplace=True)
    coordLookup.set_index('label', inplace=True, drop=True)
    xIdx, yIdx = ssplt.coordsToIndices(
        rippleMapDF['xcoords'], rippleMapDF['ycoords'])
    #
    paramMetaData = {
        'N1P2': {
            'units': 'uV',
            'label': 'N1-P2 Amplitude'
        },
        'n1_center': {
            'units': 'msec',
            'label': 'N1 Peak'
        },
        'p2_center': {
            'units': 'msec',
            'label': 'P2 Peak'
        },
        'n2_center': {
            'units': 'msec',
            'label': 'N2 Peak'
        }}
    whichParamsToPlot = list(paramMetaData.keys())
    for pName in whichParamsToPlot:
        vLims = 2 * meanParams.loc[:, pName].quantile([.25, .75]) - meanParams.loc[:, pName].median()
        paramMetaData[pName].update({
            'vmin': vLims.iloc[0], 'vmax': vLims.iloc[-1]})
    pdfPath = os.path.join(
        alignedFeaturesFolder,
        prefix + '_{}_{}_lmfit_heatmaps.pdf'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    xSize = meanParams['xCoords'].unique().size
    minX = meanParams['xCoords'].min()
    maxX = meanParams['xCoords'].max()
    ySize = meanParams['yCoords'].unique().size
    minY = meanParams['yCoords'].min()
    maxY = meanParams['yCoords'].max()
    def cTransform(absV, minV, maxV, vSize):
        return (absV - minV) / (maxV - minV) * vSize
    # plot heatmaps

    # # export model params and confidence intervals
    outputParams = modelParams.reset_index()
    outputParams.columns = outputParams.columns.astype(str)
    resultPath = os.path.join(
        resultFolder,
        prefix + '_{}_{}_lmfit_model_parameters.parquet'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    outputParams.to_parquet(resultPath, engine="fastparquet")
    #
    outputComps = compsAndTargetDF.reset_index()
    outputComps.columns = outputComps.columns.astype(str)
    resultPath = os.path.join(
        resultFolder,
        prefix + '_{}_{}_lmfit_signals.parquet'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    outputComps.to_parquet(resultPath, engine="fastparquet")
    #
    outputCI = paramsCI.reset_index()
    outputCI.columns = outputCI.columns.astype(str)
    resultPath = os.path.join(
        resultFolder,
        prefix + '_{}_{}_lmfit_std_errs.parquet'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    outputCI.to_parquet(resultPath, engine="fastparquet")
    #
    with PdfPages(pdfPath) as pdf:
        for name, group in tqdm(meanParams.groupby(['electrode', arguments['amplitudeFieldName']])):
            for pName, plotMeta in paramMetaData.items():
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 24)
                data = group.loc[:, ['xCoords', 'yCoords', pName]]
                # data.loc[group['r2'] < 0.9, pName] = np.nan
                dataSquare = data.pivot('yCoords', 'xCoords', pName)
                anns = group.loc[:, ['xCoords', 'yCoords', 'feature']]
                anns.loc[:, 'feature'] = anns['feature'].str.replace('a#0', '')
                anns.loc[:, 'feature'] = anns['feature'].str.replace('b#0', '')
                anns.loc[:, 'feature'] = anns['feature'].str.replace('_', '\n')
                annsSquare = anns.pivot('yCoords', 'xCoords', 'feature')
                ax = sns.heatmap(
                    dataSquare, annot=annsSquare, fmt='s',
                    annot_kws={'fontsize': 'xx-small'}, cmap='Blues_r',
                    vmin=plotMeta['vmin'], vmax=plotMeta['vmax'],
                    ax=ax)
                # overAx = ax.twinx()
                elecConfig = stimConfigLookup[name[0]]
                for eName in elecConfig['cathodes']:
                    x = coordLookup.loc[eName, 'xcoords']
                    tX = cTransform(x, minX, maxX, xSize)
                    y = coordLookup.loc[eName, 'ycoords']
                    tY = cTransform(y, minY, maxY, ySize)
                    ax.plot(tX, tY, marker='*', c='b', ms=10, ls=None)
                for eName in elecConfig['anodes']:
                    x = coordLookup.loc[eName, 'xcoords']
                    tX = cTransform(x, minX, maxX, xSize)
                    y = coordLookup.loc[eName, 'ycoords']
                    tY = cTransform(y, minY, maxY, ySize)
                    ax.plot(tX, tY, marker='*', c='r', ms=10, ls=None)
                titleText = '{} ({})'.format(plotMeta['label'], plotMeta['units'])
                titleText += '\n stim on {} ({} uA)'.format(name[0], name[1])
                fig.suptitle(titleText)
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
    ###########################
    timeScales = ['3', '8', '40']
    # timeScales = ['8']
    plotDF.set_index(compsAndTargetDF.index.names + ['columnLabel', 'rowLabel'], inplace=True)
    for timeScale in timeScales:
        pdfPath = os.path.join(
            alignedFeaturesFolder,
            prefix + '_{}_{}_lmfit_{}_msec.pdf'.format(
                arguments['inputBlockSuffix'], arguments['window'], timeScale))
        with PdfPages(pdfPath) as pdf:
            for name, group in tqdm(plotDF.groupby('feature')):
                plotGroup = group.stack().to_frame(name='signal').reset_index()
                g = sns.relplot(
                    # data=plotGroup,
                    data=plotGroup.query('(bin < {}e-3) & (bin >= 0.9e-3)'.format(timeScale)),
                    x='bin', y='signal',
                    hue='regrID',
                    row='rowLabel', col='columnLabel',
                    facet_kws=dict(sharey=False),
                    **relplotKWArgs)
                g.fig.suptitle(name)
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()