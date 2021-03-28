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
##########################################################
##########################################################
useCachedResult = False
DEBUGGING = False
SMALLDATASET = True
##########################################################
##########################################################
import os, sys, re
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
from tqdm import tqdm
import pdb, traceback, shutil
import random
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
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
#
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


if DEBUGGING:
    daskOpts['daskProgBar'] = False
    daskOpts['daskComputeOpts']['scheduler'] = 'single-threaded'
    funKWArgs.update({
        'maxIter': 1,
        # 'plotting': False, 'verbose': False
        'plotting': True, 'verbose': True
        })


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x-center)**2 / (2 * sigma ** 2))


# exp1 = Model(offsetExponential, prefix='exp1_')
# exp2 = Model(offsetExponential, prefix='exp2_')
exp1 = ExponentialModel(prefix='exp1_')
exp2 = ExponentialModel(prefix='exp2_')
exp3 = ExponentialModel(prefix='exp3_')
#
p1 = Model(gaussian, prefix='p1_')
n1 = Model(gaussian, prefix='n1_')
p2 = Model(gaussian, prefix='p2_')
p3 = Model(gaussian, prefix='p3_')
n2 = Model(gaussian, prefix='n2_')
p4 = Model(gaussian, prefix='p4_')
exp_mod = exp1 + exp2 + exp3
gauss_mod = p1 + n1 + p2 + p3 + n2 + p4
full_mod = exp_mod + gauss_mod
#
expPars = exp1.make_params()
expPars.update(exp2.make_params())
expPars.update(exp3.make_params())
#
expPars['exp1_amplitude'].set(value=1)
expPars['exp1_decay'].set(min=30., value=250., max=500.)
#
expPars['exp2_amplitude'].set(value=1)
expPars['exp2_decay'].set(min=2., value=10., max=20.)
#
expPars['exp3_amplitude'].set(value=1)
expPars['exp3_decay'].set(min=.05, value=.1, max=.5)
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
gaussPars['n1_center'].set(expr='n1_offset + n1_sigma')
gaussPars['p1_center'].set(expr='n1_offset - 2 * p1_sigma')
gaussPars['p2_center'].set(expr='n1_center + 2 * n1_sigma + 2 * p2_sigma')
gaussPars['p3_center'].set(expr='p2_center + 2 * p2_sigma + p3_offset + 2 * p3_sigma')
gaussPars['n2_center'].set(expr='p3_center + 2 * p3_sigma + 2 * n2_sigma')
gaussPars['p4_center'].set(expr='n2_center + 2 * n2_sigma + 2 * p4_sigma')
#
gaussPars['n1_sigma'].set(value=100e-3, min=50e-3, max=150e-3)
gaussPars['p1_sigma'].set(value=100e-3, min=50e-3, max=150e-3)
gaussPars['p2_sigma'].set(value=200e-3, min=100e-3, max=300e-3)
gaussPars['p3_sigma'].set(value=200e-3, min=100e-3, max=300e-3)
gaussPars['n2_sigma'].set(value=400e-3, min=200e-3, max=600e-3)
gaussPars['p4_sigma'].set(value=400e-3, min=200e-3, max=1200e-3)
#
absMaxPotential = 1e3 # uV
pars = expPars.copy()
pars.update(gaussPars)

dependentParamNames = [
    'n1_center', 'p1_center',
    'p2_center', 'p3_center',
    'n2_center', 'p4_center'
    ]

modelPrefixes = ['exp1_', 'exp2_', 'exp3_', 'p1_', 'n1_', 'p2_', 'p3_', 'n2_', 'p4_']
modelParamNames = [
    'p4_amplitude', 'p4_center', 'p4_sigma', 'n2_amplitude', 'n2_center',
    'n2_sigma', 'p3_amplitude', 'p3_center', 'p3_sigma', 'p2_amplitude',
    'p2_center', 'p2_sigma', 'n1_amplitude', 'n1_center', 'n1_sigma',
    'p1_amplitude', 'p1_center', 'p1_sigma', 'exp3_amplitude', 'exp3_decay',
    'exp2_amplitude', 'exp2_decay', 'exp1_amplitude', 'exp1_decay']
modelStatsNames = ['chisqr', 'r2', 'model']


def applyModel(
        x, y,
        method='least_squares', scoreBounds=None,
        slowExpTBounds=None, medExpTBounds=None, fastExpTBounds=None,
        assessSlow=False, assessMed=False, assessFast=False,
        verbose=True, plotting=False):
    #
    fullPars = pars.copy()
    dummy = pd.Series(0, index=x)
    dummyAnns = pd.Series({key: 0 for key in modelParamNames + modelStatsNames})
    #
    prelim_stats = np.percentile(y, q=[1, 99])
    iqr = prelim_stats[1] - prelim_stats[0]
    if iqr == 0:
        return dummy, dummyAnns, None
    ################################################
    #
    #  Slow exp fitting
    #
    ################################################
    if slowExpTBounds is not None:
        exp_tStart = slowExpTBounds[0] if slowExpTBounds[0] is not None else x[0]
        exp_tStop = slowExpTBounds[1] if slowExpTBounds[1] is not None else x[-1] + 1
        exp_xMask = (x >= (1e3 * exp_tStart)) & (x < (1e3 * exp_tStop))
        exp_x = x[exp_xMask]
        exp_y = y[exp_xMask]
    else:
        exp_x = x
        exp_y = y
    expGuess = exp1.guess(exp_y, x=exp_x)
    signGuess = np.sign(np.nanmean(exp_y, axis=None))
    ampGuess = signGuess * expGuess['exp1_amplitude'].value
    decayGuess = np.clip(expGuess['exp1_decay'].value, fullPars['exp1_decay'].min, fullPars['exp1_decay'].max)
    #
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
        tempPars['exp1_decay'].set(
            value=decayGuess,
            min=fullPars['exp1_decay'].min,
            max=fullPars['exp1_decay'].max
            )
        ###
        exp_out = exp1.fit(exp_y, tempPars, x=exp_x, method=method)
        if assessSlow:
            try:
                exp_out.conf_interval()
                ciDF = pd.DataFrame(exp_out.ci_out)
                paramsCI = {}
                modelFitsWellEnough = True
                for pName in ciDF.columns:
                    paramsCI[pName] = (ciDF.loc[5, pName][1] - ciDF.loc[1, pName][1]) / exp_out.best_values[pName]
                    if paramsCI[pName] > 1:
                        # print('exponential param {} has relative CI {}! discarding...'.format(pName, paramsCI[pName]))
                        modelFitsWellEnough = False
            except Exception:
                traceback.print_exc()
                modelFitsWellEnough = False
        else:
            modelFitsWellEnough = True
        #
        if modelFitsWellEnough:
            fullPars['exp1_amplitude'].set(value=exp_out.best_values['exp1_amplitude'], vary=False)
            fullPars['exp1_decay'].set(value=exp_out.best_values['exp1_decay'], vary=False)
        else:
            print(exp_out.ci_report())
            fullPars['exp1_amplitude'].set(value=0, vary=False)
            fullPars['exp1_decay'].set(value=fullPars['exp1_decay'].max, vary=False)
        #
        print('####')
        if verbose:
            print(fullPars['exp1_amplitude'])
            print(fullPars['exp1_decay'])
        print('####')
        #
        intermed_y = y - exp_out.eval(fullPars, x=x)
        if verbose:
            print(exp_out.fit_report())
        intermed_stats = np.percentile(intermed_y, q=[1, 99])
        intermed_iqr = intermed_stats[1] - intermed_stats[0]
    except Exception:
        traceback.print_exc()
        return dummy, dummyAnns, None
    ################################################
    #
    # medium scale exp fitting
    #
    ################################################
    if medExpTBounds is not None:
        exp2_tStart = medExpTBounds[0] if medExpTBounds[0] is not None else x[0]
        exp2_tStop = medExpTBounds[1] if medExpTBounds[1] is not None else x[-1] + 1
        exp2_xMask = (x >= (1e3 * exp2_tStart)) & (x < (1e3 * exp2_tStop))
        #
        exp2_x = x[exp2_xMask]
        exp2_y = intermed_y[exp2_xMask]
    else:
        exp2_x = x
        exp2_y = intermed_y
    expGuess = exp2.guess(exp2_y, x=exp2_x)
    signGuess = np.sign(np.nanmean(exp2_y, axis=None))
    ampGuess = signGuess * expGuess['exp2_amplitude'].value
    decayGuess = np.clip(expGuess['exp2_decay'].value, fullPars['exp2_decay'].min, fullPars['exp2_decay'].max)
    try:
        tempPars = exp2.make_params()
        if ampGuess > 0:
            tempPars['exp2_amplitude'].set(value=ampGuess, max=3 * ampGuess, min=1e-3 * ampGuess)
        else:
            tempPars['exp2_amplitude'].set(value=ampGuess, max=1e-3 * ampGuess, min=3 * ampGuess)
        #
        tempPars['exp2_decay'].set(
            value=decayGuess,
            min=fullPars['exp2_decay'].min,
            max=fullPars['exp2_decay'].max)
        ###
        exp_out = exp2.fit(exp2_y, tempPars, x=exp2_x, method=method)
        if assessMed:
            try:
                exp_out.conf_interval()
                ciDF = pd.DataFrame(exp_out.ci_out)
                paramsCI = {}
                modelFitsWellEnough = True
                for pName in ciDF.columns:
                    paramsCI[pName] = (ciDF.loc[5, pName][1] - ciDF.loc[1, pName][1]) / exp_out.best_values[pName]
                    if paramsCI[pName] > 1:
                        # print('exponential param {} has relative CI {}! discarding...'.format(pName, paramsCI[pName]))
                        modelFitsWellEnough = False
            except Exception:
                traceback.print_exc()
                modelFitsWellEnough = False
        else:
            modelFitsWellEnough = True
        #
        if modelFitsWellEnough:
            fullPars['exp2_amplitude'].set(value=exp_out.best_values['exp2_amplitude'], vary=False)
            fullPars['exp2_decay'].set(value=exp_out.best_values['exp2_decay'], vary=False)
        else:
            print(exp_out.ci_report())
            fullPars['exp2_amplitude'].set(value=0, vary=False)
            fullPars['exp2_decay'].set(value=fullPars['exp2_decay'].max, vary=False)
        #
        print('####')
        if verbose:
            print(fullPars['exp2_amplitude'])
            print(fullPars['exp2_decay'])
        print('####')
        #
        intermed_y = intermed_y - exp_out.eval(fullPars, x=x)
        if verbose:
            print(exp_out.fit_report())
        intermed_stats = np.percentile(intermed_y, q=[1, 99])
        intermed_iqr = intermed_stats[1] - intermed_stats[0]
    except Exception:
        traceback.print_exc()
        return dummy, dummyAnns, None
    ################################################
    #
    # super fast exp fitting
    #
    ################################################
    if fastExpTBounds is not None:
        exp3_tStart = fastExpTBounds[0] if fastExpTBounds[0] is not None else x[0]
        exp3_tStop = fastExpTBounds[1] if fastExpTBounds[1] is not None else x[-1] + 1
        exp3_xMask = (x >= (1e3 * exp3_tStart)) & (x < (1e3 * exp3_tStop))
        #
        exp3_x = x[exp3_xMask]
        exp3_y = intermed_y[exp3_xMask]
    else:
        exp3_x = x
        exp3_y = intermed_y
    expGuess = exp3.guess(exp3_y, x=exp3_x)
    signGuess = np.sign(np.nanmean(exp3_y, axis=None))
    ampGuess = signGuess * expGuess['exp3_amplitude'].value
    decayGuess = np.clip(expGuess['exp3_decay'].value, fullPars['exp3_decay'].min, fullPars['exp3_decay'].max)
    try:
        tempPars = exp3.make_params()
        if ampGuess > 0:
            tempPars['exp3_amplitude'].set(value=ampGuess, max=3 * ampGuess, min=1e-3 * ampGuess)
        else:
            tempPars['exp3_amplitude'].set(value=ampGuess, max=1e-3 * ampGuess, min=3 * ampGuess)
        #
        tempPars['exp3_decay'].set(
            value=decayGuess,
            min=fullPars['exp3_decay'].min,
            max=fullPars['exp3_decay'].max)
        ###
        exp_out = exp3.fit(exp3_y, tempPars, x=exp3_x, method=method)
        if assessFast:
            try:
                exp_out.conf_interval()
                ciDF = pd.DataFrame(exp_out.ci_out)
                paramsCI = {}
                modelFitsWellEnough = True
                for pName in ciDF.columns:
                    paramsCI[pName] = (ciDF.loc[5, pName][1] - ciDF.loc[1, pName][1]) / exp_out.best_values[pName]
                    if paramsCI[pName] > 1:
                        # print('exponential param {} has relative CI {}! discarding...'.format(pName, paramsCI[pName]))
                        modelFitsWellEnough = False
            except Exception:
                traceback.print_exc()
                modelFitsWellEnough = False
        else:
            modelFitsWellEnough = True
        #
        if modelFitsWellEnough:
            fullPars['exp3_amplitude'].set(value=exp_out.best_values['exp3_amplitude'], vary=False)
            fullPars['exp3_decay'].set(value=exp_out.best_values['exp3_decay'], vary=False)
        else:
            print(exp_out.ci_report())
            fullPars['exp3_amplitude'].set(value=0, vary=False)
            fullPars['exp3_decay'].set(value=fullPars['exp3_decay'].max, vary=False)
        #
        print('####')
        if verbose:
            print(fullPars['exp3_amplitude'])
            print(fullPars['exp3_decay'])
        print('####')
        #
        intermed_y = intermed_y - exp_out.eval(fullPars, x=x)
        if verbose:
            print(exp_out.fit_report())
        intermed_stats = np.percentile(intermed_y, q=[1, 99])
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
        for pref in ['n1', 'p1', 'p2', 'p3', 'n2', 'p4']:
            pName = '{}_sigma'.format(pref)
            fullPars[pName].set(
                value=np.random.uniform(
                    fullPars[pName].min,
                    fullPars[pName].max))
        # Randomize n1_offset
        fullPars['n1_offset'].set(
            value=np.random.uniform(
                fullPars['n1_offset'].min, fullPars['n1_offset'].max
                ))
        # positives
        for pref in ['p1', 'p2', 'p3', 'p4']:
            pName = '{}_amplitude'.format(pref)
            maxAmp = min(intermed_iqr, absMaxPotential)
            fullPars[pName].set(
                # value=.1 * maxAmp,  # vary=False,
                value=0,
                min=1e-3 * maxAmp, max=maxAmp
                )
        # negatives
        for pref in ['n1', 'n2']:
            pName = '{}_amplitude'.format(pref)
            maxAmp = min(intermed_iqr, absMaxPotential)
            fullPars[pName].set(
                value=0,
                # value=-.1 * maxAmp,
                max=-1e-3 * maxAmp,
                min=-maxAmp
                )
        # freeze any?
        freezeGaussians = ['p1']
        # freezeGaussians = []
        # freezeGaussians = ['p1', 'n1', 'p2', 'p3']
        if len(freezeGaussians):
            for pref in freezeGaussians:
                pName = '{}_amplitude'.format(pref)
                fullPars[pName].set(value=0, vary=False)
                pName = '{}_sigma'.format(pref)
                fullPars[pName].set(
                    value=fullPars[pName].min,
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
        group, dataColNames=None,
        tBounds=None, verbose=False, slowExpTBounds=None,
        fastExpTBounds=None, medExpTBounds=None,
        assessSlow=False, assessMed=False, assessFast=False,
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
                scoreBounds=scBnds,
                slowExpTBounds=slowExpTBounds,
                fastExpTBounds=fastExpTBounds,
                medExpTBounds=medExpTBounds,
                assessSlow=assessSlow, assessMed=assessMed,
                assessFast=assessFast,
                verbose=verbose, plotting=plotting)
            #
            outList['best_fit'].append(resultsSrs)
            outList['params'].append(resultsParams)
            outList['model'].append(mod)
    prelimParams = pd.DataFrame(outList['params'])
    prelimDF = pd.DataFrame(outList['best_fit'])
    bestIdx = prelimParams['r2'].argmax()
    resultDF = pd.concat([prelimDF, prelimParams], axis='columns').loc[bestIdx, :].to_frame().T
    resultDF.index = [group.index[0]]
    bestModel = outList['model'][bestIdx]
    if bestModel is not None:
        bestModel.conf_interval()
    resultDF.loc[group.index[0], 'model'] = bestModel
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
        featNames = dataDF.index.get_level_values('feature')
        elecNames = dataDF.index.get_level_values('electrode')
        rates = dataDF.index.get_level_values('RateInHz')
        amps = dataDF.index.get_level_values(arguments['amplitudeFieldName'])
        print('Available rates are {}'.format(np.unique(rates)))
        if SMALLDATASET:
            dbMask = (
                # featNames.str.contains('rostralY_e12') &
                # elecNames.str.contains('caudalZ_e23') &
                # (featNames.str.contains('caudalY_e11') | featNames.str.contains('rostralY_e11')) &
                elecNames.str.contains('caudalY_e11') &
                (rates < funKWArgs['tBounds'][-1] ** (-1)) &
                (amps < -720)
                )
        else:
            dbMask = (rates < funKWArgs['tBounds'][-1] ** (-1))
        #
        dataDF = dataDF.loc[dbMask, :]
        daskClient = Client()
        resDF = ash.splitApplyCombine(
            dataDF,
            fun=shapeFit, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, **daskOpts)
        if not (useCachedResult and os.path.exists(resultPath)):
            if os.path.exists(resultPath):
                shutil.rmtree(resultFolder)
                os.makedirs(resultFolder)
        resDF.index = dataDF.index
        modelsSrs = resDF['model'].copy()
        resDF.drop(columns='model', inplace=True)
        modelStatsNames.remove('model')
        modelStatsNames.append('model_index')
        resDF.loc[:, 'model_index'] = range(resDF.shape[0])
        resDF.set_index(modelParamNames + modelStatsNames, inplace=True, append=True)
        resDF.to_hdf(resultPath, 'lmfit_lfp')
        presentNames = [cn for cn in resDF.index.names if cn in dataDF.index.names]
        meansDF = dataDF.groupby(presentNames).mean()
        meansDF.to_hdf(resultPath, 'lfp')
        for idx, metaIdx in enumerate(modelsSrs.index):
            modelResult = modelsSrs[metaIdx]
            if not np.isnan(modelResult):
                thisPath = os.path.join(
                    resultFolder,
                    prefix + '_{}_{}_lmfit_{}.sav'.format(
                        arguments['inputBlockSuffix'], arguments['window'], idx))
                try:
                    save_modelresult(modelResult, thisPath)
                except Exception:
                    traceback.print_exc()
    else:
        modelStatsNames.remove('model')
        modelStatsNames.append('model_index')
        resDF = pd.read_hdf(resultPath, 'lmfit_lfp')
        meansDF = pd.read_hdf(resultPath, 'lfp')
    #########################################################################
    modelParams = resDF.index.to_frame()  #.reset_index(drop=True)
    allIdxNames = resDF.index.names
    modelIndex = pd.Series(
        resDF.index.get_level_values('model_index'),
        index=resDF.index)
    resDF = resDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    modelParams = modelParams.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    modelIndex = modelIndex.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    allIdxNames = meansDF.index.names
    meansDF = meansDF.droplevel([cn for cn in allIdxNames if cn not in groupBy])
    resDF.columns = resDF.columns.astype(np.float) / 1e3 + funKWArgs.pop('tOffset', 0)
    compsDict = {}
    paramsCI = pd.DataFrame(np.nan, index=modelParams.index, columns=modelParamNames)
    for rowIdx, row in resDF.iterrows():
        thisMIdx = modelIndex.loc[rowIdx]
        modelPath = thisPath = os.path.join(
            resultFolder,
            prefix + '_{}_{}_lmfit_{}.sav'.format(
                arguments['inputBlockSuffix'], arguments['window'], thisMIdx))
        try:
            model = load_modelresult(modelPath)
            if hasattr(model, 'ci_out'):
                ciDF = pd.DataFrame(model.ci_out)
                for pName in ciDF.columns:
                    paramsCI.loc[rowIdx, pName] = ciDF.loc[5, pName][1] - ciDF.loc[1, pName][1]
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
                    compsDict[key] = [value]
                else:
                    compsDict[key].append(value)
    relativeCI = (paramsCI / modelParams).loc[:, modelParamNames]
    modelParams = modelParams.loc[:, modelParamNames]
    meanParams = modelParams.groupby(['electrode', arguments['amplitudeFieldName'], 'feature']).mean() * np.nan
    for name, group in modelParams.groupby(['electrode', arguments['amplitudeFieldName'], 'feature']):
        for pref in ['n1_', 'n2_', 'p1_', 'p2_', 'p3_', 'p4_']:
            prefMask = group.columns.str.startswith(pref)
            subGroup = group.loc[:, prefMask].reset_index(drop=True)
            subGroupCI = relativeCI.loc[group.index, prefMask].reset_index(drop=True)
            goodFitMask = (subGroupCI[pref + 'amplitude'].abs() < 1).to_numpy()
            subGroupMean = subGroup.loc[goodFitMask, :].mean()
            if np.isnan(subGroupMean[pref + 'amplitude']):
                subGroupMean.loc[pref + 'amplitude'] = 0
            meanParams.loc[name, subGroupMean.index] = subGroupMean
    meanParams.loc[:, 'N1P2'] = meanParams['p2_amplitude'] - meanParams['n1_amplitude']
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
    compsAndTargetDF.columns.name = 'bin'
    #
    plotDF = compsAndTargetDF.stack().to_frame(name='signal').reset_index()
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
            'label': 'N1 Latency'
        },
        'n1_sigma': {
            'units': 'msec',
            'label': 'N1 Sigma',
        },
        'n2_center': {
            'units': 'msec',
            'label': 'N2 Latency'
        },
        'n2_sigma': {
            'units': 'msec',
            'label': 'N2 Sigma'
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
    with PdfPages(pdfPath) as pdf:
        for name, group in tqdm(meanParams.groupby(['electrode', arguments['amplitudeFieldName']])):
            for pName, plotMeta in paramMetaData.items():
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 24)
                data = group.loc[:, ['xCoords', 'yCoords', pName]]
                # data.loc[group['r2'] < 0.9, pName] = np.nan
                dataSquare = data.pivot('yCoords', 'xCoords', pName)
                anns = group.loc[:, ['xCoords', 'yCoords', 'feature']]
                annsSquare = anns.pivot('yCoords', 'xCoords', 'feature')
                ax = sns.heatmap(
                    dataSquare, annot=annsSquare, fmt='s',
                    annot_kws={'fontsize': 'xx-small'},
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
                plt.close()
                # plt.show()
    ###########################
    '''
    pdfPath = os.path.join(
        alignedFeaturesFolder,
        prefix + '_{}_{}_lmfit_3_msec.pdf'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    with PdfPages(pdfPath) as pdf:
        for name, group in tqdm(plotDF.groupby('feature')):
            g = sns.relplot(
                # data=group,
                data=group.query('bin < 3e-3'),
                x='bin', y='signal',
                hue='regrID',
                row='rowLabel', col='columnLabel',
                facet_kws=dict(sharey=False),
                **relplotKWArgs)
            g.fig.suptitle(name)
            pdf.savefig()
            plt.close()
            # plt.show()
    '''
    # # export model params and confidence intervals
    # pdb.set_trace()
    outputParams = modelParams.reset_index()
    outputParams.columns = outputParams.columns.astype(np.str)
    resultPath = os.path.join(
        resultFolder,
        prefix + '_{}_{}_lmfit_model_parameters.parquet'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    outputParams.to_parquet(resultPath, engine="fastparquet")
    #
    outputComps = compsAndTargetDF.reset_index()
    outputComps.columns = outputComps.columns.astype(np.str)
    resultPath = os.path.join(
        resultFolder,
        prefix + '_{}_{}_lmfit_signals.parquet'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    outputComps.to_parquet(resultPath, engine="fastparquet")
    # 
    outputCI = paramsCI.reset_index()
    outputCI.columns = outputCI.columns.astype(np.str)
    resultPath = os.path.join(
        resultFolder,
        prefix + '_{}_{}_lmfit_confidence_intervals.parquet'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    outputCI.to_parquet(resultPath, engine="fastparquet")