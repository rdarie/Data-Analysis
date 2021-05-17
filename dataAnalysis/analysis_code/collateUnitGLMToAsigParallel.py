"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --debugging                            print diagnostics? [default: False]
    --attemptMPI                           whether to try to load MPI [default: False]
    --makePredictionPDF                    make pdf of y vs yhat? [default: False]
    --plottingCovariateFilters             make pdf of y vs yhat? [default: False]
    --makeCovariatePDF                     make pdf of y vs yhat? [default: False]
    --profile                              print time and mem diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --plottingIndividual                   plots? [default: False]
    --plottingOverall                      plots? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --estimatorName=estimator              estimator filename
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockName=inputBlockName        filename for resulting estimator [default: fr_sqrt]
"""
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import os, glob, re
from scipy import stats
from sklearn import utils
import pandas as pd
import numpy as np
import pdb, traceback
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle
import pyglmnet
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests as mt
from patsy import (
    ModelDesc, EvalEnvironment, Term, Sum, Treatment, INTERCEPT,
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices, build_design_matrices)
from dataAnalysis.custom_transformers.tdr import *
from dataAnalysis.custom_transformers.tdr import _poisson_pseudoR2, _FUDE
from itertools import product, combinations
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
import matplotlib
# matplotlib.use('Agg')  # generate postscript output 
matplotlib.use('QT5Agg')  # generate postscript output
import matplotlib.ticker as ticker
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("darkgrid")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

try:
    if not arguments['attemptMPI']:
        raise(Exception('MPI aborted by cmd line argument'))
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    HAS_MPI = True
except Exception:
    traceback.print_exc()
    RANK = 0
    SIZE = 1
    HAS_MPI = False
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName']
    )
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
fullEstimatorName = '{}_{}_{}_{}'.format(
    prefix,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])
#
estimatorSubFolder = os.path.join(
        alignSubFolder, fullEstimatorName)
estimatorFiguresFolder = os.path.join(
    GLMFiguresFolder, fullEstimatorName)
if not os.path.exists(estimatorFiguresFolder):
    os.makedirs(estimatorFiguresFolder, exist_ok=True)
#
variantMetaDataPath = os.path.join(
    estimatorSubFolder, 'variant_metadata.xlsx')
featuresMetaDataPath = os.path.join(
    estimatorSubFolder, 'features_meta.pickle')

nDebuggingPlots = 5
cPalettes = sns.color_palette()

varFolders = sorted(
    glob.glob(os.path.join(alignSubFolder, fullEstimatorName, 'var_*')))
variantList = [os.path.basename(i) for i in varFolders]
if arguments['debugging']:
    showNow = False
    # variantList = ['var_{:03d}'.format(i) for i in range(6)]
    variantList = ['var_{:03d}'.format(i) for i in [6]]
else:
    showNow = False

with open(featuresMetaDataPath, 'rb') as f:
    featuresMetaData = pickle.load(f)
    ihbasis = featuresMetaData['ihbasis']
    iht = featuresMetaData['iht']
#
cBasis = pd.read_excel(
    variantMetaDataPath, sheet_name='covariateBasis', index_col=0)


def applyCBasis(vec, cbCol):
    return np.convolve(
        vec.to_numpy(),
        cBasis[cbCol].to_numpy(),
        mode='same')

useInvLink = True
if useInvLink:
    invLinkFun = lambda x: np.exp(x)
else:
    invLinkFun = lambda x: x
#
covPalette = sns.color_palette("Set1", n_colors=8)
basicCovariateNames = [
    'GT_Right', 'K_Right', 'M_Right',
    'program0_amplitude', 'program1_amplitude', 'program2_amplitude'
    ]
examplePlottedCovariateNames = [
    'GT_Right_angle', 'K_Right_angle', 'M_Right_angle',
    'program0_amplitude', 'program1_amplitude', 'program2_amplitude']
covFilterPlotOpts = {
    'GT_Right': {'c': covPalette[0], 'ls': '-', 'lw': 2},
    'K_Right': {'c': covPalette[1], 'ls': '-', 'lw': 2},
    'M_Right': {'c': covPalette[2], 'ls': '-', 'lw': 2},
    'stimulation': {'c': 'r', 'ls': '-', 'lw': 2},
    'kinematics': {'c': 'b', 'ls': '-', 'lw': 2},
    'present': {'c': 'g', 'ls': '-', 'lw': 2},
    'past': {'c': 'y', 'ls': '-', 'lw': 2},
    'future': {'c': 'c', 'ls': '-', 'lw': 2},
    'GT_Right_angle': {'c': covPalette[0], 'ls': '-', 'lw': 3},
    'K_Right_angle': {'c': covPalette[1], 'ls': '-', 'lw': 3},
    'M_Right_angle': {'c': covPalette[2], 'ls': '-', 'lw': 3},
    'GT_Right_angular_velocity': {'c': covPalette[0], 'ls': '--', 'lw': 3},
    'K_Right_angular_velocity': {'c': covPalette[1], 'ls': '--', 'lw': 3},
    'M_Right_angular_velocity': {'c': covPalette[2], 'ls': '--', 'lw': 3},
    'GT_Right_angular_acceleration': {'c': covPalette[0], 'ls': '.', 'lw': 3},
    'K_Right_angular_acceleration': {'c': covPalette[1], 'ls': '.', 'lw': 3},
    'M_Right_angular_acceleration': {'c': covPalette[2], 'ls': '.', 'lw': 3},
    'program0_amplitude': {'c': covPalette[3], 'ls': '-', 'lw': 3},
    'program1_amplitude': {'c': covPalette[4], 'ls': '-', 'lw': 3},
    'program2_amplitude': {'c': covPalette[5], 'ls': '-', 'lw': 3},
    'program3_amplitude': {'c': covPalette[6], 'ls': '-', 'lw': 3},
    'RateInHz': {'c': covPalette[6], 'ls': '-', 'lw': 3},
    'program0_amplitude:RateInHz': {'c': covPalette[3], 'ls': '-', 'lw': 1},
    'program1_amplitude:RateInHz': {'c': covPalette[4], 'ls': '-', 'lw': 1},
    'program2_amplitude:RateInHz': {'c': covPalette[5], 'ls': '-', 'lw': 1},
    'program3_amplitude:RateInHz': {'c': covPalette[6], 'ls': '-', 'lw': 1},
    'deriv(program0_amplitude)': {'c': covPalette[3], 'ls': '--', 'lw': 3},
    'deriv(program1_amplitude)': {'c': covPalette[4], 'ls': '--', 'lw': 3},
    'deriv(program2_amplitude)': {'c': covPalette[5], 'ls': '--', 'lw': 3},
    'deriv(program3_amplitude)': {'c': covPalette[6], 'ls': '--', 'lw': 3},
    'deriv(program0_amplitude):RateInHz': {'c': covPalette[3], 'ls': '--', 'lw': 1},
    'deriv(program1_amplitude):RateInHz': {'c': covPalette[4], 'ls': '--', 'lw': 1},
    'deriv(program2_amplitude):RateInHz': {'c': covPalette[5], 'ls': '--', 'lw': 1},
    'deriv(program3_amplitude):RateInHz': {'c': covPalette[6], 'ls': '--', 'lw': 1},
    }
termsToCovLookup = {
    'kvsdP1P2P3': 'program0_amplitude',
    'kvsdP0P2P3': 'program1_amplitude',
    'kvsdP0P1P3': 'program2_amplitude',
    'kvsdP0P1P2': 'program3_amplitude',
    'kv': 'stimulation',
    'sdP0P1P2P3': 'kinematics'
    }
binSize = (
    rasterOpts['binInterval'] *
    glmOptsLookup[arguments['estimatorName']]['subsampleOpts']['decimate'])
rollingWindow = int(150e-3 * binSize ** -1)
tolList = [1e-2, 1e-4, 1e-6]
rollingWindowList = [
    int(i / binSize)
    for i in [50e-3, 100e-3, 200e-3]
    ]
deriv = lambda x: pd.Series(x).diff().fillna(0).to_numpy()
#
ihbasisDF = pd.read_excel(
    variantMetaDataPath, sheet_name='ensembleBasis', index_col=0)
variantInfo = pd.read_excel(variantMetaDataPath, sheet_name='variantInfo')
variantInfo['variantName'] = ['var_{:03d}'.format(i) for i in variantInfo.index]
iht = np.array(ihbasisDF.index)
ihbasis = ihbasisDF.to_numpy()
saveR2 = {}
saveEstimatorParams = {}
saveBetas = {}
saveLL = {}

def calcMinIter(iterBetaNorm, tol_list=[1e-6], plotting=False):
    if isinstance(iterBetaNorm, pd.Series):
        ibn = iterBetaNorm
    else:
        ibn = pd.Series(iterBetaNorm)
    updateSize = (ibn.diff().abs() / ibn.shift(1)).fillna(0)
    terminateIndices = []
    if plotting:
        fig, ax = plt.subplots()
        ax.plot(ibn)
    for tol in tol_list:
        tolMask = updateSize >= tol
        if (tolMask).any():
            idxOverTol = updateSize.index[tolMask]
            terminIdx = idxOverTol.max()
        else:
            terminIdx = updateSize.index[-1]
        terminateIndices.append(terminIdx)
        if plotting:
            ax.plot(terminIdx, ibn[terminIdx], '*')
    if plotting:
        return terminateIndices, fig, ax
    else:
        return terminateIndices


for variantName in variantList:
    variantIdx = int(variantName[-3:])
    variantInfo.loc[variantIdx, 'singleLag'] = (
        (
            variantInfo.loc[variantIdx, 'minKinLag'] ==
            variantInfo.loc[variantIdx, 'maxKinLag']) and
        (
            variantInfo.loc[variantIdx, 'minStimLag'] ==
            variantInfo.loc[variantIdx, 'maxStimLag'])
        )
    if variantInfo.loc[variantIdx, 'singleLag']:
        variantInfo.loc[variantIdx, 'stimLag'] = (
            variantInfo.loc[variantIdx, 'minStimLag'])
        variantInfo.loc[variantIdx, 'kinLag'] = (
            variantInfo.loc[variantIdx, 'minKinLag'])
    variantInfo.loc[variantIdx, 'stimLagStr'] = (
        '{:.3f} to {:.3f}'.format(
            variantInfo.loc[variantIdx, 'minStimLag'],
            variantInfo.loc[variantIdx, 'maxStimLag']))
    variantInfo.loc[variantIdx, 'kinLagStr'] = (
        '{:.3f} to {:.3f}'.format(
            variantInfo.loc[variantIdx, 'minKinLag'],
            variantInfo.loc[variantIdx, 'maxKinLag']))
    equalLag = (
        variantInfo.loc[variantIdx, 'kinLagStr'] ==
        variantInfo.loc[variantIdx, 'kinLagStr'])
    variantInfo.loc[variantIdx, 'equalLag'] = equalLag
    terms = ''
    variantInfo.loc[variantIdx, 'anyKin'] = False
    variantInfo.loc[variantIdx, 'anyStim'] = False
    if variantInfo.loc[variantIdx, 'ensembleTerms']:
        terms += 'e'
    if variantInfo.loc[variantIdx, 'angle']:
        terms += 'k'
        variantInfo.loc[variantIdx, 'anyKin'] = True
    if variantInfo.loc[variantIdx, 'angularVelocity']:
        terms += 'v'
        variantInfo.loc[variantIdx, 'anyKin'] = True
    if variantInfo.loc[variantIdx, 'angularAcceleration']:
        terms += 'a'
        variantInfo.loc[variantIdx, 'anyKin'] = True
    if variantInfo.loc[variantIdx, 'stimAmplitude']:
        terms += 's'
        variantInfo.loc[variantIdx, 'anyStim'] = True
    if variantInfo.loc[variantIdx, 'stimVelocity']:
        terms += 'd'
        variantInfo.loc[variantIdx, 'anyStim'] = True
    if variantInfo.loc[variantIdx, 'program0_amplitude']:
        terms += 'P0'
    if variantInfo.loc[variantIdx, 'program1_amplitude']:
        terms += 'P1'
    if variantInfo.loc[variantIdx, 'program2_amplitude']:
        terms += 'P2'
    if variantInfo.loc[variantIdx, 'program3_amplitude']:
        terms += 'P3'
    variantInfo.loc[variantIdx, 'kinAndStim'] = (
        variantInfo.loc[variantIdx, 'anyKin'] and
        variantInfo.loc[variantIdx, 'anyStim']
        )
    variantInfo.loc[variantIdx, 'kinOnly'] = (
        variantInfo.loc[variantIdx, 'anyKin'] and
        (not variantInfo.loc[variantIdx, 'anyStim'])
        )
    variantInfo.loc[variantIdx, 'stimOnly'] = (
        (not variantInfo.loc[variantIdx, 'anyKin']) and
        variantInfo.loc[variantIdx, 'anyStim']
        )
    variantInfo.loc[variantIdx, 'terms'] = terms

if (RANK == 0):
    targetH5Path = os.path.join(
        estimatorSubFolder, '_raster_long.h5')
    targetDF = pd.read_hdf(targetH5Path, 'target')
    regressorH5Path = targetH5Path.replace('_raster', '_rig')
    featuresDF = pd.read_hdf(regressorH5Path, 'feature')
    featuresDF.columns.name = 'feature'
else:
    featuresDF = None
    targetDF = None
if HAS_MPI:
    COMM.Barrier()  # sync MPI threads, wait for 0
    featuresDF = COMM.bcast(featuresDF, root=0)
    targetDF = COMM.bcast(targetDF, root=0)

for vIdx, variantName in enumerate(variantList):
    variantIdx = int(variantName[-3:])
    estimatorMetadataPath = os.path.join(
       alignSubFolder,
       fullEstimatorName, variantName, 'estimator_metadata.pickle')
    estimatorPath = os.path.join(
        estimatorSubFolder, variantName, 'estimator.joblib')
    if (RANK == 0):
        with open(estimatorMetadataPath, 'rb') as f:
            estimatorMetadata = pickle.load(f)
        if os.path.exists(estimatorPath):
            print('loading {}'.format(estimatorPath))
            estimator = jb.load(estimatorPath)
        else:
            partialMatches = sorted(glob.glob(
                os.path.join(
                    alignSubFolder, fullEstimatorName, variantName,
                    'estimator_*.joblib')))
            if len(partialMatches):
                print('loading {}'.format(partialMatches[0]))
                estimator = jb.load(partialMatches[0])
                for otherMatch in partialMatches[1:]:
                    print('loading {}'.format(otherMatch))
                    newEstimator = jb.load(otherMatch)
                    estimator.regressionList.update(newEstimator.regressionList)
                    estimator.betas = pd.concat(
                        [estimator.betas, newEstimator.betas])
                    if hasattr(estimator, 'pvals'):
                        estimator.pvals = pd.concat(
                            [estimator.pvals, newEstimator.pvals])
                    if hasattr(estimator, 'significantBetas'):
                        estimator.significantBetas = pd.concat([
                            estimator.significantBetas,
                            newEstimator.significantBetas
                            ])
                    os.remove(otherMatch)
                os.remove(partialMatches[0])
                jb.dump(estimator, estimatorPath)
        # #  Remake X and y
        if vIdx == 0:
            train = estimatorMetadata['trainIdx']
            test = estimatorMetadata['testIdx']
            cv_folds = estimatorMetadata['cv_folds']
            cv_test_masks = [
                featuresDF.index.isin(featuresDF.index[train][cv_idx[1]])
                for cv_idx in cv_folds]
            cv_train_masks = [
                featuresDF.index.isin(featuresDF.index[train][cv_idx[0]])
                for cv_idx in cv_folds]
            testFoldMask = featuresDF.index.isin(featuresDF.index[test])
            trainFeatures = featuresDF.iloc[train, :]
            testFeatures = featuresDF.iloc[test, :]
        xTrainList = []
        xTestList = []
        if variantInfo.loc[variantIdx, 'addIntercept']:
            xTrainList.append(dmatrix(
                ModelDesc([], [Term([])]),
                featuresDF.iloc[train, :],
                return_type='dataframe'))
            xTestList.append(dmatrix(
                ModelDesc([], [Term([])]),
                featuresDF.iloc[test, :],
                return_type='dataframe'))
        if 'kinModelDesc' in estimatorMetadata.keys():
            kinModelDesc = ModelDesc.from_formula(estimatorMetadata['kinModelDesc'])
            kinXTrain = dmatrix(
                kinModelDesc,
                featuresDF.iloc[train, :],
                return_type='dataframe')
            kinXTest = build_design_matrices(
                [kinXTrain.design_info], featuresDF.iloc[test, :],
                return_type='dataframe')[0]
            kinModelDescFinal = ModelDesc.from_formula(estimatorMetadata['kinModelDescFinal'])
            kinXTrainFinal = dmatrix(
                kinModelDescFinal,
                kinXTrain,
                return_type='dataframe')
            xTrainList.append(kinXTrainFinal)
            kinXTestFinal = dmatrix(
                kinModelDescFinal,
                kinXTest,
                return_type='dataframe')
            xTestList.append(kinXTestFinal)
        if 'stimModelDesc' in estimatorMetadata.keys():
            stimModelDesc = ModelDesc.from_formula(estimatorMetadata['stimModelDesc'])
            stimXTrain = dmatrix(
                stimModelDesc,
                featuresDF.iloc[train, :],
                return_type='dataframe')
            stimXTest = build_design_matrices(
                [stimXTrain.design_info], featuresDF.iloc[test, :],
                return_type='dataframe')[0]
            stimModelDescFinal = ModelDesc.from_formula(estimatorMetadata['stimModelDescFinal'])
            stimXTrainFinal = dmatrix(
                stimModelDescFinal,
                stimXTrain,
                return_type='dataframe')
            xTrainList.append(stimXTrainFinal)
            stimXTestFinal = dmatrix(
                stimModelDescFinal,
                stimXTest,
                return_type='dataframe')
            xTestList.append(stimXTestFinal)
        if 'histDesc' in estimatorMetadata.keys():
            histDesc = ModelDesc.from_formula(estimatorMetadata['histDesc'])
            xTrainList.append(dmatrix(
                histDesc,
                featuresDF.iloc[train, :],
                return_type='dataframe'))
            xTestList.append(dmatrix(
                histDesc,
                featuresDF.iloc[test, :],
                return_type='dataframe'))
        #
        unitNames = list(estimator.regressionList.keys())
        #
        estimator.xTrain = pd.concat(xTrainList, axis='columns')
        estimator.xTest = pd.concat(xTestList, axis='columns')
        estimator.yTrain = targetDF.iloc[train, :]
        estimator.yTest = targetDF.iloc[test, :]
    else:
        estimator = None
        estimatorMetadata = None
        unitNames = None
        train = None
        test = None
        cv_folds = None
        cv_test_masks = None
        cv_train_masks = None
        testFoldMask = None
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        estimator = COMM.bcast(estimator, root=0)
        estimatorMetadata = COMM.bcast(estimatorMetadata, root=0)
        unitNames = COMM.bcast(unitNames, root=0)
        train = COMM.bcast(train, root=0)
        test = COMM.bcast(test, root=0)
        cv_folds = COMM.bcast(cv_folds, root=0)
        cv_test_masks = COMM.bcast(cv_test_masks, root=0)
        cv_train_masks = COMM.bcast(cv_train_masks, root=0)
        testFoldMask = COMM.bcast(testFoldMask, root=0)
    #
    estimatorDiagnosticsPath = os.path.join(
       alignSubFolder,
       fullEstimatorName, variantName, 'estimator_diagnostics.h5')
    if os.path.exists(estimatorDiagnosticsPath):
        raise(Exception('Only implemnted for novel calculation'))
    else:
        yColIdx = range(RANK, targetDF.shape[1], SIZE)
        thisRankUnitNames = [unitNames[i] for i in yColIdx]
        print('RANK = {}, analyzing units {}'.format(RANK, thisRankUnitNames))
        # can be broken down by unit
        yHat = pd.DataFrame(
            np.nan,
            index=targetDF.index,
            columns=thisRankUnitNames,
            # columns=targetDF.columns
            )
        # can be broken down by unit
        pR2 = pd.DataFrame(
            np.nan,
            index=thisRankUnitNames,
            columns=['gs', 'cv', 'valid', 'valid_LL'])
        #
        # can be broken down by unit
        estimatorParams = pd.DataFrame(
            np.nan,
            index=thisRankUnitNames,
            columns=['minIter', 'reg_lambda'])
        # will make pR2smooth
        pR2smoothDict = {
            i: pd.DataFrame(
                np.nan,
                index=thisRankUnitNames,
                columns=['pR2'])
            for i in rollingWindowList
            }
        toleranceParamsList = []
        # calculate yHat, pR2, pR2smooth, toleranceParams, saveCVBetas
        for uIdx, unitName in enumerate(thisRankUnitNames):
            regDict = estimator.regressionList[unitName]
            yHat.loc[testFoldMask, unitName] = (
                regDict['reg'].predict(estimator.xTest))
            if hasattr(regDict['reg'], 'get_params'):
                if 'distr' in regDict['reg'].get_params():
                    pR2.loc[unitName, 'valid_LL'] = pyglmnet._logL(
                        regDict['reg'].distr,
                        estimator.yTest[unitName],
                        yHat.loc[testFoldMask, unitName])
                    pyglmnet._logL(
                        regDict['reg'].distr,
                        estimator.yTest[unitName],
                        estimator.yTest[unitName].mean())
            if hasattr(regDict['reg'], 'model_'):
                try:
                    regDict['reg'].model_.endog = estimator.xTrain
                    regDict['reg'].model_.exog = estimator.yTrain.loc[:, unitName]
                except Exception:
                    traceback.print_exc()
            #
            if 'gridsearch_best_mean_test_score' in regDict:
                pR2.loc[unitName, 'gs'] = regDict['gridsearch_best_mean_test_score']
            if 'cross_val_mean_test_score' in regDict:
                pR2.loc[unitName, 'cv'] = regDict['cross_val_mean_test_score']
            if 'validationScore' in regDict:
                pR2.loc[unitName, 'valid'] = regDict['validationScore']
            if hasattr(regDict['reg'], 'reg_lambda'):
                estimatorParams.loc[unitName, 'reg_lambda'] = (
                    regDict['reg'].reg_lambda)
            if 'gridsearch' in regDict:
                gs = regDict['gridsearch']
                splitScores = [
                    i
                    for i in gs.cv_results_.keys()
                    if '_test_score' in i and 'split' in i]
                for splitScore in splitScores:
                    splitName = splitScore.split('_test_score')[0]
                    pR2.loc[unitName, 'gs_' + splitName] = (
                        gs.cv_results_[splitScore][gs.best_index_])
            if 'cross_val' in regDict:
                saveCVBetas = True
                cv_scores = regDict['cross_val']
                if uIdx == 0:
                    cvBetas = [
                        pd.DataFrame(
                            0,
                            index=thisRankUnitNames,
                            columns=estimator.betas.columns)
                        for i in cv_scores['estimator']]
                for cvIdx, cvEst in enumerate(cv_scores['estimator']):
                    cvBetas[cvIdx].loc[unitName, :] = cvEst.beta_
                    cv_test = cv_folds[cvIdx][1]
                    yHat.loc[cv_test_masks[cvIdx], unitName] = (
                        cvEst.predict(estimator.xTrain.iloc[cv_test, :]))
            else:
                saveCVBetas = False
            if 'iterBetaNorm' in regDict:
                iterBetaNorm = regDict['iterBetaNorm']
                iterScore = regDict['iterScore']
                termIndices = calcMinIter(iterBetaNorm, tol_list=tolList)
                theseParams = pd.DataFrame({
                    'tol': [
                        tolList[i]
                        for i in range(len(tolList))],
                    'minIter': [
                        termIndices[i]
                        for i in range(len(tolList))],
                    'score': [
                        iterScore[termIndices[i]]
                        for i in range(len(tolList))]})
                theseParams['marginalIter'] = theseParams['minIter'].diff()
                theseParams['marginalScore'] = theseParams['score'].diff()
                toleranceParamsList.append(theseParams)
                estimatorParams.loc[unitName, 'minIter'] = termIndices[-1]
            for rIdx, thisRollingWindow in enumerate(rollingWindowList):
                pR2smoothDict[thisRollingWindow].loc[unitName, 'pR2'] = _poisson_pseudoR2(
                    estimator.yTest[unitName]
                    .rolling(thisRollingWindow, center=True).mean()
                    .dropna().iloc[int(thisRollingWindow/2)::thisRollingWindow]
                    .to_numpy(),
                    yHat.loc[testFoldMask, unitName]
                    .rolling(thisRollingWindow, center=True).mean()
                    .dropna().iloc[int(thisRollingWindow/2)::thisRollingWindow]
                    .to_numpy())
        #
        pR2smooth = pd.concat(pR2smoothDict, names=['rollingWindow']).reset_index()
        toleranceParams = pd.concat(toleranceParamsList, ignore_index=True)
        if HAS_MPI:
            COMM.Barrier()  # sync MPI threads, wait for 0
            if RANK == 0:
                print('Gathering pieces from first round')
            yHat_pieces = COMM.gather(yHat, root=0)
            pR2_pieces = COMM.gather(pR2, root=0)
            pR2smooth_pieces = COMM.gather(pR2smooth, root=0)
            toleranceParams_pieces = COMM.gather(toleranceParams, root=0)
            estimatorParams_pieces = COMM.gather(estimatorParams, root=0)
            if saveCVBetas:
                cvBetas_pieces = COMM.gather(cvBetas, root=0)
            if RANK == 0:
                print('yHat_pieces[0].shape = {}'.format(yHat_pieces[0].shape))
                print('yHat_pieces[0].columns = {}'.format(yHat_pieces[0].columns))
                yHat = pd.concat(yHat_pieces, axis='columns')
                print('yHat.shape = {}'.format(yHat.shape))
                print('pR2_pieces[0].shape = {}'.format(pR2_pieces[0].shape))
                print('pR2_pieces[0].columns = {}'.format(pR2_pieces[0].columns))
                pR2 = pd.concat(pR2_pieces)
                print('pR2.shape = {}'.format(pR2.shape))
                print('pR2smooth_pieces[0].shape = {}'.format(pR2smooth_pieces[0].shape))
                print('pR2smooth_pieces[0].columns = {}'.format(pR2smooth_pieces[0].columns))
                pR2smooth = pd.concat(pR2smooth_pieces)
                print('pR2smooth.shape = {}'.format(pR2smooth.shape))
                print('toleranceParams_pieces[0].shape = {}'.format(toleranceParams_pieces[0].shape))
                print('toleranceParams_pieces[0].columns = {}'.format(toleranceParams_pieces[0].columns))
                toleranceParams = pd.concat(toleranceParams_pieces)
                print('toleranceParams.shape = {}'.format(toleranceParams.shape))
                print('estimatorParams_pieces[0].shape = {}'.format(estimatorParams_pieces[0].shape))
                print('estimatorParams_pieces[0].columns = {}'.format(estimatorParams_pieces[0].columns))
                estimatorParams = pd.concat(estimatorParams_pieces)
                print('estimatorParams.shape = {}'.format(estimatorParams.shape))
                print('estimatorParams.columns = {}'.format(estimatorParams.columns))
                if saveCVBetas:
                    cvBetasReassembled = [[] for i in range(len((cvBetas_pieces[0])))]
                    for rankIdx, cvBetasPiece in enumerate(cvBetas_pieces):
                        for esbIdx, cvBetasThisFold in enumerate(cvBetasPiece):
                            cvBetasReassembled[esbIdx].append(cvBetasThisFold)
                    cvBetas = [pd.concat(cvBetasThisFold) for cvBetasThisFold in cvBetasReassembled]
                # pdb.set_trace()
                ##################################################################
                # save results
        if RANK == 0:
            yHat.to_hdf(estimatorDiagnosticsPath, '/yHat')
            pR2.to_hdf(estimatorDiagnosticsPath, '/pR2')
            estimatorParams.to_hdf(estimatorDiagnosticsPath, '/estimatorParams')
            pR2smooth.to_hdf(estimatorDiagnosticsPath, '/pR2smooth')
            toleranceParams.to_hdf(estimatorDiagnosticsPath, '/toleranceParams')
            if saveCVBetas:
                for eIdx, estB in enumerate(cvBetas):
                    estB.to_hdf(
                        estimatorDiagnosticsPath,
                        '/cvBetas/cv{}'.format(eIdx))
                estimatorBetasList = cvBetas + [estimator.betas]
            else:
                estimatorBetasList = [estimator.betas]
            unitNames = pR2.sort_values('valid', ascending=False).index.to_list()
        else:
            unitNames = None
            estimatorBetasList = None
        if HAS_MPI:
            COMM.Barrier()  # sync MPI threads, wait for 0
            unitNames = COMM.bcast(unitNames, root=0)
            yHat = COMM.bcast(yHat, root=0)
            pR2 = COMM.bcast(pR2, root=0)
            estimatorParams = COMM.bcast(estimatorParams, root=0)
            pR2smooth = COMM.bcast(pR2smooth, root=0)
            toleranceParams = COMM.bcast(toleranceParams, root=0)
            estimatorBetasList = COMM.bcast(estimatorBetasList, root=0)
            cvBetas = COMM.bcast(cvBetas, root=0)
            
        estimator.yHat = yHat
        # calculate covariate splines
        regex1 = r'\"bs\(([\S\s]*), df=([\d]+)\)\[([\d]+)\][:]*([\S\s]*)\"'#
        regex2 = r"(deriv\()*applyCBasis\(Q\(\"([\S\s]*)\"\),[\s]*([-*\d\.]*)"
        regex3 = r"bs\(([\S\s]*), df=([\d]+)\)\[([\d]+)\][:]*([\s\S]*)"
        # make dummy matrix to get lists of names of splined objects
        if RANK == 0:
            covariateSplines = []
            #
            for cName in estimator.betas.columns:
                matches = re.search(regex1, cName, re.MULTILINE)
                if matches:
                    mainCovariateName = matches.groups()[0]
                    if mainCovariateName not in covariateSplines:
                        covariateSplines.append(mainCovariateName)
            #
            dummyCovariate = pd.DataFrame(
                0,
                index=range(100),
                columns=featuresDF.columns)
            for cName in covariateSplines:
                dummyCovariate[cName] = np.linspace(
                    featuresDF[cName].min(),
                    featuresDF[cName].max(), 100)
            # TODO make this work for any nonlinear covariate
            if 'stimModelDesc' in estimatorMetadata.keys():
                dInfo = stimXTrain.design_info
                dummyX = build_design_matrices(
                    [dInfo], dummyCovariate, return_type='dataframe')[0]
                if len(covariateSplines) and arguments['plottingIndividual']:
                    pdfPath = os.path.join(
                        estimatorFiguresFolder,
                        'covariate_splines.pdf')
                    if not os.path.exists(pdfPath):
                        fig = plt.figure(figsize=(12, 9))
                        gs = gridspec.GridSpec(nrows=1, ncols=1)
                        ax0 = fig.add_subplot(gs[0, 0])
                        for cName in dummyX.columns:
                            splineMatches = re.search(regex3, cName, re.MULTILINE)
                            if any(dummyX[cName]):
                                ax0.plot(
                                    dummyCovariate[splineMatches.groups()[0]],
                                    dummyX[cName], label=cName)
                        ax0.set_label('Normalized amplitude')
                        ax0.set_title(
                            'cell {} covariate weights, model {}'
                            .format(unitName, variantName))
                        plt.legend()
                        plt.savefig(pdfPath)
                        plt.savefig(pdfPath.replace('.pdf', '.png'))
                        if showNow:
                            plt.show()
                        else:
                            plt.close()
            else:
                dummyX = None
        else:
            dummyX = None
            dummyCovariate = None
        if HAS_MPI:
            COMM.Barrier()  # sync MPI threads, wait for 0
            if 'stimModelDesc' in estimatorMetadata.keys():
                dummyX = COMM.bcast(dummyX, root=0)
            dummyCovariate = COMM.bcast(dummyCovariate, root=0)
        #
        covariateFilters = [
            {
                i: pd.DataFrame(index=cBasis.index, columns=[])
                for i in thisRankUnitNames}
            for esb in estimatorBetasList]
        xFiltered = {
            i: pd.DataFrame(index=featuresDF.index, columns=[])
            for i in thisRankUnitNames}
        covariateSplineFilters = [
            {i: {} for i in thisRankUnitNames}
            for esb in estimatorBetasList]
        anyCovariateSplineFilters = False
        for esbIdx, estimatorBetas in (enumerate(estimatorBetasList)):
            print(
                'Calculating covariate filters for fold {} of {}\n'
                .format(esbIdx+1, len(estimatorBetasList)))
            # for cIdx, cName in enumerate(tqdm(estimatorBetas.columns)):
            for cIdx, cName in enumerate(estimatorBetas.columns):
                # print(
                #     'Calculating covariate filters for term {} of {}'
                #     .format(cIdx+1, len(estimatorBetas.columns)))
                matches = re.search(regex2, cName, re.MULTILINE)
                if matches:
                    covariateName = matches.groups()[1]
                    derivTerm = matches.groups()[0] is not None
                    lag = float(matches.groups()[2])
                    splineMatches = re.search(
                        regex3, covariateName, re.MULTILINE)
                    #
                    if not splineMatches:
                        covSaveName = covariateName
                        if derivTerm:
                            covSaveName = 'deriv({})'.format(covSaveName)
                    else:
                        anyCovariateSplineFilters = True
                        mainCovariateName = splineMatches.groups()[0]
                        nDF = splineMatches.groups()[1]
                        splIdx = splineMatches.groups()[2]
                        splName = covariateName.split(':')[0]
                        spl = dummyX[splName]
                        interactName = splineMatches.groups()[3]
                        covSaveName = mainCovariateName
                        if derivTerm:
                            covSaveName = 'deriv({})'.format(covSaveName)
                        if interactName:
                            covSaveName = covSaveName + ':' + interactName
                    for uIdx, unitName in enumerate(thisRankUnitNames):
                        thisCoefficient = estimatorBetas.loc[unitName, cName]
                        alreadyStarted = (
                            covSaveName in covariateFilters[esbIdx][unitName])
                        if not alreadyStarted:
                            # create an entry in the list of filters
                            covariateFilters[esbIdx][unitName][covSaveName] = 0
                            # create an entry in the list of predictions
                            xFiltered[unitName].loc[:, covSaveName] = 0
                            if splineMatches:
                                # create an entry in the list of amp dependent filters
                                covariateSplineFilters[esbIdx][unitName][covSaveName] = (
                                    pd.DataFrame(
                                        np.zeros((
                                            cBasis.shape[0],
                                            dummyCovariate.shape[0])),
                                        index=cBasis.index,
                                        columns=dummyCovariate[mainCovariateName]))
                        # validation fold goes last
                        if esbIdx == (len(estimatorBetasList) - 1):
                            xFiltered[unitName].loc[testFoldMask, covSaveName] += (
                                thisCoefficient * estimator.xTest[cName])
                        elif saveCVBetas:
                            cv_test = cv_folds[esbIdx][1]
                            xFiltered[unitName].loc[cv_test_masks[esbIdx], covSaveName] += (
                                thisCoefficient * estimator.xTrain[cName].iloc[cv_test])
                        if not splineMatches:
                            covariateFilters[esbIdx][unitName][covSaveName] += (
                                cBasis[lag] * thisCoefficient)
                        else:
                            if not alreadyStarted:
                                covariateFilters[esbIdx][unitName][covSaveName] = 0
                            thisSplineFilter = (
                                np.outer(cBasis[lag], spl) * thisCoefficient)
                            covariateFilters[esbIdx][unitName][covSaveName] += (thisSplineFilter[:, -1])
                            covariateSplineFilters[esbIdx][unitName][covSaveName] += thisSplineFilter
        #
        if HAS_MPI:
            COMM.Barrier()  # sync MPI threads, wait for 0
            covariateFilters_pieces = COMM.gather(covariateFilters, root=0)
            covariateSplineFilters_pieces = COMM.gather(covariateSplineFilters, root=0)
            xFiltered_pieces = COMM.gather(xFiltered, root=0)
            if (RANK == 0):
                print('type(covariateFilters_pieces) {}'.format(type(covariateFilters_pieces)))
                for rankIdx, covFilterList in enumerate(covariateFilters_pieces):
                    for esbIdx, covFilterDict in enumerate(covFilterList):
                        covariateFilters[esbIdx].update(covFilterDict)
                        covariateSplineFilters[esbIdx].update(
                            covariateSplineFilters_pieces[rankIdx][esbIdx])
                    xFiltered.update(xFiltered_pieces[rankIdx])
        if (RANK == 0):
            regularFilterNames = covariateFilters[0][unitNames[0]].columns
            covariateFiltersMedian = {
                i: pd.DataFrame(index=cBasis.index, columns=regularFilterNames)
                for i in unitNames}
            covariateFiltersLB = {
                i: pd.DataFrame(index=cBasis.index, columns=regularFilterNames)
                for i in unitNames}
            covariateFiltersHB = {
                i: pd.DataFrame(index=cBasis.index, columns=regularFilterNames)
                for i in unitNames}
            if anyCovariateSplineFilters:
                splineFilterNames = list(
                    covariateSplineFilters[0][unitNames[0]].keys())
                dummySplineFilter = (
                    covariateSplineFilters[0][unitNames[0]][splineFilterNames[0]] * 0)
                covariateSplineFiltersMedian = {
                    cn: {un: dummySplineFilter.copy() for un in unitNames}
                    for cn in splineFilterNames}
                covariateSplineFiltersLB = {
                    cn: {un: dummySplineFilter.copy() for un in unitNames}
                    for cn in splineFilterNames}
                covariateSplineFiltersHB = {
                    cn: {un: dummySplineFilter.copy() for un in unitNames}
                    for cn in splineFilterNames}
            for unitName in unitNames:
                covariateFiltersStack = np.stack(
                    [arr[unitName].to_numpy() for arr in covariateFilters],
                    axis=-1)
                covariateFiltersMedian[unitName].loc[:, :] = np.median(
                    covariateFiltersStack, axis=-1)
                covariateFiltersLB[unitName].loc[:, :] = np.percentile(
                    covariateFiltersStack, 2.5, axis=-1)
                covariateFiltersHB[unitName].loc[:, :] = np.percentile(
                    covariateFiltersStack, 97.5, axis=-1)
                if anyCovariateSplineFilters:
                    for covariateName in covariateSplineFilters[0][unitName].keys():
                        covariateFiltersStack = np.stack(
                            [
                                arr[unitName][covariateName].to_numpy()
                                for arr in covariateSplineFilters],
                            axis=-1)
                        covariateSplineFiltersMedian[covariateName][unitName].loc[:, :] = np.median(
                            covariateFiltersStack, axis=-1)
                        covariateSplineFiltersLB[covariateName][unitName].loc[:, :] = np.percentile(
                            covariateFiltersStack, 2.5, axis=-1)
                        covariateSplineFiltersHB[covariateName][unitName].loc[:, :] = np.percentile(
                            covariateFiltersStack, 97.5, axis=-1)
            covariateFiltersMedianDF = pd.concat(covariateFiltersMedian, names=['target', 'lag']).unstack(level='lag')
            covariateFiltersMedianDF.columns.names = ['covariate', 'lag']
            covariateFiltersHBDF = pd.concat(covariateFiltersHB, names=['target', 'lag']).unstack(level='lag')
            covariateFiltersHBDF.columns.names = ['covariate', 'lag']
            covariateFiltersLBDF = pd.concat(covariateFiltersLB, names=['target', 'lag']).unstack(level='lag')
            covariateFiltersLBDF.columns.names = ['covariate', 'lag']
            if anyCovariateSplineFilters:
                covariateSplineFiltersMedianDF = {}
                covariateSplineFiltersHBDF = {}
                covariateSplineFiltersLBDF = {}
                for cName in splineFilterNames:
                    covariateSplineFiltersMedianDF[cName] = pd.concat(
                        covariateSplineFiltersMedian[cName],
                        names=['target', 'lag']).unstack(level='lag')
                    covariateSplineFiltersHBDF[cName] = pd.concat(
                        covariateSplineFiltersHB[cName],
                        names=['target', 'lag']).unstack(level='lag')
                    covariateSplineFiltersLBDF[cName] = pd.concat(
                        covariateSplineFiltersLB[cName],
                        names=['target', 'lag']).unstack(level='lag')
            #
            covariateFiltersMagnitudeDict = {
                k: (v.abs()).mean() for k, v in covariateFiltersMedian.items()}
            if anyCovariateSplineFilters:
                for uIdx, unitName in enumerate(unitNames):
                    for cName in splineFilterNames:
                        covariateFiltersMagnitudeDict[unitName][cName] = (
                            covariateSplineFiltersMedian[cName][unitName]
                            .iloc[:, -1].abs().mean())
            covariateFiltersMagnitude = pd.concat(
                covariateFiltersMagnitudeDict, axis='columns').T
            ##################################################################
            # save results
            covariateFiltersMedianDF.to_hdf(
                estimatorDiagnosticsPath, '/covariateFilters/temporal/median')
            covariateFiltersHBDF.to_hdf(
                estimatorDiagnosticsPath, '/covariateFilters/temporal/hb')
            covariateFiltersLBDF.to_hdf(
                estimatorDiagnosticsPath, '/covariateFilters/temporal/lb')
            #
            for unitName in unitNames:
                xFiltered[unitName].to_hdf(
                    estimatorDiagnosticsPath,
                    '/xFiltered/{}'.format(unitName))
            if anyCovariateSplineFilters:
                pd.Series(splineFilterNames).to_hdf(
                    estimatorDiagnosticsPath,
                    '/covariateFilters/timeAmplitude/splineFilterNames')
                for cName in splineFilterNames:
                    safeName = cName.replace(':', '')
                    for unsafeChar in ['(', ')']:
                        safeName = safeName.replace(unsafeChar, '')
                    covariateSplineFiltersMedianDF[cName].to_hdf(
                        estimatorDiagnosticsPath,
                        '/covariateFilters/timeAmplitude/{}/median'.format(safeName))
                    covariateSplineFiltersHBDF[cName].to_hdf(
                        estimatorDiagnosticsPath,
                        '/covariateFilters/timeAmplitude/{}/hb'.format(safeName))
                    covariateSplineFiltersLBDF[cName].to_hdf(
                        estimatorDiagnosticsPath,
                        '/covariateFilters/timeAmplitude/{}/lb'.format(safeName))
            covariateFiltersMagnitude.to_hdf(
                estimatorDiagnosticsPath,
                '/covariateFilters/magnitude/median')