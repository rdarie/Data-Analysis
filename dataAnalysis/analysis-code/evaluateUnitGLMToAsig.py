"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --debugging                            print diagnostics? [default: False]
    --makePredictionPDF                    make pdf of y vs yhat? [default: False]
    --plottingCovariateFilters             make pdf of y vs yhat? [default: False]
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
#
import matplotlib
import matplotlib.ticker as ticker
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Agg')  # generate postscript output 
matplotlib.use('QT5Agg')  # generate postscript output
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
#
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
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
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

varFolders = sorted(
    glob.glob(os.path.join(alignSubFolder, fullEstimatorName, 'var_*')))
variantList = [os.path.basename(i) for i in varFolders]
if arguments['debugging']:
    showNow = False
    # variantList = ['var_{:03d}'.format(i) for i in range(10)]
    variantList = ['var_{:03d}'.format(i) for i in [2, 10, 20]]
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


if arguments['plottingOverall']:
    pdfPath = os.path.join(
        estimatorFiguresFolder,
        'covariate_basis.pdf')
    plt.plot(cBasis)
    plt.xlabel('Time (sec)')
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    if showNow:
        plt.show()
    else:
        plt.close()

binSize = (
    rasterOpts['binInterval'] *
    glmOptsLookup[arguments['estimatorName']]['decimate'])
rollingWindow = int(60e-3 / binSize)
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
    if variantInfo.loc[variantIdx, 'stim']:
        terms += 's'
        variantInfo.loc[variantIdx, 'anyStim'] = True
    if variantInfo.loc[variantIdx, 'stimVelocity']:
        terms += 'd'
        variantInfo.loc[variantIdx, 'anyStim'] = True
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

targetH5Path = os.path.join(
    estimatorSubFolder, '_raster_long.h5')
targetDF = pd.read_hdf(targetH5Path, 'target')
regressorH5Path = targetH5Path.replace('_raster', '_rig')
featuresDF = pd.read_hdf(regressorH5Path, 'feature')

for vIdx, variantName in enumerate(variantList):
    variantIdx = int(variantName[-3:])
    estimatorMetadataPath = os.path.join(
       alignSubFolder,
       fullEstimatorName, variantName, 'estimator_metadata.pickle')
    with open(estimatorMetadataPath, 'rb') as f:
        estimatorMetadata = pickle.load(f)
    estimatorPath = os.path.join(
        estimatorSubFolder, variantName, 'estimator.joblib')
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
    ##  Remake X and y
    if vIdx == 0:
        train = estimatorMetadata['trainIdx']
        test = estimatorMetadata['testIdx']
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
    #
    unitNames = estimator.regressionList.keys()
    #
    estimator.xTrain = pd.concat(xTrainList, axis='columns')
    estimator.xTest = pd.concat(xTestList, axis='columns')
    estimator.yTrain = targetDF.iloc[train, :]
    estimator.yTest = targetDF.iloc[test, :]
    #
    estimatorDiagnosticsPath = os.path.join(
       alignSubFolder,
       fullEstimatorName, variantName, 'estimator_diagnostics.h5')
    if os.path.exists(estimatorDiagnosticsPath):
        with pd.HDFStore(estimatorDiagnosticsPath,  mode='r') as store:
            estimator.yHat = pd.read_hdf(store, '/yHat')
            pR2 = pd.read_hdf(store, '/pR2')
            unitNames = pR2.sort_values('valid', ascending=False).index.to_list()
            estimatorParams = pd.read_hdf(store, '/estimatorParams')
            pR2smooth = pd.read_hdf(store, '/pR2smooth')
            rollingWindowList = pd.unique(pR2smooth['rollingWindow']).tolist()
            toleranceParams = pd.read_hdf(store, '/toleranceParams')
            tolList = pd.unique(toleranceParams['tol']).tolist()
            covariateFiltersMedianDF = pd.read_hdf(
                store, '/covariateFilters/temporal/median')
            covariateFiltersHBDF = pd.read_hdf(
                store, '/covariateFilters/temporal/hb')
            covariateFiltersLBDF = pd.read_hdf(
                store, '/covariateFilters/temporal/lb')
            #
            spFiNaIdx = '/'.join([
                '/covariateFilters', 'timeAmplitude',
                'splineFilterNames'
                ])
            anyCovariateSplineFilters = (
                spFiNaIdx in store.keys())
            covariateFiltersMagnitude = pd.read_hdf(
                store, '/covariateFilters/magnitude/median')
            if anyCovariateSplineFilters:
                splineFilterNames = pd.read_hdf(
                    store, spFiNaIdx).to_list()
        if anyCovariateSplineFilters:
            covariateSplineFiltersMedianDF = {}
            covariateSplineFiltersHBDF = {}
            covariateSplineFiltersLBDF = {}
            for cName in splineFilterNames:
                safeName = cName.replace(':', '')
                for unsafeChar in ['(', ')']:
                    safeName = safeName.replace(unsafeChar, '')
                try:
                    covariateSplineFiltersMedianDF[cName] = pd.read_hdf(
                        estimatorDiagnosticsPath, '/'.join([
                            '/covariateFilters', 'timeAmplitude',
                            '{}/median'.format(safeName)
                        ]))
                    covariateSplineFiltersHBDF[cName] = pd.read_hdf(
                        estimatorDiagnosticsPath, '/'.join([
                            '/covariateFilters', 'timeAmplitude',
                            '{}/hb'.format(safeName)
                        ]))
                    covariateSplineFiltersLBDF[cName] = pd.read_hdf(
                        estimatorDiagnosticsPath, '/'.join([
                            '/covariateFilters', 'timeAmplitude',
                            '{}/lb'.format(safeName)
                        ]))
                except Exception:
                    traceback.print_exc()
    else:
        estimator.yHat = pd.DataFrame(
            np.nan,
            index=estimator.yTest.index,
            columns=estimator.yTest.columns)
        pR2 = pd.DataFrame(
            np.nan,
            index=unitNames,
            columns=['gs', 'cv', 'valid', 'valid_LL'])
        #
        estimatorParams = pd.DataFrame(
            np.nan,
            index=unitNames,
            columns=['minIter', 'reg_lambda'])
        toleranceParamsList = []
        tolList = [1e-2, 1e-4, 1e-6]
        rollingWindowList = [
            int(i / binSize)
            for i in [50e-3, 100e-3, 200e-3]
            ]
        pR2smoothDict = {
            i: pd.DataFrame(
                np.nan,
                index=unitNames,
                columns=['pR2'])
            for i in rollingWindowList
            }
        #
        for uIdx, unitName in enumerate(unitNames):
            regDict = estimator.regressionList[unitName]
            estimator.yHat.loc[:, unitName] = (
                regDict['reg'].predict(estimator.xTest))
            if hasattr(regDict['reg'], 'get_params'):
                if 'distr' in regDict['reg'].get_params():
                    pR2.loc[unitName, 'valid_LL'] = pyglmnet._logL(
                        regDict['reg'].distr,
                        estimator.yTest[unitName],
                        estimator.yHat[unitName])
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
                            index=unitNames,
                            columns=estimator.betas.columns)
                        for i in cv_scores['estimator']]
                for cvIdx, cvEst in enumerate(cv_scores['estimator']):
                    cvBetas[cvIdx].loc[unitName, :] = cvEst.beta_
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
                    estimator.yHat[unitName]
                    .rolling(thisRollingWindow, center=True).mean()
                    .dropna().iloc[int(thisRollingWindow/2)::thisRollingWindow]
                    .to_numpy())
        #
        pR2smooth = pd.concat(pR2smoothDict, names=['rollingWindow']).reset_index()
        toleranceParams = pd.concat(toleranceParamsList, ignore_index=True)
        #
        unitNames = pR2.sort_values('valid').index.to_list()
        #
        covariateSplines = []
        regex1 = r'\"bs\(([\S\s]*), df=([\d]+)\)\[([\d]+)\][:]*([\S\s]*)\"'#
        regex2 = r"(deriv\()*applyCBasis\(Q\(\"([\S\s]*)\"\),[\s]*([-*\d\.]*)"
        regex3 = r"bs\(([\S\s]*), df=([\d]+)\)\[([\d]+)\][:]*([\s\S]*)"
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
        #
        if saveCVBetas:
            estimatorBetasList = cvBetas + [estimator.betas]
        else:
            estimatorBetasList = [estimator.betas]
        covariateFilters = [
            {
                i: pd.DataFrame(index=cBasis.index, columns=[])
                for i in unitNames}
            for esb in estimatorBetasList]
        covariateSplineFilters = [
            {i: {} for i in unitNames}
            for esb in estimatorBetasList]
        anyCovariateSplineFilters = False
        for esbIdx, estimatorBetas in enumerate(estimatorBetasList):
            for cName in estimatorBetas.columns:
                matches = re.search(regex2, cName, re.MULTILINE)
                if matches:
                    covariateName = matches.groups()[1]
                    derivTerm = matches.groups()[0] is not None
                    lag = np.float(matches.groups()[2])
                    splineMatches = re.search(regex3, covariateName, re.MULTILINE)
                    #
                    if not splineMatches:
                        covariateSaveName = covariateName
                        if derivTerm:
                            covariateSaveName = 'deriv({})'.format(covariateSaveName)
                    else:
                        anyCovariateSplineFilters = True
                        mainCovariateName = splineMatches.groups()[0]
                        nDF = splineMatches.groups()[1]
                        splIdx = splineMatches.groups()[2]
                        splName = covariateName.split(':')[0]
                        spl = dummyX[splName]
                        interactName = splineMatches.groups()[3]
                        covariateSaveName = mainCovariateName
                        if derivTerm:
                            covariateSaveName = 'deriv({})'.format(covariateSaveName)
                        if interactName:
                            covariateSaveName = covariateSaveName + ':' + interactName
                    #
                    for targetIdx in estimatorBetas.index:
                        thisCoefficient = estimatorBetas.loc[targetIdx, cName]
                        if not splineMatches:
                            if not (covariateSaveName in covariateFilters[esbIdx][targetIdx]):
                                covariateFilters[esbIdx][targetIdx][covariateSaveName] = 0
                            covariateFilters[esbIdx][targetIdx][covariateSaveName] += (
                                cBasis[lag] * thisCoefficient)
                        else:
                            alreadyStarted = (
                                covariateSaveName in covariateSplineFilters[esbIdx][targetIdx])
                            if not alreadyStarted:
                                covariateSplineFilters[esbIdx][targetIdx][covariateSaveName] = (
                                    pd.DataFrame(
                                        np.zeros((
                                            cBasis.shape[0],
                                            dummyCovariate.shape[0])),
                                        index=cBasis.index,
                                        columns=dummyCovariate[mainCovariateName]))
                            covariateSplineFilters[esbIdx][targetIdx][covariateSaveName] += (
                                np.outer(cBasis[lag], spl) * thisCoefficient)
                            # spline
        #
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
        estimator.yHat.to_hdf(estimatorDiagnosticsPath, '/yHat')
        pR2.to_hdf(estimatorDiagnosticsPath, '/pR2')
        estimatorParams.to_hdf(estimatorDiagnosticsPath, '/estimatorParams')
        pR2smooth.to_hdf(estimatorDiagnosticsPath, '/pR2smooth')
        toleranceParams.to_hdf(estimatorDiagnosticsPath, '/toleranceParams')
        if saveCVBetas:
            for eIdx, estB in enumerate(cvBetas):
                estB.to_hdf(
                    estimatorDiagnosticsPath,
                    '/cvBetas/cv{}'.format(eIdx))
        covariateFiltersMedianDF.to_hdf(
            estimatorDiagnosticsPath, '/covariateFilters/temporal/median')
        covariateFiltersHBDF.to_hdf(
            estimatorDiagnosticsPath, '/covariateFilters/temporal/hb')
        covariateFiltersLBDF.to_hdf(
            estimatorDiagnosticsPath, '/covariateFilters/temporal/lb')
        # pdb.set_trace()
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
    #
    kinNames = [i for i in covariateFiltersMagnitude.columns if '_Right' in i]
    kinBaseNames = pd.unique([i.split('_Right')[0] for i in kinNames])
    covariateFiltersReducedMag = pd.DataFrame(
        np.nan, index=covariateFiltersMagnitude.index, columns=kinBaseNames)
    for kbn in kinBaseNames:
        kbnMask = covariateFiltersMagnitude.columns.str.contains(kbn + '_Right')
        covariateFiltersReducedMag.loc[:, kbn] = covariateFiltersMagnitude.loc[:, kbnMask].sum(axis=1)
    stimNames = [i for i in covariateFiltersMagnitude.columns if '_Right' not in i]
    stimBaseNames = pd.unique([i.split('_amplitude')[0][-8:] for i in stimNames])
    for sbn in stimBaseNames:
        sbnMask = covariateFiltersMagnitude.columns.str.contains(sbn)
        covariateFiltersReducedMag.loc[:, sbn] = covariateFiltersMagnitude.loc[:, sbnMask].sum(axis=1)
    covariateCorrelations = pd.DataFrame(np.nan, index=kinNames, columns=stimNames)
    for pairNames in product(kinNames, stimNames):
        covariateCorrelations.loc[pairNames] = np.corrcoef(
            covariateFiltersMagnitude[pairNames[0]].to_numpy(),
            covariateFiltersMagnitude[pairNames[1]].to_numpy())[0, 1]
    covariateReducedCorrelations = pd.DataFrame(np.nan, index=kinBaseNames, columns=stimBaseNames)
    for pairNames in product(kinBaseNames, stimBaseNames):
        try:
            x = covariateFiltersReducedMag[pairNames[0]].to_numpy()
            y = covariateFiltersReducedMag[pairNames[1]].to_numpy()
            coeffs = np.polyfit(x, y, 1)
            covariateReducedCorrelations.loc[pairNames] = np.corrcoef(
                x, y)[0, 1]
        except:
            continue
    allY = pd.concat(
            {
                'test_data': estimator.yTest,
                'test_prediction': estimator.yHat},
            names=['regressionDataType'])
    #
    dataQuery = '&'.join([
        # '((RateInHz==100)|(amplitude==0))',
        # '(amplitude==0)',
        '(pedalSizeCat=="M")'
        ])
    #
    #######################################################
    variantInfo.loc[variantIdx, 'median_valid'] = pR2['valid'].median()
    variantInfo.loc[variantIdx, 'median_gs'] = pR2['gs'].median()
    saveR2[variantName] = pR2
    saveEstimatorParams[variantName] = estimatorParams
    saveBetas[variantName] = {
        'betas': estimator.betas,
        'significantBetas': estimator.significantBetas
        }
    ########################################################
    if arguments['plottingIndividual']:
        cPalettes = sns.color_palette()
        # plot example trace and predictions
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'traces_example_{}.pdf'.format(
                variantName))
        with sns.plotting_context('notebook', font_scale=1):
            try:
                # fig, ax = estimator.plot_xy(smoothY=rollingWindow)
                # fig, ax = estimator.plot_xy(maxPR2=0.025)
                fig, ax = estimator.plot_xy(
                    # unitName='elec10_2',
                    smoothY=5,
                    binInterval=rasterOpts['binInterval'],
                    decimated=glmOptsLookup[arguments['estimatorName']]['decimate'],
                    winSize=glmOptsLookup[arguments['estimatorName']]['rollingWindow'],
                    selT=slice(1900, 2500),
                    useInvLink=True)
                # for cAx in ax:
                #     cAx.set_xlim([100, 2000])
                fig.savefig(
                    pdfPath,
                    # bbox_extra_artists=(i.get_legend() for i in ax),
                    # bbox_inches='tight'
                    )
                fig.savefig(
                    pdfPath.replace('.pdf', '.png'),
                    # bbox_extra_artists=(i.get_legend() for i in ax),
                    # bbox_inches='tight'
                    )
                if showNow:
                    plt.show()
                else:
                    plt.close()
            except Exception:
                traceback.print_exc()
        # plot example trace and predictions
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'rsq_distribution_{}.pdf'.format(
                variantName))
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        sns.distplot(pR2['valid'], ax=ax[0])
        ax[0].set_title('pseudoR^2 (median {:.3f})'.format(pR2['valid'].median()))
        ax[0].set_ylabel('Count')
        ax[0].set_xlabel('pseudoR^2')
        # ax[0].xaxis.set_major_formatter(ticker.EngFormatter(unit='', places=3))
        ax[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        # ax[0].set_xlim([-0.025, .15])
        sns.boxplot(data=pR2['valid'], ax=ax[1], orient="h")
        ax[1].set_xlabel('pseudoR^2')
        ax[1].set_xlim([-0.025, .15])
        ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
        #
        fig, ax = plt.subplots(figsize=(9, 6))
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'lambda_distribution_{}.pdf'.format(
                variantName))
        sns.countplot(estimatorParams['reg_lambda'], ax=ax)
        ax.set_title('lambda (median {0:.2e})'.format(estimatorParams['reg_lambda'].median()))
        ax.set_ylabel('Count')
        ax.set_xlabel('lambda')
        ax.set_xticklabels(['{:.2e}'.format(float(i.get_text())) for i in ax.get_xticklabels()])
        # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2e}'))
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
        #
        nScoreBins = 200
        #
        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'minIter_distribution_{}.pdf'.format(
                variantName))
        fig.set_tight_layout(True)
        for idx, (tol, group) in enumerate(toleranceParams.groupby('tol')):
            labelText = '{}'.format(tol)
            scoreCounts, scoreBins = np.histogram(group['score'], nScoreBins)
            ax[0].plot(scoreBins[:-1], scoreCounts.cumsum() / scoreCounts.sum(), label=labelText)
            if tol != tolList[0]:
                scoreCounts, scoreBins = np.histogram(group['marginalScore'], nScoreBins)
                ax[1].plot(scoreBins[:-1], scoreCounts.cumsum() / scoreCounts.sum(), label=labelText)
            scoreCounts, scoreBins = np.histogram(group['minIter'], nScoreBins)
            ax[2].plot(scoreBins[:-1], scoreCounts.cumsum() / scoreCounts.sum(), label=labelText)
            if idx == 0:
                ax[0].set_xlim((group['score'].quantile(0.05), group['score'].quantile(0.95)))
                ax[1].set_xlim((group['marginalScore'].quantile(0.05), group['marginalScore'].quantile(0.95)))
                ax[2].set_xlim((group['minIter'].quantile(0.05), group['minIter'].quantile(0.95)))
            else:
                ax[0].set_xlim(
                    (
                        min(ax[0].get_xlim()[0], group['score'].quantile(0.05)),
                        max(ax[0].get_xlim()[1], group['score'].quantile(0.95)))
                    )
                ax[1].set_xlim(
                    (
                        min(ax[1].get_xlim()[0], group['marginalScore'].quantile(0.05)),
                        max(ax[1].get_xlim()[1], group['marginalScore'].quantile(0.95)))
                    )
                ax[2].set_xlim(
                    (
                        min(ax[2].get_xlim()[0], group['minIter'].quantile(0.05)),
                        max(ax[2].get_xlim()[1], group['minIter'].quantile(0.95)))
                    )
        ax[0].set_xlabel('Validation Score')
        ax[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        ax[1].set_xlabel(
            'marginal score (vs. tol={0:.2E})'.format(tolList[0]))
        ax[1].set_ylabel('Count')
        ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        ax[2].set_xlabel('Iteration count')
        ax[2].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        plt.legend()
        #
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
        #
        fig, ax = plt.subplots(2, 1, sharex=True)
        #
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'rsq_change_with_binSize_{}.pdf'.format(
                variantName))
        for rIdx, thisRollingWindow in enumerate(rollingWindowList):
            rMask = pR2smooth['rollingWindow'] == thisRollingWindow
            lblText = '{} sample average; (median pR2 {:.3f})'.format(
                thisRollingWindow, pR2smooth.loc[rMask, 'pR2'].median())
            sns.distplot(
                pR2smooth.loc[rMask, 'pR2'], ax=ax[0],
                label=lblText, color=cPalettes[rIdx])
        #
        sns.boxplot(
            x='pR2', y='rollingWindow',
            data=pR2smooth,
            ax=ax[1], orient="h", palette=cPalettes)
        ax[0].legend()
        ax[0].set_ylabel('Count')
        # ax[0].set_xlabel('pseudoR^2')
        # ax[0].set_xlim([-0.025, .15])
        ax[1].set_xlabel('pseudoR^2')
        ax[1].set_xlim([-0.025, .15])
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
        #
        if variantInfo.loc[variantIdx, 'kinAndStim']:
            pdfPath = os.path.join(
                estimatorFiguresFolder,
                'corr_between_beta_{}.pdf'.format(
                    variantName))
            plotKinBaseNames = [i for i in kinBaseNames if covariateReducedCorrelations.loc[i, :].notna().any()]
            plotStimBaseNames = [j for j in stimBaseNames if covariateReducedCorrelations.loc[:, j].notna().any()]
            with PdfPages(pdfPath) as pdf:
                sns.heatmap(
                    covariateReducedCorrelations.loc[plotKinBaseNames, plotStimBaseNames],
                    cmap="cubehelix")
                plt.suptitle('Correlation coefficients')
                pdf.savefig()
                sns.pairplot(
                    covariateFiltersReducedMag,
                    x_vars=plotKinBaseNames, y_vars=plotStimBaseNames)
                plt.suptitle('Model parameter magnitudes')
                pdf.savefig()
                if showNow:
                    plt.show()
                else:
                    plt.close()
        if arguments['plottingCovariateFilters']:
            pdfPath = os.path.join(
                estimatorFiguresFolder,
                'covariate_filters_{}.pdf'.format(variantName))
            with PdfPages(pdfPath) as pdf:
                for uIdx, unitName in enumerate(unitNames):
                    filterGroupMedian = covariateFiltersMedianDF.loc[unitName, :].unstack(level='covariate')
                    filterGroupHB = covariateFiltersHBDF.loc[unitName, :].unstack(level='covariate')
                    filterGroupLB = covariateFiltersLBDF.loc[unitName, :].unstack(level='covariate')
                    print('Printing {} covariate filters'.format(unitName))
                    covPalette = sns.color_palette("Set1", n_colors=25)
                    if not filterGroupMedian.empty:
                        fig = plt.figure(figsize=(12, 9))
                        gs = gridspec.GridSpec(nrows=1, ncols=1)
                        ax0 = fig.add_subplot(gs[0, 0])
                        for cIdx, cName in enumerate(filterGroupMedian.columns):
                            ax0.fill_between(
                                filterGroupLB.index,
                                filterGroupLB[cName], filterGroupHB[cName],
                                facecolor=covPalette[cIdx], alpha=0.3)
                            ax0.plot(filterGroupMedian[cName], c=covPalette[cIdx], label=cName)
                        ax0.set_title(
                            'cell {} covariate weights, model {}'
                            .format(unitName, variantName))
                        plt.legend()
                        pdf.savefig()
                        if showNow:
                            plt.show()
                        else:
                            plt.close()
                    #
                    if anyCovariateSplineFilters:
                        for cName in splineFilterNames:
                            splFilt = covariateSplineFiltersMedianDF[cName].loc[unitName, :].unstack(level=0)
                            splFiltLB = covariateSplineFiltersLBDF[cName].loc[unitName, :].unstack(level=0)
                            splFiltHB = covariateSplineFiltersHBDF[cName].loc[unitName, :].unstack(level=0)
                            splFilt.index = ['{:.2f}'.format(i) for i in splFilt.index]
                            splFilt.columns = ['{:.1f}'.format(i) for i in splFilt.columns]
                            splFiltLB.index = ['{:.2f}'.format(i) for i in splFiltLB.index]
                            splFiltLB.columns = ['{:.1f}'.format(i) for i in splFiltLB.columns]
                            splFiltHB.index = ['{:.2f}'.format(i) for i in splFiltHB.index]
                            splFiltHB.columns = ['{:.1f}'.format(i) for i in splFiltHB.columns]
                            fig = plt.figure(figsize=(24, 9))
                            gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1])
                            vmin = splFiltLB.min().min()
                            vmax = splFiltHB.max().max()
                            ax0 = fig.add_subplot(gs[0, 0])
                            sns.heatmap(
                                splFiltLB, ax=ax0, center=0, vmin=vmin, vmax=vmax, cbar=False, cmap="coolwarm",
                                xticklabels=10, yticklabels=5, edgecolors='face')
                            ax0.set_title('lower bound')
                            ax1 = fig.add_subplot(gs[0, 1])
                            sns.heatmap(
                                splFilt, ax=ax1, center=0, vmin=vmin, vmax=vmax, cbar=False,  cmap="coolwarm",
                                xticklabels=10, yticklabels=5, edgecolors='face')
                            ax1.set_title('median')
                            ax2 = fig.add_subplot(gs[0, 2])
                            sns.heatmap(
                                splFiltHB, ax=ax2, center=0, vmin=vmin, vmax=vmax, cbar=True,  cmap="coolwarm",
                                xticklabels=10, yticklabels=5, edgecolors='face')
                            ax2.set_title('higher bound')
                            plt.suptitle(
                                'cell {}, {} weights, model {}'
                                .format(unitName, cName, variantName))
                            ax0.set_ylabel('Lag (sec)')
                            ax1.set_xlabel('Amplitude (a.u.)')
                            pdf.savefig()
                            if showNow:
                                plt.show()
                            else:
                                plt.close()
        #
        if arguments['makePredictionPDF']:
            print('Rolling window is {}'.format(rollingWindow))
            allBins = allY.query(dataQuery).index.get_level_values('bin')
            plotBins = allBins.unique()[int(rollingWindow/2)::rollingWindow]
            plotMask = allBins.isin(plotBins)
            plotData = (
                allY.query(dataQuery)
                .rolling(rollingWindow, center=True).mean()
                .iloc[plotMask, :]
                .dropna()
                .reset_index())
            pdfPath = os.path.join(
                estimatorFiguresFolder,
                'prediction_examples_{}.pdf'.format(
                    variantName))
            relplotKWArgs.update({
                'palette': "ch:1.6,-.3,dark=.1,light=0.7,reverse=1",
                })
            with PdfPages(pdfPath) as pdf:
                for uIdx, unitName in enumerate(unitNames):
                    thisReg = estimator.regressionList[unitName]
                    print('Plotting {}'.format(unitName))
                    g = sns.relplot(
                        x='bin', y=unitName,
                        data=plotData,
                        style='regressionDataType',
                        hue='amplitude',
                        col='electrode', **relplotKWArgs)
                    axMax = plotData.query('amplitude==0')[unitName].quantile(0.95)
                    g.axes.flatten()[0].set_ylim([0, axMax])
                    plt.suptitle(
                        '{}: pR^2 = {:.3f} ({} model)'.format(
                            unitName,
                            thisReg['validationScore'],
                            variantInfo.loc[variantIdx, 'terms']))
                    pdf.savefig()
                    plt.close()
                    if arguments['debugging']:
                        if uIdx > 15:
                            break
    
    #####################################################################
LLSat = pd.Series(np.nan, index=unitNames)
for unitName, thisReg in estimator.regressionList.items():
    if hasattr(thisReg['reg'], 'get_params'):
        if 'distr' in thisReg['reg'].get_params():
            LLSat[unitName] = pyglmnet._logL(
                thisReg['reg'].distr,
                estimator.yTest[unitName], estimator.yTest[unitName])
# plots to compare between models (saveR2 saveEstimatorParams saveBetas)

allR2 = pd.concat(saveR2, names=['variant', 'target']).stack().reset_index()
allR2.columns = ['variant', 'target', 'grp', 'pR2']
mappingTargets = [
    'terms', 'stimAmpSpline', 'stimLagStr',
    'kinLagStr', 'anyKin', 'anyStim', 'kinAndStim',
    'kinOnly', 'stimOnly']
for mappingTarget in mappingTargets:
    mapping = dict(variantInfo[['variantName', mappingTarget]].values)
    allR2[mappingTarget] = allR2['variant'].map(mapping)
allR2['kinOnly'] = allR2['terms'] == ('kva')
allR2['stimOnly'] = allR2['terms'] == ('sd')
allR2['kinAndStim'] = allR2['terms'] == ('kvasd')
allR2['gs_split_all'] = allR2['grp'].str.contains('split')
allR2['splinedStim'] = allR2['stimAmpSpline'].str.contains('bs')
#
stimLagR2 = allR2.query('(gs_split_all) and (stimOnly) and (splinedStim)')
stimLagVarList = stimLagR2['variant'].unique()
#
stimLagPairedT = pd.DataFrame(
    np.nan, index=stimLagVarList, columns=stimLagVarList)
stimLagPairedPval = pd.DataFrame(
    np.nan, index=stimLagVarList, columns=stimLagVarList)
for pairName in (combinations(stimLagVarList,2)):
    x = stimLagR2.query('variant == "{}"'.format(pairName[0]))
    y = stimLagR2.query('variant == "{}"'.format(pairName[1]))
    T, pv = stats.wilcoxon(x['pR2'], y['pR2'], zero_method='pratt', correction=True)
    stimLagPairedT.loc[pairName] = T
    stimLagPairedPval.loc[pairName] = pv
#
kinLagR2 = allR2.query('(gs_split_all) and (kinOnly)')
kinLagVarList = kinLagR2['variant'].unique()
kinLagPairedT = pd.DataFrame(
    np.nan, index=kinLagVarList, columns=kinLagVarList)
kinLagPairedPval = pd.DataFrame(
    np.nan, index=kinLagVarList, columns=kinLagVarList)
for pairName in (combinations(kinLagVarList,2)):
    x = kinLagR2.query('variant == "{}"'.format(pairName[0]))
    y = kinLagR2.query('variant == "{}"'.format(pairName[1]))
    # all(x['grp'].values == y['grp'].values)
    T, pv = stats.wilcoxon(x['pR2'], y['pR2'], zero_method='pratt', correction=True)
    kinLagPairedT.loc[pairName] = T
    kinLagPairedPval.loc[pairName] = pv
#
lagR2 = allR2.query('(gs_split_all) and (kinAndStim)')
lagVarList = lagR2['variant'].unique()
lagPairedT = pd.DataFrame(
    np.nan, index=lagVarList, columns=lagVarList)
lagPairedPval = pd.DataFrame(
    np.nan, index=lagVarList, columns=lagVarList)
for pairName in (combinations(lagVarList,2)):
    x = lagR2.query('variant == "{}"'.format(pairName[0]))
    y = lagR2.query('variant == "{}"'.format(pairName[1]))
    # all(x['grp'].values == y['grp'].values)
    T, pv = stats.wilcoxon(x['pR2'], y['pR2'], zero_method='pratt', correction=True)
    lagPairedT.loc[pairName] = T
    lagPairedPval.loc[pairName] = pv
#
if arguments['plottingOverall']:
    pdfPath = os.path.join(
        estimatorFiguresFolder,
        'dynamic_and_nonlinear.pdf')
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=3, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        y='kinLagStr', x='pR2', hue='grp',
        data=allR2.query('((grp=="gs") or (grp=="valid")) and (kinOnly)'),
        ax=ax0, orient='h')
    ax0.set_xlim([lagR2['pR2'].quantile(0.05), lagR2['pR2'].quantile(0.95)])
    ax1 = fig.add_subplot(gs[2, 0])
    sns.boxplot(
        y='stimLagStr', x='pR2', hue='stimAmpSpline',
        data=stimLagR2,
        ax=ax1, orient='h', palette="Set2")
    ax1.set_xlim([lagR2['pR2'].quantile(0.05), lagR2['pR2'].quantile(0.95)])
    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        y='stimLagStr', x='pR2', hue='stimAmpSpline',
        data=lagR2,
        ax=ax2, orient='h', palette="Set3")
    ax2.set_xlim([lagR2['pR2'].quantile(0.05), lagR2['pR2'].quantile(0.95)])
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    if showNow:
        plt.show()
    else:
        plt.close()

allBeta = pd.concat(
    {k: v['betas'] for k, v in saveBetas.items()},
    names=['variant', 'target']).stack().reset_index()

bestVariants = pd.DataFrame(index=[], columns=variantInfo.columns)
for name, group in variantInfo.groupby('terms'):
    maxVarIdx = group['median_valid'].idxmax()
    bestVariants.loc[maxVarIdx, :] = variantInfo.loc[maxVarIdx, :]
bestValidation = pd.concat({
    i: saveR2[bestVariants.query("terms == '{}'".format(i))['variantName'].iloc[0]]['valid']
    for i in bestVariants['terms'].to_list()},
    axis=1, sort=True)
bestValidationLL = pd.concat({
    i: saveR2[bestVariants.query("terms == '{}'".format(i))['variantName'].iloc[0]]['valid_LL']
    for i in bestVariants['terms'].to_list()},
    axis=1, sort=True)
bestBetasSignificance = pd.concat({
    i: saveBetas[bestVariants.query("terms == '{}'".format(i))['variantName'].iloc[0]]['significantBetas']
    for i in bestVariants['terms'].to_list()},
    axis=0, sort=True, names=['terms', 'target']).fillna(False)
bestBetas = pd.concat({
    i: saveBetas[bestVariants.query("terms == '{}'".format(i))['variantName'].iloc[0]]['betas']
    for i in bestVariants['terms'].to_list()},
    axis=0, sort=True, names=['terms', 'target'])

# maxOverallVariant = variantInfo['median_valid'].idxmax()
# mostModulated = saveR2[maxOverallVariant].sort_values('valid').index.to_list()
# print(mostModulated)
# mostModSelector = {
#     'outputFeatures': sorted([i for i in mostModulated[-64:]])
#     }
# modSelectorPath = os.path.join(
#     alignSubFolder, '{}_maxmod.pickle'.format(fullEstimatorName))
# print(modSelectorPath)
# with open(modSelectorPath, 'wb') as f:
#     pickle.dump(mostModSelector, f)


def plotPR2Diff(p1, p2, ll1, ll2, lls, color=None):
    deltaPR2 = p2 - p1
    FUDE = 1 - (lls - ll2) / (lls - ll1)
    print(FUDE.sort_values(ascending=False).index)
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    sns.scatterplot(x=p1, y=p2, ax=ax0, color=color)
    ax0.plot([0, 1], [0, 1], 'k--')
    maxP = max(p1.max(), p2.max())
    ax0.set_xlim((0, maxP))
    ax0.set_ylim((0, maxP))
    ax0.set_xlabel(p1.name)
    ax0.set_ylabel(p2.name)
    ax0.set_title(
        'pR^2 of {} vs {} (median difference {:0.3f})'
        .format(p1.name, p2.name, deltaPR2.median()))
    ax = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1])
        ]
    sns.distplot(FUDE, ax=ax[0], color=color)
    ax[0].set_ylabel('Count')
    ax[0].set_title('FUDE (median {:0.3f})'.format(FUDE.median()))
    # ax[0].set_xlim([-0.025, .15])
    sns.boxplot(data=FUDE, ax=ax[1], orient="h", color=color)
    ax[1].set_xlabel('FUDE')
    # ax[1].set_xlim([-0.025, .15])
    pdfPath = os.path.join(
        estimatorFiguresFolder,
        'rsq_diff_{}_vs_{}.pdf'.format(p1.name, p2.name))
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    if showNow:
        plt.show()
    else:
        plt.close()


if arguments['plottingOverall']:
    try:
        plotPR2Diff(
            bestValidation['kva'], bestValidation['kvasd'],
            bestValidationLL['kva'], bestValidationLL['kvasd'], LLSat, color='b')
    except Exception:
        pass
    try:
        plotPR2Diff(
            bestValidation['sd'], bestValidation['kvasd'],
            bestValidationLL['sd'], bestValidationLL['kvasd'], LLSat, color='r')
    except Exception:
        pass
