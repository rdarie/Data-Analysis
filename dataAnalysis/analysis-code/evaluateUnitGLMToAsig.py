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
import os, glob
from scipy import stats
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
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices)
from dataAnalysis.custom_transformers.tdr import *
from dataAnalysis.custom_transformers.tdr import _poisson_pseudoR2, _FUDE
#
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
    estimatorSubFolder, 'variant_metadata.pickle')
with open(variantMetaDataPath, 'rb') as f:
    allModelOpts = pickle.load(f)
featuresMetaDataPath = os.path.join(
    estimatorSubFolder, 'features_meta.pickle')
dummyEstimatorMetadataPath = os.path.join(
   estimatorSubFolder, 'var_001', 'estimator_metadata.pickle')
with open(dummyEstimatorMetadataPath, 'rb') as f:
    dummyEstimatorMetadata = pickle.load(f)
featuresMetaDataPath = os.path.join(
    estimatorSubFolder, 'features_meta.pickle')
with open(featuresMetaDataPath, 'rb') as f:
    featuresMetaData = pickle.load(f)
    ihbasis = featuresMetaData['ihbasis']
    iht = featuresMetaData['iht']
#
targetH5Path = dummyEstimatorMetadata['targetPath']
targetDF = pd.read_hdf(targetH5Path, 'target')
regressorH5Path = targetH5Path.replace(
    '_raster', '_rig')
featuresDF = pd.read_hdf(regressorH5Path, 'feature')
saveR2 = {}
saveEstimatorParams = {}
saveBetas = {}
varFolders = sorted(glob.glob(os.path.join(alignSubFolder, fullEstimatorName, 'var_*')))
variantList = [os.path.basename(i) for i in varFolders]
# variantList = range(15)
if arguments['debugging']:
    showNow = False
    # variantList = ['bak/var_000']
    variantList = ['var_{:03d}'.format(i) for i in range(55)]
else:
    showNow = False


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


variantInfo = pd.DataFrame(
    np.nan,
    index=variantList,
    columns=[
        'kinLag', 'terms',
        'stimLag', 'singleLag', 'equalLag'])


for variantName in variantList:
    variantIdx = int(variantName[-3:])
    modelOpts = allModelOpts[variantIdx]
    singleLag = (
        len(modelOpts['kinLag']) == 1 and
        len(modelOpts['stimLag']) == 1)
    variantInfo.loc[variantName, 'singleLag'] = singleLag
    if singleLag:
        variantInfo.loc[variantName, 'stimLag'] = modelOpts['stimLag'][0]
        variantInfo.loc[variantName, 'kinLag'] = modelOpts['stimLag'][0]
    equalLag = modelOpts['kinLag'] == modelOpts['stimLag']
    variantInfo.loc[variantName, 'equalLag'] = equalLag
    terms = ''
    if modelOpts['ensembleHistoryTerms']:
        terms += 'e'
    if modelOpts['kinematicTerms']:
        terms += 'k'
    if modelOpts['stimTerms']:
        terms += 's'
    variantInfo.loc[variantName, 'terms'] = terms
    #
    estimatorMetadataPath = os.path.join(
       alignSubFolder,
       fullEstimatorName, variantName, 'estimator_metadata.pickle')
    with open(estimatorMetadataPath, 'rb') as f:
        estimatorMetadata = pickle.load(f)
    
    cBasis = estimatorMetadata['cBasis']
    def applyCBasis(x, cbCol):
        return np.convolve(
            x.to_numpy(),
            cBasis[cbCol].to_numpy(),
            mode='same')

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
                estimator.betas = pd.concat([estimator.betas, newEstimator.betas])
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
    #
    descStr = estimatorMetadata['modelDescStr']
    desc = ModelDesc.from_formula(descStr)
    train = estimatorMetadata['trainIdx']
    test = estimatorMetadata['testIdx']
    try:
        estimator.xTrain = dmatrix(
            desc, featuresDF.iloc[train, :],
            return_type='dataframe')
        estimator.xTest = dmatrix(
            desc, featuresDF.iloc[test, :],
            return_type='dataframe')
        estimator.yTrain = targetDF.iloc[train, :]
        estimator.yTest = targetDF.iloc[test, :]
    except Exception:
        traceback.print_exc()
    for idx, unitName in enumerate(estimator.regressionList.keys()):
        reg = estimator.regressionList[unitName]['reg']
        if hasattr(reg, 'model_'):
            try:
                reg.model_.endog = estimator.xTrain
                reg.model_.exog = estimator.yTrain.loc[:, unitName]
            except Exception:
                traceback.print_exc()
    ######################################################################
    # trialInfo = estimator.yTest.index.to_frame(index=False)
    # nBins = trialInfo['bin'].unique().size
    # nTrials = trialInfo.groupby([
    #     'RateInHz', 'amplitude',
    #     'electrode', 'pedalSizeCat'])['bin'].count() / nBins
    # print('Test set contains:\n{}'.format(nTrials))
    countPerBin = (
        rasterOpts['binInterval'] *
        glmOptsLookup[arguments['estimatorName']]['decimate'])
    rollingWindow = int(60e-3 / countPerBin)
    if arguments['plottingIndividual']:
        cPalettes = sns.color_palette()
        #
        with sns.plotting_context('notebook', font_scale=1):
            try:
                fig, ax = estimator.plot_xy(smoothY=rollingWindow)
                # fig, ax = estimator.plot_xy(maxPR2=0.025)
                # fig, ax = estimator.plot_xy(unitName='elec75#1_p000', smoothY=10)
                for cAx in ax:
                    cAx.set_xlim([100, 2000])
                pdfPath = os.path.join(
                    estimatorFiguresFolder,
                    'traces_example_{}.pdf'.format(
                        variantName))
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
    #
    pR2 = pd.DataFrame(
        np.nan,
        index=estimator.regressionList.keys(),
        columns=['gs', 'cv', 'valid', 'valid_LL'])
    estimatorParams = pd.DataFrame(
        np.nan,
        index=estimator.regressionList.keys(),
        columns=['minIter', 'reg_lambda'])
    toleranceParamsList = []
    tolList = [1e-2, 1e-4, 1e-6]
    for unit, regDict in estimator.regressionList.items():
        if 'gridsearch_best_mean_test_score' in regDict:
            pR2.loc[unit, 'gs'] = regDict['gridsearch_best_mean_test_score']
        if 'cross_val_mean_test_score' in regDict:
            pR2.loc[unit, 'cv'] = regDict['cross_val_mean_test_score']
        if 'validationScore' in regDict:
            pR2.loc[unit, 'valid'] = regDict['validationScore']
        if hasattr(regDict['reg'], 'reg_lambda'):
            estimatorParams.loc[unit, 'reg_lambda'] = (
                regDict['reg'].reg_lambda
            )
        if 'gridsearch' in regDict:
            gs = regDict['gridsearch']
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
            estimatorParams.loc[unit, 'minIter'] = termIndices[-1]
    #
    toleranceParams = pd.concat(toleranceParamsList, ignore_index=True)
    #######################################################
    variantInfo.loc[variantName, 'median_valid'] = pR2['valid'].median()
    variantInfo.loc[variantName, 'median_gs'] = pR2['gs'].median()
    saveR2[variantName] = pR2
    saveEstimatorParams[variantName] = estimatorParams
    saveBetas[variantName] = {
        'betas': estimator.betas,
        'significantBetas': estimator.significantBetas
        }
    ########################################################
    if arguments['plottingIndividual']:
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
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'rsq_distribution_{}.pdf'.format(
                variantName))
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
        #
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.countplot(estimatorParams['reg_lambda'], ax=ax)
        ax.set_title('lambda (median {0:.2e})'.format(estimatorParams['reg_lambda'].median()))
        ax.set_ylabel('Count')
        ax.set_xlabel('lambda')
        ax.set_xticklabels(['{:.2e}'.format(float(i.get_text())) for i in ax.get_xticklabels()])
        # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2e}'))
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'lambda_distribution_{}.pdf'.format(
                variantName))
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
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'minIter_distribution_{}.pdf'.format(
                variantName))
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
    # def plot_average_xy
    estimator.yHat = pd.DataFrame(
        np.nan,
        index=estimator.yTest.index,
        columns=estimator.yTest.columns)
    #
    for unitName, thisReg in estimator.regressionList.items():
        estimator.yHat.loc[:, unitName] = (
            thisReg['reg'].predict(estimator.xTest))
        if hasattr(thisReg['reg'], 'get_params'):
            if 'distr' in thisReg['reg'].get_params():
                pR2.loc[unitName, 'valid_LL'] = pyglmnet._logL(
                    thisReg['reg'].distr,
                    estimator.yTest[unitName], estimator.yHat[unitName])
                pyglmnet._logL(
                    thisReg['reg'].distr,
                    estimator.yTest[unitName], estimator.yTest[unitName].mean())
    #
    ########################################################################
    rollingWindowList = [
        int(i / countPerBin)
        for i in [50e-3, 100e-3, 200e-3]
        ]
    pR2smoothDict = {
        i: pd.DataFrame(
            np.nan,
            index=estimator.regressionList.keys(),
            columns=['pR2'])
        for i in rollingWindowList
        }
    for uIdx, unitName in enumerate(sorted(estimator.regressionList.keys())):
        for rIdx, thisRollingWindow in enumerate(rollingWindowList):
            pR2smoothDict[thisRollingWindow].loc[unitName, 'pR2'] = _poisson_pseudoR2(
                estimator.yTest[unitName]
                .rolling(thisRollingWindow, center=True).mean()
                .dropna().iloc[int(thisRollingWindow/2)::thisRollingWindow]
                .to_numpy(),
                estimator.yHat[unitName]
                .rolling(thisRollingWindow, center=True).mean()
                .dropna().iloc[int(thisRollingWindow/2)::thisRollingWindow]
                .to_numpy()
                )
    #
    pR2smooth = pd.concat(pR2smoothDict, names=['rollingWindow']).reset_index()
    #
    if arguments['plottingIndividual']:
        fig, ax = plt.subplots(2, 1, sharex=True)
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
        #
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'rsq_distribution_binSize_{}.pdf'.format(
                variantName))
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if showNow:
            plt.show()
        else:
            plt.close()
    #
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
    if arguments['makePredictionPDF'] and arguments['plottingIndividual']:
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
        with PdfPages(pdfPath) as pdf:
            for uIdx, unitName in enumerate(pR2.sort_values('valid').index.to_list()):
                thisReg = estimator.regressionList[unitName]
                print('Plotting {}'.format(unitName))
                g = sns.relplot(
                    x='bin', y=unitName,
                    data=plotData,
                    style='regressionDataType',
                    hue='amplitude',
                    col='electrode', ci='sem',
                    kind='line')
                axMax = plotData.query('amplitude==0')[unitName].quantile(0.95)
                g.axes.flatten()[0].set_ylim([0, axMax])
                plt.suptitle(
                    '{}: pR^2 = {:.3f} ({} model)'.format(
                        unitName,
                        thisReg['validationScore'],
                        variantInfo.loc[variantName, 'terms']))
                pdf.savefig()
                plt.close()
                #  if arguments['debugging']:
                #      if uIdx > 3:
                #          break
    #####################################################################
LLSat = pd.Series(np.nan, index=estimator.regressionList.keys())
for unitName, thisReg in estimator.regressionList.items():
    if hasattr(thisReg['reg'], 'get_params'):
        if 'distr' in thisReg['reg'].get_params():
            LLSat[unitName] = pyglmnet._logL(
                thisReg['reg'].distr,
                estimator.yTest[unitName], estimator.yTest[unitName])
# plots to compare between models (saveR2 saveEstimatorParams saveBetas)
print(variantInfo)
csvPath = os.path.join(
    estimatorFiguresFolder,
    'variant_info.csv')
variantInfo.to_csv(csvPath)
#
bestVariants = pd.DataFrame(index=[], columns=variantInfo.columns)
for name, group in variantInfo.groupby('terms'):
    maxVarIdx = group['median_valid'].idxmax()
    bestVariants.loc[maxVarIdx, :] = variantInfo.loc[maxVarIdx, :]
bestValidation = pd.concat({
    i: saveR2[bestVariants.query("terms == '{}'".format(i)).index[0]]['valid']
    for i in bestVariants['terms'].to_list()},
    axis=1, sort=True)
bestValidationLL = pd.concat({
    i: saveR2[bestVariants.query("terms == '{}'".format(i)).index[0]]['valid_LL']
    for i in bestVariants['terms'].to_list()},
    axis=1, sort=True)
bestBetasSignificance = pd.concat({
    i: saveBetas[bestVariants.query("terms == '{}'".format(i)).index[0]]['significantBetas']
    for i in bestVariants['terms'].to_list()},
    axis=0, sort=True, names=['terms', 'target']).fillna(False)
bestBetas = pd.concat({
    i: saveBetas[bestVariants.query("terms == '{}'".format(i)).index[0]]['betas']
    for i in bestVariants['terms'].to_list()},
    axis=0, sort=True, names=['terms', 'target'])

maxOverallVariant = variantInfo['median_valid'].idxmax()
mostModulated = saveR2[maxOverallVariant].sort_values('valid').index.to_list()
print(mostModulated)
mostModSelector = {
    'outputFeatures': sorted([i for i in mostModulated[-64:]])
    }
modSelectorPath = os.path.join(
    alignSubFolder, '{}_maxmod.pickle'.format(fullEstimatorName))
print(modSelectorPath)
with open(modSelectorPath, 'wb') as f:
    pickle.dump(mostModSelector, f)


def plotPR2Diff(p1, p2, ll1, ll2, lls):
    deltaPR2 = p2 - p1
    FUDE = 1 - (lls - ll2) / (lls - ll1)
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    sns.scatterplot(x=p1, y=p2, ax=ax0)
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
    sns.distplot(FUDE, ax=ax[0])
    ax[0].set_ylabel('Count')
    ax[0].set_title('FUDE (median {:0.3f})'.format(FUDE.median()))
    # ax[0].set_xlim([-0.025, .15])
    sns.boxplot(data=FUDE, ax=ax[1], orient="h")
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
            bestValidation['ek'], bestValidation['eks'],
            bestValidationLL['ek'], bestValidationLL['eks'], LLSat)
    except Exception:
        pass
    try:
        plotPR2Diff(
            bestValidation['es'], bestValidation['eks'],
            bestValidationLL['es'], bestValidationLL['eks'], LLSat)
    except Exception:
        pass
    try:
        plotPR2Diff(
            bestValidation['k'], bestValidation['ks'],
            bestValidationLL['k'], bestValidationLL['ks'], LLSat)
    except Exception:
        pass
    try:
        plotPR2Diff(
            bestValidation['s'], bestValidation['ks'],
            bestValidationLL['s'], bestValidationLL['ks'], LLSat)
    except Exception:
        pass

unitNames = bestBetas.index.get_level_values('target').unique()

allUnitFiltersDict = {}

for name, betaGroup in bestBetas.iterrows():
    if 'e' in name[0]:
        unitFilters = pd.DataFrame(
            index=iht,
            columns=unitNames
            )
        for unitName in unitNames:
            unitCoeffMask = betaGroup.index.str.contains(unitName.replace('#', '_'))
            unitCoeffs = betaGroup.iloc[unitCoeffMask]
            unitFilters.loc[:, unitName] = np.dot(ihbasis, unitCoeffs)
        allUnitFiltersDict[name] = unitFilters
allUnitFilters = pd.concat(
    allUnitFiltersDict, axis=1, names=['terms', 'filter', 'target'])
pdfPath = os.path.join(
    estimatorFiguresFolder,
    'ensemble_filters.pdf')
#
if arguments['plottingOverall']:
    with PdfPages(pdfPath) as pdf:
        for name, filterGroup in allUnitFilters.groupby(level=['terms', 'target'], axis=1):
            print('Printing {}'.format(name))
            selfFilters = filterGroup.columns.get_level_values('filter') == name[1]
            fig = plt.figure(figsize=(12, 9))
            gs = gridspec.GridSpec(nrows=2, ncols=1)
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.plot(filterGroup.loc[:, selfFilters], c='r')
            ax0.set_title('cell {}, {} model, self weights'.format(name[1], name[0]))
            ax1 = fig.add_subplot(gs[1, 0])
            ax1.plot(filterGroup.loc[:, ~selfFilters])
            ax1.set_title('ensemble weights')
            ax1.set_xlabel('Time (sec)')
            ax1.set_xlim([0, 0.4])
            pdf.savefig()
            plt.close()
