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
from statsmodels.stats.multitest import multipletests as mt
from patsy import (
    ModelDesc, EvalEnvironment, Term, Sum, Treatment, INTERCEPT,
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices)
from dataAnalysis.custom_transformers.tdr import _poisson_pseudoR2
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
featuresMetaDataPath = os.path.join(
    alignSubFolder,
    fullEstimatorName, 'features_meta.pickle')
# if arguments['debugging']:
#     featuresMetaDataPath = os.path.join(
#         alignSubFolder,
#         fullEstimatorName, 'bak', 'features_meta.pickle')
dummyEstimatorMetadataPath = os.path.join(
   alignSubFolder,
   fullEstimatorName, 'var_000', 'estimator_metadata.pickle')
with open(dummyEstimatorMetadataPath, 'rb') as f:
    dummyEstimatorMetadata = pickle.load(f)
#
targetH5Path = dummyEstimatorMetadata['targetPath']
# if arguments['debugging']:
#     targetH5Path = os.path.join(
#         alignSubFolder, fullEstimatorName, 'bak', '_raster_long.h5')
targetDF = pd.read_hdf(targetH5Path, 'target')
regressorH5Path = targetH5Path.replace(
    '_raster', '_rig')
featuresDF = pd.read_hdf(regressorH5Path, 'feature')
# confirm targetDF
if False:
    exampleUnitName = 'elec75#0_p000'
    averageTarget = targetDF[exampleUnitName].unstack(level='bin')
    plt.plot(averageTarget.mean().to_numpy()); plt.show()
saveR2 = {}
saveEstimatorParams = {}
varFolders = sorted(glob.glob(os.path.join(alignSubFolder, fullEstimatorName, 'var_*')))
variantList = [os.path.basename(i) for i in varFolders]
# variantList = range(15)
if arguments['debugging']:
    showNow = True
    # variantList = ['bak/var_000']
    variantList = ['var_000']
else:
    showNow = False


def calcMinIter(iterBetaNorm, tol_list=[1e-6], plotting=False):
    if isinstance(iterBetaNorm, pd.Series):
        ibn = iterBetaNorm
    else:
        ibn = pd.Series(iterBetaNorm)
    updateSize = ibn.diff().abs() / ibn.shift(1)
    terminateIndices = []
    if plotting:
        fig, ax = plt.subplots()
        ax.plot(ibn)
    for tol in tol_list:
        tolMask = updateSize >= tol
        if (~tolMask).any():
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
    variantBaseName = os.path.basename(variantName)
    estimatorFiguresFolder = os.path.join(
        GLMFiguresFolder, fullEstimatorName)
    if not os.path.exists(estimatorFiguresFolder):
        os.makedirs(estimatorFiguresFolder, exist_ok=True)
    #
    estimatorMetadataPath = os.path.join(
       alignSubFolder,
       fullEstimatorName, variantName, 'estimator_metadata.pickle')
    with open(estimatorMetadataPath, 'rb') as f:
        estimatorMetadata = pickle.load(f)
    estimatorPath = os.path.join(
        alignSubFolder,
        fullEstimatorName, variantName, 'estimator.joblib')
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
    with sns.plotting_context('notebook', font_scale=1):
        try:
            fig, ax = estimator.plot_xy(smoothY=25)
            # fig, ax = estimator.plot_xy(maxPR2=0.025)
            # fig, ax = estimator.plot_xy(unitName='elec75#1_p000', smoothY=10)
            for cAx in ax:
                cAx.set_xlim([100, 2000])
            pdfPath = os.path.join(
                estimatorFiguresFolder,
                'traces_example_{}.pdf'.format(
                    variantBaseName))
            fig.savefig(
                pdfPath,
                bbox_extra_artists=(i.get_legend() for i in ax),
                bbox_inches='tight')
            fig.savefig(
                pdfPath.replace('.pdf', '.png'),
                bbox_extra_artists=(i.get_legend() for i in ax),
                bbox_inches='tight')
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
        columns=['gs', 'cv', 'valid'])
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
        if 'iterBetaNorm' in regDict:
            iterBetaNorm = regDict['iterBetaNorm']
            iterScore = regDict['iterScore']
            termIndices = calcMinIter(iterBetaNorm, tol_list=tolList)
            theseParams = pd.DataFrame(
                {
                    'tol': [tolList[i] for i in range(len(tolList))],
                    'minIter': [termIndices[i] for i in range(len(tolList))],
                    'score': [iterScore[termIndices[i]] for i in range(len(tolList))]
                }
            )
            theseParams['marginalIter'] = theseParams['minIter'].diff()
            theseParams['marginalScore'] = theseParams['score'].diff()
            toleranceParamsList.append(theseParams)
            estimatorParams.loc[unit, 'minIter'] = termIndices[-1]
        toleranceParams = pd.concat(toleranceParamsList, ignore_index=True)
        if hasattr(regDict['reg'], 'reg_lambda'):
            estimatorParams.loc[unit, 'reg_lambda'] = (
                regDict['reg'].reg_lambda
            )
        if 'gridsearch' in regDict:
            gs = regDict['gridsearch']
    saveR2[variantName] = pR2
    saveEstimatorParams[variantName] = estimatorParams
    #
    fig, ax = plt.subplots(2, 1)
    sns.distplot(pR2['valid'], ax=ax[0])
    ax[0].set_title('pseudoR^2 (median {0:.3f})'.format(pR2['valid'].median()))
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel('pseudoR^2')
    ax[0].set_xlim([-0.025, .15])
    sns.boxplot(data=pR2['valid'], ax=ax[1], orient="h")
    ax[1].set_xlabel('pseudoR^2')
    ax[1].set_xlim([-0.025, .15])
    pdfPath = os.path.join(
        estimatorFiguresFolder,
        'rsq_distribution_{}.pdf'.format(
            variantBaseName))
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    if showNow:
        plt.show()
    else:
        plt.close()
    #
    nBins = 200
    #
    fig, ax = plt.subplots(3, 1)
    fig.set_tight_layout(True)
    for idx, (tol, group) in enumerate(toleranceParams.groupby('tol')):
        labelText = '{}'.format(tol)
        scoreCounts, scoreBins = np.histogram(group['score'], nBins)
        ax[0].plot(scoreBins[:-1], scoreCounts.cumsum() / scoreCounts.sum(), label=labelText)
        if tol != tolList[0]:
            scoreCounts, scoreBins = np.histogram(group['marginalScore'], nBins)
            ax[1].plot(scoreBins[:-1], scoreCounts.cumsum() / scoreCounts.sum(), label=labelText)
        scoreCounts, scoreBins = np.histogram(group['minIter'], nBins)
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
    ax[1].set_xlabel(
        'marginal score (vs. tol={0:.2E})'.format(tolList[0]))
    ax[1].set_ylabel('Count')
    ax[2].set_xlabel('Iteration count')
    plt.legend()
    #
    pdfPath = os.path.join(
        estimatorFiguresFolder,
        'minIter_distribution_{}.pdf'.format(
            variantBaseName))
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    if showNow:
        plt.show()
    else:
        plt.close()

    # def plot_average_xy
    estimator.yHat = pd.DataFrame(
        np.nan,
        index=estimator.yTest.index, columns=estimator.yTest.columns)
    #
    for unitName, thisReg in estimator.regressionList.items():
        estimator.yHat.loc[:, unitName] = thisReg['reg'].predict(estimator.xTest)
    #
    allY = pd.concat(
        {
            'test_data': estimator.yTest,
            'test_prediction': estimator.yHat},
        names=['regressionDataType'])
    #
    dataQuery = '&'.join([
        '((RateInHz==100)|(amplitude==0))',
        '(pedalSizeCat=="M")'
        ])
    #
    #
    # maxPR2 = None
    # unitName = None
    # scores = [
    #     {
    #         'unit': k, 'score': v['gridsearch_best_mean_test_score'],
    #         'std_score': v['gridsearch_best_std_test_score']}
    #     for k, v in estimator.regressionList.items()]
    # scoresDF = pd.DataFrame(scores)
    # #
    # if unitName is None:
    #     if maxPR2 is not None:
    #         uIdx = (
    #             scoresDF
    #             .loc[scoresDF['score'] < maxPR2, 'score']
    #             .idxmax()
    #             )
    #     else:
    #         uIdx = scoresDF['score'].idxmax()
    #     unitName = scoresDF.loc[uIdx, 'unit']
    # else:
    #     uIdx = scoresDF.loc[scoresDF['unit'] == unitName, :].index[0]
    #
    '''>>> allY.index.names
        FrozenList([
            'regressionDataType', 'segment', 'originalIndex',
            't', 'RateInHz', 'activeGroup', 'amplitude',
            'amplitudeCat', 'electrode', 'pedalDirection',
            'pedalMetaCat', 'pedalMovementCat', 'pedalMovementDuration',
            'pedalSize', 'pedalSizeCat', 'pedalVelocityCat', 'program', 'bin'])
    '''
    trialInfo = estimator.yTest.index.to_frame(index=False)
    nBins = trialInfo['bin'].unique().size
    nTrials = trialInfo.groupby([
        'RateInHz', 'amplitude',
        'electrode', 'pedalSizeCat'])['bin'].count() / nBins
    print('Test set contains:\n{}'.format(nTrials))
    ########################################################################
    rollingWindowList = [1, 20, 30, 50, 100, 200]
    pR2smoothDict = {
        i: pd.DataFrame(
            np.nan,
            index=estimator.regressionList.keys(),
            columns=['pR2'])
        for i in rollingWindowList
        }
    for uIdx, unitName in enumerate(sorted(estimator.regressionList.keys())):
        thisReg = estimator.regressionList[unitName]['reg']
        yhat = pd.Series(
            thisReg.predict(estimator.xTest),
            index=estimator.yTest[unitName].index)
        for rIdx, rollingWindow in enumerate(rollingWindowList):
            pR2smoothDict[rollingWindow].loc[unitName, 'pR2'] = _poisson_pseudoR2(
                estimator.yTest[unitName].rolling(rollingWindow, center=True).mean().dropna().iloc[int(rollingWindow/2)::rollingWindow].to_numpy(),
                yhat.rolling(rollingWindow, center=True).mean().dropna().iloc[int(rollingWindow/2)::rollingWindow].to_numpy())
    #
    pR2smooth = pd.concat(pR2smoothDict, names=['rollingWindow']).reset_index()
    #
    #
    cPalettes = sns.color_palette()
    fig, ax = plt.subplots(2, 1, sharex=True)
    for rIdx, rollingWindow in enumerate(rollingWindowList):
        rMask = pR2smooth['rollingWindow'] == rollingWindow
        lblText = '{} sample average; (median pR2 {:.3f})'.format(
            rollingWindow, pR2smooth.loc[rMask, 'pR2'].median())
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
            variantBaseName))
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    if showNow:
        plt.show()
    else:
        plt.close()
    if arguments['makePredictionPDF']:
        rollingWindow = 20
        plotData = (
            allY.query(dataQuery)
            .rolling(rollingWindow).mean()
            .dropna().iloc[int(rollingWindow/2)::rollingWindow]
            .reset_index())
        pdfPath = os.path.join(
            estimatorFiguresFolder,
            'prediction_examples_{}.pdf'.format(
                variantBaseName))
        with PdfPages(pdfPath) as pdf:
            for uIdx, unitName in enumerate(sorted(estimator.regressionList.keys())):
                thisReg = estimator.regressionList[unitName]
                print('Plotting {}'.format(unitName))
                ax = sns.relplot(
                    x='bin', y=unitName,
                    data=plotData,
                    style='regressionDataType',
                    hue='amplitude',
                    col='electrode',
                    kind='line')
                plt.suptitle('pR^2 = {:.3f}'.format(thisReg['validationScore']))
                pdf.savefig()
                plt.close()
    #####################################################################
betasForPlot = (
    estimator.betas.mask(~estimator.significantBetas).stack().stack()
    .to_frame(name='beta').reset_index())
betasForPlot['beta_abs'] = betasForPlot['beta'].abs()
betasForPairgrid = betasForPlot.pivot(
    index='unit', columns='regressor', values='beta_abs')
#
significantNames = (
    betasForPairgrid
    .columns[betasForPairgrid.notna().any()]
    .to_list())
#
betaStats = (
    betasForPlot
    .groupby(['unit', 'regressor'])
    .agg({'beta': ['mean', 'std']}))
betaStats.columns = betaStats.columns.droplevel(0)
betaStats['cv'] = betaStats['std'] / betaStats['mean']
#
with sns.plotting_context('notebook', font_scale=0.75):
    g = sns.FacetGrid(
        betasForPlot,
        sharex=False, sharey=False,
        col='regressor', hue='regressor_lag')
    # bins = np.linspace(0, 1e-24, 10)
    g.map(
        plt.hist, 'beta_abs',
        # bins=bins
        ).add_legend().set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
    #
    pdfPath = os.path.join(
        GLMFiguresFolder,
        '{}_evaluation_betasAllBins.pdf'.format(
            fullEstimatorName))
    plt.savefig(pdfPath)
    plt.close()
    # plt.show()
#
with sns.plotting_context('notebook', font_scale=0.75):
    g = sns.pairplot(betasForPairgrid, vars=significantNames)
    #
    for r in range(g.axes.shape[0]):
        newLim = g.axes[r, r].get_xlim()
        for c in range(g.axes.shape[1]):
            g.axes[r, c].xaxis.set_major_formatter(ticker.EngFormatter())
            g.axes[r, c].yaxis.set_major_formatter(ticker.EngFormatter())
            if c != r:
                g.axes[r, c].set_ylim(newLim)
    #
    pdfPath = os.path.join(
        GLMFiguresFolder,
        '{}_evaluation_betas.pdf'.format(
            fullEstimatorName))
    #
    plt.tight_layout()
    plt.savefig(pdfPath)
    plt.close()
    # plt.show()
