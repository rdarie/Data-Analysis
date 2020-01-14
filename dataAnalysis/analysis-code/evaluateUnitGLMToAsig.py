"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
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
#
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import os, glob
from scipy import stats
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle
from statsmodels.stats.multitest import multipletests as mt
from patsy import (
    ModelDesc, EvalEnvironment, Term, Sum, Treatment, INTERCEPT,
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices)
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
#
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
#
dummyEstimatorMetadataPath = os.path.join(
   alignSubFolder,
   fullEstimatorName, 'var_000', 'estimator_metadata.pickle')
with open(dummyEstimatorMetadataPath, 'rb') as f:
    dummyEstimatorMetadata = pickle.load(f)
targetH5Path = dummyEstimatorMetadata['targetPath']
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
variantList = range(15)
if DEBUGGING:
    variantList = [1]
for variantName in ['var_{:03d}'.format(i) for i in variantList]:
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
    estimator.xTrain = dmatrix(
        desc, featuresDF.iloc[train, :],
        return_type='dataframe')
    estimator.xTest = dmatrix(
        desc, featuresDF.iloc[test, :],
        return_type='dataframe')
    estimator.yTrain = targetDF.iloc[train, :]
    estimator.yTest = targetDF.iloc[test, :]
    for idx, colName in enumerate(estimator.yTrain.columns):
        reg = estimator.regressionList[colName]['reg']
        if hasattr(reg, 'model_'):
            reg.model_.endog = estimator.xTrain
            reg.model_.exog = estimator.yTrain.loc[:, colName]
    ######################################################################
    with sns.plotting_context('notebook', font_scale=1):
        fig, ax = estimator.plot_xy()
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
            bbox_extra_artists=(i.get_legend() for i in ax),
            bbox_inches='tight')
        fig.savefig(
            pdfPath.replace('.pdf', '.png'),
            bbox_extra_artists=(i.get_legend() for i in ax),
            bbox_inches='tight')
        plt.close()
    #
    pR2 = pd.DataFrame(
        np.nan,
        index=estimator.regressionList.keys(),
        columns=['gs', 'cv', 'valid'])
    for unit, regDict in estimator.regressionList.items():
        pR2.loc[unit, 'gs'] = regDict['gridsearch_mean_test_score']
        pR2.loc[unit, 'cv'] = regDict['cross_val_mean_test_score']
        pR2.loc[unit, 'valid'] = regDict['validationScore']
    saveR2[variantName] = pR2
    ax = sns.distplot(pR2['valid'])
    ax.set_title('pseudoR^2 for population of units')
    ax.set_ylabel('Count')
    ax.set_xlabel('pseudoR^2')
    ax.set_xlim([-0.025, .15])
    pdfPath = os.path.join(
        estimatorFiguresFolder,
        'rsq_distribution_{}.pdf'.format(
            variantName))
    plt.savefig(pdfPath)
    plt.savefig(pdfPath.replace('.pdf', '.png'))
    plt.close()
#
if DEBUGGING:
    pdb.set_trace()
#
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
