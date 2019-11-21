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
    --window=window                        process with short window? [default: short]
    --estimator=estimator                  estimator filename
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockName=inputBlockName        filename for resulting estimator [default: fr_sqrt]
"""
#
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')  # generate postscript output 
matplotlib.use('QT5Agg')  # generate postscript output
import matplotlib.pyplot as plt
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import os
from scipy import stats
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle

from statsmodels.stats.multitest import multipletests as mt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")
import matplotlib.ticker as ticker
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
#
estimatorPath = os.path.join(
    analysisSubFolder,
    arguments['estimator'] + '.joblib')
with open(
    os.path.join(
        analysisSubFolder,
        arguments['estimator'] + '_meta.pickle'),
        'rb') as f:
    estimatorMetadata = pickle.load(f)
estimator = jb.load(
    os.path.join(analysisSubFolder, estimatorMetadata['path']))
# #)
with sns.plotting_context('notebook', font_scale=1):
    estimator.plot_xy()
regressorH5Path = os.path.join(
    analysisSubFolder,
    estimatorMetadata['trainingDataPath']
    .replace('_raster', '_{}_rig'.format(estimatorMetadata['name']))
    .replace('.nix', '.h5')
    )
featuresDF = pd.read_hdf(regressorH5Path, 'feature')
for idx, (name, regDict) in enumerate(estimator.regressionList.items()):
    reg = regDict['reg']
    gs = regDict['gridSearchCV']
    if idx == 0:
        regressorNames = []
        regressorTypes = []
        regressorLags = []
        for paramIdx, featureMask in enumerate(gs.param_grid['featureMask']):
            theseFeatureNames = featuresDF.iloc[:, featureMask].columns
            regressorNames.append(theseFeatureNames)
            regressorLags.append(theseFeatureNames.to_list()[0][1])
            if np.any(['Hz' in i[0] for i in theseFeatureNames.to_list()]):
                regressorTypes.append('iar')
            elif np.any(['ACR' in i[0] for i in theseFeatureNames.to_list()]):
                regressorTypes.append('acr')
            else:
                regressorTypes.append('kin')
        regressorIndex = pd.MultiIndex.from_arrays(
            [regressorTypes, regressorLags], names=('type', 'lag'))
        targetNames = [i[0] for i in estimator.regressionList.keys()]
        regressionScores = pd.DataFrame(np.nan, index=regressorIndex, columns=targetNames)
    regressionScores.loc[:, name[0]] = gs.cv_results_['mean_test_score']
    regressionScores.columns.name = 'target'

maxScore = (
    regressionScores
    .groupby('type').agg('max')
    .unstack().reset_index()
    .rename(columns={0: 'score'}))
from itertools import combinations
uniqueTypes = maxScore['type'].unique()
tTestPval = pd.DataFrame(np.nan, index=uniqueTypes, columns=uniqueTypes)
for names in combinations(uniqueTypes, 2):
    stat, pval = stats.ttest_rel(
        maxScore.loc[maxScore['type'] == names[0], 'score'].to_numpy(),
        maxScore.loc[maxScore['type'] == names[1], 'score'].to_numpy()
        )
    tTestPval.loc[names[0], names[1]] = pval
f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(
    x="score", y="type", data=maxScore,
    whis="range", palette="vlag")
sns.swarmplot(
    x="score", y="type", data=maxScore,
    size=3, color=".3", linewidth=0)

pdfPath = os.path.join(
    figureFolder,
    '{}_evaluation_rsq.pdf'.format(
        arguments['estimator']))
plt.savefig(pdfPath)
plt.close()

maxLag = (
    regressionScores
    .groupby('type').agg('idxmax')
    .unstack().reset_index()
    .rename(columns={0: 'lag'}))
maxLag['lag'] = maxLag['lag'].apply(lambda x: x[1])
f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(
    x="lag", y="type", data=maxLag,
    whis="range", palette="vlag")
sns.swarmplot(
    x="lag", y="type", data=maxLag,
    size=3, color=".3", linewidth=0)
rsquared = np.array(
    [
        regDict['mean_test_score']
        for unit, regDict in estimator.regressionList.items()
        if np.isfinite(regDict['mean_test_score'])])

pdfPath = os.path.join(
    figureFolder,
    '{}_evaluation_lags.pdf'.format(
        arguments['estimator']))
plt.savefig(pdfPath)
plt.close()

'''
ax = sns.distplot(rsquared)
ax.set_title('R^2 for population of units')
ax.set_ylabel('Count')
ax.set_xlabel('R^2')
pdfPath = os.path.join(
    figureFolder,
    '{}_evaluation_rsq.pdf'.format(
        arguments['estimator']))
plt.savefig(pdfPath)
plt.close()
'''
#
betas = estimator.betas
betas.index.names = ['unit', 'unit_lag']
betas.columns.names = ['regressor', 'regressor_lag']
pvals = estimator.pvals
pvals.index.names = ['unit', 'unit_lag']
pvals.columns.names = ['regressor', 'regressor_lag']
significantBetas = estimator.significantBetas
significantBetas.index.names = ['unit', 'unit_lag']
significantBetas.columns.names = ['regressor', 'regressor_lag']
print((significantBetas).aggregate('sum'))
#
betasForPlot = (
    betas
    .mask(~significantBetas)
    .stack().stack()
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
        figureFolder,
        '{}_evaluation_betasAllBins.pdf'.format(
            arguments['estimator']))
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
        figureFolder,
        '{}_evaluation_betas.pdf'.format(
            arguments['estimator']))
    #
    plt.tight_layout()
    plt.savefig(pdfPath)
    plt.close()
    # plt.show()
