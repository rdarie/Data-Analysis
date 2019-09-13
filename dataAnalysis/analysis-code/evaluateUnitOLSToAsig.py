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
#
rsquared = np.array([regDict['reg'].rsquared for regDict in estimator.regressionList if np.isfinite(regDict['reg'].rsquared)])
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

betas = estimator.betas
betaMax = estimator.betaMax.stack(level='positionBin')
#
pvals = pd.DataFrame(np.nan, index=betas.index, columns=betas.columns)
for idx, regDict in enumerate(estimator.regressionList):
    pvals.iloc[idx, :] = regDict['reg'].pvalues

# pdb.set_trace()
origShape = pvals.shape
flatPvals = pvals.to_numpy().reshape(-1)
try:
    _, fixedPvals, _, _ = mt(flatPvals, method='holm')
except Exception:
    fixedPvals = flatPvals / flatPvals.size
pvals.iloc[:, :] = fixedPvals.reshape(origShape)
pvalsMax = pvals.loc[:, betaMax.columns]
betaMax.columns = betaMax.columns.droplevel('lag')
pvalsMax.columns = betaMax.columns
#
alpha = 0.01
significantBetas = pvals < alpha
significantBetaMax = pvalsMax < alpha
print((significantBetas).aggregate('sum'))
#
betasForPlot = (
    betas
    .mask(~significantBetas)
    .stack().stack()
    .to_frame(name='beta').reset_index())
betasForPlot['beta_abs'] = betasForPlot['beta'].abs()
betasForPairgrid = (
    betaMax
    .mask(~significantBetaMax)
    )
significantNames = betasForPairgrid.columns[betasForPairgrid.notna().any()].to_list()
#
betaStats = betasForPlot.groupby(['unit', 'taskVariable']).agg({'beta': ['mean', 'std']}).dropna()
betaStats.columns = betaStats.columns.droplevel(0)
betaStats['cv'] = betaStats['std'] / betaStats['mean']
g = sns.FacetGrid(
    betasForPlot,
    sharex=False, sharey=False,
    col="taskVariable", hue='lag')
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
# TODO: betas from different lags don't appear as the same datapoint in the pairgrid
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
#
g = sns.FacetGrid(
    pd.melt(pvals, var_name='coefficient', value_name='p-value'),
    col="coefficient", margin_titles=True)
bins = np.linspace(0, 1e-24, 10)
g.map(
    plt.hist, 'p-value',
    bins=bins
    )
pdfPath = os.path.join(
    figureFolder,
    '{}_evaluation_pvalues.pdf'.format(
        arguments['estimator']))
plt.savefig(pdfPath)
plt.close()
#