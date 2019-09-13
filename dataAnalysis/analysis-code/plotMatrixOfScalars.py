"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --trialIdx=trialIdx                       which trial to analyze [default: 1]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --processAll                              process entire experimental day? [default: False]
    --verbose                                 print diagnostics? [default: True]
    --window=window                           process with short window? [default: short]
    --inputBlockName=inputBlockName           which trig_ block to pull [default: pca]
    --secondaryBlockName=secondaryBlockName   filename for secondary inputs [default: RC]
    --resultName=resultName                   name of field to request [default: emgMaxCrossCorr]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output
# matplotlib.use('Qt5Agg')   # generate interactive output
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("paper")
sns.set_style("whitegrid")

from namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
resultPath = os.path.join(
    analysisSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockName'], arguments['window'],
        arguments['resultName']))

dataDF = pd.read_hdf(resultPath, arguments['resultName'])
#cmap = ListedColormap(
#    sns.color_palette(
#        nrnRelplotKWArgs['palette'], 512))
colMask = np.array(['CH' in i for i in dataDF.columns], dtype=np.bool)
colMask = colMask & ~np.array(['CH15' in i for i in dataDF.columns], dtype=np.bool)
rowMask = np.array([True for i in dataDF.index], dtype=np.bool)
plotDF = dataDF.loc[rowMask, colMask]
cmap = sns.cubehelix_palette(
    n_colors=256, start=0, rot=.5, reverse=True, as_cmap=True)
plotDist = False
if plotDist:
    ax = sns.distplot(np.ravel(plotDF))
    plt.show()
dataFlat = np.ravel(plotDF)
vMin, vMax = np.quantile(dataFlat, [0.05, 0.95])
# pdb.set_trace()
with PdfPages(pdfPath) as pdf:
    f, ax = plt.subplots()
    w = 4 # size in inches
    f.set_size_inches(w, w * plotDF.shape[0] / plotDF.shape[1])
    sns.heatmap(
        plotDF, cmap=cmap,
        robust=True,
        vmin=vMin, vmax=vMax,
        ax=ax,
        square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_xticklabels(
        [i.get_text().split('#')[0] for i in ax.get_xticklabels()],
        fontdict={'fontsize': ax.get_xticklabels()[0].get_size()})
    ax.set_yticklabels(
        [i.get_text().split('_')[0] for i in ax.get_yticklabels()],
        fontdict={'fontsize': ax.get_xticklabels()[0].get_size()})
    pdf.savefig()
    plt.close()
