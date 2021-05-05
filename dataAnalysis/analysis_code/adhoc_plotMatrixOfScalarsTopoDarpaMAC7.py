"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --blockIdx=blockIdx                       which trial to analyze [default: 1]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName         append a name to the resulting blocks? [default: motion]
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
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.preproc.ns5 as ns5
from scipy import stats
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("paper")
sns.set_style("whitegrid")
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName'])
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#

def getPairResult(resultPath, rippleMapDF, resultName):
    dataDF = pd.read_hdf(resultPath, resultName)
    dataDF.columns = dataDF.columns.droplevel(level='lag')
    dataDF.columns.name = 'from'
    dataDF.index = dataDF.index.droplevel(level='lag')
    dataDF.index.name = 'to'
    #
    duplicatesMask = np.zeros_like(dataDF,  dtype=bool)
    duplicatesMask[np.triu_indices_from(duplicatesMask)] = True
    #
    dataStack = (
        dataDF
        .mask(duplicatesMask)
        .stack().dropna()
        .to_frame(name='corr'))
    dataStack['rsquared'] = dataStack['corr'] ** 2
    dataStack.loc[:, 'xdist'] = np.nan
    dataStack.loc[:, 'ydist'] = np.nan
    dataStack.loc[:, 'dist'] = np.nan
    tos = dataStack.index.get_level_values('to')
    froms = dataStack.index.get_level_values('from')
    dataStack['pairOriginArray'] = 'mixed'
    for arrayName in ['rostral', 'caudal']:
        dataStack.loc[
            tos.str.contains(arrayName) &
            froms.str.contains(arrayName), 'pairOriginArray'] = arrayName
    dataStack['pairOriginBank'] = 'mixed'
    for bankName in ['X', 'Y', 'Z']:
        dataStack.loc[
            tos.str.contains(bankName) &
            froms.str.contains(bankName), 'pairOriginBank'] = bankName
    for fromLabel in rippleMapDF['label']:
        for toLabel in rippleMapDF['label']:
            catIndex = (fromLabel + '#0', toLabel + '#0')
            if catIndex in dataStack.index:
                xposFrom = rippleMapDF.loc[rippleMapDF['label'] == fromLabel, 'xcoords']
                assert xposFrom.size == 1
                xposFrom = xposFrom.iloc[0]
                yposFrom = rippleMapDF.loc[rippleMapDF['label'] == fromLabel, 'ycoords']
                assert yposFrom.size == 1
                yposFrom = yposFrom.iloc[0]
                #
                xposTo = rippleMapDF.loc[rippleMapDF['label'] == toLabel, 'xcoords']
                assert xposTo.size == 1
                xposTo = xposTo.iloc[0]
                yposTo = rippleMapDF.loc[rippleMapDF['label'] == toLabel, 'ycoords']
                assert yposTo.size == 1
                yposTo = yposTo.iloc[0]
                dataStack.loc[catIndex, 'xdist'] = np.abs(xposFrom - xposTo)
                dataStack.loc[catIndex, 'ydist'] = np.abs(yposFrom - yposTo)
                dataStack.loc[catIndex, 'dist'] = np.sqrt(
                    (xposFrom - xposTo) ** 2 + (yposFrom - yposTo) ** 2)
    return dataStack

resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_prepoststim.pdf'.format(
        arguments['inputBlockName'], arguments['window']))
rippleMapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])

dataStackDict = {
    'corrPreStim': getPairResult(resultPath, rippleMapDF, 'corrPreStim'),
    'corr': getPairResult(resultPath, rippleMapDF, 'corr')
    }
dataStack = pd.concat(dataStackDict, names=['window', 'from', 'to'])
# pdb.set_trace()
stat, pv = stats.wilcoxon(
    dataStackDict['corrPreStim']['rsquared'],
    dataStackDict['corr']['rsquared'])
with PdfPages(pdfPath) as pdf:
    f = plt.figure(
        constrained_layout=True,
        figsize=(11, 9))
    spec = gridspec.GridSpec(
        ncols=1, nrows=1, figure=f)
    ax = f.add_subplot(spec[0, 0])
    sns.boxplot(
        x='window', y='rsquared', order=['corrPreStim', 'corr'],
        data=dataStack.reset_index(), ax=ax)
    # ax.set_ylim([0, 1])
    # for arrayName, group in dataStack.groupby(['pairOriginBank', 'pairOriginArray']):
    #     sns.regplot(x='dist', y='rsquared', data=group, label=arrayName, ax=ax)
    # plt.legend()
    plt.title('p = {:.4}'.format(pv))
    pdf.savefig()
    plt.show()
