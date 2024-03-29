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
matplotlib.use('PS')   # generate postscript output
# matplotlib.use('Qt5Agg')   # generate interactive output
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
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
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName'])
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockName'], arguments['window'],
        arguments['resultName']))

dataDF = pd.read_hdf(resultPath, arguments['resultName'])
dataDF.columns = dataDF.columns.droplevel(level='lag')
dataDF.columns.name = 'from'
dataDF.index = dataDF.index.droplevel(level='lag')
dataDF.index.name = 'to'

pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockName'], arguments['window'],
        arguments['resultName']))
# pdb.set_trace()
asp.plotCorrelationMatrix(
    dataDF ** 2, pdfPath,
    heatmap_kws={
        'xticklabels': 4,
        'yticklabels': 4,
        'vmin': 0, 'vmax': 1,
    }, xticklabels_kws={'rotation': 45},
    yticklabels_kws={'rotation': 'horizontal'})
