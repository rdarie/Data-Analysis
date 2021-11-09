"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                      which experimental day to analyze
    --blockIdx=blockIdx                            which trial to analyze [default: 1]
    --processAll                                   process entire experimental day? [default: False]
    --plotting                                     make plots? [default: False]
    --showFigures                                  show plots? [default: False]
    --verbose=verbose                              print diagnostics? [default: 0]
    --debugging                                    print diagnostics? [default: False]
    --estimatorName=estimatorName                  filename for resulting estimator (cross-validated n_comps)
    --datasetPrefix=datasetPrefix                  filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName                  filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName                    append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName              append a name to the resulting blocks? [default: motion]
    --iteratorSuffixList=iteratorSuffixList        which iterators to compare
"""

import logging
logging.captureWarnings(True)
import matplotlib, os
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import os, sys
from itertools import combinations
import pingouin as pg
from sklearn.pipeline import make_pipeline, Pipeline
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
import colorsys
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS
import joblib as jb
import dill as pickle
import gc
import vg
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
from numpy.random import default_rng
from itertools import product
from scipy import stats
import scipy
import umap
import umap.plot
from tqdm import tqdm
#
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }

mplRCParams = {
    'figure.titlesize': 7,
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1,  # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

for arg in sys.argv:
    print(arg)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'selectionName': 'lfp_CAR', 'verbose': '1', 'exp': 'exp202101271100',
        'analysisName': 'hiRes', 'blockIdx': '2', 'alignFolderName': 'motion', 'processAll': True,
        'plotting': True, 'datasetPrefix': 'Block_XL_df', 'iteratorSuffixList': 'ca, cb, ccs, ccm, ra',
        'debugging': False, 'estimatorName': 'mahal_ledoit', 'showFigures': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder,
        arguments['analysisName'], 'dimensionality')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
#
estimatorsSubFolder = os.path.join(
    analysisSubFolder, 'estimators')
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'], 'dimensionality')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder)

selectionName = arguments['selectionName']
#
estimatorName = arguments['estimatorName']
fullEstimatorName = '{}_{}_{}'.format(
    estimatorName, arguments['datasetPrefix'], selectionName)
resultPath = os.path.join(estimatorsSubFolder, '{}.h5'.format(fullEstimatorName))
listOfIteratorSuffixes = [x.strip() for x in arguments['iteratorSuffixList'].split(',')]
summariesDict = {}
for iterIdx, iteratorSuffix in enumerate(listOfIteratorSuffixes):
    datasetName = arguments['datasetPrefix'] + '_{}'.format(iteratorSuffix)
    #thisEstimatorName = '{}_{}_{}'.format(
    #    estimatorName, datasetName, selectionName)
    #estimatorPath = os.path.join(
    #    estimatorsSubFolder,
    #    thisEstimatorName + '.h5'
    #    )
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    with open(loadingMetaPath, 'rb') as _f:
        dataLoadingMeta = pickle.load(_f)
        cvIterator = dataLoadingMeta['iteratorsBySegment'][0]
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    breakDownData, breakDownText, breakDownHtml = hf.calcBreakDown(
        dataDF.index.to_frame().drop_duplicates(['segment', 'originalIndex', 't']).reset_index(drop=True),
        breakDownBy=stimulusConditionNames)
    breakDownData.rename(columns={'count': 'numTrials'}, inplace=True)
    breakDownData.loc[:, 'numSamples'] = dataDF.groupby(breakDownData.index.names).count().iloc[:, 0].to_numpy()
    summariesDict[iteratorSuffix] = breakDownData
#
allSummariesDF = pd.concat(summariesDict, names=['iterator'])
allSummariesDF.to_hdf(resultPath, 'sampleCount')
print('Done saving summary of sample nums to {}:sampleCount'.format(resultPath))
