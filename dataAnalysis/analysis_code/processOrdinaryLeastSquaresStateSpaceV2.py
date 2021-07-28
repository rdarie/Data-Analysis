"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose=verbose                        print diagnostics?
    --debugging                              print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
"""

import logging
logging.captureWarnings(True)
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
from scipy.stats import mode
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklego.preprocessing import PatsyTransformer
from dataAnalysis.analysis_code.regression_parameters import *
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
import patsy
from itertools import product
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
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
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
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
    'figure.titlesize': 7
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

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_ra', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr_pca_ta',
        'blockIdx': '2', 'exp': 'exp202101271100'}
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
        figureFolder, arguments['analysisName'], 'regression')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder)
#
datasetName = arguments['datasetName']
fullEstimatorName = '{}_{}'.format(
    arguments['estimatorName'], arguments['datasetName'])
#
estimatorsSubFolder = os.path.join(
    analysisSubFolder, 'estimators')
dataFramesFolder = os.path.join(
    analysisSubFolder, 'dataframes')
datasetPath = os.path.join(
    dataFramesFolder,
    datasetName + '.h5'
    )
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
estimatorMetaDataPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '_meta.pickle'
    )
with open(estimatorMetaDataPath, 'rb') as _f:
    estimatorMeta = pickle.load(_f)
#
loadingMetaPath = estimatorMeta['loadingMetaPath']
with open(loadingMetaPath, 'rb') as _f:
    loadingMeta = pickle.load(_f)
    iteratorOpts = loadingMeta['iteratorOpts']
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
#
for hIdx, histOpts in enumerate(addHistoryTerms):
    locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
thisEnv = patsy.EvalEnvironment.capture()

iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
# cv_kwargs = loadingMeta['cv_kwargs'].copy()
cvIterator = iteratorsBySegment[0]
lastFoldIdx = cvIterator.n_splits
#
selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
#
lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormula', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(lambda x: ' + '.join(x), axis='columns')
#
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
#
with pd.HDFStore(estimatorPath) as store:
    ADF = pd.read_hdf(store, 'A')
    BDF = pd.read_hdf(store, 'B')
    CDF = pd.read_hdf(store, 'C')
    DDF = pd.read_hdf(store, 'D')
    eigDF = pd.read_hdf(store, 'eigenvalues')
    eigValPalette = pd.read_hdf(store, 'eigValPalette')
# check eigenvalue persistence over reduction of the state space dimensionality
eigDF.loc[:, 'freqBandName'] = eigDF.reset_index()['rhsMaskIdx'].map(rhsMasksInfo['freqBandName']).to_numpy()
eigDF.loc[:, 'designFormula'] = eigDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['designFormula']).to_numpy()
eigDF.loc[:, 'designFormulaLabel'] = eigDF.reset_index()['designFormula'].apply(lambda x: x.replace(' + ', ' +\n')).to_numpy()
eigDF.loc[:, 'fullFormula'] = eigDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
eigDF.loc[:, 'fullFormulaLabel'] = eigDF.reset_index()['fullFormula'].apply(lambda x: x.replace(' + ', ' +\n')).to_numpy()
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalue_reduction'))
with PdfPages(pdfPath) as pdf:
    for name, thisPlotEig in eigDF.groupby(['fullFormula', 'freqBandName']):
        height, width = 3, 3
        aspect = width / height
        g = sns.relplot(
            col='stateNDim', col_wrap=3,
            x='real', y='imag', hue='eigValType',
            height=height, aspect=aspect,
            facet_kws={'margin_titles': True},
            palette=eigValPalette.to_dict(),
            kind='scatter', data=thisPlotEig.reset_index(), rasterized=True, edgecolor=None)
        g.suptitle('design: {} freqBand: {}'.format(*name))
        for ax in g.axes.flatten():
            c = Circle((0, 0), 1, ec=(0, 0, 0, 1.), fc=(0, 0, 0, 0))
            ax.add_artist(c)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
#
plotEig = eigDF.loc[eigDF['nDimIsMax'], :]
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalues'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    g = sns.relplot(
        row='fullFormulaLabel', col='freqBandName',
        x='real', y='imag', hue='eigValType',
        height=height, aspect=aspect,
        palette=eigValPalette.to_dict(),
        facet_kws={'margin_titles': True}, rasterized=True, edgecolor=None,
        kind='scatter', data=plotEig)
    for ax in g.axes.flatten():
        c = Circle((0, 0), 1, ec=(0, 0, 0, 1.), fc=(0, 0, 0, 0))
        ax.add_artist(c)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()

'''pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalues'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    plotMask = plotEig['designFormula'] == lOfDesignFormulas[1]
    g = sns.relplot(
        row='designFormulaLabel', col='freqBandName',
        x='real', y='imag', hue='eigValType',
        height=height, aspect=aspect,
        kind='scatter',
        facet_kws={'margin_titles': True}, edgecolor=None, rasterized=True,
        data=plotEig.loc[plotMask, :])
    g.suptitle('design: {} freqBand: {}'.format(*name))
    for ax in g.axes.flatten():
        c = Circle((0,0), 1, ec=(0, 0, 0, 1.), fc=(0, 0, 0, 0))
        ax.add_artist(c)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
'''