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

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
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
    'tight_layout.pad': 3e-1, # units of font size
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
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_d', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr2_ta_spectral',
        'blockIdx': '2', 'exp': 'exp202101281100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    scratchPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings'
    scratchFolder = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202101201100-Rupert'
    figureFolder = '/gpfs/data/dborton/rdarie/Neural Recordings/processed/202101201100-Rupert/figures'
    
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

#
if estimatorMeta['pipelinePathRhs'] is not None:
    transformedSelectionNameRhs = '{}_{}'.format(
        selectionNameRhs, estimatorMeta['arguments']['transformerNameRhs'])
    transformedRhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(transformedSelectionNameRhs))
    pipelineScoresRhsDF = pd.read_hdf(estimatorMeta['pipelinePathRhs'], 'work')
    workingPipelinesRhs = pipelineScoresRhsDF['estimator']
else:
    workingPipelinesRhs = None
#
if estimatorMeta['pipelinePathLhs'] is not None:
    pipelineScoresLhsDF = pd.read_hdf(estimatorMeta['pipelinePathLhs'], 'work')
    workingPipelinesLhs = pipelineScoresLhsDF['estimator']
else:
    workingPipelinesLhs = None
#
estimatorsDF = pd.read_hdf(estimatorPath, 'cv_estimators')
scoresDF = pd.read_hdf(estimatorPath, 'cv_scores')
lhsDF = pd.read_hdf(estimatorMeta['lhsDatasetPath'], '/{}/data'.format(selectionNameLhs))
rhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(selectionNameRhs))
lhsMasks = pd.read_hdf(estimatorMeta['lhsDatasetPath'], '/{}/featureMasks'.format(selectionNameLhs))
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
#
trialInfoLhs = lhsDF.index.to_frame().reset_index(drop=True)
trialInfoRhs = rhsDF.index.to_frame().reset_index(drop=True)
checkSameMeta = stimulusConditionNames + ['bin', 'trialUID', 'conditionUID']
assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
trialInfo = trialInfoLhs
#
lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
#
binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
#
with pd.HDFStore(estimatorPath) as store:
    ADF = pd.read_hdf(store, 'A')
    BDF = pd.read_hdf(store, 'B')
    CDF = pd.read_hdf(store, 'C')
    DDF = pd.read_hdf(store, 'D')
    eigDF = pd.read_hdf(store, 'eigenvalues')
# check eigenvalue persistence over reduction of the state dimension
eigenValueDict = {}
iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold', 'electrode']
for name, thisA in ADF.groupby(iRGroupNames):
    for stateSpaceDim in range(1, thisA.shape[0], 10):
        print(stateSpaceDim)
        w, v = np.linalg.eig(thisA.iloc[:stateSpaceDim, :stateSpaceDim])
        entryName = tuple([n for n in name] + [stateSpaceDim])
        eigenValueDict[entryName] = pd.Series(w)
        eigenValueDict[entryName].index.name = 'eigenvalueIndex'
reducedEigDF = pd.concat(eigenValueDict, names=iRGroupNames + ['spaceDim'])
plotRedEig = pd.concat({
    'magnitude': reducedEigDF.apply(lambda x: np.absolute(x)),
    'phase': reducedEigDF.apply(lambda x: np.angle(x)),
    'real': reducedEigDF.apply(lambda x: x.real),
    'imag': reducedEigDF.apply(lambda x: x.imag)}, axis='columns')
plotRedEig.reset_index(inplace=True)
rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
plotRedEig.loc[:, 'freqBandName'] = plotRedEig['rhsMaskIdx'].map(rhsMasksInfo['freqBandName'])
plotRedEig.loc[:, 'designFormula'] = plotRedEig['lhsMaskIdx'].map(lhsMasksInfo['designFormula'])
plotRedEig.loc[:, 'designFormulaLabel'] = plotRedEig['designFormula'].apply(lambda x: x.replace(' + ', ' +\n'))
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalue_reduction'))
with PdfPages(pdfPath) as pdf:
    for name, thisPlotEig in plotRedEig.groupby(['designFormula', 'freqBandName']):
        height, width = 3, 3
        aspect = width / height
        g = sns.relplot(
            col='spaceDim', col_wrap=3,
            x='real', y='imag',
            height=height, aspect=aspect,
            facet_kws={'margin_titles': True},
            kind='scatter', data=thisPlotEig)
        g.suptitle('design: {} freqBand: {}'.format(*name))
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
# sanity check that signal dynamics are independent of fold, electrode
meanA = ADF.groupby(['rhsMaskIdx', 'state']).mean()
stdA = ADF.groupby(['rhsMaskIdx', 'state']).std()
fig, ax = plt.subplots(1, 2)
sns.heatmap(meanA.reset_index(drop=True), ax=ax[0])
sns.heatmap(stdA.reset_index(drop=True), ax=ax[1])
#
plotEig = pd.concat({
    'magnitude': eigDF.apply(lambda x: np.absolute(x)),
    'phase': eigDF.apply(lambda x: np.angle(x)),
    'real': eigDF.apply(lambda x: x.real),
    'imag': eigDF.apply(lambda x: x.imag)}, axis='columns')
plotEig.reset_index(inplace=True)
rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
plotEig.loc[:, 'freqBandName'] = plotEig['rhsMaskIdx'].map(rhsMasksInfo['freqBandName'])
plotEig.loc[:, 'designFormula'] = plotEig['lhsMaskIdx'].map(lhsMasksInfo['designFormula'])
plotEig.loc[:, 'designFormulaLabel'] = plotEig['designFormula'].apply(lambda x: x.replace(' + ', ' +\n'))
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalues_all'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    g = sns.relplot(
        row='designFormula', col='freqBandName',
        x='real', y='imag',
        height=height, aspect=aspect,
        facet_kws={'margin_titles': True},
        kind='scatter', data=plotEig)
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalues'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    plotMask = plotEig['designFormula'] == lOfDesignFormulas[1]
    g = sns.relplot(
        row='designFormulaLabel', col='freqBandName',
        x='real', y='imag',
        height=height, aspect=aspect,
        kind='scatter',
        facet_kws={'margin_titles': True},
        data=plotEig.loc[plotMask, :])
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
