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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os, sys
from sklearn.pipeline import make_pipeline, Pipeline
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.decomposition import PCA, FactorAnalysis
import joblib as jb
import dill as pickle
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
from numpy.random import default_rng
from itertools import product
from scipy import stats
import umap
import umap.plot
rng = default_rng()
idxSl = pd.IndexSlice
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=0.5, color_codes=True)
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# pdb.set_trace()
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'exp': 'exp202101281100', 'blockIdx': '2', 'selectionName': 'lfp_CAR_spectral',
        'plotting': True, 'showFigures': False, 'iteratorSuffixList': 'a, b', 'estimatorName': 'pca',
        'datasetPrefix': 'Block_XL_df', 'analysisName': 'default', 'processAll': True, 'verbose': '1',
        'debugging': False, 'alignFolderName': 'motion'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

# refEstimator = estimatorsSrs.loc[('a', 'higamma', 7)]
# testEstimator = estimatorsSrs.loc[('b', 'higamma', 7)]

def weightsCovariance(est):
    components_ = est.components_.copy()
    if isinstance(est, PCA):
        exp_var_diff = np.maximum(est.explained_variance_ - est.noise_variance_, 0.)
        exp_var = est.explained_variance_
        if est.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - est.noise_variance_, 0.)
        cov = np.dot(components_.T * exp_var_diff, components_)
    elif isinstance(est, FactorAnalysis):
        cov = np.dot(components_.T, components_)
    else:
        cov = None
    return cov

def KL(
        refEstimator=None, testEstimator=None,
        recenter=False, includeNoiseVariance=True):
    # pdb.set_trace()
    if isinstance(refEstimator, Pipeline):
        refEstimator = refEstimator.named_steps['dim_red']
    if isinstance(testEstimator, Pipeline):
        testEstimator = testEstimator.named_steps['dim_red']
    #
    if includeNoiseVariance:
        if isinstance(refEstimator, EmpiricalCovariance):
            S0 = refEstimator.covariance_
        else:
            S0 = refEstimator.get_covariance()
        if isinstance(testEstimator, EmpiricalCovariance):
            S1 = testEstimator.covariance_
        else:
            S1 = testEstimator.get_covariance()
    else:
        S0 = weightsCovariance(refEstimator)
        # assert ((S0 + np.eye(S0.shape[0]) * refEstimator.noise_variance_) == refEstimator.get_covariance())
        S1 = weightsCovariance(testEstimator)
    #
    if recenter:
        u0 = np.zeros(refEstimator.mean_.shape)
        u1 = np.zeros(testEstimator.mean_.shape)
    else:
        if isinstance(refEstimator, EmpiricalCovariance):
            u0 = refEstimator.location_
        else:
            u0 = refEstimator.mean_
        if isinstance(testEstimator, EmpiricalCovariance):
            u1 = testEstimator.location_
        else:
            u1 = testEstimator.mean_
    #
    S1inv = np.linalg.inv(S1)
    # np.matmul(S1inv, S0)
    t1 = np.trace(S1inv @ S0)
    # (np.eye(S1inv.shape[0]) * S0) == S0
    # (np.eye(S1inv.shape[0]) @ S0) == S0
    t2 = (u1 - u0).T @ S1inv @ (u1 - u0)
    #
    rank0 = np.linalg.matrix_rank(S0)
    rank1 = np.linalg.matrix_rank(S1)
    assert rank0 == rank1
    t3 = rank0
    # adjustment for singular matrices
    det0 = np.linalg.det(S0)
    det1 = np.linalg.det(S1)
    t4 = np.log(det1 / det0)
    eps = np.spacing(1)
    if np.isnan(t4):
        if det0 == 0:
            adjDet0 = det0 + eps
        else:
            adjDet0 = det0
        if det1 == 0:
            adjDet1 = det1 + eps
        else:
            adjDet1 = det1
        t4 = np.log(adjDet1 / adjDet0)
        assert (not np.isnan(t4))
    # correct for numerical issues, dKL should
    # be strictly positive
    dKL = max(
        1/2 * (t1 + t2 - t3 + t4),
        0.)
    return dKL

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
estimatorName = arguments['estimatorName']
fullEstimatorName = '{}_{}_{}'.format(
    estimatorName, arguments['datasetPrefix'], selectionName)
listOfIteratorSuffixes =[x.strip() for x in arguments['iteratorSuffixList'].split(',')]
covMatDict = {}
eVSDict = {}
estimatorsDict = {}
for iteratorSuffix in listOfIteratorSuffixes:
    datasetName = arguments['datasetPrefix'] + '_{}'.format(iteratorSuffix)
    thisEstimatorName = '{}_{}_{}'.format(
        estimatorName, datasetName, selectionName)
    # pdb.set_trace()
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        thisEstimatorName + '.h5'
        )
    #
    with pd.HDFStore(estimatorPath, 'r') as store:
        if 'cv_covariance_matrices' in store:
            covMatDict[iteratorSuffix] = pd.read_hdf(store, 'cv_covariance_matrices')
        # covMatDict[iteratorSuffix].index.names == ['freqBandName', 'fold', 'feature']
        # covMatDict[iteratorSuffix].columns.name == 'feature'
        if 'full_explained_variance' in store:
            eVSDict[iteratorSuffix] = pd.read_hdf(store, 'full_explained_variance')
        if 'full_estimators' in store:
            estimatorsDict[iteratorSuffix] = pd.read_hdf(store, 'full_estimators')
        elif 'cv_estimators' in store:
            estimatorsDict[iteratorSuffix] = pd.read_hdf(store, 'cv_estimators')
            # estimatorsDict[iteratorSuffix].index.names == ['freqBandName', 'fold']
            # Series
    # pdb.set_trace()
covMatDF = pd.concat(covMatDict, names=['iterator'])
if any(eVSDict):
    eVSDF = pd.concat(eVSDict, names=['iterator'])
else:
    eVSDF = None
estimatorsSrs = pd.concat(estimatorsDict, names=['iterator'])
#
del covMatDict, eVSDict, estimatorsDict
#
klDict = {}
projectedDict = {}
for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
    klDF = pd.DataFrame(np.nan, index=estimatorGroup.index, columns=estimatorGroup.index)
    for refIdx, testIdx in product(estimatorGroup.index, estimatorGroup.index):
        klDF.loc[refIdx, testIdx] = KL(
            estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx])
    klDict[freqBandName] = klDF.copy()
    uMapper = umap.UMAP(metric='precomputed')
    uMapper.fit(klDF)
    projectedDict[freqBandName] = pd.DataFrame(
        uMapper.transform(klDF), index=klDF.index, columns=['umap{}'.format(fid) for fid in range(2)])
    projectedDict[freqBandName].columns.name = 'feature'
# pdb.set_trace()
pdfPath = os.path.join(
    figureOutputFolder, '{}_KLDiv_comparison_2D.pdf'.format(
        fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    for freqBandName, umapDF in projectedDict.items():
        g = sns.relplot(x='umap0', y='umap1', kind='scatter', hue='iterator', data=umapDF.reset_index())
        g.suptitle(freqBandName)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
pdfPath = os.path.join(
    figureOutputFolder, '{}_KLDiv_comparison.pdf'.format(
        fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    for freqBandName, klDF in klDict.items():
        fig, ax = plt.subplots(figsize=(12, 12))
        ax = sns.heatmap(
            klDF,
            # norm=LogNorm()
            )
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
if eVSDF is not None:
    pdfPath = os.path.join(
        figureOutputFolder, '{}_explained_variance_comparison.pdf'.format(
            fullEstimatorName))
    with PdfPages(pdfPath) as pdf:
        for name, group in eVSDF.groupby('freqBandName'):
            plotDF = group.xs('test', level='trialType').to_frame(name='signal').reset_index()
            g = sns.relplot(
                data=plotDF, x='component', hue='iterator',
                y='signal', kind='line', alpha=0.5, lw=0.5, errorbar='se')
            g.fig.set_size_inches((12, 8))
            g.fig.suptitle('{}'.format(name))
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=.3)
            pdf.savefig(bbox_inches='tight')
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()