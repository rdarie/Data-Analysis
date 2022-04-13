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
import matplotlib.ticker as tix
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import os, sys
from itertools import combinations, permutations, product
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
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
from numpy.random import default_rng
from itertools import product
from scipy import stats
import scipy
import umap
import umap.plot
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
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
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "legend.frameon": True,
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

fontSizeInches = snsRCParams["font.size"] / 72
mplRCParams = {
    'figure.subplot.left': 0.01,
    'figure.subplot.right': 0.99,
    'figure.subplot.bottom': 0.01,
    'figure.subplot.top': 0.99,
    'figure.titlesize': 7,
    'axes.titlepad': 3.5,
    'axes.labelpad': 2.5,
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 2e-1,  # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc=snsRCParams)
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
    'verbose': '1', 'blockIdx': '2', 'datasetPrefix': 'Block_XL_df', 'alignFolderName': 'motion',
    'iteratorSuffixList': 'ca, cb, ccs, ccm', 'showFigures': False, 'processAll': True,
    'selectionName': 'laplace_scaled', 'analysisName': 'hiRes', 'debugging': False,
    'estimatorName': 'mahal_ledoit', 'exp': 'exp202101211100', 'plotting': True}
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
        if isinstance(refEstimator, EmpiricalCovariance):
            u0 = np.zeros(refEstimator.location_.shape)
        else:
            u0 = np.zeros(refEstimator.mean_.shape)
        if isinstance(testEstimator, EmpiricalCovariance):
            u1 = np.zeros(testEstimator.location_.shape)
        else:
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
    try:
        signDet0, logDet0 = np.linalg.slogdet(S0)
        signDet1, logDet1 = np.linalg.slogdet(S1)
        assert signDet0 == signDet1
        t4 = logDet1 - logDet0
    except Exception:
        traceback.print_exc()
        # adjustment for singular matrices
        det0 = np.linalg.det(S0)
        det1 = np.linalg.det(S1)
        #
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
    # correct any numerical issues; (dKL should
    # be strictly positive)
    dKL = max(
        1/2 * (t1 + t2 - t3 + t4),
        0.)
    return dKL

def covMatErrorNorm(
        refEstimator=None, testEstimator=None,
        norm='frobenius', scaling=False, squared=False,
        recenter=False, includeNoiseVariance=True):
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
    error = S1 - S0
    if norm == "frobenius":
        squaredNorm = np.sum(error ** 2)
    elif norm == "spectral":
        squaredNorm = np.amax(scipy.linalg.svdvals(np.dot(error.T, error)))
    else:
        raise NotImplementedError(
            "Only spectral and frobenius norms are implemented")
    # optionally scale the error norm
    if scaling:
        squaredNorm = squaredNorm / error.shape[0]
    # finally get either the squared norm or the norm
    if squared:
        result = squaredNorm
    else:
        result = np.sqrt(squaredNorm)
    return result

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
resultPath = os.path.join(estimatorsSubFolder, '{}_{}.h5'.format(fullEstimatorName, 'covarianceMatrixCalc'))
listOfIteratorSuffixes = [x.strip() for x in arguments['iteratorSuffixList'].split(',')]
covMatDict = {}
eVSDict = {}
estimatorsDict = {}
for iterIdx, iteratorSuffix in enumerate(listOfIteratorSuffixes):
    datasetName = arguments['datasetPrefix'] + '_{}'.format(iteratorSuffix)
    thisEstimatorName = '{}_{}_{}'.format(
        estimatorName, datasetName, selectionName)
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        thisEstimatorName + '.h5'
        )
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    with open(loadingMetaPath, 'rb') as _f:
        dataLoadingMeta = pickle.load(_f)
        cvIterator = dataLoadingMeta['iteratorsBySegment'][0]
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
    #
    if iterIdx == 0:
        datasetPath = os.path.join(
            dataFramesFolder,
            datasetName + '.h5'
            )
        dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
        mapDF = dataDF.columns.to_frame().reset_index(drop=True).set_index('feature')
        del dataDF
covMatDF = pd.concat(covMatDict, names=['iterator'])
if any(eVSDict):
    eVSDF = pd.concat(eVSDict, names=['iterator'])
else:
    eVSDF = None
estimatorsSrs = pd.concat(estimatorsDict, names=['iterator'])
lastFoldIdx = estimatorsSrs.index.get_level_values('fold').max()
workEstimators = estimatorsSrs.xs(lastFoldIdx, level='fold')
estimatorsSrs.drop(lastFoldIdx, axis='index', level='fold', inplace=True)
# discard the all band?
if 'spectral' in arguments['selectionName']:
    estimatorsSrs.drop('all', axis='index', level='freqBandName', inplace=True)
#
# pdb.set_trace()
del covMatDict, eVSDict, estimatorsDict
calcAllDistances = True
distancesDict = {}
projectedDict = {}
if calcAllDistances:
    lOfDistanceTypes = ['KL', 'frobenius', 'spectral']
    lOfDistanceTypesExtended = lOfDistanceTypes + ['compound']
else:
    lOfDistanceTypes = ['frobenius']
    lOfDistanceTypesExtended = lOfDistanceTypes
for distanceType in lOfDistanceTypesExtended:
    distancesDict[distanceType] = {}
    projectedDict[distanceType] = {}
for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
    theseDistances = {}
    for distanceType in lOfDistanceTypesExtended:
        theseDistances[distanceType] = pd.DataFrame(
            np.nan, index=estimatorGroup.index, columns=estimatorGroup.index)
    print('Calculating covariance distances per frequency band; On freq. band {}'.format(freqBandName))
    tIterator = tqdm(estimatorGroup.index, mininterval=30., maxinterval=120.)
    for refIdx in tIterator:
        for testIdx in estimatorGroup.index:
            if refIdx == testIdx:
                for distanceType in lOfDistanceTypesExtended:
                    theseDistances[distanceType].loc[refIdx, testIdx] = 0.
            else:
                theseDistances['frobenius'].loc[refIdx, testIdx] = covMatErrorNorm(
                    estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx], norm='frobenius')
                if calcAllDistances:
                    theseDistances['KL'].loc[refIdx, testIdx] = KL(
                        estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx], recenter=True)
                    theseDistances['spectral'].loc[refIdx, testIdx] = covMatErrorNorm(
                        estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx], norm='spectral')
    if calcAllDistances:
        theseDistances['compound'] = pd.DataFrame(
            0., index=estimatorGroup.index, columns=estimatorGroup.index)
        for distanceType in lOfDistanceTypes:
            theseDistances['compound'] += (theseDistances[distanceType] / np.linalg.norm(theseDistances[distanceType])).to_numpy() / len(lOfDistanceTypes)
    for distanceType in lOfDistanceTypesExtended:
        distancesDict[distanceType][freqBandName] = theseDistances[distanceType].copy()
        distancesDict[distanceType][freqBandName].name = distanceType
        distancesDict[distanceType][freqBandName].to_hdf(resultPath, '/distanceMatrices/{}/{}'.format(freqBandName, distanceType))
        #
        # uMapper = umap.UMAP(metric='precomputed', n_components=2, n_neighbors=2*lastFoldIdx, min_dist=1e-3)
        uMapper = MDS(metric='dissimilarity', n_components=2, max_iter=600, n_init=8)
        projectedDict[distanceType][freqBandName] = pd.DataFrame(
            uMapper.fit_transform(theseDistances[distanceType]),
            index=theseDistances[distanceType].index,
            columns=['umap_{}'.format(fid) for fid in range(2)])
        projectedDict[distanceType][freqBandName].columns.name = 'feature'
        projectedDict[distanceType][freqBandName].name = '{}_umap'.format(distanceType)
        projectedDict[distanceType][freqBandName].to_hdf(resultPath, '/umapDistanceMatrices/{}/{}'.format(freqBandName, distanceType))

colorMaps = {
    # coldish
    'covMat': 'crest_r',
    'points': 'deep',
    # warmish
    'distMat': 'flare_r',
    'distHist': 'muted'
    }
iteratorDescriptions = pd.Series({
    'ca': 'B',
    'cb': 'M',
    'ccs': 'S',
    'ccm': 'C',
    'ra': 'A'
    })

def getLabelFromIters(iter1, iter2):
    sortedIters = sorted([iter1, iter2])
    sortedLabels = [iteratorDescriptions.loc[si] for si in sortedIters]
    return '{}-{}'.format(*sortedLabels)

permuteGroupLabels = pd.unique([getLabelFromIters(it1, it2) for it1, it2 in permutations(iteratorDescriptions.index, 2)])
selfGroupLabels = pd.unique([getLabelFromIters(it1, it2) for it1, it2 in zip(iteratorDescriptions.index, iteratorDescriptions.index)])
groupLabels = np.concatenate([selfGroupLabels, permuteGroupLabels])
groupLabelsDtype = pd.CategoricalDtype(categories=groupLabels, ordered=True)

permuteIters = pd.unique(['{}_{}'.format(*sorted([it1, it2])) for it1, it2 in permutations(iteratorDescriptions.index, 2)])
selfIters = pd.unique(['{}_{}'.format(*sorted([it1, it2])) for it1, it2 in zip(iteratorDescriptions.index, iteratorDescriptions.index)])
groupIters = np.concatenate([selfIters, permuteIters])
groupItersDtype = pd.CategoricalDtype(categories=groupIters, ordered=True)
testRefLookup = pd.Series(groupLabels, index=groupIters, dtype=groupLabelsDtype)

diagonalTerms = ['{}_{}'.format(tn, tn) for tn in listOfIteratorSuffixes]
NACategory = 'Other'
categoryDescriptions = pd.Series({
    'WE': diagonalTerms,
    'deltaM': ['ca_cb'],
    'deltaS': ['ca_ccs', 'cb_ccs'],
    'deltaC': ['cb_ccm', 'ca_ccm'],
    'deltaSS': ['ccm_ccs'],
    NACategory: []
    })
categoryDtype = pd.CategoricalDtype(categories=[catN for catN in categoryDescriptions.index], ordered=True)
categoryLabels = pd.Series({
    'WE': '$WE$',
    'deltaM': '$BE_{\mathbf{M}}$',
    'deltaS': '$BE_{\mathbf{S}}$',
    'deltaC': '$BE_{\mathbf{C}}$',
    'deltaSS': '$BE_{\mathbf{SS}}$',
    NACategory: NACategory
    })
#
pickingColors = False
# singlesPalette = pd.DataFrame(
#     sns.color_palette(colorMaps['points'], 5), columns=['r', 'g', 'b'], index=iteratorDescriptions.to_list())
singlesPalette = pd.DataFrame(
    [sns.color_palette(colorMaps['points'])[ii] for ii in [7, 3, 0, 2, 5]], columns=['r', 'g', 'b'], index=iteratorDescriptions.to_list())
singlesPaletteHLS = singlesPalette.apply(lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']), axis='columns')
if pickingColors:
    sns.palplot(singlesPalette.apply(lambda x: tuple(x), axis='columns'))
    palAx = plt.gca()
    for tIdx, tN in enumerate(singlesPalette.index):
        palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)

for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
    for distanceType in lOfDistanceTypesExtended:
        distanceDF = distancesDict[distanceType][freqBandName]
        indexInfo = distanceDF.index.to_frame().reset_index(drop=True).drop(columns='freqBandName')
        indexInfo.loc[:, 'hasStim'] = indexInfo['iterator'].isin(['ccs', 'ccm'])
        indexInfo.loc[:, 'hasMovement'] = indexInfo['iterator'].isin(['cb', 'ccm'])
        refIndexInfo = indexInfo.copy()
        refIndexInfo.rename(columns=lambda x: 'ref_' + x, inplace=True)
        testIndexInfo = indexInfo.copy()
        testIndexInfo.rename(columns=lambda x: 'test_' + x, inplace=True)
        plotDF = pd.DataFrame(
            distanceDF.to_numpy())
        plotDF.index.name = 'test'
        plotDF.columns.name = 'ref'
        plotDF = plotDF.stack()
        # remove comparisons between self and self
        plotDF = plotDF.loc[plotDF.index.get_level_values('test') != plotDF.index.get_level_values('ref')]
        plotDF = pd.concat([
            testIndexInfo.loc[plotDF.index.get_level_values('test'), :].reset_index(drop=True),
            refIndexInfo.loc[plotDF.index.get_level_values('ref'), :].reset_index(drop=True),
            plotDF.reset_index(drop=True).to_frame(name='distance')
            ], axis='columns')
        groupLookup = plotDF[['test_iterator', 'ref_iterator']].drop_duplicates().reset_index(drop=True)
        groupLookup.loc[:, 'test_label'] = groupLookup['test_iterator'].map(iteratorDescriptions)
        groupLookup.loc[:, 'ref_label'] = groupLookup['ref_iterator'].map(iteratorDescriptions)
        groupLookup.loc[:, 'test_ref'] = (
            groupLookup[['test_iterator', 'ref_iterator']]
            .apply(lambda x: '{}_{}'.format(*sorted(x.to_list())), axis='columns'))
        def sortedLabel(x):
            iterators = sorted(x.to_list())
            label = '{}-{}'.format(*[iteratorDescriptions[it] for it in iterators])
            return label
        groupLookup.loc[:, 'test_ref_label'] = pd.Categorical(
            groupLookup[['test_iterator', 'ref_iterator']]
            .apply(sortedLabel, axis='columns'), dtype=groupLabelsDtype)
        groupLookup.sort_values(by='test_ref_label', inplace=True)
        groupLookup.reset_index(inplace=True, drop=True)
        # groupLookup.loc[:, 'test_ref_label'] = (
        #     groupLookup[['test_iterator', 'ref_iterator']]
        #     .apply(sortedLabel, axis='columns'))
        #
        pairIndex = plotDF[['test_iterator', 'ref_iterator']].apply(lambda x: tuple(x), axis='columns')
        for key in ['test_ref', 'test_ref_label']:
            plotDF.loc[:, key] = pairIndex.map(groupLookup[['test_iterator', 'ref_iterator', key]].set_index(['test_iterator', 'ref_iterator'])[key])
        #
        categoryLookup = groupLookup[['test_ref', 'test_ref_label']].drop_duplicates().reset_index(drop=True)
        for key, value in categoryDescriptions.items():
            categoryLookup.loc[categoryLookup['test_ref'].isin(value), 'category'] = key
        categoryLookup.loc[:, 'category'] = pd.Categorical(categoryLookup['category'].fillna(NACategory), dtype=categoryDtype)
        reOrderList = categoryDescriptions.index.to_list() + [NACategory]
        categoryLookup.sort_values(by=['category', 'test_ref_label'], inplace=True)
        categoryLookup.reset_index(drop=True, inplace=True)
        categoryLookup.loc[:, 'category_label'] = categoryLookup['category'].map(categoryLabels)
        nGroups = categoryLookup.shape[0]
        # rawGroupColors = sns.color_palette(colorMaps['distHist'], nGroups)
        # groupsPalette = pd.DataFrame(
        #     rawGroupColors, columns=['r', 'g', 'b'],
        #     index=categoryLookup['test_ref_label'].to_list())
        rawGroupPalette = sns.color_palette(colorMaps['distHist'])
        # rawGroupColors = {
        #     'B-B': rawGroupPalette[7],
        #     'M-M': rawGroupPalette[7],
        #     'C-C': rawGroupPalette[7],
        #     'S-S': rawGroupPalette[7],
        #     #
        #     'B-M': rawGroupPalette[3],
        #     #
        #     'B-S': rawGroupPalette[0],
        #     'M-S': rawGroupPalette[0],
        #     #
        #     'M-C': rawGroupPalette[2],
        #     'C-S': rawGroupPalette[2],
        #     'B-C': rawGroupPalette[2]
        #     }
        rawGroupColors = {
            'B-B': sns.utils.alter_color(rawGroupPalette[7], l=.6),  # gray
            'M-M': sns.utils.alter_color(rawGroupPalette[7], l=.3),
            'S-S': sns.utils.alter_color(rawGroupPalette[7], l=-.3),
            'C-C': sns.utils.alter_color(rawGroupPalette[7], l=-.6),
            #
            'B-M': rawGroupPalette[3],  # red
            #
            'B-S': rawGroupPalette[0],  # blue
            'M-S': sns.utils.alter_color(rawGroupPalette[4]),  # purple
            #
            'B-C': sns.utils.alter_color(rawGroupPalette[2]),  # green
            'M-C': sns.utils.alter_color(rawGroupPalette[9], l=-.3),  # light blue
            'C-S': sns.utils.alter_color(rawGroupPalette[8], l=-.3),  # yellow
            }
        groupsPalette = pd.DataFrame(
            rawGroupColors).T
        groupsPalette.columns = ['r', 'g', 'b']
        if pickingColors:
            sns.palplot(groupsPalette.apply(lambda x: tuple(x), axis='columns'))
            palAx = plt.gca()
            for tIdx, tN in enumerate(groupsPalette.index):
                palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
        ####
        groupLookup.loc[:, 'color'] = groupLookup['test_ref_label'].astype(np.object).map(groupsPalette.apply(lambda x: tuple(x), axis='columns'))
        categoryLookup.loc[:, 'color'] = categoryLookup['test_ref_label'].astype(np.object).map(groupsPalette.apply(lambda x: tuple(x), axis='columns'))
        # uniqueCats = categoryLookup['category'].unique()
        # nCats = uniqueCats.shape[0]
        # categoryPalette = pd.DataFrame(
        #     sns.color_palette(colorMaps['distHist'], nCats), columns=['r', 'g', 'b'],
        #     index=uniqueCats)
        # if pickingColors:
        #     sns.palplot(categoryPalette.apply(lambda x: tuple(x), axis='columns'))
        #     palAx = plt.gca()
        #     for tIdx, tN in enumerate(categoryPalette.index):
        #         palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
        # categoryLookup.loc[:, 'color'] = categoryLookup['category'].map(categoryPalette.apply(lambda x: tuple(x), axis='columns'))
        break
    break

categoryLookupCopy = categoryLookup.copy()
categoryLookupCopy.loc[:, 'category'] = categoryLookupCopy['category'].astype(np.str)
categoryLookupCopy.loc[:, 'category_label'] = categoryLookupCopy['category_label'].astype(np.str)
categoryLookupCopy.loc[:, 'test_ref_label'] = categoryLookupCopy['test_ref_label'].astype(np.str)
categoryLookupCopy.to_hdf(resultPath, '/categoryLookup')
groupLookupCopy = groupLookup.copy()
groupLookupCopy.loc[:, 'test_ref_label'] = groupLookup['test_ref_label'].astype(np.str)
groupLookupCopy.to_hdf(resultPath, '/groupLookup')
singlesPalette.to_hdf(resultPath, '/singlesPalette')

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}_covMatSimilarity_comparison.pdf'.format(
        expDateTimePathStr, fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
        nSubGroups = estimatorGroup.groupby('iterator').ngroups
        suppressTicks = True
        gridspecKWArgs = {
            'width_ratios': [10, 1], 'wspace': 0.2,
            'height_ratios': [10] * nSubGroups, 'hspace': 0.2}
        axHeightInches = 2.
        avgAxHeight = np.mean(gridspecKWArgs['height_ratios'])
        heightInUnits = np.sum(gridspecKWArgs['height_ratios']) + gridspecKWArgs['hspace'] * avgAxHeight * (nSubGroups - 1)
        avgAxWidth = np.mean(gridspecKWArgs['width_ratios'])
        widthInUnits = np.sum(gridspecKWArgs['width_ratios']) + gridspecKWArgs['wspace'] * avgAxWidth * 1.
        axWidthInches = axHeightInches * widthInUnits / heightInUnits
        cBarExtraWid = (mplRCParams["axes.labelpad"] + 2 * snsRCParams["font.size"] + snsRCParams["axes.titlesize"]) / 72
        totalHeight = axHeightInches / 0.98
        totalWidth = axWidthInches / 0.98 + cBarExtraWid
        fig, ax = plt.subplots(
            nSubGroups, 2,
            figsize=(totalWidth, totalHeight),
            gridspec_kw=gridspecKWArgs,
            )
        middleAxIdx = int(nSubGroups / 2)
        groupCovMat = covMatDF.loc[idxSl[:, freqBandName, lastFoldIdx, :], :]
        groupCovMinVal, groupCovMaxVal = np.percentile(groupCovMat.to_numpy().flatten(), [2.5, 97.5])
        commonHeatmapOpts = dict(
            cmap=colorMaps['covMat'], vmin=groupCovMinVal, vmax=groupCovMaxVal,
            linewidths=0, square=True,
            )
        for axIdx, (iteratorName, _) in enumerate(estimatorGroup.groupby('iterator')):
            maxFoldIdx = covMatDF.loc[idxSl[iteratorName, freqBandName, :, :], :].index.get_level_values('fold').max()
            thisCovMat = covMatDF.loc[idxSl[iteratorName, freqBandName, maxFoldIdx, :], :].copy().clip(lower=groupCovMinVal, upper=groupCovMaxVal)
            thisCovMat.index = thisCovMat.index.get_level_values('feature')
            newTickLabels = thisCovMat.index.to_numpy()
            cBarAx = ax[axIdx, 1]
            mainAx = ax[axIdx, 0]
            if suppressTicks:
                mainAx.set_xticks([])
                mainAx.set_yticks([])
            for tlIdx, tl in enumerate(newTickLabels):
                if suppressTicks:
                    # suppress all
                    newTickLabels[tlIdx] = ""
                else:
                    # plot every n-th label
                    if (tlIdx % int(newTickLabels.shape[0] / 10)) != 0:
                        newTickLabels[tlIdx] = ""
                    else:
                        newTickLabels[tlIdx] = tl
            mask = np.zeros_like(thisCovMat)
            mask[np.triu_indices_from(mask)] = True
            commonHeatmapOpts.update(dict(
                xticklabels=False, yticklabels=False,
                rasterized=True, mask=mask, square=True,
                ))
            if axIdx == middleAxIdx:
                sns.heatmap(thisCovMat, ax=mainAx, cbar_ax=cBarAx, **commonHeatmapOpts)
                cBarAx.set_ylabel('Covariance (a.u.)') #($uV^2$)
                cBarAx.yaxis.set_major_locator(tix.AutoLocator())
                cBarAx.yaxis.set_major_formatter("{x:+3.2f}")
                # mainAx.set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
                # mainAx.set_yticklabels(newTickLabels, rotation=30, ha='right', va='top')
            elif axIdx == (nSubGroups - 1):
                sns.heatmap(thisCovMat, ax=mainAx, cbar=False, **commonHeatmapOpts)
                cBarAx.set_xticks([])
                cBarAx.set_yticks([])
                sns.despine(fig=fig, ax=cBarAx, top=True, bottom=True, left=True, right=True)
                # mainAx.set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
            else:
                sns.heatmap(thisCovMat, ax=mainAx, cbar=False, **commonHeatmapOpts)
                cBarAx.set_xticks([])
                cBarAx.set_yticks([])
                sns.despine(fig=fig, ax=cBarAx, top=True, bottom=True, left=True, right=True)
                # ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
            #
            mainAx.set_title(iteratorDescriptions.loc[iteratorName])
            mainAx.set_xlabel(None)
            mainAx.set_ylabel(None)
        bboxExtraArtists = []
        figXLabel = fig.supxlabel(
            'LFP feature\n({})'.format(prettyNameLookup[freqBandName]),
            x=0.5, y=-0.02, ha='center', va='top')
        bboxExtraArtists.append(figXLabel)
        figYLabel = fig.supylabel(
            'LFP feature ({})'.format(prettyNameLookup[freqBandName]),
            x=-0.02, y=0.5, ha='right', va='center')
        bboxExtraArtists.append(figYLabel)
        # footer = fig.suptitle('LFP feature ({})'.format(prettyNameLookup[freqBandName]), y=0.02, ha='center', va='bottom')
        # bboxExtraArtists.append(footer)
        # fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(
            bbox_inches='tight', pad_inches=0, bbox_extra_artists=bboxExtraArtists)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        ##################################################################################################
        gridspecKWArgs = {
            'width_ratios': [10] * nSubGroups + [1], 'wspace': 0.1,
            'height_ratios': [10], 'hspace': 0.1,
            }
        axWidthInches = 2.
        avgAxWidth = np.mean(gridspecKWArgs['width_ratios'])
        widthInUnits = np.sum(gridspecKWArgs['width_ratios']) + gridspecKWArgs['wspace'] * avgAxWidth * (len(gridspecKWArgs['height_ratios']) - 1)
        avgAxHeight = np.mean(gridspecKWArgs['height_ratios'])
        heightInUnits = np.sum(gridspecKWArgs['height_ratios']) + gridspecKWArgs['hspace'] * avgAxHeight * (len(gridspecKWArgs['width_ratios']) - 1)
        axHeightInches = axWidthInches * heightInUnits / widthInUnits
        cBarExtraWid = (mplRCParams["axes.labelpad"] + 4 * snsRCParams["font.size"] + snsRCParams["axes.labelsize"]) / 72
        totalHeight = axHeightInches / 0.98
        totalWidth = axWidthInches / 0.98 + cBarExtraWid
        fig, ax = plt.subplots(
            len(gridspecKWArgs['height_ratios']),
            len(gridspecKWArgs['width_ratios']),
            figsize=(totalWidth, totalHeight),
            gridspec_kw=gridspecKWArgs,
            )
        groupCovMat = covMatDF.loc[idxSl[:, freqBandName, lastFoldIdx, :], :]
        groupCovMinVal, groupCovMaxVal = np.percentile(groupCovMat.to_numpy().flatten(), [2.5, 97.5])
        commonHeatmapOpts = dict(
            cmap=colorMaps['covMat'], vmin=groupCovMinVal, vmax=groupCovMaxVal,
            linewidths=0, square=True,
            )
        cBarAx = ax[-1]
        for axIdx, (iteratorName, _) in enumerate(estimatorGroup.groupby('iterator')):
            maxFoldIdx = covMatDF.loc[idxSl[iteratorName, freqBandName, :, :], :].index.get_level_values('fold').max()
            thisCovMat = covMatDF.loc[idxSl[iteratorName, freqBandName, maxFoldIdx, :], :].copy().clip(lower=groupCovMinVal, upper=groupCovMaxVal)
            thisCovMat.index = thisCovMat.index.get_level_values('feature')
            newTickLabels = thisCovMat.index.to_numpy()
            mainAx = ax[axIdx]
            if suppressTicks:
                mainAx.set_xticks([])
                mainAx.set_yticks([])
            for tlIdx, tl in enumerate(newTickLabels):
                if suppressTicks:
                    # suppress all
                    newTickLabels[tlIdx] = ""
                else:
                    # plot every n-th label
                    if (tlIdx % int(newTickLabels.shape[0] / 10)) != 0:
                        newTickLabels[tlIdx] = ""
                    else:
                        newTickLabels[tlIdx] = tl
            mask = np.zeros_like(thisCovMat)
            mask[np.triu_indices_from(mask)] = True
            commonHeatmapOpts.update(dict(
                xticklabels=False, yticklabels=False,
                rasterized=True, mask=mask, square=True,
                ))
            if axIdx == (nSubGroups - 1):
                sns.heatmap(thisCovMat, ax=mainAx, cbar_ax=cBarAx, **commonHeatmapOpts)
            else:
                sns.heatmap(thisCovMat, ax=mainAx, cbar=False, **commonHeatmapOpts)
            #
            mainAx.set_title(iteratorDescriptions.loc[iteratorName])
            mainAx.set_xlabel(None)
            mainAx.set_ylabel(None)
        cBarAx.set_ylabel('Covariance (a.u.)')  # ($uV^2$)
        cBarAx.yaxis.set_major_locator(tix.AutoLocator())
        cBarAx.yaxis.set_major_formatter("{x:+3.2f}")
        bboxExtraArtists = []
        figXLabel = fig.supxlabel(
            'LFP feature ({})'.format(prettyNameLookup[freqBandName]),
            x=0.5, y=-0.02, ha='center', va='top')
        bboxExtraArtists.append(figXLabel)
        figYLabel = fig.supylabel(
            'LFP feature\n({})'.format(prettyNameLookup[freqBandName]),
            x=-0.02, y=0.5, ha='right', va='center')
        bboxExtraArtists.append(figYLabel)
        # footer = fig.suptitle('LFP feature ({})'.format(prettyNameLookup[freqBandName]), y=0.02, ha='center', va='bottom')
        # bboxExtraArtists.append(footer)
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(
            bbox_inches='tight', pad_inches=0, bbox_extra_artists=bboxExtraArtists)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        fig, ax = plt.subplots(
            1, nSubGroups, figsize=(2 * nSubGroups, 2), sharey=True)
        #
        for axIdx, (iteratorName, _) in enumerate(estimatorGroup.groupby('iterator')):
            maxFoldIdx = covMatDF.loc[idxSl[iteratorName, freqBandName, :, :], :].index.get_level_values('fold').max()
            #
            thisCovMat = covMatDF.loc[idxSl[iteratorName, freqBandName, maxFoldIdx, :], :].copy()  #.clip(lower=groupCovMinVal, upper=groupCovMaxVal)
            thisCovMat.index = thisCovMat.index.get_level_values('feature')
            thisCovMat.index.name = 'fromFeat'
            thisCovMat.columns = thisCovMat.columns.get_level_values('feature')
            thisCovMat.columns.name = 'toFeat'
            #
            flatCovMat = thisCovMat.where(np.triu(np.ones(thisCovMat.shape)).astype(np.bool)).stack()
            flatCovMat = flatCovMat.to_frame(name='covariance').reset_index()
            flatCovMat.loc[:, 'siteDistance'] = flatCovMat.apply(
                asp.distanceBetweenSites, axis='columns', mapDF=mapDF)
            #
            sdSrs = (
                flatCovMat
                    .loc[flatCovMat['fromFeat'] == flatCovMat['toFeat'], ['fromFeat', 'covariance']]
                    .set_index('fromFeat').apply(np.sqrt).rename(columns={'covariance': 'sd'})['sd'])
            flatCovMat.loc[:, 'correlation'] = flatCovMat.apply(lambda x: x['covariance'] / (sdSrs[x['fromFeat']] * sdSrs[x['toFeat']]), axis='columns')
            sns.scatterplot(
                x='siteDistance', y='covariance',
                data=flatCovMat.loc[flatCovMat['siteDistance'] > 0, :],
                ax=ax[axIdx],
                s=2, color=".2", alpha=0.2, marker = "+",
                )
            sns.lineplot(
                x='siteDistance', y='covariance',
                data=flatCovMat.loc[flatCovMat['siteDistance'] > 0, :],
                errorbar='sd', color=".2",
                ax=ax[axIdx]
                )
            ax[axIdx].set_title(iteratorDescriptions.loc[iteratorName])
            ax[axIdx].set_xlabel('Distance (um)')
            if axIdx == 0:
                ax[-1].set_ylabel('Correlation (a.u.)')
            else:
                ax[axIdx].set_ylabel('')
        #
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(
            bbox_inches='tight', pad_inches=0,
            #bbox_extra_artists=[footer]
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        for distanceType in lOfDistanceTypesExtended:
            distanceDF = distancesDict[distanceType][freqBandName]
            distanceForPlot = distanceDF.copy()
            newIndex = distanceDF.index.get_level_values('iterator').map(iteratorDescriptions)
            # distanceForPlot.index = newIndex
            # distanceForPlot.columns = newIndex
            newTickLabels = newIndex.to_numpy()
            for tlIdx, tl in enumerate(newTickLabels):
                if (tlIdx % lastFoldIdx) != int(lastFoldIdx / 2):
                    newTickLabels[tlIdx] = ''
                else:
                    newTickLabels[tlIdx] = tl.replace(',', ',\n')
            gridspecKWArgs = {
                'height_ratios': [20], 'hspace': 0.1,
                'width_ratios': [20, 1], 'wspace': 0.1}
            axWidthInches = 2.
            avgAxWidth = np.mean(gridspecKWArgs['width_ratios'])
            widthInUnits = np.sum(gridspecKWArgs['width_ratios']) + gridspecKWArgs['wspace'] * avgAxWidth * (len(gridspecKWArgs['height_ratios']) - 1)
            avgAxHeight = np.mean(gridspecKWArgs['height_ratios'])
            heightInUnits = np.sum(gridspecKWArgs['height_ratios']) + gridspecKWArgs['hspace'] * avgAxHeight * (len(gridspecKWArgs['width_ratios']) - 1)
            axHeightInches = axWidthInches * heightInUnits / widthInUnits
            cBarExtraWid = (mplRCParams["axes.labelpad"] + 2 * snsRCParams["font.size"] + snsRCParams["axes.labelsize"]) / 72
            totalHeight = axHeightInches / 0.98
            totalWidth = axWidthInches / 0.98 + cBarExtraWid
            print('distance heatmap size is {} by {}'.format(totalHeight, totalWidth))
            fig, ax = plt.subplots(
                len(gridspecKWArgs['height_ratios']),
                len(gridspecKWArgs['width_ratios']),
                figsize=(totalWidth, totalHeight),
                gridspec_kw=gridspecKWArgs)
            mask = np.zeros_like(distanceForPlot)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(
                distanceForPlot, cmap=colorMaps['distMat'],
                xticklabels=newTickLabels, yticklabels=newTickLabels,
                ax=ax[0], linewidths=0, cbar_ax=ax[1],
                mask=mask
                )
            ax[0].tick_params(left=False, bottom=False)
            accentLineWidth = 0
            inset = 0
            # boxDict = {cN: [] for cN in categoryDescriptions.index}
            # boxAnnDict = {cN: [] for cN in categoryDescriptions.index}
            boxDict = {cN: [] for cN in categoryLookup['test_ref_label'].unique()}
            boxAnnDict = {cN: [] for cN in categoryLookup['test_ref_label'].unique()}
            for rowIdx in range(0, distanceForPlot.shape[0], lastFoldIdx):
                left = rowIdx + inset
                right = rowIdx + (lastFoldIdx - inset)
                test_iter = distanceForPlot.index[left:right].get_level_values('iterator').unique()[0]
                for colIdx in range(0, distanceForPlot.shape[1], lastFoldIdx):
                    top = colIdx + inset
                    bottom = colIdx + (lastFoldIdx - inset)
                    ref_iter = distanceForPlot.columns[top:bottom].get_level_values('iterator').unique()[0]
                    test_ref = '{}_{}'.format(*sorted([test_iter, ref_iter]))
                    thisCat = categoryLookup.set_index('test_ref').loc[test_ref, :]
                    ####################################################################################################
                    #  if thisCat['category'] not in [NACategory]:
                    boxDict[thisCat['test_ref_label']].append(Rectangle((left, bottom), (right - left), (top - bottom)))
                    textArgs = (((left + right) / 2), ((top + bottom) / 2), thisCat['test_ref_label'])
                    textKWArgs = dict(ha='center', va='center', weight='bold')
                    h, l, s = colorsys.rgb_to_hls(*thisCat['color'])
                    textKWArgs['c'] = (0., 0., 0.)if l > 0.5 else (1., 1., 1.)
                    boxAnnDict[thisCat['test_ref_label']].append([textArgs, textKWArgs])
                    ####################################################################################################
            ###
            ax[0].set_ylabel('Epoch')
            ax[0].set_xlabel('Epoch')
            heatMapXLim = ax[0].get_xlim()
            heatMapYLim = ax[0].get_ylim()
            heatMapXWid = heatMapXLim[1] - heatMapXLim[0]
            heatMapYWid = heatMapYLim[1] - heatMapYLim[0]
            # ax[0].set_xticklabels(newTickLabels, rotation=-30, ha='center', va='top')
            # ax[0].set_yticklabels(newTickLabels, rotation=30, ha='center', va='bottom')
            ax[1].set_ylabel('Distance (a.u.)')
            figTitle = fig.suptitle('{} distances ({})'.format(distanceType.capitalize(), prettyNameLookup[freqBandName]))
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0, bbox_extra_artists=[figTitle])
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            drawBoxesAroundGroups = True
            if drawBoxesAroundGroups:
                mapFig, mapAx = plt.subplots(1, 1, figsize=(1., 1.))
                mapAx.set_xlim(heatMapXLim)
                mapAx.set_ylim(heatMapYLim)
                for cN in categoryLookup['test_ref_label'].unique():
                    thisCatIdx = categoryLookup.index[categoryLookup['test_ref_label'] == cN][0]
                    thisCat = categoryLookup.loc[thisCatIdx, :]
                    thisColor = tuple([rgb for rgb in thisCat['color']] + [1.])
                    if len(boxDict[thisCat['test_ref_label']]):
                        for rectIdx, thisRectangle in enumerate(boxDict[thisCat['test_ref_label']]):
                            theseVerts = pd.DataFrame(thisRectangle.get_verts(), columns=['x', 'y'])
                            thisCenter = theseVerts.iloc[:4, :].mean()
                            aboveDiag = (thisCenter['x'] <= thisCenter['y'])
                            if aboveDiag:
                                # Create patch collection with specified colour/alpha
                                pc = PatchCollection(
                                    [thisRectangle], facecolor=thisColor,
                                    edgecolor=(0., 0., 0., 0.), linewidth=accentLineWidth)
                                # Add collection to axes
                                mapAx.add_collection(pc)
                                # Add text
                                textArgs, textKWArgs = boxAnnDict[thisCat['test_ref_label']][rectIdx]
                                mapAx.text(*textArgs, **textKWArgs)
                                # for (textArgs, textKWArgs) in boxAnnDict[thisCat['test_ref_label']]:
                                #     mapAx.text(*textArgs, **textKWArgs)
                mapAx.set_xticks([])
                mapAx.set_yticks([])
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #############################
            indexInfo = distanceDF.index.to_frame().reset_index(drop=True).drop(columns='freqBandName')
            indexInfo.loc[:, 'hasStim'] = indexInfo['iterator'].isin(['ccs', 'ccm'])
            indexInfo.loc[:, 'hasMovement'] = indexInfo['iterator'].isin(['cb', 'ccm'])
            refIndexInfo = indexInfo.copy()
            refIndexInfo.rename(columns=lambda x: 'ref_' + x, inplace=True)
            testIndexInfo = indexInfo.copy()
            testIndexInfo.rename(columns=lambda x: 'test_' + x, inplace=True)
            plotDF = pd.DataFrame(
                distanceDF.to_numpy())
            plotDF.index.name = 'test'
            plotDF.columns.name = 'ref'
            plotDF = plotDF.stack()
            # remove comparisons between self and self
            plotDF = plotDF.loc[plotDF.index.get_level_values('test')!=plotDF.index.get_level_values('ref')]
            plotDF = pd.concat([
                testIndexInfo.loc[plotDF.index.get_level_values('test'), :].reset_index(drop=True),
                refIndexInfo.loc[plotDF.index.get_level_values('ref'), :].reset_index(drop=True),
                plotDF.reset_index(drop=True).to_frame(name='distance')
                ], axis='columns')
            #
            pairIndex = plotDF[['test_iterator', 'ref_iterator']].apply(lambda x: tuple(x), axis='columns')
            for key in ['test_ref', 'test_ref_label']:
                plotDF.loc[:, key] = pairIndex.map(groupLookup[['test_iterator', 'ref_iterator', key]].set_index(['test_iterator', 'ref_iterator'])[key])
            for key in ['category']:
                plotDF.loc[:, key] = plotDF['test_ref'].map(categoryLookup[['test_ref', key]].set_index('test_ref')[key])
            plotDF.loc[:, 'xDummy'] = 0.
            plotDF.drop(columns=['test_ref_label', 'category']).to_hdf(resultPath, '/distanceMatricesForPlot/{}/{}'.format(freqBandName, distanceType))
            normalizedClusterDistDF = plotDF.loc[plotDF['category'] != 'Other', :].copy()
            # normFactor = plotDF.loc[plotDF['category'] == 'WE', 'distance'].median()
            # normalizedClusterDistDF.loc[:, 'distance'] = normalizedClusterDistDF['distance'] / normFactor
            scaler = StandardScaler()
            scaler.fit(plotDF.loc[plotDF['category'] == 'WE', ['distance']])
            normalizedClusterDistDF.loc[:, 'distance'] = scaler.transform(normalizedClusterDistDF.loc[:, ['distance']])
            #
            statsDictGroups = {}
            for catName, catGroup in plotDF.groupby('category'):
                catSubGroups = [sg for sn, sg in catGroup.groupby('test_ref')]
                if len(catSubGroups) > 1:
                    for subGroup1, subGroup2 in combinations(catSubGroups, 2):
                        thisIndex = tuple([catName, subGroup1['test_ref'].unique()[0], subGroup2['test_ref'].unique()[0]])
                        statsDictGroups[thisIndex] = tdr.tTestNadeauCorrection(
                            subGroup1['distance'], subGroup2['distance'], tTestArgs={}, powerAlpha=0.05,
                            test_size=cvIterator.splitter.sampler.test_size)
            groupStatsDF = pd.concat(statsDictGroups, axis='columns', names=['category', 'test_ref1', 'test_ref2'])
            thisPalette = categoryLookup[['test_ref_label', 'color']].drop_duplicates(subset=['test_ref_label']).set_index('test_ref_label')['color']
            height = 3
            width = 1.5
            aspect = width / height
            g = sns.catplot(
                data=plotDF, height=height, aspect=aspect,
                y='distance', x='xDummy',
                hue='test_ref_label',
                palette=thisPalette.to_dict(),
                hue_order=thisPalette.index.to_list(),
                kind='box', whis=np.inf)
            g.set_axis_labels("", "Distance (a.u.)")
            g.axes[0, 0].set_xticks([])
            g.suptitle('{} distances ({})'.format(distanceType.capitalize(), prettyNameLookup[freqBandName]))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            asp.reformatFacetGridLegend(
                g, titleOverrides={'test_ref_label': 'Epoch pair'},
                contentOverrides={},
                styleOpts=styleOpts)
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            plotByCat = False
            if plotByCat:
                statsDictCats = {}
                catSubGroups = [sg for sn, sg in plotDF.groupby('category')]
                if len(catSubGroups) > 1:
                    for subGroup1, subGroup2 in combinations(catSubGroups, 2):
                        thisIndex = tuple([subGroup1['category'].unique()[0], subGroup2['category'].unique()[0]])
                        statsDictCats[thisIndex] = tdr.tTestNadeauCorrection(
                            subGroup1['distance'], subGroup2['distance'], tTestArgs={}, powerAlpha=0.05,
                            test_size=cvIterator.splitter.sampler.test_size)
                catStatsDF = pd.concat(statsDictCats, axis='columns', names=['category1', 'category2'])
                thisPalette = (
                    categoryLookup
                        .loc[categoryLookup['category'].isin(normalizedClusterDistDF['category']), ['category', 'color', 'category_label']]
                        .drop_duplicates(subset=['category'])
                        .set_index('category')[['color', 'category_label']])
                height = 3
                width = 1.5
                aspect = width / height
                g = sns.catplot(
                    data=normalizedClusterDistDF,
                    y='distance', x='xDummy', hue='category', height=height, aspect=aspect,
                    palette=thisPalette['color'].to_dict(), hue_order=thisPalette.index.to_list(),
                    kind='box', whis=np.inf)
                xLim = g.axes[0, 0].get_xlim()
                xLen = xLim[1] - xLim[0]
                thisPalette.loc[:, 'x'] = xLim[0] + (1 + np.arange(thisPalette.shape[0])) * xLen / (thisPalette.shape[0] + 1)
                thisPalette.loc[:, 'y'] = normalizedClusterDistDF.groupby('category').max()['distance']
                thisPalette.loc[:, 'yOff'] = normalizedClusterDistDF.groupby('category').std()['distance']
                for cN in [('WE', 'deltaM'), ('WE', 'deltaS'), ('WE', 'deltaC')]:
                    theseStats = catStatsDF.loc[:, cN]
                    if theseStats['p-val'] < 1e-3:
                        pointsToPlot = thisPalette.loc[cN, ['x', 'y']].copy()
                        pointsToPlot.loc[:, 'y'] = pointsToPlot['y'].max() + float(thisPalette.loc[cN, ['yOff']].max() / 2)
                        asp.annotateLine(
                            pointsToPlot, g.axes[0, 0],
                            x_var='x', y_var='y',
                            offsets=None, plotKWArgs=dict(c='k'),
                            textLocation='average', text='*', textKWArgs={})
                g.set_axis_labels("", "Distance (a.u.)")
                g.axes[0, 0].set_xticks([])
                g.suptitle('{} distances ({})'.format(distanceType.capitalize(), prettyNameLookup[freqBandName]))
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                asp.reformatFacetGridLegend(
                    g, titleOverrides={'test_ref_label': 'Epoch pair'},
                    contentOverrides=thisPalette['category_label'],
                    styleOpts=styleOpts)
                g.resize_legend(adjust_subtitles=True)
                pdf.savefig(bbox_inches='tight', pad_inches=0)
                # normalizedClusterDistDF.to_hdf(resultPath, '/distanceMatricesForPlotNormalized/{}/{}'.format(freqBandName, distanceType))
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            ################################################################################################
            umapDF = projectedDict[distanceType][freqBandName]
            umapPlotDF = umapDF.reset_index()
            umapPlotDF.loc[:, 'iterator_label'] = umapPlotDF['iterator'].map(iteratorDescriptions)
            thisPalette = (
                singlesPalette
                    .loc[singlesPalette.index.isin(umapPlotDF['iterator_label'])]
                    .apply(lambda x: tuple(x), axis='columns'))
            #
            x_var, y_var = 'umap_0', 'umap_1'
            height = 3
            width = 1.5
            aspect = width / height
            g = sns.relplot(
                x=x_var, y=y_var, kind='scatter', height=height, aspect=aspect,
                edgecolor=None, alpha=0.6,
                hue='iterator_label', hue_order=thisPalette.index,
                palette=thisPalette.to_dict(), data=umapPlotDF)
            meanPositions = umapPlotDF.groupby('iterator').mean()[umapDF.columns]
            stdPositions = umapPlotDF.groupby('iterator').std()[umapDF.columns]
            for test_iter, ref_iter in combinations(meanPositions.index, 2):
                test_ref = '{}_{}'.format(*sorted([test_iter, ref_iter]))
                thisCat = categoryLookup.set_index('test_ref').loc[test_ref, :]
                if thisCat['test_ref_label'] in ['B-M', 'B-S', 'B-C', 'M-S', 'M-C', 'C-S']:
                    pointsToPlot = meanPositions.loc[[test_iter, ref_iter], :].copy()
                    thisColor = tuple([rgb for rgb in thisCat['color']] + [0.6])
                    plotKWArgs = dict(color=thisColor, linewidth=snsRCParams['lines.markersize'] / 2)
                    offsets = 3 * stdPositions.max(axis='columns')
                    textLocation = 'average'
                    text = '   {}'.format(thisCat['test_ref_label'])
                    textKWArgs = dict(c=thisColor, ha='left', va='bottom')
                    asp.annotateLine(
                        pointsToPlot, g.axes[0, 0],
                        offsets=offsets,
                        x_var=x_var, y_var=y_var,
                        plotKWArgs=plotKWArgs,
                        textLocation=textLocation, text=text, textKWArgs=textKWArgs)
            g.suptitle('MDS projection based on\n{} distance ({})'.format(distanceType.capitalize(), prettyNameLookup[freqBandName]))
            for ax in g.axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            g.despine(top=True, right=True, left=True, bottom=True)
            g.set_axis_labels("", "")
            asp.reformatFacetGridLegend(
                g, titleOverrides={'iterator_label': 'Epoch'},
                contentOverrides=iteratorDescriptions,
                styleOpts=styleOpts)
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
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
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight')
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()