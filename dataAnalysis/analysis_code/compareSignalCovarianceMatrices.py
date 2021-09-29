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
listOfIteratorSuffixes =[x.strip() for x in arguments['iteratorSuffixList'].split(',')]
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
#
del covMatDict, eVSDict, estimatorsDict
#
distancesDict = {}
projectedDict = {}
lOfDistanceTypes = ['KL', 'frobenius', 'spectral']
for distanceType in lOfDistanceTypes + ['compound']:
    distancesDict[distanceType] = {}
    projectedDict[distanceType] = {}
for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
    theseDistances = {}
    for distanceType in lOfDistanceTypes:
        theseDistances[distanceType] = pd.DataFrame(
            np.nan, index=estimatorGroup.index, columns=estimatorGroup.index)
    print('Calculating covariance distances per frequency band; On freq. band {}'.format(freqBandName))
    tIterator = tqdm(estimatorGroup.index, mininterval=30., maxinterval=120.)
    for refIdx in tIterator:
        for testIdx in estimatorGroup.index:
            if refIdx == testIdx:
                for distanceType in ['KL', 'frobenius', 'spectral']:
                    theseDistances[distanceType].loc[refIdx, testIdx] = 0.
            else:
                theseDistances['KL'].loc[refIdx, testIdx] = KL(
                    estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx], recenter=True)
                theseDistances['frobenius'].loc[refIdx, testIdx] = covMatErrorNorm(
                    estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx], norm='frobenius')
                theseDistances['spectral'].loc[refIdx, testIdx] = covMatErrorNorm(
                    estimatorGroup.loc[refIdx], estimatorGroup.loc[testIdx], norm='spectral')
    theseDistances['compound'] = pd.DataFrame(
        0., index=estimatorGroup.index, columns=estimatorGroup.index)
    for distanceType in lOfDistanceTypes:
        theseDistances['compound'] += (theseDistances[distanceType] / np.linalg.norm(theseDistances[distanceType])).to_numpy() / len(lOfDistanceTypes)
    for distanceType in lOfDistanceTypes + ['compound']:
        distancesDict[distanceType][freqBandName] = theseDistances[distanceType].copy()
        distancesDict[distanceType][freqBandName].name = distanceType
        distancesDict[distanceType][freqBandName].to_hdf(resultPath, '/distanceMatrices/{}/{}'.format(freqBandName, distanceType))
        #
        # uMapper = umap.UMAP(metric='precomputed', n_neighbors=3*lastFoldIdx, min_dist=0.5)
        uMapper = MDS(metric='dissimilarity', n_components=2)
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
    'points': 'mako',
    # warmish
    'distMat': 'flare_r',
    'distHist': 'rocket'
    }
iteratorDescriptions = pd.Series({
    'ca': 'B',
    'cb': 'M',
    'ccs': 'S',
    'ccm': 'SM',
    'ra': 'A'
    })
diagonalTerms = ['{}_{}'.format(tn, tn) for tn in listOfIteratorSuffixes]
categoryDescriptions = pd.Series({
    'WE': diagonalTerms,
    'deltaM': ['ca_cb'],
    'deltaS': ['cb_ccm', 'ca_ccs'],
    'deltaSM': ['ccm_ccs'],
    })
categoryLabels = pd.Series({
    'WE': '$WE$',
    'deltaM': '$BE_{\mathbf{M}}$',
    'deltaS': '$BE_{\mathbf{S}}$',
    'deltaSM': '$BE_{\mathbf{SM}}$',
    })
NACategory = 'Other'
#
pickingColors = False
singlesPalette = pd.DataFrame(
    sns.color_palette(colorMaps['points'], 5), columns=['r', 'g', 'b'], index=iteratorDescriptions.to_list())
singlesPaletteHLS = singlesPalette.apply(lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']), axis='columns')
if pickingColors:
    sns.palplot(singlesPalette.apply(lambda x: tuple(x), axis='columns'))
    palAx = plt.gca()
    for tIdx, tN in enumerate(singlesPalette.index):
        palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
    for distanceType in ['frobenius']:
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
        plotDF = plotDF.loc[plotDF.index.get_level_values('test')!=plotDF.index.get_level_values('ref')]
        plotDF = pd.concat([
            testIndexInfo.loc[plotDF.index.get_level_values('test'), :].reset_index(drop=True),
            refIndexInfo.loc[plotDF.index.get_level_values('ref'), :].reset_index(drop=True),
            plotDF.reset_index(drop=True).to_frame(name='distance')
            ], axis='columns')
        groupsLookup = plotDF[['test_iterator', 'ref_iterator']].drop_duplicates().reset_index(drop=True)
        groupsLookup.loc[:, 'test_label'] = groupsLookup['test_iterator'].map(iteratorDescriptions)
        groupsLookup.loc[:, 'ref_label'] = groupsLookup['ref_iterator'].map(iteratorDescriptions)
        groupsLookup.loc[:, 'test_ref'] = (
            groupsLookup[['test_iterator', 'ref_iterator']]
            .apply(lambda x: '{}_{}'.format(*sorted(x.to_list())), axis='columns'))
        def sortedLabel(x):
            iterators = sorted(x.to_list())
            label = '{}-{}'.format(*[iteratorDescriptions[it] for it in iterators])
            return label
        groupsLookup.loc[:, 'test_ref_label'] = (
            groupsLookup[['test_iterator', 'ref_iterator']]
            .apply(sortedLabel, axis='columns'))
        #
        pairIndex = plotDF[['test_iterator', 'ref_iterator']].apply(lambda x: tuple(x), axis='columns')
        for key in ['test_ref', 'test_ref_label']:
            plotDF.loc[:, key] = pairIndex.map(groupsLookup[['test_iterator', 'ref_iterator', key]].set_index(['test_iterator', 'ref_iterator'])[key])
        #
        categoryLookup = groupsLookup[['test_ref', 'test_ref_label']].drop_duplicates().reset_index(drop=True)
        for key, value in categoryDescriptions.items():
            categoryLookup.loc[categoryLookup['test_ref'].isin(value), 'category'] = key
        categoryLookup.loc[:, 'category'].fillna(NACategory, inplace=True)
        reOrderList = categoryDescriptions.index.to_list() + [NACategory]
        categoryLookup.loc[:, 'category_num'] = categoryLookup['category'].apply(lambda x: reOrderList.index(x))
        categoryLookup.sort_values(by=['category_num', 'test_ref'], inplace=True)
        categoryLookup.reset_index(drop=True, inplace=True)
        categoryLookup.loc[:, 'group_num'] = categoryLookup.index.to_numpy()
        categoryLookup.loc[:, 'category_label'] = categoryLookup['category'].map(categoryLabels)
        nGroups = categoryLookup.shape[0]
        rawGroupColors = sns.color_palette(colorMaps['distHist'], nGroups)
        groupsPalette = pd.DataFrame(
            rawGroupColors, columns=['r', 'g', 'b'],
            index=categoryLookup['test_ref_label'].to_list())
        if pickingColors:
            sns.palplot(groupsPalette.apply(lambda x: tuple(x), axis='columns'))
            palAx = plt.gca()
            for tIdx, tN in enumerate(groupsPalette.index):
                palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
        ####
        groupsLookup.loc[:, 'color'] = groupsLookup['test_ref_label'].map(groupsPalette.apply(lambda x: tuple(x), axis='columns'))
        uniqueCats = categoryLookup['category'].unique()
        nCats = uniqueCats.shape[0]
        categoryPalette = pd.DataFrame(
            sns.color_palette(colorMaps['distHist'], nCats), columns=['r', 'g', 'b'],
            index=uniqueCats)
        if pickingColors:
            sns.palplot(categoryPalette.apply(lambda x: tuple(x), axis='columns'))
            palAx = plt.gca()
            for tIdx, tN in enumerate(categoryPalette.index):
                palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
        categoryLookup.loc[:, 'color'] = categoryLookup['category'].map(categoryPalette.apply(lambda x: tuple(x), axis='columns'))
        break
    break

def annotateLine(
        pointsToPlot, ax,
        x_var=None, y_var=None,
        offsets=None, plotKWArgs=None,
        textLocation=None, text=None, textKWArgs={}):
    left, right = pointsToPlot[x_var].idxmin(), pointsToPlot[x_var].idxmax()
    bottom, top = pointsToPlot[y_var].idxmin(), pointsToPlot[y_var].idxmax()
    p1 = pointsToPlot.loc[left, :].to_numpy()
    p1 = np.append(p1, [0])
    p2 = pointsToPlot.loc[right, :].to_numpy()
    p2 = np.append(p2, [0])
    deltaPDir = vg.normalize(p2 - p1)
    if offsets is not None:
        p1 = p1 + offsets[left] * deltaPDir
        p2 = p2 - offsets[right] * deltaPDir
    pointsToPlot.loc[left, :] = p1[:2]
    pointsToPlot.loc[right, :] = p2[:2]
    ax.plot(
        pointsToPlot[x_var], pointsToPlot[y_var], **plotKWArgs,
        )
    if text is not None:
        if textLocation == 'average':
            xpos, ypos = pointsToPlot[x_var].mean(), pointsToPlot[y_var].mean()
        else:
            xpos, ypos = textLocation
        ax.text(xpos, ypos, text, **textKWArgs)
    return

def distanceBetweenSites(
        inputSrs=None, mapDF=None,
        spacing=1., xSpacing=1., ySpacing=1.):
    # fromFeat = 'utah1'
    # toFeat = 'utah10'
    dX = xSpacing * (mapDF.loc[inputSrs['fromFeat'], 'xCoords'] - mapDF.loc[inputSrs['toFeat'], 'xCoords'])
    dY = ySpacing * (mapDF.loc[inputSrs['fromFeat'], 'yCoords'] - mapDF.loc[inputSrs['toFeat'], 'yCoords'])
    distance = np.sqrt(dX**2 + dY**2)
    return distance

pdfPath = os.path.join(
    figureOutputFolder, '{}_covMatSimilarity_comparison.pdf'.format(
        fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    for freqBandName, estimatorGroup in estimatorsSrs.groupby('freqBandName'):
        nSubGroups = estimatorGroup.groupby('iterator').ngroups
        fig, ax = plt.subplots(
            1, nSubGroups + 1, figsize=(2 * nSubGroups, 2),
            gridspec_kw={
                'width_ratios': [10 for aaa in range(nSubGroups)] + [1],
                'wspace': 0.1})
        groupCovMat = covMatDF.loc[idxSl[:, freqBandName, lastFoldIdx, :], :]
        groupCovMinVal, groupCovMaxVal = np.percentile(groupCovMat.to_numpy().flatten(), [2.5, 97.5])
        commonHeatmapOpts = dict(
            cmap=colorMaps['covMat'], vmin=groupCovMinVal, vmax=groupCovMaxVal,
            linewidths=0
            )
        suppressTicks = True
        for axIdx, (iteratorName, _) in enumerate(estimatorGroup.groupby('iterator')):
            maxFoldIdx = covMatDF.loc[idxSl[iteratorName, freqBandName, :, :], :].index.get_level_values('fold').max()
            thisCovMat = covMatDF.loc[idxSl[iteratorName, freqBandName, maxFoldIdx, :], :].copy().clip(lower=groupCovMinVal, upper=groupCovMaxVal)
            thisCovMat.index = thisCovMat.index.get_level_values('feature')
            newTickLabels = thisCovMat.index.to_numpy()
            if suppressTicks:
                ax[axIdx].set_xticks([])
                ax[axIdx].set_yticks([])
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
            commonHeatmapOpts.update(dict(
                xticklabels=False, yticklabels=False,
                rasterized=True,
                ))
            if axIdx == 0:
                sns.heatmap(thisCovMat, ax=ax[axIdx], cbar_ax=ax[-1], **commonHeatmapOpts)
                ax[axIdx].set_ylabel('LFP feature')
                # ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
                # ax[axIdx].set_yticklabels(newTickLabels, rotation=30, ha='right', va='top')
            elif axIdx == (nSubGroups - 1):
                sns.heatmap(thisCovMat, ax=ax[axIdx], cbar=False, **commonHeatmapOpts)
                ax[axIdx].set_ylabel('')
                # ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
            else:
                sns.heatmap(thisCovMat, ax=ax[axIdx], cbar=False, **commonHeatmapOpts)
                ax[axIdx].set_ylabel('')
                # ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
            #
            ax[axIdx].set_title(iteratorDescriptions.loc[iteratorName])
            ax[axIdx].set_xlabel('')
        ax[-1].set_ylabel('Covariance ($uV^2$)')
        footer = fig.suptitle('LFP feature (frequency band: {})'.format(freqBandName), y=0.02, ha='center', va='bottom')
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(
            bbox_inches='tight', pad_inches=0, bbox_extra_artists=[footer])
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
                distanceBetweenSites, axis='columns', mapDF=mapDF)
            #
            sdSrs = (
                flatCovMat
                    .loc[flatCovMat['fromFeat'] == flatCovMat['toFeat'], ['fromFeat', 'covariance']]
                    .set_index('fromFeat').apply(np.sqrt).rename(columns={'covariance': 'sd'})['sd'])
            flatCovMat.loc[:, 'correlation'] = flatCovMat.apply(lambda x: x['covariance'] / (sdSrs[x['fromFeat']] * sdSrs[x['toFeat']]), axis='columns')
            sns.scatterplot(
                x='siteDistance', y='correlation',
                data=flatCovMat.loc[flatCovMat['siteDistance'] > 0, :],
                ax=ax[axIdx],
                s=2, color=".2", alpha=0.2, marker = "+",
                )
            sns.lineplot(
                x='siteDistance', y='correlation',
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
        for distanceType in ['frobenius']:
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
            fig, ax = plt.subplots(
                1, 2, figsize=(3, 3),
                gridspec_kw={
                    'width_ratios': [20, 1],
                    'wspace': 0.1})
            sns.heatmap(
                distanceForPlot, cmap=colorMaps['distMat'],
                xticklabels=newTickLabels, yticklabels=newTickLabels,
                ax=ax[0], linewidths=0, cbar_ax=ax[1]
                )
            ###
            accentLineWidth = 0
            inset = 0
            boxDict = {cN: [] for cN in categoryDescriptions.index}
            boxAnnDict = {cN: [] for cN in categoryDescriptions.index}
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
                    if thisCat['category'] not in [NACategory]:
                        boxDict[thisCat['category']].append(Rectangle((left, bottom), (right - left), (top - bottom)))
                        textArgs = (((left + right)/ 2), ((top + bottom) / 2), thisCat['test_ref_label'])
                        textKWArgs = dict(ha='center', va='center', weight='bold')
                        h, l, s = colorsys.rgb_to_hls(*thisCat['color'])
                        textKWArgs['c'] = (0., 0., 0.)if l > 0.5 else (1., 1., 1.)
                        boxAnnDict[thisCat['category']].append([textArgs, textKWArgs])
            ###
            ax[0].set_ylabel('Epoch')
            ax[0].set_xlabel('Epoch')
            # ax[0].set_xticklabels(newTickLabels, rotation=-30, ha='center', va='top')
            # ax[0].set_yticklabels(newTickLabels, rotation=30, ha='center', va='bottom')
            ax[1].set_ylabel('Distance (a.u.)')
            fig.suptitle('{} covariance distances (frequency band: {})'.format(distanceType, freqBandName))
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            drawBoxesAroundGroups = True
            if drawBoxesAroundGroups:
                mapFig, mapAx = plt.subplots(1, 1, figsize=(1.5, 1.5))
                mapAx.set_xlim(ax[0].get_xlim())
                mapAx.set_ylim(ax[0].get_ylim())
                for cN in categoryDescriptions.index:
                    thisCatIdx = categoryLookup.index[categoryLookup['category'] == cN][0]
                    thisCat = categoryLookup.loc[thisCatIdx, :]
                    thisColor = tuple([rgb for rgb in thisCat['color']] + [1.])
                    if len(boxDict[thisCat['category']]):
                        # Create patch collection with specified colour/alpha
                        pc = PatchCollection(
                            boxDict[thisCat['category']], facecolor=thisColor,
                            edgecolor=(0., 0., 0., 0.), linewidth=accentLineWidth)
                        # Add collection to axes
                        mapAx.add_collection(pc)
                        # Add text
                        for (textArgs, textKWArgs) in boxAnnDict[thisCat['category']]:
                            mapAx.text(*textArgs, **textKWArgs)
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
                plotDF.loc[:, key] = pairIndex.map(groupsLookup[['test_iterator', 'ref_iterator', key]].set_index(['test_iterator', 'ref_iterator'])[key])
            for key in ['category', 'category_num', 'group_num']:
                plotDF.loc[:, key] = plotDF['test_ref'].map(categoryLookup[['test_ref', key]].set_index('test_ref')[key])
            plotDF.loc[:, 'xDummy'] = 0.
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
            thisPalette = groupsLookup[['test_ref_label', 'color']].set_index('test_ref_label')['color'].drop_duplicates()
            g = sns.catplot(
                data=plotDF, height=2, aspect=1,
                y='distance', x='xDummy',
                hue='test_ref_label',
                palette=thisPalette.to_dict(),
                hue_order=categoryLookup['test_ref_label'],
                kind='violin', cut=0)
            g.set_axis_labels("", "Distance (a.u.)")
            g.axes[0, 0].set_xticks([])
            g.suptitle('{} distance (frequency band: {})'.format(distanceType, freqBandName))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            normFactor = plotDF.loc[plotDF['category'] == 'WE', 'distance'].median()
            normalizedClusterDistDF = plotDF.loc[plotDF['category'] != 'Other', :].copy()
            normalizedClusterDistDF.loc[:, 'distance'] = normalizedClusterDistDF['distance'] / normFactor
            #
            statsDictCats = {}
            catSubGroups = [sg for sn, sg in plotDF.groupby('category')]
            for subGroup1, subGroup2 in combinations(catSubGroups, 2):
                thisIndex = tuple([subGroup1['category'].unique()[0], subGroup2['category'].unique()[0]])
                statsDictCats[thisIndex] = tdr.tTestNadeauCorrection(
                    subGroup1['distance'], subGroup2['distance'], tTestArgs={}, powerAlpha=0.05,
                    test_size=cvIterator.splitter.sampler.test_size)
            catStatsDF = pd.concat(statsDictCats, axis='columns', names=['category1', 'category2'])
            thisPalette = (
                categoryLookup
                    .loc[categoryLookup['category'].isin(normalizedClusterDistDF['category']), ['category', 'color', 'category_label']]
                    .set_index('category')[['color', 'category_label']].drop_duplicates())
            g = sns.catplot(
                data=normalizedClusterDistDF,
                y='distance', x='xDummy', hue='category', height=2, aspect=1,
                palette=thisPalette['color'].to_dict(), hue_order=thisPalette.index.to_list(),
                kind='violin', cut=0)
            xLim = g.axes[0, 0].get_xlim()
            xLen = xLim[1] - xLim[0]
            thisPalette.loc[:, 'x'] = xLim[0] + (1 + np.arange(thisPalette.shape[0])) * xLen / (thisPalette.shape[0] + 1)
            thisPalette.loc[:, 'y'] = normalizedClusterDistDF.groupby('category').max()['distance']
            thisPalette.loc[:, 'yOff'] = normalizedClusterDistDF.groupby('category').std()['distance']
            for cN in [('WE', 'deltaM'), ('WE', 'deltaS'), ('WE', 'deltaSM')]:
                theseStats = catStatsDF.loc[:, cN]
                if theseStats['p-val'] < 1e-3:
                    pointsToPlot = thisPalette.loc[cN, ['x', 'y']].copy()
                    pointsToPlot.loc[:, 'y'] = pointsToPlot['y'].max() + float(thisPalette.loc[cN, ['yOff']].max() / 2)
                    annotateLine(
                        pointsToPlot, g.axes[0, 0],
                        x_var='x', y_var='y',
                        offsets=None, plotKWArgs=dict(c='k'),
                        textLocation='average', text='*', textKWArgs={})
            g.set_axis_labels("", "Distance (a.u.)")
            g.axes[0, 0].set_xticks([])
            g.suptitle('Clusterability of {} distance (frequency band: {})'.format(distanceType, freqBandName))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            asp.reformatFacetGridLegend(
                g, titleOverrides={'category': 'Distance group'},
                contentOverrides=thisPalette['category_label'],
                styleOpts=styleOpts)
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ################################################################################################
            umapDF = projectedDict[distanceType][freqBandName]
            umapPlotDF = umapDF.reset_index()
            umapPlotDF.loc[:, 'iterator_label'] = umapPlotDF['iterator'].map(iteratorDescriptions)
            thisPalette = singlesPalette.apply(lambda x: tuple(x), axis='columns')
            x_var, y_var = 'umap_0', 'umap_1'
            g = sns.relplot(
                x=x_var, y=y_var, kind='scatter', height=3, aspect=1,
                edgecolor=None, alpha=0.6,
                hue='iterator_label', hue_order=thisPalette.index,
                palette=thisPalette.to_dict(), data=umapPlotDF)
            meanPositions = umapPlotDF.groupby('iterator').mean()[umapDF.columns]
            stdPositions = umapPlotDF.groupby('iterator').std()[umapDF.columns]
            for test_iter, ref_iter in combinations(meanPositions.index, 2):
                test_ref = '{}_{}'.format(*sorted([test_iter, ref_iter]))
                thisCat = categoryLookup.set_index('test_ref').loc[test_ref, :]
                if thisCat['category'] in ['deltaS', 'deltaSM', 'deltaM']:
                    pointsToPlot = meanPositions.loc[[test_iter, ref_iter], :].copy()
                    thisColor = tuple([rgb for rgb in thisCat['color']] + [0.6])
                    plotKWArgs = dict(color=thisColor, linewidth=snsRCParams['lines.markersize'] / 2)
                    offsets = 3 * stdPositions.max(axis='columns')
                    textLocation = 'average'
                    text = '   {}'.format(thisCat['category_label'])
                    textKWArgs = dict(c=thisColor, ha='left', va='bottom')
                    annotateLine(
                        pointsToPlot, g.axes[0, 0],
                        offsets=offsets,
                        x_var=x_var, y_var=y_var,
                        plotKWArgs=plotKWArgs,
                        textLocation=textLocation, text=text, textKWArgs=textKWArgs)
            g.suptitle('MDS projection of {} distance (frequency band: {})'.format(distanceType, freqBandName))
            for ax in g.axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            g.despine(top=True, right=True, left=True, bottom=True)
            g.set_axis_labels("", "")
            asp.reformatFacetGridLegend(
                g, titleOverrides={'iterator_label': 'Epoch'},
                contentOverrides=iteratorDescriptions,
                styleOpts=styleOpts)
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