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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os, sys
from sklearn.pipeline import make_pipeline, Pipeline
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
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
from tqdm import tqdm
import umap.plot
rng = default_rng()
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
    context='paper', style='whitegrid',
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
        'plotting': True, 'datasetPrefix': 'Block_XL_df', 'iteratorSuffixList': 'ca, cb, ccm, ccs',
        'debugging': False, 'estimatorName': 'mahal', 'showFigures': False}
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
for iteratorSuffix in listOfIteratorSuffixes:
    datasetName = arguments['datasetPrefix'] + '_{}'.format(iteratorSuffix)
    thisEstimatorName = '{}_{}_{}'.format(
        estimatorName, datasetName, selectionName)
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
    print('On {}'.format(freqBandName))
    for refIdx in tqdm(estimatorGroup.index):
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
        uMapper = umap.UMAP(metric='precomputed', n_neighbors=2*lastFoldIdx)
        uMapper.fit(theseDistances[distanceType])
        projectedDict[distanceType][freqBandName] = pd.DataFrame(
            uMapper.transform(theseDistances[distanceType]), index=theseDistances[distanceType].index,
            columns=['umap_{}'.format(fid) for fid in range(2)])
        projectedDict[distanceType][freqBandName].columns.name = 'feature'
        projectedDict[distanceType][freqBandName].name = '{}_umap'.format(distanceType)
        projectedDict[distanceType][freqBandName].to_hdf(resultPath, '/umapDistanceMatrices/{}/{}'.format(freqBandName, distanceType))


primaryPalette = pd.DataFrame(sns.color_palette('colorblind'), columns=['r', 'g', 'b'])
pickingColors = False
if pickingColors:
    sns.palplot(primaryPalette.apply(lambda x: tuple(x), axis='columns'))
    palAx = plt.gca()
    for tIdx, tN in enumerate(primaryPalette.index):
        palAx.text(tIdx, .5, '{}'.format(tN))
rgb = pd.DataFrame(
    primaryPalette.iloc[[1, 0, 2, 4, 7, 3], :].to_numpy(),
    columns=['r', 'g', 'b'], index=['v', 'a', 'r', 'ens', 'prediction', 'outsideStim'])
hls = rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']), axis='columns')
hls.loc['a*r', :] = hls.loc[['a', 'r'], :].mean()
hls.loc['v*r', :] = hls.loc[['v', 'r'], :].mean()
hls.loc['v*a', :] = hls.loc[['v', 'v', 'a'], :].mean()
hls.loc['v*a*r', :] = hls.loc[['v', 'a', 'r'], :].mean()
for sN in ['a*r', 'v*r', 'v*a', 'v*a*r']:
    hls.loc[sN, 's'] = hls.loc[sN, 's'] * 0.75
    hls.loc[sN, 'l'] = hls.loc[sN, 'l'] * 1.2
hls.loc['v*a*r', 's'] = hls.loc['v*a*r', 's'] * 0.5
hls.loc['v*a*r', 'l'] = hls.loc['v*a*r', 'l'] * 1.5
hls.loc['ground_truth', :] = hls.loc['prediction', :]
hls.loc['ground_truth', 'l'] = hls.loc['prediction', 'l'] * 0.25
primarySourcePalette = hls.apply(lambda x: pd.Series(colorsys.hls_to_rgb(*x), index=['r', 'g', 'b']), axis='columns')
sourcePalette = primarySourcePalette.apply(lambda x: tuple(x), axis='columns')
sourcePalette.loc['ca'] = sourcePalette['prediction'] # no stim no movement
sourcePalette.loc['cb'] = sourcePalette['v'] # no stim, movement
sourcePalette.loc['ccs'] = sourcePalette['a*r'] # stim, no movement
sourcePalette.loc['ccm'] = sourcePalette['v*a*r'] # stim, movement
sourcePalette.loc['withinStim'] = sourcePalette['a']
sourcePalette.loc['acrossStim'] = sourcePalette['r']
keepEntries = ['ca', 'cb', 'ccs', 'ccm', 'withinStim', 'acrossStim', 'outsideStim']
sourcePalette = sourcePalette.loc[keepEntries].to_frame(name='color')
iteratorDescriptions = {
    'ca':  'no stim, no movement',
    'cb':  'no stim, movement',
    'ccs': 'stim, no movement',
    'ccm': 'stim, movement',
    }
sourcePalette.loc[:, 'description'] = pd.Series(iteratorDescriptions)
if pickingColors:
    sns.palplot(sourcePalette['color'], size=sourcePalette.shape[0])
    palAx = plt.gca()
    for tIdx, tN in enumerate(sourcePalette.index):
        palAx.text(tIdx, .5, '{}'.format(tN))
sourcePalette.to_hdf(resultPath, 'plotOpts')
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
            cmap='mako', vmin=groupCovMinVal, vmax=groupCovMaxVal,
            linewidths=0
            )
        for axIdx, (iteratorName, _) in enumerate(estimatorGroup.groupby('iterator')):
            thisCovMat = covMatDF.loc[idxSl[iteratorName, freqBandName, lastFoldIdx, :], :].clip(lower=groupCovMinVal, upper=groupCovMaxVal)
            thisCovMat.index = thisCovMat.index.get_level_values('feature')
            newTickLabels = thisCovMat.index.to_numpy()
            for tlIdx, tl in enumerate(newTickLabels):
                if (tlIdx % int(newTickLabels.shape[0] / 10)) != 0:
                    newTickLabels[tlIdx] = ''
                else:
                    newTickLabels[tlIdx] = tl
            commonHeatmapOpts.update(dict(
                xticklabels=newTickLabels
                ))
            if axIdx == 0:
                sns.heatmap(thisCovMat, ax=ax[axIdx], cbar_ax=ax[-1], yticklabels=newTickLabels, **commonHeatmapOpts)
                ax[axIdx].set_yticklabels(newTickLabels, rotation=30, ha='right', va='top')
                ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
            elif axIdx == (nSubGroups - 1):
                sns.heatmap(thisCovMat, ax=ax[axIdx], cbar_ax=ax[-1], yticklabels=False, **commonHeatmapOpts)
                ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
                ax[axIdx].set_ylabel('')
            else:
                sns.heatmap(thisCovMat, ax=ax[axIdx], cbar=False, yticklabels=False, **commonHeatmapOpts)
                ax[axIdx].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
                ax[axIdx].set_ylabel('')
            ax[axIdx].set_title(sourcePalette.loc[iteratorName, 'description'])
        ax[-1].set_ylabel('Covariance ($uV^2$)')
        fig.suptitle('Example covariance matrices (frequency band: {})'.format(freqBandName))
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        for distanceType in ['compound']:
            distanceDF = distancesDict[distanceType][freqBandName]
            distanceForPlot = distanceDF.copy()
            newIndex = distanceDF.index.get_level_values('iterator').map(sourcePalette['description'])
            distanceForPlot.index = newIndex
            distanceForPlot.columns = newIndex
            newTickLabels = newIndex.to_numpy()
            for tlIdx, tl in enumerate(newTickLabels):
                if (tlIdx % lastFoldIdx) != 0:
                    newTickLabels[tlIdx] = ''
                else:
                    newTickLabels[tlIdx] = tl.replace(',', ',\n')
            fig, ax = plt.subplots(
                1, 2, figsize=(3, 3),
                gridspec_kw={
                    'width_ratios': [20, 1],
                    'wspace': 0.1})
            sns.heatmap(
                distanceForPlot, cmap='rocket',
                xticklabels=newTickLabels, yticklabels=newTickLabels,
                ax=ax[0], linewidths=0, cbar_ax=ax[1]
                # norm=LogNorm()
                )
            ax[0].set_ylabel('data subset')
            ax[0].set_xlabel('data subset')
            ax[0].set_xticklabels(newTickLabels, rotation=-30, ha='left', va='top')
            ax[0].set_yticklabels(newTickLabels, rotation=30, ha='right', va='top')
            ax[1].set_ylabel('Distance (a.u.)')
            fig.suptitle('{} covariance distances (frequency band: {})'.format(distanceType, freqBandName))
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
            plotDF.loc[plotDF['ref_hasStim'] & plotDF['test_hasStim'], 'stimNoStim'] = 'withinCluster'
            plotDF.loc[plotDF['ref_hasStim'] & (~plotDF['test_hasStim']), 'stimNoStim'] = 'betweenCluster'
            plotDF.loc[(~plotDF['ref_hasStim']) & plotDF['test_hasStim'], 'stimNoStim'] = 'betweenCluster'
            plotDF.loc[(~plotDF['ref_hasStim']) & (~plotDF['test_hasStim']), 'stimNoStim'] = 'NA'
            #
            plotDF.loc[:, 'stimPerimovementStim'] = 'NA'
            plotDF.loc[(plotDF['test_iterator'] == 'ccs') & (plotDF['ref_iterator'] == 'ccm'), 'stimPerimovementStim'] = 'betweenCluster'
            plotDF.loc[(plotDF['test_iterator'] == 'ccm') & (plotDF['ref_iterator'] == 'ccs'), 'stimPerimovementStim'] = 'betweenCluster'
            plotDF.loc[(plotDF['test_iterator'] == 'ccs') & (plotDF['ref_iterator'] == 'ccs'), 'stimPerimovementStim'] = 'withinCluster'
            plotDF.loc[(plotDF['test_iterator'] == 'ccm') & (plotDF['ref_iterator'] == 'ccm'), 'stimPerimovementStim'] = 'withinCluster'
            #
            plotDF.loc[:, 'stimByClass'] = 'NA'
            plotDF.loc[(plotDF['test_iterator'] == 'ccs') & (plotDF['ref_iterator'] == 'ccs'), 'stimByClass'] = 'withinStim'
            plotDF.loc[(plotDF['test_iterator'] == 'ccm') & (plotDF['ref_iterator'] == 'ccm'), 'stimByClass'] = 'withinStim'
            plotDF.loc[(plotDF['test_iterator'] == 'ccs') & (plotDF['ref_iterator'] == 'ccm'), 'stimByClass'] = 'acrossStim'
            plotDF.loc[(plotDF['test_iterator'] == 'ccm') & (plotDF['ref_iterator'] == 'ccs'), 'stimByClass'] = 'acrossStim'
            plotDF.loc[(plotDF['test_iterator'].isin(['ccs', 'ccm'])) & (~plotDF['ref_iterator'].isin(['ccs', 'ccm'])), 'stimByClass'] = 'outsideStim'
            plotDF.loc[(~plotDF['test_iterator'].isin(['ccs', 'ccm'])) & (plotDF['ref_iterator'].isin(['ccs', 'ccm'])), 'stimByClass'] = 'outsideStim'
            normFactor = plotDF.loc[plotDF['stimByClass'] == 'withinStim', 'distance'].mean()
            normalizedClusterDistDF = plotDF.loc[plotDF['stimByClass'] != 'NA', :].copy()
            normalizedClusterDistDF.loc[:, 'distance'] = normalizedClusterDistDF['distance'] / normFactor
            normalizedClusterDistDF.loc[:, 'xDummy'] = 0.
            thisPalette = sourcePalette['color'].loc[sourcePalette.index.isin(normalizedClusterDistDF['stimByClass'])]
            #
            g = sns.catplot(
                data=normalizedClusterDistDF,
                y='distance', x='xDummy', hue='stimByClass', height=2, aspect=1,
                palette=thisPalette.to_dict(), hue_order=thisPalette.index.to_list(),
                kind='violin')
            g.set_axis_labels("", "Distance (a.u.)")
            g.suptitle('Clusterability of {} distance (frequency band: {})'.format(distanceType, freqBandName))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            umapDF = projectedDict[distanceType][freqBandName]
            plotDF = umapDF.reset_index()
            thisPalette = sourcePalette['color'].loc[sourcePalette.index.isin(plotDF['iterator'])]
            g = sns.relplot(
                x='umap_0', y='umap_1', kind='scatter', height=3, aspect=1,
                hue='iterator', palette=thisPalette.to_dict(), data=plotDF)
            xLim = g.axes[0, 0].get_xlim()
            yLim = g.axes[0, 0].get_ylim()
            offset = (xLim[1] - xLim[0]) / 100, (yLim[1] - yLim[0]) / 100
            examplePoints = [
                ('ccs', 'ccs', 'withinStim'),
                ('ccm', 'ccm', 'withinStim'),
                ('ccm', 'ccs', 'acrossStim'),
                ('ccm', 'cb', 'outsideStim'),
                ('ccs', 'cb', 'outsideStim'),
                ('ccm', 'ca', 'outsideStim'),
                ('ccs', 'ca', 'outsideStim')]
            rng = np.random.default_rng()
            nPoints = 1
            for pairName in examplePoints:
                testIdxList = rng.choice(umapDF.xs(pairName[0], level='iterator', drop_level=False).index, nPoints)
                refIdxList = rng.choice(umapDF.xs(pairName[1], level='iterator', drop_level=False).index, nPoints)
                # testIdx = umapDF.xs(pairName[0], level='iterator', drop_level=False).index[pointIdx]
                # refIdx = umapDF.xs(pairName[1], level='iterator', drop_level=False).index[pointIdx+1]
                for testIdx, refIdx in zip(testIdxList, refIdxList):
                    pointsToPlot = umapDF.loc[[testIdx, refIdx], :].to_numpy()
                    alpha = 0.5
                    useColor = tuple([rgb for rgb in sourcePalette.loc[pairName[2], 'color']] + [alpha])
                    g.axes[0, 0].plot(pointsToPlot[:, 0], pointsToPlot[:, 1], c=useColor)
                textPos = np.mean(pointsToPlot, axis=0)
                g.axes[0, 0].text(
                    textPos[0] - offset[0], textPos[1] - offset[1], pairName[2], c=useColor, va='top', ha='right')
            g.suptitle('umap projection of {} distance (frequency band: {})'.format(distanceType, freqBandName))
            asp.reformatFacetGridLegend(
                g, titleOverrides={'iterator': 'data subset'},
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
            g.tight_layout(pad=.3)
            pdf.savefig(bbox_inches='tight')
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()