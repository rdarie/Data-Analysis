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
    --datasetPrefix=datasetPrefix                  filename for resulting estimator (cross-validated n_comps) [default: Block_XL_df]
    --selectionName=selectionName                  filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName                    append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName              append a name to the resulting blocks? [default: motion]
    --targetList=targetList                        which iterators to compare
    --iteratorSuffixList=iteratorSuffixList        which iterators to compare
    --expList=expList                              which experiments to compare
    --freqBandGroup=freqBandGroup                  what will the plot be aligned to? [default: outboundWithStim]
    --plotSuffix=plotSuffix                        switches between a few different processing options [default: all]
"""

import logging
logging.captureWarnings(True)
import matplotlib as mpl
import os
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.colors import LogNorm, Normalize
# from matplotlib.ticker import MaxNLocator
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
import seaborn as sns
import os, sys
from itertools import combinations, permutations, product
# import pingouin as pg
# from sklearn.pipeline import make_pipeline, Pipeline
# import dataAnalysis.helperFunctions.profiling as prf
# import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
# import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
# from dataAnalysis.analysis_code.namedQueries import namedQueries
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import pdb, traceback
import numpy as np
import pandas as pd
import colorsys
# import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
# from sklearn.covariance import EmpiricalCovariance
# from sklearn.metrics import explained_variance_score
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
# from sklearn.decomposition import PCA, FactorAnalysis
# from sklearn.manifold import MDS
# import joblib as jb
import dill as pickle
# import gc
# import vg
from statannotations.Annotator import Annotator
# from statannotations.stats.StatResult import StatResult
# from statannotations.PValueFormat import PValueFormat
from dataAnalysis.analysis_code.currentExperiment import *
from docopt import docopt
# from numpy.random import default_rng
# from itertools import product
# from scipy import stats
# import scipy
# import umap
# import umap.plot
# from tqdm import tqdm
from sklearn.metrics import silhouette_samples, silhouette_score
#
idxSl = pd.IndexSlice
#
useDPI = 300
dpiFactor = 72 / useDPI
snsContext = 'talk'
snsRCParams = {
    'axes.facecolor': 'w',
    #
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    #
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": False,
    #
    "xtick.bottom": True,
    "xtick.top": False,
    "ytick.left": True,
    "ytick.right": False,
    }
snsRCParams.update(customSeabornContexts[snsContext])
mplRCParams = {
    'figure.dpi': useDPI, 'savefig.dpi': useDPI,
    #
    'axes.titlepad': 1.5,
    'axes.labelpad': 0.75,
    #
    'figure.subplot.left': 0.02,
    'figure.subplot.right': 0.98,
    'figure.subplot.bottom': 0.02,
    'figure.subplot.top': 0.98,
    #
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }

styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 1.,
    'panel_heading.pad': 1.
    }

for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

sns.set(
    context=snsContext, style='white',
    palette='dark', font='sans-serif',
    color_codes=True, rc=snsRCParams)

for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
# full list 'exp202101201100, exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100'
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
    'showFigures': True, 'iteratorSuffixList': 'ca, cb, ccs, ccm', 'debugging': False,
    'estimatorName': 'mahal_ledoit', 'alignFolderName': 'motion', 'analysisName': 'hiRes', 'plotting': True,
    'verbose': '1', 'blockIdx': '2', 'selectionName': 'laplace_spectral_scaled', 'datasetPrefix': 'Block_XL_df',
    'targetList': "laplace_spectral_scaled, laplace_scaled",
    'processAll': True, 'expList': 'exp202101271100, exp202101211100',
    }
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

listOfExpSuffixes = [x.strip() for x in arguments['expList'].split(',')]
listOfIteratorSuffixes = [x.strip() for x in arguments['iteratorSuffixList'].split(',')]
listOfSelectionNames = [x.strip() for x in arguments['targetList'].split(',')]
hasLoadedMap = False
covMatDict = {}
estimatorsDict = {}
loadRawDistanceMatrix = True
if loadRawDistanceMatrix:
    distancesDict = {}
distancesStackDict = {}
silhouetteDict = {}
for expName in listOfExpSuffixes:
    print('load experiment {}'.format(expName))
    arguments['exp'] = expName
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    figureOutputFolder = os.path.join(
        remoteBasePath, 'figures', 'dimensionality_across_exp')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    pdfPath = os.path.join(
        figureOutputFolder, '{}_covMatSimilarity_comparison_fbg{}_{}.pdf'.format(
            subjectName, arguments['freqBandGroup'], arguments['plotSuffix']))
    #
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    lOfDistanceTypes = ['KL', 'frobenius', 'spectral']
    for selectionName in listOfSelectionNames:
        print('load selection {}'.format(selectionName))
        estimatorName = arguments['estimatorName']
        fullEstimatorName = '{}_{}_{}'.format(
            estimatorName, arguments['datasetPrefix'], selectionName)
        resultPath = os.path.join(estimatorsSubFolder, '{}_{}.h5'.format(fullEstimatorName, 'covarianceMatrixCalc'))
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
                    covMatDict[expName, selectionName, iteratorSuffix] = pd.read_hdf(store, 'cv_covariance_matrices')
                if 'full_estimators' in store:
                    estimatorsDict[expName, selectionName, iteratorSuffix] = pd.read_hdf(store, 'full_estimators')
                elif 'cv_estimators' in store:
                    estimatorsDict[expName, selectionName, iteratorSuffix] = pd.read_hdf(store, 'cv_estimators')
            if 'spectral' in selectionName:
                estimatorsDict[expName, selectionName, iteratorSuffix].drop('all', axis='index', level='freqBandName', inplace=True)
            if not hasLoadedMap:
                datasetPath = os.path.join(
                    dataFramesFolder,
                    datasetName + '.h5'
                    )
                dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
                mapDF = dataDF.columns.to_frame().reset_index(drop=True).set_index('feature')
                hasLoadedMap = True
                del dataDF
        thisFreqBandNameList = np.unique(estimatorsDict[expName, selectionName, iteratorSuffix].index.get_level_values('freqBandName'))
        for freqBandName in thisFreqBandNameList:
            print('    load freq band {}'.format(freqBandName))
            for distanceType in lOfDistanceTypes + ['compound']:
                if loadRawDistanceMatrix:
                    thisDistanceDF = pd.read_hdf(resultPath, '/distanceMatrices/{}/{}'.format(freqBandName, distanceType))
                    for testIter, refIter in combinations(thisDistanceDF.index.get_level_values('iterator').unique(), 2):
                        sortedIter = sorted([testIter, refIter])
                        yMask = thisDistanceDF.index.get_level_values('iterator').isin(sortedIter)
                        xMask = thisDistanceDF.columns.get_level_values('iterator').isin(sortedIter)
                        msDF = thisDistanceDF.loc[yMask, xMask]
                        # plt.matshow(msDF); plt.show()
                        msSH = pd.Series(
                            silhouette_samples(msDF.to_numpy(), np.asarray(msDF.columns.get_level_values('iterator')), metric='precomputed'),
                            index=msDF.index)
                        # plt.matshow(msSH); plt.show()
                        msSH.index = msSH.index.droplevel('freqBandName')
                        silhouetteDict[(expName, selectionName, '{}_silh'.format(distanceType), freqBandName, sortedIter[0], sortedIter[1])] = msSH.to_frame(name='distance')
                    thisDistanceDF.index = thisDistanceDF.index.droplevel('freqBandName')
                    # selfCompareMask = thisDistanceDF.apply(lambda x: x.index.get_level_values('iterator'), axis='index') == thisDistanceDF.T.apply(lambda x: x.index.get_level_values('iterator'), axis='index').T
                    # for diagIdx in range(thisDistanceDF.shape[0]):
                    #     selfCompareMask.iloc[diagIdx, diagIdx] = False
                    # scaleFactor = np.std(thisDistanceDF.to_numpy()[selfCompareMask.to_numpy()])
                    # plt.matshow((thisDistanceDF).to_numpy()); plt.show()
                    distancesDict[(expName, selectionName, distanceType, freqBandName)] = thisDistanceDF
                try:
                    thisDistanceStackDF = pd.read_hdf(resultPath, '/distanceMatricesForPlot/{}/{}'.format(freqBandName, distanceType))
                    normalizeMe = False
                    if normalizeMe:
                        scaler = StandardScaler()
                        scaler.fit(thisDistanceStackDF.loc[thisDistanceStackDF['category'] == 'WE', ['distance']])
                        thisDistanceStackDF.loc[:, 'distance'] = scaler.transform(thisDistanceStackDF.loc[:, ['distance']])
                    distancesStackDict[(expName, selectionName, distanceType, freqBandName)] = thisDistanceStackDF
                except Exception:
                    # traceback.print_exc()
                    pass
                # try:
                #     thisDistanceStackNormDF = pd.read_hdf(resultPath, '/distanceMatricesForPlotNormalized/{}/{}'.format(freqBandName, distanceType))
                #     distancesStackNormDict[(expName, selectionName, distanceType, freqBandName)] = thisDistanceStackNormDF
                # except Exception:
                #     # traceback.print_exc()
                #     pass
#
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
#
diagonalTerms = ['{}_{}'.format(tn, tn) for tn in listOfIteratorSuffixes]
NACategory = 'Other'
'''categoryDescriptions = pd.Series({
    'WE': diagonalTerms,
    'deltaM': ['ca_cb'],
    'deltaS': ['ca_ccs'],
    'deltaC': ['cb_ccm'],
    'deltaSS': ['ccm_ccs'],
    NACategory: ['ca_ccm', 'cb_ccs']
    })'''
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
prettyNameLookup['0.'] = ''
maskNoStimElecNames = False
spinalMapDF = spinalElectrodeMaps[subjectName].sort_values(['xCoords', 'yCoords'])
spinalElecCategoricalDtype = pd.CategoricalDtype(spinalMapDF.index.to_list(), ordered=True)
spinalMapDF.index = pd.Index(spinalMapDF.index, dtype=spinalElecCategoricalDtype)
# covMatDF = pd.concat(covMatDict, names=['expName', 'selectionName', 'iterator'])
del covMatDict
if loadRawDistanceMatrix:
    distanceDF = pd.concat(distancesDict, names=['expName', 'selectionName', 'distanceType', 'freqBandName'])
    del distancesDict

distanceStackDF = pd.concat(distancesStackDict, names=['expName', 'selectionName', 'distanceType', 'freqBandName', 'temp'])
distanceStackDF.reset_index(inplace=True)
dropTheseCols = ['temp', 'test_hasStim', 'ref_hasStim', 'ref_hasMovement', 'test_hasMovement', 'test_fold', 'ref_fold', 'category_num', 'group_num', 'category']
distanceStackDF.drop(columns=[cN for cN in dropTheseCols if cN in distanceStackDF.columns], inplace=True)

silhDF = pd.concat(silhouetteDict,
    names=['expName', 'selectionName', 'distanceType', 'freqBandName', 'test_iterator',
           'ref_iterator', 'iterator', 'fold']).reset_index()
silhDF.drop(columns=['iterator', 'fold'], inplace=True)
silhDF.loc[:, 'xDummy'] = 0.
silhDF.loc[:, 'test_ref'] = (
    silhDF[['test_iterator', 'ref_iterator']]
        .apply(lambda x: '{}_{}'.format(*sorted(x.to_list())), axis='columns'))
silhDF.loc[:, 'test_ref_label'] = silhDF['test_ref'].map(testRefLookup)
#
distanceStackDF = pd.concat([distanceStackDF, silhDF.loc[:, distanceStackDF.columns]], axis='index').reset_index(drop=True)
elecNames = distanceStackDF['expName'].apply(lambda x: expNameElectrodeLookup[subjectName][x][0])
if maskNoStimElecNames:
    noStimMask = ~(distanceStackDF['ref_hasStim'] | distanceStackDF['test_hasStim'])
    elecNames.loc[noStimMask.to_numpy()] = 'NA'
distanceStackDF.loc[:, 'electrode'] = pd.Categorical(elecNames.to_numpy(), dtype=spinalElecCategoricalDtype)
del distancesStackDict, elecNames
estimatorsSrs = pd.concat(estimatorsDict, names=['expName', 'selectionName', 'iterator'])
lastFoldIdx = estimatorsSrs.index.get_level_values('fold').max()
estimatorsSrs.drop(lastFoldIdx, axis='index', level='fold', inplace=True)

if arguments['plotSuffix'] == 'all':
    if subjectName == 'Rupert':
        lOfElectrodes = ['NA', '-E13+E16', '-E09+E16', '-E11+E16', '-E02+E16', '-E05+E16', '-E04+E16']
    elif subjectName == 'Murdoc':
        lOfElectrodes = []
    distanceStackDF = distanceStackDF.loc[distanceStackDF['electrode'].isin(lOfElectrodes), :]
    estimatorsSrs = estimatorsSrs.loc[estimatorsSrs.index.get_level_values('expName').isin(distanceStackDF['expName'].unique())]
elif arguments['plotSuffix'] == 'best_three':
    if subjectName == 'Rupert':
        lOfElectrodes = ['NA', '-E11+E16', '-E05+E16', '-E04+E16']
    elif subjectName == 'Murdoc':
        lOfElectrodes = []
    distanceStackDF = distanceStackDF.loc[distanceStackDF['electrode'].isin(lOfElectrodes), :]
    estimatorsSrs = estimatorsSrs.loc[estimatorsSrs.index.get_level_values('expName').isin(distanceStackDF['expName'].unique())]
elif arguments['plotSuffix'] == 'E04':
    if subjectName == 'Rupert':
        lOfElectrodes = ['NA', '-E04+E16']
    elif subjectName == 'Murdoc':
        lOfElectrodes = []
    distanceStackDF = distanceStackDF.loc[distanceStackDF['electrode'].isin(lOfElectrodes), :]
    estimatorsSrs = estimatorsSrs.loc[estimatorsSrs.index.get_level_values('expName').isin(distanceStackDF['expName'].unique())]

freqBandOrderExtended = ['NA', 'all', 'broadband', 'cross_frequency'] + freqBandOrder
freqBandNameList = [
    bn
    for bn in freqBandOrderExtended
    if bn in estimatorsSrs.index.get_level_values('freqBandName').to_list()]
if arguments['freqBandGroup'] == '0':
    freqBandShortList = ['all']
elif arguments['freqBandGroup'] == '1':
    freqBandShortList = ['all'] + freqBandOrder
elif arguments['freqBandGroup'] == '2':
    freqBandShortList = ['all', 'beta', 'gamma']
elif arguments['freqBandGroup'] == '3':
    freqBandShortList = ['gamma', 'higamma']
elif arguments['freqBandGroup'] == '4':
    freqBandShortList = ['all', 'beta']
expNameList = estimatorsSrs.index.get_level_values('expName').unique().to_list()

remakePalettes = True
if remakePalettes:
    pickingColors = False
    singlesPalette = pd.DataFrame(
        sns.color_palette(colorMaps['points'], 5), columns=['r', 'g', 'b'], index=iteratorDescriptions.to_list())
    singlesPaletteHLS = singlesPalette.apply(
        lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']),
        axis='columns')
    if pickingColors:
        sns.palplot(singlesPalette.apply(lambda x: tuple(x), axis='columns'))
        palAx = plt.gca()
        for tIdx, tN in enumerate(singlesPalette.index):
            palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
    ######
    # for freqBandName in freqBandNameList:
    #     for distanceType in ['frobenius']:
    freqBandName = freqBandNameList[0]
    distanceType = 'frobenius'
    exampleSliceMask = (
        (distanceStackDF['freqBandName'] == freqBandName) &
        (distanceStackDF['distanceType'] == distanceType)
        )
    thisDistanceDF = distanceStackDF.loc[exampleSliceMask].drop(columns=['freqBandName', 'distanceType'])
    groupLookup = thisDistanceDF[['test_iterator', 'ref_iterator']].drop_duplicates().reset_index(drop=True)
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
    #
    pairIndex = thisDistanceDF[['test_iterator', 'ref_iterator']].apply(lambda x: tuple(x), axis='columns')
    for key in ['test_ref', 'test_ref_label']:
        thisDistanceDF.loc[:, key] = pairIndex.map(groupLookup[['test_iterator', 'ref_iterator', key]].set_index(['test_iterator', 'ref_iterator'])[key])
    #
    categoryLookup = groupLookup[['test_ref', 'test_ref_label']].drop_duplicates().reset_index(drop=True)
    for key, value in categoryDescriptions.items():
        categoryLookup.loc[categoryLookup['test_ref'].isin(value), 'category'] = key
    categoryLookup.loc[:, 'category'] = pd.Categorical(categoryLookup['category'].fillna(NACategory), dtype=categoryDtype)
    reOrderList = categoryDescriptions.index.to_list() + [NACategory]
    categoryLookup.loc[:, 'category_num'] = categoryLookup['category'].apply(lambda x: reOrderList.index(x))
    categoryLookup.sort_values(by=['category_num', 'test_ref'], inplace=True)
    categoryLookup.reset_index(drop=True, inplace=True)
    categoryLookup.loc[:, 'group_num'] = categoryLookup.index.to_numpy()
    categoryLookup.loc[:, 'category_label'] = categoryLookup['category'].map(categoryLabels)
    nGroups = categoryLookup.shape[0]
    rawGroupColors = sns.color_palette(colorMaps['distHist'], nGroups)
    groupPalette = pd.DataFrame(
        rawGroupColors, columns=['r', 'g', 'b'],
        index=categoryLookup['test_ref_label'])
    if pickingColors:
        sns.palplot(groupPalette.apply(lambda x: tuple(x), axis='columns'))
        palAx = plt.gca()
        for tIdx, tN in enumerate(groupPalette.index):
            palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
    ####
    groupLookup.loc[:, 'color'] = groupLookup['test_ref_label'].astype(np.object).map(groupPalette.apply(lambda x: tuple(x), axis='columns'))
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
    categoryLookup.loc[:, 'color'] = categoryLookup['category'].astype(np.object).map(
        categoryPalette.apply(lambda x: tuple(x), axis='columns'))
else:
    categoryLookup = pd.read_hdf(resultPath, '/categoryLookup')
    groupLookup = pd.read_hdf(resultPath, '/groupsLookup')
    singlesPalette = pd.read_hdf(resultPath, '/singlesPalette')

distanceStackDF.loc[:, 'test_ref_label'] = distanceStackDF['test_ref'].map(testRefLookup)
distanceStackDF.loc[:, 'category'] = distanceStackDF['test_ref'].map(categoryLookup[['test_ref', 'category']].set_index('test_ref')['category'])
# distanceStackDF.loc[distanceStackDF['category'] == 'WE', 'electrode'] = 'NA'
# rename S to Sn for each electrode
subGroupLookup = distanceStackDF[['test_iterator', 'ref_iterator', 'test_ref', 'test_ref_label', 'electrode', 'expName']].drop_duplicates().reset_index(drop=True)
subGroupLookup.loc[:, 'test_ref_exp'] = subGroupLookup.apply(lambda x: '{}_{}'.format(x['test_ref'], x['expName']), axis='columns')
subGroupLookup.loc[:, 'test_ref_elec'] = subGroupLookup.apply(lambda x: '{}_{}'.format(x['test_ref'], x['electrode']), axis='columns')
subGroupLookup.loc[:, 'test_ref_elec_label'] = subGroupLookup.apply(lambda x: '{} ({})'.format(x['test_ref_label'], x['electrode']), axis='columns')
subGroupLookup.sort_values(['electrode', 'test_ref_label'], inplace=True)
subGroupLookup.reset_index(drop=True, inplace=True)
#
subCategoryLookup = subGroupLookup[['test_ref_elec', 'test_ref_elec_label', 'test_ref_exp', 'electrode', 'expName', 'test_ref', 'test_ref_label']].drop_duplicates().reset_index(drop=True)
for key, value in categoryDescriptions.items():
    subCategoryLookup.loc[subCategoryLookup['test_ref'].isin(value), 'category'] = key
subCategoryLookup.loc[:, 'category'] = pd.Categorical(subCategoryLookup['category'].fillna(NACategory), dtype=categoryDtype)
subCategoryLookup.loc[:, 'category_label'] = subCategoryLookup['category'].map(categoryLabels)
subCategoryLookup.loc[:, 'category_elec'] = subCategoryLookup.apply(lambda x: '{} ({})'.format(x['category'], x['electrode']), axis='columns')
subCategoryLookup.loc[:, 'category_elec_label'] = subCategoryLookup.apply(lambda x: '{} ({})'.format(x['category_label'], x['electrode']), axis='columns')
subCategoryLookup.loc[:, 'xDummy'] = 0
subCategoryLookup.sort_values(['category', 'test_ref_label', 'electrode'], inplace=True)
subCategoryLookup.reset_index(drop=True, inplace=True)
#
distanceStackDF.loc[:, 'test_ref_elec'] = distanceStackDF.apply(lambda x: '{}_{}'.format(x['test_ref'], x['electrode']), axis='columns')
distanceStackDF.loc[:, 'test_ref_elec_label'] = distanceStackDF.apply(lambda x: '{} ({})'.format(x['test_ref_label'], x['electrode']), axis='columns')
distanceStackDF.loc[:, 'category_elec'] = distanceStackDF.apply(lambda x: '{} ({})'.format(x['category'], x['electrode']), axis='columns')
distanceStackDF.loc[:, 'category_label'] = distanceStackDF['category'].map(categoryLabels)
distanceStackDF.loc[:, 'category_elec_label'] = distanceStackDF.apply(lambda x: '{} ({})'.format(x['category_label'], x['electrode']), axis='columns')
# distanceStackDF.loc[:, ['expName', 'distanceType', 'electrode']].drop_duplicates()

# plotOnlyTheseBands = ['all', 'beta', 'gamma', 'higamma']
# masterDF = distanceStackDF.loc[distanceStackDF['freqBandName'].isin(plotOnlyTheseBands), :]
masterDF = distanceStackDF
#
highLightGroups = {
    'unscaled': masterDF['test_ref_label'].isin(['B-B', 'B-M', 'B-S', 'B-C', 'M-S', 'M-C', 'C-S']), #  & masterDF['freqBandName'].isin(['all'])
    'relative to B-M': masterDF['test_ref_label'].isin(['B-M', 'B-S', 'B-C', 'M-S', 'M-C', 'C-S']),
    'relative to M-S': masterDF['test_ref_label'].isin(['M-S', 'M-C', 'C-S']),
    'Everything': masterDF['test_ref_label'].notna()
    }
groupPagesBy = ['distanceType']
hueVar = 'test_ref_label'
hueVarLabel = None
rowVar = 'xDummy'
rowOrder = [0.]
xVar = 'freqBandName'
colVar = 'electrode'
colOrder = subCategoryLookup[colVar].sort_values().unique()
uniqueHues = subCategoryLookup[hueVar].unique()
numHues = uniqueHues.shape[0]
#
rawGroupPalette = sns.color_palette(colorMaps['distHist'])
rawGroupColors = {
    'B-B': sns.utils.alter_color(rawGroupPalette[7], l=.6), # gray
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
subCategoryPalette = pd.DataFrame(
    rawGroupColors).T
subCategoryPalette.columns = ['r', 'g', 'b']
if pickingColors:
    sns.palplot(subCategoryPalette.apply(lambda x: tuple(x), axis='columns'))
    palAx = plt.gca()
    for tIdx, tN in enumerate(subCategoryPalette.index):
        palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
subCategoryLookup.loc[:, 'color'] = subCategoryLookup['test_ref_label'].astype(np.object).map(subCategoryPalette.apply(lambda x: tuple(x), axis='columns'))
#
confidence_alpha = 0.05
dictForHueStats = {}
plotDistances = ['frobenius']
for name, group in masterDF.groupby(groupPagesBy + [rowVar, colVar, xVar]):
    tempStatsDict = {}
    hueSubGroups = [sg for sn, sg in group.groupby(hueVar) if not sg.empty]
    if len(hueSubGroups) > 1:
        for subGroup1, subGroup2 in combinations(hueSubGroups, 2):
            thisIndex = tuple([subGroup1[hueVar].unique()[0], subGroup2[hueVar].unique()[0]])
            tempStatsDict[thisIndex] = tdr.tTestNadeauCorrection(
                subGroup1['distance'], subGroup2['distance'], tTestArgs={}, powerAlpha=0.05,
                test_size=cvIterator.splitter.sampler.test_size)
            tempStatsDict[thisIndex].rename(index={'p-val': 'pval'}, inplace=True)
        dictForHueStats[name] = pd.concat(tempStatsDict, axis='columns', names=['category1', 'category2']).T
hueStatsDF = pd.concat(dictForHueStats, names=groupPagesBy + [rowVar, colVar, xVar])

with PdfPages(pdfPath) as pdf:
    for hgName, hgMask in highLightGroups.items():
        hgDF = masterDF.loc[hgMask, :]
        print('on highlight group {}'.format(hgName))
        for pageNames, pageGroup in hgDF.groupby(groupPagesBy):
            print(
                'grouping pages by {}\non page group {}'.format(
                    groupPagesBy, pageNames))
            distanceType = pageNames
            if distanceType != 'frobenius':
                continue
            plotDF = pageGroup.copy()
            if (distanceType == 'frobenius') and (hgName == 'unscaled'):
                for _, group in plotDF.groupby(['freqBandName', 'expName']):
                    normFactor = group.loc[group['test_ref_label'] == 'B-B', 'distance'].median()
                    rescaled = group['distance'] / normFactor
                    plotDF.loc[group.index, 'distance'] = rescaled
                ## thisHueOrder = ['B-B', 'B-M', 'B-S', 'B-C']
                thisHueOrder = ['B-B', 'B-M', 'B-S', 'B-C', 'M-S', 'M-C', 'C-S']
                xVar = 'xDummy'
                xOrder = [0.]
                #
                colVar = 'freqBandName'
                colOrder = [
                    fbn
                    for fbn in freqBandShortList
                    if fbn in plotDF['freqBandName'].to_list()]
                rowVar = 'electrode'
                rowOrder = [
                    eN
                    for eN in subCategoryLookup['electrode'].sort_values().unique()
                    if eN in plotDF['electrode'].to_list()]
                shareY = False
                colWrap = None
                shareYByRow = False
            elif (distanceType == 'frobenius') and (hgName == 'Everything'):
                for _, group in plotDF.groupby(['freqBandName', 'expName']):
                    normFactor = group.loc[group['test_ref_label'] == 'B-B', 'distance'].median()
                    rescaled = group['distance'] / normFactor
                    plotDF.loc[group.index, 'distance'] = rescaled
                thisHueOrder = np.unique(subCategoryLookup[hueVar]).tolist()
                #
                xVar = 'xDummy'
                xOrder = [0.]
                colVar = 'freqBandName'
                colOrder = [
                    fbn
                    for fbn in freqBandShortList
                    if fbn in plotDF['freqBandName'].to_list()]
                rowVar = 'electrode'
                rowOrder = [
                    eN for eN in subCategoryLookup['electrode'].sort_values().unique()
                    if eN in plotDF['electrode'].to_list()]
                #
                colWrap = None
                shareY = False
                shareYByRow = False
            elif (distanceType == 'frobenius') and (hgName == 'relative to B-M'):
                for _, group in plotDF.groupby(['freqBandName', 'expName']):
                    normFactor = group.loc[group['test_ref_label'] == 'B-M', 'distance'].median()
                    rescaled = group['distance'] / normFactor
                    plotDF.loc[group.index, 'distance'] = rescaled
                thisHueOrder = ['B-M', 'B-S', 'B-C', 'M-S', 'M-C', 'C-S']
                #
                xVar = 'xDummy'
                xOrder = [0.]
                colVar = 'freqBandName'
                colOrder = [
                    fbn
                    for fbn in freqBandShortList
                    if fbn in plotDF['freqBandName'].to_list()]
                #
                rowVar = 'electrode'
                rowOrder = [
                    eN
                    for eN in subCategoryLookup['electrode'].sort_values().unique()
                    if eN in plotDF['electrode'].to_list()]
                #
                colWrap = None
                shareY = False
            elif (distanceType == 'frobenius') and (hgName == 'relative to M-S'):
                for _, group in plotDF.groupby(['freqBandName', 'expName']):
                    normFactor = group.loc[group['test_ref_label'] == 'M-S', 'distance'].median()
                    rescaled = group['distance'] / normFactor
                    plotDF.loc[group.index, 'distance'] = rescaled
                thisHueOrder = ['M-S', 'M-C', 'C-S']
                #
                rowVar = 'xDummy'
                rowOrder = [0.]
                xVar = 'freqBandName'
                xOrder = [
                    fbn
                    for fbn in freqBandShortList
                    if fbn in plotDF['freqBandName'].to_list()]
                colVar = 'electrode'
                colOrder = [
                    eN
                    for eN in subCategoryLookup['electrode'].sort_values().unique()
                    if eN in plotDF['electrode'].to_list()]
                #
                shareY = False
                colWrap = None
                shareYByRow = False
            ############################################################
            if rowOrder is not None:
                figHeight = 1.5 + 3. * len(rowOrder)
                height = figHeight / len(rowOrder)
            else:
                figHeight = 4.5
                height = figHeight
            #
            if colOrder is not None:
                figWidth = 1. + .5 * len(thisHueOrder) * len(xOrder) * len(colOrder)
                width = figWidth / len(colOrder)
            else:
                figWidth = 1. + .5 * len(thisHueOrder) * len(xOrder)
                width = figWidth
            #
            argsForBoxPlot = dict(whis=np.inf, width=0.9)
            thisPalette = (
                subCategoryLookup
                    .loc[subCategoryLookup[hueVar].isin(plotDF.loc[:, hueVar]), [hueVar, 'color']]
                    .drop_duplicates().set_index(hueVar)[['color']])
            argsForCatPlot = dict(
                y='distance', x=xVar, order=xOrder,
                hue=hueVar, palette=thisPalette['color'].to_dict(), hue_order=thisHueOrder,)
            aspect = width / height
            g = sns.catplot(
                row=rowVar, row_order=rowOrder,
                col=colVar, col_order=colOrder, col_wrap=colWrap,
                data=plotDF, height=height, aspect=aspect,
                margin_titles=True, sharey=shareY, kind='box',
                **argsForCatPlot, **argsForBoxPlot)
            g.set_titles(row_template="{row_name}", col_template="{col_name}")
            plotProcFuns = [
                asp.genTitleChanger(prettyNameLookup)]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            for (row_val, col_val), ax in g.axes_dict.items():
                statsSubset = hueStatsDF.xs(distanceType, level='distanceType').reset_index()
                if g._row_var is not None:
                    print('g._row_var = {}'.format(g._row_var))
                    dataSubset = plotDF.loc[plotDF[g._row_var] == row_val, :]
                    statsSubset = statsSubset.loc[statsSubset[g._row_var] == row_val, :]
                else:
                    dataSubset = plotDF
                    statsSubset = statsSubset
                #
                if g._col_var is not None:
                    print('g._col_var = {}'.format(g._col_var))
                    dataSubset = dataSubset.loc[dataSubset[g._col_var] == col_val, :]
                    statsSubset = statsSubset.loc[statsSubset[g._col_var] == col_val, :]
                elif (distanceType == 'frobenius') and (hgName == 'unscaled'):
                    dataSubset = dataSubset.loc[dataSubset['freqBandName'] == 'all', :]
                    statsSubset = statsSubset.loc[statsSubset['freqBandName'] == 'all', :]
                #
                if xOrder is not None:
                    print('xVar = {}'.format(xVar))
                    dataSubset = dataSubset.loc[dataSubset[xVar].isin(xOrder), :]
                    statsSubset = statsSubset.loc[statsSubset[xVar].isin(xOrder), :]
                #
                enableAnnotations = True
                if enableAnnotations:
                    pairs = []
                    pvalAnns = []
                    for xName, statsSubGroup in statsSubset.groupby(xVar):
                        for rowIdx, row in statsSubGroup.iterrows():
                            isThisPlotted = (row['category1'] in thisPalette.index) and (row['category2'] in thisPalette.index)
                            isThisSignificant = row['pval'] < confidence_alpha
                            if (distanceType == 'frobenius') and (hgName == 'relative to B-M'):
                                extraCondition = (row['category1'] == 'B-M') or (row['category2'] == 'B-M')
                            elif (distanceType == 'frobenius') and (hgName == 'relative to M-S'):
                                extraCondition = ((row['category1'] != 'M-S') and (row['category2'] != 'M-S'))
                            elif (distanceType == 'frobenius') and (hgName == 'unscaled'):
                                # extraCondition = (row['category1'] == 'B-B') or (row['category2'] == 'B-B')
                                extraCondition = False
                            elif (distanceType == 'frobenius') and (hgName == 'Everything'):
                                extraCondition = False
                            condition = isThisSignificant and isThisPlotted and extraCondition
                            if condition:
                                pairs.append(
                                    (
                                        (xName, row['category1']), (xName, row['category2'])
                                    )
                                )
                                pvalAnns.append(row['pval'])
                    if len(pairs):
                        try:
                            annotator = Annotator(
                                ax, pairs,
                                data=dataSubset, plot='boxplot',
                                **argsForCatPlot)
                            annotator.configure(
                                test=None, test_short_name='',
                                line_width=sns.plotting_context()['lines.linewidth'],
                                pvalue_format=dict(
                                    fontsize=sns.plotting_context()["font.size"]))
                            annotator.set_pvalues(pvalAnns).annotate()
                        except Exception:
                            traceback.print_exc()
                            pdb.set_trace()
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    newXTickLabels = [
                        applyPrettyNameLookup(tL.get_text())
                        for tL in xTickLabels]
                    ax.set_xticklabels(newXTickLabels, rotation=30, va='top', ha='right')
                if (distanceType == 'frobenius'):
                    for xJ in range(1, len(xOrder), 2):
                        ax.axvspan(-0.5 + xJ, 0.5 + xJ, color="0.1", alpha=0.1, zorder=1.)
                ax.axhline(1., c='0.5', lw=1., ls='--', zorder=.9)
                ax.set_xlim([-0.5, 0.5 + len(xOrder) - 1])
                if shareYByRow and (not ax.is_first_col()):
                    ax.set_ylabel(None)
                    ax.set_yticklabels([])
            if shareYByRow:
                for ro in range(g.axes.shape[0]):
                    allYLim = pd.DataFrame([colAx.get_ylim() for colAx in g.axes[ro, :]], columns=['inf', 'sup'])
                    newYLims = [allYLim['inf'].min(), allYLim['sup'].max()]
                    for co in range(g.axes.shape[1]):
                        g.axes[ro, co].set_ylim(newYLims)
                        if co > 0:
                            g.axes[ro, 0].get_shared_y_axes().join(g.axes[ro, 0], g.axes[ro, co])
                            # g.axes[ro, co].set_ylim(g.axes[ro, 0].get_ylim())
            # g.axes[0, 0].set_xticks([])
            # g.axes[0, 0].set_xlim([-0.5, 0.5])
            g.suptitle('{} distance ({})'.format(distanceType.capitalize(), hgName))
            g.set_axis_labels(prettyNameLookup[xVar], 'Normalized\ndistance\n(a.u.)')
            contentOverrides = {hueVar: 'Epoch  pair'}
            if hueVarLabel is not None:
                pass
            asp.reformatFacetGridLegend(
                g, titleOverrides=contentOverrides,
                contentOverrides=contentOverrides,
                styleOpts=styleOpts)
            if (distanceType == 'frobenius') and (hgName == 'Everything'):
                pvalueAnnotLegend = '''
                p-value annotation legend:
                      ns: p > 5e-02
                       *: 1e-02 < p <= 5e-02
                      **: 1e-03 < p <= 1e-02
                     ***: 1e-04 < p <= 1e-03
                    ****: p <= 1e-04
                '''
                g.axes[0, 0].text(
                    1, 1, pvalueAnnotLegend, ha='right', va='top',
                    transform=g.fig.transFigure)
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight', pad_inches=0,
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()