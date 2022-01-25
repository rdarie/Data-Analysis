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
    --memoryEfficientLoad                    print diagnostics? [default: False]
    --forceReprocess                         print diagnostics? [default: False]
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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LinearLocator)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dask.distributed import Client, LocalCluster
import os, traceback
from tqdm import tqdm
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_rd', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'pls_select_scaled', 'forceReprocess': True,
        'blockIdx': '2', 'exp': 'exp202101271100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.plotting.aligned_signal_plots as asp
from dataAnalysis.custom_transformers.tdr import reconstructionR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
from datetime import datetime
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer, r2_score
import joblib as jb
import patsy
from sklego.preprocessing import PatsyTransformer
import dill as pickle
pickle.settings['recurse'] = True
import gc, sys
from copy import deepcopy
import colorsys
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
for arg in sys.argv:
    print(arg)
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
        'lines.markersize': 2.4,
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


if __name__ == '__main__':
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')

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
        binInterval = iteratorOpts['forceBinInterval'] if (iteratorOpts['forceBinInterval'] is not None) else rasterOpts['binOpts'][loadingMeta['arguments']['analysisName']]['binInterval']
        dt = binInterval
    #
    histOptsForExportDict = {}
    for hIdx, histOpts in enumerate(addEndogHistoryTerms):
        formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
        locals().update({'enhto{}'.format(hIdx): formattedHistOpts})
        histOptsForExportDict['enhto{}'.format(hIdx)] = formattedHistOpts
    for hIdx, histOpts in enumerate(addExogHistoryTerms):
        formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
        locals().update({'exhto{}'.format(hIdx): formattedHistOpts})
        histOptsForExportDict['exhto{}'.format(hIdx)] = formattedHistOpts
    thisEnv = patsy.EvalEnvironment.capture()

    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    lastFoldIdx = cvIterator.n_splits
    #
    selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
    selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
    #
    lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
    lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
    allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets').xs(arguments['estimatorName'], level='regressorName')
    allTargetsPLS = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargetsPLS')
    allTargetsPLS.set_index(['lhsMaskIdx', 'rhsMaskIdx'], inplace=True)
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    ##
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    #
    stimConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'stimConditionLookup')
    kinConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'kinConditionLookup')
    ################ collect estimators and scores
    #
    trialTypeToPlot = 'validation'
    #
    predIndexNames = None
    R2PerIndexNames = None
    #
    memoryEfficientLoad = True
    if memoryEfficientLoad:
        predDF = None
        R2Per = None
    else:
        # predDF = None
        predList = []
        R2PerList = []
    #
    if processSlurmTaskCount is not None:
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
        allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
        lhsTargetMask = allTargetsDF.index.get_level_values('lhsMaskIdx').isin(lhsMasksOfInterest['plotPredictions'])
        jobsToLoad = allTargetsDF.loc[lhsTargetMask, 'parentProcess'].unique()
        for modelIdx in range(processSlurmTaskCount):
            if modelIdx not in jobsToLoad:
                continue
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(modelIdx))
            prf.print_memory_usage('Loading predictions from {}'.format(thisEstimatorPath))
            with pd.HDFStore(thisEstimatorPath) as store:
                thisPred = pd.read_hdf(store, 'predictions')
                if predIndexNames is None:
                    predIndexNames = thisPred.index.names
                thisPred.reset_index(inplace=True)
                maskForPlot = (
                    (thisPred['lhsMaskIdx'].isin(lhsMasksOfInterest['plotPredictions'])) &
                    (thisPred['trialType'] == trialTypeToPlot) &
                    (thisPred['trialRateInHz'] > 50)
                    )
                if maskForPlot.any():
                    thisPred = thisPred.loc[maskForPlot, :]
                    print('these predictions, thisPred.shape = {}'.format(thisPred.shape))
                    if memoryEfficientLoad:
                        if predDF is None:
                            predDF = thisPred
                        else:
                            predDF = predDF.append(thisPred)
                    else:
                        predList.append(thisPred)
                    ##
                    thisR2Per = pd.read_hdf(store, 'processedR2')
                    if R2PerIndexNames is None:
                        R2PerIndexNames = thisR2Per.index.names
                    thisR2Per.reset_index(inplace=True)
                    maskForPlotScore = (
                            (thisR2Per['lhsMaskIdx'].isin(lhsMasksOfInterest['plotPredictions'])) &
                            (thisR2Per['trialType'] == trialTypeToPlot)
                        )
                    print('these R2, thisR2.shape = {}'.format(thisR2Per.shape))
                    if memoryEfficientLoad:
                        if R2Per is None:
                            R2Per = thisR2Per
                        else:
                            R2Per = R2Per.append(thisR2Per)
                    else:
                        R2PerList.append(thisR2Per)
    #  else:
    #      print('Loading predictions from {}'.format(estimatorPath))
    #      thisPred = pd.read_hdf(estimatorPath, 'predictions')
    #      predList.append(thisPred)
    ###
    prf.print_memory_usage('concatenating predictions from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        predDF = pd.concat(predList, copy=False)
        del predList
    predDF.set_index(predIndexNames, inplace=True)
    print('all predictions, predDF.shape = {}'.format(predDF.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating predictions from .h5 array')
    ###
    prf.print_memory_usage('concatenating R2 from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        R2Per = pd.concat(R2PerList, copy=False)
        del R2PerList
    R2Per.set_index(R2PerIndexNames, inplace=True)
    print('all predictions, R2PerDF.shape = {}'.format(R2Per.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating R2 from .h5 array')
    #
    estimatorsDict = {}
    # scoresDict = {}
    for rowIdx, row in allTargetsDF.iterrows():
        lhsMaskIdx, rhsMaskIdx, targetName = row.name
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        freqBandName = rhsMaskParams['freqBandName']
        if processSlurmTaskCount is not None:
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
            print('Loading data from {}'.format(thisEstimatorPath))
        else:
            thisEstimatorPath = estimatorPath
        thisEstimatorJBPath = os.path.join(
            thisEstimatorPath.replace('.h5', ''),
            'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
            ))
        thisEstimatorJBDict = jb.load(thisEstimatorJBPath)
        thisEstimatorJB = pd.Series(thisEstimatorJBDict)
        thisEstimatorJB.index.name = 'fold'
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = thisEstimatorJB
    prf.print_memory_usage('concatenating estimators from .h5 array')
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    del estimatorsDict
    prf.print_memory_usage('done concatenating estimators from .h5 array')
    with pd.HDFStore(estimatorPath) as store:
        coefDF = pd.read_hdf(store, 'coefficients')
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
        predictionLineStyleDF = pd.read_hdf(store, 'termLineStyleDF')
    #
    targetRHSLookup = (
        allTargetsDF.index.to_frame()
            .reset_index(drop=True)[['rhsMaskIdx', 'target']]
            .drop_duplicates().set_index('target')['rhsMaskIdx']
        )
    scoresFromLL = R2Per.xs('all', level='electrode').reset_index()
    scoresFromLL.loc[:, 'rhsMaskIdx'] = scoresFromLL['target'].map(targetRHSLookup)
    scoresFromLL.sort_values(
        ['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'],
        kind='mergesort', inplace=True)
    scoresFromLL.loc[:, 'fullFormulaDescr'] = scoresFromLL['lhsMaskIdx'].map(
        lhsMasksInfo['fullFormulaDescr']).to_numpy()
    scoresFromLL.loc[:, 'fullFormulaAsLabel'] = scoresFromLL['fullFormulaDescr'].apply(lambda x: x.replace(' + ', ' +\n'))
    featsToPlot = (
        scoresFromLL.loc[scoresFromLL['fullFormulaDescr'].str.contains('NULL'), :].groupby('target').mean()['score'].idxmax(),
        scoresFromLL.loc[scoresFromLL['fullFormulaDescr'].str.contains('NULL'), :].groupby('target').mean()['score'].idxmin())
    #
    height, width = 2, 4
    aspect = width / height
    commonOpts = dict(
        )
    groupPagesBy = ['rhsMaskIdx', 'lhsMaskIdx']
    groupSubPagesBy = ['trialType', 'electrode', 'trialRateInHz', 'target']
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'reconstructions_single_trial'))
    with PdfPages(pdfPath) as pdf:
        for name0, predGroup0 in predDF.groupby(groupPagesBy):
            nmLk0 = {key: value for key, value in zip(groupPagesBy, name0)}  # name lookup
            nmLk0['design'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'designFormula']
            nmLk0['fullFormulaDescr'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'fullFormulaDescr']
            if not (nmLk0['lhsMaskIdx'] in lhsMasksOfInterest['plotPredictions']):
                continue
            scoreMasks = [
                scoresFromLL[cN] == nmLk0[cN]
                for cN in groupPagesBy]
            plotScores = scoresFromLL.loc[np.logical_and.reduce(scoreMasks), :]
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            g = sns.catplot(
                data=plotScores, hue='trialType',
                x='target', y='score',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=height, aspect=aspect,
                kind='box')
            g.set_xticklabels(rotation=-30, ha='left')
            g.suptitle('R^2 of model {fullFormulaDescr}'.format(**nmLk0))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ####
            '''for name1, predGroup1 in predGroup0.groupby(groupSubPagesBy):
                nmLk1 = {key: value for key, value in zip(groupSubPagesBy, name1)}  # name lookup
                nmLk0.update(nmLk1)
                if not (nmLk0['target'] in featsToPlot):
                    continue
                # if nmLk0['trialType'] != trialTypeToPlot:
                #     continue
                # if nmLk0['trialRateInHz'] < 100:
                #     continue'''
            # rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            annScoreLookup = plotScores.loc[plotScores['trialType'] == trialTypeToPlot, ['target', 'score']].set_index('target')
            plotDF = (predGroup0.xs(100, level='trialRateInHz', axis='index', drop_level=False).xs(trialTypeToPlot, level='trialType', axis='index', drop_level=False))
            plotDF.sort_index(level=['trialUID', 'target', 'bin'], kind='mergesort', sort_remaining=False, inplace=True)
            trialInfo = plotDF.index.to_frame().reset_index(drop=True)
            trialInfo.loc[:, 'groupUID'] = trialInfo['trialUID'].map({v: k for k, v in enumerate(np.unique(trialInfo['trialUID']))})
            trialDurs = trialInfo.groupby('groupUID').max()['bin'] - trialInfo.groupby('groupUID').min()['bin']
            trialTOffsets = trialDurs.shift(1).fillna(0).cumsum()
            trialInfo.loc[:, 't'] = trialInfo['bin'] - trialInfo['bin'].min() + trialInfo['groupUID'].map(trialTOffsets)
            confPlotWinSize = 10. # seconds
            plotRounds = trialInfo['t'].apply(lambda x: np.floor(x / confPlotWinSize))
            targetChans = annScoreLookup.sort_values(by='score').index
            for pr in tqdm(plotRounds.unique()):
                plotRoundMask = (plotRounds == pr).to_numpy()
                fig, ax = plt.subplots(
                    1, 1, figsize=(confPlotWinSize * 3, targetChans.size))
                extraArtists = []
                yTickLocs = []
                for cIdx, cN in enumerate(targetChans):
                    scaler = MinMaxScaler()
                    thisColor = sourcePalette.loc[cN]
                    plotMask = (trialInfo['target'] == cN).to_numpy() & plotRoundMask
                    plotTrace = plotDF['ground_truth'].iloc[plotMask].to_numpy()
                    plotTrace = scaler.fit_transform(plotTrace.reshape(-1, 1)).flatten() + cIdx * 1
                    thisT = trialInfo.loc[plotMask, 't'].to_numpy()
                    ax.plot(thisT, plotTrace, label=cN, alpha=1., ls='-', lw=0.5, c=thisColor)
                    baselineLoc = scaler.transform(np.asarray([0.]).reshape(1, -1))[0, 0] + cIdx * 1
                    yTickLocs.append(baselineLoc) # cIdx * 1. + 0.5
                    traceLabel = ax.text(np.min(thisT), baselineLoc, '{} ($R^2$ = {:.2f})'.format(cN, annScoreLookup.loc[cN, 'score']), va='center', ha='right')
                    extraArtists.append(traceLabel)
                    plotTraceP = plotDF['prediction'].iloc[plotMask].to_numpy()
                    plotTraceP = scaler.transform(plotTraceP.reshape(-1, 1)).flatten() + cIdx * 1
                    ax.plot(thisT, plotTraceP, label=cN, alpha=0.75, ls='--', lw=0.5, c=thisColor)
                    ax.set_xlim([thisT.min(), thisT.max()])
                xTickLocs = []
                for name, group in trialInfo.loc[plotRoundMask, :].groupby('groupUID'):
                    ax.axvline(group['t'].min(), c='k')
                    xTickLocs.append(group['t'].min())
                    if group['bin'].min() < 0:
                        ax.axvspan(
                            group['t'].min() - group['bin'].min(),
                            group['t'].max(), alpha=0.1,
                            color='red', zorder=-10)
                    metaInfo = group.iloc[0, :]
                    captionText = (
                        'movement category = {}\n'.format(metaInfo['pedalMovementCat']) +
                        'stim electrode = {}\n'.format(metaInfo['electrode']) +
                        'stim amplitude = {} uA\n'.format(metaInfo['trialAmplitude']) +
                        'stim rate = {} Hz\n'.format(metaInfo['trialRateInHz'])
                        )
                    axCaption = ax.text(group['t'].min(), ax.get_ylim()[1], captionText, ha='left', va='bottom')
                    extraArtists.append(axCaption)
                figTitle = fig.supylabel('{}'.format(metaInfo['design']))
                extraArtists.append(figTitle)
                # fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                # ax.legend(loc='lower left')
                ax.set_xlabel('Time (s)')
                yTickLocs = np.unique(np.round(yTickLocs, decimals=3)).tolist()
                ax.set_yticks(yTickLocs)
                ax.set_ylim([0, (targetChans.size + 1) * 1.])
                ax.set_yticklabels(['' for yt in yTickLocs])
                xTickLocs = np.unique(np.round(xTickLocs, decimals=3)).tolist()
                ax.xaxis.set_minor_locator(LinearLocator(5))
                ax.set_xticks(xTickLocs)
                fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                # extraArtists.append(ax.get_legend())
                figSaveOpts = dict(
                    bbox_extra_artists=tuple(extraArtists),
                    bbox_inches='tight')
                pdf.savefig(**figSaveOpts)
                plt.close()
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')