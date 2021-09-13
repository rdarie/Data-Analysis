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
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dask.distributed import Client, LocalCluster
import os, traceback

# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_rd', 'plotting': True,
        'showFigures': True, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'pls_select_scaled', 'forceReprocess': True,
        'blockIdx': '2', 'exp': 'exp202101271100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
# otherwise, get arguments from console:

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))
# from dataAnalysis.analysis_code.regression_parameters_history_len_determine import *
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
        binInterval = iteratorOpts['forceBinInterval'] if (iteratorOpts['forceBinInterval'] is not None) else rasterOpts['binInterval']
    #
    histOptsForExportDict = {}
    for hIdx, histOpts in enumerate(addHistoryTerms):
        formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
        locals().update({'hto{}'.format(hIdx): formattedHistOpts})
        histOptsForExportDict['hto{}'.format(hIdx)] = formattedHistOpts
        # locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    histOptsForExportDF = pd.DataFrame(histOptsForExportDict)
    histOptsHtmlPath = os.path.join(
        figureOutputFolder, '{}_{}.html'.format(fullEstimatorName, 'histOpts'))
    histOptsForExportDF.to_html(histOptsHtmlPath)
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
    allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets')
    allTargetsPLS = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargetsPLS')
    allTargetsPLS.set_index(['lhsMaskIdx', 'rhsMaskIdx'], inplace=True)
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    #
    stimConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'stimConditionLookup')
    kinConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'kinConditionLookup')
    ##
    ################ collect estimators and scores
    #
    llIndexNames = None
    aicIndexNames = None
    R2PerIndexNames = None
    #
    memoryEfficientLoad = True
    if memoryEfficientLoad:
        scoresStack = None
        llDF = None
        aicDF = None
        R2Per = None
    else:
        #
        scoresStackList = []
        llList = []
        aicList = []
        R2PerList = []
    #
    if processSlurmTaskCountPLS is not None:
        slurmGroupSize = int(np.ceil(allTargetsPLS.shape[0] / processSlurmTaskCountPLS))
        allTargetsPLS.loc[:, 'parentProcess'] = allTargetsPLS['targetIdx'] // slurmGroupSize
        for modelIdx in range(processSlurmTaskCountPLS):
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(modelIdx))
            prf.print_memory_usage('Loading predictions from {}'.format(thisEstimatorPath))
            with pd.HDFStore(thisEstimatorPath) as store:
                ##
                thisScoresStack = pd.read_hdf(store, 'processedCVScores')
                print('these scoresStack, thisScoresStack.shape = {}'.format(thisScoresStack.shape))
                if memoryEfficientLoad:
                    if scoresStack is None:
                        scoresStack = thisScoresStack
                    else:
                        scoresStack = scoresStack.append(thisScoresStack)
                else:
                    scoresStackList.append(thisScoresStack)
                ##
                thisLl = pd.read_hdf(store, 'processedLogLike')
                if llIndexNames is None:
                    llIndexNames = thisLl.index.names
                thisLl.reset_index(inplace=True)
                print('these ll, thisLl.shape = {}'.format(thisLl.shape))

                if memoryEfficientLoad:
                    if llDF is None:
                        llDF = thisLl
                    else:
                        llDF = llDF.append(thisLl)
                else:
                    llList.append(thisLl)

                ##
                thisAic = pd.read_hdf(store,  'processedAIC')
                if aicIndexNames is None:
                    aicIndexNames = thisAic.index.names
                thisAic.reset_index(inplace=True)
                print('these Aic, thisAic.shape = {}'.format(thisAic.shape))

                if memoryEfficientLoad:
                    if aicDF is None:
                        aicDF = thisAic
                    else:
                        aicDF = aicDF.append(thisAic)
                else:
                    aicList.append(thisAic)

                #
                thisR2Per = pd.read_hdf(store, 'processedR2')
                if R2PerIndexNames is None:
                    R2PerIndexNames = thisR2Per.index.names
                thisR2Per.reset_index(inplace=True)
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
    prf.print_memory_usage('concatenating ll from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        llDF = pd.concat(llList, copy=False)
        del llList
    llDF.set_index(llIndexNames, inplace=True)
    print('all l, llDF.shape = {}'.format(llDF.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating ll from .h5 array')
    ###
    prf.print_memory_usage('concatenating aic from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        aicDF = pd.concat(aicList, copy=False)
        del aicList
    aicDF.set_index(aicIndexNames, inplace=True)
    print('all predictions, predDF.shape = {}'.format(aicDF.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating aic from .h5 array')
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
    ###
    prf.print_memory_usage('concatenating scoresStack from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        scoresStack = pd.concat(scoresStackList, copy=False)
        del scoresStackList
    print('all scoresStack, scoresStack.shape = {}'.format(scoresStack.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating scoresStack from .h5 array')
    #
    estimatorsDict = {}
    # scoresDict = {}
    for rowIdx, row in allTargetsPLS.iterrows():
        lhsMaskIdx, rhsMaskIdx = row.name
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        freqBandName = rhsMaskParams['freqBandName']
        if processSlurmTaskCountPLS is not None:
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
            print('Loading data from {}'.format(thisEstimatorPath))
        else:
            thisEstimatorPath = estimatorPath
        '''estimatorsDict[(lhsMaskIdx, rhsMaskIdx, freqBandName)] = pd.read_hdf(
            thisEstimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, freqBandName
                ))'''
        thisEstimatorJBPath = os.path.join(
            thisEstimatorPath.replace('.h5', ''),
            'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                lhsMaskIdx, rhsMaskIdx, freqBandName
            ))
        thisEstimatorJBDict = jb.load(thisEstimatorJBPath)
        thisEstimatorJB = pd.Series(thisEstimatorJBDict)
        thisEstimatorJB.index.name = 'fold'
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, freqBandName)] = thisEstimatorJB
    prf.print_memory_usage('concatenating estimators from .h5 array')
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    del estimatorsDict
    prf.print_memory_usage('done concatenating estimators from .h5 array')
    # prf.print_memory_usage('concatenating scores from .h5 array')
    # scoresDF = pd.concat(scoresDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    # del scoresDict
    # prf.print_memory_usage('done concatenating scores from .h5 array')
    # gc.collect()
    #
    with pd.HDFStore(estimatorPath) as store:
        #
        coefDF = pd.read_hdf(store, 'coefficients')
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
        if ('modelCompareFUDE' in store) and (not arguments['forceReprocess']):
            modelCompareFUDE = pd.read_hdf(store, 'modelCompareFUDE')
        if ('modelCompareFUDEStats' in store) and (not arguments['forceReprocess']):
            modelCompareFUDEStats = pd.read_hdf(store, 'modelCompareFUDEStats')
        if ('modelCompareScores' in store) and (not arguments['forceReprocess']):
            modelCompareScores = pd.read_hdf(store, 'modelCompareScores')
    FUDEDict = {}
    FUDEStatsDict = {}
    ScoresDict = {}
    for modelIdx, modelToTest in modelsToTestDF.iterrows():
        testDesign = modelToTest['testDesign']
        refDesign = modelToTest['refDesign']
        ##
        theseFUDE = tdr.partialR2(llDF['ll'], refDesign=refDesign, testDesign=testDesign, designLevel='lhsMaskIdx')
        theseFUDEStats = tdr.correctedResampledPairedTTest(
            theseFUDE.xs('test', level='trialType'), y=0., groupBy=['target', 'electrode'], cvIterator=cvIterator)
        comparisonParams = tuple(modelToTest.loc[['testDesign', 'refDesign', 'testType', 'lagSpec', 'testHasEnsembleHistory']])
        FUDEDict[comparisonParams] = theseFUDE
        FUDEStatsDict[comparisonParams] = theseFUDEStats
        theseTestScores = tdr.partialR2(
            llDF['ll'], refDesign=None, testDesign=testDesign,
            designLevel='lhsMaskIdx').reset_index().rename(columns={'ll': 'test_score'})
        theseRefScores = tdr.partialR2(
            llDF['ll'], refDesign=None, testDesign=refDesign,
            designLevel='lhsMaskIdx').reset_index().rename(columns={'ll': 'ref_score'})
        assert theseTestScores.shape == theseRefScores.shape
        checkSameOn = ['target', 'fold', 'electrode']
        assert (theseRefScores.loc[:, checkSameOn] == theseTestScores.loc[:, checkSameOn]).all().all()
        theseTestScores.loc[:, 'ref_score'] = theseRefScores['ref_score'].to_numpy()
        ScoresDict[comparisonParams] = theseTestScores
    #
    modelCompareFUDE = pd.concat(FUDEDict, names=['testLhsMaskIdx', 'refLhsMaskIdx', 'testType', 'lagSpec', 'testHasEnsembleHistory'])
    modelCompareFUDE.name = 'score'
    modelCompareFUDEStats = pd.concat(FUDEStatsDict, names=['testLhsMaskIdx', 'refLhsMaskIdx', 'testType', 'lagSpec', 'testHasEnsembleHistory'])
    modelCompareScores = pd.concat(ScoresDict, names=['testLhsMaskIdx', 'refLhsMaskIdx', 'testType', 'lagSpec', 'testHasEnsembleHistory'])
    modelCompareFUDE.to_hdf(estimatorPath, 'modelCompareFUDE')
    modelCompareFUDEStats.to_hdf(estimatorPath, 'modelCompareFUDEStats')
    modelCompareScores.to_hdf(estimatorPath, 'modelCompareScores')
    print('Loaded and saved scores and partial scores')
    targetRHSLookup = allTargetsDF.index.to_frame().reset_index(drop=True)[['rhsMaskIdx', 'target']].drop_duplicates().set_index('target')['rhsMaskIdx']
    plotAIC = aicDF.xs('all', level='electrode').reset_index()
    plotAIC.loc[:, 'fullDesignAsLabel'] = plotAIC['fullFormulaDescr'].apply(lambda x: x.replace(' + ', ' +\n'))
    plotAIC.loc[:, 'rhsMaskIdx'] = plotAIC['target'].map(targetRHSLookup)
    #
    scoresFromLL = R2Per.xs('all', level='electrode').reset_index()
    scoresFromLL.loc[:, 'rhsMaskIdx'] = scoresFromLL['target'].map(targetRHSLookup)
    #
    scoresStack = scoresStack.loc[scoresStack['fold'] != lastFoldIdx, :]
    plotAIC = plotAIC.loc[plotAIC['fold'] != lastFoldIdx, :]
    scoresFromLL = scoresFromLL.loc[scoresFromLL['fold'] != lastFoldIdx, :]
    #
    plotAIC.loc[:, 'dAIC'] = np.nan
    for name, group in plotAIC.groupby(['rhsMaskIdx', 'target', 'trialType']):
        # print('{}, {}'.format(name, group['aic'].min()))
        plotAIC.loc[group.index, 'dAIC'] = group['aic'] - group['aic'].min()
    #
    scoresStack.sort_values(['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'], kind='mergesort', inplace=True)
    plotAIC.sort_values(['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'], kind='mergesort', inplace=True)
    scoresFromLL.sort_values(['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'], kind='mergesort', inplace=True)
    #
    scoresFromLL.loc[:, 'aic'] = plotAIC['aic'].to_numpy()
    scoresFromLL.loc[:, 'dAIC'] = plotAIC['dAIC'].to_numpy()
    scoresFromLL.loc[:, 'fullDesignAsLabel'] = plotAIC['fullDesignAsLabel'].to_numpy()

    def drawUnityLine(g, ro, co, hu, dataSubset):
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset.iloc[:, 0].isna().all()))
        if not hasattr(g.axes[ro, co], 'axHasUnityLine'):
            g.axes[ro, co].axHasUnityLine = True
            if not emptySubset:
                currXLim, currYLim = g.axes[ro, co].get_xlim(), g.axes[ro, co].get_ylim()
                leftEdge = min(currXLim[0], currYLim[0])
                rightEdge = max(currXLim[1], currYLim[1])
                g.axes[ro, co].plot(
                    [leftEdge, rightEdge], [leftEdge, rightEdge], '-',
                    c=(0., 0., 0., 0.75), zorder=1.9)
                g.axes[ro, co].set_xlim(currXLim)
                g.axes[ro, co].set_ylim(currYLim)
        return

    def annotateWithPVal(g, ro, co, hu, dataSubset):
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset.iloc[:, 0].isna().all()))
        if not hasattr(g.axes[ro, co], 'axHasPValAnnotation'):
            g.axes[ro, co].axHasPValAnnotation = True
            if not emptySubset:
                nSig = dataSubset.loc[dataSubset['fold'] == 0., 'significant'].sum()
                nTot = dataSubset.loc[dataSubset['fold'] == 0., 'significant'].shape[0]
                messageStr = 'n = {}/{} significant'.format(nSig, nTot)
                g.axes[ro, co].text(
                    0.9, 1, messageStr, ha='right', va='top',
                    fontsize=snsRCParams['font.size'], transform=g.axes[ro, co].transAxes)
            return

    def annotateWithQuantile(g, ro, co, hu, dataSubset):
        qMin, qMed, qMax = 0.25, 0.5, 0.75
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset.iloc[:, 0].isna().all()))
        if not hasattr(g.axes[ro, co], 'axHasQuantileAnnotation'):
            g.axes[ro, co].axHasQuantileAnnotation = True
            if not emptySubset:
                qMinVal, qMedVal, qMaxVal = dataSubset['parameter'].quantile([qMin, qMed, qMax])
                messageStr = 'quantiles([{:.3g}-{:.3g}-{:.3g}]) = {:.3g}-{:.3g}-{:.3g}'.format(qMin, qMed, qMax, qMinVal, qMedVal, qMaxVal)
                g.axes[ro, co].text(
                    0.9, 1, messageStr, ha='right', va='top',
                    fontsize=snsRCParams['font.size'], transform=g.axes[ro, co].transAxes)
            return
    #
    pdfPath = os.path.join(
        figureOutputFolder,
        '{}_{}.pdf'.format(fullEstimatorName, 'partial_scores'))
    with PdfPages(pdfPath) as pdf:
        height, width = 3, 3
        aspect = width / height
        # maskSecondOrderTests = modelsToTestDF['testType'] == 'secondOrderInteractions'
        for testTypeName, modelsToTestGroup in modelsToTestDF.groupby('testType'):
            if 'captionStr' in modelsToTestGroup:
                titleText = modelsToTestGroup['captionStr'].iloc[0]
            else:
                titleText = 'partial R2 scores for {} compared to {}'.format(
                    modelsToTestGroup['testDesign'].iloc[0], modelsToTestGroup['refDesign'].iloc[0])
            if 'refCaption' in modelsToTestGroup:
                refCaption = modelsToTestGroup['refCaption'].iloc[0]
            else:
                refCaption = lhsMasksInfo.loc[modelsToTestGroup['refDesign'].iloc[0], 'fullFormulaDescr']
            if 'testCaption' in modelsToTestGroup:
                testCaption = modelsToTestGroup['testCaption'].iloc[0]
            else:
                testCaption = lhsMasksInfo.loc[modelsToTestGroup['testDesign'].iloc[0], 'fullFormulaDescr']
            #
            plotFUDE = modelCompareFUDE.xs(testTypeName, level='testType').xs('all', level='electrode').reset_index()
            plotFUDE = plotFUDE.loc[plotFUDE['trialType'].isin(['test']), :]
            plotFUDEStats = modelCompareFUDEStats.xs(testTypeName, level='testType').xs('all', level='electrode').reset_index()
            plotScores = modelCompareScores.loc[modelCompareScores['electrode'] == 'all'].xs(testTypeName, level='testType').reset_index()
            plotScores = plotScores.loc[plotScores['trialType'].isin(['test']), :]
            #
            lookupBasedOn = ['testLhsMaskIdx', 'refLhsMaskIdx', 'target']
            lookupAt = pd.MultiIndex.from_frame(plotScores.loc[:, lookupBasedOn])
            lookupFrom = plotFUDEStats.loc[:, lookupBasedOn + ['p-val']].set_index(lookupBasedOn)['p-val']
            plotPVals = lookupFrom.loc[lookupAt]
            plotScores.loc[:, 'significant'] = (plotPVals < 0.01).to_numpy()
            ###
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            g = sns.catplot(
                data=plotFUDE, kind='box',
                y='score', x='target', hue='trialType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=height, aspect=aspect,
                sharey=True
                )
            g.set_xticklabels(rotation=30, ha='right')
            g.set_titles(template="data subset: {col_name}")
            g.suptitle(titleText)
            g.set_axis_labels('target signal', '{} R2 w.r.t. {}'.format(testCaption, refCaption))
            print('Saving {}\n to {}'.format(titleText, pdfPath))
            # g.axes.flat[0].set_ylim(allScoreQuantiles)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            tc = thisPalette.loc['test']
            signiPalette = {
                True: (tc[0], tc[1], tc[2], .75),
                False: (tc[0], tc[1], tc[2], .25)
                }
            #
            g = sns.relplot(
                data=plotScores, kind='scatter',
                y='test_score', x='ref_score',
                hue='significant',
                height=height, aspect=aspect,
                edgecolor=None,
                hue_order=[True, False],
                palette=signiPalette,
                style='target',
                # hue_order=thisPalette.index.to_list(),
                # palette=thisPalette.to_dict(),
                )
            g.set_axis_labels(refCaption, testCaption)
            g.set_titles(template="data subset: {col_name}")
            g.suptitle(titleText)
            plotProcFuns = [drawUnityLine, annotateWithPVal]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            print('Saving {}\n to {}'.format(titleText, pdfPath))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    #
    # estimatorsDF.iloc[0].regressor_.named_steps['regressor'].results_.summary()
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'r2'))
    with PdfPages(pdfPath) as pdf:
        height, width = 3, 4
        aspect = width / height
        for rhsMaskIdx, plotScores in scoresFromLL.groupby(['rhsMaskIdx'], sort=False):
            rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            #
            for annotationName in ['historyLen', 'designFormula', 'ensembleTemplate']:
                plotScores.loc[:, annotationName] = plotScores['lhsMaskIdx'].map(lhsMasksInfo[annotationName])
            #
            plotScores.loc[:, 'designType'] = ''
            thisDesignTypeMask = (
                    (plotScores['designFormula'] == 'NULL') &
                    (plotScores['ensembleTemplate'] == 'NULL')
                )
            assert (not thisDesignTypeMask.any())
            thisDesignTypeMask = (
                    (plotScores['designFormula'] == 'NULL') &
                    (plotScores['ensembleTemplate'] != 'NULL')
                )
            plotScores.loc[thisDesignTypeMask, 'designType'] = 'ensembleOnly'
            thisDesignTypeMask = (
                    (plotScores['designFormula'] != 'NULL') &
                    (plotScores['ensembleTemplate'] == 'NULL')
                )
            plotScores.loc[thisDesignTypeMask, 'designType'] = 'exogenousOnly'
            thisDesignTypeMask = (
                    (plotScores['designFormula'] != 'NULL') &
                    (plotScores['ensembleTemplate'] != 'NULL')
                )
            plotScores.loc[thisDesignTypeMask, 'designType'] = 'ensembleAndExogenous'
            #
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            g = sns.catplot(
                data=plotScores,
                y='score', hue='trialType',
                x='fullDesignAsLabel',
                col='target', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect, sharey=False)
            g.suptitle('R2 (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            g = sns.catplot(
                data=plotScores.loc[plotScores['trialType'] == 'train', :],
                hue='trialType',
                y='dAIC',
                x='fullDesignAsLabel',
                col='target', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect, sharey=False)
            g.suptitle('AIC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            g = sns.catplot(
                data=plotScores,
                y='score', hue='trialType',
                x='fullDesignAsLabel', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect, sharey=False)
            g.suptitle('R2 (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            g = sns.catplot(
                data=plotScores.loc[plotScores['trialType'] == 'train', :],
                y='dAIC', hue='trialType',
                x='fullDesignAsLabel', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect, sharey=False)
            g.suptitle('AIC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

    '''
    pName = 'alpha'
    estimatorsDF.apply(
        lambda x: print(x.regressor_.named_steps['regressor'].get_params()[pName]))
    estimatorsDF.apply(
        lambda x: print(x.regressor_.named_steps['regressor']))
    '''
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'hyperparameters'))
    with PdfPages(pdfPath) as pdf:
        for pName in ['L1_wt', 'alpha', 'n_components']:
            try:
                hyperParams = estimatorsDF.apply(
                    lambda x: x.regressor_.named_steps['regressor'].get_params()[pName])
                titleText = pName
                plotDF = hyperParams.to_frame(name='parameter').reset_index()
                g = sns.catplot(
                    x='rhsMaskIdx', y='parameter',
                    col='rhsMaskIdx', data=plotDF, kind='violin'
                    )
                g.suptitle(titleText)
                plotProcFuns = [annotateWithQuantile]
                for (ro, co, hu), dataSubset in g.facet_data():
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                print('Saving {}\n to {}'.format(titleText, pdfPath))
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                pdf.savefig(
                    bbox_inches='tight',
                    )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            except Exception:
                traceback.print_exc()
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')