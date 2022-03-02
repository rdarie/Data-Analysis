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
        'lines.linewidth': .5,
        'lines.markersize': 1.2,
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
    for hIdx, histOpts in enumerate(addEndogHistoryTerms):
        formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
        locals().update({'enhto{}'.format(hIdx): formattedHistOpts})
    for hIdx, histOpts in enumerate(addExogHistoryTerms):
        formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
        locals().update({'exhto{}'.format(hIdx): formattedHistOpts})
    thisEnv = patsy.EvalEnvironment.capture()

    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    lastFoldIdx = cvIterator.n_splits
    #
    selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
    selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
    #
    # lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
    # pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsDF')
    lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
    allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets').xs(arguments['estimatorName'], level='regressorName')
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    lhsMasksInfo.loc[:, 'fullFormulaShort'] = lhsMasksInfo['fullFormulaDescr'].apply(lambda x: x.replace('rcb(', '(').replace(', **exhto', ', h').replace(', **enhto', ', h'))
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    #
    rhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(selectionNameRhs))
    targetInfo = rhsDF.columns.to_frame().reset_index(drop=True).set_index('feature')
    del rhsDF
    gc.collect()
    #
    lhsMasksPresent = allTargetsDF.index.get_level_values('lhsMaskIdx').unique()
    modelsValidMask = (modelsToTestDF['testDesign'].isin(lhsMasksPresent)) & (modelsToTestDF['refDesign'].isin(lhsMasksPresent))
    modelsToTestDF = modelsToTestDF.loc[modelsValidMask, :]
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
        ccDF = None
        R2Per = None
    else:
        #
        scoresStackList = []
        llList = []
        aicList = []
        ccList = []
        R2PerList = []
    #
    if processSlurmTaskCount is not None:
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
        allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
        for modelIdx in range(processSlurmTaskCount):
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(modelIdx))
            try:
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

                    ##
                    thisCC = pd.read_hdf(store,  'processedCC')
                    thisCC.reset_index(inplace=True)
                    print('these CC, thisCC.shape = {}'.format(thisCC.shape))

                    if memoryEfficientLoad:
                        if aicDF is None:
                            aicDF = thisAic
                        else:
                            aicDF = aicDF.append(thisAic)
                    else:
                        aicList.append(thisAic)
                    if memoryEfficientLoad:
                        if ccDF is None:
                            ccDF = thisCC
                        else:
                            ccDF = ccDF.append(thisCC)
                    else:
                        ccList.append(thisCC)

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
                if arguments['verbose']:
                    prf.print_memory_usage('Loaded predictions from {}'.format(thisEstimatorPath))
            except:
                traceback.print_exc()
    #  else:
    #      print('Loading predictions from {}'.format(estimatorPath))
    #      thisPred = pd.read_hdf(estimatorPath, 'predictions')
    #      predList.append(thisPred)
    ###
    prf.print_memory_usage('concatenating ll from .h5 array')
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
        ccDF = pd.concat(ccList, copy=False)
        del ccList
    aicDF.set_index(aicIndexNames, inplace=True)
    ccDF.set_index(aicIndexNames, inplace=True)
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
    ##
    estimatorsDict = {}
    gsScoresDict = {}
    nItersDict = {}
    convergenceDict = {}
    for rowIdx, row in allTargetsDF.iterrows():
        lhsMaskIdx, rhsMaskIdx, targetName = row.name
        if processSlurmTaskCount is not None:
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
            # print('Loading data from {}'.format(thisEstimatorPath))
        else:
            thisEstimatorPath = estimatorPath
        # scoresDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
        #     thisEstimatorPath,
        #     'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
        #         lhsMaskIdx, rhsMaskIdx, targetName
        #         ))
        '''estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            thisEstimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))'''
        thisEstimatorJBPath = os.path.join(
            thisEstimatorPath.replace('.h5', ''),
            'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
            ))
        try:
            thisEstimatorJBDict = jb.load(thisEstimatorJBPath)
            thisEstimatorJB = pd.Series(thisEstimatorJBDict)
            thisEstimatorJB.index.name = 'fold'
            estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = thisEstimatorJB
            ####
            convergenceInfo = pd.concat(
                (
                    thisEstimatorJB
                    .apply(lambda x: x.regressor_.named_steps['regressor'].convergence_history)
                    .to_dict()),
                names=['fold', 'filler']
                ).reset_index().drop(columns=['filler'])
            convergenceInfo.loc[:, 'scaledCost'] = MinMaxScaler().fit_transform(convergenceInfo.loc[:, ['cost']])
            convergenceDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = convergenceInfo
            #####
            gsScoresDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
                thisEstimatorPath, 'gs_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                    lhsMaskIdx, rhsMaskIdx, targetName
                ))
        except:
            traceback.print_exc()
        try:
            thisNIters = pd.read_hdf(
                thisEstimatorPath, 'cv_estimators/lhsMask_{}/rhsMask_{}/{}/cv_estimators_n_iter'.format(
                    lhsMaskIdx, rhsMaskIdx, targetName
                ))
            nItersDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = thisNIters
        except:
            pass
    ##
    savingResults = True
    prf.print_memory_usage('concatenating estimators from .h5 array')
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    del estimatorsDict
    prf.print_memory_usage('done concatenating estimators from .h5 array')
    gsScoresDF = pd.concat(gsScoresDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    gsScoresDF.drop(columns=['lhsMaskIdx', 'rhsMaskIdx'], inplace=True)
    gsScoresDF.loc[:, 'cc'] = np.sqrt(gsScoresDF['score'])
    del gsScoresDict
    prf.print_memory_usage('done concatenating gs scores from h5 file')
    if len(nItersDict):
        nItersDF = pd.concat(nItersDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    else:
        nItersDF = None
    if len(convergenceDict):
        convergenceDF = pd.concat(convergenceDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    else:
        convergenceDF = None
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
    if savingResults:
        modelCompareFUDE.to_hdf(estimatorPath, 'modelCompareFUDE')
        modelCompareFUDEStats.to_hdf(estimatorPath, 'modelCompareFUDEStats')
        modelCompareScores.to_hdf(estimatorPath, 'modelCompareScores')
    print('Loaded and saved scores and partial scores')
    plotAIC = aicDF.xs('all', level='electrode').reset_index()
    plotAIC.loc[:, 'fullDesignAsLabel'] = plotAIC['fullFormulaDescr'].apply(lambda x: x.replace(' + ', ' +\n'))
    plotAIC.loc[:, 'rhsMaskIdx'] = plotAIC['target'].map(scoresStack[['rhsMaskIdx', 'target']].drop_duplicates().set_index('target')['rhsMaskIdx'])
    #
    plotAIC.loc[:, 'dAIC'] = np.nan
    for name, group in plotAIC.groupby(['rhsMaskIdx', 'target', 'trialType']):
        plotAIC.loc[group.index, 'dAIC'] = group['aic'] - group['aic'].min()
    scoresStack.sort_values(['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'], kind='mergesort', inplace=True)
    plotAIC.sort_values(['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'], kind='mergesort', inplace=True)
    #
    plotCC = ccDF.xs('all', level='electrode').reset_index()
    plotCC.loc[:, 'rhsMaskIdx'] = plotCC['target'].map(scoresStack[['rhsMaskIdx', 'target']].drop_duplicates().set_index('target')['rhsMaskIdx'])
    plotCC.sort_values(['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'trialType'], kind='mergesort', inplace=True)
    #
    scoresStack.loc[:, 'aic'] = plotAIC['aic'].to_numpy()
    scoresStack.loc[:, 'dAIC'] = plotAIC['dAIC'].to_numpy()
    scoresStack.loc[:, 'cc'] = plotCC['cc'].to_numpy()
    if savingResults:
        scoresStack.to_hdf(estimatorPath, 'processedScores')

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
        qMin, qMax = 0.25, 0.75
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset.iloc[:, 0].isna().all()))
        if not hasattr(g.axes[ro, co], 'axHasQuantileAnnotation'):
            g.axes[ro, co].axHasQuantileAnnotation = True
            if not emptySubset:
                qMinVal, qMaxVal = dataSubset['parameter'].quantile([0.25, 0.75])
                messageStr = 'quantiles([{:.3g}, {:.3g}]) = ({:.3g}, {:.3g})'.format(qMin, qMax, qMinVal, qMaxVal)
                g.axes[ro, co].text(
                    0.9, 1, messageStr, ha='right', va='top',
                    fontsize=snsRCParams['font.size'], transform=g.axes[ro, co].transAxes)
            return
    #
    try:
        pdfPath = os.path.join(
            figureOutputFolder,
            '{}_{}_{}.pdf'.format(expDateTimePathStr, fullEstimatorName, 'partial_scores'))
        with PdfPages(pdfPath) as pdf:
            height, width = 2, 2
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
                plotScores.loc[:, 'freqBandName'] = plotScores['target'].map(targetInfo['freqBandName'])
                thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
                g = sns.catplot(
                    data=plotFUDE, kind='box',
                    y='score', x='target', hue='trialType',
                    hue_order=thisPalette.index.to_list(),
                    palette=thisPalette.to_dict(),
                    height=height, aspect=aspect,
                    sharey=True,
                    whis=np.inf,
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
                    style='freqBandName',
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
    except:
        traceback.print_exc()
    # estimatorsDF.iloc[0].regressor_.named_steps['regressor'].results_.summary()
    try:
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, fullEstimatorName, 'r2'))
        with PdfPages(pdfPath) as pdf:
            height, width = 1.5, 2
            aspect = width / height
            # for rhsMaskIdx, plotScores in scoresStack.groupby(['rhsMaskIdx'], sort=False):
            # rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            plotScores = scoresStack.copy()
            #
            for annotationName in ['historyLen', 'designFormula', 'ensembleTemplate', 'selfTemplate', 'designType']:
                plotScores.loc[:, annotationName] = plotScores['lhsMaskIdx'].map(lhsMasksInfo[annotationName])
            plotScores.loc[:, 'freqBandName'] = plotScores['target'].map(targetInfo['freqBandName'])
            trialTypesToPlot = ['test', 'train']
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            g = sns.catplot(
                data=plotScores.loc[plotScores['trialType'].isin(trialTypesToPlot), :],
                y='cc', hue='trialType',
                x='fullDesignAsLabel',
                col='freqBandName', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect, sharey=True,
                whis=np.inf,)
            # g.suptitle('CC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
            g.suptitle('CC')
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            trialTypesToPlot = ['train']
            g = sns.catplot(
                data=plotScores.loc[plotScores['trialType'].isin(trialTypesToPlot), :],
                hue='trialType',
                y='dAIC',
                x='fullDesignAsLabel',
                col='freqBandName', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                whis=np.inf,
                kind='box', height=height, aspect=aspect, sharey=True)
            # g.suptitle('AIC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
            g.suptitle('AIC')
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            trialTypesToPlot = ['test', 'train']
            g = sns.catplot(
                data=plotScores.loc[plotScores['trialType'].isin(trialTypesToPlot), :],
                y='cc', hue='trialType',
                x='fullDesignAsLabel', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                # kind='box',
                kind='violin', cut=0, inner='point',
                height=height, aspect=aspect, sharey=True)
            g.set_titles(template="{row_var}\n{row_name}")
            # g.suptitle('CC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.suptitle('CC')
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            trialTypesToPlot = ['train']
            g = sns.catplot(
                data=plotScores.loc[plotScores['trialType'].isin(trialTypesToPlot), :],
                y='dAIC', hue='trialType',
                x='fullDesignAsLabel', row='designType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                whis=np.inf,
                kind='box', height=height, aspect=aspect, sharey=True)
            g.set_titles(template="{row_var}\n{row_name}")
            # g.suptitle('AIC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.suptitle('AIC')
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    except:
        traceback.print_exc()

    '''
    pName = 'alpha'
    estimatorsDF.apply(lambda x: print(x.regressor_.named_steps['regressor'].get_params()[pName]))
    estimatorsDF.apply(
        lambda x: print(x.regressor_.named_steps['regressor']))
    '''
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, fullEstimatorName, 'hyperparameters_gridSearch'))
    with PdfPages(pdfPath) as pdf:
        height, width = 1.5, 2
        aspect = width / height
        hpNames = [cN for cN in gsScoresDF.columns if 'regressor__regressor__' in cN]
        gsScoresDF.reset_index(inplace=True)
        gsScoresDF.drop(columns=['level_3'], inplace=True)
        gsScoresDF.rename(columns={cN: cN.replace('regressor__regressor__', '') for cN in hpNames}, inplace=True)
        hpNames = [cN.replace('regressor__regressor__', '') for cN in hpNames]
        issueMask = gsScoresDF['score'].abs() > 1e3
        if issueMask.any():
            # gsScoresDF.loc[issueMask, 'lhsMaskIdx']
            gsScoresDF.loc[issueMask, 'score'] = np.nan
            print(gsScoresDF.loc[issueMask, :].drop_duplicates(subset=['lhsMaskIdx', 'rhsMaskIdx', 'fold']))
        #
        bestParamsDict = {}
        for gsName, gsGroup in gsScoresDF.groupby(['lhsMaskIdx', 'rhsMaskIdx', 'fold', 'target']):
            maxIdx = gsGroup.loc[gsGroup['foldType'] == 'test', :].set_index(hpNames)['score'].idxmax()
            bestParamsDict[gsName] = pd.Series({k: v for k, v in zip(hpNames, maxIdx)})
        bestParamsDF = pd.concat(bestParamsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'fold', 'target'], axis='columns').T.reset_index()
        #
        for annotationName in ['historyLen', 'designFormula', 'ensembleTemplate', 'selfTemplate', 'designType', 'fullFormulaShort']:
            gsScoresDF.loc[:, annotationName] = gsScoresDF['lhsMaskIdx'].map(lhsMasksInfo[annotationName])
            bestParamsDF.loc[:, annotationName] = bestParamsDF['lhsMaskIdx'].map(lhsMasksInfo[annotationName])
        #
        gsScoresDF.loc[:, 'freqBandName'] = gsScoresDF['target'].map(targetInfo['freqBandName'])
        bestParamsDF.loc[:, 'designType'] = bestParamsDF['lhsMaskIdx'].map(gsScoresDF.loc[:, ['lhsMaskIdx', 'designType']].drop_duplicates().set_index('lhsMaskIdx')['designType'])
        #
        # thisNIters
        plotCoefDF = coefDF.to_frame(name='coef').reset_index()
        plotCoefDF.loc[:, 'abs_coef'] = plotCoefDF['coef'].abs()
        plotCoefDF.loc[:, 'xDummy'] = 0
        g = sns.catplot(
            y='abs_coef',
            x='xDummy',
            hue='factor', hue_order=plotCoefDF['factor'].unique(),
            row='lhsMaskIdx', col='rhsMaskIdx',
            height=height, aspect=aspect,
            facet_kws=dict(sharey=False),
            whis=np.inf,
            data=plotCoefDF.query('coef != 0'), kind='box',
            legend=False
            )
        g.suptitle('magnitude of coefficients')
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        if convergenceDF is not None:
            plotConvDF = convergenceDF.reset_index()
            g = sns.relplot(
                y='cost', x='iter',
                row='lhsMaskIdx', col='rhsMaskIdx',
                hue='target',
                data=plotConvDF, kind='line', errorbar='se',
                legend=False,
                )
            g.suptitle('regression convergence (cost)')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            g = sns.relplot(
                y='scaledCost', x='iter',
                row='lhsMaskIdx', col='rhsMaskIdx',
                hue='target',
                data=plotConvDF, kind='line', errorbar='se',
                legend=False,
                )
            g.suptitle('regression convergence (minmax scaled cost)')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        if nItersDF is not None:
            plotNIters = nItersDF.to_frame(name='nIters').reset_index()
            g = sns.displot(
                data=plotNIters,
                row='lhsMaskIdx', col='rhsMaskIdx',
                x='nIters',
                height=height, aspect=aspect,
                facet_kws=dict(sharey=False),
                kind='hist'
                )
            g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
            g.suptitle('number of iterations')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        height, width = 1., 2.
        aspect = width / height
        for hpN in hpNames:
            #
            # plotScores = gsScoresDF.loc[gsScoresDF['rhsMaskIdx'] == rhsMaskIdx, :].groupby(hpNames + ['lhsMaskIdx', 'rhsMaskIdx', 'trialType', 'designType', 'historyLen', 'designFormula', 'ensembleTemplate', 'fold']).mean().reset_index()
            # plotScores = gsScoresDF.loc[gsScoresDF['rhsMaskIdx'] == rhsMaskIdx, :]
            #
            paramFromCV = estimatorsDF.apply(lambda x: x.regressor_.named_steps['regressor'].get_params()[hpN]).to_frame(name=hpN).reset_index()
            paramFromCV.loc[:, 'freqBandName'] = paramFromCV['target'].map(targetInfo['freqBandName'])
            for annName in ['designType', 'fullFormulaShort']:
                paramFromCV.loc[:, annName] = paramFromCV['lhsMaskIdx'].map(gsScoresDF.loc[:, ['lhsMaskIdx', annName]].drop_duplicates().set_index('lhsMaskIdx')[annName])
            #
            trialTypeMask = (
                trialTypePalette.index.isin(gsScoresDF['trialType']) &
                trialTypePalette.index.isin(['train'])
                )
            thisPalette = trialTypePalette.loc[trialTypeMask]
            if len(gsScoresDF[hpN].unique()) > 1:
                g = sns.relplot(
                    y='score', x=hpN,
                    row='fullFormulaShort', col='freqBandName',
                    hue='trialType',
                    data=gsScoresDF, kind='line', errorbar='se',
                    hue_order=thisPalette.index.to_list(),
                    palette=thisPalette.to_dict(),
                    height=height, aspect=aspect,
                    facet_kws=dict(sharex=False, sharey=False)
                    )
                if hpN in ['alpha']:
                    g.set(xscale='log')
                # xLimsRel = [a.get_xlim() for a in g.axes.flat]
                g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
                g.suptitle('Cross validated R^2')
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                pdf.savefig(bbox_inches='tight', pad_inches=0)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                '''
                plotScoresCV = scoresStack.loc[scoresStack['rhsMaskIdx'] == rhsMaskIdx, :]
                designTypeLookup = plotScores.loc[:, ['lhsMaskIdx', 'designType']].drop_duplicates().set_index('lhsMaskIdx')['designType']
                plotScoresCV.loc[:, 'designType'] = plotScoresCV['lhsMaskIdx'].map(designTypeLookup)
                paramLookup = estimatorsDF.apply(lambda x: x.regressor_.named_steps['regressor'].get_params()[hpN])
                paramLookup = paramLookup.xs(rhsMaskIdx, level='rhsMaskIdx', drop_level=False)
                paramLookup = paramLookup.reset_index(name=hpN).set_index(['lhsMaskIdx', 'fold'])[hpN]
                plotScoresCV.loc[:, hpN] = plotScoresCV.set_index(['lhsMaskIdx', 'fold']).index.map(paramLookup)
                '''
            else:
                g = sns.catplot(
                    data=gsScoresDF,
                    row='fullFormulaShort', col='freqBandName',
                    x=hpN, y='score',
                    hue='trialType',
                    hue_order=thisPalette.index.to_list(),
                    palette=thisPalette.to_dict(),
                    height=height, aspect=aspect,
                    facet_kws=dict(sharey=False),
                    kind='violin'
                    )
                # for aIdx, a in enumerate(g.axes.flat):
                #     a.set_xlim(xLimsRel[aIdx])
                g.suptitle('Cross validated R^2')
                g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                pdf.savefig(bbox_inches='tight', pad_inches=0)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            try:
                titleText = hpN
                if paramFromCV[hpN].dtype.kind in 'biufc':
                    g = sns.displot(
                        x=hpN,
                        row='fullFormulaShort', col='freqBandName',
                        data=paramFromCV, kind='hist'
                        )
                    for ax in g.axes.flat:
                        ax.xaxis.set_major_formatter(lambda x, pos: f'{x:.2e}')
                else:
                    g = sns.catplot(
                        x=hpN,
                        row='fullFormulaShort', col='freqBandName',
                        data=paramFromCV, kind='count'
                        )
                g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
                g.suptitle(titleText)
                '''plotProcFuns = [annotateWithQuantile]
                for (ro, co, hu), dataSubset in g.facet_data():
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)'''
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