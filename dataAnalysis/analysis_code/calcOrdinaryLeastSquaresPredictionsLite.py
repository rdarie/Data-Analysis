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

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_rb', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr_select_scaled', 'forceReprocess': True,
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
import pingouin as pg
from copy import deepcopy
import colorsys
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
for arg in sys.argv:
    print(arg)
idxSl = pd.IndexSlice
from datetime import datetime as dt

if __name__ == '__main__':
    try:
        print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
    except:
        pass
    for arg in sys.argv:
        print(arg)
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
    designMatrixPath = estimatorMeta['designMatrixPath']
    #
    lhsDF = pd.read_hdf(designMatrixPath, 'lhsDF')
    lhsMasks = pd.read_hdf(designMatrixPath, 'featureMasks')
    allTargetsDF = pd.read_hdf(designMatrixPath, 'allTargets').xs(arguments['estimatorName'], level='regressorName')
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    #
    lhsMasksInfo = pd.read_hdf(designMatrixPath, 'lhsMasksInfo')
    ####
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        slurmTaskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
    else:
        slurmTaskID = 0
    if os.getenv('SLURM_ARRAY_TASK_COUNT') is not None:
        slurmTaskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    else:
        slurmTaskCount = 1
    ########
    # if rerunning a subset of jobs from an original array
    slurmTaskCount = processSlurmTaskCount
    #### how many regressions per job?
    slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / slurmTaskCount))
    ####
    if os.getenv('SLURM_ARRAY_TASK_MIN') is not None:
        slurmTaskMin = int(os.getenv('SLURM_ARRAY_TASK_MIN'))
    else:
        slurmTaskMin = 0
    savingResults = True
    ################
    DEBUGGING = False
    if DEBUGGING:
        savingResults = False
        slurmTaskID = 9
        slurmTaskCount = processSlurmTaskCount
        slurmTaskMin = 0
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / slurmTaskCount))
        estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
    ################ collect estimators and scores
    estimatorsDict = {}
    scoresDict = {}
    for rowIdx, row in allTargetsDF.iterrows():
        lhsMaskIdx, rhsMaskIdx, targetName = row.name
        if (row['targetIdx'] // slurmGroupSize) != slurmTaskID:
            continue
        '''estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            estimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))'''
        thisEstimatorJBPath = os.path.join(
            estimatorPath.replace('.h5', ''),
            'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
            ))
        thisEstimatorJBDict = jb.load(thisEstimatorJBPath)
        thisEstimatorJB = pd.Series(thisEstimatorJBDict)
        thisEstimatorJB.index.name = 'fold'
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = thisEstimatorJB
        scoresDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            estimatorPath,
            'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    scoresDF = pd.concat(scoresDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    # prep rhs dataframes
    histDesignInfoDict = {}
    histSourceTermDict = {}
    for rhsMaskIdx in range(rhsMasks.shape[0]):
        prf.print_memory_usage('Prepping RHS dataframes (rhsRow: {})'.format(rhsMaskIdx))
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        rhGroup = pd.read_hdf(designMatrixPath, 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
        for ensTemplate in lOfHistTemplates:
            if ensTemplate != 'NULL':
                histSourceTermDict.update({ensTemplate.format(cN): cN for cN in rhGroup.columns})
                ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
                ensFormula += ' - 1'
                print('Generating endog design info for {}'.format(ensFormula))
                ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
                exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
                ensPt.fit(exampleRhGroup)
                ensDesignMatrix = ensPt.transform(exampleRhGroup)
                ensDesignInfo = ensDesignMatrix.design_info
                print(ensDesignInfo.term_names)
                print('\n')
                histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
    #
    designInfoDict = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        #
        if designFormula != 'NULL':
            if designFormula not in designInfoDict:
                print('Generating exog design info for {}'.format(designFormula))
                formulaIdx = lOfDesignFormulas.index(designFormula)
                lhGroup = lhsDF.loc[:, lhsMask]
                pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
                exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
                designMatrix = pt.fit_transform(exampleLhGroup)
                designInfo = designMatrix.design_info
                designInfoDict[designFormula] = designInfo
    #
    designInfoDF = pd.Series(designInfoDict).to_frame(name='designInfo')
    designInfoDF.index.name = 'design'
    histDesignInfoDF = pd.DataFrame(
        [value for key, value in histDesignInfoDict.items()],
        columns=['designInfo'])
    histDesignInfoDF.index = pd.MultiIndex.from_tuples(
        [key for key, value in histDesignInfoDict.items()],
        names=['rhsMaskIdx', 'ensTemplate'])
    ################################################################################################
    predDF = None
    residualsAutoCorrDF = None
    resNormTestDict = {}
    processedTargetIndices = []
    print('Calculating predicted waveforms per model')
    indexNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'foldType']
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        thisModelHistoryLen = lhsMasksInfo.loc[lhsMaskIdx, 'historyLen']
        #
        designFormula = lhsMaskParams['designFormula']
        lhGroup = lhsDF.loc[:, lhsMask]
        if designFormula != 'NULL':
            designInfo = designInfoDict[designFormula]
            formulaIdx = lOfDesignFormulas.index(designFormula)
            ##
            theseColumns = designInfo.column_names
            #
            parentFormula = masterExogLookup[designFormula]
            parentFormulaIdx = masterExogFormulas.index(parentFormula)
            designDF = pd.read_hdf(designMatrixPath, 'designs/exogParents/formula_{}'.format(parentFormulaIdx))
            try:
                designDF = designDF.loc[:, theseColumns]
            except:
                traceback.print_exc()
            thisFeatureInfo = pd.read_hdf(designMatrixPath, 'designs/exogParents/term_lookup_{}'.format(parentFormulaIdx))
            ##
            try:
                designDF.columns = pd.MultiIndex.from_frame(thisFeatureInfo.loc[thisFeatureInfo['factor'].isin(theseColumns), :])
            except:
                traceback.print_exc()
            designDF = designDF.loc[:, designDF.columns.get_level_values('factor').isin(theseColumns)]
            exogList = [designDF]
            designTermNames = designInfo.term_names
        else:
            exogList = []
            designInfo = None
            designDF = None
            designTermNames = []
        #
        # add ensemble to designDF?
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        for rhsMaskIdx in range(rhsMasks.shape[0]):
            rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
            rhGroup = pd.read_hdf(
                designMatrixPath,
                'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            if ensTemplate != 'NULL':
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
            if selfTemplate != 'NULL':
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
            ####
            for targetName in rhGroup.columns:
                if (lhsMaskIdx, rhsMaskIdx, targetName) not in allTargetsDF.index:
                    continue
                targetIdx = allTargetsDF.loc[(lhsMaskIdx, rhsMaskIdx, targetName), 'targetIdx']
                if (targetIdx // slurmGroupSize) != slurmTaskID:
                    # print("targetIdx ({}) // slurmGroupSize = {}".format(targetIdx, targetIdx // slurmGroupSize))
                    # print('slurmTaskID = {}'.format(slurmTaskID))
                    # print('Skipping...')
                    continue
                else:
                    processedTargetIndices.append(targetIdx)
                # add targetDF to designDF?
                if ensTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(ensTemplate)
                    thisEnsDesign = pd.read_hdf(
                        designMatrixPath,
                        'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    thisEnsFeatureInfo = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/term_lookup_{}'.format(rhsMaskIdx, templateIdx))
                    thisEnsDesign.columns = pd.MultiIndex.from_frame(thisEnsFeatureInfo)
                    ensHistList = [
                        thisEnsDesign.iloc[:, sl]
                        for key, sl in ensDesignInfo.term_name_slices.items()
                        if key != ensTemplate.format(targetName)]
                    del thisEnsDesign
                    ensTermNames = [
                        tN
                        for tN in ensDesignInfo.term_names
                        if tN != ensTemplate.format(targetName)]
                else:
                    ensHistList = []
                    ensTermNames = []
                #
                if selfTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(selfTemplate)
                    thisSelfDesign = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    thisSelfFeatureInfo = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/term_lookup_{}'.format(rhsMaskIdx, templateIdx))
                    thisSelfDesign.columns = pd.MultiIndex.from_frame(thisSelfFeatureInfo)
                    selfHistList = [
                        thisSelfDesign.iloc[:, sl].copy()
                        for key, sl in selfDesignInfo.term_name_slices.items()
                        if key == selfTemplate.format(targetName)]
                    del thisSelfDesign
                    selfTermNames = [
                        tN
                        for tN in selfDesignInfo.term_names
                        if tN == selfTemplate.format(targetName)]
                    # print('selfHistList:\n{}\nselfTermNames:\n{}'.format(selfHistList, selfTermNames))
                else:
                    selfHistList = []
                    selfTermNames = []
                #
                fullDesignList = exogList + ensHistList + selfHistList
                fullDesignDF = pd.concat(fullDesignList, axis='columns')
                del fullDesignList
                #
                sortOrder = {cN: cIdx for cIdx, cN in enumerate(rhGroup.columns)}
                sortKey = lambda x: x.map(sortOrder)
                fullDesignDF.sort_index(
                    axis='columns', level='source',
                    key=sortKey, kind='mergesort',
                    sort_remaining=False, inplace=True)
                fullDesignColumnInfo = fullDesignDF.columns.to_frame().reset_index(drop=True)
                fullDesignDF.columns = fullDesignDF.columns.get_level_values('factor')
                #
                for foldIdx in range(cvIterator.n_splits + 1):
                    targetDF = rhGroup.loc[:, [targetName]]
                    estimatorIdx = (lhsMaskIdx, rhsMaskIdx, targetName, foldIdx)
                    if int(arguments['verbose']) > 3:
                        print('estimator: {}'.format(estimatorIdx))
                        print('in dataframe: {}'.format(estimatorIdx in estimatorsDF.index))
                    if not estimatorIdx in estimatorsDF.index:
                        continue
                    if foldIdx == cvIterator.n_splits:
                        # work and validation folds
                        trainIdx, testIdx = cvIterator.workIterator.split(rhGroup)[0]
                        trainStr, testStr = 'work', 'validation'
                        foldType = 'validation'
                    else:
                        trainIdx, testIdx = cvIterator.raw_folds[foldIdx]
                        trainStr, testStr = 'train', 'test'
                        foldType = 'train'
                    estimator = estimatorsDF.loc[estimatorIdx]
                    regressor = estimator.regressor_.named_steps['regressor']
                    estPreprocessorRhs = estimator.transformer_
                    ####
                    '''predictionPerSource = pd.concat([
                        estPreprocessorRhs.transform(targetDF.iloc[trainIdx, 0]),
                        estPreprocessorRhs.transform(targetDF.iloc[testIdx, 0])
                        ]).to_frame(name='ground_truth')'''
                    predictionPerSource = pd.concat({
                        trainStr: estPreprocessorRhs.transform(targetDF.iloc[trainIdx, 0]),
                        testStr: estPreprocessorRhs.transform(targetDF.iloc[testIdx, 0])
                        }, names=['trialType']).to_frame(name='ground_truth')
                    predictionPerSource.loc[:, 'prediction'] = np.concatenate([
                        estimator.predict(fullDesignDF.iloc[trainIdx, :]),
                        estimator.predict(fullDesignDF.iloc[testIdx, :])
                        ])
                    predictionPerSource.loc[:, 'residuals'] = predictionPerSource['ground_truth'] - predictionPerSource['prediction']
                    predIndexDF = predictionPerSource.index.to_frame().reset_index(drop=True)
                    indexValues = [lhsMaskIdx, designFormula, rhsMaskIdx, targetName, foldIdx, foldType]
                    for indexName, indexValue in zip(indexNames[::-1], indexValues[::-1]):
                        predIndexDF.insert(0, indexName, indexValue)
                    predictionPerSource.index = pd.MultiIndex.from_frame(predIndexDF)
                    '''if True:
                        plotPred = predictionPerSource.xs(0, level='trialUID').reset_index()
                        plt.plot(plotPred['bin'], plotPred['ground_truth'])
                        plt.plot(plotPred['bin'], plotPred['prediction'])
                        plt.show()'''
                    if predDF is None:
                        predDF = predictionPerSource
                    else:
                        predDF = predDF.append(predictionPerSource)
                    prf.print_memory_usage('Calculated predictions for {}'.format(indexValues))
                    print('predDF.shape = {}'.format(predDF.shape))
                    #### residual autocorrelation
                    nLags = int(thisModelHistoryLen / binInterval)
                    nLagsSec = np.arange(0, nLags) * binInterval
                    residualsAutoCorrDict = {}
                    for lagIdx, lagSec in enumerate(nLagsSec):
                        residualsAutoCorrDict[lagSec] = predictionPerSource['residuals'].groupby('trialUID').apply(lambda x: x.autocorr(lag=lagIdx))
                    theseResAutoCorrDF = pd.concat(residualsAutoCorrDict, names=['lag', 'trialUID'])
                    del residualsAutoCorrDict
                    resAutoCorrIndexDF = theseResAutoCorrDF.index.to_frame().reset_index(drop=True)
                    for indexName, indexValue in zip(indexNames[::-1], indexValues[::-1]):
                        resAutoCorrIndexDF.insert(0, indexName, indexValue)
                    theseResAutoCorrDF.index = pd.MultiIndex.from_frame(resAutoCorrIndexDF)
                    if residualsAutoCorrDF is None:
                        residualsAutoCorrDF = theseResAutoCorrDF
                    else:
                        residualsAutoCorrDF = residualsAutoCorrDF.append(theseResAutoCorrDF)
                    ##### residuals goodness of fit tests
                    minBin = predIndexDF['bin'].min()
                    shiftedRes = predictionPerSource['residuals'].shift(1).drop(minBin, level='bin')
                    residT = predictionPerSource['residuals'].drop(minBin, level='bin')
                    residTDiff = residT - shiftedRes
                    kDW = (residTDiff ** 2).sum() / (predictionPerSource['residuals'] ** 2).sum()
                    #
                    resNormAlpha = 0.05
                    resNormality = pg.normality(
                        predictionPerSource['residuals'], method='normaltest', alpha=resNormAlpha).iloc[0, :]
                    resNormality.loc['kDW'] = kDW
                    resNormality.loc['alpha'] = resNormAlpha
                    resNormTestDict[tuple(indexValues)] = resNormality
                    #
    resNormTestDF = pd.concat(resNormTestDict, names=indexNames, axis='columns').T
    predDF.columns.name = 'term'
    prf.print_memory_usage('Saving prediction DF')
    if savingResults:
        predDF.to_hdf(estimatorPath, 'predictions')
        resNormTestDF.to_hdf(estimatorPath, 'residualNormality')
        residualsAutoCorrDF.to_hdf(estimatorPath, 'residualAutocorrelation')
    print('Loaded and saved predictions and coefficients')
    #
    scoresStack = pd.concat({
        'test': scoresDF['test_score'],
        'train': scoresDF['train_score']},
        names=['foldType']
        ).to_frame(name='score').reset_index()
    #
    lastFoldMask = (scoresStack['fold'] == cvIterator.n_splits)
    trainMask = (scoresStack['foldType'] == 'train')
    testMask = (scoresStack['foldType'] == 'test')
    #
    scoresStack.loc[:, 'trialType'] = ''
    scoresStack.loc[(trainMask & lastFoldMask), 'trialType'] = 'work'
    scoresStack.loc[(trainMask & (~lastFoldMask)), 'trialType'] = 'train'
    scoresStack.loc[(testMask & lastFoldMask), 'trialType'] = 'validation'
    scoresStack.loc[(testMask & (~lastFoldMask)), 'trialType'] = 'test'
    scoresStack.loc[:, 'dummyX'] = 0
    scoresStack.loc[:, 'design'] = scoresStack['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'designFormula'])
    scoresStack.loc[:, 'designAsLabel'] = scoresStack['design'].apply(lambda x: x.replace(' + ', ' +\n'))
    scoresStack.loc[:, 'fullDesign'] = scoresStack['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
    scoresStack.loc[:, 'fullDesignAsLabel'] = scoresStack['fullDesign'].apply(lambda x: x.replace(' + ', ' +\n'))
    if savingResults:
        scoresStack.to_hdf(estimatorPath, 'processedCVScores')
    llDict1 = {}
    aicDict1 = {}
    ccDict1 = {}
    for scoreName, targetScores in scoresStack.groupby(['lhsMaskIdx', 'target', 'fold']):
        print('Calculating scores ({})'.format(scoreName))
        lhsMaskIdx, targetName, fold = scoreName
        designFormula = lhsMasksInfo.loc[lhsMaskIdx, 'designFormula']
        estimator = estimatorsDF.loc[idxSl[lhsMaskIdx, :, targetName, fold]].iloc[0]
        regressor = estimator.regressor_.steps[-1][1]
        K = regressor.results_.df_model
        thesePred = predDF.xs(targetName, level='target').xs(lhsMaskIdx, level='lhsMaskIdx').xs(fold, level='fold')
        llDict2 = {}
        aicDict2 = {}
        ccDict2 = {}
        if hasattr(regressor, 'results_'):
            probaFamily = regressor.results_.family
        else:
            from statsmodels.genmod.families.family import Gaussian
            probaFamily = Gaussian()
        for name, predGroup in thesePred.groupby(['electrode', 'trialType']):
            llDict3 = dict()
            llDict3['llSat'] = probaFamily.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
            nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
            llDict3['llNull'] = probaFamily.loglike(nullModel, predGroup['ground_truth'].to_numpy())
            llDict3['llFull'] = probaFamily.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
            llDict2[name] = pd.Series(llDict3)
            aicDict2[name] = 2 * K - 2 * llDict3['llFull']
        for trialType, predGroup in thesePred.groupby('trialType'):
            llDict3 = dict()
            llDict3['llSat'] = probaFamily.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
            nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
            llDict3['llNull'] = probaFamily.loglike(nullModel, predGroup['ground_truth'].to_numpy())
            llDict3['llFull'] = probaFamily.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
            llDict2[('all', trialType)] = pd.Series(llDict3)
            aicDict2[('all', trialType)] = 2 * K - 2 * llDict3['llFull']
            ccDict2[('all', trialType)] = np.corrcoef(
                predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())[0][1]
        llDict1[(lhsMaskIdx, designFormula, targetName, fold)] = pd.concat(llDict2, names=['electrode', 'trialType', 'llType'])
        aicSrs = pd.Series(aicDict2)
        aicSrs.index.names = ['electrode', 'trialType']
        aicDict1[(lhsMaskIdx, designFormula, targetName, fold)] = aicSrs
        ccSrs = pd.Series(ccDict2)
        ccSrs.index.names = ['electrode', 'trialType']
        ccDict1[(lhsMaskIdx, designFormula, targetName, fold)] = ccSrs
    llDF = pd.concat(
        llDict1, names=['lhsMaskIdx', 'design', 'target', 'fold', 'electrode', 'trialType', 'llType']).to_frame(name='ll')
    llDF.loc[:, 'fullFormulaDescr'] = llDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
    llDF.set_index('fullFormulaDescr', append=True, inplace=True)
    if savingResults:
        llDF.to_hdf(estimatorPath, 'processedLogLike')
    aicDF = pd.concat(
        aicDict1, names=['lhsMaskIdx', 'design', 'target', 'fold', 'electrode', 'trialType']).to_frame(name='aic')
    aicDF.loc[:, 'fullFormulaDescr'] = aicDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
    aicDF.set_index('fullFormulaDescr', append=True, inplace=True)
    if savingResults:
        aicDF.to_hdf(estimatorPath, 'processedAIC')
    ccDF = pd.concat(
        ccDict1, names=['lhsMaskIdx', 'design', 'target', 'fold', 'electrode', 'trialType']).to_frame(name='cc')
    ccDF.loc[:, 'fullFormulaDescr'] = ccDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
    ccDF.set_index('fullFormulaDescr', append=True, inplace=True)
    if savingResults:
        ccDF.to_hdf(estimatorPath, 'processedCC')
    #
    R2Per = llDF['ll'].groupby(['lhsMaskIdx', 'design', 'target', 'electrode', 'fold', 'trialType']).apply(
        tdr.getR2).to_frame(name='score')
    R2Per.loc[:, 'fullFormulaDescr'] = R2Per.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
    R2Per.set_index('fullFormulaDescr', append=True, inplace=True)
    if savingResults:
        R2Per.to_hdf(estimatorPath, 'processedR2')
    print('Loaded and saved scores and partial scores')
    # sanity check that all the scoring methods are equivalent
    # if True:
    #     trainR2 = scoresStack.loc[scoresStack['trialType'] == 'train', 'score'].to_numpy()
    #     trainR2FromLL = R2Per.xs('train', level='trialType').xs('all', level='electrode')['score'].to_numpy()
    #     trainCC = ccDF.xs('train', level='trialType')['cc'].to_numpy()
    #     fig, ax = plt.subplots()
    #     ax.plot(np.sqrt(trainR2) - trainCC)
    #     plt.show()

    print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
