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
from tqdm import tqdm
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

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

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

if __name__ == '__main__':
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
    #
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
    # histOptsForExportDF = pd.DataFrame(histOptsForExportDict)
    # histOptsHtmlPath = os.path.join(
    #     figureOutputFolder, '{}_{}.html'.format(fullEstimatorName, 'histOpts'))
    # histOptsForExportDF.to_html(histOptsHtmlPath)
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
    allTargetsPLS = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargetsPLS')
    allTargetsPLS.set_index(['lhsMaskIdx', 'rhsMaskIdx'], inplace=True)
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    #
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
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
    slurmTaskCount = processSlurmTaskCountPLS
    #### how many regressions per job?
    slurmGroupSize = int(np.ceil(allTargetsPLS.shape[0] / slurmTaskCount))
    ####
    if os.getenv('SLURM_ARRAY_TASK_MIN') is not None:
        slurmTaskMin = int(os.getenv('SLURM_ARRAY_TASK_MIN'))
    else:
        slurmTaskMin = 0
    savingResults = True
    ################
    # savingResults = False
    # slurmTaskID = 1
    # slurmTaskCount = processSlurmTaskCountPLS
    # slurmTaskMin = 0
    # slurmGroupSize = int(np.ceil(allTargetsPLS.shape[0] / slurmTaskCount))
    # estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
    ################ collect estimators and scores
    estimatorsDict = {}
    scoresDict = {}
    for rowIdx, row in allTargetsPLS.iterrows():
        lhsMaskIdx, rhsMaskIdx = row.name
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        freqBandName = rhsMaskParams['freqBandName']
        if (row['targetIdx'] // slurmGroupSize) != slurmTaskID:
            continue
        '''estimatorsDict[(lhsMaskIdx, rhsMaskIdx, freqBandName)] = pd.read_hdf(
            estimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, freqBandName
                ))'''
        thisEstimatorJBPath = os.path.join(
            estimatorPath.replace('.h5', ''),
            'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                lhsMaskIdx, rhsMaskIdx, freqBandName
            ))
        thisEstimatorJBDict = jb.load(thisEstimatorJBPath)
        thisEstimatorJB = pd.Series(thisEstimatorJBDict)
        thisEstimatorJB.index.name = 'fold'
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, freqBandName)] = thisEstimatorJB
        #
        scoresDict[(lhsMaskIdx, rhsMaskIdx, freqBandName)] = pd.read_hdf(
            estimatorPath,
            'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, freqBandName
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
        rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
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
            designDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'designs/formula_{}'.format(formulaIdx))
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
        #
        for rhsMaskIdx in range(rhsMasks.shape[0]):
            rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
            freqBandName = rhsMaskParams['freqBandName']
            targetIdx = allTargetsPLS.loc[(lhsMaskIdx, rhsMaskIdx), 'targetIdx']
            if (targetIdx // slurmGroupSize) != slurmTaskID:
                continue
            rhGroup = pd.read_hdf(
                estimatorMeta['designMatrixPath'],
                'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            if ensTemplate != 'NULL':
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
                templateIdx = lOfHistTemplates.index(ensTemplate)
                thisEnsDesign = pd.read_hdf(
                    estimatorMeta['designMatrixPath'],
                    'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                ensHistList = [thisEnsDesign]
                ensTermNames = [tN for tN in ensDesignInfo.term_names]
            else:
                ensHistList = []
                ensTermNames = []
            #
            fullDesignList = exogList + ensHistList
            fullDesignDF = pd.concat(fullDesignList, axis='columns')
            del fullDesignList
            ####
            for foldIdx in range(cvIterator.n_splits + 1):
                estimatorIdx = (lhsMaskIdx, rhsMaskIdx, freqBandName, foldIdx)
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
                indicesThisFold = np.concatenate([trainIdx, testIdx])
                estimator = estimatorsDF.loc[estimatorIdx]
                regressor = estimator.regressor_.steps[-1][1]
                estPreprocessorLhs = Pipeline(estimator.regressor_.steps[:-1])
                estPreprocessorRhs = estimator.transformer_
                termNames = designTermNames + ensTermNames
                # pdb.set_trace()
                all_coefs = estimator.regressor_.named_steps['regressor'].coef_
                _x_mean = estimator.regressor_.named_steps['regressor']._x_mean
                _y_mean = estimator.regressor_.named_steps['regressor']._y_mean
                _x_std = estimator.regressor_.named_steps['regressor']._x_std
                _y_std = estimator.regressor_.named_steps['regressor']._y_std
                tIterator = enumerate(tqdm(rhGroup.columns, mininterval=30., maxinterval=120.))
                for subTargetIdx, subTarget in tIterator:
                    coefs = pd.Series(
                        all_coefs[:, subTargetIdx], index=fullDesignDF.columns)
                    K = (coefs.abs() > 0).sum() # dof
                    predictionPerComponent = pd.concat({
                        trainStr: estPreprocessorLhs.transform((fullDesignDF.iloc[trainIdx, :] - _x_mean) / _x_std) * coefs,
                        testStr: estPreprocessorLhs.transform((fullDesignDF.iloc[testIdx, :] - _x_mean) / _x_std) * coefs
                        }, names=['trialType'])
                    print('predictionPerComponent.index.names =\n{}'.format(predictionPerComponent.index.names))
                    predictionSrs = predictionPerComponent.sum(axis='columns')
                    # sanity check
                    #############################
                    doSanityCheck = False
                    if doSanityCheck:
                        predictionsNormalWay = (
                            (estimator.predict(fullDesignDF.iloc[indicesThisFold, :]) * _y_std - _y_mean)[:, subTargetIdx]
                            )
                        mismatch = predictionSrs - predictionsNormalWay.reshape(-1)
                        if int(arguments['verbose']) > 3:
                            print('max mismatch is {}'.format(mismatch.abs().max()))
                        try:
                            assert (mismatch.abs().max() < 1e-3)
                        except:
                            '''
                            fig, ax = plt.subplots()
                            ax.plot(predictionSrs.to_numpy(), label='from summation')
                            ax.plot(predictionsNormalWay, '--', label='from prediction')
                            # ax.plot(predictionSrs.to_numpy() / predictionsNormalWay , label='ratio')
                            # ax.plot(predictionPerSource.loc[:, 'ground_truth'].to_numpy(), label='gt')
                            ax.legend()
                            '''
                            print('Attention! max mismatch is {}'.format(mismatch.abs().max()))
                    predictionPerSource = pd.DataFrame(
                        np.nan, index=predictionPerComponent.index,
                        columns=termNames)
                    for termName in designTermNames:
                        termSlice = designInfo.term_name_slices[termName]
                        factorNames = designInfo.column_names[termSlice]
                        predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                    if ensTemplate != 'NULL':
                        for termName in ensTermNames:
                            factorNames = ensDesignInfo.column_names[ensDesignInfo.term_name_slices[termName]]
                            predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                    predictionPerSource.loc[:, 'prediction'] = predictionSrs
                    predictionPerSource.loc[:, 'ground_truth'] = np.concatenate([
                        estPreprocessorRhs.transform(rhGroup.iloc[trainIdx, :] * _y_std - _y_mean).iloc[:, subTargetIdx],
                        estPreprocessorRhs.transform(rhGroup.iloc[testIdx, :] * _y_std - _y_mean).iloc[:, subTargetIdx]
                        ])
                    predictionPerSource.loc[:, 'residuals'] = predictionPerSource['ground_truth'] - predictionSrs
                    predIndexDF = predictionPerSource.index.to_frame().reset_index(drop=True)
                    indexValues = [lhsMaskIdx, designFormula, rhsMaskIdx, subTarget, foldIdx, foldType]
                    for indexName, indexValue in zip(indexNames[::-1], indexValues[::-1]):
                        predIndexDF.insert(0, indexName, indexValue)
                    predictionPerSource.index = pd.MultiIndex.from_frame(predIndexDF)
                    if predDF is None:
                        predDF = predictionPerSource
                    else:
                        predDF = predDF.append(predictionPerSource)
                    prf.print_memory_usage('\nCalculated predictions for\n[lhsMaskIdx, designFormula, rhsMaskIdx, subTarget, foldIdx, foldType]\n{}'.format(indexValues))
    resNormTestDF = pd.concat(resNormTestDict, names=indexNames, axis='columns').T
    predDF.columns.name = 'term'
    prf.print_memory_usage('Saving prediction DF')
    if savingResults:
        predDF.to_hdf(estimatorPath, 'predictions_full')
    print('Loaded and saved predictions and coefficients')