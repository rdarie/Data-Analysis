"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             load from raw, or regular? [default: False]
    --showFigures                          load from raw, or regular? [default: False]
    --debugging                            load from raw, or regular? [default: False]
    --averageByTrial                       load from raw, or regular? [default: False]
    --calculateFullModels                  load from raw, or regular? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
    --selectMethod=selectMethod            filename for resulting estimator (cross-validated n_comps)
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
#
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dask.distributed import Client, LocalCluster
import os, traceback
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.custom_transformers.tdr import reconstructionR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))

import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer, r2_score
import joblib as jb
import dill as pickle
pickle.settings['recurse'] = True
import gc, sys

from copy import deepcopy
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
###
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)


if __name__ == '__main__':
    ##
    '''
    
    consoleDebugging = True
    if consoleDebugging:
        arguments = {
            'processAll': True, 'debugging': True, 'estimatorName': 'select', 'exp': 'exp202101271100',
            'analysisName': 'hiRes', 'lazy': False, 'blockIdx': '2', 'averageByTrial': False, 'verbose': '2',
            'selectionName': 'lfp_CAR', 'showFigures': False, 'datasetName': 'Block_XL_df_ra', 'plotting': True,
            'window': 'long', 'alignFolderName': 'motion'}
        os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
    ##
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    #
    idxSl = pd.IndexSlice
    arguments['verbose'] = int(arguments['verbose'])
    #
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'], 'dimensionality')
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    #
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
    estimatorName = arguments['estimatorName']
    fullEstimatorName = '{}_{}_{}'.format(
        estimatorName, datasetName, selectionName)
    #
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    estimatorMetaDataPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_meta.pickle'
        )
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
        iteratorsBySegment = loadingMeta['iteratorsBySegment']
        # cv_kwargs = loadingMeta['cv_kwargs']
    for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
        loadingMeta['arguments'].pop(argName, None)
    arguments.update(loadingMeta['arguments'])
    cvIterator = iteratorsBySegment[0]
    ###
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    # only use zero lag targets
    lagMask = dataDF.columns.get_level_values('lag') == 0
    dataDF = dataDF.loc[:, lagMask]
    featureMasks = featureMasks.loc[:, lagMask]
    #
    estimatorClass = ColumnTransformer
    # listOfColumns = [
    #     ('utah18', 0, 8.0, 9.0, 'NA', 'NA'),
    #     ('utah29', 0, 6.0, 0.0, 'NA', 'NA'),
    #     ('utah32', 0, 6.0, 3.0, 'NA', 'NA'),
    #     ('utah35', 0, 6.0, 6.0, 'NA', 'NA'),
    #     ('utah38', 0, 6.0, 9.0, 'NA', 'NA'),
    #     ( 'utah4', 0, 9.0, 4.0, 'NA', 'NA'),
    #     ('utah59', 0, 3.0, 0.0, 'NA', 'NA'),
    #     ('utah62', 0, 3.0, 3.0, 'NA', 'NA'),
    #     ('utah65', 0, 3.0, 6.0, 'NA', 'NA'),
    #     ('utah68', 0, 3.0, 9.0, 'NA', 'NA'),
    #     ( 'utah7', 0, 9.0, 7.0, 'NA', 'NA'),
    #     ('utah79', 0, 1.0, 0.0, 'NA', 'NA'),
    #     ( 'utah9', 0, 8.0, 0.0, 'NA', 'NA'),
    #     ('utah90', 0, 0.0, 2.0, 'NA', 'NA'),
    #     ('utah93', 0, 0.0, 5.0, 'NA', 'NA'),
    #     ('utah96', 0, 0.0, 8.0, 'NA', 'NA')
    #     ]
    if arguments['selectMethod'] == 'fromRegression':
        referenceRegressionName = 'ols_select_baseline_{}'.format(datasetName)
        referenceRegressionPath = os.path.join(
            estimatorsSubFolder,
            referenceRegressionName + '.h5'
            )
        # pdb.set_trace()
        referenceScores = pd.read_hdf(referenceRegressionPath, 'processedScores')
        referenceTrainMask = (referenceScores['trialType'] == 'train')
        trainCCDF = referenceScores.loc[referenceTrainMask, ['target', 'cc']].groupby('target').mean()
        listOfTargetNames = trainCCDF.sort_values('cc', ascending=False, kind='mergesort').index.to_list()
        listOfColumns = [cN for cN in dataDF.columns if cN[0] in listOfTargetNames[:16]]
    if arguments['selectMethod'] == 'decimateSpace':
        featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
        keepX = np.unique(featureInfo['xCoords'])[::3]
        keepY = np.unique(featureInfo['yCoords'])[::3]
        xyMask = (
            featureInfo['xCoords'].isin(keepX) &
            featureInfo['yCoords'].isin(keepY)
            )
        listOfColumns = dataDF.columns[xyMask.to_numpy()].to_list()
    elif arguments['selectMethod'] == 'keepAll':
        listOfColumns = dataDF.columns.to_list()
    elif arguments['selectMethod'] == 'mostModulated':
        blockBaseName, _ = hf.processBasicPaths(arguments)
        raucResultsPath = os.path.join(
            dataFramesFolder,
            '{}_{}_{}_rauc.h5'.format(blockBaseName, selectionName, loadingMeta['arguments']['window'])
            )
        relativeStatsDF = pd.read_hdf(raucResultsPath, 'relativeStatsDF')
        relativeStatsDF.loc[:, 'T_abs'] = relativeStatsDF['T'].abs()
        statsRankingDF = relativeStatsDF.groupby(dataDF.columns.names).mean().sort_values('T', ascending=False, kind='mergesort')
        # add to list based on dataDF to maintain ordering
        print('Choosing top 16 features from statsRankingDF')
        listOfColumns = [cN for cN in dataDF.columns.to_list() if cN in statsRankingDF.index[:16]]
    #
    excludeFreqBands = ['alpha', 'spb']
    listOfColumns = [cN for cN in listOfColumns if cN[4] not in excludeFreqBands]
    selectedColumnsStr = '\n'.join(['{}'.format(cN) for cN in listOfColumns])
    print('Selecting {} columns:\n{}\n'.format(len(listOfColumns), selectedColumnsStr))
    print(', '.join(["'{}#0'".format(cN[0]) for cN in listOfColumns]))
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    workIdx = cvIterator.work
    workingDataDF = dataDF.iloc[workIdx, :]
    prf.print_memory_usage('just loaded data, fitting')
    #
    cvScoresDict = {}  # cross validated best estimator
    featuresList = []
    featureColumnFields = dataDF.columns.names
    #
    for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
        maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
        dataGroup = dataDF.loc[:, featureMask]
        nFeatures = dataGroup.columns.shape[0]
        trfName = '{}_{}'.format(estimatorName, maskParams['freqBandName'])
        ###
        columnsMask = dataGroup.columns.isin(listOfColumns)
        estimatorKWArgs = {
            'transformers':
                [('select', 'passthrough', columnsMask)]}
        cvScores = {
            foldIdx: Pipeline([
                ('averager', tdr.DataFrameBinTrimmer(burnInPeriod=burnInPeriod)),
                ('dim_red', estimatorClass(**estimatorKWArgs))])
            for foldIdx in range(cvIterator.n_splits + 1)}
        # pdb.set_trace()
        for foldIdx, selector in cvScores.items():
            selector.fit(dataGroup)
        cvScoresDF = pd.Series(cvScores).to_frame(name='estimator')
        cvScoresDF.loc[:, ['test_score', 'train_score']] = 1.
        cvScoresDF.index.name = 'fold'
        cvScoresDict[maskParams['freqBandName']] = cvScoresDF
        estimatorInstance = Pipeline([
            ('averager', tdr.DataFrameBinTrimmer(burnInPeriod=burnInPeriod)),
            ('dim_red', estimatorClass(**estimatorKWArgs))])
        featureColumns = dataGroup.columns[columnsMask].to_frame().reset_index(drop=True)
        featureColumns.loc[:, 'freqBandName'] = maskParams['freqBandName']
        preEst = Pipeline(estimatorInstance.steps[:-1])
        xInterDF = preEst.fit_transform(dataGroup)
        if isinstance(xInterDF, pd.DataFrame):
            featuresIndex = xInterDF.index
        else:
            featuresIndex = pd.Index(range(xInterDF.shape[0]))
        # else:
        #     featuresIndex = dataGroup.index
        featuresList.append(pd.DataFrame(
            estimatorInstance.fit_transform(dataGroup),
            index=featuresIndex, columns=pd.MultiIndex.from_frame(featureColumns)))
    featuresDF = pd.concat(featuresList, axis='columns')
    #
    scoresDF = pd.concat(cvScoresDict, names=['freqBandName'])
    lastFoldIdx = scoresDF.index.get_level_values('fold').max()
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        print('Deleting contents of {}'.format(estimatorPath))
        os.remove(estimatorPath)
    #
    if arguments['averageByTrial']:
        for rowIdx, row in scoresDF.iterrows():
            newEstimatorInstance = Pipeline([
                ('averager', tdr.DataFrameAverager(
                    stimConditionNames=stimulusConditionNames + ['bin'],
                    addIndexFor=stimulusConditionNames, burnInPeriod=burnInPeriod)),
                ('dim_red', row['estimator'].named_steps['dim_red'])])
            scoresDF.loc[rowIdx, 'estimator'] = newEstimatorInstance
    print('\n\nSaving {}\n\n'.format(estimatorPath))
    try:
        scoresDF.loc[idxSl[:, lastFoldIdx], :].to_hdf(estimatorPath, 'work')
        #############################################################################
        scoresDF.loc[:, ['test_score', 'train_score']].to_hdf(estimatorPath, 'cv_scores')
    except Exception:
        traceback.print_exc()
    #
    if hasattr(scoresDF['estimator'].iloc[0].named_steps['dim_red'], 'n_iter_'):
        nIters = scoresDF['estimator'].apply(lambda es: es.named_steps['dim_red'].n_iter_)
        print('n iterations per estimator:\n{}'.format(nIters))
        nIters.to_hdf(estimatorPath, 'cv_estimators_n_iter')
    try:
        scoresDF['estimator'].to_hdf(estimatorPath, 'cv_estimators')
    except Exception:
        traceback.print_exc()
        pdb.set_trace()
    #
    '''jb.dump(chosenEstimator, estimatorPath.replace('.h5', '.joblib'))'''
    #
    estimatorMetadata = {
        'path': os.path.basename(estimatorPath),
        'name': estimatorName,
        'datasetName': datasetName,
        'selectionName': selectionName,
        'outputFeatures': featuresDF.columns
        }
    with open(estimatorMetaDataPath, 'wb') as f:
        pickle.dump(estimatorMetadata, f)
    outputSelectionName = '{}_{}'.format(
        selectionName, estimatorName)
    outputLoadingMeta = deepcopy(loadingMeta)
    if 'pca' in estimatorName:
        outputLoadingMeta['arguments']['unitQuery'] = 'pca'
    elif 'fa' in estimatorName:
        outputLoadingMeta['arguments']['unitQuery'] = 'factor'
    #
    # 'decimate', 'procFun', 'addLags' were already applied, no need to apply them again
    for k in ['decimate', 'procFun', 'addLags']:
        outputLoadingMeta['alignedAsigsKWargs'].pop(k, None)
    # 'normalizeDataset', 'unNormalizeDataset' were already applied, no need to apply them again
    for k in ['normalizeDataset', 'unNormalizeDataset']:
        outputLoadingMeta.pop(k, None)
        #
        def passthr(df, params):
            return df
        #
        outputLoadingMeta[k] = passthr
    # 
    featuresDF.to_hdf(
        datasetPath,
        '/{}/data'.format(outputSelectionName),
        mode='a')
    #
    maskList = []
    allGroupIdx = pd.MultiIndex.from_tuples(
        [tuple('all' for fgn in featureColumnFields)],
        names=featureColumnFields)
    # iterate principle components by feature
    for name, group in featuresDF.groupby('feature', axis='columns'):
        attrValues = ['all' for fgn in featureColumnFields]
        attrValues[featureColumnFields.index('feature')] = name
        thisMask = pd.Series(
            featuresDF.columns.isin(group.columns),
            index=featuresDF.columns).to_frame()
        thisMask.columns = pd.MultiIndex.from_tuples(
            (attrValues, ), names=featureColumnFields)
        maskList.append(thisMask.T)
    # iterate principle components by frequency band of input data
    for name, group in featuresDF.groupby('freqBandName', axis='columns'):
        attrValues = ['all' for fgn in featureColumnFields]
        attrValues[featureColumnFields.index('freqBandName')] = name
        thisMask = pd.Series(
            featuresDF.columns.isin(group.columns),
            index=featuresDF.columns).to_frame()
        thisMask.columns = pd.MultiIndex.from_tuples(
            (attrValues, ), names=featureColumnFields)
        maskList.append(thisMask.T)
    #
    maskDF = pd.concat(maskList)
    maskParams = [
        {k: v for k, v in zip(maskDF.index.names, idxItem)}
        for idxItem in maskDF.index
        ]
    maskParamsStr = [
        '{}'.format(idxItem).replace("'", '')
        for idxItem in maskParams]
    maskDF.loc[:, 'maskName'] = maskParamsStr
    maskDF.set_index('maskName', append=True, inplace=True)
    maskDF.to_hdf(
        datasetPath,
        '/{}/featureMasks'.format(outputSelectionName),
        mode='a')
    ###
    outputLoadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(outputSelectionName) + '_meta.pickle'
        )
    with open(outputLoadingMetaPath, 'wb') as f:
        pickle.dump(outputLoadingMeta, f)
print('\n' + '#' * 50 + '\n{}\nComplete.\n'.format(__file__) + '#' * 50 + '\n')