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
    --verbose=verbose                      print diagnostics? [default: 0]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
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
from docopt import docopt
from copy import deepcopy
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
##
##
'''def compute_scores(
        X, estimator,
        nComponentsToTest,
        cv, estimatorKWArgs={},
        ,
        verbose=False):
    
        scores[n] = cross_validate(
            instance, X.to_numpy(), cv=cv,
            return_train_score=return_train_score,
            return_estimator=return_estimator)
    return scores'''

def shrunk_cov_score(
        X, cvIterator):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(
        ShrunkCovariance(),
        {'shrinkage': shrinkages}, cv=cvIterator)
    cv.fit(X)
    cvResDF = pd.DataFrame(cv.cv_results_)
    return cvResDF['mean_test_score'].max()


def calc_lw_score(
        X, cv):
    result = cross_val_score(
        LedoitWolf(), X.to_numpy(), cv=cv)
    return result


if __name__ == '__main__':
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    ##
    '''
    consoleDebugging = True
    if consoleDebugging:
        arguments = {
            'alignFolderName': 'motion', 'lazy': False, 'debugging': False,
            'datasetName': 'Block_XL_df_b', 'blockIdx': '2', 'window': 'long',
            'processAll': True, 'plotting': True, 'verbose': '1', 'exp': 'exp202101281100',
            'showFigures': False, 'selectionName': 'lfp_CAR', 'estimatorName': 'pca',
            'analysisName': 'default'}
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
    if 'pca' in estimatorName:
        estimatorClass = PCA
        estimatorKWArgs = dict(svd_solver='auto')
    elif 'fa' in estimatorName:
        estimatorClass = FactorAnalysis
        estimatorKWArgs = dict(
            max_iter=1000,
            iterated_power=3,
            tol=1e-2
        )
    gridSearchKWArgs = dict(
        return_train_score=True,
        cv=cvIterator,
        refit=False, scoring=reconstructionR2,
        param_grid=dict())
    crossvalKWArgs = dict(
        cv=cvIterator,
        return_train_score=True, return_estimator=True,
        scoring=reconstructionR2,
    )
    ###
    scoreIsLikelihood = False
    ###
    joblibBackendArgs = dict(
        # backend='dask'
        backend='loky',
        )
    if joblibBackendArgs['backend'] == 'dask':
        daskComputeOpts = dict(
            # scheduler='threads'
            scheduler='processes'
            # scheduler='single-threaded'
            )
        if daskComputeOpts['scheduler'] == 'single-threaded':
            daskClient = None
        elif daskComputeOpts['scheduler'] == 'processes':
            # daskClient = Client(LocalCluster(processes=True))
            daskClient = Client(LocalCluster(processes=True))
        elif daskComputeOpts['scheduler'] == 'threads':
            daskClient = Client(LocalCluster(processes=False))
        else:
            print('Scheduler name is not correct!')
            daskClient = Client()
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    # only use zero lag targets
    lagMask = dataDF.columns.get_level_values('lag') == 0
    dataDF = dataDF.loc[:, lagMask]
    featureMasks = featureMasks.loc[:, lagMask]
    #
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    workIdx = cvIterator.work
    workingDataDF = dataDF.iloc[workIdx, :]
    prf.print_memory_usage('just loaded data, fitting')
    #
    cvScoresDict = {}  # cross validated best estimator
    fmScoresDict = {}  # cross validated estimator with max n comps
    gsScoresDict = {}
    gridSearcherDict = {}
    '''lOfColumnTransformers = []'''
    ###
    # remove the 'all' column?
    removeAllColumn = False
    if removeAllColumn:
        featureMasks = featureMasks.loc[~ featureMasks.all(axis='columns'), :]
    ###
    '''outputFeatureNameList = []'''
    featuresList = []
    featureColumnFields = dataDF.columns.names
    ###

    maxNCompsToTest = min(80, featureMasks.sum(axis='columns').min())
    # pdb.set_trace()
    listOfNCompsToTest = []
    for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
        maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
        dataGroup = dataDF.loc[:, featureMask]
        nFeatures = dataGroup.columns.shape[0]
        if arguments['debugging']:
            nCompsToTest = range(1, min(maxNCompsToTest, nFeatures), 5)
        else:
            nCompsToTest = range(1, min(maxNCompsToTest, nFeatures), 1)
        listOfNCompsToTest.append(nCompsToTest)
        trfName = '{}_{}'.format(estimatorName, maskParams['freqBandName'])
        ###
        if arguments['averageByTrial']:
            estimatorInstance = Pipeline([
                ('averager', tdr.DataFrameAverager(
                    stimConditionNames=stimulusConditionNames + ['bin'],
                    addIndexFor=stimulusConditionNames, burnInPeriod=500e-3)),
                ('dim_red', estimatorClass(**estimatorKWArgs))])
        else:
            estimatorInstance = Pipeline([
                ('averager', tdr.DataFramePassThrough()),
                ('dim_red', estimatorClass(**estimatorKWArgs))])
        ###
        if arguments['verbose']:
            print('Fitting {} ...'.format(trfName))
        gridSearchKWArgs['param_grid']['dim_red__n_components'] = [nc for nc in nCompsToTest]
        print('n_components = {}'.format(gridSearchKWArgs['param_grid']['dim_red__n_components']))
        ###
        estimatorInstance.fit_transform(dataGroup)
        cvScores, gridSearcherDict[maskParams['freqBandName']], gsScoresDict[maskParams['freqBandName']] = tdr.gridSearchHyperparameters(
            dataGroup,
            # estimatorClass=estimatorClass, estimatorKWArgs=estimatorKWArgs,
            estimatorInstance=estimatorInstance,
            verbose=int(arguments['verbose']),
            recalculateBestEstimator=True,
            gridSearchKWArgs=gridSearchKWArgs,
            crossvalKWArgs=crossvalKWArgs,
            joblibBackendArgs=joblibBackendArgs
            )
        cvScoresDF = pd.DataFrame(cvScores)
        cvScoresDF.index.name = 'fold'
        cvScoresDF.dropna(axis='columns', inplace=True)
        cvScoresDict[maskParams['freqBandName']] = cvScoresDF
        #
        fullEstimatorKWArgs = estimatorKWArgs.copy()
        fullEstimatorKWArgs['n_components'] = maxNCompsToTest
        if arguments['averageByTrial']:
            fullEstimatorInstance = Pipeline([
                ('averager', tdr.DataFrameAverager(
                    stimConditionNames=stimulusConditionNames + ['bin'], addIndexFor=stimulusConditionNames)),
                ('dim_red', estimatorClass(**fullEstimatorKWArgs))])
        else:
            fullEstimatorInstance = Pipeline([
                ('averager', tdr.DataFramePassThrough()),
                ('dim_red', estimatorClass(**fullEstimatorKWArgs))])
        fullModelScores = tdr.crossValidationScores(
            dataGroup,
            # estimatorClass=estimatorClass, estimatorKWArgs=fullEstimatorKWArgs,
            estimatorInstance=fullEstimatorInstance,
            crossvalKWArgs=crossvalKWArgs,
            joblibBackendArgs=joblibBackendArgs,
            verbose=int(arguments['verbose']),)
        fmScoresDF = pd.DataFrame(fullModelScores)
        fmScoresDF.index.name = 'fold'
        fmScoresDF.dropna(axis='columns', inplace=True)
        fmScoresDict[maskParams['freqBandName']] = fmScoresDF
        #
        lastFoldIdx = cvScoresDF.index.get_level_values('fold').max()
        bestEstimator = cvScoresDF.loc[lastFoldIdx, 'estimator']
        # bestEstimator = gridSearcherDict[maskParams['freqBandName']].best_estimator_
        '''lOfColumnTransformers.append((
            # transformer name
            trfName,
            # estimator
            bestEstimator,
            # columns
            dataGroup.columns.copy()
            ))'''
        featureColumns = pd.DataFrame(
            np.nan,
            index=range(bestEstimator.get_params()['dim_red__n_components']),
            columns=featureColumnFields)
        for fcn in featureColumnFields:
            if fcn == 'feature':
                featureColumns.loc[:, fcn] = [
                    '{}{:0>3d}'.format(trfName, nc)
                    for nc in range(1, bestEstimator.get_params()['dim_red__n_components'] + 1)]
            elif fcn == 'lag':
                featureColumns.loc[:, fcn] = 0
            else:
                featureColumns.loc[:, fcn] = maskParams[fcn]
        '''outputFeatureNameList.append(featureColumns)'''
        # if arguments['averageByTrial']:
        preEst = Pipeline(bestEstimator.steps[:-1])
        xInterDF = preEst.fit_transform(dataGroup)
        if isinstance(xInterDF, pd.DataFrame):
            featuresIndex = xInterDF.index
        else:
            featuresIndex = pd.Index(range(xInterDF.shape[0]))
        # else:
        #     featuresIndex = dataGroup.index
        featuresList.append(pd.DataFrame(
            bestEstimator.transform(dataGroup),
            index=featuresIndex, columns=pd.MultiIndex.from_frame(featureColumns)))
    #
    '''chosenEstimator = ColumnTransformer(lOfColumnTransformers)
    chosenEstimator.fit(workingDataDF)'''
    #
    '''outputFeaturesIndex = pd.MultiIndex.from_frame(
        pd.concat(outputFeatureNameList).reset_index(drop=True))'''
    featuresDF = pd.concat(featuresList, axis='columns')
    #
    scoresDF = pd.concat(cvScoresDict, names=['freqBandName'])
    scoresFullModel = pd.concat(fmScoresDict, names=['freqBandName'])
    lastFoldIdx = scoresDF.index.get_level_values('fold').max()
    #
    gsScoresDF = pd.concat(gsScoresDict, names=['freqBandName'])
    #
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        print('Deleting contents of {}'.format(estimatorPath))
        os.remove(estimatorPath)
    #
    print('\n\nSaving {}\n\n'.format(estimatorPath))
    try:
        scoresDF.loc[idxSl[:, lastFoldIdx], :].to_hdf(estimatorPath, 'work')
        scoresFullModel.loc[:, ['test_score', 'train_score']].to_hdf(estimatorPath, 'full_scores')
        scoresDF.loc[:, ['test_score', 'train_score']].to_hdf(estimatorPath, 'cv_scores')
    except Exception:
        traceback.print_exc()
        pdb.set_trace()
    #
    try:
        scoresDF['estimator'].to_hdf(estimatorPath, 'cv_estimators')
        scoresFullModel['estimator'].to_hdf(estimatorPath, 'full_estimators')
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
    if arguments['plotting']:
        pdfPath = os.path.join(
                figureOutputFolder,
                '{}_dimensionality.pdf'.format(
                    fullEstimatorName))
        print('Saving plots to {}'.format(pdfPath))
        with PdfPages(pdfPath) as pdf:
            for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
                maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
                dataGroup = dataDF.loc[:, featureMask]
                nFeatures = dataGroup.columns.shape[0]
                nCompsToTest = listOfNCompsToTest[idx]
                scoresForPlotDict = {
                    'PCA_test': gsScoresDF.xs(maskParams['freqBandName']).loc[:, 'test_score'],
                    'PCA_train': gsScoresDF.xs(maskParams['freqBandName']).loc[:, 'train_score']}
                if scoreIsLikelihood:
                    if arguments['verbose']:
                        print('Calculating ledoit-wolf # of components...')
                    lWScores = calc_lw_score(dataGroup, cv=cvIterator)
                    lWForPlot = pd.concat({
                        nc: pd.DataFrame({'test_score': lWScores})
                        for nc in nCompsToTest}, names=['dim_red__n_components', 'fold'])
                    scoresForPlotDict['ledoitWolfMLE'] = lWForPlot.swaplevel('dim_red__n_components', 'fold')['test_score']
                scoresForPlot = pd.concat(scoresForPlotDict, names=['evalType']).to_frame(name='score').reset_index()
                validationMask = (
                    (scoresForPlot['fold'] == lastFoldIdx) &
                    (scoresForPlot['evalType'] == 'PCA_test'))
                scoresForPlot.loc[validationMask, 'evalType'] = 'PCA_validation'
                workingMask = (
                    (scoresForPlot['fold'] == lastFoldIdx) &
                    (scoresForPlot['evalType'] == 'PCA_train'))
                scoresForPlot.loc[workingMask, 'evalType'] = 'PCA_work'
                #
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 8)
                sns.lineplot(
                    data=scoresForPlot,
                    x='dim_red__n_components', y='score',
                    hue='evalType', errorbar='se', ax=ax)
                handles, labels = ax.get_legend_handles_labels()
                textDescr = 'Chosen parameters and corresponding score'
                bestScore = cvScoresDict[maskParams['freqBandName']].mean()['test_score']
                bestEstimator = cvScoresDict[maskParams['freqBandName']].loc[lastFoldIdx, ['estimator']].iloc[0]
                lineMax, = ax.plot(
                    bestEstimator.get_params()['dim_red__n_components'], bestScore,
                    'r*', label=textDescr)
                handles.append(lineMax)
                labels.append(textDescr)
                ax.legend(handles, labels)
                ax.set_xlabel('number of components')
                if scoreIsLikelihood:
                    ax.set_ylabel('average log-likelihood')
                elif 'scoring' in crossvalKWArgs:
                    ax.set_ylabel('{}'.format(crossvalKWArgs['scoring']))
                fig.tight_layout(pad=1)
                titleText = fig.suptitle('{}'.format(maskParams['freqBandName']))
                pdf.savefig(bbox_inches='tight', pad_inches=0)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                if 'pca' in estimatorName:
                    cumExplVariance = pd.Series(
                        np.cumsum(bestEstimator.steps[-1][1].explained_variance_ratio_),
                        index=[idx + 1 for idx in range(bestEstimator.steps[-1][1].explained_variance_ratio_.shape[0])])
                ######
                    fig, ax = plt.subplots()
                    fig.set_size_inches(12, 8)
                    ax.plot(cumExplVariance, label='cumulative explained variance')
                    handles, labels = ax.get_legend_handles_labels()
                    textDescr = 'Chosen parameters and corresponding num. components'
                    lineMax, = ax.plot(
                        bestEstimator.steps[-1][1].n_components,
                        cumExplVariance.loc[bestEstimator.steps[-1][1].n_components],
                        'r*', label=textDescr)
                    handles.append(lineMax)
                    labels.append(textDescr)
                    ax.set_ylim((0, 1))
                    ax.set_xlabel('number of components')
                    ax.set_ylabel('explained variance')
                    fig.tight_layout(pad=1)
                    pdf.savefig(bbox_inches='tight', pad_inches=0)
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
    del dataDF
    gc.collect()
    #
    prf.print_memory_usage('Done fitting')
