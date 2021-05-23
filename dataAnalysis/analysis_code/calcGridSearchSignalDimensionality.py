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
    --verbose=verbose                      print diagnostics? [default: 0]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
from dask.distributed import Client, LocalCluster
import os, traceback
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer
import joblib as jb
import dill as pickle
import gc
from docopt import docopt
from copy import deepcopy

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
    idxSl = pd.IndexSlice
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    #
    arguments['verbose'] = int(arguments['verbose'])
    #
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], arguments['alignFolderName'])
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    #
    '''fullEstimatorName = '{}_{}_to_{}{}_{}_{}'.format(
        arguments['estimatorName'],
        arguments['unitQueryLhs'], arguments['unitQueryRhs'],
        iteratorSuffix,
        arguments['window'],
        arguments['alignQuery'])'''
    datasetName = arguments['datasetName']
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], arguments['datasetName'])
    #
    estimatorsSubFolder = os.path.join(
        alignSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(alignSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    #
    with open(datasetPath.replace('.h5', '_meta.pickle'), 'rb') as _f:
        loadingMeta = pickle.load(_f)
        # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
        iteratorsBySegment = loadingMeta['iteratorsBySegment']
        cv_kwargs = loadingMeta['cv_kwargs']
    for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
        loadingMeta['arguments'].pop(argName, None)
    arguments.update(loadingMeta['arguments'])
    cvIterator = iteratorsBySegment[0]
    if 'pca' in arguments['estimatorName']:
        estimatorClass = PCA
        estimatorKWArgs = dict(svd_solver='auto')
    elif 'fa' in arguments['estimatorName']:
        estimatorClass = FactorAnalysis
        estimatorKWArgs = dict(
            max_iter=5000,
            iterated_power=4,
            tol=5e-3
        )
    gridSearchKWArgs = dict(
        return_train_score=True,
        cv=cvIterator,
        param_grid=dict())
    crossvalKWArgs = dict(
        cv=cvIterator,
        return_train_score=True, return_estimator=True)
    joblibBackendArgs = dict(
        backend='dask'
        # backend='loky'
        )
    if joblibBackendArgs['backend'] == 'dask':
        daskComputeOpts = dict(
            # scheduler='threads'
            scheduler='processes'
            # scheduler='single-threaded'
            )
        if joblibBackendArgs['backend'] == 'dask':
            if daskComputeOpts['scheduler'] == 'single-threaded':
                daskClient = Client(LocalCluster(n_workers=1))
            elif daskComputeOpts['scheduler'] == 'processes':
                daskClient = Client(LocalCluster(processes=True))
            elif daskComputeOpts['scheduler'] == 'threads':
                daskClient = Client(LocalCluster(processes=False))
            else:
                print('Scheduler name is not correct!')
                daskClient = Client()
    dataDF = pd.read_hdf(datasetPath, datasetName)
    featureMasks = pd.read_hdf(datasetPath, datasetName + '_featureMasks')
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
    cvScoresDict = {}
    gsScoresDict = {}
    gridSearcherDict = {}
    lOfColumnTransformers = []
    ###
    # remove the 'all' column?
    removeAllColumn = True
    if removeAllColumn:
        featureMasks = featureMasks.loc[~ featureMasks.all(axis='columns'), :]
    ###
    outputFeatureList = []
    featureColumnFields = dataDF.columns.names
    ###
    for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
        maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
        dataGroup = dataDF.loc[:, featureMask]
        nFeatures = dataGroup.columns.shape[0]
        nCompsToTest = range(1, min(120, nFeatures + 1), 3)
        trfName = '{}_{}'.format(arguments['estimatorName'], maskParams['freqBandName'])
        if arguments['verbose']:
            print('Fitting {} ...'.format(trfName))
        if arguments['debugging']:
            nCompsToTest = range(1, min(80, nFeatures + 1), 3)
        gridSearchKWArgs['param_grid']['n_components'] = [nc for nc in nCompsToTest]
        cvScores, gridSearcherDict[maskParams['freqBandName']], gsScoresDict[maskParams['freqBandName']] = tdr.gridSearchHyperparameters(
            dataGroup, estimatorClass=estimatorClass,
            verbose=int(arguments['verbose']),
            recalculateBestEstimator=True,
            gridSearchKWArgs=gridSearchKWArgs,
            crossvalKWArgs=crossvalKWArgs,
            estimatorKWArgs=estimatorKWArgs,
            joblibBackendArgs=joblibBackendArgs
            )
        cvScoresDF = pd.DataFrame(cvScores)
        cvScoresDF.index.name = 'fold'
        cvScoresDF.dropna(axis='columns', inplace=True)
        cvScoresDict[maskParams['freqBandName']] = cvScoresDF
        lastFoldIdx = cvScoresDF.index.get_level_values('fold').max()
        bestEstimator = cvScoresDF.loc[lastFoldIdx, 'estimator']
        # bestEstimator = gridSearcherDict[maskParams['freqBandName']].best_estimator_
        lOfColumnTransformers.append((
            # transformer name
            trfName,
            # estimator
            bestEstimator,
            # columns
            dataGroup.columns.copy()
            ))
        featureColumns = pd.DataFrame(
            np.nan,
            index=range(bestEstimator.n_components),
            columns=featureColumnFields)
        for fcn in featureColumnFields:
            if fcn == 'feature':
                featureColumns.loc[:, fcn] = [
                    '{}{:0>3d}#0'.format(trfName, nc)
                    for nc in range(1, bestEstimator.n_components + 1)]
            elif fcn == 'lag':
                featureColumns.loc[:, fcn] = 0
            else:
                featureColumns.loc[:, fcn] = maskParams[fcn]
        outputFeatureList.append(featureColumns)
    #
    chosenEstimator = ColumnTransformer(lOfColumnTransformers)
    chosenEstimator.fit(workingDataDF)
    outputFeaturesIndex = pd.MultiIndex.from_frame(
        pd.concat(outputFeatureList).reset_index(drop=True))
    #
    scoresDF = pd.concat(cvScoresDict, names=['freqBandName'])
    lastFoldIdx = scoresDF.index.get_level_values('fold').max()
    #
    gsScoresDF = pd.concat(gsScoresDict, names=['freqBandName'])
    #
    '''nCompsMaxMLE = scoresDF.groupby(['nComponents']).mean()['test_score'].idxmax()
    maxMLE = scoresDF.xs(nCompsMaxMLE, level='nComponents', axis='index').mean()['test_score']
    mostComps = scoresDF.index.get_level_values('nComponents').max()
    pcaFull = scoresDF.loc[(mostComps, lastFoldIdx), 'estimator']
    chosenEstimator = scoresDF.loc[(nCompsMaxMLE, lastFoldIdx), 'estimator']
    #
    '''
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    # 
    try:
        scoresDF.loc[idxSl[:, lastFoldIdx], :].to_hdf(estimatorPath, 'work')
        scoresDF.loc[:, ['test_score', 'train_score']].to_hdf(estimatorPath, 'cv')
    except Exception:
        traceback.print_exc()
    #
    try:
        scoresDF['estimator'].to_hdf(estimatorPath, 'cv_estimators')
    except Exception:
        traceback.print_exc()
    #
    jb.dump(chosenEstimator, estimatorPath.replace('.h5', '.joblib'))
    #
    estimatorMetadata = {
        'path': os.path.basename(estimatorPath),
        'name': arguments['estimatorName'],
        'datasetName': datasetName,
        'outputFeatures': outputFeaturesIndex
        }
    with open(estimatorPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(estimatorMetadata, f)

    features = chosenEstimator.transform(dataDF)
    #
    featuresDF = pd.DataFrame(
        features, index=dataDF.index,
        columns=outputFeaturesIndex)
    outputDatasetName = '{}_{}_{}_{}_{}'.format(
        arguments['unitQuery'], arguments['estimatorName'],
        arguments['iteratorSuffix'], arguments['window'], arguments['alignQuery'])
    outputDFPath = os.path.join(
        dataFramesFolder, outputDatasetName + '.h5'
        )
    outputLoadingMeta = deepcopy(loadingMeta)
    if 'pca' in arguments['estimatorName']:
        outputLoadingMeta['arguments']['unitQuery'] = 'pca'
    elif 'fa' in arguments['estimatorName']:
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
        outputDFPath, outputDatasetName,
        mode='a')
    #
    maskList = []
    allGroupIdx = pd.MultiIndex.from_tuples(
        [tuple('all' for fgn in featureColumnFields)],
        names=featureColumnFields)
    if arguments['unitQuery'] == 'lfp_CAR_spectral':
        # each freq band
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
    maskDF.to_hdf(outputDFPath, outputDatasetName + '_featureMasks', mode='a')
    ###
    with open(outputDFPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(outputLoadingMeta, f)
    if arguments['plotting']:
        figureOutputPath = os.path.join(
                figureOutputFolder,
                '{}_{}_dimensionality.pdf'.format(
                    arguments['estimatorName'], datasetName))
        with PdfPages(figureOutputPath) as pdf:
            for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
                maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
                dataGroup = dataDF.loc[:, featureMask]
                nFeatures = dataGroup.columns.shape[0]
                nCompsToTest = range(1, min(120, nFeatures + 1), 3)
                if arguments['verbose']:
                    print('Calculating ledoit-wolf # of components...')
                lWScores = calc_lw_score(dataGroup, cv=cvIterator)
                lWForPlot = pd.concat({
                    nc: pd.DataFrame({'test_score': lWScores})
                    for nc in nCompsToTest}, names=['n_components', 'fold'])
                scoresForPlot = pd.concat(
                    {
                        'PCA_test': gsScoresDF.xs(maskParams['freqBandName']).loc[:, 'test_score'],
                        'PCA_train': gsScoresDF.xs(maskParams['freqBandName']).loc[:, 'train_score'],
                        'ledoitWolfMLE': lWForPlot.swaplevel('n_components', 'fold')['test_score']},
                    names=['evalType']).to_frame(name='score').reset_index()
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
                    x='n_components', y='score',
                    hue='evalType', ci='sem', ax=ax)
                handles, labels = ax.get_legend_handles_labels()
                textDescr = 'Chosen parameters and corresponding score'
                # pdb.set_trace()
                bestScore = cvScoresDict[maskParams['freqBandName']].loc[lastFoldIdx, ['test_score']].iloc[0]
                bestEstimator = cvScoresDict[maskParams['freqBandName']].loc[lastFoldIdx, ['estimator']].iloc[0]
                lineMax, = ax.plot(
                    bestEstimator.n_components, bestScore,
                    'r*', label=textDescr)
                handles.append(lineMax)
                labels.append(textDescr)
                ax.legend(handles, labels)
                ax.set_xlabel('number of components')
                ax.set_ylabel('average log-likelihood')
                fig.tight_layout(pad=1)
                titleText = fig.suptitle('{}'.format(maskIdx))
                pdf.savefig(bbox_inches='tight', pad_inches=0)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                if 'pca' in arguments['estimatorName']:
                    cumExplVariance = pd.Series(
                        np.cumsum(bestEstimator.explained_variance_ratio_),
                        index=[idx + 1 for idx in range(bestEstimator.explained_variance_ratio_.shape[0])])
                ######
                    fig, ax = plt.subplots()
                    fig.set_size_inches(12, 8)
                    ax.plot(cumExplVariance, label='cumulative explained variance')
                    handles, labels = ax.get_legend_handles_labels()
                    textDescr = 'Chosen parameters and corresponding num. components'
                    lineMax, = ax.plot(
                        bestEstimator.n_components,
                        cumExplVariance.loc[bestEstimator.n_components],
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
