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
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from docopt import docopt
from copy import deepcopy
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
# pdb.set_trace()
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
    iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
    cv_kwargs = loadingMeta.pop('cv_kwargs')
for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
    loadingMeta['arguments'].pop(argName, None)
arguments.update(loadingMeta['arguments'])


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
    cvIterator = iteratorsBySegment[0]
    if 'pca' in arguments['estimatorName']:
        estimatorClass = PCA
        estimatorKWArgs = dict(svd_solver='auto')
    elif 'fa' in arguments['estimatorName']:
        estimatorClass = FactorAnalysis
        estimatorKWArgs = dict()
    crossvalKWArgs = dict(
        cv=cvIterator,
        return_train_score=True, return_estimator=True)
    joblibBackendArgs = dict(
        backend='dask'
        )
    if joblibBackendArgs['backend'] == 'dask':
        daskComputeOpts = dict(
            # scheduler='threads'
            # scheduler='processes'
            scheduler='single-threaded'
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
    # only use zero lag targets    
    dataDF = dataDF.xs(0, level='lag', axis='columns')
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    workIdx = cvIterator.work
    workingDataDF = dataDF.iloc[workIdx, :]
    prf.print_memory_usage('just loaded data, fitting')
    nFeatures = dataDF.columns.shape[0]
    #
    nCompsToTest = range(1, nFeatures + 1, 3)
    if arguments['debugging']:
        nCompsToTest = range(50, 100, 5)
    #
    scores = {}
    for n in nCompsToTest:
        if arguments['verbose']:
            print('tdr.crossValidationScores evaluating with {} components'.format(n))
        ekwa = estimatorKWArgs.copy()
        ekwa['n_components'] = n
        scores[n] = tdr.crossValidationScores(
            dataDF, None, estimatorClass,
            estimatorKWArgs=ekwa,
            crossvalKWArgs=crossvalKWArgs,
            joblibBackendArgs=joblibBackendArgs,
            verbose=arguments['verbose']
            )
    scoresDF = pd.concat(
        {nc: pd.DataFrame(scr) for nc, scr in scores.items()},
        names=['nComponents', 'fold'])
    scoresDF.dropna(axis='columns', inplace=True)
    #
    nCompsMaxMLE = scoresDF.groupby(['nComponents']).mean()['test_score'].idxmax()
    maxMLE = scoresDF.xs(nCompsMaxMLE, level='nComponents', axis='index').mean()['test_score']
    lastFoldIdx = scoresDF.index.get_level_values('fold').max()
    mostComps = scoresDF.index.get_level_values('nComponents').max()
    pcaFull = scoresDF.loc[(mostComps, lastFoldIdx), 'estimator']
    chosenEstimator = scoresDF.loc[(nCompsMaxMLE, lastFoldIdx), 'estimator']
    pdb.set_trace()
    # make transformed dataset
    if 'pca' in arguments['estimatorName']:
        outputFeatures = ['pca{:03d}'.format(nc) for nc in range(1, chosenEstimator.n_components + 1)]
    elif 'fa' in arguments['estimatorName']:
        outputFeatures = ['factor{:03d}'.format(nc) for nc in range(1, chosenEstimator.n_components + 1)]
    #
    features = chosenEstimator.transform(dataDF)
    featureColumnFields = dataDF.columns.names
    featureColumns = pd.DataFrame(np.nan, index=range(features.shape[1]), columns=featureColumnFields)
    for fcn in featureColumnFields:
        if fcn == 'feature':
            featureColumns.loc[:, fcn] = outputFeatures
        else:
            featureColumns.loc[:, fcn] = 'NA'
    #
    featuresDF = pd.DataFrame(features, index=dataDF.index, columns=pd.MultiIndex.from_frame(featureColumns))
    outputDatasetName = '{}_{}_{}_{}_{}'.format(
        arguments['unitQuery'], arguments['estimatorName'],
        arguments['iteratorSuffix'], arguments['window'], arguments['alignQuery'])
    outputDFPath = os.path.join(
        dataFramesFolder, outputDatasetName + '.h5'
    )
    outputLoadingMeta = deepcopy(loadingMeta)
    # these were already applied, no need to apply them again
    for k in ['decimate', 'procFun', 'addLags']:
        outputLoadingMeta['alignedAsigsKWargs'].pop(k, None)
    for k in ['normalizeDataset', 'unNormalizeDataset']:
        outputLoadingMeta.pop(k, None)
        outputLoadingMeta[k] = lambda x: x
    featuresDF.to_hdf(
        outputDFPath,
        '{}_{}'.format(arguments['unitQuery'], arguments['estimatorName']),
        mode='a')
    with open(outputDFPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(outputLoadingMeta, f)
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
        'outputFeatures': outputFeatures
        }
    with open(estimatorPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(estimatorMetadata, f)
    '''
    alignedAsigsKWargs['unitNames'] = saveUnitNames
    alignedAsigsKWargs['unitQuery'] = None
    alignedAsigsKWargs.pop('dataQuery', None)
    # pdb.set_trace()
        '''
    if 'pca' in arguments['estimatorName']:
        cumExplVariance = pd.Series(
            np.cumsum(pcaFull.explained_variance_ratio_),
            index=[idx + 1 for idx in range(pcaFull.explained_variance_ratio_.shape[0])])
        ######
        varCutoff = 0.95
        nCompsCutoff = cumExplVariance.index[cumExplVariance > varCutoff].min()
    ######
    if arguments['verbose']:
        print('Calculating ledoit-wolf # of components...')
    lWScores = calc_lw_score(dataDF, cv=cvIterator)
    lWForPlot = pd.concat({
        nc: pd.DataFrame({'test_score': lWScores})
        for nc in nCompsToTest}, names=['nComponents', 'fold'])
    scoresForPlot = pd.concat(
        {
            'PCA_test': scoresDF['test_score'],
            'PCA_train': scoresDF['train_score'],
            'ledoitWolfMLE': lWForPlot['test_score']},
        names=['evalType']).to_frame(name='score').reset_index()
    validationMask = (
        (scoresForPlot['fold'] == lastFoldIdx) &
        (scoresForPlot['evalType'] == 'PCA_test'))
    scoresForPlot.loc[validationMask, 'evalType'] = 'PCA_validation'
    workingMask = (
        (scoresForPlot['fold'] == lastFoldIdx) &
        (scoresForPlot['evalType'] == 'PCA_train'))
    scoresForPlot.loc[workingMask, 'evalType'] = 'PCA_work'
    # pdb.set_trace()
    if arguments['plotting']:
        figureOutputPath = os.path.join(
                figureOutputFolder,
                '{}_{}_dimensionality.pdf'.format(
                    arguments['estimatorName'], datasetName))
        with PdfPages(figureOutputPath) as pdf:
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 8)
            sns.lineplot(
                data=scoresForPlot,
                x='nComponents', y='score',
                hue='evalType', ci='sem', ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            textDescr = 'num. components that maximize likelihood'
            lineMax, = ax.plot(
                nCompsMaxMLE, maxMLE,
                'r*', label=textDescr)
            handles.append(lineMax)
            labels.append(textDescr)
            ax.legend(handles, labels)
            ax.set_xlabel('number of components')
            ax.set_ylabel('average log-likelihood')
            fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            if 'pca' in arguments['estimatorName']:
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 8)
                ax.plot(cumExplVariance)
                ax.plot(
                    nCompsMaxMLE,
                    cumExplVariance.loc[nCompsMaxMLE],
                    '*')
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