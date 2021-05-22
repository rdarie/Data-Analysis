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
from copy import deepcopy
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from docopt import docopt

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
    # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
    iteratorsBySegment = loadingMeta['iteratorsBySegment']
    cv_kwargs = loadingMeta['cv_kwargs']
for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
    loadingMeta['arguments'].pop(argName, None)
arguments.update(loadingMeta['arguments'])


if __name__ == '__main__':
    cvIterator = iteratorsBySegment[0]
    if 'mahal' in arguments['estimatorName']:
        # estimatorClass = EmpiricalCovariance
        estimatorClass = tdr.EmpiricalCovarianceTransformer
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
    # remove the 'all' column
    featureMasks = featureMasks.loc[~ featureMasks.all(axis='columns'), :]
    #
    outputFeatureList = []
    featureColumnFields = dataDF.columns.names
    cvScoresDict = {}
    lOfColumnTransformers = []
    # pdb.set_trace()
    for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
        maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
        dataGroup = dataDF.loc[:, featureMask]
        trfName = '{}_{}'.format(arguments['estimatorName'], maskParams['freqBandName'])
        if arguments['verbose']:
            print('Fitting {} ...'.format(trfName))
        nFeatures = dataGroup.columns.shape[0]
        cvScores = tdr.crossValidationScores(
            dataGroup, None, estimatorClass,
            estimatorKWArgs=estimatorKWArgs,
            crossvalKWArgs=crossvalKWArgs,
            joblibBackendArgs=joblibBackendArgs,
            verbose=arguments['verbose']
            )
        cvScoresDF = pd.DataFrame(cvScores)
        cvScoresDF.index.name = 'fold'
        cvScoresDF.dropna(axis='columns', inplace=True)
        cvScoresDict[maskParams['freqBandName']] = cvScoresDF
        #
        lastFoldIdx = cvScoresDF.index.get_level_values('fold').max()
        bestEstimator = cvScoresDF.loc[lastFoldIdx, 'estimator']
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
            index=[idx],
            columns=featureColumnFields)
        for fcn in featureColumnFields:
            if fcn == 'feature':
                featureColumns.loc[idx, fcn] = trfName
            elif fcn == 'lag':
                featureColumns.loc[idx, fcn] = 0
            else:
                featureColumns.loc[idx, fcn] = maskParams[fcn]
        outputFeatureList.append(featureColumns)
    #
    chosenEstimator = ColumnTransformer(lOfColumnTransformers)
    chosenEstimator.fit(workingDataDF)
    outputFeaturesIndex = pd.MultiIndex.from_frame(
        pd.concat(outputFeatureList).reset_index(drop=True))
    #
    scoresDF = pd.concat(cvScoresDict, names=['freqBandName'])
    scoresDF.dropna(axis='columns', inplace=True)
    #
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    # 
    try:
        scoresDF.loc[lastFoldIdx, :].to_hdf(estimatorPath, 'work')
        scoresDF.loc[:, ['test_score', 'train_score']].to_hdf(estimatorPath, 'cv')
    except Exception:
        traceback.print_exc()
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
    #
    features = chosenEstimator.transform(dataDF)
    #
    featuresDF = pd.DataFrame(
        features, index=dataDF.index,
        columns=pd.MultiIndex.from_frame(featureColumns))
    outputDatasetName = '{}_{}_{}_{}_{}'.format(
        arguments['unitQuery'], arguments['estimatorName'],
        arguments['iteratorSuffix'], arguments['window'], arguments['alignQuery'])
    outputDFPath = os.path.join(
        dataFramesFolder, outputDatasetName + '.h5'
        )
    #
    maskList = []
    haveAllGroup = False
    allGroupIdx = pd.MultiIndex.from_tuples(
        [tuple('all' for fgn in featureColumnFields)],
        names=featureColumnFields)
    allMask = pd.Series(True, index=featuresDF.columns).to_frame()
    allMask.columns = allGroupIdx
    maskList.append(allMask.T)
    if arguments['unitQuery'] == 'lfp_CAR_spectral':
        # each freq band
        for name, group in featuresDF.groupby('freqBandName', axis='columns'):
            attrValues = ['all' for fgn in featureColumnFields]
            attrValues[featureColumnFields.index('freqBandName')] = name
            thisMask = pd.Series(
                featuresDF.columns.isin(group.columns),
                index=featuresDF.columns).to_frame()
            if np.all(thisMask):
                haveAllGroup = True
                thisMask.columns = allGroupIdx
            else:
                thisMask.columns = pd.MultiIndex.from_tuples(
                    (attrValues, ), names=featureColumnFields)
            maskList.append(thisMask.T)
    # each lag    
    for name, group in featuresDF.groupby('lag', axis='columns'):
        attrValues = ['all' for fgn in featureColumnFields]
        attrValues[featureColumnFields.index('lag')] = name
        thisMask = pd.Series(
            featuresDF.columns.isin(group.columns),
            index=featuresDF.columns).to_frame()
        if not np.all(thisMask):
            # all group already covered
            thisMask.columns = pd.MultiIndex.from_tuples(
                (attrValues, ), names=featureColumnFields)
            maskList.append(thisMask.T)
    '''
    # each parent feature
    for name, group in featuresDF.groupby('parentFeature', axis='columns'):
        attrValues = ['all' for fgn in featureColumnFields]
        attrValues[featureColumnFields.index('parentFeature')] = name
        thisMask = pd.Series(
            featuresDF.columns.isin(group.columns),
            index=featuresDF.columns).to_frame()
        if np.all(thisMask):
            haveAllGroup = True
            thisMask.columns = allGroupIdx
        else:
            thisMask.columns = pd.MultiIndex.from_tuples(
                (attrValues, ), names=featureColumnFields)
        maskList.append(thisMask.T)'''
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
    #
    outputLoadingMeta = deepcopy(loadingMeta)
    outputLoadingMeta['arguments']['unitQuery'] = 'mahal'
    # these were already applied, no need to apply them again
    for k in ['decimate', 'procFun', 'addLags']:
        outputLoadingMeta['alignedAsigsKWargs'].pop(k, None)
    for k in ['normalizeDataset', 'unNormalizeDataset']:
        outputLoadingMeta.pop(k, None)
        #
        def passthr(df, params):
            return df
        #
        outputLoadingMeta[k] = passthr
    featuresDF.to_hdf(
        outputDFPath, outputDatasetName,
        mode='a')
    with open(outputDFPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(outputLoadingMeta, f)
