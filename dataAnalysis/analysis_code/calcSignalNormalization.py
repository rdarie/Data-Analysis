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
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
"""

import logging
logging.captureWarnings(True)
import matplotlib, os, sys
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from docopt import docopt
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''

arguments = {
    'analysisName': 'hiRes', 'processAll': True, 'selectionName': 'lfp_CAR_spectral', 'datasetName': 'Block_XL_df_rd',
    'window': 'long', 'estimatorName': 'scaled', 'verbose': 2, 'exp': 'exp202101271100',
    'alignFolderName': 'motion', 'showFigures': False, 'blockIdx': '2', 'debugging': False,
    'plotting': True, 'lazy': False}
os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')

'''

idxSl = pd.IndexSlice
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
        figureFolder, arguments['analysisName'])
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
    fullEstimatorName + '.joblib'
    )
#
with open(loadingMetaPath, 'rb') as _f:
    loadingMeta = pickle.load(_f)
    # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
    iteratorsBySegment = loadingMeta['iteratorsBySegment']
    # cv_kwargs = loadingMeta['cv_kwargs']
    if 'normalizeDataset' in loadingMeta:
        normalizeDataset = loadingMeta['normalizeDataset']
        unNormalizeDataset = loadingMeta['unNormalizeDataset']
        normalizationParams = loadingMeta['normalizationParams']
    else:
        normalizeDataset = None
for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
    loadingMeta['arguments'].pop(argName, None)
arguments.update(loadingMeta['arguments'])


if __name__ == '__main__':
    cvIterator = iteratorsBySegment[0]
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    # only use zero lag targets
    lagMask = dataDF.columns.get_level_values('lag') == 0
    dataDF = dataDF.loc[:, lagMask]
    featureMasks = featureMasks.loc[:, lagMask]
    #
    freqBands = featureMasks.index.get_level_values('freqBandName').unique()
    if freqBands.size > 1:
        # remove the all mask
        freqBandMask = ~(featureMasks.index.get_level_values('freqBandName') == 'all')
        featureMasks = featureMasks.loc[freqBandMask, :]
    #
    lOfColumnTransformers = []
    outputFeatureNameList = []
    # pdb.set_trace()
    for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
        maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
        dataGroup = dataDF.loc[:, featureMask]
        print('dataGroup.shape = {}'.format(dataGroup.shape))
        trfName = '{}_{}'.format(estimatorName, maskParams['freqBandName'])
        estimator = tdr.flatStandardScaler()
        lOfColumnTransformers.append((
            # transformer name
            trfName,
            # estimator
            estimator,
            # columns
            dataGroup.columns.copy()
            ))
        outputFeatureNameList.append(dataGroup.columns.to_frame().reset_index(drop=True))
    #
    chosenEstimator = ColumnTransformer(lOfColumnTransformers)
    chosenEstimator.fit(dataDF)
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    #
    jb.dump(chosenEstimator, estimatorPath)
    #
    outputFeatureNames = pd.concat(outputFeatureNameList)
    outputFeatureNames.loc[:, 'feature'] = outputFeatureNames['feature'].apply(lambda x: '{}#0'.format(x))
    outputFeatureIndex = pd.MultiIndex.from_frame(outputFeatureNames)
    features = chosenEstimator.transform(dataDF)
    featuresDF = pd.DataFrame(
        features,
        index=dataDF.index, columns=outputFeatureIndex)
    estimatorMetadata = {
        'path': os.path.basename(estimatorPath),
        'name': estimatorName,
        'datasetName': datasetName,
        'selectionName': selectionName,
        'outputFeatures': featuresDF.columns
        }
    with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
        pickle.dump(estimatorMetadata, f)
    outputSelectionName = '{}_{}'.format(
        selectionName, estimatorName)
    # pdb.set_trace()
    featuresDF.sort_index(
        axis='columns', inplace=True,
        level=['feature', 'lag'],
        kind='mergesort', sort_remaining=False)
    ##
    maskList = []
    haveAllGroup = False
    allGroupIdx = pd.MultiIndex.from_tuples(
        [tuple('all' for fgn in featuresDF.columns.names)],
        names=featuresDF.columns.names)
    allMask = pd.Series(True, index=featuresDF.columns).to_frame()
    allMask.columns = allGroupIdx
    maskList.append(allMask.T)
    if 'lfp_CAR_spectral' in selectionName:
        # each freq band
        for name, group in featuresDF.groupby('freqBandName', axis='columns'):
            attrValues = ['all' for fgn in featuresDF.columns.names]
            attrValues[featuresDF.columns.names.index('freqBandName')] = name
            thisMask = pd.Series(
                featuresDF.columns.isin(group.columns),
                index=featuresDF.columns).to_frame()
            if np.all(thisMask):
                haveAllGroup = True
                thisMask.columns = allGroupIdx
            else:
                thisMask.columns = pd.MultiIndex.from_tuples(
                    (attrValues, ), names=featuresDF.columns.names)
            maskList.append(thisMask.T)
    # each lag    
    for name, group in featuresDF.groupby('lag', axis='columns'):
        attrValues = ['all' for fgn in featuresDF.columns.names]
        attrValues[featuresDF.columns.names.index('lag')] = name
        thisMask = pd.Series(
            featuresDF.columns.isin(group.columns),
            index=featuresDF.columns).to_frame()
        if not np.all(thisMask):
            # all group already covered
            thisMask.columns = pd.MultiIndex.from_tuples(
                (attrValues, ), names=featuresDF.columns.names)
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
    #
    outputLoadingMeta = deepcopy(loadingMeta)
    outputLoadingMeta['arguments']['unitQuery'] = 'lfp'
    outputLoadingMeta['arguments']['selectionName'] = outputSelectionName
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
        datasetPath,
        '/{}/data'.format(outputSelectionName),
        mode='a')
    outputLoadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(outputSelectionName) + '_meta.pickle'
        )
    with open(outputLoadingMetaPath, 'wb') as f:
        pickle.dump(outputLoadingMeta, f)
