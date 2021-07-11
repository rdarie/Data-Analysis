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

import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('PS')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
#
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dask.distributed import Client, LocalCluster
import os, traceback
from dataAnalysis.analysis_code.regression_parameters import *
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
from docopt import docopt
from copy import deepcopy
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
for arg in sys.argv:
    print(arg)
##
# pdb.set_trace()
if __name__ == '__main__':
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    ##
    '''
    
    consoleDebugging = True
    if consoleDebugging:
        arguments = {
            'showFigures': False, 'estimatorName': 'regressor', 'datasetName': 'Synthetic_XL_df_g',
            'exp': 'exp202101281100', 'averageByTrial': False, 'verbose': '1', 'lazy': False,
            'analysisName': 'hiRes', 'blockIdx': '2', 'alignFolderName': 'motion', 'debugging': True,
            'plotting': True, 'window': 'long', 'selectionName': 'rig', 'processAll': True}
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
            figureFolder, arguments['analysisName'], 'regression')
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
        iteratorOpts = loadingMeta['iteratorOpts']
    for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
        loadingMeta['arguments'].pop(argName, None)
    arguments.update(loadingMeta['arguments'])
    #
    cvIterator = iteratorsBySegment[0]
    #
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    #
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    prf.print_memory_usage('just loaded data, fitting')
    #
    # featuresDF = ...
    transformersLookup = {
        # 'forceMagnitude': MinMaxScaler(feature_range=(0., 1)),
        # 'forceMagnitude_prime': MinMaxScaler(feature_range=(-1., 1)),
        'amplitude': MinMaxScaler(feature_range=(0., 1)),
        'RateInHz': MinMaxScaler(feature_range=(0., .5)),
        'velocity': MinMaxScaler(feature_range=(-1., 1.)),
        'velocity_abs': MinMaxScaler(feature_range=(0., 1.)),
        }
    lOfTransformers = []
    for cN in dataDF.columns:
        if cN[0] not in transformersLookup:
            lOfTransformers.append(([cN], None,))
        else:
            lOfTransformers.append(([cN], transformersLookup[cN[0]],))
    lhsScaler = DataFrameMapper(lOfTransformers, input_df=True, )
    lhsScaler.fit(dataDF)
    featuresDF = pd.DataFrame(
        lhsScaler.transform(dataDF), index=dataDF.index, columns=dataDF.columns)
    # pdb.set_trace()
    # plt.plot(featuresDF.xs('velocity_abs', axis='columns', level='feature').to_numpy())
    regressorsFromMetadata = ['electrode']
    columnAdder = tdr.DataFrameMetaDataToColumns(addColumns=regressorsFromMetadata)
    featuresDF = columnAdder.fit_transform(featuresDF)
    #
    estimatorMetadata = {
        'path': os.path.basename(estimatorPath),
        'name': estimatorName,
        'datasetName': datasetName,
        'selectionName': selectionName,
        'outputFeatures': featuresDF.columns
        }
    with open(estimatorMetaDataPath, 'wb') as _f:
        pickle.dump(estimatorMetadata, _f)
    #
    outputSelectionName = '{}_{}'.format(
        selectionName, estimatorName)
    outputLoadingMeta = deepcopy(loadingMeta)
    #
    # 'decimate', 'procFun', 'addLags' were already applied, no need to apply them again
    for k in ['decimate', 'procFun', 'addLags', 'rollingWindow']:
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
    keepMask = featuresDF.columns.get_level_values('feature').isin(regressionColumnsToUse)
    featuresDF = featuresDF.loc[:, keepMask]
    featuresDF.rename(columns=regressionColumnRenamer, level='feature', inplace=True)
    #
    featuresDF.to_hdf(
        datasetPath,
        '/{}/data'.format(outputSelectionName),
        mode='a')
    #####################################################################################################################
    ###
    '''
    from ttictoc import tic, toc
    featuresDF.columns = featuresDF.columns.get_level_values('feature')
    # get one of each condition, to fit the patsy transformer on
    # (it needs to see all of the categoricals)
    exampleFeaturesDF = featuresDF.loc[featuresDF.index.get_level_values('conditionUID') == 0, :]
    # featuresDF = featuresDF.iloc[:800, :]
    #
    for hIdx, histOpts in enumerate(addHistoryTerms):
        locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    # hto1['logBasis'] = False
    # hto1['addInputToOutput'] = True
    # hto1['preprocFun'] = lambda x: x.diff().fillna(0)
    # hto1['selectColumns'] = slice(2, 4)
    # hto1['causal'] = False
    raisedCosBaser = tdr.raisedCosTransformer(hto0)
    #
    if arguments['plotting']:
        pdfPath = os.path.join(
            figureOutputFolder, 'history_basis.pdf'
            )
        fig, ax = raisedCosBaser.plot_basis()
        plt.savefig(pdfPath)
        plt.savefig(pdfPath.replace('.pdf', '.png'))
        if arguments['debugging']:
            plt.show()
        else:
            plt.close()
    #####
    rcb = tdr.patsyRaisedCosTransformer
    thisEnv = patsy.EvalEnvironment.capture()
    timingInfo = {df: {} for df in lOfDesignFormulas}
    for designFormula in lOfDesignFormulas:
        print(designFormula)
        tic()
        pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
        # train on example
        pt.fit(exampleFeaturesDF)
        # transform features
        designMatrix = pt.transform(exampleFeaturesDF)
        timingInfo[designFormula]['elapsed'] = toc()
        timingInfo[designFormula]['designMatrix'] = designMatrix
        print('Elapsed time: {}'.format(timingInfo[designFormula]['elapsed']))
        ##
    for designFormula in lOfDesignFormulas:
        print('\n' * 5)
        designMatrix = timingInfo[designFormula]['designMatrix']
        designInfo = designMatrix.design_info
        print(designInfo.describe())
        print(designMatrix.shape)
        print('\n'.join(designInfo.column_names))
        designDF = (
            pd.DataFrame(
                designMatrix,
                index=exampleFeaturesDF.index,
                columns=designInfo.column_names))
        fig, ax = plt.subplots(2, 1, sharex=True)
        for cN in ['v', 'a', 'r']:
            ax[0].plot(exampleFeaturesDF[cN].to_numpy(), label='input {}'.format(cN))
        ax[0].legend()
        for cN in designDF.columns:
            ax[1].plot(designDF[cN].to_numpy(), label=cN)
        ax[1].legend()
        plt.show()
        '''
    #####################################################################################################################
    #
    maskList = []
    attrNames = ['feature', 'lag', 'designFormula', 'ensembleTemplate', 'selfTemplate']
    for designFormula in lOfDesignFormulas:
        for ensembleTemplate in lOfEnsembleTemplates:
            for selfTemplate in lOfSelfTemplates:
                attrValues = ['all', 0, designFormula, ensembleTemplate, selfTemplate]
                thisMask = pd.Series(
                    True,
                    index=featuresDF.columns).to_frame()
                thisMask.columns = pd.MultiIndex.from_tuples(
                    (attrValues, ), names=attrNames)
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
    # pdb.set_trace()
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
