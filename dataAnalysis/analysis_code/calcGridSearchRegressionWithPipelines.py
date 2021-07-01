"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                               process entire experimental day? [default: False]
    --analysisName=analysisName                append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName          append a name to the resulting blocks? [default: motion]
    --window=window                            process with short window? [default: long]
    --winStart=winStart                        start of window [default: 200]
    --winStop=winStop                          end of window [default: 400]
    --lazy                                     load from raw, or regular? [default: False]
    --plotting                                 make plots? [default: False]
    --debugging                                restrict datasets for debugging? [default: False]
    --showFigures                              show plots? [default: False]
    --verbose=verbose                          print diagnostics? [default: 0]
    --alignQuery=alignQuery                    what will the plot be aligned to? [default: midPeak]
    --datasetNameRhs=datasetNameRhs            which trig_ block to pull [default: Block]
    --selectionNameRhs=selectionNameRhs        how to restrict channels? [default: fr_sqrt]
    --transformerNameRhs=transformerNameRhs    how to restrict channels?
    --datasetNameLhs=datasetNameLhs            which trig_ block to pull [default: Block]
    --selectionNameLhs=selectionNameLhs        how to restrict channels? [default: fr_sqrt]
    --transformerNameLhs=transformerNameLhs    how to restrict channels?
    --estimatorName=estimatorName              filename for resulting estimator (cross-validated n_comps)
    --selector=selector                        filename if using a unit selector
"""

import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('PS')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
#
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
from sklearn.pipeline import make_pipeline, Pipeline
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
import statsmodels.api as sm
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor, LinearRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from dataAnalysis.custom_transformers.target_transformer import TransformedTargetRegressor
from dataAnalysis.analysis_code.regression_parameters import *
from sklego.preprocessing import PatsyTransformer
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.linear_model._coordinate_descent import _alpha_grid
import joblib as jb
from copy import copy, deepcopy
import dill as pickle
import patsy
import sys
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
from itertools import product
idxSl = pd.IndexSlice
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)

for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'selectionNameLhs': 'rig_regressor', 'selectionNameRhs': 'lfp_CAR', 'verbose': '2',
        'datasetNameRhs': 'Synthetic_XL_df_g', 'transformerNameRhs': 'pca_ta', 'selector': None,
        'debugging': True, 'processAll': True, 'winStop': '400', 'showFigures': True, 'window': 'long',
        'lazy': False, 'datasetNameLhs': 'Synthetic_XL_df_g', 'alignFolderName': 'motion',
        'plotting': True, 'estimatorName': 'enr', 'transformerNameLhs': None, 'blockIdx': '2',
        'analysisName': 'hiRes', 'alignQuery': 'midPeak', 'winStart': '200', 'exp': 'exp202101281100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    
'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


if __name__ == '__main__':
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], 'regression')
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    #
    rhsDatasetPath = os.path.join(
        dataFramesFolder,
        arguments['datasetNameRhs'] + '.h5'
        )
    assert os.path.exists(rhsDatasetPath)
    lhsDatasetPath = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '.h5'
        )
    assert os.path.exists(lhsDatasetPath)
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], arguments['datasetNameLhs'])
    #
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    cvEstimatorsPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_cv_estimators.pickle'
        )
    #
    estimatorMetaDataPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_meta.pickle'
        )
    estimatorMetadata = {}
    loadingMetaPathLhs = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '_' + arguments['selectionNameLhs'] + '_meta.pickle'
        )
    #
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        iteratorOpts = loadingMeta['iteratorOpts']
    #
    # data loading meta
    estimatorMetadata['loadingMetaPath'] = loadingMetaPathLhs
    #
    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    #
    joblibBackendArgs = dict(
        # backend='dask',
        backend='loky',
        #### n_jobs=1
    )
    #
    estimatorMetadata['joblibBackendArgs'] = joblibBackendArgs
    #
    if joblibBackendArgs['backend'] == 'dask':
        daskComputeOpts = dict(
            scheduler='processes'
            # scheduler='single-threaded'
        )
        if daskComputeOpts['scheduler'] == 'single-threaded':
            daskClient = Client(LocalCluster(n_workers=1))
        elif daskComputeOpts['scheduler'] == 'processes':
            daskClient = Client(LocalCluster(processes=True))
        elif daskComputeOpts['scheduler'] == 'threads':
            daskClient = Client(LocalCluster(processes=False))
        else:
            print('Scheduler name is not correct!')
            daskClient = Client()
    #
    crossvalKWArgs = dict(
        return_train_score=True, return_estimator=True)
    estimatorMetadata['crossvalKWArgs'] = crossvalKWArgs
    ###
    nAlphas = 10
    ###
    ## statsmodels elasticnet
    regressorKWArgs = {
        'sm_class': sm.GLM,
        'family': sm.families.Gaussian(),
        'alpha': 1e-12, 'L1_wt': .1,
        'refit': True, 'tol': 1e-2,
        'maxiter': 1000, 'disp': False
        }
    regressorClass =  tdr.SMWrapper
    regressorInstance = tdr.SMWrapper(**regressorKWArgs)
    ## names of things in statsmodels
    l1_ratio_name, alpha_name = 'regressor__regressor__L1_wt', 'regressor__regressor__alpha'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid={l1_ratio_name: [.1, .5, .95]}
        )
    #
    '''regressorClass = ElasticNet
    regressorKWArgs = dict(fit_intercept=False, l1_ratio=0.5, alpha=1e-12)
    # names of things in sklearn
    l1_ratio_name, alpha_name = 'regressor__regressor__l1_ratio', 'regressor__regressor__alpha'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid={l1_ratio_name: [.1, .5, .95]}
        )'''
    #
    '''regressorClass = LinearRegression
    regressorKWArgs = dict(fit_intercept=False)
    # names of things in sklearn
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False
        )'''
    for hIdx, histOpts in enumerate(addHistoryTerms):
        locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    thisEnv = patsy.EvalEnvironment.capture()
    #
    crossvalKWArgs['cv'] = cvIterator
    gridSearchKWArgs['cv'] = cvIterator
    #
    estimatorMetadata['regressorClass'] = regressorClass
    estimatorMetadata['regressorInstance'] = regressorInstance
    estimatorMetadata['regressorKWArgs'] = regressorKWArgs
    estimatorMetadata['gridSearchKWArgs'] = gridSearchKWArgs
    estimatorMetadata['nAlphas'] = nAlphas
    #
    lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
    lhsMasks = pd.read_hdf(lhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameLhs']))
    rhsMasks = pd.read_hdf(rhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameRhs']))
    #
    estimatorMetadata['arguments'] = arguments.copy()
    estimatorMetadata['lhsDatasetPath'] = lhsDatasetPath
    estimatorMetadata['rhsDatasetPath'] = rhsDatasetPath
    #
    if arguments['transformerNameRhs'] is not None:
        transformedSelectionNameRhs = '{}_{}'.format(
            arguments['selectionNameRhs'], arguments['transformerNameRhs'])
        transformedRhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(transformedSelectionNameRhs))
        transformedRhsMasks = pd.read_hdf(rhsDatasetPath, '/{}/featureMasks'.format(transformedSelectionNameRhs))
        #
        # get rid of rhsMasks that don't output single features
        transformedRhsMasks = transformedRhsMasks.loc[transformedRhsMasks.sum(axis=1).to_numpy() == 1, :]
        #
        pipelineNameRhs = '{}_{}_{}'.format(
            arguments['transformerNameRhs'], arguments['datasetNameRhs'], arguments['selectionNameRhs'])
        pipelinePathRhs = os.path.join(
            estimatorsSubFolder,
            pipelineNameRhs + '.h5'
            )
        pipelineMetaDataPathRhs = os.path.join(
            estimatorsSubFolder,
            pipelineNameRhs + '_meta.pickle'
            )
        workingScoresRhsDF = pd.read_hdf(pipelinePathRhs, 'work')
        workingPipelinesRhs = workingScoresRhsDF['estimator']
        with open(pipelineMetaDataPathRhs, 'rb') as _f:
            pipelineMetaRhs = pickle.load(_f)
        estimatorMetadata['pipelineMetaDataPathRhs'] = pipelineMetaDataPathRhs
        estimatorMetadata['pipelinePathRhs'] = pipelinePathRhs
    else:
        workingPipelinesRhs = None
        estimatorMetadata['pipelineMetaDataPathRhs'] = None
        estimatorMetadata['pipelinePathRhs'] = None
    #
    if arguments['transformerNameLhs'] is not None:
        pass
    else:
        workingPipelinesLhs = None
        estimatorMetadata['pipelineMetaDataPathLhs'] = None
        estimatorMetadata['pipelinePathLhs'] = None
    #
    trialInfoLhs = lhsDF.index.to_frame().reset_index(drop=True)
    trialInfoRhs = rhsDF.index.to_frame().reset_index(drop=True)
    checkSameMeta = stimulusConditionNames + ['bin', 'trialUID', 'conditionUID']
    assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
    trialInfo = trialInfoLhs
    #
    lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
    rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
    #
    '''# fill zeros, e.g. if trials do not have measured position, positions will be NaN
    lhsDF.fillna(0, inplace=True)'''
    # moved to assembleDataFrames
    #
    '''if ('amplitude' in lhsDF.columns.get_level_values('feature')) and ('RateInHz' in lhsDF.columns.get_level_values('feature')):
        lhsDF.loc[:, idxSl['RateInHz', :, :, :, :]] = lhsDF.loc[:, idxSl['RateInHz', :, :, :, :]] * (lhsDF.loc[:, idxSl['amplitude', :, :, :, :]].abs() > 0).to_numpy(dtype=float)'''
    # moved to analysis maker
    #
    '''lOfDesignFormulas = ["velocity + electrode:(amplitude/RateInHz)"]
    estimatorMetadata['lOfDesignFormulas'] = lOfDesignFormulas
    transformersLookup = {
        # 'forceMagnitude': MinMaxScaler(feature_range=(0., 1)),
        # 'forceMagnitude_prime': MinMaxScaler(feature_range=(-1., 1)),
        'amplitude': MinMaxScaler(feature_range=(0., 1)),
        'RateInHz': MinMaxScaler(feature_range=(0., .5)),
        'velocity': MinMaxScaler(feature_range=(-1., 1.)),
        }
    lOfTransformers = []
    for cN in lhsDF.columns:
        if cN[0] not in transformersLookup:
            lOfTransformers.append(([cN], None,))
        else:
            lOfTransformers.append(([cN], transformersLookup[cN[0]],))
    lhsScaler = DataFrameMapper(lOfTransformers, input_df=True,)
    lhsDF = pd.DataFrame(
        lhsScaler.fit_transform(lhsDF), index=lhsDF.index, columns=lhsDF.columns)
    #
    estimatorMetadata['lhsScaler'] = lhsScaler
    #
    lhsMasks.iloc[0, :] = lhsMasks.columns.get_level_values('feature').isin(transformersLookup.keys())
    regressorsFromMetadata = ['electrode']
    #
    estimatorMetadata['regressorsFromMetadata'] = regressorsFromMetadata'''
    # moved to prep regressor
    #
    '''
    workIdx = cvIterator.work
    workingLhsDF = lhsDF.iloc[workIdx, :]
    workingRhsDF = rhsDF.iloc[workIdx, :]
    nFeatures = lhsDF.columns.shape[0]
    nTargets = rhsDF.columns.shape[0]
    '''
    #
    allScores = []
    lhGroupNames = lhsMasks.index.names
    if 'backend' in joblibBackendArgs:
        if joblibBackendArgs['backend'] == 'dask':
            daskClient = Client()
    #
    cvScoresDict0 = {}
    gridSearcherDict0 = {}
    gsScoresDict0 = {}
    #
    figureOutputPath = os.path.join(
        figureOutputFolder,
        '{}_signals.pdf'.format(fullEstimatorName))
    for lhsRowIdx, rhsRowIdx in product(list(range(lhsMasks.shape[0])), list(range(rhsMasks.shape[0]))):
        lhsMask = lhsMasks.iloc[lhsRowIdx, :]
        rhsMask = rhsMasks.iloc[rhsRowIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMask.index.names, lhsMask.name)}
        rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
        lhGroup = lhsDF.loc[:, lhsMask]
        #
        lhGroup.columns = lhGroup.columns.get_level_values('feature')
        designFormula = lhsMask.name[lhsMasks.index.names.index('designFormula')]
        #
        print(designFormula)
        pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
        exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
        pt.fit(exampleLhGroup)
        designMatrix = pt.transform(lhGroup)
        designInfo = designMatrix.design_info
        designDF = (
            pd.DataFrame(
                designMatrix,
                index=lhGroup.index,
                columns=designInfo.column_names))
        designDF.columns.name = 'feature'
        rhGroup = rhsDF.loc[:, rhsMask].copy()
        # transform to PCs
        if workingPipelinesRhs is not None:
            transformPipelineRhs = workingPipelinesRhs.xs(rhsMaskParams['freqBandName'], level='freqBandName').iloc[0]
            rhsPipelineMinusAverager = Pipeline(transformPipelineRhs.steps[1:])
            rhsPipelineAverager = transformPipelineRhs.named_steps['averager']
            rhTransformedColumns = transformedRhsDF.columns[transformedRhsDF.columns.get_level_values('freqBandName') == rhsMaskParams['freqBandName']]
            rhGroup = pd.DataFrame(
                rhsPipelineMinusAverager.transform(rhGroup),
                index=lhGroup.index, columns=rhTransformedColumns)
        else:
            rhsPipelineAverager = Pipeline[('averager', tdr.DataFramePassThrough(), )]
        rhGroup.columns = rhGroup.columns.get_level_values('feature')
        ####
        # if arguments['debugging']:
        #     if rhsMaskParams['freqBandName'] not in ['beta', 'gamma', 'higamma', 'all']:
        #         # if maskParams['lag'] not in [0]:
        #         continue
        pipelineRhs = Pipeline([('averager', rhsPipelineAverager, ), ])
        pipelineLhs = Pipeline([('averager', rhsPipelineAverager, ), ('regressor', regressorClass(**regressorKWArgs)), ])
        estimatorInstance = TransformedTargetRegressor(regressor=pipelineLhs, transformer=pipelineRhs, check_inverse=False)
        gsParamsPerTarget = None
        if 'param_grid' in gridSearchKWArgs:
            gsKWA = deepcopy(gridSearchKWArgs)
            if l1_ratio_name in gsKWA['param_grid']:
                dummyDesign = clone(pipelineLhs.named_steps['averager']).fit_transform(designDF)
                dummyRhs = pd.DataFrame(
                    clone(pipelineRhs.named_steps['averager']).fit_transform(rhGroup),
                    columns=rhGroup.columns)
                paramGrid = gsKWA.pop('param_grid')
                lOfL1Ratios = paramGrid.pop(l1_ratio_name)
                gsParamsPerTarget = {}
                for targetName in rhGroup.columns:
                    gsParamsPerTarget[targetName] = []
                    for l1Ratio in lOfL1Ratios:
                        alphas = _alpha_grid(
                            dummyDesign, dummyRhs.loc[:, [targetName]],
                            l1_ratio=l1Ratio, n_alphas=nAlphas)
                        gsParamsPerTarget[targetName].append(
                            {
                                l1_ratio_name: [l1Ratio],
                                # alpha_name: [1e-12]
                                alpha_name: np.atleast_1d(alphas).tolist()
                            }
                        )
        cvScoresDict1 = {}
        gridSearcherDict1 = {}
        gsScoresDict1 = {}
        for targetName in rhGroup.columns:
            ###########
            ###########
            if targetName not in rhGroup.columns[:2]:
                continue
            ###########
            ###########
            print('Fitting {} to {}...'.format(lhsMask.name[-1], targetName))
            if gsParamsPerTarget is not None:
                gsKWA['param_grid'] = gsParamsPerTarget[targetName]
            ##
            targetDF = rhGroup.loc[:, [targetName]]
            cvScores, gridSearcherDict1[targetName], gsScoresDict1[targetName] = tdr.gridSearchHyperparameters(
                designDF, rhGroup.loc[:, [targetName]],
                estimatorInstance=estimatorInstance,
                verbose=int(arguments['verbose']),
                gridSearchKWArgs=gsKWA,
                crossvalKWArgs=crossvalKWArgs,
                joblibBackendArgs=joblibBackendArgs
                )
            cvScoresDF = pd.DataFrame(cvScores)
            cvScoresDF.index.name = 'fold'
            cvScoresDF.dropna(axis='columns', inplace=True)
            cvScoresDict1[targetName] = cvScoresDF
            gridSearcherDict0[(lhsRowIdx, rhsRowIdx)] = gridSearcherDict1
            # pdb.set_trace()
        cvScoresDict0[(lhsRowIdx, rhsRowIdx)] = pd.concat(cvScoresDict1, names=['target'])
        gsScoresDict0[(lhsRowIdx, rhsRowIdx)] = pd.concat(gsScoresDict1, names=['target'])
    #
    allCVScores = pd.concat(cvScoresDict0, names=['lhsMaskIdx', 'rhsMaskIdx'])
    allGSScores = pd.concat(gsScoresDict0, names=['lhsMaskIdx', 'rhsMaskIdx'])
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    lastFoldIdx = cvIterator.n_splits
    print('\n\nSaving {}\n\n'.format(estimatorPath))
    try:
        allCVScores.loc[idxSl[:, :, :, lastFoldIdx], :].to_hdf(estimatorPath, 'work_scores_estimators')
        allCVScores.loc[:, ['test_score', 'train_score']].to_hdf(estimatorPath, 'cv_scores')
    except Exception:
        traceback.print_exc()
        pdb.set_trace()
    try:
        # with open(cvEstimatorsPath, 'wb') as _f:
        #     pickle.dump(allCVScores['estimator'].to_dict(), _f)
        allCVScores['estimator'].to_hdf(estimatorPath, 'cv_estimators')
    except Exception:
        traceback.print_exc()
        pdb.set_trace()
    #
    if os.path.exists(estimatorMetaDataPath):
        os.remove(estimatorMetaDataPath)
    with open(estimatorMetaDataPath, 'wb') as f:
        pickle.dump(estimatorMetadata, f)
    if arguments['plotting']:
        figureOutputPath = os.path.join(
                figureOutputFolder,
                '{}_r2.pdf'.format(fullEstimatorName))
        scoresForPlot = pd.concat(
            {'test': allCVScores['test_score'], 'train': allCVScores['train_score']},
            names=['evalType']).to_frame(name='score').reset_index()
        lastFoldIdx = cvIterator.n_splits
        validationMask = (
            (scoresForPlot['fold'] == lastFoldIdx) &
            (scoresForPlot['evalType'] == 'test'))
        scoresForPlot.loc[validationMask, 'evalType'] = 'validation'
        workingMask = (
            (scoresForPlot['fold'] == lastFoldIdx) &
            (scoresForPlot['evalType'] == 'train'))
        scoresForPlot.loc[workingMask, 'evalType'] = 'work'
        colWrap = np.ceil(np.sqrt(scoresForPlot['lhsMaskIdx'].unique().size)).astype(int)
        with PdfPages(figureOutputPath) as pdf:
            # fig, ax = plt.subplots()
            # fig.set_size_inches(12, 8)
            g = sns.catplot(
                data=scoresForPlot, hue='evalType',
                col='lhsMaskIdx', col_wrap=colWrap,
                x='target', y='score',
                kind='box')
            g.fig.suptitle('R^2')
            newYLims = scoresForPlot['score'].quantile([0.25, 1 - 1e-3]).to_list()
            for ax in g.axes.flat:
                ax.set_xlabel('regression target')
                ax.set_ylabel('R2 of ordinary least squares fit')
                ax.set_ylim(newYLims)
            g.fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    del lhsDF, rhsDF
    # gc.collect()
