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
from dask.distributed import Client, LocalCluster
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
from sklearn.pipeline import make_pipeline, Pipeline
import dataAnalysis.helperFunctions.profiling as prf
# import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr_temp as tdr_temp
# from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
# import dataAnalysis.preproc.ns5 as ns5
import statsmodels.api as sm
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor, LinearRegression, enet_path
# from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
# from sklearn.svm import LinearSVR
# from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
import shutil
import time
# from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from dataAnalysis.custom_transformers.target_transformer import TransformedTargetRegressor
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'selector': None, 'transformerNameRhs': 'select', 'analysisName': 'hiRes', 'alignFolderName': 'motion',
        'selectionNameRhs': 'lfp_CAR_scaled', 'processAll': True, 'datasetNameRhs': 'Block_XL_df_ra',
        'datasetNameLhs': 'Block_XL_df_ra', 'selectionNameLhs': 'rig', 'verbose': '1',
        'transformerNameLhs': None, 'blockIdx': '2', 'winStop': '400', 'alignQuery': 'midPeak',
        'winStart': '200', 'debugging': True, 'window': 'long', 'showFigures': False,
        'estimatorName': 'enr3_select_scaled', 'lazy': False, 'plotting': True, 'exp': 'exp202101271100'
        }
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    
'''

exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetNameRhs'].split('_')[-1]))
from sklego.preprocessing import PatsyTransformer
# from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.linear_model._coordinate_descent import _alpha_grid
import joblib as jb
from copy import copy, deepcopy
import dill as pickle
import patsy
import sys
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
# from itertools import product
idxSl = pd.IndexSlice
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)


if __name__ == '__main__':
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    #
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
    designMatrixDatasetName = '{}_{}_{}_{}_regression_design_matrices'.format(
        arguments['datasetNameLhs'], arguments['selectionNameLhs'], arguments['selectionNameRhs'],
        arguments['transformerNameRhs'])
    designMatrixPath = os.path.join(
        dataFramesFolder,
        designMatrixDatasetName + '.h5'
        )
    #
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        iteratorOpts = loadingMeta['iteratorOpts']
    #
    # data loading meta
    estimatorMetadata['loadingMetaPath'] = loadingMetaPathLhs
    estimatorMetadata['designMatrixPath'] = designMatrixPath
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
        'refit': True, 'tol': 1e-4,
        'maxiter': 100, 'disp': False,
        'calc_frequency_weights': True
        }
    regressorClass = tdr_temp.SMWrapper
    regressorInstance = tdr_temp.SMWrapper(**regressorKWArgs)
    ## names of things in statsmodels
    l1_ratio_name, alpha_name = 'regressor__regressor__L1_wt', 'regressor__regressor__alpha'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid={l1_ratio_name: [.5]}
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
    lhsMasks = pd.read_hdf(designMatrixPath, '/featureMasks')
    lhsMasksInfo = pd.read_hdf(designMatrixPath, '/lhsMasksInfo')
    allTargetsDF = pd.read_hdf(designMatrixPath, 'allTargets')
    rhsMasks = pd.read_hdf(rhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameRhs']))
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
        estimatorMetadata['pipelinePathRhs'] = pipelinePathRhs
        pipelineMetaDataPathRhs = os.path.join(
            estimatorsSubFolder,
            pipelineNameRhs + '_meta.pickle'
        )
        workingScoresRhsDF = pd.read_hdf(pipelinePathRhs, 'work')
        workingPipelinesRhs = workingScoresRhsDF['estimator']
        with open(pipelineMetaDataPathRhs, 'rb') as _f:
            pipelineMetaRhs = pickle.load(_f)
    else:
        workingPipelinesRhs = None
    #
    if arguments['transformerNameLhs'] is not None:
        pass
    else:
        workingPipelinesLhs = None
    estimatorMetadata['arguments'] = arguments.copy()
    estimatorMetadata['lhsDatasetPath'] = lhsDatasetPath
    estimatorMetadata['rhsDatasetPath'] = rhsDatasetPath
    #
    allScores = []
    lhGroupNames = lhsMasks.index.names
    if 'backend' in joblibBackendArgs:
        if joblibBackendArgs['backend'] == 'dask':
            daskClient = Client()
    #
    # prep rhs dataframes
    rhsPipelineAveragerDict = {}
    histDesignInfoDict = {}
    #
    for rhsMaskIdx in range(rhsMasks.shape[0]):
        #
        prf.print_memory_usage('\n    On rhsRow {}\n'.format(rhsMaskIdx))
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        rhGroup = pd.read_hdf(designMatrixPath, 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
        # transform to PCs
        if workingPipelinesRhs is not None:
            transformPipelineRhs = workingPipelinesRhs.xs(rhsMaskParams['freqBandName'], level='freqBandName').iloc[0]
            rhsPipelineAveragerDict[rhsMaskIdx] = transformPipelineRhs.named_steps['averager']
        else:
            rhsPipelineAveragerDict[rhsMaskIdx] = Pipeline[('averager', tdr_temp.DataFramePassThrough(),)]
        #
        for ensTemplate in lOfHistTemplates:
            if ensTemplate != 'NULL':
                ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
                ensFormula += ' - 1'
                prf.print_memory_usage('Calculating history terms as {}'.format(ensFormula))
                ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
                exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
                ensPt.fit(exampleRhGroup)
                ensDesignMatrix = ensPt.transform(exampleRhGroup)
                ensDesignInfo = ensDesignMatrix.design_info
                histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
    ####################
    DEBUGGING = True
    ####################
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        slurmTaskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
    else:
        slurmTaskID = 0
    ####################
    if DEBUGGING:
        slurmTaskID = 4
    ####################
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    estimatorPathJoblib = estimatorPath.replace('.h5', '')
    if os.path.exists(estimatorPathJoblib):
        shutil.rmtree(estimatorPathJoblib)
    os.makedirs(estimatorPathJoblib)
    ####
    if os.getenv('SLURM_ARRAY_TASK_COUNT') is not None:
        slurmTaskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    else:
        slurmTaskCount = 1
    # if rerunning a subset of tasks, get the original task count from the file
    slurmTaskCount = processSlurmTaskCount
    ####################
    if DEBUGGING:
        slurmTaskCount = processSlurmTaskCount
    ####################
    print('slurmTaskCount = {}'.format(slurmTaskCount))
    ############
    slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / slurmTaskCount))
    if os.getenv('SLURM_ARRAY_TASK_MIN') is not None:
        slurmTaskMin = int(os.getenv('SLURM_ARRAY_TASK_MIN'))
    else:
        slurmTaskMin = 0
    ####################
    if DEBUGGING:
        slurmTaskMin = 0
    ####################
    if arguments['transformerNameRhs'] is not None:
        del transformedRhsDF, transformedRhsMasks
    #
    if slurmTaskID == slurmTaskMin:
        if os.path.exists(estimatorMetaDataPath):
            os.remove(estimatorMetaDataPath)
        with open(estimatorMetaDataPath, 'wb') as f:
            pickle.dump(estimatorMetadata, f)
    else:
        time.sleep(30)
    #
    #
    lhsDF = pd.read_hdf(designMatrixPath, 'lhsDF')
    exampleLhGroup = lhsDF.loc[lhsDF.index.get_level_values('conditionUID') == 0, :].copy()
    del lhsDF
    #
    cvScoresDict0 = {}
    gridSearcherDict0 = {}
    gsScoresDict0 = {}
    processedTargetIndices = []
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        if designFormula != 'NULL':
            pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
            # pdb.set_trace()
            designMatrixExample = pt.fit_transform(exampleLhGroup)
            designInfo = designMatrixExample.design_info
            theseColumns = designInfo.column_names
            #
            parentFormula = masterExogLookup[designFormula]
            parentFormulaIdx = masterExogFormulas.index(parentFormula)
            designDF = pd.read_hdf(designMatrixPath, 'designs/exogParents/formula_{}'.format(parentFormulaIdx))
            thisFeatureInfo = pd.read_hdf(designMatrixPath, 'designs/exogParents/term_lookup_{}'.format(parentFormulaIdx))
            # pdb.set_trace()
            designDF.columns = pd.MultiIndex.from_frame(thisFeatureInfo)
            designDF = designDF.loc[:, designDF.columns.get_level_values('factor').isin(theseColumns)]
            exogList = [designDF]
        else:
            exogList = []
        # add ensemble to designDF?
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        if (ensTemplate == 'NULL') and (selfTemplate == 'NULL') and (designFormula =='NULL'):
            continue
        #
        for rhsMaskIdx in range(rhsMasks.shape[0]):
            rhsPipelineAverager = rhsPipelineAveragerDict[rhsMaskIdx]
            ####
            pipelineRhs = Pipeline([('averager', rhsPipelineAverager, ), ])
            pipelineLhs = Pipeline([('averager', rhsPipelineAverager, ), ('regressor', regressorClass(**regressorKWArgs)), ])
            estimatorInstance = TransformedTargetRegressor(
                regressor=pipelineLhs, transformer=pipelineRhs, check_inverse=False)
            ###
            if ensTemplate != 'NULL':
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
            if selfTemplate != 'NULL':
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
            ###
            rhGroup = pd.read_hdf(designMatrixPath, 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            ###
            cvScoresDict1 = {}
            gridSearcherDict1 = {}
            gsScoresDict1 = {}
            #
            for targetName in rhGroup.columns:
                targetIdx = allTargetsDF.loc[(lhsMaskIdx, rhsMaskIdx, targetName), 'targetIdx']
                if (targetIdx // slurmGroupSize) != slurmTaskID:
                    print("targetIdx ({}) // slurmGroupSize = {}".format(targetIdx, targetIdx // slurmGroupSize))
                    print('slurmTaskID = {}'.format(slurmTaskID))
                    print('Skipping...')
                    continue
                else:
                    processedTargetIndices.append(targetIdx)
                print('\nslurmTaskID == {} targetIdx == {}\n'.format(slurmTaskID, targetIdx))
                prf.print_memory_usage('Fitting {} to {}...'.format(lhsMask.name[-1], targetName))
                ##
                targetDF = rhGroup.loc[:, [targetName]]
                # add targetDF to designDF?
                if ensTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(ensTemplate)
                    thisEnsDesign = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    thisEnsFeatureInfo = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/term_lookup_{}'.format(rhsMaskIdx, templateIdx))
                    thisEnsDesign.columns = pd.MultiIndex.from_frame(thisEnsFeatureInfo)
                    ensHistList = [
                        thisEnsDesign.iloc[:, sl].copy()
                        for key, sl in ensDesignInfo.term_name_slices.items()
                        if key != ensTemplate.format(targetName)]
                    # del thisEnsDesign
                else:
                    ensHistList = []
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
                    # del thisSelfDesign
                else:
                    selfHistList = []
                #
                fullDesignList = exogList + ensHistList + selfHistList
                fullDesignDF = pd.concat(fullDesignList, axis='columns')
                sortOrder = {cN: cIdx for cIdx, cN in enumerate(rhGroup.columns)}
                sortKey = lambda x: x.map(sortOrder)
                fullDesignDF.sort_index(
                    axis='columns', level='source',
                    key=sortKey, kind='mergesort',
                    sort_remaining=False, inplace=True)
                fullDesignDF.columns = fullDesignDF.columns.get_level_values('factor')
                # pdb.set_trace()
                del fullDesignList
                gsKWA = deepcopy(gridSearchKWArgs)
                if 'param_grid' in gridSearchKWArgs:
                    if l1_ratio_name in gsKWA['param_grid']:
                        dummyDesign = clone(pipelineLhs.named_steps['averager']).fit_transform(fullDesignDF)
                        dummyRhs = pd.DataFrame(
                            clone(pipelineRhs.named_steps['averager']).fit_transform(rhGroup),
                            columns=rhGroup.columns).loc[:, [targetName]]
                        paramGrid = gsKWA.pop('param_grid')
                        lOfL1Ratios = paramGrid.pop(l1_ratio_name)
                        gsParams = []
                        for l1Ratio in lOfL1Ratios:
                            alphas = _alpha_grid(
                                dummyDesign, dummyRhs,
                                fit_intercept=False, eps=1e-4,
                                l1_ratio=l1Ratio, n_alphas=nAlphas)
                            # enet_alphas, _, _ = enet_path(dummyDesign, dummyRhs, l1_ratio=l1Ratio, n_alphas=nAlphas, eps=1e-4)
                            alphas = np.atleast_1d(alphas).tolist()
                            fastTrack = True
                            if fastTrack:
                                alphas = [alphas[-1], alphas[0]]
                            gsParams.append(
                                {
                                    l1_ratio_name: [l1Ratio],
                                    alpha_name: alphas,
                                    'regressor__regressor__calc_frequency_weights': [True, False]
                                })
                            alphasStr = ['{:.3g}'.format(a) for a in alphas]
                            print('Evaluating alphas: {}'.format(alphasStr))
                        gsParams.append(
                            {
                                l1_ratio_name: [.5],
                                alpha_name: [0.],
                                'regressor__regressor__calc_frequency_weights': [True, False]
                            })
                        gsKWA['param_grid'] = gsParams
                #############
                #  estimatorInstance.fit(fullDesignDF, targetDF)
                ##############
                cvScores, gridSearcherDict1[targetName], gsScoresDF = tdr_temp.gridSearchHyperparameters(
                    fullDesignDF, targetDF,
                    estimatorInstance=estimatorInstance,
                    verbose=int(arguments['verbose']),
                    gridSearchKWArgs=gsKWA,
                    crossvalKWArgs=crossvalKWArgs,
                    joblibBackendArgs=joblibBackendArgs
                    )
                cvScoresDF = pd.DataFrame(cvScores)
                cvScoresDF.index.name = 'fold'
                cvScoresDF.dropna(axis='columns', inplace=True)
                #
                cvScoresDF.loc[:, ['test_score', 'train_score']].to_hdf(
                    estimatorPath,
                    'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
                        lhsMaskIdx, rhsMaskIdx, targetName
                        ))
                '''
                cvScoresDF['estimator'].to_hdf(
                    estimatorPath,
                    'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                        lhsMaskIdx, rhsMaskIdx, targetName
                        ))'''
                estimatorSaveName = 'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                    lhsMaskIdx, rhsMaskIdx, targetName
                    )
                estimatorSavePath = os.path.join(estimatorPathJoblib, estimatorSaveName)
                jb.dump(cvScoresDF['estimator'].to_dict(), estimatorSavePath)
                print('Saved {} ...'.format(estimatorSavePath))
                #
                if hasattr(cvScoresDF['estimator'].iloc[0].regressor_.named_steps['regressor'], 'n_iter_'):
                    nIters = cvScoresDF['estimator'].apply(lambda es: es.regressor_.named_steps['regressor'].n_iter_)
                    print('n iterations per estimator:\n{}'.format(nIters))
                    nIters.to_hdf(
                    estimatorPath,
                    'cv_estimators/lhsMask_{}/rhsMask_{}/{}/cv_estimators_n_iter'.format(
                        lhsMaskIdx, rhsMaskIdx, targetName
                        ))
                #
                gsScoresStack = pd.concat({
                    'test': gsScoresDF['test_score'],
                    'train': gsScoresDF['train_score']},
                    names=['foldType']
                    ).to_frame(name='score').reset_index()
                #
                lastFoldMask = (gsScoresStack['fold'] == cvIterator.n_splits)
                trainMask = (gsScoresStack['foldType'] == 'train')
                testMask = (gsScoresStack['foldType'] == 'test')
                #
                gsScoresStack.loc[:, 'trialType'] = ''
                gsScoresStack.loc[(trainMask & lastFoldMask), 'trialType'] = 'work'
                gsScoresStack.loc[(trainMask & (~lastFoldMask)), 'trialType'] = 'train'
                gsScoresStack.loc[(testMask & lastFoldMask), 'trialType'] = 'validation'
                gsScoresStack.loc[(testMask & (~lastFoldMask)), 'trialType'] = 'test'
                gsScoresStack.loc[:, 'lhsMaskIdx'] = lhsMaskIdx
                gsScoresStack.loc[:, 'rhsMaskIdx'] = rhsMaskIdx
                gsScoresStack.loc[:, 'dummyX'] = 0
                gsScoresStack.loc[:, 'design'] = gsScoresStack['lhsMaskIdx'].apply(
                    lambda x: lhsMasksInfo.loc[x, 'designFormula'])
                gsScoresStack.loc[:, 'designAsLabel'] = gsScoresStack['design'].apply(
                    lambda x: x.replace(' + ', ' +\n'))
                gsScoresStack.loc[:, 'fullDesign'] = gsScoresStack['lhsMaskIdx'].apply(
                    lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
                gsScoresStack.loc[:, 'fullDesignAsLabel'] = gsScoresStack['fullDesign'].apply(
                    lambda x: x.replace(' + ', ' +\n'))
                gsScoresStack.to_hdf(
                    estimatorPath,
                    'gs_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                        lhsMaskIdx, rhsMaskIdx, targetName
                        ))
                prf.print_memory_usage('\n\nCompleted fit {} to {}...\n\n'.format(lhsMask.name[-1], targetName))
print('Processed these targets: {}'.format(processedTargetIndices))
prf.print_memory_usage('All fits complete.')
