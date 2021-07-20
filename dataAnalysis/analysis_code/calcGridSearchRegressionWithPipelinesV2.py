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
        'selector': None, 'debugging': True, 'processAll': True, 'winStop': '400', 'showFigures': True, 'window': 'long',
        'lazy': False, 'alignFolderName': 'motion', 'plotting': True,
        'estimatorName': 'enr3_ta',
        'datasetNameRhs': 'Synthetic_XL_df_g', 'transformerNameRhs': 'pca_ta',
        'datasetNameLhs': 'Synthetic_XL_df_g', 'transformerNameLhs': None,
        'blockIdx': '2', 'analysisName': 'hiRes', 'alignQuery': 'midPeak',
        'winStart': '200', 'exp': 'exp202101281100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    
'''


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
    nAlphas = 25
    ###
    ## statsmodels elasticnet
    regressorKWArgs = {
        'sm_class': sm.GLM,
        'family': sm.families.Gaussian(),
        'alpha': 1e-12, 'L1_wt': .1,
        'refit': True, 'tol': 1e-2,
        'maxiter': 1000, 'disp': False
        }
    regressorClass = tdr.SMWrapper
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
    lhsMasks = pd.read_hdf(designMatrixPath, '/featureMasks')
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
            rhsPipelineAveragerDict[rhsMaskIdx] = Pipeline[('averager', tdr.DataFramePassThrough(),)]
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
    #
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        slurmTaskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    else:
        slurmTaskID = 0
    if os.getenv('SLURM_ARRAY_TASK_COUNT') is not None:
        slurmTaskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    else:
        slurmTaskCount = 1
    if os.getenv('SLURM_ARRAY_TASK_MIN') is not None:
        slurmTaskMin = int(os.getenv('SLURM_ARRAY_TASK_MIN'))
    else:
        slurmTaskMin = 0
    #
    #
    if arguments['transformerNameRhs'] is not None:
        del transformedRhsDF, transformedRhsMasks
    #
    if slurmTaskID == slurmTaskMin:
        if os.path.exists(estimatorPath):
            os.remove(estimatorPath)
        if os.path.exists(estimatorMetaDataPath):
            os.remove(estimatorMetaDataPath)
        with open(estimatorMetaDataPath, 'wb') as f:
            pickle.dump(estimatorMetadata, f)
    #
    cvScoresDict0 = {}
    gridSearcherDict0 = {}
    gsScoresDict0 = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        if designFormula != 'NULL':
            formulaIdx = lOfDesignFormulas.index(designFormula)
            designDF = pd.read_hdf(designMatrixPath, 'designs/formula_{}'.format(formulaIdx))
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
            estimatorInstance = TransformedTargetRegressor(regressor=pipelineLhs, transformer=pipelineRhs, check_inverse=False)
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
                if (targetIdx % slurmTaskCount) != slurmTaskID:
                    continue
                print('\nslurmTaskID == {} targetIdx == {}\n'.format(slurmTaskID, targetIdx))
                prf.print_memory_usage('Fitting {} to {}...'.format(lhsMask.name[-1], targetName))
                ##
                targetDF = rhGroup.loc[:, [targetName]]
                # add targetDF to designDF?
                if ensTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(ensTemplate)
                    thisEnsDesign = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    ensHistList = [
                        thisEnsDesign.iloc[:, sl].copy()
                        for key, sl in ensDesignInfo.term_name_slices.items()
                        if key != ensTemplate.format(targetName)]
                    del thisEnsDesign
                else:
                    ensHistList = []
                #
                if selfTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(selfTemplate)
                    thisSelfDesign = pd.read_hdf(
                        designMatrixPath, 'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    selfHistList = [
                        thisSelfDesign.iloc[:, sl].copy()
                        for key, sl in selfDesignInfo.term_name_slices.items()
                        if key == selfTemplate.format(targetName)]
                    del thisSelfDesign
                else:
                    selfHistList = []
                #
                fullDesignList = exogList + ensHistList + selfHistList
                fullDesignDF = pd.concat(fullDesignList, axis='columns')
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
                                l1_ratio=l1Ratio, n_alphas=nAlphas)
                            alphasStr = ['{:.3g}'.format(a) for a in alphas]
                            print('Evaluating alphas: {}'.format(alphasStr))
                            gsParams.append(
                                {
                                    l1_ratio_name: [l1Ratio],
                                    alpha_name: np.atleast_1d(alphas).tolist()
                                }
                            )
                        gsKWA['param_grid'] = gsParams
                cvScores, gridSearcherDict1[targetName], gsScoresDict1[targetName] = tdr.gridSearchHyperparameters(
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
                cvScoresDF['estimator'].to_hdf(
                    estimatorPath,
                    'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                        lhsMaskIdx, rhsMaskIdx, targetName
                        ))
                prf.print_memory_usage('\n\nCompleted fit {} to {}...\n\n'.format(lhsMask.name[-1], targetName))