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
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor, LinearRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from dataAnalysis.custom_transformers.target_transformer import TransformedTargetRegressor
from sklego.preprocessing import PatsyTransformer
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.linear_model._coordinate_descent import _alpha_grid
import joblib as jb
from copy import copy, deepcopy
import dill as pickle
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
        'verbose': '2', 'winStop': '400', 'selectionNameRhs': 'lfp_CAR', 'processAll': True,
        'transformerNameLhs': None, 'analysisName': 'hiRes', 'alignFolderName': 'motion',
        'estimatorName': 'enr', 'exp': 'exp202101281100', 'datasetNameRhs': 'Block_XL_df_d',
        'lazy': False, 'winStart': '200', 'selector': None, 'blockIdx': '2', 'window': 'long',
        'debugging': True, 'datasetNameLhs': 'Block_XL_df_d', 'transformerNameRhs': 'pca_ta',
        'alignQuery': 'midPeak', 'selectionNameLhs': 'rig', 'showFigures': False, 'plotting': True}
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
            arguments['analysisName'], 'pls')
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
    fullEstimatorName = '{}_{}_{}'.format(
        arguments['estimatorName'], arguments['datasetNameLhs'], arguments['selectionNameLhs'])
    #
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    loadingMetaPathLhs = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '_' + arguments['selectionNameLhs'] + '_meta.pickle'
        )
    #
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
    #
    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    workIdx = cvIterator.work
    #
    joblibBackendArgs = dict(
        # backend='dask',
        backend='loky',
        #### n_jobs=1
    )
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
    crossvalKWArgs = dict(
        return_train_score=True, return_estimator=True)
    #
    estimatorClass = ElasticNet
    estimatorKWArgs = dict(fit_intercept=False, l1_ratio=0.5, alpha=1e-12)
    # names of things in sklearn
    l1_ratio_name, alpha_name = 'regressor__regressor__l1_ratio', 'regressor__regressor__alpha'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid=dict(
            regressor__regressor__l1_ratio=[.1, .5, .95, 1])
        )
    nAlphas = 20
    #
    '''estimatorClass = LinearRegression
    estimatorKWArgs = dict(fit_intercept=False)
    # names of things in sklearn
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False
        )'''
    #
    crossvalKWArgs['cv'] = cvIterator
    gridSearchKWArgs['cv'] = cvIterator

    lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
    lhsMasks = pd.read_hdf(lhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameLhs']))
    rhsMasks = pd.read_hdf(rhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameRhs']))
    #
    if arguments['transformerNameRhs'] is not None:
        transformedSelectionNameRhs = '{}_{}'.format(
            arguments['selectionNameRhs'], arguments['transformerNameRhs'])
        transformedRhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(transformedSelectionNameRhs))
        transformedRhsMasks = pd.read_hdf(rhsDatasetPath, '/{}/featureMasks'.format(transformedSelectionNameRhs))
        transformedRhsMasks.index.get_level_values('feature')
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
    else:
        workingPipelinesRhs = None
    #
    if arguments['transformerNameLhs'] is not None:
        pass
    else:
        workingPipelinesLhs = None
    #
    trialInfoLhs = lhsDF.index.to_frame().reset_index(drop=True)
    trialInfoRhs = rhsDF.index.to_frame().reset_index(drop=True)
    checkSameMeta = stimulusConditionNames + ['bin', 'trialUID']
    assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
    trialInfo = trialInfoLhs
    #
    lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
    rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
    # fill zeros, e.g. if trials do not have measured position, positions will be NA
    lhsDF.fillna(0, inplace=True)
    # plt.plot(lhsDF.loc[:, idxSl['trialRateInHz', :, :, :, :]].to_numpy()); plt.show()
    if ('amplitude' in lhsDF.columns.get_level_values('feature')) and ('RateInHz' in lhsDF.columns.get_level_values('feature')):
        lhsDF.loc[:, idxSl['RateInHz', :, :, :, :]] = lhsDF.loc[:, idxSl['RateInHz', :, :, :, :]] * (lhsDF.loc[:, idxSl['amplitude', :, :, :, :]].abs() > 0).to_numpy(dtype=float)
    # plt.plot(lhsDF.loc[:, idxSl['RateInHz', :, :, :, :]].to_numpy()); plt.show()
    # for cN in ['electrode', 'RateInHz']
    lOfDesignFormulas = ["velocity + electrode:(amplitude/RateInHz)"]
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
    scaledLhsDF = pd.DataFrame(
        lhsScaler.fit_transform(lhsDF), index=lhsDF.index, columns=lhsDF.columns)
    #
    lhsMasks.iloc[0, :] = lhsMasks.columns.get_level_values('feature').isin(transformersLookup.keys())
    regressorsFromMetadata = ['electrode']
    #
    workingLhsDF = scaledLhsDF.iloc[workIdx, :]
    workingRhsDF = rhsDF.iloc[workIdx, :]
    nFeatures = scaledLhsDF.columns.shape[0]
    nTargets = rhsDF.columns.shape[0]
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
    with PdfPages(figureOutputPath) as pdf:
        for lhsRowIdx, rhsRowIdx in product(list(range(lhsMasks.shape[0])), list(range(rhsMasks.shape[0]))):
            lhsMask = lhsMasks.iloc[lhsRowIdx, :]
            rhsMask = rhsMasks.iloc[rhsRowIdx, :]
            lhsMaskParams = {k: v for k, v in zip(lhsMask.index.names, lhsMask.name)}
            rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
            lhGroup = scaledLhsDF.loc[:, lhsMask].copy()
            for cN in regressorsFromMetadata:
                cNKey = (cN, 0,) + ('NA',) * 4
                lhGroup.loc[:, cNKey] = trialInfoLhs.loc[:, cN].to_numpy()
            #
            lhGroup.columns = lhGroup.columns.get_level_values('feature')
            #
            pt = PatsyTransformer(lOfDesignFormulas[lhsRowIdx], return_type="matrix")
            designMatrix = pt.fit_transform(lhGroup)
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
                # test: rhst = transformPipelineRhs.transform(workingRhsDF)
                rhTransformedColumns = transformedRhsDF.columns[transformedRhsDF.columns.get_level_values('freqBandName') == rhsMaskParams['freqBandName']]
                rhGroup = pd.DataFrame(
                    rhsPipelineMinusAverager.transform(rhGroup),
                    index=lhGroup.index, columns=rhTransformedColumns)
            else:
                rhsPipelineAverager = Pipeline[('averager', tdr.DataFramePassThrough(), )]
            rhGroup.columns = rhGroup.columns.get_level_values('feature')
            ####
            if arguments['debugging']:
                if rhsMaskParams['freqBandName'] not in ['beta', 'gamma', 'higamma', 'all']:
                    # if maskParams['lag'] not in [0]:
                    continue
            pipelineRhs = Pipeline([('averager', rhsPipelineAverager, ), ])
            pipelineLhs = Pipeline([('averager', rhsPipelineAverager, ), ('regressor', estimatorClass(**estimatorKWArgs)), ])
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
            ####################################################################################################
            #### diagnostic plots
            didDesignPlot = False
            #########################################################################################################
            cvScoresDict1 = {}
            gridSearcherDict1 = {}
            gsScoresDict1 = {}
            for targetName in rhGroup.columns:
                ####
                if arguments['debugging']:
                    if targetName not in rhGroup.columns[:2]:
                        continue
                ###
                print('Fitting {} to {}...'.format(lhsMask.name[-1], targetName))
                if gsParamsPerTarget is not None:
                    gsKWA['param_grid'] = gsParamsPerTarget[targetName]
                ##
                if False:
                    randomNumGen = np.random.default_rng()
                    targetDF = (designDF.sum(axis='columns') + randomNumGen.normal(0, .1, designDF.shape[0])).to_frame(name=targetName)
                else:
                    targetDF = rhGroup.loc[:, [targetName]]
                ########################################
                #### diagnostic plots
                estForPlot = clone(estimatorInstance)
                estForPlot.fit(designDF, targetDF)
                designTr = Pipeline(estForPlot.regressor.steps[:-1]).transform(designDF)
                if not didDesignPlot:
                    plotDesignDF = designTr.stack().to_frame(name='signal')
                    g = sns.relplot(
                        col='electrode', row='pedalMovementCat', x='bin', y='signal', hue='feature',
                        data=plotDesignDF, kind='line', errorbar='se'
                        )
                    plt.show()
                    didDesignPlot = True
                rhTr = estForPlot.transformer.transform(targetDF)
                # check that it doesn't matter if we average first or project first
                '''rhTr2 = pd.DataFrame(
                    transformPipelineRhs.transform(rhsDF.loc[:, rhsMask].copy()),
                    index=rhTr.index, columns=rhTransformedColumns)
                rhTr2.columns = rhTr2.columns.get_level_values('feature')
                rhTr2 = rhTr2.loc[:, [targetName]]
                #
                (rhTr2 - rhTr).abs().max()'''
                rhPred = pd.DataFrame(
                    estForPlot.predict(designDF),
                    index=rhTr.index, columns=[targetName])
                plotTargetDF = pd.concat(
                    {'y_pred': rhPred, 'y_true': rhTr}, names=['signalType', 'feature'], axis='columns')
                plotTargetDF = plotTargetDF.stack(['signalType', 'feature']).to_frame(name='signal').reset_index()
                sns.relplot(
                    col='electrode', row='pedalMovementCat', x='bin', y='signal', style='signalType',
                    data=plotTargetDF, kind='line', errorbar='se'
                    )
                plt.show()
                ########################################
                '''cvScores, gridSearcherDict1[targetName], gsScoresDict1[targetName] = tdr.gridSearchHyperparameters(
                    designDF, rhGroup.loc[:, [targetName]],
                    estimatorInstance=estimatorInstance,
                    # estimatorKWArgs=estimatorKWArgs, estimatorClass=estimatorClass,
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
                pdb.set_trace()
            cvScoresDict0[(lhsRowIdx, rhsRowIdx)] = pd.concat(cvScoresDict1, names=['target'])
            gsScoresDict0[(lhsRowIdx, rhsRowIdx)] = pd.concat(gsScoresDict1, names=['target'])'''
    #
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    allScoresDF.to_hdf(estimatorPath, 'cv')
    '''loadingMeta['arguments'] = arguments.copy()
    loadingMeta['lhGroupNames'] = lhGroupNames
    loadingMeta['lhsNormalizationParams'] = lhsNormalizationParams
    loadingMeta['rhsNormalizationParams'] = rhsNormalizationParams
    with open(estimatorPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(loadingMeta, f)'''
    #
    if arguments['plotting']:
        figureOutputPath = os.path.join(
                figureOutputFolder,
                '{}_r2.pdf'.format(fullEstimatorName))
        scoresForPlot = pd.concat(
            {'test': allScoresDF['test_score'], 'train': allScoresDF['train_score']},
            names=['evalType']).to_frame(name='score').reset_index()
        lastFoldIdx = scoresForPlot['fold'].max()
        validationMask = (
            (scoresForPlot['fold'] == lastFoldIdx) &
            (scoresForPlot['evalType'] == 'test'))
        scoresForPlot.loc[validationMask, 'evalType'] = 'validation'
        workingMask = (
            (scoresForPlot['fold'] == lastFoldIdx) &
            (scoresForPlot['evalType'] == 'train'))
        scoresForPlot.loc[workingMask, 'evalType'] = 'work'
        colWrap = np.ceil(np.sqrt(scoresForPlot['maskName'].unique().size)).astype(int)
        with PdfPages(figureOutputPath) as pdf:
            # fig, ax = plt.subplots()
            # fig.set_size_inches(12, 8)
            g = sns.catplot(
                data=scoresForPlot, hue='evalType',
                col='maskName', col_wrap=colWrap,
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
