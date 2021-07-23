"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                  which experimental day to analyze
    --blockIdx=blockIdx                        which trial to analyze [default: 1]
    --processAll                               process entire experimental day? [default: False]
    --analysisName=analysisName                append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName          append a name to the resulting blocks? [default: motion]
    --window=window                            process with short window? [default: long]
    --lazy                                     load from raw, or regular? [default: False]
    --plotting                                 load from raw, or regular? [default: False]
    --showFigures                              load from raw, or regular? [default: False]
    --debugging                                load from raw, or regular? [default: False]
    --maxNumFeatures=maxNumFeatures            load from raw, or regular? [default: 32]
    --verbose=verbose                          print diagnostics? [default: 0]
    --datasetNameRhs=datasetNameRhs            which trig_ block to pull [default: Block]
    --selectionNameRhs=selectionNameRhs        how to restrict channels? [default: fr_sqrt]
    --transformerNameRhs=transformerNameRhs    how to restrict channels?
    --datasetNameLhs=datasetNameLhs            which trig_ block to pull [default: Block]
    --selectionNameLhs=selectionNameLhs        how to restrict channels? [default: fr_sqrt]
    --transformerNameLhs=transformerNameLhs    how to restrict channels?
    --estimatorName=estimatorName              filename for resulting estimator (cross-validated n_comps)
"""
import logging
logging.captureWarnings(True)
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
    context='talk', style='darkgrid',
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
        'datasetNameLhs': 'Block_XL_df_ra',
        'window': 'long', 'lazy': False, 'showFigures': False, 'blockIdx': '2',
        'debugging': True, 'exp': 'exp202101281100',
        'transformerNameLhs': None, 'transformerNameRhs': 'pca_ta', 'plotting': True,
        'verbose': '1', 'processAll': True, 'datasetNameRhs': 'Block_XL_df_ra',
        'selectionNameLhs': 'rig', 'analysisName': 'hiRes', 'alignFolderName': 'motion',
        'estimatorName': '', 'selectionNameRhs': 'lfp_CAR'}
        os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
    ##
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
    designMatrixDatasetName = '{}_{}_{}_{}_regression_design_matrices'.format(
        arguments['datasetNameLhs'], arguments['selectionNameLhs'], arguments['selectionNameRhs'], arguments['transformerNameRhs'])
    designMatrixPath = os.path.join(
        dataFramesFolder,
        designMatrixDatasetName + '.h5'
        )
    if os.path.exists(designMatrixPath):
        os.remove(designMatrixPath)
    loadingMetaPathLhs = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '_' + arguments['selectionNameLhs'] + '_meta.pickle'
        )
    #
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        iteratorOpts = loadingMeta['iteratorOpts']
        iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
        cvIterator = iteratorsBySegment[0]
    #
    lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
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
    checkSameMeta = stimulusConditionNames + ['bin', 'trialUID', 'conditionUID']
    assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
    trialInfo = trialInfoLhs
    #
    lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
    rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
    ##################### end of data loading
    ## scale external covariates
    transformersLookup = {
        # 'forceMagnitude': MinMaxScaler(feature_range=(0., 1)),
        # 'forceMagnitude_prime': MinMaxScaler(feature_range=(-1., 1)),
        'amplitude': MinMaxScaler(feature_range=(0., 1)),
        'RateInHz': MinMaxScaler(feature_range=(0., .5)),
        'velocity': MinMaxScaler(feature_range=(-1., 1.)),
        'velocity_abs': MinMaxScaler(feature_range=(0., 1.)),
        }
    lOfTransformers = []
    for cN in lhsDF.columns:
        if cN[0] not in transformersLookup:
            lOfTransformers.append(([cN], None,))
        else:
            lOfTransformers.append(([cN], transformersLookup[cN[0]],))
    lhsScaler = DataFrameMapper(lOfTransformers, input_df=True, )
    lhsScaler.fit(lhsDF)
    lhsDF = pd.DataFrame(
        lhsScaler.transform(lhsDF), index=lhsDF.index, columns=lhsDF.columns)
    regressorsFromMetadata = ['electrode']
    columnAdder = tdr.DataFrameMetaDataToColumns(addColumns=regressorsFromMetadata)
    lhsDF = columnAdder.fit_transform(lhsDF)
    keepMask = lhsDF.columns.get_level_values('feature').isin(regressionColumnsToUse)
    lhsDF = lhsDF.loc[:, keepMask]
    lhsDF.rename(columns=regressionColumnRenamer, level='feature', inplace=True)
    lhsDF.columns = lhsDF.columns.get_level_values('feature')
    lhsDF.to_hdf(designMatrixPath, 'lhsDF', mode='a')
    ######## make lhs masks
    maskList = []
    attrNames = ['feature', 'lag', 'designFormula', 'ensembleTemplate', 'selfTemplate']
    for designFormula in lOfDesignFormulas:
        for ensembleTemplate, selfTemplate in lOfEnsembleTemplates:
            if (ensembleTemplate == 'NULL') and (selfTemplate == 'NULL') and (designFormula == 'NULL'):
                continue
            attrValues = ['all', 0, designFormula, ensembleTemplate, selfTemplate]
            thisMask = pd.Series(
                True,
                index=lhsDF.columns).to_frame()
            thisMask.columns = pd.MultiIndex.from_tuples(
                (attrValues,), names=attrNames)
            maskList.append(thisMask.T)
    #
    lhsMasks = pd.concat(maskList)
    maskParams = [
        {k: v for k, v in zip(lhsMasks.index.names, idxItem)}
        for idxItem in lhsMasks.index
    ]
    maskParamsStr = [
        '{}'.format(idxItem).replace("'", '')
        for idxItem in maskParams]
    lhsMasks.loc[:, 'maskName'] = maskParamsStr
    lhsMasks.set_index('maskName', append=True, inplace=True)
    # pdb.set_trace()
    lhsMasks.to_hdf(
        designMatrixPath, 'featureMasks', mode='a')
    ###
    if arguments['plotting']:
        pdfPath = os.path.join(
            figureOutputFolder, 'history_basis.pdf'
            )
        cm = PdfPages(pdfPath)
    else:
        import contextlib
        cm = contextlib.nullcontext()
    with cm as pdf:
        for hIdx, histOpts in enumerate(addHistoryTerms):
            formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
            locals().update({'hto{}'.format(hIdx): formattedHistOpts})
            raisedCosBaser = tdr.raisedCosTransformer(formattedHistOpts)
            if arguments['plotting']:
                fig, ax = raisedCosBaser.plot_basis()
                fig.suptitle('hto{}'.format(hIdx))
                fig.tight_layout()
                pdf.savefig(
                    bbox_inches='tight',
                    )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
    thisEnv = patsy.EvalEnvironment.capture()
    # prep rhs dataframes
    histDesignDict = {}
    targetsList = []
    for rhsMaskIdx in range(rhsMasks.shape[0]):
        prf.print_memory_usage('Prepping RHS on rhsRow {}'.format(rhsMaskIdx))
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        rhGroup = rhsDF.loc[:, rhsMask].copy()
        # transform to PCs
        if workingPipelinesRhs is not None:
            transformPipelineRhs = workingPipelinesRhs.xs(rhsMaskParams['freqBandName'], level='freqBandName').iloc[0]
            rhsPipelineMinusAverager = Pipeline(transformPipelineRhs.steps[1:])
            rhTransformedColumns = transformedRhsDF.columns[
                transformedRhsDF.columns.get_level_values('freqBandName') == rhsMaskParams['freqBandName']]
            rhGroup = pd.DataFrame(
                rhsPipelineMinusAverager.transform(rhGroup),
                index=rhsDF.index, columns=rhTransformedColumns)
        rhGroup.columns = rhGroup.columns.get_level_values('feature')
        #####################
        theseMaxNumFeatures = min(rhGroup.shape[1], int(arguments['maxNumFeatures']))
        print('Restricting target group to its first {} features'.format(theseMaxNumFeatures))
        rhGroup = rhGroup.iloc[:, :theseMaxNumFeatures]
        ####################
        rhGroup.to_hdf(designMatrixPath, 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
        targetsList.append(pd.Series(rhGroup.columns).to_frame(name='target'))
        targetsList[-1].loc[:, 'rhsMaskIdx'] = rhsMaskIdx
        #
        for templateIdx, ensTemplate in enumerate(lOfHistTemplates):
            if ensTemplate != 'NULL':
                ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
                ensFormula += ' - 1'
                prf.print_memory_usage('Calculating history terms as {}'.format(ensFormula))
                ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
                exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
                ensPt.fit(exampleRhGroup)
                ensDesignMatrix = ensPt.transform(rhGroup)
                ensDesignInfo = ensDesignMatrix.design_info
                thisHistDesign = (
                    pd.DataFrame(
                        ensDesignMatrix,
                        index=rhGroup.index,
                        columns=ensDesignInfo.column_names))
                thisHistDesign.columns.name = 'factor'
                thisHistDesign.to_hdf(designMatrixPath, 'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
    del rhsDF
    # prep lhs dataframes
    designDict = {}
    for formulaIdx, designFormula in enumerate(lOfDesignFormulas):
        if designFormula != 'NULL':
            prf.print_memory_usage('calculating exog terms for: {}'.format(designFormula))
            pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
            exampleLhGroup = lhsDF.loc[lhsDF.index.get_level_values('conditionUID') == 0, :]
            pt.fit(exampleLhGroup)
            designMatrix = pt.transform(lhsDF)
            designInfo = designMatrix.design_info
            designDF = (
                pd.DataFrame(
                    designMatrix,
                    index=lhsDF.index,
                    columns=designInfo.column_names))
            designDF.columns.name = 'factor'
            designDF.to_hdf(designMatrixPath, 'designs/formula_{}'.format(formulaIdx))

    allTargetsList = []
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        allTargetsList.append(pd.concat(targetsList))
        allTargetsList[-1].loc[:, 'lhsMaskIdx'] = lhsMaskIdx
    allTargetsDF = pd.concat(allTargetsList, ignore_index=True)
    allTargetsDF.index.name = 'targetIdx'
    allTargetsDF.reset_index(inplace=True)
    allTargetsDF.set_index(['lhsMaskIdx', 'rhsMaskIdx', 'target'], inplace=True)
    print('Complete. Saving allTargets to {}'.format(designMatrixPath))
    allTargetsDF.to_hdf(designMatrixPath, 'allTargets')
    htmlPath = os.path.join(figureOutputFolder, '{}.html'.format(designMatrixDatasetName))
    allTargetsDF.to_html(htmlPath)
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
    hto1['preprocFun'] = lambda x: x.diff().fillna(0)
    raisedCosBaser = tdr.raisedCosTransformer(hto1)
    #
    if arguments['plotting']:
        pdfPath = os.path.join(
            figureOutputFolder, 'history_basis.pdf'
            )
        fig, ax = raisedCosBaser.plot_basis()
        plt.savefig(pdfPath)
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
            ax[0].plot(exampleFeaturesDF[cN].to_numpy(), '.-', label='input {}'.format(cN))
        ax[0].legend()
        for cN in designDF.columns:
            ax[1].plot(designDF[cN].to_numpy(), '.-', label=cN)
        ax[1].legend()
        plt.show()
        '''
    #####################################################################################################################
    #

