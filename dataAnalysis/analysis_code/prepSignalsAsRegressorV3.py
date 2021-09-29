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
    --maxNumFeatures=maxNumFeatures            load from raw, or regular? [default: 100]
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
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
#
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dask.distributed import Client, LocalCluster
import os, traceback
#
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetNameRhs'].split('_')[-1]))
#
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
from copy import deepcopy
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
for arg in sys.argv:
    print(arg)
##
if __name__ == '__main__':
    ##
    '''
    consoleDebugging = True
    if consoleDebugging:
        arguments = {
        'datasetNameLhs': 'Block_XL_df_ra',
        'window': 'long', 'lazy': False, 'showFigures': False, 'blockIdx': '2',
        'debugging': True, 'exp': 'exp202101281100',
        'transformerNameLhs': None, 'transformerNameRhs': 'pca_ta', 'plotting': True,
        'verbose': '1', 'processAll': True, 'datasetNameRhs': 'Block_XL_df_rc',
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
        'amplitude_raster': MinMaxScaler(feature_range=(0., 1)),
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
    #
    for designFormula, ensembleTemplate, selfTemplate in lOfEndogAndExogTemplates:
        if (ensembleTemplate == 'NULL') and (selfTemplate == 'NULL') and (designFormula == 'NULL'):
                continue
        attrValues = ['all', 0, designFormula, ensembleTemplate, selfTemplate]
        thisMask = pd.Series(
            True,
            index=lhsDF.columns).to_frame()
        thisMask.columns = pd.MultiIndex.from_tuples(
            (attrValues,), names=attrNames)
        maskList.append(thisMask.T)
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
    lhsMasks.to_hdf(
        designMatrixPath, 'featureMasks', mode='a')
    ####
    if arguments['plotting']:
        pdfPath = os.path.join(
            figureOutputFolder, '{}_history_basis.pdf'.format(designMatrixDatasetName)
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
    ##
    # prep rhs dataframes
    # histDesignDict = {}
    thisEnv = patsy.EvalEnvironment.capture()
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
    ###
    allTargetsList = []
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        allTargetsList.append(pd.concat(targetsList))
        allTargetsList[-1].loc[:, 'lhsMaskIdx'] = lhsMaskIdx
    allTargetsDF = pd.concat(allTargetsList, ignore_index=True)
    allTargetsDF.index.name = 'targetIdx'
    allTargetsDF.reset_index(inplace=True)
    allTargetsDF.set_index(['lhsMaskIdx', 'rhsMaskIdx', 'target'], inplace=True)
    print('Saving list of all targets to {}'.format(designMatrixPath))
    allTargetsDF.to_hdf(designMatrixPath, 'allTargets')
    htmlPath = os.path.join(figureOutputFolder, '{}.html'.format(designMatrixDatasetName))
    allTargetsDF.to_html(htmlPath)
    ###
    allTargetsPLS = allTargetsDF.reset_index().drop(['target', 'targetIdx'], axis='columns').drop_duplicates(subset=['lhsMaskIdx', 'rhsMaskIdx']).reset_index(drop=True)
    allTargetsPLS.loc[:, 'targetIdx'] = range(allTargetsPLS.shape[0])
    print('Saving list of all pls targets to {}'.format(designMatrixPath))
    allTargetsPLS.to_hdf(designMatrixPath, 'allTargetsPLS')
    htmlPathPLS = os.path.join(figureOutputFolder, '{}_pls.html'.format(designMatrixDatasetName))
    allTargetsPLS.to_html(htmlPathPLS)
    # prep lhs dataframes
    for parentFormulaIdx, parentFormula in enumerate(masterExogFormulas):
        prf.print_memory_usage('calculating exog terms for: {}'.format(parentFormula))
        pt = PatsyTransformer(parentFormula, eval_env=thisEnv, return_type="matrix")
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
        designDF.to_hdf(designMatrixPath, 'designs/exogParents/formula_{}'.format(parentFormulaIdx))
    #
    for formulaIdx, designFormula in enumerate(lOfDesignFormulas):
        if designFormula != 'NULL':
            prf.print_memory_usage('Looking up exog terms for: {}'.format(designFormula))
            parentFormula = masterExogLookup[designFormula]
            parentFormulaIdx = masterExogFormulas.index(parentFormula)
            pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
            exampleLhGroup = lhsDF.loc[lhsDF.index.get_level_values('conditionUID') == 0, :]
            #
            designMatrixExample = pt.fit_transform(exampleLhGroup)
            designInfo = designMatrixExample.design_info
            theseColumns = designInfo.column_names
            designDF = pd.read_hdf(designMatrixPath, 'designs/exogParents/formula_{}'.format(parentFormulaIdx))
            designDF = designDF.loc[:, theseColumns]
            designDF.to_hdf(designMatrixPath, 'designs/formula_{}'.format(formulaIdx))
    #####################################################################################################################
    ###
    #
    '''
    lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
    lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
    lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
    lhsMasksInfo.loc[:, 'designFormulaShortHand'] = lhsMasksInfo['designFormula'].apply(lambda x: formulasShortHand[x])
    lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormulaShortHand', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(lambda x: ' + '.join(x), axis='columns')
    '''
    rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
    rhsMasksInfo.to_hdf(designMatrixPath, 'rhsMasksInfo')
    ###
    lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
    lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
    lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
    lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormula', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(
        lambda x: ' + '.join(x), axis='columns')
    for key in ['nb', 'logBasis', 'historyLen', 'useOrtho', 'normalize', 'addInputToOutput']:
        # lhsMasksInfo.loc[:, key] = np.nan
        for rowIdx, row in lhsMasksInfo.iterrows():
            if row['designFormula'] in designHistOptsDict:
                theseHistOpts = designHistOptsDict[row['designFormula']]
                lhsMasksInfo.loc[rowIdx, key] = theseHistOpts[key]
            elif row['ensembleTemplate'] in templateHistOptsDict:
                theseHistOpts = templateHistOptsDict[row['ensembleTemplate']]
                lhsMasksInfo.loc[rowIdx, key] = theseHistOpts[key]
            else:
                lhsMasksInfo.loc[rowIdx, key] = 'NULL'
    # lhsMasksInfo.loc[:, 'lagSpec'] = np.nan
    for rowIdx, row in lhsMasksInfo.iterrows():
        if row['designFormula'] in designHistOptsDict:
            theseHistOpts = designHistOptsDict[row['designFormula']]
            lhsMasksInfo.loc[rowIdx, 'lagSpec'] = 'hto{}'.format(addHistoryTerms.index(theseHistOpts))
        elif row['ensembleTemplate'] in templateHistOptsDict:
            theseHistOpts = templateHistOptsDict[row['ensembleTemplate']]
            lhsMasksInfo.loc[rowIdx, 'lagSpec'] = 'hto{}'.format(addHistoryTerms.index(theseHistOpts))
        else:
            lhsMasksInfo.loc[rowIdx, 'lagSpec'] = 'NULL'
    htmlPath = os.path.join(
        figureOutputFolder, '{}_{}.html'.format(designMatrixDatasetName, 'designs_info'))
    lhsMasksInfo.drop(columns=['lag', 'maskName']).to_html(htmlPath)
    lhsMasksInfo.to_hdf(designMatrixPath, 'lhsMasksInfo')
    ###
    trialInfo = lhsDF.index.to_frame().reset_index(drop=True)
    stimCondition = pd.Series(np.nan, index=trialInfo.index)
    stimOrder = []
    for name, group in trialInfo.groupby(['electrode', 'trialRateInHz']):
        stimCondition.loc[group.index] = '{} {}'.format(*name)
        stimOrder.append('{} {}'.format(*name))
    trialInfo.loc[:, 'stimCondition'] = stimCondition
    stimConditionLookup = (
        trialInfo
            .loc[:, ['electrode', 'trialRateInHz', 'stimCondition']]
            .drop_duplicates()
            .set_index(['electrode', 'trialRateInHz'])['stimCondition'])
    kinCondition = pd.Series(np.nan, index=trialInfo.index)
    kinOrder = []
    for name, group in trialInfo.groupby(['pedalMovementCat', 'pedalDirection']):
        kinCondition.loc[group.index] = '{} {}'.format(*name)
        kinOrder.append('{} {}'.format(*name))
    trialInfo.loc[:, 'kinCondition'] = kinCondition
    kinConditionLookup = (
        trialInfo
            .loc[:, ['pedalMovementCat', 'pedalDirection', 'kinCondition']]
            .drop_duplicates()
            .set_index(['pedalMovementCat', 'pedalDirection'])['kinCondition'])
    #
    stimConditionLookup.to_hdf(designMatrixPath, 'stimConditionLookup')
    kinConditionLookup.to_hdf(designMatrixPath, 'kinConditionLookup')
    #####
    ################ define model comparisons
    # "test" should be the "bigger" model (we are adding coefficients and asking whether they improved performance
    modelsToTest = []
    ######### backup of version from comparing 56 models
    '''
    for lhsMaskIdx in [1, 2, 20, 21, 39, 40]:
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx + 2,
            'testCaption': 'v + a + r + va + vr + ar (+ var)',
            'refCaption': 'v + a + r + va + vr + ar',
            'captionStr': 'partial R2 of adding second order interaction terms',
            'testType': 'secondOrderInteractions',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
    for lhsMaskIdx in [3, 4, 22, 23, 41, 42]:
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx + 2,
            'testCaption': 'v + a + r + va + vr (+ ar)',
            'refCaption': 'v + a + r + va + vr',
            'captionStr': 'partial R2 of adding the AR interaction term',
            'testType': 'ARInteractions',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx + 4,
            'testCaption': 'v + a + r (+ va + vr) + ar',
            'refCaption': 'v + a + r + ar',
            'captionStr': 'partial R2 of adding the VR and VA interaction terms',
            'testType': 'VAVRInteractions',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
    for lhsMaskIdx in [9, 10, 28, 29, 47, 48]:
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx + 8,
            'testCaption': 'v + (a + r)',
            'refCaption': 'v',
            'captionStr': 'partial R2 of adding terms for A and R',
            'testType': 'ARTerms',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx + 6,
            'testCaption': '(v +) a + r',
            'refCaption': 'a + r',
            'captionStr': 'partial R2 of adding terms for V',
            'testType': 'VTerms',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
    for lhsMaskIdx in [10, 29, 48]:
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx - 1,
            'testCaption': 'v + a + r (+ ensemble_history)',
            'refCaption': 'v + a + r',
            'captionStr': 'partial R2 of adding terms for signal ensemble history to V+A+R',
            'testType': 'ensembleTerms',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx - 10,
            'testCaption': '(v + a + r +) ensemble_history',
            'refCaption': 'ensemble_history',
            'captionStr': 'partial R2 of adding terms for V+A+R to ensemble history',
            'testType': 'VARNoEnsembleTerms',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })'''
    for lhsMaskIdx in lhsMasksOfInterest['varVsEnsemble']:
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx - 1,
            'testCaption': 'v + a + r (+ ensemble_history)',
            'refCaption': 'v + a + r',
            'captionStr': 'partial R2 of adding terms for signal ensemble history to V+A+R',
            'testType': 'ensembleTerms',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
        modelsToTest.append({
            'testDesign': lhsMaskIdx,
            'refDesign': lhsMaskIdx - 2,
            'testCaption': '(v + a + r +) ensemble_history',
            'refCaption': 'ensemble_history',
            'captionStr': 'partial R2 of adding terms for V+A+R to ensemble history',
            'testType': 'VARNoEnsembleTerms',
            'testHasEnsembleHistory': lhsMasksInfo.loc[lhsMaskIdx, 'selfTemplate'] != 'NULL',
            'lagSpec': lhsMasksInfo.loc[lhsMaskIdx, 'lagSpec'],
        })
    modelsToTestDF = pd.DataFrame(modelsToTest)
    modelsToTestDF.to_hdf(designMatrixPath, 'modelsToTest')
    print('\n' + '#' * 50 + '\n{}\nComplete.\n'.format(__file__) + '#' * 50 + '\n')