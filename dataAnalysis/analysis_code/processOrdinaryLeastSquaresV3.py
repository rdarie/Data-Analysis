"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose=verbose                        print diagnostics?
    --debugging                              print diagnostics? [default: False]
    --forceReprocess                         print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
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
import dataAnalysis.plotting.aligned_signal_plots as asp
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
import colorsys
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
for arg in sys.argv:
    print(arg)
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
        'lines.markersize': 2.4,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 7
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV


if __name__ == '__main__':
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    # if debugging in a console:
    '''
    consoleDebugging = True
    if consoleDebugging:
        arguments = {
            'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_rd', 'plotting': True,
            'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
            'verbose': '1', 'debugging': False, 'estimatorName': 'enr_fa_ta', 'forceReprocess': False,
            'blockIdx': '2', 'exp': 'exp202101271100'}
        os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''

    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'], 'regression')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    #
    datasetName = arguments['datasetName']
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], arguments['datasetName'])
    #
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    dataFramesFolder = os.path.join(
        analysisSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    estimatorMetaDataPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_meta.pickle'
        )
    with open(estimatorMetaDataPath, 'rb') as _f:
        estimatorMeta = pickle.load(_f)
    #
    loadingMetaPath = estimatorMeta['loadingMetaPath']
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        iteratorOpts = loadingMeta['iteratorOpts']
        binInterval = iteratorOpts['forceBinInterval'] if (iteratorOpts['forceBinInterval'] is not None) else rasterOpts['binInterval']
    #
    histOptsForExportDict = {}
    for hIdx, histOpts in enumerate(addHistoryTerms):
        formattedHistOpts = getHistoryOpts(histOpts, iteratorOpts, rasterOpts)
        locals().update({'hto{}'.format(hIdx): formattedHistOpts})
        histOptsForExportDict['hto{}'.format(hIdx)] = formattedHistOpts
        # locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    histOptsForExportDF = pd.DataFrame(histOptsForExportDict)
    histOptsHtmlPath = os.path.join(
        figureOutputFolder, '{}_{}.html'.format(fullEstimatorName, 'histOpts'))
    histOptsForExportDF.to_html(histOptsHtmlPath)
    thisEnv = patsy.EvalEnvironment.capture()

    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    lastFoldIdx = cvIterator.n_splits
    #
    selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
    selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
    #
    lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
    lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
    allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets')
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    #
    rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
    ###
    lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
    lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
    lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
    lhsMasksInfo.loc[:, 'designFormulaShortHand'] = lhsMasksInfo['designFormula'].apply(lambda x: formulasShortHand[x])
    lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormulaShortHand', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(lambda x: ' + '.join(x), axis='columns')
    for key in ['nb', 'logBasis', 'historyLen', 'useOrtho', 'normalize', 'addInputToOutput']:
        # lhsMasksInfo.loc[:, key] = np.nan
        for rowIdx, row in lhsMasksInfo.iterrows():
            if row['designFormula'] in designHistOptsDict:
                theseHistOpts = designHistOptsDict[row['designFormula']]
                lhsMasksInfo.loc[rowIdx, key] = theseHistOpts[key]
            else:
                lhsMasksInfo.loc[rowIdx, key] = 'NULL'
    # lhsMasksInfo.loc[:, 'lagSpec'] = np.nan
    for rowIdx, row in lhsMasksInfo.iterrows():
        if row['designFormula'] in designHistOptsDict:
            theseHistOpts = designHistOptsDict[row['designFormula']]
            lhsMasksInfo.loc[rowIdx, 'lagSpec'] = 'hto{}'.format(addHistoryTerms.index(theseHistOpts))
        else:
            lhsMasksInfo.loc[rowIdx, 'lagSpec'] = 'NULL'
    htmlPath = os.path.join(
        figureOutputFolder, '{}_{}.html'.format(fullEstimatorName, 'designs_info'))
    lhsMasksInfo.drop(columns=['lag', 'maskName']).to_html(htmlPath)
    ##

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
    ####
    ################ define model comparisons
    # "test" should be the "bigger" model (we are adding coefficients and asking whether they improved performance
    modelsToTest = []
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
            })
    modelsToTestDF = pd.DataFrame(modelsToTest)
    modelsToTestDF.to_hdf(estimatorPath, 'modelsToTest')
    ################ end define model comparisons
    #
    ################ collect estimators and scores
    estimatorsDict = {}
    scoresDict = {}
    if processSlurmTaskCount is not None:
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
        allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
    for rowIdx, row in allTargetsDF.iterrows():
        lhsMaskIdx, rhsMaskIdx, targetName = row.name
        if processSlurmTaskCount is not None:
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
        else:
            thisEstimatorPath = estimatorPath
        scoresDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            thisEstimatorPath,
            'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            thisEstimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    scoresDF = pd.concat(scoresDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    ################ end define model comparisons
    #
    ################ collect estimators and scores
    with pd.HDFStore(estimatorPath) as store:
        if 'coefficients' in store:
            coefDF = pd.read_hdf(store, 'coefficients')
            loadedCoefs = True
        else:
            loadedCoefs = False
        if 'sourcePalette' in store:
            sourcePalette = pd.read_hdf(store, 'sourcePalette')
            termPalette = pd.read_hdf(store, 'termPalette')
            factorPalette = pd.read_hdf(store, 'factorPalette')
            trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
            sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
            loadedPlotOpts = True
        else:
            loadedPlotOpts = False
        if 'predictions' in store:
            predDF = pd.read_hdf(store, 'predictions')
            loadedPreds = True
        else:
            loadedPreds = False
        if 'processedCVScores' in store:
            scoresStack = pd.read_hdf(store, 'processedCVScores')
            llDF = pd.read_hdf(store, 'processedLogLike')
            aicDF = pd.read_hdf(store, 'processedAIC')
            R2Per = pd.read_hdf(store, 'processedR2')
            modelCompareFUDE = pd.read_hdf(store, 'modelCompareFUDE')
            modelCompareFUDEStats = pd.read_hdf(store, 'modelCompareFUDEStats')
            modelCompareScores = pd.read_hdf(store, 'modelCompareScores')
            loadedProcessedScores = True
        else:
            loadedProcessedScores = False
    # prep rhs dataframes
    histDesignInfoDict = {}
    histSourceTermDict = {}
    for rhsMaskIdx in range(rhsMasks.shape[0]):
        prf.print_memory_usage('\n Prepping RHS dataframes (rhsRow: {})\n'.format(rhsMaskIdx))
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
        for ensTemplate in lOfHistTemplates:
            if ensTemplate != 'NULL':
                histSourceTermDict.update({ensTemplate.format(cN): cN for cN in rhGroup.columns})
                ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
                ensFormula += ' - 1'
                print('Generating endog design info for {}'.format(ensFormula))
                ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
                exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
                ensPt.fit(exampleRhGroup)
                ensDesignMatrix = ensPt.transform(exampleRhGroup)
                ensDesignInfo = ensDesignMatrix.design_info
                print(ensDesignInfo.term_names)
                print('\n')
                histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
    #
    designInfoDict = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        #
        if designFormula != 'NULL':
            if designFormula not in designInfoDict:
                print('Generating exog design info for {}'.format(designFormula))
                formulaIdx = lOfDesignFormulas.index(designFormula)
                lhGroup = lhsDF.loc[:, lhsMask]
                pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
                exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
                designMatrix = pt.fit_transform(exampleLhGroup)
                designInfo = designMatrix.design_info
                designInfoDict[designFormula] = designInfo
    #
    designInfoDF = pd.Series(designInfoDict).to_frame(name='designInfo')
    designInfoDF.index.name = 'design'
    histDesignInfoDF = pd.DataFrame(
        [value for key, value in histDesignInfoDict.items()],
        columns=['designInfo'])
    histDesignInfoDF.index = pd.MultiIndex.from_tuples(
        [key for key, value in histDesignInfoDict.items()],
        names=['rhsMaskIdx', 'ensTemplate'])
    ################################################################################################
    reProcessPredsCoefs = not (loadedCoefs and loadedPreds)
    if reProcessPredsCoefs or arguments['forceReprocess']:
        coefDict0 = {}
        predDict0 = {}
        print('Calculating predicted waveforms per model')
        for lhsMaskIdx in range(lhsMasks.shape[0]):
            lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
            lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
            designFormula = lhsMaskParams['designFormula']
            lhGroup = lhsDF.loc[:, lhsMask]
            if designFormula != 'NULL':
                designInfo = designInfoDict[designFormula]
                formulaIdx = lOfDesignFormulas.index(designFormula)
                designDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'designs/formula_{}'.format(formulaIdx))
                exogList = [designDF]
                designTermNames = designInfo.term_names
            else:
                exogList = []
                designInfo = None
                designDF = None
                designTermNames = []
            #
            # add ensemble to designDF?
            ensTemplate = lhsMaskParams['ensembleTemplate']
            selfTemplate = lhsMaskParams['selfTemplate']
            for rhsMaskIdx in range(rhsMasks.shape[0]):
                rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
                rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
                rhGroup = pd.read_hdf(
                    estimatorMeta['designMatrixPath'],
                    'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
                if ensTemplate != 'NULL':
                    ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
                if selfTemplate != 'NULL':
                    selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
                ####
                for targetName in rhGroup.columns:
                    # add targetDF to designDF?
                    if ensTemplate != 'NULL':
                        templateIdx = lOfHistTemplates.index(ensTemplate)
                        thisEnsDesign = pd.read_hdf(
                            estimatorMeta['designMatrixPath'],
                            'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                        ensHistList = [
                            thisEnsDesign.iloc[:, sl]
                            for key, sl in ensDesignInfo.term_name_slices.items()
                            if key != ensTemplate.format(targetName)]
                        del thisEnsDesign
                        ensTermNames = [
                            tN
                            for tN in ensDesignInfo.term_names
                            if tN != ensTemplate.format(targetName)]
                    else:
                        ensHistList = []
                        ensTermNames = []
                    #
                    if selfTemplate != 'NULL':
                        templateIdx = lOfHistTemplates.index(selfTemplate)
                        thisSelfDesign = pd.read_hdf(
                            estimatorMeta['designMatrixPath'], 'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                        selfHistList = [
                            thisSelfDesign.iloc[:, sl].copy()
                            for key, sl in selfDesignInfo.term_name_slices.items()
                            if key == selfTemplate.format(targetName)]
                        del thisSelfDesign
                        selfTermNames = [
                            tN
                            for tN in selfDesignInfo.term_names
                            if tN == selfTemplate.format(targetName)]
                        # print('selfHistList:\n{}\nselfTermNames:\n{}'.format(selfHistList, selfTermNames))
                    else:
                        selfHistList = []
                        selfTermNames = []
                    #
                    fullDesignList = exogList + ensHistList + selfHistList
                    fullDesignDF = pd.concat(fullDesignList, axis='columns')
                    del fullDesignList
                    for foldIdx in range(cvIterator.n_splits + 1):
                        targetDF = rhGroup.loc[:, [targetName]]
                        estimatorIdx = (lhsMaskIdx, rhsMaskIdx, targetName, foldIdx)
                        if int(arguments['verbose']) > 0:
                            print('estimator: {}'.format(estimatorIdx))
                            if int(arguments['verbose']) > 3:
                                print('in dataframe: {}'.format(estimatorIdx in estimatorsDF.index))
                        if not estimatorIdx in estimatorsDF.index:
                            continue
                        if foldIdx == cvIterator.n_splits:
                            # work and validation folds
                            trainIdx, testIdx = cvIterator.workIterator.split(rhGroup)[0]
                            trainStr, testStr = 'work', 'validation'
                            foldType = 'validation'
                        else:
                            trainIdx, testIdx = cvIterator.raw_folds[foldIdx]
                            trainStr, testStr = 'train', 'test'
                            foldType = 'train'
                        estimator = estimatorsDF.loc[estimatorIdx]
                        coefs = pd.Series(
                            estimator.regressor_.named_steps['regressor'].coef_, index=fullDesignDF.columns)
                        coefDict0[(lhsMaskIdx, designFormula, rhsMaskIdx, targetName, foldIdx)] = coefs
                        estPreprocessorLhs = Pipeline(estimator.regressor_.steps[:-1])
                        estPreprocessorRhs = estimator.transformer_
                        predictionPerComponent = pd.concat({
                            trainStr: estPreprocessorLhs.transform(fullDesignDF.iloc[trainIdx, :]) * coefs,
                            testStr: estPreprocessorLhs.transform(fullDesignDF.iloc[testIdx, :]) * coefs
                            }, names=['trialType'])
                        predictionSrs = predictionPerComponent.sum(axis='columns')
                        # sanity check
                        #############################
                        indicesThisFold = np.concatenate([trainIdx, testIdx])
                        predictionsNormalWay = np.concatenate([
                            estimator.predict(fullDesignDF.iloc[trainIdx, :]),
                            estimator.predict(fullDesignDF.iloc[testIdx, :])
                            ])
                        mismatch = predictionSrs - predictionsNormalWay.reshape(-1)
                        if int(arguments['verbose']) > 3:
                            print('max mismatch is {}'.format(mismatch.abs().max()))
                        try:
                            assert (mismatch.abs().max() < 1e-3)
                        except:
                            print('Attention! max mismatch is {}'.format(mismatch.abs().max()))
                        termNames = designTermNames + ensTermNames + selfTermNames
                        predictionPerSource = pd.DataFrame(
                            np.nan, index=predictionPerComponent.index,
                            columns=termNames)
                        for termName in designTermNames:
                            termSlice = designInfo.term_name_slices[termName]
                            factorNames = designInfo.column_names[termSlice]
                            predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                        if ensTemplate is not None:
                            for termName in ensTermNames:
                                factorNames = ensDesignInfo.column_names[ensDesignInfo.term_name_slices[termName]]
                                predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                        if selfTemplate is not None:
                            for termName in selfTermNames:
                                factorNames = selfDesignInfo.column_names[selfDesignInfo.term_name_slices[termName]]
                                predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                        #
                        predictionPerSource.loc[:, 'prediction'] = predictionSrs
                        predictionPerSource.loc[:, 'ground_truth'] = np.concatenate([
                            estPreprocessorRhs.transform(targetDF.iloc[trainIdx, :]),
                            estPreprocessorRhs.transform(targetDF.iloc[testIdx, :])
                            ])
                        predDict0[(lhsMaskIdx, designFormula, rhsMaskIdx, targetName, foldIdx, foldType)] = predictionPerSource
        #
        predDF = pd.concat(predDict0, names=['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'foldType'])
        predDF.columns.name = 'term'
        coefDF = pd.concat(coefDict0, names=['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'factor'])
        coefDF.to_hdf(estimatorPath, 'coefficients')
        predDF.to_hdf(estimatorPath, 'predictions')
        print('Loaded and saved predictions and coefficients')
    else:
        print('Predictions and coefficients loaded from .h5 file')
    ################################################################################################
    if (not loadedPlotOpts) or arguments['forceReprocess']:
        termPalette = pd.concat({
            'exog': pd.Series(np.unique(np.concatenate([di.term_names for di in designInfoDF['designInfo']]))),
            'endog': pd.Series(np.unique(np.concatenate([di.term_names for di in histDesignInfoDF['designInfo']]))),
            'other': pd.Series(['prediction', 'ground_truth']),
            }, names=['type', 'index']).to_frame(name='term')
        sourceTermLookup = pd.concat({
            'exog': pd.Series(sourceTermDict).to_frame(name='source'),
            'endog': pd.Series(histSourceTermDict).to_frame(name='source'),
            'other': pd.Series(
                ['prediction', 'ground_truth'],
                index=['prediction', 'ground_truth']).to_frame(name='source'),}, names=['type', 'term'])
        #
        primaryPalette = pd.DataFrame(sns.color_palette('colorblind'), columns=['r', 'g', 'b'])
        pickingColors = False
        if pickingColors:
            sns.palplot(primaryPalette.apply(lambda x: tuple(x), axis='columns'))
            palAx = plt.gca()
            for tIdx, tN in enumerate(primaryPalette.index):
                palAx.text(tIdx, .5, '{}'.format(tN))
        rgb = pd.DataFrame(
            primaryPalette.iloc[[1, 0, 2, 4, 7], :].to_numpy(),
            columns=['r', 'g', 'b'], index=['v', 'a', 'r', 'ens', 'prediction'])
        hls = rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']), axis='columns')
        hls.loc['a*r', :] = hls.loc[['a', 'r'], :].mean()
        hls.loc['v*r', :] = hls.loc[['v', 'r'], :].mean()
        hls.loc['v*a', :] = hls.loc[['v', 'v', 'a'], :].mean()
        hls.loc['v*a*r', :] = hls.loc[['v', 'a', 'r'], :].mean()
        for sN in ['a*r', 'v*r', 'v*a', 'v*a*r']:
            hls.loc[sN, 's'] = hls.loc[sN, 's'] * 0.75
            hls.loc[sN, 'l'] = hls.loc[sN, 'l'] * 1.2
        hls.loc['v*a*r', 's'] = hls.loc['v*a*r', 's'] * 0.5
        hls.loc['v*a*r', 'l'] = hls.loc['v*a*r', 'l'] * 1.5
        for rhsMaskIdx in range(rhsMasks.shape[0]):
            rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            lumVals = np.linspace(0.3, 0.7, rhGroup.shape[1])
            for cIdx, cN in enumerate(rhGroup.columns):
                hls.loc[cN, :] = hls.loc['ens', :]
                hls.loc[cN, 'l'] = lumVals[cIdx]
        hls.loc['ground_truth', :] = hls.loc['prediction', :]
        hls.loc['ground_truth', 'l'] = hls.loc['prediction', 'l'] * 0.25
        primarySourcePalette = hls.apply(lambda x: pd.Series(colorsys.hls_to_rgb(*x), index=['r', 'g', 'b']), axis='columns')
        sourcePalette = primarySourcePalette.apply(lambda x: tuple(x), axis='columns')
        if pickingColors:
            sns.palplot(sourcePalette, size=sourcePalette.shape[0])
            palAx = plt.gca()
            for tIdx, tN in enumerate(sourcePalette.index):
                palAx.text(tIdx, .5, '{}'.format(tN))
        ########################################################################################
        factorPaletteDict = {}
        endoFactors = []
        for designFormula, row in designInfoDF.iterrows():
            for tN, factorIdx in row['designInfo'].term_name_slices.items():
                thisSrs = pd.Series({fN: tN for fN in row['designInfo'].column_names[factorIdx]})
                thisSrs.name = 'term'
                thisSrs.index.name = 'factor'
                endoFactors.append(thisSrs)
        factorPaletteDict['endo'] = pd.concat(endoFactors).to_frame(name='term').reset_index().drop_duplicates(subset='factor')
        exoFactors = []
        for (rhsMaskIdx, ensTemplate), row in histDesignInfoDF.iterrows():
            for tN, factorIdx in row['designInfo'].term_name_slices.items():
                thisSrs = pd.Series({fN: tN for fN in row['designInfo'].column_names[factorIdx]})
                thisSrs.name = 'term'
                thisSrs.index.name = 'factor'
                exoFactors.append(thisSrs)
        factorPaletteDict['exo'] = pd.concat(exoFactors).to_frame(name='term').reset_index().drop_duplicates(subset='factor')
        factorPalette = pd.concat(factorPaletteDict, names=['type', 'index'])
        ############# workaround inconsistent use of whitespace with patsy
        sourceTermLookup.reset_index(inplace=True)
        sourceTermLookup.loc[:, 'term'] = sourceTermLookup['term'].apply(lambda x: x.replace(' ', ''))
        sourceTermLookup.set_index(['type', 'term'], inplace=True)
        ######
        termPalette.loc[:, 'termNoWS'] = termPalette['term'].apply(lambda x: x.replace(' ', ''))
        termPalette.loc[:, 'source'] = termPalette['termNoWS'].map(sourceTermLookup.reset_index(level='type')['source'])
        termPalette = termPalette.sort_values('source', kind='mergesort').sort_index(kind='mergesort')
        #
        factorPalette.loc[:, 'termNoWS'] = factorPalette['term'].apply(lambda x: x.replace(' ', ''))
        factorPalette.loc[:, 'source'] = factorPalette['termNoWS'].map(sourceTermLookup.reset_index(level='type')['source'])
        factorPalette = factorPalette.sort_values('source', kind='mergesort').sort_index(kind='mergesort')
        ############
        termPalette.loc[:, 'color'] = termPalette['source'].map(sourcePalette)
        factorPalette.loc[:, 'color'] = factorPalette['source'].map(sourcePalette)
        #
        trialTypeOrder = ['train', 'work', 'test', 'validation']
        trialTypePalette = pd.Series(
            sns.color_palette('Paired', 12)[::-1][:len(trialTypeOrder)],
            index=trialTypeOrder)
        # not allowed to save plot options - processOrdinaryLeastSquaresTransferFunction does it now.
        '''
            sourceTermLookup.to_hdf(estimatorPath, 'sourceTermLookup')
            sourcePalette.to_hdf(estimatorPath, 'sourcePalette')
            termPalette.to_hdf(estimatorPath, 'termPalette')
            factorPalette.to_hdf(estimatorPath, 'factorPalette')
            trialTypePalette.to_hdf(estimatorPath, 'trialTypePalette')
            print('Loaded and saved plot options')'''
    ###
    if (not loadedProcessedScores) or arguments['forceReprocess']:
        scoresStack = pd.concat({
                'test': scoresDF['test_score'],
                'train': scoresDF['train_score']},
            names=['trialType']
            ).to_frame(name='score').reset_index()
        #
        lastFoldMask = (scoresStack['fold'] == cvIterator.n_splits)
        trainMask = (scoresStack['trialType'] == 'train')
        testMask = (scoresStack['trialType'] == 'test')
        #
        scoresStack.loc[:, 'foldType'] = ''
        scoresStack.loc[(trainMask & lastFoldMask), 'foldType'] = 'work'
        scoresStack.loc[(trainMask & (~lastFoldMask)), 'foldType'] = 'train'
        scoresStack.loc[(testMask & lastFoldMask), 'foldType'] = 'validation'
        scoresStack.loc[(testMask & (~lastFoldMask)), 'foldType'] = 'test'
        scoresStack.loc[:, 'dummyX'] = 0
        scoresStack.loc[:, 'design'] = scoresStack['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'designFormula'])
        scoresStack.loc[:, 'designAsLabel'] = scoresStack['design'].apply(lambda x: x.replace(' + ', ' +\n'))
        scoresStack.loc[:, 'fullDesign'] = scoresStack['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
        scoresStack.loc[:, 'fullDesignAsLabel'] = scoresStack['fullDesign'].apply(lambda x: x.replace(' + ', ' +\n'))
        scoresStack.to_hdf(estimatorPath, 'processedCVScores')
        llDict1 = {}
        aicDict1 = {}
        for scoreName, targetScores in scoresStack.groupby(['lhsMaskIdx', 'target', 'fold']):
            print('Calculating scores ({})'.format(scoreName))
            lhsMaskIdx, targetName, fold = scoreName
            designFormula = lhsMasksInfo.loc[lhsMaskIdx, 'designFormula']
            estimator = estimatorsDF.loc[idxSl[lhsMaskIdx, :, targetName, fold]].iloc[0]
            regressor = estimator.regressor_.steps[-1][1]
            K = regressor.results_.df_model
            thesePred = predDF.xs(targetName, level='target').xs(lhsMaskIdx, level='lhsMaskIdx').xs(fold, level='fold')
            llDict2 = {}
            aicDict2 = {}
            for name, predGroup in thesePred.groupby(['electrode', 'trialType']):
                llDict3 = dict()
                llDict3['llSat'] = regressor.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
                nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
                llDict3['llNull'] = regressor.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
                llDict3['llFull'] = regressor.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
                llDict2[name] = pd.Series(llDict3)
                aicDict2[name] = 2 * K - 2 * llDict3['llFull']
            for trialType, predGroup in thesePred.groupby('trialType'):
                llDict3 = dict()
                llDict3['llSat'] = regressor.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
                nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
                llDict3['llNull'] = regressor.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
                llDict3['llFull'] = regressor.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
                llDict2[('all', trialType)] = pd.Series(llDict3)
                aicDict2[('all', trialType)] = 2 * K - 2 * llDict3['llFull']
            llDict1[(lhsMaskIdx, designFormula, targetName, fold)] = pd.concat(llDict2, names=['electrode', 'trialType', 'llType'])
            aicSrs = pd.Series(aicDict2)
            aicSrs.index.names = ['electrode', 'trialType']
            aicDict1[(lhsMaskIdx, designFormula, targetName, fold)] = aicSrs
        llDF = pd.concat(llDict1, names=['lhsMaskIdx', 'design', 'target', 'fold', 'electrode', 'trialType', 'llType']).to_frame(name='ll')
        llDF.loc[:, 'fullFormulaDescr'] = llDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
        llDF.set_index('fullFormulaDescr', append=True, inplace=True)
        llDF.to_hdf(estimatorPath, 'processedLogLike')
        aicDF = pd.concat(aicDict1, names=['lhsMaskIdx', 'design', 'target', 'fold', 'electrode', 'trialType']).to_frame(name='aic')
        aicDF.loc[:, 'fullFormulaDescr'] = aicDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
        aicDF.set_index('fullFormulaDescr', append=True, inplace=True)
        aicDF.to_hdf(estimatorPath, 'processedAIC')
        #
        R2Per = llDF['ll'].groupby(['lhsMaskIdx', 'design', 'target', 'electrode', 'fold', 'trialType']).apply(tdr.getR2).to_frame(name='score')
        R2Per.to_hdf(estimatorPath, 'processedR2')
        #
        FUDEDict = {}
        FUDEStatsDict = {}
        ScoresDict = {}
        for modelIdx, modelToTest in modelsToTestDF.iterrows():
            testDesign = modelToTest['testDesign']
            refDesign = modelToTest['refDesign']
            ##
            theseFUDE = tdr.partialR2(llDF['ll'], refDesign=refDesign, testDesign=testDesign, designLevel='lhsMaskIdx')
            theseFUDEStats = tdr.correctedResampledPairedTTest(
                theseFUDE.xs('test', level='trialType'), y=0., groupBy=['target', 'electrode'], cvIterator=cvIterator)
            comparisonParams = tuple(modelToTest.loc[['testDesign', 'refDesign', 'testType', 'lagSpec', 'testHasEnsembleHistory']])
            FUDEDict[comparisonParams] = theseFUDE
            FUDEStatsDict[comparisonParams] = theseFUDEStats
            theseTestScores = tdr.partialR2(
                llDF['ll'], refDesign=None, testDesign=testDesign,
                designLevel='lhsMaskIdx').reset_index().rename(columns={'ll': 'test_score'})
            theseRefScores = tdr.partialR2(
                llDF['ll'], refDesign=None, testDesign=refDesign,
                designLevel='lhsMaskIdx').reset_index().rename(columns={'ll': 'ref_score'})
            assert theseTestScores.shape == theseRefScores.shape
            checkSameOn = ['target', 'fold', 'electrode']
            assert (theseRefScores.loc[:, checkSameOn] == theseTestScores.loc[:, checkSameOn]).all().all()
            theseTestScores.loc[:, 'ref_score'] = theseRefScores['ref_score'].to_numpy()
            ScoresDict[comparisonParams] = theseTestScores
        modelCompareFUDE = pd.concat(FUDEDict, names=['testLhsMaskIdx', 'refLhsMaskIdx', 'testType', 'lagSpec', 'testHasEnsembleHistory'])
        modelCompareFUDE.name = 'score'
        modelCompareFUDEStats = pd.concat(FUDEStatsDict, names=['testLhsMaskIdx', 'refLhsMaskIdx', 'testType', 'lagSpec', 'testHasEnsembleHistory'])
        modelCompareScores = pd.concat(ScoresDict, names=['testLhsMaskIdx', 'refLhsMaskIdx', 'testType', 'lagSpec', 'testHasEnsembleHistory'])
        modelCompareFUDE.to_hdf(estimatorPath, 'modelCompareFUDE')
        modelCompareFUDEStats.to_hdf(estimatorPath, 'modelCompareFUDEStats')
        modelCompareScores.to_hdf(estimatorPath, 'modelCompareScores')
        print('Loaded and saved scores and partial scores')
    #
    def drawUnityLine(g, ro, co, hu, dataSubset):
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset.iloc[:, 0].isna().all()))
        if not hasattr(g.axes[ro, co], 'axHasUnityLine'):
            g.axes[ro, co].axHasUnityLine = True
            if not emptySubset:
                currXLim, currYLim = g.axes[ro, co].get_xlim(), g.axes[ro, co].get_ylim()
                leftEdge = min(currXLim[0], currYLim[0])
                rightEdge = max(currXLim[1], currYLim[1])
                g.axes[ro, co].plot(
                    [leftEdge, rightEdge], [leftEdge, rightEdge], '-',
                    c=(0., 0., 0., 0.75), zorder=1.9)
                g.axes[ro, co].set_xlim(currXLim)
                g.axes[ro, co].set_ylim(currYLim)
        return

    def annotateWithPVal(g, ro, co, hu, dataSubset):
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset.iloc[:, 0].isna().all()))
        if not hasattr(g.axes[ro, co], 'axHasPValAnnotation'):
            g.axes[ro, co].axHasPValAnnotation = True
            if not emptySubset:
                nSig = dataSubset.loc[dataSubset['fold'] == 0., 'significant'].sum()
                nTot = dataSubset.loc[dataSubset['fold'] == 0., 'significant'].shape[0]
                messageStr = 'n = {}/{} significant'.format(nSig, nTot)
                g.axes[ro, co].text(
                    0.9, 1, messageStr, ha='right', va='top',
                    fontsize=snsRCParams['font.size'], transform=g.axes[ro, co].transAxes)
            return
    #
    pdfPath = os.path.join(
        figureOutputFolder,
        '{}_{}.pdf'.format(fullEstimatorName, 'partial_scores'))
    with PdfPages(pdfPath) as pdf:
        height, width = 3, 3
        aspect = width / height
        # maskSecondOrderTests = modelsToTestDF['testType'] == 'secondOrderInteractions'
        for testTypeName, modelsToTestGroup in modelsToTestDF.groupby('testType'):
            if 'captionStr' in modelsToTestGroup:
                titleText = modelsToTestGroup['captionStr'].iloc[0]
            else:
                titleText = 'partial R2 scores for {} compared to {}'.format(
                    modelsToTestGroup['testDesign'].iloc[0], modelsToTestGroup['refDesign'].iloc[0])
            if 'refCaption' in modelsToTestGroup:
                refCaption = modelsToTestGroup['refCaption'].iloc[0]
            else:
                refCaption = lhsMasksInfo.loc[modelsToTestGroup['refDesign'].iloc[0], 'fullFormulaDescr']
            if 'testCaption' in modelsToTestGroup:
                testCaption = modelsToTestGroup['testCaption'].iloc[0]
            else:
                testCaption = lhsMasksInfo.loc[modelsToTestGroup['testDesign'].iloc[0], 'fullFormulaDescr']
            plotFUDE = modelCompareFUDE.xs(testTypeName, level='testType').xs('all', level='electrode').reset_index()
            plotFUDE = plotFUDE.loc[plotFUDE['trialType'].isin(['test']), :]
            plotFUDEStats = modelCompareFUDEStats.xs(testTypeName, level='testType').xs('all', level='electrode').xs('hto1', level='lagSpec').reset_index()
            plotScores = modelCompareScores.loc[modelCompareScores['electrode'] == 'all'].xs(testTypeName, level='testType').xs('hto1', level='lagSpec').reset_index()
            plotScores = plotScores.loc[plotScores['trialType'].isin(['test']), :]
            #
            lookupBasedOn = ['testLhsMaskIdx', 'refLhsMaskIdx', 'target']
            lookupAt = pd.MultiIndex.from_frame(plotScores.loc[:, lookupBasedOn])
            lookupFrom = plotFUDEStats.loc[:, lookupBasedOn + ['p-val']].set_index(lookupBasedOn)['p-val']
            plotPVals = lookupFrom.loc[lookupAt]
            plotScores.loc[:, 'significant'] = (plotPVals < 0.01).to_numpy()
            ###
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            g = sns.catplot(
                data=plotFUDE, kind='box',
                y='score', x='target', hue='trialType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=height, aspect=aspect,
                sharey=True
                )
            g.set_xticklabels(rotation=30, ha='right')
            g.set_titles(template="data subset: {col_name}")
            g.suptitle(titleText)
            g.set_axis_labels('target signal', '{} R2 w.r.t. {}'.format(testCaption, refCaption))
            print('Saving {}\n to {}'.format(titleText, pdfPath))
            # g.axes.flat[0].set_ylim(allScoreQuantiles)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            tc = thisPalette.loc['test']
            signiPalette = {
                True: (tc[0], tc[1], tc[2], .75),
                False: (tc[0], tc[1], tc[2], .25)
                }
            #
            g = sns.relplot(
                data=plotScores, kind='scatter',
                y='test_score', x='ref_score',
                hue='significant',
                height=height, aspect=aspect,
                edgecolor=None,
                hue_order=[True, False],
                palette=signiPalette,
                style='target',
                # hue_order=thisPalette.index.to_list(),
                # palette=thisPalette.to_dict(),
                )
            g.set_axis_labels(refCaption, testCaption)
            g.set_titles(template="data subset: {col_name}")
            g.suptitle(titleText)
            plotProcFuns = [drawUnityLine, annotateWithPVal]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            print('Saving {}\n to {}'.format(titleText, pdfPath))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'r2'))
    with PdfPages(pdfPath) as pdf:
        height, width = 8, 12
        aspect = width / height
        plotAIC = aicDF.xs('all', level='electrode').reset_index()
        plotAIC.loc[:, 'fullDesignAsLabel'] = plotAIC['fullFormulaDescr'].apply(lambda x: x.replace(' + ', ' +\n'))
        plotAIC.loc[:, 'rhsMaskIdx'] = plotAIC['target'].map(scoresStack[['rhsMaskIdx', 'target']].drop_duplicates().set_index('target')['rhsMaskIdx'])
        for rhsMaskIdx, plotScores in scoresStack.groupby(['rhsMaskIdx']):
            rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['foldType'])]
            g = sns.catplot(
                data=plotScores, y='score', hue='foldType',
                x='fullDesignAsLabel', col='target',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect)
            g.suptitle('R2 (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            g = sns.catplot(
                data=plotAIC.loc[plotAIC['rhsMaskIdx'] == rhsMaskIdx, :], hue='trialType',
                y='aic',
                x='fullDesignAsLabel', col='target',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                kind='box', height=height, aspect=aspect)
            g.suptitle('AIC (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
            g.set_xticklabels(rotation=-30, ha='left')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

    height, width = 2, 4
    trialTypeToPlot = 'test'
    aspect = width / height
    commonOpts = dict(
        )
    groupPagesBy = ['rhsMaskIdx', 'lhsMaskIdx']
    groupSubPagesBy = ['trialType', 'foldType', 'electrode', 'trialRateInHz', 'target']
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'reconstructions'))
    with PdfPages(pdfPath) as pdf:
        for name0, predGroup0 in predDF.groupby(groupPagesBy):
            nmLk0 = {key: value for key, value in zip(groupPagesBy, name0)} # name lookup
            nmLk0['design'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'designFormula']
            nmLk0['fullFormulaDescr'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'fullFormulaDescr']
            if not (nmLk0['lhsMaskIdx'] in [19, 28, 29]):
                continue
            scoreMasks = [
                scoresStack[cN] == nmLk0[cN]
                for cN in groupPagesBy]
            plotScores = scoresStack.loc[np.logical_and.reduce(scoreMasks), :]
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['foldType'])]
            g = sns.catplot(
                data=plotScores, hue='foldType',
                x='target', y='score',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=height, aspect=aspect,
                kind='box')
            g.set_xticklabels(rotation=-30, ha='left')
            g.suptitle('R^2 of model {fullFormulaDescr}'.format(**nmLk0))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ####
            for name1, predGroup1 in predGroup0.groupby(groupSubPagesBy):
                nmLk1 = {key: value for key, value in zip(groupSubPagesBy, name1)}  # name lookup
                if not (nmLk1['target'] in ['fa_ta_all001']):
                    continue
                nmLk0.update(nmLk1)
                if nmLk0['trialType'] != trialTypeToPlot:
                    continue
                if nmLk0['trialRateInHz'] < 100:
                    continue
                plotDF = predGroup1.stack().to_frame(name='signal').reset_index()
                plotDF.loc[:, 'predType'] = 'component'
                plotDF.loc[plotDF['term'] == 'ground_truth', 'predType'] = 'ground_truth'
                plotDF.loc[plotDF['term'] == 'prediction', 'predType'] = 'prediction'
                plotDF.loc[:, 'kinCondition'] = plotDF.loc[:, ['pedalMovementCat', 'pedalDirection']].apply(lambda x: tuple(x), axis='columns').map(kinConditionLookup)
                plotDF.loc[:, 'stimCondition'] = plotDF.loc[:, ['electrode', 'trialRateInHz']].apply(lambda x: tuple(x), axis='columns').map(stimConditionLookup)
                plotDF.loc[:, 'fullFormulaDescr'] = plotDF['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
                kinOrder = kinConditionLookup.loc[kinConditionLookup.isin(plotDF['kinCondition'])].to_list()
                stimOrder = stimConditionLookup.loc[stimConditionLookup.isin(plotDF['stimCondition'])].to_list()
                thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
                theseColors = thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color'].to_dict()
                theseLegendUpdates = thisTermPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict()
                predictionLineStyleDF = pd.DataFrame([
                    {
                        'factor': .5,
                        'component': .5,
                        'prediction': 1.,
                        'ground_truth': 1.,
                        },
                    {
                        'factor': (1, 1),
                        'component': (3, 1),
                        'prediction': (2, 1),
                        'ground_truth': (8, 0),
                        }
                ], index=['sizes', 'dashes'])
                #
                def plotRoutine(inputDF, inputColorPalette, inputLegendUpdates, hueName=None):
                    g = sns.relplot(
                        data=inputDF,
                        col='trialAmplitude', row='kinCondition',
                        row_order=kinOrder,
                        x='bin', y='signal', hue=hueName,
                        height=height, aspect=aspect, palette=inputColorPalette,
                        kind='line', errorbar='sd',
                        size='predType', sizes=predictionLineStyleDF.loc['sizes', :].to_dict(),
                        style='predType', dashes=predictionLineStyleDF.loc['dashes', :].to_dict(),
                        style_order=predictionLineStyleDF.columns,
                        facet_kws=dict(margin_titles=True),
                        )
                    g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
                    titleText = 'model {fullFormulaDescr}\n{target}, electrode {electrode} rate {trialRateInHz} Hz ({trialType})'.format(
                        **nmLk0)
                    print('Saving plot of {}...'.format(titleText))
                    g.suptitle(titleText)
                    asp.reformatFacetGridLegend(
                        g, titleOverrides={},
                        contentOverrides=inputLegendUpdates,
                        styleOpts=styleOpts)
                    g.tight_layout(pad=styleOpts['tight_layout.pad'])
                    pdf.savefig(
                        bbox_inches='tight',
                        # bbox_extra_artists=[figTitle, g.legend]
                        )
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
                    return g
                plotRoutine(plotDF.loc[plotDF['predType'].isin(['ground_truth', 'prediction']), :], theseColors, theseLegendUpdates, hueName='term')
                plotRoutine(plotDF.loc[plotDF['predType'] == 'component', :], theseColors, theseLegendUpdates, hueName='term')
                ####
                designFormula = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'designFormula']
                if designFormula != 'NULL':
                    designInfo = designInfoDict[designFormula]
                    formulaIdx = lOfDesignFormulas.index(designFormula)
                    designDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'designs/formula_{}'.format(formulaIdx))
                    for key, value in nmLk1.items():
                        if key in designDF.index.names:
                            designDF = designDF.xs(value, level=key, drop_level=False)
                    designTermNames = designInfo.term_names
                    plotDesign = designDF.stack().to_frame(name='signal').reset_index()
                    plotDesign = plotDesign.loc[plotDesign['bin'] > (plotDesign['bin'].min() + burnInPeriod), :]
                    plotDesign.loc[:, 'predType'] = 'factor'
                    plotDesign.loc[:, 'kinCondition'] = plotDesign.loc[:, ['pedalMovementCat', 'pedalDirection']].apply(lambda x: tuple(x), axis='columns').map(kinConditionLookup)
                    thisFactorPalette = factorPalette.loc[factorPalette['factor'].isin(plotDesign['factor']), :]
                    plotDesign.loc[:, 'term'] = plotDesign['factor'].map(thisFactorPalette[['factor', 'term']].set_index('factor')['term'])
                    theseColorsFactor = thisFactorPalette.loc[:, ['factor', 'color']].set_index('factor')['color'].to_dict()
                    theseLegendUpdatesFactor = thisFactorPalette.loc[:, ['factor', 'source']].set_index('factor')['source'].to_dict()
                    coefValues = coefDF.loc[idxSl[nmLk0['lhsMaskIdx'], nmLk0['design'], nmLk0['rhsMaskIdx'], nmLk0['target'], :]].reset_index(name='coef')
                    coefValues.loc[:, 'term'] = coefValues['factor'].map(factorPalette[['factor', 'term']].set_index('factor')['term'])
                    meanCoefs = coefValues.groupby('term').mean()['coef']
                    for termName, plotDesignTerm in plotDesign.groupby('term'):
                        if meanCoefs[termName] != 0:
                            plotRoutine(plotDesignTerm, theseColorsFactor, theseLegendUpdatesFactor, hueName='factor')
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')
    '''
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'regressors'))
    with PdfPages(pdfPath) as pdf:
        for (lhsMaskIdx, designFormula), row in designInfoDF.iterrows():
            #
            designInfo = row['designInfo']
            designDF = allDesignDF.xs(lhsMaskIdx, level='lhsMaskIdx').xs(designFormula, level='design').loc[:, designInfo.column_names]
            factorNames = designDF.columns.to_frame().reset_index(drop=True)
            factorNames.loc[:, 'term'] = np.nan
            termDF = pd.DataFrame(
                np.nan, index=designDF.index,
                columns=designInfo.term_names)
            termDF.columns.name = 'term'
            for termName, termSlice in designInfo.term_name_slices.items():
                termDF.loc[:, termName] = designDF.iloc[:, termSlice].sum(axis='columns')
                factorNames.iloc[termSlice, 1] = termName
            #
            trialInfo = termDF.index.to_frame().reset_index(drop=True)
            stimCondition = pd.Series(np.nan, index=trialInfo.index)
            for name, group in trialInfo.groupby(['electrode', 'trialRateInHz']):
                stimCondition.loc[group.index] = '{} {}'.format(*name)
            trialInfo.loc[:, 'stimCondition'] = stimCondition
            kinCondition = pd.Series(np.nan, index=trialInfo.index)
            for name, group in trialInfo.groupby(['pedalMovementCat', 'pedalDirection']):
                kinCondition.loc[group.index] = '{} {}'.format(*name)
            trialInfo.loc[:, 'kinCondition'] = kinCondition
            #
            #
            lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
            lhsMaskParams = {k: v for k, v in zip(lhsMask.index.names, lhsMask.name)}
            plotDF = lhsDF.loc[:, lhsMask].copy()
            plotDF.columns = plotDF.columns.get_level_values('feature')
            plotDF.drop(columns=['e'], inplace=True)
            plotDF.index = pd.MultiIndex.from_frame(trialInfo)
            plotDF = plotDF.stack().to_frame(name='signal')
            plotDF.reset_index(inplace=True)
            g = sns.relplot(
                row='kinCondition', col='stimCondition',
                x='bin', y='signal', hue='feature',
                kind='line', errorbar='se', data=plotDF
                )
            g.fig.suptitle('Features for model {}'.format(designFormula))
            leg = g._legend
            if leg is not None:
                t = leg.get_title()
                tContent = t.get_text()
                # if tContent in titleOverrides:
                #     t.set_text(titleOverrides[tContent])
                # for t in leg.texts:
                #     tContent = t.get_text()
                #     if tContent in titleOverrides:
                #         t.set_text(titleOverrides[tContent])
                #     elif tContent in emgNL.index:
                #         t.set_text('{}'.format(emgNL[tContent]))
                for l in leg.get_lines():
                    # l.set_lw(2 * l.get_lw())
                    l.set_lw(styleOpts['legend.lw'])
            g.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            designDF.index = pd.MultiIndex.from_frame(trialInfo)
            designDF = designDF.stack().to_frame(name='signal')
            designDF.reset_index(inplace=True)
            factorPalette = factorNames.set_index('factor')['term'].map(termPalette)
            g = sns.relplot(
                row='kinCondition', col='stimCondition',
                x='bin', y='signal', hue='factor',
                kind='line', errorbar='se', data=designDF,
                palette=factorPalette.to_dict()
                )
            g.fig.suptitle('Terms for model {}'.format(designFormula))
            leg = g._legend
            if leg is not None:
                t = leg.get_title()
                tContent = t.get_text()
                # if tContent in titleOverrides:
                #     t.set_text(titleOverrides[tContent])
                # for t in leg.texts:
                #     tContent = t.get_text()
                #     if tContent in titleOverrides:
                #         t.set_text(titleOverrides[tContent])
                #     elif tContent in emgNL.index:
                #         t.set_text('{}'.format(emgNL[tContent]))
                for l in leg.get_lines():
                    # l.set_lw(2 * l.get_lw())
                    l.set_lw(styleOpts['legend.lw'])
            g.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            termDF.index = pd.MultiIndex.from_frame(trialInfo)
            termDF = termDF.stack().to_frame(name='signal')
            termDF.reset_index(inplace=True)
            g = sns.relplot(
                row='kinCondition', col='stimCondition',
                x='bin', y='signal', hue='term',
                kind='line', errorbar='se', data=termDF,
                palette=termPalette.to_dict()
                )
            g.fig.suptitle('Terms for model {}'.format(designFormula))
            leg = g._legend
            if leg is not None:
                t = leg.get_title()
                tContent = t.get_text()
                # if tContent in titleOverrides:
                #     t.set_text(titleOverrides[tContent])
                # for t in leg.texts:
                #     tContent = t.get_text()
                #     if tContent in titleOverrides:
                #         t.set_text(titleOverrides[tContent])
                #     elif tContent in emgNL.index:
                #         t.set_text('{}'.format(emgNL[tContent]))
                for l in leg.get_lines():
                    # l.set_lw(2 * l.get_lw())
                    l.set_lw(styleOpts['legend.lw'])
            g.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    '''
    ####