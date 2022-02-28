"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --oneTrialOnly                           show plots? [default: False]
    --verbose=verbose                        print diagnostics?
    --debugging                              print diagnostics? [default: False]
    --memoryEfficientLoad                    print diagnostics? [default: False]
    --forceReprocess                         print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
    --eraMethod=eraMethod                    append a name to the resulting blocks? [default: ERA]
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

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_rd', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'pls_select_scaled', 'forceReprocess': True,
        'blockIdx': '2', 'exp': 'exp202101271100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.plotting.aligned_signal_plots as asp
from dataAnalysis.custom_transformers.tdr import reconstructionR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
from datetime import datetime
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
import colorsys
for arg in sys.argv:
    print(arg)
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .2,
        'lines.markersize': .4,
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
    'figure.titlesize': 9
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
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')

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
    transferFuncPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_{}_tf.h5'.format(arguments['eraMethod'])
        )

    for hIdx, histOpts in enumerate(addEndogHistoryTerms):
        locals().update({'enhto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    for hIdx, histOpts in enumerate(addExogHistoryTerms):
        locals().update({'exhto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
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
    allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets').xs(arguments['estimatorName'], level='regressorName')
    # allTargetsPLS = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargetsPLS')
    # allTargetsPLS.set_index(['lhsMaskIdx', 'rhsMaskIdx'], inplace=True)
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    ##
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    #
    stimConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'stimConditionLookup')
    kinConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'kinConditionLookup')
    ################ collect estimators and scores
    scoresStack = pd.read_hdf(estimatorPath, 'processedScores')
    #
    trialTypeToPlot = 'test'
    #
    predIndexNames = None
    R2PerIndexNames = None
    #
    memoryEfficientLoad = True
    if memoryEfficientLoad:
        predDF = None
        R2Per = None
        ccDF = None
        inputDrivenDF = None
        oskDF = None
    else:
        # predDF = None
        predList = []
        R2PerList = []
        ccList = []
        inputDrivenList = []
        oskList = []
    #
    if processSlurmTaskCount is not None:
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
        allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
        lhsTargetMask = allTargetsDF.index.get_level_values('lhsMaskIdx').isin(lhsMasksOfInterest['plotPredictions'])
        jobsToLoad = allTargetsDF.loc[lhsTargetMask, 'parentProcess'].unique()
        #
        for modelIdx in range(processSlurmTaskCount):
            if modelIdx not in jobsToLoad:
                continue
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(modelIdx))
            prf.print_memory_usage('Loading predictions from {}'.format(thisEstimatorPath))
            with pd.HDFStore(thisEstimatorPath) as store:
                thisPred = pd.read_hdf(store, 'predictions')
                if predIndexNames is None:
                    predIndexNames = thisPred.index.names
                thisPred.reset_index(inplace=True)
                maskForPlot = (
                    (thisPred['lhsMaskIdx'].isin(lhsMasksOfInterest['plotPredictions'])) &
                    (thisPred['trialType'] == trialTypeToPlot) &
                    (thisPred['trialRateInHz'] != 50)
                    )
                if maskForPlot.any():
                    thisPred = thisPred.loc[maskForPlot, :]
                    print('these predictions, thisPred.shape = {}'.format(thisPred.shape))
                    if memoryEfficientLoad:
                        if predDF is None:
                            predDF = thisPred
                        else:
                            predDF = predDF.append(thisPred)
                    else:
                        predList.append(thisPred)
        hasAnyInputDriven = False
        for tfIdx in range(processSlurmTaskCount):
            # load transfer funs if they exist
            thisTFPath = transferFuncPath.replace('_tf.h5', '_{}_tf.h5'.format(tfIdx))
            if os.path.exists(thisTFPath):
                thisOSK = pd.read_hdf(thisTFPath, 'oneStepKalman')
                thisInputDriven = pd.read_hdf(thisTFPath, 'inputDriven')
                thisIDTrialInfo = thisInputDriven.index.to_frame().reset_index(drop=True)
                maskForPlot = (
                    (thisIDTrialInfo['lhsMaskIdx'].isin(lhsMasksOfInterest['plotPredictions'])) &
                    # (thisIDTrialInfo['trialType'] == trialTypeToPlot) &
                    (thisIDTrialInfo['trialRateInHz'] != 50)
                    ).to_numpy()
                if maskForPlot.any():
                    thisInputDriven = thisInputDriven.loc[maskForPlot, :]
                    thisOSK = thisOSK.loc[maskForPlot, :]
                    print('these predictions, thisInputDriven.shape = {}'.format(thisInputDriven.shape))
                    if memoryEfficientLoad:
                        if inputDrivenDF is None:
                            inputDrivenDF = thisInputDriven
                            oskDF = thisOSK
                        else:
                            inputDrivenDF = inputDrivenDF.append(thisInputDriven)
                            oskDF = oskDF.append(thisOSK)
                    else:
                        inputDrivenList.append(thisInputDriven)
                        oskList.append(thisOSK)
                    hasAnyInputDriven = True
    ### else
    ###    print('Loading predictions from {}'.format(estimatorPath))
    ###    thisPred = pd.read_hdf(estimatorPath, 'predictions')
    ###    predList.append(thisPred)
    ###
    prf.print_memory_usage('concatenating predictions from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        predDF = pd.concat(predList, copy=False)
        del predList
    ##
    if hasAnyInputDriven:
        if not memoryEfficientLoad:
            inputDrivenDF = pd.concat(inputDrivenList, copy=False)
            oskDF = pd.concat(oskList, copy=False)
            del inputDrivenList, oskList
        idTrialInfo = inputDrivenDF.index.to_frame().reset_index(drop=True)
        foldsUID = idTrialInfo.apply(lambda x: (x['fold'], x['trialUID'], ), axis='columns')
        typeLookup = predDF.loc[:, ['fold', 'foldType', 'trialType', 'trialUID']].drop_duplicates().set_index(['fold', 'trialUID'])
        idTrialInfo.loc[:, 'foldType'] = foldsUID.map(typeLookup['foldType'])
        idTrialInfo.loc[:, 'trialType'] = foldsUID.map(typeLookup['trialType'])
        #
        nanMask = idTrialInfo.isna().any(axis='columns').to_numpy()
        inputDrivenDF = inputDrivenDF.loc[~nanMask, :]
        oskDF = oskDF.loc[~nanMask, :]
        idTrialInfo = idTrialInfo.loc[~nanMask, :]
        #
        inputDrivenDF.index = pd.MultiIndex.from_frame(idTrialInfo)
        inputDrivenDF = inputDrivenDF.stack().to_frame(name='inputDriven')
        inputDrivenDF.reset_index(inplace=True)
        inputDrivenDF = inputDrivenDF.loc[:, predIndexNames + ['inputDriven']]
        #
        oskDF.index = pd.MultiIndex.from_frame(idTrialInfo)
        oskDF = oskDF.stack().to_frame(name='oneStepKalman')
        oskDF.reset_index(inplace=True)
        oskDF = oskDF.loc[:, predIndexNames + ['oneStepKalman']]
        # inputDrivenDF.loc[:, predIndexNames]
        # predDF.loc[:, predIndexNames]
        inputDrivenDF.set_index(predIndexNames, inplace=True)
        oskDF.set_index(predIndexNames, inplace=True)
        predDF.set_index(predIndexNames, inplace=True)
        predDF = pd.concat([predDF, inputDrivenDF.loc[predDF.index, :], oskDF.loc[predDF.index, :]], axis='columns')
        predDF.columns.name = 'term'
        del inputDrivenDF
    else:
        predDF.set_index(predIndexNames, inplace=True)
    ######
    if arguments['oneTrialOnly']:
        tInfo = predDF.index.to_frame().reset_index(drop=True)
        #
        preMask = (tInfo['fold'] == 0).to_numpy()
        tInfo = tInfo.loc[preMask, :]
        predDF = predDF.loc[preMask, :]
        duplicatedMask = ~tInfo.duplicated(subset=['lhsMaskIdx', 'bin', 'target', 'trialType', 'conditionUID']).to_numpy()
        predDF = predDF.loc[duplicatedMask, :]
        tInfo = tInfo.loc[duplicatedMask, :]
        otherRelPlotKWArgs = dict(estimator=None, errorbar=None, units='trialUID', facet_kws=dict(margin_titles=True, sharey=False))
    else:
        otherRelPlotKWArgs = dict(errorbar='se', facet_kws=dict(margin_titles=True, sharey=True))
    print('all predictions, predDF.shape = {}'.format(predDF.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating predictions from .h5 array')
    #
    # inputDrivenDF
    estimatorsDict = {}
    # scoresDict = {}
    for rowIdx, row in allTargetsDF.iterrows():
        lhsMaskIdx, rhsMaskIdx, targetName = row.name
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        freqBandName = rhsMaskParams['freqBandName']
        if processSlurmTaskCount is not None:
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
            # print('Loading data from {}'.format(thisEstimatorPath))
        else:
            thisEstimatorPath = estimatorPath
        '''estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            thisEstimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))'''
        thisEstimatorJBPath = os.path.join(
            thisEstimatorPath.replace('.h5', ''),
            'cv_estimators__lhsMask_{}__rhsMask_{}__{}.joblib'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
            ))
        thisEstimatorJBDict = jb.load(thisEstimatorJBPath)
        thisEstimatorJB = pd.Series(thisEstimatorJBDict)
        thisEstimatorJB.index.name = 'fold'
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = thisEstimatorJB
    prf.print_memory_usage('concatenating estimators from .h5 array')
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    del estimatorsDict
    prf.print_memory_usage('done concatenating estimators from .h5 array')
    with pd.HDFStore(estimatorPath) as store:
        coefDF = pd.read_hdf(store, 'coefficients')
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
        predictionLineStyleDF = pd.read_hdf(store, 'termLineStyleDF')
    targetRHSLookup = (
        allTargetsDF.index.to_frame()
            .reset_index(drop=True)[['rhsMaskIdx', 'target']]
            .drop_duplicates().set_index('target')['rhsMaskIdx']
        )
    scoresStack.loc[:, 'fullFormulaDescr'] = scoresStack['lhsMaskIdx'].map(
        lhsMasksInfo['fullFormulaDescr']).to_numpy()
    scoresStack.loc[:, 'fullFormulaAsLabel'] = scoresStack['fullFormulaDescr'].apply(lambda x: x.replace(' + ', ' +\n'))
    rankingMask = (~scoresStack['fullDesign'].str.contains('self')) & (scoresStack['trialType'] == 'test')
    scoreRanking = scoresStack.loc[rankingMask, :].groupby('target').mean()['cc'].dropna().sort_values()
    featsToPlot = [scoreRanking.index[j] for j in [0, -1]]
    # featsToPlot = (
    #     scoresStack.loc[scoresStack['fullFormulaDescr'].str.contains('NULL'), :].groupby('target').mean()['score'].idxmax(),
    #     scoresStack.loc[scoresStack['fullFormulaDescr'].str.contains('NULL'), :].groupby('target').mean()['score'].idxmin())
    # scoresStack.loc[:, 'score'] = np.sqrt(scoresStack['score'])
    height, width = 1.5, 3
    aspect = width / height
    commonOpts = dict(
        )
    groupPagesBy = ['rhsMaskIdx', 'lhsMaskIdx']
    groupSubPagesBy = ['trialType', 'electrode', 'trialRateInHz', 'target']
    # for name, group in tInfo.groupby(groupPagesBy + groupSubPagesBy + ['trialAmplitude', 'pedalMovementCat']): break
    plotSingleTrials = False
    if plotSingleTrials:
        predDF = predDF.xs(0, level='conditionUID')
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, arguments['estimatorName'], 'reconstructions_single_trial'))
    else:
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, arguments['estimatorName'], 'reconstructions'))
    with PdfPages(pdfPath) as pdf:
        for name0, predGroup0 in predDF.groupby(groupPagesBy):
            nmLk0 = {key: value for key, value in zip(groupPagesBy, name0)}  # name lookup
            nmLk0['design'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'designFormula']
            nmLk0['fullFormulaDescr'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'fullFormulaDescr']
            nmLk0['fullFormulaAsLabel'] = nmLk0['fullFormulaDescr'].replace(' + ', ' +\n')
            if not (nmLk0['lhsMaskIdx'] in lhsMasksOfInterest['plotPredictions']):
                continue
            scoreMasks = [
                scoresStack[cN] == nmLk0[cN]
                for cN in groupPagesBy]
            plotScores = scoresStack.loc[np.logical_and.reduce(scoreMasks), :]
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            g = sns.catplot(
                data=plotScores, hue='trialType',
                x='target', y='cc',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=3, aspect=aspect,
                kind='box', whis=np.inf)
            g.set_xticklabels(rotation=-30, ha='left')
            g.suptitle('CC of model\n{fullFormulaAsLabel}'.format(**nmLk0))
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ####
            for name1, predGroup1 in predGroup0.groupby(groupSubPagesBy):
                nmLk1 = {key: value for key, value in zip(groupSubPagesBy, name1)}  # name lookup
                nmLk0.update(nmLk1)
                #
                nmLk0['averageCC'] = scoreRanking[nmLk0['target']]
                if not (nmLk0['target'] in featsToPlot):
                    continue
                plotDF = predGroup1.stack().to_frame(name='signal').reset_index()
                plotDF.loc[:, 'predType'] = 'component'
                plotDF.loc[plotDF['term'] == 'ground_truth', 'predType'] = 'ground_truth'
                plotDF.loc[plotDF['term'] == 'prediction', 'predType'] = 'prediction'
                plotDF.loc[plotDF['term'] == 'residuals', 'predType'] = 'residuals'
                plotDF.loc[plotDF['term'] == 'inputDriven', 'predType'] = 'inputDriven'
                plotDF.loc[plotDF['term'] == 'oneStepKalman', 'predType'] = 'oneStepKalman'
                plotDF.loc[:, 'kinCondition'] = plotDF.loc[:, ['pedalMovementCat', 'pedalDirection']].apply(lambda x: tuple(x), axis='columns').map(kinConditionLookup)
                plotDF.loc[:, 'stimCondition'] = plotDF.loc[:, ['electrode', 'trialRateInHz']].apply(lambda x: tuple(x), axis='columns').map(stimConditionLookup)
                plotDF.loc[:, 'fullFormulaDescr'] = plotDF['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
                kinOrder = kinConditionLookup.loc[kinConditionLookup.isin(plotDF['kinCondition'])].to_list()
                stimOrder = stimConditionLookup.loc[stimConditionLookup.isin(plotDF['stimCondition'])].to_list()
                thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
                theseColors = thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color'].to_dict()
                theseLegendUpdates = thisTermPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict()
                #
                def plotRoutine(
                        inputDF=None, inputColorPalette=None, inputLegendUpdates=None,
                        rowOrder=None, hueName=None, axLims=None, relPlotKWArgs={}):
                    gg = sns.relplot(
                        data=inputDF.query('bin < 1.0'),
                        col='trialAmplitude', row='kinCondition',
                        row_order=rowOrder,
                        x='bin', y='signal', hue=hueName,
                        height=height, aspect=aspect, palette=inputColorPalette,
                        kind='line',
                        size='predType', sizes=predictionLineStyleDF.loc['sizes', :].to_dict(),
                        style='predType', dashes=predictionLineStyleDF.loc['dashes', :].to_dict(),
                        style_order=predictionLineStyleDF.columns,
                        **relPlotKWArgs,
                        )
                    gg.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
                    titleText = 'model {fullFormulaAsLabel}\n{target} (ave. CC: {averageCC:.2f}), electrode {electrode} rate {trialRateInHz} Hz ({trialType})'.format(**nmLk0)
                    print('Saving plot of {}...'.format(titleText))
                    gg.suptitle(titleText)
                    asp.reformatFacetGridLegend(
                        gg, titleOverrides={},
                        contentOverrides=inputLegendUpdates,
                        styleOpts=styleOpts)
                    currAxLims = {
                        'x': gg.axes.flatten()[0].get_xlim(),
                        'y': gg.axes.flatten()[0].get_ylim()
                        }
                    if axLims is not None:
                        if 'x' in axLims:
                            gg.axes.flatten()[0].set_xlim(axLims['x'])
                        if 'y' in axLims:
                            gg.axes.flatten()[0].set_ylim(axLims['y'])
                    gg.tight_layout(pad=styleOpts['tight_layout.pad'])
                    pdf.savefig(
                        bbox_inches='tight',
                        # bbox_extra_artists=[figTitle, g.legend]
                        )
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
                    return gg, currAxLims
                print('ground truth and prediction')
                g, predAxLims = plotRoutine(
                        plotDF.loc[plotDF['predType'].isin(['ground_truth', 'prediction', 'inputDriven', 'oneStepKalman']), :],
                        theseColors, theseLegendUpdates, hueName='term', relPlotKWArgs=otherRelPlotKWArgs)
                # print('residuals')
                #
                # g, _ = plotRoutine(
                #         plotDF.loc[plotDF['predType'].isin(['residuals']), :],
                #         theseColors, theseLegendUpdates, hueName='term',
                #         axLims=predAxLims)
                # print('components')
                # g, _ = plotRoutine(
                #         plotDF.loc[plotDF['predType'] == 'component', :],
                #         theseColors, theseLegendUpdates, hueName='term',
                #         axLims=predAxLims)
                ####
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')