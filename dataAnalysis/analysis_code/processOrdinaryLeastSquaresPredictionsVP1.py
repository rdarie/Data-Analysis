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
    --memoryEfficientLoad                    print diagnostics? [default: False]
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
from docopt import docopt
from copy import deepcopy
import colorsys
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
for arg in sys.argv:
    print(arg)
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
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    # if debugging in a console:
    '''
    
    consoleDebugging = True
    if consoleDebugging:
        arguments = {
            'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_ra', 'plotting': True,
            'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
            'verbose': '1', 'debugging': False, 'estimatorName': 'enr_fa', 'forceReprocess': True,
            'blockIdx': '2', 'exp': 'exp202101281100'}
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
    ##
    ################ collect estimators and scores
    #
    trialTypeToPlot = 'test'
    #
    predIndexNames = None
    #
    memoryEfficientLoad = True
    if memoryEfficientLoad:
        predDF = None
        scoresStack = None
    else:
        # predDF = None
        predList = []
        scoresStackList = []
    #
    if processSlurmTaskCount is not None:
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
        allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
        for modelIdx in range(processSlurmTaskCount):
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(modelIdx))
            prf.print_memory_usage('Loading predictions from {}'.format(thisEstimatorPath))
            with pd.HDFStore(thisEstimatorPath) as store:
                thisPred = pd.read_hdf(store, 'predictions')
                if predIndexNames is None:
                    predIndexNames = thisPred.index.names
                thisPred.reset_index(inplace=True)
                # pdb.set_trace()
                maskForPlot = (
                    (thisPred['lhsMaskIdx'].isin([19, 28, 29])) &
                    (thisPred['target'] == 'fa_all001') &
                    (thisPred['trialType'] == trialTypeToPlot) &
                    (thisPred['trialRateInHz'] > 50)
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
                    ##
                    thisScoresStack = pd.read_hdf(store, 'processedCVScores')
                    # pdb.set_trace()
                    maskForPlotScore = (
                            (thisScoresStack['lhsMaskIdx'].isin([19, 28, 29])) &
                            (thisScoresStack['target'] == 'fa_all001') &
                            (thisScoresStack['foldType'] == trialTypeToPlot)
                        )
                    print('these scoresStack, thisScoresStack.shape = {}'.format(thisScoresStack.shape))
                    if memoryEfficientLoad:
                        if scoresStack is None:
                            scoresStack = thisScoresStack
                        else:
                            scoresStack = scoresStack.append(thisScoresStack)
                    else:
                        scoresStackList.append(thisScoresStack)
    #  else:
    #      print('Loading predictions from {}'.format(estimatorPath))
    #      thisPred = pd.read_hdf(estimatorPath, 'predictions')
    #      predList.append(thisPred)
    ###
    prf.print_memory_usage('concatenating predictions from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        predDF = pd.concat(predList, copy=False)
        del predList
    predDF.set_index(predIndexNames, inplace=True)
    print('all predictions, predDF.shape = {}'.format(predDF.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating predictions from .h5 array')
    ###
    prf.print_memory_usage('concatenating scoresStack from .h5 array')
    gc.collect()
    if not memoryEfficientLoad:
        scoresStack = pd.concat(scoresStackList, copy=False)
        del scoresStackList
    print('all scoresStack, scoresStack.shape = {}'.format(scoresStack.shape))
    gc.collect()
    prf.print_memory_usage('done concatenating scoresStack from .h5 array')
    #
    estimatorsDict = {}
    # scoresDict = {}
    for rowIdx, row in allTargetsDF.iterrows():
        lhsMaskIdx, rhsMaskIdx, targetName = row.name
        if processSlurmTaskCount is not None:
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
            print('Loading data from {}'.format(thisEstimatorPath))
        else:
            thisEstimatorPath = estimatorPath
        # scoresDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
        #     thisEstimatorPath,
        #     'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
        #         lhsMaskIdx, rhsMaskIdx, targetName
        #         ))
        estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
            thisEstimatorPath,
            'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
                lhsMaskIdx, rhsMaskIdx, targetName
                ))
    prf.print_memory_usage('concatenating estimators from .h5 array')
    estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    del estimatorsDict
    prf.print_memory_usage('done concatenating estimators from .h5 array')
    # prf.print_memory_usage('concatenating scores from .h5 array')
    # scoresDF = pd.concat(scoresDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
    # del scoresDict
    # prf.print_memory_usage('done concatenating scores from .h5 array')
    # gc.collect()
    #
    with pd.HDFStore(estimatorPath) as store:
        modelsToTestDF = pd.read_hdf(store, 'modelsToTest')
        rhsMasksInfo = pd.read_hdf(store, 'rhsMasksInfo')
        lhsMasksInfo = pd.read_hdf(store, 'lhsMasksInfo')
        #
        stimConditionLookup = pd.read_hdf(store, 'stimConditionLookup')
        kinConditionLookup = pd.read_hdf(store, 'kinConditionLookup')
        coefDF = pd.read_hdf(store, 'coefficients')
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
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
    height, width = 2, 4
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
            # if not (nmLk0['lhsMaskIdx'] in [19, 28, 29]):
            #     continue
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
                nmLk0.update(nmLk1)
                # if not (nmLk0['target'] in ['fa_ta_all001']):
                #     continue
                # if nmLk0['trialType'] != trialTypeToPlot:
                #     continue
                # if nmLk0['trialRateInHz'] < 100:
                #     continue
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