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
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
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
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
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
    'figure.titlesize': 7,
    'mathtext.default': 'regular',
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
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
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    #
    estimatorsDict = {}
    scoresDict = {}
    if processSlurmTaskCount is not None:
        slurmGroupSize  = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
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
        iRsExist = (
            ('impulseResponsePerTerm' in store) &
            ('impulseResponsePerFactor' in store)
            )
        if iRsExist:
            iRPerTerm = pd.read_hdf(store, 'impulseResponsePerTerm')
            iRPerFactor = pd.read_hdf(store, 'impulseResponsePerFactor')
            stimConditionLookupIR = pd.read_hdf(store, 'impulseResponseStimConditionLookup')
            kinConditionLookupIR = pd.read_hdf(store, 'impulseResponseKinConditionLookup')
            loadedIR = True
        else:
            loadedIR = False
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
    ###
    trialInfo = lhsDF.index.to_frame().reset_index(drop=True)
    stimCondition = pd.Series(np.nan, index=trialInfo.index)
    stimOrder = []
    for name, group in trialInfo.groupby(['electrode', 'trialRateInHz']):
        stimCondition.loc[group.index] = '{}_{}'.format(*name)
        stimOrder.append('{} {}'.format(*name))
    trialInfo.loc[:, 'stimCondition'] = stimCondition
    stimConditionLookup = (
        trialInfo
            .loc[:, ['electrode', 'trialRateInHz', 'stimCondition']]
            .drop_duplicates()
            .set_index(['electrode', 'trialRateInHz'])['stimCondition'])
    kinCondition = pd.Series(np.nan, index=trialInfo.index)
    kinOrder = []
    for name, group in trialInfo.groupby(['pedalDirection', 'pedalMovementCat']):
        kinCondition.loc[group.index] = '{}_{}'.format(*name)
        kinOrder.append('{} {}'.format(*name))
    trialInfo.loc[:, 'kinCondition'] = kinCondition
    kinConditionLookup = (
        trialInfo
            .loc[:, ['pedalMovementCat', 'pedalDirection', 'kinCondition']]
            .drop_duplicates()
            .set_index(['pedalMovementCat', 'pedalDirection'])['kinCondition'])
    ####
    # prep rhs dataframes
    histDesignInfoDict = {}
    histSourceTermDict = {}
    featureReadableLookup = {}
    featureReadableLookup['prediction'] = 'Regression prediction'
    featureReadableLookup['ground_truth'] = 'Ground truth'
    featureReadableLookup['factor'] = 'Regression factor'
    featureReadableLookup['component'] = 'Regression component'
    #
    for rhsMaskIdx in range(rhsMasks.shape[0]):
        prf.print_memory_usage('\n Prepping RHS dataframes (rhsRow: {})\n'.format(rhsMaskIdx))
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
        rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
        #
        transformerName = estimatorMeta['arguments']['transformerNameRhs']
        freqBandName = rhsMasksInfo.loc[rhsMaskIdx, 'freqBandName']
        featureNames = rhGroup.columns.unique()
        for fidx, fn in enumerate(featureNames):
            nameParts = ['Factor']
            if freqBandName != 'all':
                nameParts.append(freqBandName)
            nameParts.append('{}'.format(fidx + 1))
            featureReadableLookup[fn] = ' '.join(nameParts)
        #
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
    impulseDict = {}
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
    designInfoDF = pd.Series(designInfoDict).to_frame(name='designInfo')
    designInfoDF.index.name = 'design'
    histDesignInfoDF = pd.DataFrame(
        [value for key, value in histDesignInfoDict.items()],
        columns=['designInfo'])
    histDesignInfoDF.index = pd.MultiIndex.from_tuples(
        [key for key, value in histDesignInfoDict.items()],
        names=['rhsMaskIdx', 'ensTemplate'])

    titleLookup = {
        'trialAmplitude = {}'.format(ta): '{} uA'.format(ta) for ta in trialInfo['trialAmplitude'].unique()
        }
    titleLookup.update({
        'kinCondition = NA_NA': 'No movement',
        'kinCondition = CW_outbound': 'Start of movement (extension)',
        'kinCondition = CW_return': 'Start of movement (flexion)',
        })

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
        '{}_{}.pdf'.format(fullEstimatorName, 'partial_scores_export'))
    with PdfPages(pdfPath) as pdf:
        height, width = 3, 3
        aspect = width / height
        # maskSecondOrderTests = modelsToTestDF['testType'] == 'secondOrderInteractions'
        # for testTypeName, modelsToTestGroup in modelsToTestDF.groupby('testType'):
        for testTypeName in ['VTerms', 'ARTerms', 'VAVRInteractions', 'VARNoEnsembleTerms']:
            testTypeMask = (modelsToTestDF['testType'] == testTypeName)
            modelsToTestGroup = modelsToTestDF.loc[testTypeMask, :].copy()
            ###
            if 'refCaption' in modelsToTestGroup:
                refCaption = modelsToTestGroup['refCaption'].iloc[0]
            else:
                refCaption = lhsMasksInfo.loc[modelsToTestGroup['refDesign'].iloc[0], 'fullFormulaDescr']
            if refCaption in modelsTestReadable:
                refCaption = modelsTestReadable[refCaption]
            if 'testCaption' in modelsToTestGroup:
                testCaption = modelsToTestGroup['testCaption'].iloc[0]
            else:
                testCaption = lhsMasksInfo.loc[modelsToTestGroup['testDesign'].iloc[0], 'fullFormulaDescr']
            if testCaption in modelsTestReadable:
                testCaption = modelsTestReadable[testCaption]
            print('\n\n')
            print(testTypeName)
            print(testCaption)
            print(lhsMasksInfo.loc[modelsToTestGroup['testDesign'], 'fullFormulaDescr'])
            print(refCaption)
            print(lhsMasksInfo.loc[modelsToTestGroup['refDesign'], 'fullFormulaDescr'])
            print('\n\n')
            titleText = '{} compared to {}'.format(testCaption, refCaption)
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
            plotScores.loc[:, 'targetLabel'] = plotScores['target'].map(featureReadableLookup)
            FUDE_alpha = 0.01
            plotScores.loc[:, 'significant'] = (plotPVals < FUDE_alpha).to_numpy()
            ###
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            plotFUDE.loc[:, 'targetLabel'] = plotFUDE['target'].map(featureReadableLookup)
            g = sns.catplot(
                data=plotFUDE, kind='box',
                y='score', x='targetLabel', hue='trialType',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=height, aspect=aspect,
                sharey=True
                )
            g.set_xticklabels(rotation=-30, ha='left')
            g.suptitle(titleText)
            g.set_axis_labels('Regression target', 'FUDE')
            print('Saving {}\n to {}'.format(titleText, pdfPath))
            # g.axes.flat[0].set_ylim(allScoreQuantiles)
            g.legend.remove()
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
                style='targetLabel',
                # hue_order=thisPalette.index.to_list(),
                # palette=thisPalette.to_dict(),
                )
            g.set_axis_labels('$R^2$, {}'.format(refCaption), '$R^2$, {}'.format(testCaption))
            plotProcFuns = [drawUnityLine, annotateWithPVal, asp.genTitleChanger(titleLookup)]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            asp.reformatFacetGridLegend(
                g, titleOverrides={'significant': ' '},
                contentOverrides={
                    'True': 'p < {}'.format(FUDE_alpha),
                    'False': 'p > {}'.format(FUDE_alpha),
                    'targetLabel': 'Regression target'
                },
                styleOpts=styleOpts)
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'r2_export'))
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
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'reconstructions_export'))
    plotProcFuns = [
        asp.genTitleChanger(titleLookup)]
    with PdfPages(pdfPath) as pdf:
        for name0, predGroup0 in predDF.groupby(groupPagesBy):
            nmLk0 = {key: value for key, value in zip(groupPagesBy, name0)} # name lookup
            nmLk0['design'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'designFormula']
            nmLk0['fullFormulaDescr'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'fullFormulaDescr']
            if nmLk0['fullFormulaDescr'] in fullFormulaReadableLabels:
                nmLk0['fullFormulaReadable'] = fullFormulaReadableLabels[nmLk0['fullFormulaDescr']]
            else:
                nmLk0['fullFormulaReadable'] = nmLk0['fullFormulaDescr']
            if not (nmLk0['lhsMaskIdx'] in [29]):
                continue
            scoreMasks = [
                scoresStack[cN] == nmLk0[cN]
                for cN in groupPagesBy]
            plotScores = scoresStack.loc[np.logical_and.reduce(scoreMasks), :]
            plotScores = plotScores.loc[plotScores['foldType'] == 'test', :]
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['foldType'])]
            plotScores.loc[:, 'targetLabel'] = plotScores['target'].map(featureReadableLookup)
            g = sns.catplot(
                data=plotScores, hue='foldType',
                x='targetLabel', y='score',
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                height=height, aspect=aspect,
                kind='box')
            g.set_axis_labels('Regression target', '$R^2$')
            g.set_xticklabels(rotation=-30, ha='left')
            g.suptitle('$R^2$, {fullFormulaReadable}'.format(**nmLk0))
            g.legend.remove()
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
                nmLk0['targetLabel'] = featureReadableLookup[nmLk0['target']]
                print(nmLk0['foldType'])
                if nmLk0['trialType'] != trialTypeToPlot:
                    continue
                if nmLk0['trialRateInHz'] not in [0, 100]:
                    continue
                plotDF = predGroup1.stack().to_frame(name='signal').reset_index()
                plotDF.loc[:, 'predType'] = 'component'
                plotDF.loc[plotDF['term'] == 'ground_truth', 'predType'] = 'ground_truth'
                plotDF.loc[plotDF['term'] == 'prediction', 'predType'] = 'prediction'
                plotDF.loc[:, 'kinCondition'] = plotDF.loc[:, ['pedalMovementCat', 'pedalDirection']].apply(lambda x: tuple(x), axis='columns').map(kinConditionLookup)
                plotDF.loc[:, 'stimCondition'] = plotDF.loc[:, ['electrode', 'trialRateInHz']].apply(lambda x: tuple(x), axis='columns').map(stimConditionLookup)
                plotDF.loc[:, 'fullFormulaDescr'] = plotDF['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
                ##
                # final triage of what to plot
                finalMask = (plotDF['kinCondition'] == 'CW_return') & (plotDF['trialAmplitude'] == plotDF['trialAmplitude'].max())
                plotDF = plotDF.loc[finalMask, :]
                #
                ###
                kinOrder = kinConditionLookup.loc[kinConditionLookup.isin(plotDF['kinCondition'])].to_list()
                stimOrder = stimConditionLookup.loc[stimConditionLookup.isin(plotDF['stimCondition'])].to_list()
                thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
                theseColors = thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color']
                theseLegendUpdates = thisTermPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict()
                for k, v in theseLegendUpdates.items():
                    if v in featureReadableLookup:
                        theseLegendUpdates[k] = featureReadableLookup[v]
                theseLegendUpdates.update(featureReadableLookup)
                theseLegendUpdates.update({'term': ' ', 'predType': ' '})
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
                def plotRoutine(inputDF, inputColorPalette, inputLegendUpdates, hueName=None):
                    g = sns.relplot(
                        data=inputDF,
                        col='trialAmplitude', row='kinCondition',
                        row_order=kinOrder,
                        x='bin', y='signal', hue=hueName, hue_order=inputColorPalette.index,
                        height=height, aspect=aspect, palette=inputColorPalette.to_dict(),
                        kind='line', errorbar='sd',
                        size='predType', sizes=predictionLineStyleDF.loc['sizes', :].to_dict(),
                        style='predType', dashes=predictionLineStyleDF.loc['dashes', :].to_dict(),
                        style_order=predictionLineStyleDF.columns,
                        facet_kws=dict(margin_titles=True), err_kws=dict(alpha=0.2, edgecolor=None),
                        )
                    for (ro, co, hu), dataSubset in g.facet_data():
                        if len(plotProcFuns):
                            for procFun in plotProcFuns:
                                procFun(g, ro, co, hu, dataSubset)
                    g.set_axis_labels('Time (msec)', '{targetLabel} (a.u.)'.format(**nmLk0))
                    if not nmLk0['electrode'] == 'NA':
                        titleText = '{fullFormulaReadable}\n{targetLabel}, stimulation on {electrode} at {trialRateInHz} Hz'.format(
                            **nmLk0)
                    else:
                        titleText = '{fullFormulaReadable}\n{targetLabel}, no stimulation'.format(
                            **nmLk0)
                    print('Saving plot of {}...'.format(titleText))
                    g.suptitle(titleText)
                    asp.reformatFacetGridLegend(
                        g, titleOverrides=inputLegendUpdates,
                        contentOverrides=inputLegendUpdates,
                        styleOpts=styleOpts)
                    g.resize_legend(adjust_subtitles=True)
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

    '''pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'impulse_responses'))
    with PdfPages(pdfPath) as pdf:
        height, width = 2, 4
        aspect = width / height
        for (lhsMaskIdx, designFormula, rhsMaskIdx), thisIRPerTerm in iRPerTerm.groupby(['lhsMaskIdx', 'design', 'rhsMaskIdx']):
            lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
            lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
            thisTitleFormula = lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr']
            print('Saving impulse response plots for {}'.format(thisTitleFormula))
            if designFormula != 'NULL':
                designInfo = designInfoDict[designFormula]
                termNames = designInfo.term_names
                histLens = [sourceHistOptsDict[tN.replace(' ', '')]['historyLen'] for tN in termNames]
            else:
                histLens = []
            ensTemplate = lhsMaskParams['ensembleTemplate']
            if ensTemplate != 'NULL':
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
                histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
            selfTemplate = lhsMaskParams['selfTemplate']
            if selfTemplate != 'NULL':
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
                histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
            tBins = thisIRPerTerm.index.get_level_values('bin')
            kernelMask = (tBins >= 0) & (tBins <= max(histLens))
            plotDF = thisIRPerTerm.loc[kernelMask, :].stack().to_frame(name='signal').reset_index()
            kinOrder = kinConditionLookupIR.loc[kinConditionLookupIR.isin(plotDF['kinCondition'])].to_list()
            stimOrder = stimConditionLookupIR.loc[stimConditionLookupIR.isin(plotDF['stimCondition'])].to_list()
            thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
            g = sns.relplot(
                # row='kinCondition', row_order=kinOrder,
                # col='stimCondition', col_order=stimOrder,
                row='target',
                x='bin', y='signal', hue='term',
                hue_order=thisTermPalette['term'].to_list(),
                palette=thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color'].to_dict(),
                kind='line', errorbar='se', data=plotDF,
                )
            g.set_axis_labels("Lag (sec)", 'contribution to target')
            g.suptitle('Impulse responses (per term) for model {}'.format(thisTitleFormula))
            asp.reformatFacetGridLegend(
                g, titleOverrides={},
                contentOverrides=termPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict(),
                styleOpts=styleOpts)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()'''
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