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
from tqdm import tqdm
import contextlib
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
        'lines.linewidth': .5,
        'lines.markersize': 2.,
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
    'legend.markerscale': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc=snsRCParams)
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
    with pd.HDFStore(estimatorPath) as store:
        modelCompareFUDE = pd.read_hdf(store, 'modelCompareFUDE')
        modelCompareFUDEStats = pd.read_hdf(store, 'modelCompareFUDEStats')
        modelCompareScores = pd.read_hdf(store, 'modelCompareScores')
        #
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
        predictionLineStyleDF = pd.read_hdf(store, 'termLineStyleDF')
    stimConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'stimConditionLookup')
    kinConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'kinConditionLookup')
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    modelMetadataLookup = pd.read_hdf(transferFuncPath, 'modelMetadataLookup')
    trialMetadataLookup = pd.read_hdf(transferFuncPath, 'trialMetadataLookup')
    #
    prettyNameLookup.update({
        'isTrialAveraged = True': 'Trial averaged',
        'isTrialAveraged = False': 'Single trial',
        'train': 'Train',
        'test': 'Test',
        'term': 'Term',
        'NA': 'No  stim.',
        'predType': 'Prediction type',
        'target': 'Regression target'
    })
    predictionPrettyNamesShort = {
        'prediction': r'$\hat{y}^{VARX}_i$',
        'ground_truth': r'$y_i$',
        'inputDriven': r'$\hat{y}^{ID}_i$',
        'oneStepKalman': r'$\hat{y}^{OSK}_i$',
        }
    predictionPrettyNamesLong = {
        'prediction': 'VARX model prediction ({})'.format(predictionPrettyNamesShort['prediction']),
        'ground_truth': 'Ground truth ({})'.format(predictionPrettyNamesShort['ground_truth']),
        'inputDriven': 'Prediction input-driven component ({})'.format(predictionPrettyNamesShort['inputDriven']),
        'oneStepKalman': 'One step Kalman prediction ({})'.format(predictionPrettyNamesShort['oneStepKalman']),
        }
    prettyNameLookup.update(predictionPrettyNamesLong)
    def formatModelSpec(infoSrs):
        designShortHand = '(No exogenous)' if infoSrs['design'] == 'NULL' else formulasShortHand[infoSrs['design']]
        selfShort = 'No self' if (infoSrs['selfFormulaDescr'] == 'NULL') else 'self'
        ensShort = 'No ensemble' if (infoSrs['ensembleFormulaDescr'] == 'NULL') else 'ensemble'
        return '({}) + ({}) + {}'.format(selfShort, ensShort, designShortHand)

    modelMetadataLookup.loc[:, 'fullDesignAsLabel'] = modelMetadataLookup.apply(formatModelSpec, axis='columns')
    modelMetadataLookup.loc[:, 'fullDesignAsMath'] = modelMetadataLookup.index.get_level_values('lhsMaskIdx').map(lhsMasksDesignAsMath)
    modelMetadataLookup.loc[modelMetadataLookup['fullDesignAsMath'].isna(), 'fullDesignAsMath'] = modelMetadataLookup.loc[modelMetadataLookup['fullDesignAsMath'].isna(), 'fullDesignAsLabel']
    spinalMapDF = spinalElectrodeMaps[subjectName].sort_values(['xCoords', 'yCoords'])
    pedalDirCat = pd.CategoricalDtype(['NA', 'CW', 'CCW'], ordered=True)
    pedalMoveCat = pd.CategoricalDtype(['NA', 'outbound', 'return'], ordered=True)
    trialMetadataLookup.loc[:, 'pedalMovementCat'] = trialMetadataLookup['pedalMovementCat'].astype(pedalMoveCat)
    trialMetadataLookup.loc[:, 'pedalDirection'] = trialMetadataLookup['pedalDirection'].astype(pedalDirCat)
    spinalElecCategoricalDtype = pd.CategoricalDtype(spinalMapDF.index.to_list(), ordered=True)
    trialMetadataLookup.loc[:, 'electrode'] = trialMetadataLookup['electrode'].astype(spinalElecCategoricalDtype)
    #
    trialMetadataLookup.loc[:, 'kinCondition'] = trialMetadataLookup.apply(lambda x: '{}_{}'.format(x['pedalMovementCat'], x['pedalDirection']), axis='columns')
    uniqKinConditions = trialMetadataLookup.loc[:, ['pedalMovementCat', 'pedalDirection', 'kinCondition']].drop_duplicates().sort_values(['pedalMovementCat', 'pedalDirection']).reset_index(drop=True)
    trialMetadataLookup.loc[:, 'kinCondition'] = trialMetadataLookup['kinCondition'].astype(pd.CategoricalDtype(uniqKinConditions['kinCondition'].to_list(), ordered=True))
    trialMetadataLookup.loc[:, 'stimCondition'] = trialMetadataLookup.apply(lambda x: '{}_{}'.format(x['electrode'], x['trialRateInHz']), axis='columns')
    uniqStimConditions = trialMetadataLookup.loc[:, ['electrode', 'trialRateInHz', 'stimCondition']].drop_duplicates().sort_values(['electrode', 'trialRateInHz']).reset_index(drop=True)
    trialMetadataLookup.loc[:, 'stimCondition'] = trialMetadataLookup['stimCondition'].astype(pd.CategoricalDtype(uniqStimConditions['stimCondition'].to_list(), ordered=True))
    trialMetadataLookup.loc[:, ['trialAmplitude', 'trialRateInHz']] = trialMetadataLookup.loc[:, ['trialAmplitude', 'trialRateInHz']].astype(int)
    #
    ################ collect estimators and scores
    scoresStack = pd.read_hdf(estimatorPath, 'processedScores')
    # swap, for consistent naming
    scoresStack.loc[:, 'foldType'] = scoresStack.loc[:, 'trialType']
    #
    ssScores = pd.read_hdf(transferFuncPath, 'stateSpaceScores')
    predDF = pd.read_hdf(transferFuncPath, 'stateSpacePredictions')
    groupSubPagesBy = ['lhsMaskIdx', 'trialType', 'electrode', 'trialRateInHz', 'target']
    plotSingleTrials = False
    if plotSingleTrials:
        predDF = predDF.xs(0, level='conditionUID')
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, fullEstimatorName, 'state_space_reconstructions_single_trial_paper'))
    else:
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, fullEstimatorName, 'state_space_reconstructions_paper'))
    # with PdfPages(pdfPath) as pdf:
    def genInsetBoxplot(
            insetData=None,
            bounds=None, transform=None,
            legend=True, newLabels={}, keysForNewLabels=[],
            row=None, col=None, globalYLims=True, addTitle=False):
        def insetBoxplot(
                data=None,
                x=None, order=None,
                y=None,
                hue=None, hue_order=None, palette=None,
                color=None, *args, **kwargs):
            ax = plt.gca()
            axIns = ax.inset_axes(bounds=bounds, transform=transform, )
            _ = (row, col, legend, globalYLims, addTitle)
            if insetData is None:
                ddf = data
            else:
                ddf = insetData
                if row is not None:
                    ddf = ddf.loc[ddf[row] == (data[row].unique()[0]), :]
                if col is not None:
                    ddf = ddf.loc[ddf[col] == (data[col].unique()[0]), :]
            titleTextList = []
            if row is not None:
                titleTextList.append('{}'.format(ddf[row].unique()[0]))
            if col is not None:
                titleTextList.append('{}'.format(ddf[col].unique()[0]))
            sns.boxplot(
                data=ddf, x=x, order=order, y=y,
                hue=hue, hue_order=hue_order, palette=palette,
                ax=axIns, *args, **kwargs)
            if  x == 'xDummy':
                axIns.set_xticks([])
                axIns.set_xticklabels([])
            if not legend:
                axIns.get_legend().remove()
            if  len(keysForNewLabels):
                newNames = ddf.loc[:, keysForNewLabels].drop_duplicates()
                assert newNames.shape[0] == 1
                newNamesDict = newNames.iloc[0, :].to_dict()
            else:
                newNamesDict = {}
            if 'x' in newLabels:
                axIns.set_xlabel(newLabels['x'].format(**newNamesDict))
            if 'y' in newLabels:
                axIns.set_ylabel(newLabels['y'].format(**newNamesDict))
            if 'title' in newLabels:
                axIns.set_title(newLabels['title'].format(**newNamesDict))
            if  globalYLims:
                newLims = insetData[y].quantile([0,  1]).to_list()
                deltaLims =  newLims[1] - newLims[0]
                newLims[0] -= deltaLims * 5e-2
                newLims[1] += deltaLims * 5e-2
                axIns.set_ylim(newLims)
            axIns.grid(False)
            return
        return insetBoxplot
    
    def plotRoutine(
            inputDF=None,
            inputColorPalette=None, trimColorPalette=True,
            inputLegendUpdates=None, titleText=None,
            inputLineStylePalette=None, trimLinesPalette=True, setTitlesTemplate=None,
            plotProcFuns=[],
            axLims=None, axLabels=[None, None], relPlotKWArgs=None, plotContext=None, showFigures=False):
        defaultRelPlotKWargs = dict(
            col='trialAmplitude', row='kinCondition',
            x='bin', y='signal',
            hue='term',
            size='predType', style='predType',
            height=2, aspect=1.5,
            kind='line', errorbar='se'
            )
        rkwa = defaultRelPlotKWargs.copy()
        if relPlotKWArgs is not None:
            rkwa.update(relPlotKWArgs)
        rkwa['data'] = inputDF
        if inputColorPalette is not None:
            thisPalette = inputColorPalette.loc[:, [rkwa['hue'], 'color']].reset_index().set_index(rkwa['hue'])['color']
            if trimColorPalette:
                thisPalette = thisPalette.loc[thisPalette.index.isin(inputDF[rkwa['hue']])]
            rkwa.update(dict(hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict()))
        if ('size' in rkwa) and (inputLineStylePalette is not None):
            thisSizeLookup = inputLineStylePalette.loc['sizes', :]
            if trimLinesPalette:
                thisSizeLookup = thisSizeLookup.loc[thisSizeLookup.index.isin(inputDF[rkwa['size']])]
            rkwa.update(dict(size_order=thisSizeLookup.index.to_list(), sizes=thisSizeLookup.to_dict()))
        if ('style' in rkwa) and (inputLineStylePalette is not None):
            thisStyleLookup = inputLineStylePalette.loc['dashes', :]
            if trimLinesPalette:
                thisStyleLookup = thisStyleLookup.loc[thisStyleLookup.index.isin(inputDF[rkwa['style']])]
            rkwa.update(dict(style_order=thisStyleLookup.index.to_list(), dashes=thisStyleLookup.to_dict()))
        gg = sns.relplot(**rkwa,)
        if len(plotProcFuns):
            for (ro, co, hu), dataSubset in gg.facet_data():
                for procFun in plotProcFuns:
                    procFun(gg, ro, co, hu, dataSubset)
        gg.set_axis_labels(*axLabels)
        if setTitlesTemplate is not None:
            gg.set_titles(**setTitlesTemplate)
        if titleText is not None:
            print('Saving plot of {}...'.format(titleText))
            gg.suptitle(titleText)
        if inputLegendUpdates is not None:
            asp.reformatFacetGridLegend(
                gg, titleOverrides=inputLegendUpdates,
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
        if plotContext is not None:
            plotContext.savefig(bbox_inches='tight')
        if showFigures:
            plt.show()
        else:
            plt.close()
        return gg, currAxLims
    # with contextlib.nullcontext() as pdf:
    with PdfPages(pdfPath) as pdf:
        for rhsMaskIdx, predGroup0 in predDF.groupby('rhsMaskIdx'):
            modelMetadata = modelMetadataLookup.xs(rhsMaskIdx, level='rhsMaskIdx')
            targetsByAverageScore = scoresStack.set_index(['lhsMaskIdx', 'rhsMaskIdx']).xs([0, rhsMaskIdx], level=['lhsMaskIdx', 'rhsMaskIdx']).groupby('target').mean()['cc'].sort_values().index.to_list()
            #########################################################################################################
            plotScores = ssScores.xs(rhsMaskIdx, level='rhsMaskIdx')
            plotScores.index.names = plotScores.index.names[:-1] + ['referenceTerm']
            plotScores.columns.name = 'comparisonTerm'
            plotScores = plotScores.stack().to_frame(name='cc')
            #
            oldScores = scoresStack.loc[scoresStack['rhsMaskIdx'] == rhsMaskIdx, :].reset_index(drop=True)
            oldScores.loc[:, 'referenceTerm'] = 'ground_truth'
            oldScores.loc[:, 'comparisonTerm'] = 'original_prediction'
            oldScores.loc[:, 'isTrialAveraged'] = False
            oldScores = oldScores.loc[:, plotScores.index.names + ['cc']].set_index(plotScores.index.names)
            #
            plotScores = pd.concat([plotScores, oldScores]).reset_index()
            plotScores.loc[:, 'xDummy'] = 0
            plotScores.loc[:, 'isTrialAveraged_referenceTerm'] = plotScores.apply(lambda x: '{}_{}'.format(x['isTrialAveraged'], x['referenceTerm']), axis='columns')
            plotScores.loc[:, 'fullDesignAsMath'] = plotScores['lhsMaskIdx'].map(modelMetadata['fullDesignAsMath'])
            plotScores.loc[:, 'referenceTermAsLabel'] = plotScores['referenceTerm'].map(prettyNameLookup)
            plotScores.loc[:, 'comparisonTermAsLabel'] = plotScores['comparisonTerm'].map(predictionPrettyNamesShort)
            boxPlotKWArgs = dict(whis=np.inf)
            catPlotKWArgs = dict(
                y='cc', kind='box',
                margin_titles=True, height=1.5, aspect=1.,
                **boxPlotKWArgs)
            #
            scorePlotMaskList = []
            scorePlotMaskList.append((
                (plotScores['trialType'].isin(['train', 'test'])) &
                (plotScores['lhsMaskIdx'].isin([0, 3])) &
                (plotScores['referenceTerm'].isin(['ground_truth'])) &
                (plotScores['comparisonTerm'].isin(['prediction'])) &
                (plotScores['referenceTerm'] != plotScores['comparisonTerm'])
                ).to_numpy())
            scorePlotMaskList.append((
                (plotScores['trialType'].isin(['train', 'test'])) &
                (plotScores['lhsMaskIdx'].isin([5])) &
                (plotScores['referenceTerm'].isin(['ground_truth'])) &
                (plotScores['comparisonTerm'].isin(['prediction', 'inputDriven', 'oneStepKalman'])) &
                (plotScores['referenceTerm'] != plotScores['comparisonTerm'])
                ).to_numpy())
            scorePlotMaskList.append((
                (plotScores['trialType'].isin(['train', 'test'])) &
                (plotScores['lhsMaskIdx'].isin([5])) &
                (plotScores['isTrialAveraged'].isin([False])) &
                (plotScores['referenceTerm'].isin(['prediction'])) &
                (plotScores['comparisonTerm'].isin(['inputDriven', 'oneStepKalman'])) &
                (plotScores['referenceTerm'] != plotScores['comparisonTerm'])
                ).to_numpy())
            for scorePlotMask in scorePlotMaskList:
                if not scorePlotMask.any():
                    continue
                thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores.loc[scorePlotMask, 'trialType'])]
                g = sns.catplot(
                    data=plotScores.loc[scorePlotMask, :],
                    row='isTrialAveraged', # row_order=[True, False],
                    col='fullDesignAsMath',
                    x='comparisonTermAsLabel',
                    hue='trialType',
                    hue_order=['train', 'test'], palette=thisPalette.to_dict(),
                    **catPlotKWArgs
                    )
                g.set_titles(col_template="{col_name}", row_template="{row_var} = {row_name}")
                plotProcFuns = [
                    asp.genTitleChanger(prettyNameLookup)]
                for (ro, co, hu), dataSubset in g.facet_data():
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                titleText = 'Goodness-of-fit tests'
                if titleText is not None:
                    print('Saving plot of {}...'.format(titleText))
                    g.suptitle(titleText)
                g.set_xticklabels(ha='center', va='top')
                g.set_axis_labels('', 'CC')
                asp.reformatFacetGridLegendV2(
                    g, labelOverrides=prettyNameLookup,
                    styleOpts=styleOpts, )
                g.resize_legend(adjust_subtitles=True)
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                pdf.savefig(bbox_inches='tight')
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            ###
            plotFUDE = modelCompareFUDE.reset_index()
            plotFUDEMask = (
                (plotFUDE['testType'].isin(['exogVSExogAndSelf', 'VARVsVARInter', 'ensVSFull'])) &
                (plotFUDE['electrode'] == 'all') &
                (plotFUDE['trialType'] == 'test')
                )
            plotFUDE = plotFUDE.loc[plotFUDEMask, :]
            plotFUDE.loc[:,  'xDummy'] = 0
            plotFUDE.loc[:, 'testCaption'] = plotFUDE['testLhsMaskIdx'].map(modelMetadata['fullDesignAsMath'])
            plotFUDE.loc[:, 'refCaption'] = plotFUDE['refLhsMaskIdx'].map(modelMetadata['fullDesignAsMath'])
            plotFUDE.loc[:, 'comparisonCaption'] = plotFUDE['testType'].map(addedTermsAsMath)
            ##
            plotFUDEStats = modelCompareFUDEStats.reset_index()
            plotFUDEStatsMask = (
                (plotFUDEStats['testType'].isin(['exogVSExogAndSelf', 'VARVsVARInter', 'ensVSFull'])) &
                (plotFUDEStats['electrode'] == 'all')
                )
            plotFUDEStats = plotFUDEStats.loc[plotFUDEStatsMask, :]
            #
            plotScores = modelCompareScores.reset_index()
            plotScoresMask = (
                (plotScores['testType'].isin(['exogVSExogAndSelf', 'VARVsVARInter', 'ensVSFull'])) &
                (plotScores['electrode'] == 'all') &
                (plotScores['trialType'].isin(['test']))
                )
            plotScores = plotScores.loc[plotScoresMask, :]
            USERINSTEADOFR2 = True
            if USERINSTEADOFR2:
                plotScores.loc[:, 'test_score'] = plotScores['test_score'].apply(np.sqrt)
                plotScores.loc[:, 'ref_score'] = plotScores['ref_score'].apply(np.sqrt)
            #
            lookupBasedOn = ['testLhsMaskIdx', 'refLhsMaskIdx', 'target']
            lookupAt = pd.MultiIndex.from_frame(plotScores.loc[:, lookupBasedOn])
            lookupFrom = plotFUDEStats.loc[:, lookupBasedOn + ['p-val']].set_index(lookupBasedOn)['p-val']
            plotPVals = lookupFrom.loc[lookupAt]
            prettyNameLookup['significant'] = 'p < 0.01'
            plotScores.loc[:, 'significant'] = (plotPVals < 0.01).to_numpy()
            plotScores.loc[:, 'testCaption'] = plotScores['testLhsMaskIdx'].map(modelMetadata['fullDesignAsMath'])
            plotScores.loc[:, 'refCaption'] = plotScores['refLhsMaskIdx'].map(modelMetadata['fullDesignAsMath'])
            plotScores.loc[:, 'comparisonCaption'] = plotScores['testType'].map(addedTermsAsMath)
            thisPalette = sourcePalette.reset_index()
            thisPalette.loc[thisPalette['index'].isin(targetsByAverageScore), 'index'] = targetsByAverageScore
            thisPalette = thisPalette.set_index('index')[0]
            thisPalette = thisPalette.loc[thisPalette.index.isin(plotScores.loc[:, 'target'])]
            # thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            height =  2.
            aspect = 1.
            g = sns.relplot(
                data=plotScores, kind='scatter',
                x='ref_score', y='test_score', 
                # row='testCaption', col='refCaption',
                col='testType',
                hue='target', 
                style='significant', markers={True: 'o', False: 'X'},
                height=height, aspect=aspect,
                edgecolor=None,
                hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                facet_kws=dict(
                    margin_titles=True,
                    sharex=False, sharey=False),
                )
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
            mapDFProcFuns = [
                (
                    genInsetBoxplot(
                        insetData=plotFUDE, bounds=[ 0.6, 0.1, 0.3, 0.3],
                        legend=False, newLabels={'x': '', 'y': 'FUDE', 'title': '{comparisonCaption}'},
                        keysForNewLabels=['comparisonCaption'],
                        # row='refCaption', col='testCaption',
                        col='testType'
                        ),
                    [],
                    dict(
                        x='xDummy', y='score', hue='trialType',
                        palette=thisPalette.to_dict(), hue_order=['test'],
                        whis=np.inf)),
                    ]
            for mpdf in mapDFProcFuns:
                mpdf_fun, mpdf_args, mpdf_kwargs = mpdf
                g.map_dataframe(mpdf_fun, *mpdf_args, **mpdf_kwargs)
            plotProcFuns = [
                asp.genLineDrawer(slope=1., offset=0, plotKWArgs=None),
                asp.genAxisLabelOverride(
                    xTemplate='{refCaption}', yTemplate='{testCaption}',
                    titleTemplate='CC', colKeys=['refCaption', 'testCaption'],
                    dropNaNCol='segment')]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            titleText = 'Added explanatory value'
            if titleText is not None:
                print('Saving plot of {}...'.format(titleText))
                g.suptitle(titleText)
            asp.reformatFacetGridLegendV2(
                g=g, labelOverrides=prettyNameLookup,
                styleOpts=styleOpts, shorten=6)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #########################################################################################################
            for lhsMaskIdx, plotGroup0 in predGroup0.groupby('lhsMaskIdx'):
                if lhsMaskIdx not in [3, 5]:
                    continue
                else:
                    modelMetadata = modelMetadataLookup.loc[(lhsMaskIdx, rhsMaskIdx,)]
                trialMetadata = trialMetadataLookup.apply(lambda x: plotGroup0.index.droplevel([lN for lN in plotGroup0.index.names if lN not in trialMetadataLookup.index.names]).map(x), axis='index')
                trialMetadata = pd.concat([trialMetadata, plotGroup0.index.to_frame().reset_index(drop=True), ], axis='columns')
                plotMask0 = (
                        (trialMetadata['trialType'] == 'train') &
                        (trialMetadata['target'].isin([targetsByAverageScore[tIdx] for tIdx in [-1]])) &
                        (trialMetadata['trialAmplitude'].isin([500])) &
                        (trialMetadata['kinCondition'].isin(['outbound_CW'])) &
                        (trialMetadata['trialRateInHz'] != 50)
                    ).to_numpy()
                if plotMask0.any():
                    plotGroup1 = pd.DataFrame(plotGroup0.loc[plotMask0, :])
                    plotGroup1.index = pd.MultiIndex.from_frame(trialMetadata.loc[plotMask0, :])
                    plotGroup1.columns.name = 'term'
                    for name2, plotGroup2 in plotGroup1.groupby(groupSubPagesBy):
                        titleText = '{}\n{} {} {} Hz'.format(
                            modelMetadata.loc['fullDesignAsMath'],
                            prettyNameLookup[name2[4]], prettyNameLookup[name2[2]], name2[3])
                        ## groupSubPagesBy = ['lhsMaskIdx', 'trialType', 'electrode', 'trialRateInHz', 'target']
                        plotGroup2 = plotGroup2.stack().to_frame(name='signal').reset_index()
                        plotGroup2.loc[:, 'predType'] = plotGroup2['term'].map(termPalette.loc[:, ['term']].reset_index().set_index('term')['type'])
                        expandOther = True
                        if expandOther:
                            plotGroup2.loc[plotGroup2['predType'] == 'other', 'predType'] = plotGroup2.loc[plotGroup2['predType'] == 'other', 'term']
                        rkwa = {
                            'facet_kws': {'margin_titles': True},
                            'row_order': plotGroup2['kinCondition'].unique().sort_values().to_list(),
                            'size': None, 'style': None
                            }
                        plotMask2 = (
                                plotGroup2['term'].isin(['ground_truth', 'prediction']) &
                                (plotGroup2['bin'] < 1.)
                            ).to_numpy()
                        plotProcFuns = [
                            asp.genAxisLabelOverride(
                                xTemplate='Time (sec)', yTemplate='Normalized LFP ($z-score, a.u.$)',
                                colTitleTemplate='{trialAmplitude} $\mu A$',
                                rowTitleTemplate='{kinCondition}',
                                colKeys=['trialAmplitude', 'kinCondition'], prettyNameLookup=prettyNameLookup)]
                        gg, currAxLims = plotRoutine(
                            inputDF=plotGroup2.loc[plotMask2, :],
                            inputColorPalette=termPalette, trimColorPalette=True,
                            inputLegendUpdates=prettyNameLookup,
                            inputLineStylePalette=None, trimLinesPalette=True,
                            axLims=None, relPlotKWArgs=rkwa, titleText=titleText,
                            # axLabels=['Time (sec)', 'Normalized LFP ($z-score, a.u.$)'],
                            # setTitlesTemplate=dict(col_template='{col_name} $\mu A$', row_template=None),
                            plotProcFuns=plotProcFuns,
                            plotContext=pdf,
                            showFigures=arguments['showFigures']
                            )
                        plotMask3 = (
                                plotGroup2['term'].isin(['ground_truth', 'prediction', 'inputDriven', 'oneStepKalman']) &
                                (plotGroup2['bin'] < 1.)
                            ).to_numpy()
                        gg, currAxLims = plotRoutine(
                            inputDF=plotGroup2.loc[plotMask3, :],
                            inputColorPalette=termPalette, trimColorPalette=True,
                            inputLegendUpdates=prettyNameLookup,
                            inputLineStylePalette=None, trimLinesPalette=True,
                            axLims=None, relPlotKWArgs=rkwa, titleText=titleText,
                            # axLabels=['Time (sec)', 'Normalized LFP ($z-score, a.u.$)'],
                            # setTitlesTemplate=dict(col_template='{col_name} $\mu A$', row_template=None),
                            plotProcFuns=plotProcFuns,
                            plotContext=pdf,
                            showFigures=arguments['showFigures']
                            )
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')