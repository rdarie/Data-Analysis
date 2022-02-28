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
    with pd.HDFStore(estimatorPath) as store:
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        sourceTermLookup = pd.read_hdf(store, 'sourceTermLookup')
        predictionLineStyleDF = pd.read_hdf(store, 'termLineStyleDF')
    stimConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'stimConditionLookup')
    kinConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'kinConditionLookup')
    modelMetadataLookup = pd.read_hdf(transferFuncPath, 'modelMetadataLookup')
    trialMetadataLookup = pd.read_hdf(transferFuncPath, 'trialMetadataLookup')
    #

    def formatModelSpec(infoSrs):
        designShortHand = '(No exogenous)' if infoSrs['design'] == 'NULL' else formulasShortHand[infoSrs['design']]
        selfShort = 'No self' if (infoSrs['selfFormulaDescr'] == 'NULL') else 'self'
        ensShort = 'No ensemble' if (infoSrs['ensembleFormulaDescr'] == 'NULL') else 'ensemble'
        return '({}) + ({}) + {}'.format(selfShort, ensShort, designShortHand)

    modelMetadataLookup.loc[:, 'fullDesignAsLabel'] = modelMetadataLookup.apply(formatModelSpec, axis='columns')

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

    #
    ################ collect estimators and scores
    scoresStack = pd.read_hdf(estimatorPath, 'processedScores')
    # swap, for consistent naming
    scoresStack.loc[:, 'foldType'] = scoresStack.loc[:, 'trialType']
    #
    ssScores = pd.read_hdf(transferFuncPath, 'stateSpaceScores')
    predDF = pd.read_hdf(transferFuncPath, 'stateSpacePredictions')
    groupPagesBy = ['lhsMaskIdx', 'rhsMaskIdx']
    groupSubPagesBy = ['trialType', 'electrode', 'trialRateInHz', 'target']
    plotSingleTrials = False
    if plotSingleTrials:
        predDF = predDF.xs(0, level='conditionUID')
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, arguments['estimatorName'], 'state_space_reconstructions_single_trial'))
    else:
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, arguments['estimatorName'], 'state_space_reconstructions'))
    # with PdfPages(pdfPath) as pdf:

    def plotRoutine(
            inputDF=None,
            inputColorPalette=None, trimColorPalette=True,
            inputLegendUpdates=None, titleText=None,
            inputLineStylePalette=None, trimLinesPalette=True,
            axLims=None, relPlotKWArgs=None, plotContext=None, showFigures=False):
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
        gg.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
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
        for name0, predGroup0 in predDF.groupby(groupPagesBy):
            if not (name0 in modelMetadataLookup.index):
                continue
            else:
                modelMetadata = modelMetadataLookup.loc[name0]
                print('plotting {}'.format(modelMetadata))
            targetsByAverageScore = scoresStack.set_index(groupPagesBy).xs(name0, level=groupPagesBy).groupby('target').mean()['cc'].sort_values().index.to_list()
            #########################################################################################################
            plotScores = ssScores.xs(name0, level=groupPagesBy)
            plotScores.index.names = plotScores.index.names[:-1] + ['referenceTerm']
            plotScores.columns.name = 'comparisonTerm'
            plotScores = plotScores.stack().to_frame(name='cc')
            #
            oldScores = scoresStack.set_index(groupPagesBy).xs(name0, level=groupPagesBy).reset_index(drop=True)
            oldScores.loc[:, 'referenceTerm'] = 'ground_truth'
            oldScores.loc[:, 'comparisonTerm'] = 'original_prediction'
            oldScores.loc[:, 'isTrialAveraged'] = False
            oldScores = oldScores.loc[:, plotScores.index.names + ['cc']].set_index(plotScores.index.names)
            #
            plotScores = pd.concat([plotScores, oldScores]).reset_index()
            plotScores.loc[:, 'xDummy'] = 0
            plotScores.loc[:, 'isTrialAveraged_referenceTerm'] = plotScores.apply(lambda x: '{}_{}'.format(x['isTrialAveraged'], x['referenceTerm']), axis='columns')

            boxPlotKWArgs = dict(whis=np.inf)
            catPlotKWArgs = dict(y='cc', kind='box', margin_titles=True, height=2, aspect=1.5, **boxPlotKWArgs)

            scorePlotMask = (
                (plotScores['referenceTerm'].isin(['prediction', 'ground_truth'])) &
                (plotScores['comparisonTerm'].isin(['original_prediction', 'prediction', 'oneStepKalman', 'inputDriven'])) &
                (plotScores['referenceTerm'] != plotScores['comparisonTerm'])
                ).to_numpy()
            thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores.loc[scorePlotMask, 'trialType'])]
            g = sns.catplot(
                data=plotScores.loc[scorePlotMask, :],
                col='referenceTerm', row='isTrialAveraged',
                x='comparisonTerm',
                hue='trialType', palette=thisPalette.to_dict(),
                **catPlotKWArgs
                )
            g.suptitle(modelMetadata['fullDesignAsLabel'])
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight')
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            scorePlotMask = (
                (plotScores['trialType'].isin(['train', 'test'])) &
                (plotScores['referenceTerm'].isin(['ground_truth'])) &
                (plotScores['comparisonTerm'].isin(['prediction', 'oneStepKalman', 'inputDriven'])) &
                (plotScores['referenceTerm'] != plotScores['comparisonTerm'])
                ).to_numpy()
            thisPalette = sourcePalette.loc[sourcePalette.index.isin(plotScores.loc[scorePlotMask, 'target'])].to_dict()
            g = sns.catplot(
                data=plotScores.loc[scorePlotMask, :],
                col='trialType', row='isTrialAveraged',
                x='comparisonTerm',
                hue='target', hue_order=targetsByAverageScore, palette=thisPalette,
                **catPlotKWArgs
                )
            g.suptitle(modelMetadata['fullDesignAsLabel'])
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight')
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #########################################################################################################
            trialMetadata = trialMetadataLookup.apply(lambda x: predGroup0.index.droplevel([lN for lN in predGroup0.index.names if lN not in trialMetadataLookup.index.names]).map(x), axis='index')
            trialMetadata = pd.concat([trialMetadata, predGroup0.index.to_frame().reset_index(drop=True), ], axis='columns')
            plotMask0 = (
                    (trialMetadata['trialType'] == 'test') &
                    (trialMetadata['target'].isin([targetsByAverageScore[tIdx] for tIdx in [-1, 0]])) &
                    (trialMetadata['trialRateInHz'] != 50)
                ).to_numpy()
            if plotMask0.any():
                plotGroup0 = pd.DataFrame(predGroup0.loc[plotMask0, :])
                plotGroup0.index = pd.MultiIndex.from_frame(trialMetadata.loc[plotMask0, :])
                plotGroup0.columns.name = 'term'
                for name1, plotGroup1 in plotGroup0.groupby(groupSubPagesBy):
                    titleText = '{}\n{} {} {} Hz'.format(modelMetadata['fullDesignAsLabel'], name1[-1], name1[1], name1[2])
                    ## groupSubPagesBy = ['trialType', 'electrode', 'trialRateInHz', 'target']
                    plotGroup1 = plotGroup1.stack().to_frame(name='signal').reset_index()
                    plotGroup1.loc[:, 'predType'] = plotGroup1['term'].map(termPalette.loc[:, ['term']].reset_index().set_index('term')['type'])
                    expandOther = True
                    if expandOther:
                        plotGroup1.loc[plotGroup1['predType'] == 'other', 'predType'] = plotGroup1.loc[plotGroup1['predType'] == 'other', 'term']
                    plotMask1 = (
                            plotGroup1['term'].isin(['ground_truth', 'prediction', 'inputDriven', 'oneStepKalman']) &
                            (plotGroup1['bin'] < 1.)
                    ).to_numpy()
                    gg, currAxLims = plotRoutine(
                        inputDF=plotGroup1.loc[plotMask1, :],
                        inputColorPalette=termPalette, trimColorPalette=True,
                        inputLegendUpdates=prettyNameLookup,
                        inputLineStylePalette=predictionLineStyleDF, trimLinesPalette=True,
                        axLims=None, relPlotKWArgs=None, titleText=titleText,
                        plotContext=pdf, showFigures=arguments['showFigures'])
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')