"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --lazy                                 load from raw, or regular? [default: False]
    --showFigures                          load from raw, or regular? [default: False]
    --debugging                            load from raw, or regular? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
    --plotSuffix=plotSuffix                filename for resulting estimator (cross-validated n_comps)
    --enableOverrides                      modify default plot opts? [default: False]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list?
    --alignQuery=alignQuery                how to restrict trials
    --individualTraces                     mean+sem or individual traces? [default: False]
    --noStim                               mean+sem or individual traces? [default: False]
    --limitPages=limitPages                mean+sem or individual traces?
    --overlayStats                         overlay ANOVA significance stars? [default: False]
    --recalcStats                          overlay ANOVA significance stars? [default: False]
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from stats test
    --hueName=hueName                      break down by hue  [default: amplitude]
    --hueControl=hueControl                hues to exclude from stats test
    --sizeName=sizeName                    break down by hue  [default: RateInHz]
    --sizeControl=sizeControl              hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=styleControl            styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from stats test [default: control]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
"""

import logging
logging.captureWarnings(True)
import matplotlib, os, sys
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
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from dataAnalysis.analysis_code.plotSignalDataFrame_options import *
import pdb
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from docopt import docopt
for arg in sys.argv:
    print(arg)
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 9,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
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
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV


print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''

arguments = {
    'rowName': 'pedalDirection', 'showFigures': False, 'exp': 'exp202101271100', 'analysisName': 'hiRes',
    'individualTraces': False, 'styleControl': None, 'verbose': '1', 'sizeName': 'RateInHz', 'alignFolderName': 'motion',
    'datasetName': 'Block_XL_df_pa', 'noStim': False, 'recalcStats': False, 'winStop': '400', 'estimatorName': None,
    'rowControl': None, 'window': 'XL', 'limitPages': None, 'winStart': '200', 'hueName': 'amplitude',
    'colControl': 'control', 'enableOverrides': True, 'processAll': True, 'colName': 'electrode', 'overlayStats': False,
    'alignQuery': None, 'styleName': 'RateInHz', 'sizeControl': None, 'blockIdx': '2', 'unitQuery': None, 'lazy': False,
    'hueControl': None, 'selectionName': 'rig', 'debugging': False, 'plotSuffix': 'rig_illustration'}
os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

if __name__ == '__main__':
    idxSl = pd.IndexSlice
    styleOpts = {
        'legend.lw': 2,
        'tight_layout.pad': 3e-1, # units of font size
        'panel_heading.pad': 0.
        }
    if arguments['plotSuffix'] in styleOptsLookup:
        styleOpts.update(styleOptsLookup[arguments['plotSuffix']])
    arguments['verbose'] = int(arguments['verbose'])
    if arguments['plotSuffix'] in argumentsLookup:
        arguments.update(argumentsLookup[arguments['plotSuffix']])
    #
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    from dataAnalysis.analysis_code.new_plot_options import *
    #
    arguments['verbose'] = int(arguments['verbose'])
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    #
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
    pdfName = '{}_{}'.format(
        datasetName, selectionName)
    if arguments['plotSuffix'] is not None:
        plotSuffix = '_{}'.format(arguments['plotSuffix'])
    else:
        plotSuffix = ''
    pdfPath = os.path.join(figureOutputFolder, '{}{}.pdf'.format(pdfName, plotSuffix))
    #
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    if arguments['plotSuffix'] in statsTestOptsLookup:
        statsTestOpts = statsTestOptsLookup[arguments['plotSuffix']]
        # else uses default defined in plotSignalDataFrame_options
    if arguments['plotSuffix'] in plotProcFunsLookup:
        plotProcFuns = plotProcFunsLookup[arguments['plotSuffix']]
    else:
        plotProcFuns = []
    #
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
        iteratorsBySegment = loadingMeta['iteratorsBySegment']
        cvIterator = iteratorsBySegment[0]
        # cv_kwargs = loadingMeta['cv_kwargs']
        if 'normalizeDataset' in loadingMeta:
            normalizeDataset = loadingMeta['normalizeDataset']
            unNormalizeDataset = loadingMeta['unNormalizeDataset']
            normalizationParams = loadingMeta['normalizationParams']
        else:
            normalizeDataset = None
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
    #
    unitNames, unitQuery = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **arguments)
    if unitQuery is not None:
        featureInfo.loc[:, 'chanName'] = featureInfo['feature'].apply(lambda x: '{}#0'.format(x))
        keepIndices = featureInfo.query(unitQuery, engine='python').index
        keepMask = featureInfo.index.isin(keepIndices)
        if not keepMask.any():
            raise(Exception('query {} did not produce any results'.format(unitQuery)))
        else:
            dataDF = dataDF.loc[:, keepMask]
            featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
    #
    alignQuery = ash.processAlignQueryArgs(
        namedQueries, alignQuery=arguments['alignQuery'])
    if alignQuery is not None:
        keepIndices = trialInfo.query(alignQuery, engine='python').index
        keepMask = trialInfo.index.isin(keepIndices)
        if not keepMask.any():
            raise(Exception('query {} did not produce any results'.format(alignQuery)))
        else:
            dataDF = dataDF.loc[keepMask, :]
            trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    #
    tMask = pd.Series(True, index=trialInfo.index)
    if arguments['winStop'] is not None:
        tMask = tMask & (trialInfo['bin'] < (float(arguments['winStop']) * 1e-3))
    if arguments['winStart'] is not None:
        tMask = tMask & (trialInfo['bin'] >= (float(arguments['winStart']) * 1e-3))
    if not tMask.all():
        dataDF = dataDF.iloc[tMask.to_numpy(), :]
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        del tMask
    #
    tBins = trialInfo['bin'].unique()
    targetFrameLen = 2 * useDPI * relplotKWArgs['height'] * relplotKWArgs['aspect'] # nominal num. points per facet
    if tBins.shape[0] > targetFrameLen:
        skipFactor = tBins.shape[0] // targetFrameLen
        tMask2 = trialInfo['bin'].isin(tBins[::skipFactor])
        dataDF = dataDF.iloc[tMask2.to_numpy(), :]
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        del tMask2
    #
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz', ],
        'kinematicCondition': ['pedalDirection', 'pedalSizeCat', 'pedalMovementCat'],
        'kinematicConditionNoSize': ['pedalDirection', 'pedalMovementCat']
        }
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        trialInfo.loc[:, canName] = compoundAnn
    dataDF.index = pd.MultiIndex.from_frame(trialInfo)
    ######
    if arguments['plotSuffix'] in customCodeLookup:
        exec(customCodeLookup[arguments['plotSuffix']])
    prf.print_memory_usage('just loaded data, plotting')
    #
    rowColOpts = asp.processRowColArguments(arguments)
    if arguments['limitPages'] is not None:
        limitPages = int(arguments['limitPages'])
    else:
        limitPages = None
    if arguments['individualTraces']:
        pdfName += '_traces'
        relplotKWArgs.update(dict(
            estimator=None, units='trialUID', alpha=0.7))
        plotProcFuns.append(
            asp.genTraceAnnotator(
                unit_var='trialUID', labelsList=['segment', 't'],
                textOpts=dict(
                    ha='center', va='top', fontsize=5,
                    c=(0., 0., 0., 0.7),
                    bbox=dict(
                        boxstyle="square",
                        ec=(0., 0., 0., 0.), fc=(1., 1., 1., 0.2))
                )))
    #  Get stats results?
    if arguments['overlayStats']:
        figureStatsFolder = os.path.join(
            alignSubFolder, 'figureStats'
            )
        if not os.path.exists(figureStatsFolder):
            os.makedirs(figureStatsFolder, exist_ok=True)
        statsTestPath = os.path.join(figureStatsFolder, pdfName + '_stats.h5')
        if os.path.exists(statsTestPath) and not arguments['recalcStats']:
            sigTestResults = pd.read_hdf(statsTestPath, 'sig')
            sigTestResults.columns.name = 'bin'
        else:
            (
                pValsWide, statValsWide,
                sigTestResults) = ash.facetGridCompareMeansDataFrame(
                    dataDF, statsTestPath,
                    rowColOpts=rowColOpts,
                    limitPages=limitPages,
                    statsTestOpts=statsTestOpts, verbose=arguments['verbose'])
    else:
        sigTestResults = None
    if arguments['plotSuffix'] in relPlotKWArgsLookup:
        relplotKWArgs.update(relPlotKWArgsLookup[arguments['plotSuffix']])
    with PdfPages(pdfPath) as pdf:
        pageCount = 0
        for uIdx, unitName in enumerate(tqdm(dataDF.columns)):
            plotDF = dataDF.loc[:, unitName].reset_index(name='signal')
            rowColArgs = {}
            for axn in ['row', 'col', 'hue']:
                if rowColOpts['{}Name'.format(axn)] is not None:
                    rowColArgs[axn] = rowColOpts['{}Name'.format(axn)]
                    if '{}Order'.format(axn) in rowColOpts:
                        rowColArgs['{}_order'.format(axn)] = rowColOpts['{}Order'.format(axn)]
                    else:
                        rowColArgs['{}_order'.format(axn)] = np.unique(plotDF[rowColArgs[axn]])
            g = sns.relplot(
                x='bin', y='signal',
                **rowColArgs, **relplotKWArgs, data=plotDF,
                facet_kws=dict(margin_titles=True))
            #  iterate through plot and add significance stars
            for (ro, co, hu), dataSubset in g.facet_data():
                #  print('(ro, co, hu) = {}'.format((ro, co, hu)))
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
                #
                if sigTestResults is not None:
                    unitMask = (
                            sigTestResults.reset_index().set_index(dataDF.columns.names).index == unitName)
                    asp.addSignificanceStars(
                        g, sigTestResults.loc[unitMask, :],
                        ro, co, hu, dataSubset, sigStarOpts=asigSigStarOpts)

            if arguments['plotSuffix'] in xAxisLabelLookup:
                xAxisLabel = xAxisLabelLookup[arguments['plotSuffix']]
            else:
                xAxisLabel = 'Time (msec)'
            if arguments['plotSuffix'] in yAxisLabelLookup:
                thisYLookup = yAxisLabelLookup[arguments['plotSuffix']]
                if isinstance(thisYLookup, dict):
                    if unitName[0] in thisYLookup:
                        yAxisLabel = thisYLookup[unitName[0]]
                    else:
                        yAxisLabel = '{}'.format(unitName[0])
                else:
                    yAxisLabel = '{}'.format(thisYLookup)
            else:
                yAxisLabel = '{}'.format(unitName[0])
            g.set_axis_labels(xAxisLabel, yAxisLabel)
            if arguments['plotSuffix'] in titlesOptsLookup:
                titlesOpts = titlesOptsLookup[arguments['plotSuffix']]
                g.set_titles(**titlesOpts)
            if arguments['plotSuffix'] in titleTextLookup:
                titleText = titleTextLookup[arguments['plotSuffix']]
                g.suptitle(titleText)
            if arguments['plotSuffix'] in legendTitleOverridesLookup:
                legendTitleOverrides = legendTitleOverridesLookup[arguments['plotSuffix']]
            else:
                legendTitleOverrides = {}
            if arguments['plotSuffix'] in legendContentOverridesLookup:
                legendContentOverrides = legendContentOverridesLookup[arguments['plotSuffix']]
            else:
                legendContentOverrides = {}
            asp.reformatFacetGridLegend(
                g, titleOverrides=legendTitleOverrides,
                contentOverrides=legendContentOverrides,
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
            pageCount += 1
            if limitPages is not None:
                if pageCount > limitPages:
                    break