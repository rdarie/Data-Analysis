"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                       which experimental day to analyze
    --blockIdx=blockIdx                             which trial to analyze [default: 1]
    --processAll                                    process entire experimental day? [default: False]
    --analysisName=analysisName                     append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName               append a name to the resulting blocks? [default: motion]
    --window=window                                 process with short window? [default: long]
    --lazy                                          load from raw, or regular? [default: False]
    --saveFeatureInfoHTML                           output info from columns [default: False]
    --showFigures                                   load from raw, or regular? [default: False]
    --debugging                                     load from raw, or regular? [default: False]
    --verbose=verbose                               print diagnostics? [default: 0]
    --datasetName=datasetName                       filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName                   filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName                   filename for resulting estimator (cross-validated n_comps)
    --plotSuffix=plotSuffix                         filename for resulting estimator (cross-validated n_comps)
    --enableOverrides                               modify default plot opts? [default: False]
    --unitQuery=unitQuery                           how to restrict channels if not supplying a list?
    --alignQuery=alignQuery                         how to restrict trials
    --individualTraces                              mean+sem or individual traces? [default: False]
    --noStim                                        mean+sem or individual traces? [default: False]
    --limitPages=limitPages                         mean+sem or individual traces?
    --overlayStats                                  overlay ANOVA significance stars? [default: False]
    --recalcStats                                   overlay ANOVA significance stars? [default: False]
    --groupPagesByColumn=groupPagesByColumn         what subset of the data goes in each page? [default: feature]
    --groupPagesByIndex=groupPagesByIndex           what subset of the data goes in each page? [default: all]
    --rowName=rowName                               break down by row  [default: pedalDirection]
    --rowControl=rowControl                         rows to exclude from stats test
    --hueName=hueName                               break down by hue  [default: amplitude]
    --hueControl=hueControl                         hues to exclude from stats test
    --sizeName=sizeName                             break down by hue  [default: RateInHz]
    --sizeControl=sizeControl                       hues to exclude from stats test
    --styleName=styleName                           break down by style [default: RateInHz]
    --styleControl=styleControl                     styles to exclude from stats test
    --colName=colName                               break down by col  [default: electrode]
    --colControl=colControl                         cols to exclude from stats test [default: control]
    --winStart=winStart                             start of window [default: 200]
    --winStop=winStop                               end of window [default: 400]
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
import scipy as scp
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
#
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .4,
        'lines.markersize': .8,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 7,
        "axes.labelsize": 9,
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
    'figure.titlesize': 14,
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True, rc=snsRCParams)
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
    pdfPath = os.path.join(figureOutputFolder, '{}-{}{}.pdf'.format(expDateTimePathStr, pdfName, plotSuffix))
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
        plotProcFuns = plotProcFunsLookup[arguments['plotSuffix']] + [asp.genTitleChanger(prettyNameLookup)]
    else:
        plotProcFuns = [asp.genTitleChanger(prettyNameLookup)]
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
    print('Signal columns are:\n{}'.format(featureInfo))
    if arguments['saveFeatureInfoHTML']:
        featureInfoHtmlPath = os.path.join(figureOutputFolder, '{}_columnsInfo.html'.format(selectionName))
        featureInfo.to_html(featureInfoHtmlPath)
    # hack control time base to look like main time base
    shiftControlTrials = True
    ############
    buggyUUID = trialInfo.loc[(trialInfo['controlFlag'] == 'control') & (trialInfo['bin'] > -0.1).to_numpy(), 'originalIndex'].unique()
    dropBecauseBugMask = trialInfo['originalIndex'].isin(buggyUUID).to_numpy()
    # trialInfo.loc[dropBecauseBugMask, :]
    # trialInfo.loc[(trialInfo['controlFlag'] == 'control') & (trialInfo['bin'] > -0.1).to_numpy(), :]
    # trialInfo.loc[(trialInfo['bin'] > 0.4).to_numpy(), :]
    if shiftControlTrials:
        deltaB = trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'].min() - trialInfo.loc[trialInfo['controlFlag'] == 'main', 'bin'].min()
        trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'] -= deltaB
    dataDF.index = pd.MultiIndex.from_frame(trialInfo)
    dataDF.drop(dataDF.index[dropBecauseBugMask], inplace=True)
    trialInfo.drop(trialInfo.index[dropBecauseBugMask], inplace=True)
    #
    unitNames, unitQuery = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **arguments)
    if unitQuery is not None:
        featureInfo.loc[:, 'chanName'] = featureInfo['feature'].apply(lambda x: '{}#0'.format(x))
        keepIndices = featureInfo.query(unitQuery, engine='python').index
        keepMask = featureInfo.index.isin(keepIndices)
        if not keepMask.any():
            raise(Exception('query {} did not produce any results\n available features are:\n{}'.format(unitQuery, featureInfo['chanName'])))
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
    # pdb.set_trace()
    if tBins.shape[0] > targetFrameLen:
        skipFactor = int(np.ceil(tBins.shape[0] // targetFrameLen))
        tMask2 = trialInfo['bin'].isin(tBins[::skipFactor])
        dataDF = dataDF.iloc[tMask2.to_numpy(), :]
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        del tMask2
    #
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz'],
        'stimConditionWithDate': ['electrode', 'expName'],
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
    # ((trialInfo['trialRateInHz'] == 0) & (trialInfo['electrode'] != 'NA'))
    # trialInfo.loc[, :]
    if arguments['plotSuffix'] in customCodeLookup:
        try:
            exec(customCodeLookup[arguments['plotSuffix']])
        except Exception:
            traceback.print_exc()
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
            estimator=None, units='trialUID', alpha=0.5))
        plotProcFuns.append(
            asp.genTraceAnnotator(
                unit_var='trialUID', labelsList=[
                    'expName', 'segment',
                    'originalIndex', 't'],
                textOpts=dict(
                    ha='left', va='bottom', fontsize=2,
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
    if 'facet_kws' in relplotKWArgs:
        relplotKWArgs['facet_kws'].update(dict(margin_titles=True))
    else:
        relplotKWArgs['facet_kws'] = dict(margin_titles=True)
    with PdfPages(pdfPath) as pdf:
        pageCount = 0
        ###
        if 'hueName' in rowColOpts:
            hueVar = rowColOpts['hueName']
        elif 'hue' in relplotKWArgs:
            hueVar = relplotKWArgs['hue']
        else:
            hueVar = None
        if hueVar is not None:
            relplotKWArgs['hue'] = hueVar
            #
            if 'palette' in rowColOpts:
                thePaletteStr = rowColOpts['palette']
            elif 'palette' in relplotKWArgs:
                thePaletteStr = relplotKWArgs['palette']
            else:
                thePaletteStr = "ch:0.8,-.3,dark=.25,light=0.75,reverse=1"
            uniqHues = sorted(dataDF.groupby(hueVar).groups.keys())
            thePalette = pd.Series(
                sns.color_palette(
                    palette=thePaletteStr, n_colors=len(uniqHues)),
                index=uniqHues)
            relplotKWArgs['palette'] = thePalette.to_dict()
            relplotKWArgs['hue_order'] = uniqHues
        if arguments['groupPagesByColumn'] == 'all':
            colGrouper = [('all', dataDF)]
        else:
            groupPagesByColumn = arguments['groupPagesByColumn'].split(', ')
            if len(groupPagesByColumn) == 1:
                groupPagesByColumn = groupPagesByColumn[0]
            colGrouper = dataDF.groupby(groupPagesByColumn, axis='columns', sort=False)
        if not (arguments['groupPagesByIndex'] == 'all'):
            groupPagesByIndex = arguments['groupPagesByIndex'].split(', ')
            if len(groupPagesByIndex) == 1:
                groupPagesByIndex = groupPagesByIndex[0]
        dataDF = dataDF.droplevel([
            'controlFlag', 'segment', 'originalIndex', 't', 'trialRateInHz',
            'stimCat', 'electrode', 'pedalDirection', 'pedalMovementCat', 'pedalMetaCat',
            'bin', 'trialUID', 'conditionUID',
            'stimCondition', 'stimConditionWithDate', 'kinematicCondition',
            'kinematicConditionNoSize'])
        for colGroupName, colGroup in colGrouper:
            if arguments['groupPagesByIndex'] == 'all':
                idxGrouper = [('all', colGroup)]
            else:
                idxGrouper = colGroup.groupby(groupPagesByIndex, sort=False)
            if arguments['plotSuffix'] in shareyAcrossPagesLookup:
                shareyAcrossPages = shareyAcrossPagesLookup[arguments['plotSuffix']]
            else:
                shareyAcrossPages = False
            if shareyAcrossPages:
                smallestGroupNames = ['bin']
                if hueVar is not None:
                    smallestGroupNames.append(hueVar)
                for axn in ['row', 'col']:
                    if rowColOpts['{}Name'.format(axn)] is not None:
                        smallestGroupNames.append(rowColOpts['{}Name'.format(axn)])
                estFun = relplotKWArgs['estimator'] if 'estimator' in relplotKWArgs else np.mean
                if estFun == 'mean':
                    estFun = np.mean
                if 'errorbar' in relplotKWArgs:
                    if relplotKWArgs['errorbar'] == 'se':
                        errFun = scp.stats.sem
                    elif relplotKWArgs['errorbar'] == 'sd':
                        errFun = np.std
                    elif relplotKWArgs['errorbar'] is None:
                        errFun = None
                else:
                    errFun = scp.stats.sem
                if errFun is not None:
                    maxSem = colGroup.groupby(smallestGroupNames).apply(errFun).max().max()
                else:
                    maxSem = 0.
                newLims = [
                    colGroup.groupby(smallestGroupNames).apply(estFun).min().min() - maxSem,
                    colGroup.groupby(smallestGroupNames).apply(estFun).max().max() + maxSem
                    ]
                yLimProcFun = [asp.genYLimSetter(newLims=newLims, forceLims=True)]
                if 'facet_kws' in relplotKWArgs:
                    relplotKWArgs['facet_kws'].update(dict(sharey=False))
                else:
                    relplotKWArgs['facet_kws'] = dict(sharey=False)
            else:
                yLimProcFun = []
            for idxGroupName, idxGroup in idxGrouper:
                plotDF = idxGroup.stack(level=idxGroup.columns.names).reset_index(name='signal')
                # plotDF = dataDF.loc[:, colGroupName].reset_index(name='signal')
                print('plotDF.columns = {}'.format(plotDF.columns))
                rowColArgs = {}
                for axn in ['row', 'col']:
                    if rowColOpts['{}Name'.format(axn)] is not None:
                        rowColArgs[axn] = rowColOpts['{}Name'.format(axn)]
                        if '{}Order'.format(axn) in rowColOpts:
                            rowColArgs['{}_order'.format(axn)] = [n for n in rowColOpts['{}Order'.format(axn)] if n in plotDF[rowColArgs[axn]].to_list()]
                        else:
                            rowColArgs['{}_order'.format(axn)] = [n for n in np.unique(plotDF[rowColArgs[axn]])]
                g = sns.relplot(
                    x='bin', y='signal',
                    **rowColArgs, **relplotKWArgs, data=plotDF)
                ##
                # why is this info missing??
                g._hue_var = hueVar
                if 'palette' in relplotKWArgs:
                    g.hue_names = [hN for hN in relplotKWArgs['palette'].keys()]
                    g.hue_kws['palette'] = relplotKWArgs['palette']
                if 'hue_order' in relplotKWArgs:
                    g.hue_kws['hue_order'] = relplotKWArgs['hue_order']
                ##
                if arguments['plotSuffix'] in titlesOptsLookup:
                    titlesOpts = titlesOptsLookup[arguments['plotSuffix']]
                    g.set_titles(**titlesOpts)
                #  iterate through plot and add significance stars
                for (ro, co, hu), dataSubset in g.facet_data():
                    # print('(ro, co, hu) = {}'.format((ro, co, hu)))
                    if len(plotProcFuns):
                        for procFun in yLimProcFun + plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                    #
                    if sigTestResults is not None:
                        unitMask = (
                                sigTestResults.reset_index().set_index(dataDF.columns.names).index == colGroupName)
                        asp.addSignificanceStars(
                            g, sigTestResults.loc[unitMask, :],
                            ro, co, hu, dataSubset, sigStarOpts=asigSigStarOpts)
                #
                if arguments['plotSuffix'] in xAxisLabelLookup:
                    xAxisLabel = xAxisLabelLookup[arguments['plotSuffix']]
                elif g._x_var in prettyNameLookup:
                    xAxisLabel = prettyNameLookup[g._x_var]
                else:
                    xAxisLabel = g._x_var
                if arguments['plotSuffix'] in xAxisUnitsLookup:
                    xAxisLabel += ' {}'.format(xAxisUnitsLookup[arguments['plotSuffix']])
                if arguments['plotSuffix'] in yAxisLabelLookup:
                    thisYLookup = yAxisLabelLookup[arguments['plotSuffix']]
                    if isinstance(thisYLookup, dict):
                        if colGroupName in thisYLookup:
                            yAxisLabel = thisYLookup[colGroupName]
                        else:
                            yAxisLabel = '{}'.format(colGroupName)
                    else:
                        yAxisLabel = '{}'.format(thisYLookup)
                elif colGroupName in prettyNameLookup:
                    yAxisLabel = prettyNameLookup[colGroupName]
                elif g._y_var in prettyNameLookup:
                    yAxisLabel = prettyNameLookup[g._y_var]
                else:
                    yAxisLabel = g._y_var
                if arguments['plotSuffix'] in yAxisUnitsLookup:
                    yAxisLabel += ' {}'.format(yAxisUnitsLookup[arguments['plotSuffix']])
                g.set_axis_labels(xAxisLabel, yAxisLabel)
                if arguments['plotSuffix'] in titleTextLookup:
                    titleText = titleTextLookup[arguments['plotSuffix']]
                    g.suptitle(titleText)
                else:
                    g.suptitle('{} {}'.format(colGroupName, idxGroupName))
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
                print('Plotted page {}: {}, {}'.format(pageCount, colGroupName, idxGroupName))
                if limitPages is not None:
                    if pageCount > limitPages:
                        break