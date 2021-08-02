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
    --enableOverrides                      modify default plot opts? [default: False]
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
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
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


print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
arguments['verbose'] = int(arguments['verbose'])
'''

arguments = {
    'styleControl': '', 'winStart': '-200', 'processAll': True, 'limitPages': None,
    'hueControl': '', 'estimatorName': None, 'datasetName': 'Block_XL_df_rc',
    'alignFolderName': 'motion', 'verbose': '0', 'enableOverrides': True,
    'selectionName': 'lfp_CAR', 'lazy': False, 'debugging': False, 'hueName': 'trialAmplitude',
    'rowName': 'pedalMovementCat', 'rowControl': '', 'winStop': '500', 'blockIdx': '2', 'exp': 'exp202101271100',
    'individualTraces': False, 'overlayStats': True, 'recalcStats': True, 'colControl': '', 'noStim': True,
    'sizeName': '', 'sizeControl': '', 'styleName': '', 'window': 'XL', 'colName': 'electrode',
    'showFigures': False, 'analysisName': 'hiRes'}
os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

if __name__ == '__main__':
    idxSl = pd.IndexSlice
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    from dataAnalysis.analysis_code.new_plot_options import *
    #
    arguments['verbose'] = int(arguments['verbose'])
    #
    if 'rowColOverrides' in locals():
        arguments['rowColOverrides'] = rowColOverrides
    #############################################
    #  room for custom code
    #############################################
    #
    statsTestOpts = dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        )
    #
    minNObservations = 3
    plotProcFuns = [
        asp.xLabelsTime,
        asp.genVLineAdder([0], vLineOpts),
        asp.genLegendRounder(decimals=2),
        asp.genNumRepAnnotator(
            hue_var=arguments['hueName'], unit_var='trialUID',
            xpos=0.05, ypos=.95, textOpts=dict(
                ha='left', va='top',
                c=(0., 0., 0., 0.7),
                bbox=dict(
                    boxstyle="square",
                    ec=(0., 0., 0., 0.),
                    fc=(1., 1., 1., 0.2))
            )),
        ]
    unusedPlotProcFuns = [
        # asp.genBlockVertShader([
        #         max(0e-3, alignedAsigsKWargs['windowSize'][0]),
        #         min(.9e-3, alignedAsigsKWargs['windowSize'][1])],
        #     asigPlotShadingOpts),
        # asp.genStimVLineAdder(
        #     'RateInHz', vLineOpts, tOnset=0, tOffset=.3, includeRight=False),
        # asp.genYLimSetter(newLims=[-75, 100], forceLims=True),
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=True,
            # for evoked lfp report
            # xUnitFactor=1e3, yUnitFactor=1,
            # xUnits='msec', yUnits='uV',
            # for evoked emg report
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            ),
        asp.genTraceAnnotator(
            unit_var='trialUID', labelsList=['segment', 't'],
            textOpts=dict(ha='left', va='bottom', fontsize=4))
        ]
    titlesOpts = dict(
        col_template="{col_var}\n{col_name}",
        row_template="{row_var}\n{row_name}")
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    #
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
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
        for argName in ['plotting', 'showFigures', 'debugging', 'verbose', 'winStart', 'winStop']:
            loadingMeta['arguments'].pop(argName, None)
        arguments.update(loadingMeta['arguments'])
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    #
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
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
        'kinematicCondition': ['pedalDirection', 'pedalSizeCat', 'pedalMovementCat']
        }
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        trialInfo.loc[:, canName] = compoundAnn
    dataDF.index = pd.MultiIndex.from_frame(trialInfo)

    prf.print_memory_usage('just loaded data, plotting')
    #
    rowColOpts = asp.processRowColArguments(arguments)
    if arguments['limitPages'] is not None:
        limitPages = int(arguments['limitPages'])
    else:
        limitPages = None
    #  Get stats results?
    pdfName = '{}_{}'.format(
        datasetName, selectionName)
    #
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
    pdfPath = os.path.join(figureOutputFolder, '{}.pdf'.format(pdfName))
    with PdfPages(pdfPath) as pdf:
        pageCount = 0
        for featureMaskIdx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
            maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
            dataGroup = dataDF.loc[:, featureMask]
            print('{}\ndataGroup.shape = {}\n\n'.format(maskParams, dataGroup.shape))
            for uIdx, unitName in enumerate(tqdm(dataGroup.columns)):
                plotDF = dataGroup.loc[:, unitName].reset_index(name='signal')
                rowColArgs = {}
                for axn in ['row', 'col', 'hue']:
                    if rowColOpts['{}Name'.format(axn)] is not None:
                        rowColArgs[axn] = rowColOpts['{}Name'.format(axn)]
                        rowColArgs['{}_order'.format(axn)] = np.unique(plotDF[rowColArgs[axn]])
                g = sns.relplot(
                    x='bin', y='signal',
                    facet_kws=dict(margin_titles=True),
                    **rowColArgs, **relplotKWArgs, data=plotDF)
                #  iterate through plot and add significance stars
                for (ro, co, hu), dataSubset in g.facet_data():
                    #  print('(ro, co, hu) = {}'.format((ro, co, hu)))
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                    #
                    if sigTestResults is not None:
                        unitMask = sigTestResults.reset_index().set_index(dataGroup.columns.names).index == unitName
                        asp.addSignificanceStars(
                            g, sigTestResults.loc[unitMask, :],
                            ro, co, hu, dataSubset, sigStarOpts=asigSigStarOpts)
                xAxisLabel = 'time (msec)'
                yAxisLabel = unitName[0]
                g.set_axis_labels(xAxisLabel, yAxisLabel)
                g.set_titles(**titlesOpts)
                titleText = ''
                g.suptitle(titleText)
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