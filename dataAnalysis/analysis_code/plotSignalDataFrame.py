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
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
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
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''

arguments = {
    'analysisName': 'hiRes', 'processAll': True, 'selectionName': 'lfp_CAR', 'datasetName': 'Block_XL_df_ca',
    'window': 'long', 'estimatorName': 'mahal', 'verbose': 2, 'exp': 'exp202101251100',
    'alignFolderName': 'motion', 'showFigures': False, 'blockIdx': '2', 'debugging': False,
    'plotting': True, 'lazy': False}
os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

if __name__ == '__main__':
    idxSl = pd.IndexSlice
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
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
        testStride=200e-3,
        testWidth=200e-3,
        tStart=-400e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        )
    #
    minNObservations = 3
    plotProcFuns = [
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=True,
            # for evoked lfp report
            # xUnitFactor=1e3, yUnitFactor=1,
            # xUnits='msec', yUnits='uV',
            # for evoked emg report
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            ),
        asp.genYLabelChanger(
            lookupDict={}, removeMatch='#0'),
        # asp.genYLimSetter(newLims=[-75, 100], forceLims=True),
        asp.xLabelsTime,
        # asp.genBlockVertShader([
        #         max(0e-3, alignedAsigsKWargs['windowSize'][0]),
        #         min(.9e-3, alignedAsigsKWargs['windowSize'][1])],
        #     asigPlotShadingOpts),
        # asp.genStimVLineAdder(
        #     'RateInHz', vLineOpts, tOnset=0, tOffset=.3, includeRight=False),
        asp.genVLineAdder([0], nrnVLineOpts),
        asp.genLegendRounder(decimals=2),
        ]
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
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    #
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    prf.print_memory_usage('just loaded data, plotting')
    #
    for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
        loadingMeta['arguments'].pop(argName, None)
    arguments.update(loadingMeta['arguments'])
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
    if arguments['overlayStats']:
        figureStatsFolder = os.path.join(
            alignSubFolder, 'figureStats'
            )
        if not os.path.exists(figureStatsFolder):
            os.makedirs(figureStatsFolder, exist_ok=True)
        statsTestPath = os.path.join(figureStatsFolder, pdfName + '_stats.h5')
        if os.path.exists(statsTestPath) and not arguments['recalcStats']:
            sigValsWide = pd.read_hdf(statsTestPath, 'sig')
            sigValsWide.columns.name = 'bin'
        else:
            (
                pValsWide, statValsWide,
                sigValsWide) = ash.facetGridCompareMeansDataFrame(
                    dataDF, statsTestPath,
                    rowColOpts=rowColOpts,
                    limitPages=limitPages,
                    statsTestOpts=statsTestOpts, verbose=arguments['verbose'])
    else:
        sigValsWide = None
    pdfPath = os.path.join(figureOutputFolder, '{}.pdf'.format(pdfName))
    with PdfPages(pdfPath) as pdf:
        for featureMaskIdx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
            maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
            dataGroup = dataDF.loc[:, featureMask]
            print('dataGroup.shape = {}'.format(dataGroup.shape))
            for uIdx, unitName in enumerate(tqdm(dataGroup.columns)):
                g = sns.relplot(
                    x='bin', y='signal',
                    col=colName, row=rowName, hue=hueName,
                    col_order=colOrder, row_order=rowOrder, hue_order=hueOrder,
                    **relplotKWArgs, data=asig)
                #  iterate through plot and add significance stars
                for (ro, co, hu), dataSubset in g.facet_data():
                    #  print('(ro, co, hu) = {}'.format((ro, co, hu)))
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                    #
                    if sigTestResults is not None:
                        addSignificanceStars(
                            g, sigTestResults.query(
                                "unit == '{}'".format(unitName)),
                            ro, co, hu, dataSubset, sigStarOpts=sigStarOpts)
                pdf.savefig()
                plt.close()
        