"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierBlocks                  delete everything *except* outlier trials? [default: False]
    --enableOverrides                      modify default plot opts? [default: False]
    --individualTraces                     mean+sem or individual traces? [default: False]
    --overlayStats                         overlay ANOVA significance stars? [default: False]
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
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --limitPages=limitPages                how many pages to print, max?
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Agg')   # generate postscript output
matplotlib.use('QT5Agg')   # generate postscript output


from namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
figureStatsFolder = os.path.join(
    alignSubFolder, 'figureStats'
    )
if not os.path.exists(figureStatsFolder):
    os.makedirs(figureStatsFolder, exist_ok=True)
#
alignedFeaturesFolder = os.path.join(
    figureFolder, arguments['analysisName'],
    'alignedFeatures')
if not os.path.exists(alignedFeaturesFolder):
    os.makedirs(alignedFeaturesFolder, exist_ok=True)

calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
rowColOpts = asp.processRowColArguments(arguments)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
    namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
    ash.processUnitQueryArgs(
        namedQueries, analysisSubFolder, **arguments))
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    calcSubFolder, prefix, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    metaDataToCategories=False))

triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
pdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['inputBlockName'],
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(figureStatsFolder, pdfName + '_stats.h5')
#############################################
#  Overrides
alignedAsigsKWargs.update({'amplitudeColumn': arguments['hueName']})
if arguments['limitPages'] is not None:
    limitPages = int(arguments['limitPages'])
else:
    limitPages = None

minNObservations = 6
if arguments['individualTraces']:
    relplotKWArgs['estimator'] = None
    relplotKWArgs['units'] = 't'
    pdfName += '_traces'
    # rowColOpts['hueName'] = 't'
    # relplotKWArgs['palette'] = "ch:0,6,dark=.3,light=0.7,reverse=1"
if arguments['invertOutlierBlocks']:
    pdfName += '_outliers'
if arguments['enableOverrides']:
    relplotKWArgs.update({
        'legend': 'brief',
        'height': 4,
        'aspect': 2,
        'facet_kws': {
            'sharey': False,
            # 'legend_out': False,
            'gridspec_kws': {
                'wspace': 0.01,
                'hspace': 0.01
            }}
    })
    if 'rowColOverrides' in locals():
        if rowColOpts['colName'] in rowColOverrides:
            rowColOpts['colOrder'] = rowColOverrides[rowColOpts['colName']]
    ##########################################################################

    # alignedAsigsKWargs.update({'windowSize': (-.2, .4)})
    alignedAsigsKWargs.update({
        'windowSize': (
            float(arguments['winStart']) * (-1e-3),
            float(arguments['winStop']) * 1e-3
            )})
    ##########################################################################
    currWindow = rasterOpts['windowSizes'][arguments['window']]
    fullWinSize = currWindow[1] - currWindow[0]
    if 'windowSize' not in alignedAsigsKWargs:
        alignedAsigsKWargs['windowSize'] = rasterOpts['windowSizes'][arguments['window']]
    redWinSize = (
        alignedAsigsKWargs['windowSize'][1] -
        alignedAsigsKWargs['windowSize'][0])
    # relplotKWArgs['aspect'] = (
    #     relplotKWArgs['aspect'] * redWinSize / fullWinSize)
    statsTestOpts.update({
        'testStride': 50e-3,
        'testWidth': 50e-3,
        'tStart': 0e-3,
        'tStop': alignedAsigsKWargs['windowSize'][1]})
#  End Overrides

#  Get stats results
if arguments['overlayStats']:
    if os.path.exists(statsTestPath):
        sigValsWide = pd.read_hdf(statsTestPath, 'sig')
        sigValsWide.columns.name = 'bin'
    else:
        (
            pValsWide, statValsWide,
            sigValsWide) = ash.facetGridCompareMeans(
            dataBlock, statsTestPath,
            loadArgs=alignedAsigsKWargs,
            rowColOpts=rowColOpts,
            limitPages=limitPages,
            statsTestOpts=statsTestOpts)
else:
    sigValsWide = None
#
# import warnings
# warnings.filterwarnings("error")
asp.plotAsigsAligned(
    dataBlock,
    limitPages=limitPages,
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs,
    sigTestResults=sigValsWide,
    figureFolder=alignedFeaturesFolder,
    enablePlots=True,
    minNObservations = minNObservations,
    plotProcFuns=[
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=False,
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
        ],
    pdfName=pdfName,
    **rowColOpts,
    relplotKWArgs=relplotKWArgs, sigStarOpts=asigSigStarOpts)
if arguments['overlayStats']:
    asp.plotSignificance(
        sigValsWide,
        pdfName=pdfName + '_pCount',
        figureFolder=alignedFeaturesFolder,
        **rowColOpts,
        **statsTestOpts)
#
if arguments['lazy']:
    dataReader.file.close()
