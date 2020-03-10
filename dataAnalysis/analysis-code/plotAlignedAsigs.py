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
    --enableOverrides                      delete outlier trials? [default: False]
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from stats test
    --hueName=hueName                      break down by hue  [default: amplitude]
    --hueControl=hueControl                hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=styleControl            styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from stats test [default: control]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')   # generate postscript output
# matplotlib.use('QT5Agg')   # generate postscript output

import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")

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
alignedAsigsKWargs['outlierBlocks'] = ash.processOutlierBlocks(
    alignSubFolder, prefix, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    metaDataToCategories=False))
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(triggeredPath, lazy=arguments['lazy'])
pdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['inputBlockName'],
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(figureStatsFolder, pdfName + '_stats.h5')
#############################################
#  Overrides
alignedAsigsKWargs.update({'decimate': 10})
limitPages = None
if arguments['enableOverrides']:
    alignedAsigsKWargs.update({'windowSize': (-1000e-3, 1000e-3)})
    currWindow = rasterOpts['windowSizes'][arguments['window']]
    fullWinSize = currWindow[1] - currWindow[0]
    redWinSize = (
        alignedAsigsKWargs['windowSize'][1] -
        alignedAsigsKWargs['windowSize'][0])
    relplotKWArgs['aspect'] = (
        relplotKWArgs['aspect'] * redWinSize / fullWinSize)
    # statsTestOpts.update({
    #     'testStride': 500e-3,
    #     'testWidth': 500e-3,
    #     'tStart': -2000e-3,
    #     'tStop': 2250e-3})
    
#  End Overrides

#  Get stats results
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
#
asp.plotAsigsAligned(
    dataBlock,
    limitPages=limitPages,
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs,
    sigTestResults=sigValsWide,
    figureFolder=alignedFeaturesFolder,
    printBreakDown=True,
    enablePlots=True,
    plotProcFuns=[
        asp.genYLabelChanger(lookupDict={}, removeMatch='#0'),
        asp.xLabelsTime,
        asp.genVLineAdder(0, vLineOpts),
        asp.genLegendRounder(decimals=2),
        ],
    pdfName=pdfName,
    **rowColOpts,
    relplotKWArgs=relplotKWArgs, sigStarOpts=asigSigStarOpts)
asp.plotSignificance(
    sigValsWide,
    pdfName=pdfName + '_pCount',
    figureFolder=alignedFeaturesFolder,
    **rowColOpts,
    **statsTestOpts)
#
if arguments['lazy']:
    dataReader.file.close()
