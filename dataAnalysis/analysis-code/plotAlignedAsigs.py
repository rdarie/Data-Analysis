"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from stats test
    --hueName=hueName                      break down by hue  [default: amplitudeCat]
    --hueControl=hueControl                hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=hueControl              styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from stats test [default: control]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default
import seaborn as sns

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
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")

rowColOpts = asp.processRowColArguments(arguments)

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
    namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
    ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **arguments))
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    metaDataToCategories=False))
if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
#
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(triggeredPath, lazy=arguments['lazy'])
pdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['inputBlockName'],
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(scratchFolder, pdfName + '_stats.h5')
#  Overrides
alignedAsigsKWargs.update({'decimate': 10})
alignedAsigsKWargs.update({'windowSize': (-50e-3, 250e-3)})
statsTestOpts.update({
    'testStride': 10e-3,
    'testWidth': 20e-3,
    'tStop': 250e-3})
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
        statsTestOpts=statsTestOpts)

asp.plotAsigsAligned(
    dataBlock,
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs,
    sigTestResults=sigValsWide,
    figureFolder=figureFolder,
    printBreakDown=True,
    enablePlots=True,
    plotProcFuns=[
        asp.yLabelsEMG, asp.xLabelsTime,
        asp.truncateLegend,
        asp.genXLimSetter(alignedAsigsKWargs['windowSize'])],
    pdfName=pdfName,
    **rowColOpts,
    relplotKWArgs=relplotKWArgs)
asp.plotSignificance(
    sigValsWide,
    pdfName=pdfName + '_pCount',
    figureFolder=figureFolder,
    **rowColOpts,
    **statsTestOpts)

if arguments['lazy']:
    dataReader.close()
