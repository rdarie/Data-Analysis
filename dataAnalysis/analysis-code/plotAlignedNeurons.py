"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                       which experimental day to analyze
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --processAll                    process entire experimental day? [default: False]
    --lazy                          load from raw, or regular? [default: False]
    --verbose                       print diagnostics? [default: False]
    --window=window                 process with short window? [default: short]
    --unitQuery=unitQuery           how to restrict channels?
    --selector=selector             filename if using a unit selector
    --alignQuery=alignQuery         what will the plot be aligned to? [default: outboundWithStim]
    --rowName=rowName               break down by row  [default: pedalDirection]
    --rowControl=rowControl         rows to exclude from comparison
    --hueName=hueName               break down by hue  [default: amplitudeCat]
    --hueControl=hueControl         hues to exclude from comparison
    --styleName=styleName           break down by style [default: RateInHz]
    --styleControl=hueControl       styles to exclude from stats test
    --colName=colName               break down by col  [default: electrode]
    --colControl=colControl         cols to exclude from comparison [default: control]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default

import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")

import os
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import pandas as pd
import dill as pickle
import pdb
from copy import deepcopy
from currentExperiment import parseAnalysisOptions
from docopt import docopt
from namedQueries import namedQueries
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

rowColOpts = asp.processRowColArguments(arguments)
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    metaDataToCategories=False, removeFuzzyName=True))

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
rasterBlockPath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}.nix'.format(
        arguments['window']))
rasterReader, rasterBlock = ns5.blockFromPath(
    rasterBlockPath, lazy=arguments['lazy'])
frBlockPath = os.path.join(
    scratchFolder,
    prefix + '_fr_{}.nix'.format(
        arguments['window']))
frReader, frBlock = ns5.blockFromPath(
    frBlockPath, lazy=arguments['lazy'])
pdfName = '{}_neurons_{}_{}'.format(
    prefix,
    arguments['window'],
    arguments['alignQuery'])
statsTestOpts.update({'tStop': rasterOpts['windowSizes'][arguments['window']][1]})
statsTestPath = os.path.join(scratchFolder, pdfName + '_stats.h5')
#  Overrides
alignedAsigsKWargs.update({'windowSize': (-10e-3, 60e-3)})
statsTestOpts.update({
    'testStride': 10e-3,
    'testWidth': 5e-3,
    'tStop': 60e-3})
#  End Overrides
if os.path.exists(statsTestPath):
    sigValsWide = pd.read_hdf(statsTestPath, 'sig')
    sigValsWide.columns.name = 'bin'
else:
    alignedAsigsKWargsStats = deepcopy(alignedAsigsKWargs)
    if alignedAsigsKWargsStats['unitNames'] is not None:
        alignedAsigsKWargsStats['unitNames'] = [
            i.replace('_#0', '_fr#0')
            for i in alignedAsigsKWargsStats['unitNames']
        ]
    (
        pValsWide, statValsWide,
        sigValsWide) = ash.facetGridCompareMeans(
        frBlock, statsTestPath,
        loadArgs=alignedAsigsKWargsStats,
        rowColOpts=rowColOpts,
        statsTestOpts=statsTestOpts)

asp.plotNeuronsAligned(
    rasterBlock,
    frBlock,
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs,
    sigTestResults=sigValsWide,
    figureFolder=figureFolder,
    printBreakDown=True,
    enablePlots=True,
    plotProcFuns=[
        asp.xLabelsTime, asp.genLegendRounder(decimals=2),
        asp.yLabelsEMG, asp.genVLineAdder(0, nrnVLineOpts)],
    pdfName=pdfName,
    **rowColOpts,
    twinRelplotKWArgs=nrnRelplotKWArgs)
asp.plotSignificance(
    sigValsWide,
    pdfName=pdfName + '_pCount',
    figureFolder=figureFolder,
    **rowColOpts,
    **statsTestOpts)

if arguments['lazy']:
    frReader.close()
    rasterReader.close()
