"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --blockIdx=blockIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --plotting                                plot out the correlation matrix? [default: True]
    --inputBlockName=inputBlockName           filename for inputs [default: fr]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                   query what the units will be aligned to? [default: midPeak]
    --selector=selector                       filename if using a unit selector
    --resultName=resultName                   filename for result [default: corr]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName         append a name to the resulting blocks? [default: motion]
    --maskOutlierBlocks                       delete outlier trials? [default: False]
"""

import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
#  import numpy as np
#  import pandas as pd
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
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
#
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(0, 300e-3),
    transposeToColumns='feature', concatOn='columns',
    getMetaData=False,
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchPath, prefix, **arguments)

correlationDF = ash.applyFun(
    triggeredPath=triggeredPath, resultPath=resultPath,
    resultNames=[arguments['resultName']],
    fun="corr", loadType='all', applyType='self',
    lazy=arguments['lazy'],
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs)[0]

#  TODO turn into general pairwise analysis
if arguments['plotting']:
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.use('PS')   # generate postscript output by default
    import seaborn as sns
    import dataAnalysis.plotting.aligned_signal_plots as asp
    
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("talk")
    sns.set_style("white")
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
    #
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['inputBlockName'], arguments['window'],
            arguments['resultName']))
    asp.plotCorrelationMatrix(correlationDF, pdfPath)
