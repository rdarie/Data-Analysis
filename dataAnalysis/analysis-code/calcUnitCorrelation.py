"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --plotting                             plot out the correlation matrix? [default: True]
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --window=window                        process with short window? [default: long]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: midPeak]
    --selector=selector                    filename if using a unit selector
    --resultName=resultName                filename for result [default: corr]
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
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=5,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=False,
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)

correlationDF = ash.applyFun(
    triggeredPath=triggeredPath, resultPath=resultPath,
    resultName=arguments['resultName'],
    fun="corr", lazy=arguments['lazy'],
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs)

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

    pdfPath = os.path.join(
        figureFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['inputBlockName'], arguments['window'],
            arguments['resultName']))
    asp.plotCorrelationMatrix(correlationDF, pdfPath)