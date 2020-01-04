"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --saveResults                          load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --plotting                             plot results?
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --window=window                        process with short window? [default: long]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: all]
    --selector=selector                    filename if using a unit selector
    --resultName=resultName                filename for result [default: meanFR]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
"""

import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5

from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
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
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultPath = os.path.join(
    alignSubFolder,
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
    namedQueries, alignSubFolder, **arguments)
if arguments['verbose']:
    print('Loading dataBlock: {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
if arguments['verbose']:
    print('Loading alignedAsigs: {}'.format(triggeredPath))
frDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
# reject outlier trials
from scipy.stats import zscore


def findOutliers(
        frDF, sdThresh=6, countThresh=10):
    nOutliers = (frDF.abs().quantile(q=0.9) > sdThresh).sum()
    tooMuch = (nOutliers >= countThresh)
    if tooMuch:
        try:
            print('Found {} outlier channels'.format(nOutliers))
            print(
                    frDF
                    .index
                    .get_level_values('t')
                    .unique()
                )
        except Exception:
            pass
    return nOutliers, tooMuch


testVar = None
groupBy = ['segment', 'originalIndex']
resultNames = ['nOutliers', 'rejectTrial']
outlierTrials = ash.applyFunGrouped(
    frDF.apply(zscore), groupBy, testVar,
    fun=findOutliers, funArgs=[], funKWargs={},
    resultNames=resultNames,
    plotting=False)

if arguments['plotting']:
    import pandas as pd
    outlierCount = pd.concat(
        [
            frDF.index.to_frame().query('bin==0').reset_index(drop=True),
            outlierTrials['nOutliers'].reset_index(drop=True)],
        axis='columns')
    print(outlierCount.loc[outlierCount['all'] > 0, :])
    outlierCount.loc[outlierCount['all'] > 10, :].to_csv(os.path.join(figureFolder, 'outlierTrials.csv'))
print('found {} outlier trials in total'.format(outlierTrials['rejectTrial'].sum()))

if arguments['saveResults']:
    for resName in resultNames:
        outlierTrials[resName].to_hdf(resultPath, resName, format='fixed')