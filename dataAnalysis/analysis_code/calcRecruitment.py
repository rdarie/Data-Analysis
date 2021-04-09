"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")

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
from scipy import stats
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName']
    )
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
    namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
    ash.processUnitQueryArgs(
        namedQueries, analysisSubFolder, **arguments))
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    metaDataToCategories=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(0, 300e-3),
    transposeToColumns='bin', concatOn='index',))
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
#  Overrides
limitPages = None
resultName = 'meanRAUC'
funKWargs = dict(
    baseline='mean',
    tStart=None, tStop=None)
#  End Overrides
asigWide = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)

rAUCDF = ash.rAUC(
    asigWide, **funKWargs).to_frame(name='rauc')
rAUCDF['kruskalStat'] = np.nan
rAUCDF['kruskalP'] = np.nan
for name, group in rAUCDF.groupby(['electrode', 'feature']):
    subGroups = [i['rauc'].to_numpy() for n, i in group.groupby('nominalCurrent')]
    try:
        stat, pval = stats.kruskal(*subGroups, nan_policy='omit')
        rAUCDF.loc[group.index, 'kruskalStat'] = stat
        rAUCDF.loc[group.index, 'kruskalP'] = pval
    except Exception:
        rAUCDF.loc[group.index, 'kruskalStat'] = 0
        rAUCDF.loc[group.index, 'kruskalP'] = 1

rAUCDF.to_hdf(resultPath, resultName, format='table')

"""
RecCurve = ash.facetGridApplyFunGrouped(
    dataBlock, resultPath,
    fun=ash.rAUC, funArgs=[], funKWargs=funKWargs,
    resultNames=resultNames,
    limitPages=limitPages,
    loadArgs=alignedAsigsKWargs,
    rowColOpts=rowColOpts, verbose=arguments['verbose'])
#
for rName in resultNames:
    tempResDict = {}
    for uName, uResults in RecCurve.items():
        tempResDict.update({uName: uResults[rName]})
    tempResDF = pd.concat(tempResDict)
    tempResDF.index.names = ['unit'] + uResults[rName].index.names
    tempResDF.to_hdf(resultPath, rName, format='table')
"""
if arguments['lazy']:
    dataReader.file.close()