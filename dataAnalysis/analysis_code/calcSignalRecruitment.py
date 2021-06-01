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
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --winStart=winStart                    start of absolute window (when loading)
    --winStop=winStop                      end of absolute window (when loading)
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
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)

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
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchFolder, blockBaseName, **arguments)
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    metaDataToCategories=False,
    removeFuzzyName=False,
    decimate=1,
    transposeToColumns='bin', concatOn='index'))
#
alignedAsigsKWargs['procFun'] = ash.genDetrender(
    timeWindow=(-200e-3, -100e-3))

if 'windowSize' not in alignedAsigsKWargs:
    alignedAsigsKWargs['windowSize'] = [ws for ws in rasterOpts['windowSizes'][arguments['window']]]
if 'winStart' in arguments:
    if arguments['winStart'] is not None:
        alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
if 'winStop' in arguments:
    if arguments['winStop'] is not None:
        alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    blockBaseName + '{}_{}_rauc.h5'.format(
        inputBlockSuffix, arguments['window']))

print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
#  Overrides
limitPages = None
funKWargs = dict(
    # baseline='mean',
    tStart=-200e-3, tStop=200e-3)
#  End Overrides
asigWide = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)

rAUCDF = ash.rAUC(
    asigWide, **funKWargs).to_frame(name='rauc')
rAUCDF['kruskalStat'] = np.nan
rAUCDF['kruskalP'] = np.nan
for name, group in rAUCDF.groupby(['electrode', 'feature']):
    subGroups = [i['rauc'].to_numpy() for n, i in group.groupby(amplitudeFieldName)]
    try:
        stat, pval = stats.kruskal(*subGroups, nan_policy='omit')
        rAUCDF.loc[group.index, 'kruskalStat'] = stat
        rAUCDF.loc[group.index, 'kruskalP'] = pval
    except Exception:
        rAUCDF.loc[group.index, 'kruskalStat'] = 0
        rAUCDF.loc[group.index, 'kruskalP'] = 1

derivedAnnot = [
    'normalizedRAUC', 'standardizedRAUC']
saveIndexNames = (rAUCDF.index.names).copy()
rAUCDF.reset_index(inplace=True)
for annName in derivedAnnot:
    rAUCDF.loc[:, annName] = np.nan

rAUCDF.loc[:, amplitudeFieldName] = rAUCDF[amplitudeFieldName].abs()

qLims = (5e-3, 1-5e-3)
# for name, group in rAUCDF.groupby(['feature', 'electrode']):
for name, group in rAUCDF.groupby(['feature']):
    '''rAUCDF.loc[group.index, 'standardizedRAUC'] = (
        RobustScaler(quantile_range=[i * 100 for i in qLims])
        .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))'''
    rAUCDF.loc[group.index, 'standardizedRAUC'] = (
        StandardScaler()
            .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))
    groupQuantiles = group['rauc'].quantile(qLims)
    rauc = group['rauc'].copy()
    rauc[rauc > groupQuantiles[qLims[-1]]] = groupQuantiles[qLims[-1]]
    rauc[rauc < groupQuantiles[qLims[0]]] = groupQuantiles[qLims[0]]
    # outlierMask = rAUCDF.loc[group.index, 'standardizedRAUC'].abs() > 6
    thisScaler = MinMaxScaler()
    thisScaler.fit(
        rauc.to_numpy().reshape(-1, 1))
    rAUCDF.loc[group.index, 'normalizedRAUC'] = (
        thisScaler.transform(rauc.to_numpy().reshape(-1, 1)))
    
for name, group in rAUCDF.groupby('electrode'):
    rAUCDF.loc[group.index, 'normalizedAmplitude'] = pd.cut(
        group[amplitudeFieldName], bins=10, labels=False)

'''featurePlotOptsColumns = ['featureName', 'EMGSide', 'EMGSite', 'EMG Location']
featurePlotOptsDF = rAUCDF.drop_duplicates(subset=['feature']).loc[:, featurePlotOptsColumns]
plotOptNames = ['color', 'style']
# pdb.set_trace()
for pon in plotOptNames:
    featurePlotOptsDF.loc[:, pon] = np.nan
uniqueSiteNames = sorted(featurePlotOptsDF['EMGSite'].unique())
nColors = len(uniqueSiteNames)
paletteNames = ['deep', 'pastel']
styleNames = ['-', '--']
for gIdx, (name, group) in enumerate(featurePlotOptsDF.groupby('EMGSide')):
    thisPalette = pd.Series(
        sns.color_palette(paletteNames[gIdx], nColors),
        index=uniqueSiteNames)
    featurePlotOptsDF.loc[group.index, 'color'] = group['EMGSite'].map(thisPalette)
    featurePlotOptsDF.loc[group.index, 'style'] = styleNames[gIdx]
featurePlotOptsDF.to_hdf(resultPath, 'RAUC_plotOpts')'''

rAUCDF.set_index(saveIndexNames, inplace=True)
rAUCDF.to_hdf(resultPath, 'RAUC')
asigWide.index = rAUCDF.index
asigWide.to_hdf(resultPath, 'RAUC_raw')
#
if arguments['lazy']:
    dataReader.file.close()
