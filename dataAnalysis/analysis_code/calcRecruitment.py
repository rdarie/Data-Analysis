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
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --maskOutlierBlocks                    delete outlier trials? [default: False]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate interactive output
# matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
import seaborn as sns
#
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
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
sns.set(
    context='notebook', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
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
    decimate=1, windowSize=(-100e-3, 100e-3),
    transposeToColumns='bin', concatOn='index'))
alignedAsigsKWargs['procFun'] = ash.genDetrender(
    timeWindow=(-90e-3, -50e-3))
if arguments['processAll']:
    prefix = 'Block'
else:
    prefix = ns5FileName
#
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockSuffix'], arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['inputBlockSuffix'], arguments['window']))
outlierTrials = ash.processOutlierTrials(
    scratchFolder, prefix, **arguments)
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
#  Overrides
limitPages = None
funKWargs = dict(
    # baseline='mean',
    tStart=-40e-3, tStop=40e-3)
#  End Overrides
asigWide = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
if outlierTrials is not None:
    trialInfo = asigWide.index.copy()
    for idxName in trialInfo.names:
        if idxName not in outlierTrials.index.names:
            trialInfo = trialInfo.droplevel(idxName)
    outlierMask = pd.Series(trialInfo.to_numpy()).map(outlierTrials)
    asigWide = asigWide.loc[~outlierMask.to_numpy(), :]

# pre-standardize
preStandardizeBounds = [-90e-3, -50e-3]
# pdb.set_trace()
preStMask = (asigWide.columns >= preStandardizeBounds[0]) & (asigWide.columns < preStandardizeBounds[1])

for name, group in asigWide.groupby(['feature']):
    scaler = StandardScaler()
    scaler.fit(group.loc[:, preStMask].to_numpy().reshape(-1, 1))
    origShape = asigWide.loc[group.index, :].shape
    asigWide.loc[group.index, :] = scaler.transform(asigWide.loc[group.index, :].to_numpy().reshape(-1, 1)).reshape(origShape)

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

derivedAnnot = [
    'normalizedRAUC', 'standardizedRAUC',
    'featureName', 'EMGSide', 'EMGSite']
saveIndexNames = (rAUCDF.index.names).copy()
rAUCDF.reset_index(inplace=True)
rAUCDF.loc[:, amplitudeFieldName] = rAUCDF[amplitudeFieldName].abs()

for annName in derivedAnnot:
    rAUCDF.loc[:, annName] = np.nan

sideLookup = {'R': 'Right', 'L': 'Left'}
'''nSig = {}'''

qLims = (5e-3, 1-5e-3)
# for name, group in rAUCDF.groupby(['feature', 'electrode']):
for name, group in rAUCDF.groupby(['feature']):
    '''rAUCDF.loc[group.index, 'standardizedRAUC'] = (
        RobustScaler(quantile_range=[i * 100 for i in qLims])
        .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))'''
    scaler = StandardScaler()
    minParams = group.groupby([amplitudeFieldName, 'electrode']).mean()['rauc'].idxmin()
    minMask = (group[amplitudeFieldName] == minParams[0]) & (group['electrode'] == minParams[1])
    scaler.fit(group.loc[minMask, 'rauc'].to_numpy().reshape(-1, 1))
    rAUCDF.loc[group.index, 'standardizedRAUC'] = scaler.transform(group['rauc'].to_numpy().reshape(-1, 1))
    '''rAUCDF.loc[group.index, 'standardizedRAUC'] = (
        StandardScaler()
            .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))'''
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
    featName = name
    featNameShort = featName.replace('EmgEnv#0', '').replace('Emg#0', '')
    rAUCDF.loc[group.index, 'featureName'] = featNameShort
    rAUCDF.loc[group.index, 'EMGSite'] = featNameShort[1:]
    rAUCDF.loc[group.index, 'EMGSide'] = sideLookup[featName[0]]
    '''if os.path.exists(statsTestPath):
        theseSig = sigValsWide.xs(featName, level='unit')
        nSig.update({featName[:-8]: theseSig.sum().sum()})'''

rAUCDF.loc[:, 'EMG Location'] = (
    rAUCDF['EMGSide'] + ' ' + rAUCDF['EMGSite'])
for name, group in rAUCDF.groupby('electrode'):
    rAUCDF.loc[group.index, 'normalizedAmplitude'] = pd.cut(
        group[amplitudeFieldName], bins=10, labels=False)
featurePlotOptsColumns = ['featureName', 'EMGSide', 'EMGSite', 'EMG Location']
featurePlotOptsDF = rAUCDF.drop_duplicates(subset=['feature']).loc[:, featurePlotOptsColumns]
plotOptNames = ['color', 'style']

for pon in plotOptNames:
    featurePlotOptsDF.loc[:, pon] = np.nan
uniqueSiteNames = sorted(featurePlotOptsDF['EMGSite'].unique())
nColors = len(uniqueSiteNames)
# paletteNames = ['deep', 'pastel']
hues = [0., 0.]
lightnesses = [.6, .75]
saturations = [.65, .75]
styleNames = ['--', '-']
for gIdx, (name, group) in enumerate(featurePlotOptsDF.groupby('EMGSide')):
    thisPalette = sns.hls_palette(
        n_colors=int(nColors*2.5),
        h=hues[gIdx], l=lightnesses[gIdx], s=saturations[gIdx])
    thisPaletteSrs = pd.Series(
        # sns.color_palette(paletteNames[gIdx], nColors),
        thisPalette[:nColors],
        index=uniqueSiteNames)
    featurePlotOptsDF.loc[group.index, 'color'] = group['EMGSite'].map(thisPaletteSrs)
    featurePlotOptsDF.loc[group.index, 'style'] = styleNames[gIdx]
# sns.palplot(featurePlotOptsDF['color']); plt.show()
if False:
    fig, ax = plt.subplots()
    for name, group in rAUCDF.groupby('featureName'):
        useColor = featurePlotOptsDF.loc[featurePlotOptsDF['featureName'] == name, 'color'].iloc[0]
        sns.distplot(group['rauc'], kde=False, hist_kws={"histtype": "step", "linewidth": 3, 'color': useColor}, ax=ax, label=name)
    ax.legend()
    plt.show()
#
if os.path.exists(resultPath):
    os.remove(resultPath)
rAUCDF.set_index(saveIndexNames, inplace=True)
rAUCDF.to_hdf(resultPath, 'emgRAUC')
featurePlotOptsDF.to_hdf(resultPath, 'emgRAUC_plotOpts')
asigWide.index = rAUCDF.index
asigWide.to_hdf(resultPath, 'emgRAUC_raw')
#
if arguments['lazy']:
    dataReader.file.close()
