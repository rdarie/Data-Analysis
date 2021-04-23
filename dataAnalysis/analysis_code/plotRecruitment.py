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
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierMask                    delete outlier trials? [default: False]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
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
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
sns.set(
    context='notebook', style='whitegrid',
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
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['inputBlockSuffix'], arguments['window']))
print('loading {}'.format(resultPath))
outlierTrials = ash.processOutlierTrials(
    scratchPath, prefix, **arguments)
#  Overrides
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
recCurve = pd.read_hdf(resultPath, 'meanRAUC')
if outlierTrials is not None:
    def rejectionLookup(entry):
        key = []
        for subKey in outlierTrials.index.names:
            keyIdx = recCurve.index.names.index(subKey)
            key.append(entry[keyIdx])
        # print(key)
        return outlierTrials[tuple(key)]
    #
    outlierMask = np.asarray(
        recCurve.index.map(rejectionLookup),
        dtype=np.bool)
    if arguments['invertOutlierMask']:
        outlierMask = ~outlierMask
    recCurve = recCurve.loc[~outlierMask, :]
minNObservations = 5
trialInfo = recCurve.index.to_frame().reset_index(drop=True)
if minNObservations is not None:
    nObsCountedFeatures = ['feature']
    for extraName in ['electrode', amplitudeFieldName, 'RateInHz']:
        if extraName is not None:
            if extraName not in nObsCountedFeatures:
                nObsCountedFeatures.append(extraName)
    nObs = trialInfo.groupby(nObsCountedFeatures).count().iloc[:, 0].to_frame(name='obsCount')
    nObs['keepMask'] = nObs['obsCount'] > minNObservations
    #
    def lookupKeep(x):
        keepVal = nObs.loc[tuple(x.loc[nObsCountedFeatures]), 'keepMask']
        # print(keepVal)
        return(keepVal)
    #
    keepMask = trialInfo.apply(lookupKeep, axis=1).to_numpy()
    recCurve = recCurve.loc[keepMask, :]
    trialInfo = recCurve.index.to_frame().reset_index(drop=True)

plotRC = recCurve.reset_index()

'''figureStatsFolder = os.path.join(
    alignSubFolder, 'figureStats'
    )
alignedPdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['inputBlockSuffix'],
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(figureStatsFolder, alignedPdfName + '_stats.h5')
if os.path.exists(statsTestPath):
    sigValsWide = pd.read_hdf(statsTestPath, 'sig')
    sigValsWide.columns.name = 'bin''''

emgRC = plotRC.loc[plotRC['feature'].str.contains('EmgEnv'), :].copy()
emgRC['normalizedRAUC'] = np.nan
emgRC['standardizedRAUC'] = np.nan
emgRC['featureName'] = np.nan
emgRC['EMGSide'] = np.nan
emgRC['EMGSite'] = np.nan
# emgRC[amplitudeFieldName] *= (-1)
emgRC[amplitudeFieldName] = emgRC[amplitudeFieldName].abs()
sideLookup = {'R': 'Right', 'L': 'Left'}
'''nSig = {}'''

qLims = (0.05, 0.95)
# for name, group in emgRC.groupby(['feature', 'electrode']):
for name, group in emgRC.groupby(['feature']):
    emgRC.loc[group.index, 'standardizedRAUC'] = (
        RobustScaler(quantile_range=[i * 100 for i in qLims])
        .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))
    groupQuantiles = group['rauc'].quantile(qLims)
    rauc = group['rauc'].copy()
    rauc[rauc > groupQuantiles[qLims[-1]]] = groupQuantiles[qLims[-1]]
    rauc[rauc < groupQuantiles[qLims[0]]] = groupQuantiles[qLims[0]]
    # outlierMask = emgRC.loc[group.index, 'standardizedRAUC'].abs() > 6
    thisScaler = MinMaxScaler()
    thisScaler.fit(
        rauc.to_numpy().reshape(-1, 1))
    emgRC.loc[group.index, 'normalizedRAUC'] = (
        thisScaler.transform(rauc.to_numpy().reshape(-1, 1)))
    featName = name
    emgRC.loc[group.index, 'featureName'] = featName[:-8]
    emgRC.loc[group.index, 'EMGSite'] = featName[1:-8]
    emgRC.loc[group.index, 'EMGSide'] = sideLookup[featName[0]]
    '''if os.path.exists(statsTestPath):
        theseSig = sigValsWide.xs(featName, level='unit')
        nSig.update({featName[:-8]: theseSig.sum().sum()})'''

emgRC.loc[:, 'EMG Location'] = (
    emgRC['EMGSide'] + ' ' + emgRC['EMGSite'])
for name, group in emgRC.groupby('electrode'):
    emgRC.loc[group.index, 'normalizedAmplitude'] = pd.cut(
        group[amplitudeFieldName], bins=10, labels=False)

pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockSuffix'], arguments['window'],
        'meanRAUC'))

plotEmgRC = emgRC
if RCPlotOpts['significantOnly']:
    plotEmgRC = plotEmgRC.query("(kruskalP < 1e-3)")

if RCPlotOpts['keepElectrodes'] is not None:
    keepDataMask = emgRC['electrode'].isin(RCPlotOpts['keepElectrodes'])
    plotEmgRC = plotEmgRC.loc[keepDataMask, :]

if RCPlotOpts['keepFeatures'] is not None:
    keepDataMask = plotEmgRC['featureName'].isin(RCPlotOpts['keepFeatures'])
    plotEmgRC = plotEmgRC.loc[keepDataMask, :]
#
emgPalette = sns.color_palette('Set2', plotEmgRC['EMGSite'].unique().size)
g = sns.relplot(
    col='electrode',
    col_order=np.unique(plotEmgRC['electrode']),
    col_wrap=5,
    # row='RateInHz',
    # x='normalizedAmplitude',
    x=amplitudeFieldName,
    y='normalizedRAUC',
    style='EMGSide', style_order=['Right', 'Left'],
    hue='EMGSite', hue_order=np.unique(plotEmgRC['EMGSite']),
    kind='line', data=plotEmgRC.query('RateInHz > 50'),
    palette=emgPalette,
    height=5, aspect=1.5, ci='sem', estimator='mean',
    facet_kws=dict(sharey=True, sharex=False), lw=2,
    )
#
plt.savefig(pdfPath)
plt.show()
