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
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
consoleDebug = True
if consoleDebug:
    arguments = {
        'inputBlockSuffix': 'emg', 'blockIdx': '6', 'window': 'XXS', 'alignQuery': 'stimOn',
        'analysisName': 'loRes',
        'verbose': False, 'alignFolderName': 'stim', 'maskOutlierBlocks': False,
        'invertOutlierMask': False, 'lazy': False,
        'unitQuery': 'isiemgenv', 'exp': 'exp202012221300', 'processAll': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
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
ecapPath = os.path.join(nspFolder, 'Block0006_Config1_PNP_Info.csv')
ecapDF = pd.read_csv(ecapPath)
ecapDF.columns = [cN.strip() for cN in ecapDF.columns]
#
mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
mapDF.loc[:, 'repetition'] = mapDF['label'].apply(lambda x: x.split('_')[-1])
mapDF.loc[:, 'whichArray'] = mapDF['elecName'].apply(lambda x: x[:-1])
#
nevIDToLabelMap = mapDF.loc[:, ['nevID', 'label']].set_index('nevID')['label']
ecapDF.loc[:, 'electrode'] = ecapDF['stimElec'].map(nevIDToLabelMap).apply(lambda x: '-' + x[:-2])
recordingArray = 'rostral'
mapMask = (mapDF['whichArray'] == recordingArray) & (mapDF['repetition'] == 'a')
elecIDToLabelMap = mapDF.loc[mapMask, ['elecID', 'label']].set_index('elecID')['label']
ecapDF.loc[:, 'feature'] = ecapDF['recElec'].map(elecIDToLabelMap).apply(lambda x: x[:-2])
ecapDF.rename(columns={'Amp': 'nominalCurrent', 'Freq': 'RateInHz'}, inplace=True)
ecapDF.loc[:, 'nominalCurrent'] = ecapDF['nominalCurrent'] * (-1)
ecapMeasureNames = ['P1_y', 'N1_y', 'P2_y']
indexNames = ['electrode', 'nominalCurrent', 'RateInHz', 'feature']
ecapWideDF = ecapDF.set_index(indexNames).loc[:, ecapMeasureNames]
ecapWideDF.columns.name = 'measurement'
ecapWideDF = ecapWideDF.unstack(level='feature')
#
recCurve = pd.read_hdf(resultPath, 'meanRAUC')
recCurveWideDF = recCurve.groupby(indexNames).mean()
recCurveWideDF = recCurveWideDF.unstack(level='feature')

pdb.set_trace()
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

figureStatsFolder = os.path.join(
    alignSubFolder, 'figureStats'
    )
alignedPdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['inputBlockSuffix'],
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(figureStatsFolder, alignedPdfName + '_stats.h5')
if os.path.exists(statsTestPath):
    sigValsWide = pd.read_hdf(statsTestPath, 'sig')
    sigValsWide.columns.name = 'bin'

'''
def masterPlot(
        emgRC, targetFeature, plotElectrode,
        plotFeature, keepElectrodeList):
    meanSelectivity = pd.DataFrame(
        np.nan,
        index=emgRC['electrode'].unique(),
        columns=emgRC[targetFeature].unique())
    onTargetList = []
    offTargetList = []
    selectivityIdxMax = pd.DataFrame(
        np.nan,
        index=emgRC['electrode'].unique(),
        columns=emgRC[targetFeature].unique())
    #
    for elecName, elecGroup in emgRC.groupby(['electrode']):
        for featName, featGroup in elecGroup.groupby([targetFeature]):
            thisSelectivity = pd.DataFrame(
                np.nan, index=elecGroup[amplitudeFieldName].unique(),
                columns=['onTarget', 'offTarget'])
            for ampName, ampGroup in elecGroup.groupby(amplitudeFieldName):
                onTarget = ampGroup.loc[
                    ampGroup[targetFeature] == featName, 'normalizedRAUC'].to_frame(name='ratio')
                offTarget = ampGroup.loc[
                    ampGroup[targetFeature] != featName, 'normalizedRAUC'].to_frame(name='ratio')
                onTarget[amplitudeFieldName] = ampName
                offTarget[amplitudeFieldName] = ampName
                onTarget['electrode'] = elecName
                offTarget['electrode'] = elecName
                onTarget['targetFeature'] = featName
                offTarget['targetFeature'] = featName
                thisSelectivity.loc[ampName, "onTargetMean"] = onTarget['ratio'].mean()
                thisSelectivity.loc[ampName, "offTargetMean"] = offTarget['ratio'].mean()
                onTargetList.append(onTarget.reset_index(drop=True))
                offTargetList.append(offTarget.reset_index(drop=True))
            maxDiff = thisSelectivity['onTargetMean'] - thisSelectivity['offTargetMean']
            meanSelectivity.loc[elecName, featName] = maxDiff.max()
            selectivityIdxMax.loc[elecName, featName] = maxDiff.idxmax()
    onTargetAll = pd.concat(onTargetList, ignore_index=True)
    onTargetAll['type'] = 'on'
    offTargetAll = pd.concat(offTargetList, ignore_index=True)
    offTargetAll['type'] = 'off'
    ratiosAll = pd.concat([onTargetAll, offTargetAll], ignore_index=True)
    keepDataIdx = [cn for cn in meanSelectivity.index if cn in keepElectrodeList]
    pdfPath = os.path.join(
        figureFolder,
        prefix + '_{}_{}_{}_{}.pdf'.format(
            arguments['inputBlockSuffix'], arguments['window'],
            targetFeature,
            'selectivity_heatmap'))
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[10, 1])
    ax = fig.add_subplot(spec[0, 0])
    cb_ax = fig.add_subplot(spec[0, 1])
    sns.heatmap(data=meanSelectivity.loc[keepDataIdx, :], annot=True, fmt=".2f", ax=ax, cbar_ax=cb_ax)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=45)
    plt.savefig(pdfPath)
    plt.show()
    pdfPath = os.path.join(
        figureFolder,
        prefix + '_{}_{}_{}_selectivity_{}_{}.pdf'.format(
            arguments['inputBlockSuffix'], arguments['window'],
            targetFeature, plotElectrode, plotFeature))
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0])
    keepDataMask = (ratiosAll['electrode'] == plotElectrode) & (ratiosAll['targetFeature'] == plotFeature)
    ax = sns.boxplot(
        x=amplitudeFieldName, y='ratio', hue='type',
        data=ratiosAll.loc[keepDataMask, :])
    ax.set_ylim([
        ratiosAll.loc[keepDataMask, 'ratio'].min(),
        ratiosAll.loc[keepDataMask, 'ratio'].quantile(0.999),
        ])
    ax.set_title('{} selectivity for {}'.format(plotElectrode, plotFeature))
    plt.savefig(pdfPath)
    plt.show()

rippleMapDF = prb_meta.mapToDF(expOpts['rippleMapFile'][int(arguments['blockIdx'])])
rippleMapDF.loc[
    rippleMapDF['label'].str.contains('caudal'),
    'ycoords'] += 800
#
if 'delsysMapDict' in locals():
    delsysMapDF = pd.DataFrame(delsysMapDict)
    mapsDict = {
        'ripple': rippleMapDF,
        'delsys': delsysMapDF}
else:
    mapsDict = {
        'ripple': rippleMapDF}

lfpRC = (
    plotRC.loc[plotRC['feature'].str.contains('caudal') |
    plotRC['feature'].str.contains('rostral'), :].copy())
lfpRC['normalizedRAUC'] = np.nan
lfpRC['standardizedRAUC'] = np.nan
lfpRC['xcoords'] = np.nan
lfpRC['ycoords'] = np.nan
lfpRC['featureName'] = np.nan
nSig = {}
for name, group in lfpRC.groupby('feature'):
    print(name)
    lfpRC.loc[group.index, 'standardizedRAUC'] = (
        RobustScaler(quantile_range=(5., 95.))
        .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))
    outlierMask = lfpRC.loc[group.index, 'standardizedRAUC'].abs() > 6
    thisScaler = MinMaxScaler()
    thisScaler.fit(
        group.loc[~outlierMask, 'rauc'].to_numpy().reshape(-1, 1))
    lfpRC.loc[group.index, 'normalizedRAUC'] = (
        thisScaler.transform(group['rauc'].to_numpy().reshape(-1, 1)))
    lfpRC.loc[group.index, 'featureName'] = name[:-2]
    lfpRC.loc[group.index, 'xcoords'] = rippleMapDF.loc[rippleMapDF['label'] == name[:-2], 'xcoords'].iloc[0]
    lfpRC.loc[group.index, 'ycoords'] = rippleMapDF.loc[rippleMapDF['label'] == name[:-2], 'ycoords'].iloc[0]

uniqueX = np.unique(rippleMapDF['xcoords'].to_list() + [3400, 8500])
# xIdx = [3, 4, 6, 7]
uniqueY = np.unique(rippleMapDF['ycoords'].to_list() + [6200, 31000, 40000, 50000, 66200, 91000])
# yIdx = [0, 2, 3, 4, 6, 8, 10, 11, 12, 14, 16]
# xLookup = {k: v for k, v in zip(uniqueX, xIdx)}
# yLookup = {k: v for k, v in zip(uniqueY, yIdx)}

lfpRAUCDistr = {}
for elecName, elecGroup in lfpRC.groupby('electrode'):
    lfpRAUCDistr.update({
        elecName: pd.DataFrame(
            np.nan,
            index=uniqueY,
            columns=uniqueX)})
    for posName, posGroup in elecGroup.groupby(['xcoords', 'ycoords']):
        lfpRAUCDistr[elecName].loc[posName[1], posName[0]] = posGroup['normalizedRAUC'].mean()

for elecName in lfpRAUCDistr.keys():
    for xPos in rippleMapDF['xcoords'].unique():
        for yPos in rippleMapDF['ycoords'].unique():
            if np.isnan(lfpRAUCDistr[elecName].loc[yPos, xPos]):
                lfpRAUCDistr[elecName].loc[yPos, xPos] = 0

keepElectrodeList = ['-rostralY_e10']
targetFeature = 'featureName'
plotElectrode = '-rostralY_e10'
plotFeature = 'caudalZ_e18_a'
sns.set(rc={'axes.facecolor':'grey'})
fig = plt.figure(figsize=(5, 30), constrained_layout=True)
spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[20, 1])
ax = fig.add_subplot(spec[0, 0])
cb_ax = fig.add_subplot(spec[1, 0])
pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockSuffix'], arguments['window'],
        'meanRAUC_lfp_heatmap'))
sns.heatmap(
    lfpRAUCDistr[plotElectrode].iloc[slice(-1, None, -1), :], linewidths=.5,
    mask=lfpRAUCDistr[plotElectrode].iloc[slice(-1, None, -1), :].isna(),
    ax=ax, cbar_ax=cb_ax, cbar_kws={"orientation": "horizontal"})
plt.savefig(pdfPath)
plt.show()

masterPlot(lfpRC, targetFeature, plotElectrode, plotFeature, keepElectrodeList)
lfpPalette = sns.color_palette('Set2', lfpRC[targetFeature].unique().size)
pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockSuffix'], arguments['window'],
        'meanRAUC_lfp'))
keepDataMask = lfpRC['electrode'].isin(keepElectrodeList)
plotLfpRC = lfpRC.loc[keepDataMask, :]
g = sns.relplot(
    col='electrode', col_wrap=5, col_order=np.unique(plotLfpRC['electrode']),
    x=amplitudeFieldName, y='normalizedRAUC',
    hue=targetFeature, hue_order=np.unique(plotLfpRC[targetFeature]),
    kind='line', data=plotLfpRC,
    palette=lfpPalette,
    height=5, aspect=1.5, ci='sem', estimator='mean',
    )
# for (ro, co, hu), dataSubset in g.facet_data():
#     break
plt.savefig(pdfPath)
'''
emgRC = plotRC.loc[plotRC['feature'].str.contains('EmgEnv'), :].copy()
emgRC['normalizedRAUC'] = np.nan
emgRC['standardizedRAUC'] = np.nan
emgRC['featureName'] = np.nan
emgRC['EMGSide'] = np.nan
emgRC['EMGSite'] = np.nan
# emgRC[amplitudeFieldName] *= (-1)
emgRC[amplitudeFieldName] = emgRC[amplitudeFieldName].abs()
sideLookup = {'R': 'Right', 'L': 'Left'}
nSig = {}

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
    if os.path.exists(statsTestPath):
        theseSig = sigValsWide.xs(featName, level='unit')
        nSig.update({featName[:-8]: theseSig.sum().sum()})

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
plt.close()
