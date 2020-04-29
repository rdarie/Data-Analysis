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
from sklearn.preprocessing import RobustScaler, MinMaxScaler
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
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(resultPath))
#  Overrides
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
recCurve = pd.read_hdf(resultPath, 'meanRAUC')
plotRC = recCurve.reset_index()

figureStatsFolder = os.path.join(
    alignSubFolder, 'figureStats'
    )
alignedPdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['inputBlockName'],
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(figureStatsFolder, 'backup', alignedPdfName + '_stats.h5')
if os.path.exists(statsTestPath):
    sigValsWide = pd.read_hdf(statsTestPath, 'sig')
    sigValsWide.columns.name = 'bin'

pdb.set_trace()
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
            arguments['inputBlockName'], arguments['window'],
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
            arguments['inputBlockName'], arguments['window'],
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
        arguments['inputBlockName'], arguments['window'],
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
        arguments['inputBlockName'], arguments['window'],
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
emgRC = plotRC.loc[plotRC['feature'].str.contains('EmgEnv'), :].copy()
emgRC['normalizedRAUC'] = np.nan
emgRC['standardizedRAUC'] = np.nan
emgRC['featureName'] = np.nan
emgRC['EMGSide'] = np.nan
emgRC['EMGSite'] = np.nan
emgRC[amplitudeFieldName] *= -1
sideLookup = {'R': 'Right', 'L': 'Left'}
nSig = {}
for name, group in emgRC.groupby('feature'):
    emgRC.loc[group.index, 'standardizedRAUC'] = (
        RobustScaler(quantile_range=(1, 99.))
        .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))
    outlierMask = emgRC.loc[group.index, 'standardizedRAUC'].abs() > 6
    thisScaler = MinMaxScaler()
    thisScaler.fit(
        group.loc[~outlierMask, 'rauc'].to_numpy().reshape(-1, 1))
    emgRC.loc[group.index, 'normalizedRAUC'] = (
        thisScaler.transform(group['rauc'].to_numpy().reshape(-1, 1)))
    featName = name[1:-8]
    emgRC.loc[group.index, 'featureName'] = name[:-8]
    emgRC.loc[group.index, 'EMGSite'] = name[1:-8]
    emgRC.loc[group.index, 'EMGSide'] = sideLookup[name[0]]
    if os.path.exists(statsTestPath):
        theseSig = sigValsWide.xs(name, level='unit')
        nSig.update({name[:-8]: theseSig.sum().sum()})
#
emgRC.loc[:, 'EMG Location'] = (
    emgRC['EMGSide'] + ' ' + emgRC['EMGSite'])

# keepElectrodeList = meanSelectivity.max(axis=1).index[meanSelectivity.max(axis=1) > 0.3]
# keepElectrodeList = meanSelectivity.index.to_list()
# targetFeature = 'featureName'
#targetFeature = 'EMGSite'
# plotElectrode = '-caudalY_e12+caudalX_e05'
# plotFeature = 'RVastusLateralis'
# plotFeature = 'RPeroneusLongus'
# plotFeature = 'PeroneusLongus'
# plotFeature = 'Right'
masterPlot(
    emgRC, targetFeature, plotElectrode,
    plotFeature, keepElectrodeList)
significantOnly = True
if significantOnly:
    emgRC = emgRC.query("(kruskalP < 1e-3)")
emgPalette = sns.color_palette('Set2', emgRC['EMGSite'].unique().size)

pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockName'], arguments['window'],
        'meanRAUC'))
keepDataMask = emgRC['electrode'].isin(keepElectrodeList)
plotEmgRC = emgRC.loc[keepDataMask, :]
g = sns.relplot(
    col='electrode', col_wrap=5, col_order=np.unique(plotEmgRC['electrode']),
    x=amplitudeFieldName, y='normalizedRAUC',
    style='EMGSide', style_order=['Right', 'Left'],
    hue='EMGSite', hue_order=np.unique(plotEmgRC['EMGSite']),
    kind='line', data=plotEmgRC,
    palette=emgPalette,
    height=5, aspect=1.5, ci='sem', estimator='mean',
    )
# for (ro, co, hu), dataSubset in g.facet_data():
#     break
plt.savefig(pdfPath)
plt.show()
