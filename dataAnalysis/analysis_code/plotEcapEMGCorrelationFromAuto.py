"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisNameLFP=analysisNameLFP      append a name to the resulting blocks? [default: default]
    --analysisNameEMG=analysisNameEMG      append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --lfpBlockSuffix=lfpBlockSuffix        which trig_ block to pull [default: pca]
    --emgBlockSuffix=emgBlockSuffix        which trig_ block to pull [default: pca]
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
import dataAnalysis.plotting.spike_sorting_plots as ssplt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
consoleDebug = True
if consoleDebug:
    arguments = {
        'lfpBlockSuffix': 'lfp_raw', 'emgBlockSuffix': 'emg',
        'blockIdx': '6', 'window': 'XXS', 'alignQuery': 'stimOn',
        'analysisNameLFP': 'fullRes', 'analysisNameEMG': 'loRes',
        'verbose': False, 'alignFolderName': 'stim', 'maskOutlierBlocks': False,
        'invertOutlierMask': False, 'lazy': False,
        'unitQuery': 'isiemgenv', 'exp': 'exp202012221300', 'processAll': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    scratchFolder = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202012221300-Goat'
    figureFolder = '/gpfs/data/dborton/rdarie/Neural Recordings/processed/202012221300-Goat/figures'
    ns5FileName = 'Block006'
    scratchPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings'
#
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
sns.set(
    context='notebook', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
#
analysisSubFolderLFP = os.path.join(
    scratchFolder, arguments['analysisNameLFP'])
analysisSubFolderEMG = os.path.join(
    scratchFolder, arguments['analysisNameEMG'])
alignSubFolderLFP = os.path.join(
    analysisSubFolderLFP, arguments['alignFolderName'])
alignSubFolderEMG = os.path.join(
    analysisSubFolderEMG, arguments['alignFolderName'])
calcSubFolderLFP = os.path.join(alignSubFolderLFP, 'dataframes')
calcSubFolderEMG = os.path.join(alignSubFolderEMG, 'dataframes')
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisNameLFP'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
resultPathEMG = os.path.join(
    calcSubFolderEMG,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['emgBlockSuffix'], arguments['window']))
print('loading {}'.format(resultPathEMG))
outlierTrials = ash.processOutlierTrials(
    scratchPath, prefix, **arguments)
#  Overrides
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
ecapPath = os.path.join(
    calcSubFolderLFP, 'lmfit',
    prefix + '_{}_{}_lmfit_signals.parquet'.format(
        arguments['lfpBlockSuffix'], arguments['window']))
rawEcapDF = pd.read_parquet(ecapPath, engine='fastparquet')
rawEcapDF.loc[:, 'nominalCurrent'] = rawEcapDF['nominalCurrent'] * (-1)
dropIndices = rawEcapDF.index[~rawEcapDF['regrID'].isin(['target', 'exp_resid'])]
rawEcapDF.drop(index=dropIndices, inplace=True)
# rawEcapDF.shape
ecapDF = rawEcapDF.loc[rawEcapDF['regrID'] == 'exp_resid', :].copy()
ecapDF.drop(columns=['regrID'], inplace=True)
#
ecapDF.set_index([
        'segment', 'originalIndex', 't',
        'electrode', 'nominalCurrent', 'feature'],
    inplace=True)
ecapDF.columns = ecapDF.columns.astype(float)
#
rawEcapDF.set_index([
        'segment', 'originalIndex', 't',
        'electrode', 'nominalCurrent', 'feature', 'regrID'],
    inplace=True)
rawEcapDF.columns = rawEcapDF.columns.astype(float)
#
plotEcap = rawEcapDF.stack().reset_index()
plotEcap.rename(columns={0: 'signal'}, inplace=True)
plotEcap.loc[:, 'feature'] = plotEcap['feature'].apply(lambda x: x.replace('#0', ''))
#

def getRAUC(x, timeWindow):
    tWinStart, tWinStop = timeWindow
    tMask = (x.index >= tWinStart) & (x.index < tWinStop)
    return x.loc[tMask].abs().mean()

ecapTWinStart, ecapTWinStop = 1e-3, 4e-3
ecapRauc = ecapDF.apply(getRAUC, axis='columns', args=[(ecapTWinStart, ecapTWinStop)])
ecapRaucWideDF = ecapRauc.unstack(level='feature')
recCurve = pd.read_hdf(resultPathEMG, 'meanRAUC')
plotOpts = pd.read_hdf(resultPathEMG, 'meanRAUC_plotOpts')
emgPalette = plotOpts.loc[:, ['featureName', 'color']].set_index('featureName')['color']
rates = recCurve.index.get_level_values('RateInHz')
dbIndexMask = (rates < 30)
recCurveWideDF = recCurve.loc[dbIndexMask, :].groupby(ecapDF.index.names).mean()['rauc']
recCurveWideDF = recCurveWideDF.unstack(level='feature')
pdb.set_trace()
assert np.all(recCurveWideDF.index == ecapRaucWideDF.index)
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
        recCurveWideDF.index.map(rejectionLookup),
        dtype=np.bool)
    if arguments['invertOutlierMask']:
        outlierMask = ~outlierMask
    recCurveWideDF = recCurveWideDF.loc[~outlierMask, :]
    ecapRaucWideDF = ecapRaucWideDF.loc[~outlierMask, :]
#
fixedElectrodeNames = pd.Series(ecapDF.index.get_level_values('electrode')).apply(lambda x: x[1:])
fixedFeatureNames = pd.Series(ecapDF.index.get_level_values('feature')).apply(lambda x: x[:-4])
ecapMaskWide = (
    pd.Series(fixedFeatureNames.to_numpy() == fixedElectrodeNames.to_numpy(), index=ecapDF.index)
    .unstack(level='feature')
    .fillna(False))
#
recCurveWideDF.columns = [cN.replace('EmgEnv#0', '') for cN in recCurveWideDF.columns]
ecapRaucWideDF.columns = [cN.replace('#0', '') for cN in ecapRaucWideDF.columns]
ecapMaskWide.columns = [cN.replace('#0', '') for cN in ecapMaskWide.columns]
#
corrDFIndex = pd.MultiIndex.from_product(
    [recCurveWideDF.columns, ecapRaucWideDF.columns],
    names=['emg', 'lfp'])
annotNames = ['xcoords', 'ycoords', 'whichArray']
corrDF = pd.DataFrame(np.nan, index=corrDFIndex, columns=['R'] + annotNames)
#
mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
mapDF.loc[:, 'whichArray'] = mapDF['elecName'].apply(lambda x: x[:-1])
mapDF.loc[mapDF['whichArray'] == 'rostral', 'xcoords'] += mapDF['xcoords'].max() * 2
#
for emgName, lfpName in corrDF.index:
    rowIdx = (emgName, lfpName)
    thisEcapMask = ~ecapMaskWide[lfpName].to_numpy()
    corrDF.loc[rowIdx,'R'] = np.corrcoef(
        recCurveWideDF.loc[thisEcapMask, emgName],
        ecapRaucWideDF.loc[thisEcapMask, lfpName])[0, 1]

for annotName in annotNames:
    lookupSource = mapDF.loc[:, [annotName, 'label']].set_index('label')[annotName]
    corrDF.loc[:, annotName] = corrDF.index.get_level_values('lfp').map(lookupSource)

corrDF.loc[:, 'xIdx'], corrDF.loc[:, 'yIdx'] = ssplt.coordsToIndices(
    corrDF['xcoords'], corrDF['ycoords'],
    swapXY=True)
corrDF.loc[:, 'R2'] = corrDF['R'] ** 2
corrDF.loc[:, 'xDummy'] = 1

plotDF = corrDF.reset_index()
plotDF.sort_values(by='emg', inplace=True, kind='mergesort')
plotDF = plotDF.query("whichArray == 'rostral'")

if RCPlotOpts['keepFeatures'] is not None:
    keepDataMask = plotDF['emg'].isin(RCPlotOpts['keepFeatures'])
    plotDF = plotDF.loc[keepDataMask, :]

fig, ax = plt.subplots(1, 2, sharey=False)
exEMGName, exLFPName = plotDF.loc[plotDF['R2'].idxmin(), ['emg', 'lfp']]
thisEcapMask = ~ecapMaskWide[exLFPName].to_numpy()
ax[0] = sns.regplot(
    x=ecapRaucWideDF.loc[thisEcapMask, exLFPName],
    y=recCurveWideDF.loc[thisEcapMask, exEMGName], ax=ax[0],
    color=emgPalette[exEMGName])
ax[0].set_xlabel('ECAP RAUC ({})'.format(exLFPName))
ax[0].set_ylabel('EMG RAUC ({})'.format(exEMGName))
exEMGName, exLFPName = plotDF.loc[plotDF['R2'].idxmax(), ['emg', 'lfp']]
thisEcapMask = ~ecapMaskWide[exLFPName].to_numpy()
ax[1] = sns.regplot(
    x=ecapRaucWideDF.loc[thisEcapMask, exLFPName],
    y=recCurveWideDF.loc[thisEcapMask, exEMGName], ax=ax[1],
    color=emgPalette[exEMGName])
ax[1].set_xlabel('ECAP RAUC ({})'.format(exLFPName))
plt.show()

exEMGName, exLFPName = plotDF.loc[plotDF['R2'].idxmin(), ['emg', 'lfp']]
binMask = (plotEcap['bin'] > ecapTWinStart) & (plotEcap['bin'] < ecapTWinStop)
featureMask = plotEcap['feature'] == exLFPName
g = sns.relplot(
    x='bin', y='signal', hue='nominalCurrent', row='electrode', style='regrID',
    data=plotEcap.loc[binMask & featureMask, :], kind='line', ci='sem')
for axIdx, ax in enumerate(g.axes.flat):
    ax.set_ylabel(exLFPName)
g.axes.flat[0].set_xlabel('Time (sec)')
plt.show()

sns.set(
    context='notebook', style='dark',
    palette='dark', font='sans-serif',
    font_scale=0.7, color_codes=True)
'''
    asp.genTitleAnnotator(
        template='{}', colNames=['lfp'],
        dropNaNCol='R2', shared=False),'''
plotProcFuns = [
    asp.genGridAnnotator(
        xpos=1, ypos=1, template='{}', colNames=['lfp'],
        dropNaNCol='R2',
        textOpts={
            'verticalalignment': 'top',
            'horizontalalignment': 'right'
        }, shared=False),
        ]
g = sns.catplot(
    row='xIdx', col='yIdx', y='R2',
    x='xDummy', data=plotDF, kind='bar',
    hue='emg', palette=emgPalette.to_dict(), hue_order=sorted(plotDF['emg'].unique()))
g.set_titles('')
g.set_xlabels('')
g.set_xticklabels([''])
for (ro, co, hu), dataSubset in g.facet_data():
    emptySubset = (
            (dataSubset.empty) or
            (dataSubset['R2'].isna().all()))
    if len(plotProcFuns):
        for procFun in plotProcFuns:
            procFun(g, ro, co, hu, dataSubset)
plt.show()