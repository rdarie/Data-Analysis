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
    --showFigures                          load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
"""

enableDiagnosticPlots = False
if enableDiagnosticPlots:import matplotlib, os
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if 'CCV_HEADLESS' in os.environ:
        matplotlib.use('PS')   # generate postscript output
    else:
        matplotlib.use('QT5Agg')   # generate interactive output
    #
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
from tqdm import tqdm
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os, re
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
idsl = pd.IndexSlice
import numpy as np
import dataAnalysis.plotting.spike_sorting_plots as ssplt
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'processAll': True, 'exp': 'exp202012171300', 'alignQuery': 'stimOn',
        'analysisNameLFP': 'fullRes', 'analysisNameEMG': 'loRes', 'alignFolderName': 'stim',
        'lfpBlockSuffix': 'lfp_raw', 'emgBlockSuffix': 'emg', 'lazy': False, 'blockIdx': '1',
        'verbose': False, 'unitQuery': 'pca', 'window': 'XXS', 'showFigures': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
#
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
sns.set(
    context='notebook', style='darkgrid',
    palette='pastel', font='sans-serif',
    font_scale=1.5, color_codes=True)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName'])
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'], arguments['alignFolderName'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
if arguments['processAll']:
    prefix = 'Block'
else:
    prefix = ns5FileName
#
#  Overrides
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
ecapPath = os.path.join(
    calcSubFolder, 'lmfit',
    prefix + '_{}_{}_lmfit_signals.parquet'.format(
        arguments['inputBlockSuffix'], arguments['window']))
print('applyCARToLmFit loading {}...'.format(ecapPath))
rawEcapDF = pd.read_parquet(ecapPath, engine='fastparquet')
rawEcapDF.loc[:, 'feature'] = rawEcapDF['feature'].apply(lambda x: x[:-4])

annotNames = ['xcoords', 'ycoords', 'whichArray']
trialMetaNames = [
    'segment', 'originalIndex', 't',
    'trialRateInHz',
    'electrode', amplitudeFieldName]
featureMetaNames = annotNames + ['regrID', 'feature']
################################################################################################
# fix map file
################################################################################################
#
mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
mapDF.loc[:, 'whichArray'] = mapDF['elecName'].apply(lambda x: x[:-1])
mapDF.loc[:, 'shifted_xcoords'] = mapDF['xcoords']
mapDF.loc[mapDF['whichArray'] == 'rostral', 'shifted_xcoords'] -= mapDF['xcoords'].max() * 2
# mapDF.loc[:, 'shifted_xcoords'] = mapDF['shifted_xcoords'] - mapDF['xcoords'].max() / 2
mapDF.loc[:, 'channelRepetition'] = mapDF['label'].apply(lambda x: x.split('_')[-1])
mapDF.loc[:, 'topoName'] = mapDF['label'].apply(lambda x: x[:-2])
elecSideLookup = {
    'caudalX': 'Left',
    'caudalY': 'Midline',
    'caudalZ': 'Right',
    'rostralX': 'Left',
    'rostralY': 'Midline',
    'rostralZ': 'Right',
}
mapDF.loc[:, 'electrodeSide'] = mapDF['elecName'].map(elecSideLookup)
mapAMask = (mapDF['channelRepetition'] == 'a').to_numpy()
lfpNL = mapDF.loc[mapAMask, :].set_index('topoName')
#
uniqueX = np.unique(mapDF['xcoords'])
xUnshiftedPalette = pd.Series(
    sns.color_palette('rocket', n_colors=uniqueX.size),
    index=uniqueX
    )
uniqueY = np.unique(mapDF['ycoords'])
yUnshiftedPalette = pd.Series(
    sns.color_palette('mako', n_colors=uniqueY.size),
    index=uniqueY
    )
xPalette = (
    mapDF
        .loc[:, ['xcoords', 'shifted_xcoords']]
        .set_index('shifted_xcoords')['xcoords']
        .map(xUnshiftedPalette)
        .drop_duplicates())
mapDF.loc[:, 'xcoords'] = mapDF['shifted_xcoords'].copy()
################################################################################################

for annotName in annotNames:
    lookupSource = mapDF.loc[mapAMask, [annotName, 'topoName']].set_index('topoName')[annotName]
    rawEcapDF.loc[:, annotName] = rawEcapDF['feature'].map(lookupSource)
timeBinMask = ~rawEcapDF.columns.isin(trialMetaNames + featureMetaNames)
if RCCalcOpts['rejectFeatures'] is not None:
    rejectDataMask = rawEcapDF['feature'].isin(RCCalcOpts['rejectFeatures'])
    rawEcapDF = rawEcapDF.loc[~rejectDataMask, :]

smallDataset = True
if smallDataset:
    rawEcapDF = rawEcapDF.loc[rawEcapDF['electrode'].str.contains('caudal'), :]
    # rawEcapDF = rawEcapDF.loc[rawEcapDF['electrode'].str.contains('caudalY_e10'), :]
#
rawVals = rawEcapDF.loc[rawEcapDF['regrID'] == 'target', timeBinMask].to_numpy().flatten()
rawMean = np.mean(rawVals)
rawStd = np.std(rawVals)
upperBound = (rawMean + 15 * rawStd)
lowerBound = (rawMean - 15 * rawStd)
## fix possible bad exp_ components
fixedExp = rawEcapDF.loc[rawEcapDF['regrID'] == 'exp_', timeBinMask] * 0
# pdb.set_trace()
for cN in ['exp1_', 'exp2_']:
    dataGroup = rawEcapDF.loc[rawEcapDF['regrID'] == cN, timeBinMask].copy()
    outlierMask = ((dataGroup > (rawMean + 9 * rawStd)) | (dataGroup < (rawMean - 9 * rawStd))).any(axis='columns')
    dataGroup.loc[outlierMask, :] = 0
    fixedExp += dataGroup.to_numpy()
rawEcapDF.loc[rawEcapDF['regrID'] == 'exp_', timeBinMask] = fixedExp
originalTarget = rawEcapDF.loc[rawEcapDF['regrID'] == 'target', timeBinMask].to_numpy()
rawEcapDF.loc[rawEcapDF['regrID'] == 'exp_resid', timeBinMask] = originalTarget - fixedExp.to_numpy()
#
newComponents = []
print('Getting CAR')
for trialName, trialGroup in tqdm(rawEcapDF.groupby(['segment', 'originalIndex', 't'])):
    for arrayName, group in trialGroup.groupby('whichArray'):
        groupData = group.loc[group['regrID'] == 'exp_resid', timeBinMask].copy()
        outlierMask = ((groupData > upperBound) | (groupData < lowerBound)).any(axis='columns')
        meanExpResid = groupData.loc[~outlierMask].mean()
        expResidCAR = group.loc[group['regrID'] == 'exp_resid', :].copy()
        expResidCAR.loc[:, 'regrID'] = 'exp_resid_CAR'
        expResidMean = group.loc[group['regrID'] == 'exp_resid', :].copy()
        expResidMean.loc[:, 'regrID'] = 'exp_resid_mean'
        #
        targetData = group.loc[group['regrID'] == 'target', timeBinMask].copy()
        outlierMask = ((targetData > upperBound) | (targetData < lowerBound)).any(axis='columns')
        meanTarget = targetData.loc[~outlierMask].mean()
        targetCAR = group.loc[group['regrID'] == 'target', :].copy()
        targetCAR.loc[:, 'regrID'] = 'target_CAR'
        targetMean = group.loc[group['regrID'] == 'target', :].copy()
        targetMean.loc[:, 'regrID'] = 'target_mean'
        #
        linearDetrend = False
        if linearDetrend:
            for rowIdx, row in groupData.iterrows():
                noiseModel = np.polyfit(
                    meanExpResid, row, 1, full=True)
                noiseTerm = np.polyval(
                    noiseModel[0], meanExpResid)
                expResidCAR.loc[rowIdx, timeBinMask] = expResidCAR.loc[rowIdx, timeBinMask] - noiseTerm
                expResidMean.loc[rowIdx, timeBinMask] = noiseTerm
                #
                noiseModel = np.polyfit(
                    meanTarget, row, 1, full=True)
                noiseTerm = np.polyval(
                    noiseModel[0], meanTarget)
                targetCAR.loc[rowIdx, timeBinMask] = targetCAR.loc[rowIdx, timeBinMask] - noiseTerm
                targetMean.loc[rowIdx, timeBinMask] = noiseTerm
        else:
            expResidMean.loc[:, timeBinMask] = np.tile(meanExpResid, (expResidMean.shape[0], 1))
            expResidCAR.loc[:, timeBinMask] = expResidCAR.loc[:, timeBinMask] - expResidMean.loc[:, timeBinMask]
            #
            targetMean.loc[:, timeBinMask] = np.tile(meanTarget, (targetMean.shape[0], 1))
            targetCAR.loc[:, timeBinMask] = targetCAR.loc[:, timeBinMask] - targetMean.loc[:, timeBinMask]
        #
        expResidCAR.loc[:, timeBinMask] = expResidCAR.loc[:, timeBinMask].clip(lower=(rawMean - 9 * rawStd), upper=(rawMean + 9 * rawStd))
        targetCAR.loc[:, timeBinMask] = targetCAR.loc[:, timeBinMask].clip(lower=(rawMean - 9 * rawStd), upper=(rawMean + 9 * rawStd))
        newComponents.append(expResidCAR)
        newComponents.append(expResidMean)
        newComponents.append(targetCAR)
        newComponents.append(targetMean)
        ###
        if enableDiagnosticPlots:
            plotGroupData = groupData.copy()
            plotGroupData.columns = plotGroupData.columns.astype(float)
            axLims = 1.5e-3, 10e-3
            axMask = (plotGroupData.columns >= axLims[0]) & (plotGroupData.columns < axLims[1])
            plotGroupData = plotGroupData.loc[:, axMask]
            plotGroupData.columns.name = 'bin'
            originalDataForPlot = plotGroupData.stack().to_frame(name='signal').reset_index()
            fig, ax = plt.subplots(1, 3, sharex=True)
            sns.lineplot(
                x='bin', y='signal', hue='level_0',
                units='level_0', estimator=None,
                ax=ax[0], palette='Reds',
                data=originalDataForPlot)
            ax[0].set_title('Original')
            sns.lineplot(
                x='bin', y='signal', errorbar='se',
                # units='level_0', estimator=None,
                estimator='mean', ax=ax[1],
                data=originalDataForPlot)
            ax[1].set_title('Mean across channels')
            correctedData = expResidCAR.loc[:, timeBinMask].copy().loc[:, axMask]
            correctedData.columns = correctedData.columns.astype(float)
            correctedData.columns.name = 'bin'
            correctedDataForPlot = correctedData.stack().to_frame(name='signal').reset_index()
            sns.lineplot(
                x='bin', y='signal', hue='level_0',
                units='level_0', estimator=None,
                ax=ax[2], palette='Blues',
                data=correctedDataForPlot)
            ax[2].set_xlim(axLims)
            ax[2].set_title('Corrected')
            ax[0].get_shared_y_axes().join(ax[0], ax[1])
            plt.show()
#
ecapDF = pd.concat([rawEcapDF] + newComponents).reset_index(drop=True)
#
resultPath = os.path.join(
    calcSubFolder, 'lmfit',
    prefix + '_{}_{}_lmfit_signals_CAR.parquet'.format(
        arguments['inputBlockSuffix'], arguments['window']))
if os.path.exists(resultPath):
    os.remove(resultPath)
ecapDF.to_parquet(resultPath, engine="fastparquet")
