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
    --showFigures                          load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --lfpBlockSuffix=lfpBlockSuffix        which trig_ block to pull [default: pca]
    --emgBlockSuffix=emgBlockSuffix        which trig_ block to pull [default: pca]
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
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os, re
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import pingouin as pg
idsl = pd.IndexSlice
import numpy as np
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import dataAnalysis.plotting.spike_sorting_plots as ssplt
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer
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
    context='notebook', style='dark',
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
    figureFolder, arguments['analysisNameLFP'], arguments['alignFolderName'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
if arguments['processAll']:
    prefix = 'Block'
else:
    prefix = ns5FileName
#
corrPathEMG = os.path.join(
    calcSubFolderEMG,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['emgBlockSuffix'], arguments['window']))
resultPathEMG = os.path.join(
    calcSubFolderEMG,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['emgBlockSuffix'], arguments['window']))
print('loading {}'.format(resultPathEMG))
#  Overrides
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
trialMetaNames = [
    'segment', 'originalIndex', 't',
    ] + stimulusConditionNames
keepCols = trialMetaNames + ['feature']
mapAnnotNames = ['xcoords', 'ycoords', 'whichArray']
ecapPath = os.path.join(
    calcSubFolderLFP, 'lmfit',
    prefix + '_{}_{}_lmfit_signals_CAR.parquet'.format(
        arguments['lfpBlockSuffix'], arguments['window']))
rawEcapDF = pd.read_parquet(ecapPath, engine='fastparquet')

rawEcapDF.loc[:, 'nominalCurrent'] = rawEcapDF['nominalCurrent'] * (-1)
# simplify electrode names
rawEcapDF.loc[:, 'electrode'] = rawEcapDF['electrode'].apply(lambda x: x[1:])
#
removeStimOnRec = True
if removeStimOnRec:
    ecapRmMask = (rawEcapDF['electrode'] == rawEcapDF['feature'])
    rawEcapDF.drop(index=rawEcapDF.index[ecapRmMask], inplace=True)
for colName in keepCols:
    if colName not in rawEcapDF.columns:
        rawEcapDF.loc[:, colName] = 0.
# rawEcapDF.shape
ecapTWinStart = lmfitFunKWArgs['tBounds'][0]
ecapTWinStop = 3e-3
qLimsEcap = (5e-3, 1-5e-3)
emgTWinStart, emgTWinStop = 0, 39e-3
emgCorrThreshold = 0.75
#
barVarName = 'absR'
whichRaucLFP = 'rauc'
whichRaucEMG = 'rauc'
whichEcap = 'exp_resid_CAR'

if RCCalcOpts['keepElectrodes'] is not None:
    keepDataMask = rawEcapDF['electrode'].isin(RCCalcOpts['keepElectrodes'])
    rawEcapDF = rawEcapDF.loc[keepDataMask, :]

ecapDF = rawEcapDF.loc[rawEcapDF['regrID'] == whichEcap, :].copy()
ecapDF.drop(columns=['regrID'] + mapAnnotNames, inplace=True)
del rawEcapDF
'''targetOrExpResidMask = rawEcapDF['regrID'].isin(['target', 'exp_resid_CAR'])
dropIndices = rawEcapDF.loc[~targetOrExpResidMask].index
rawEcapDF.drop(index=dropIndices, inplace=True)'''
# 
ecapDF.set_index(
    [
        idxName
        for idxName in keepCols
        if idxName in ecapDF.columns],
    inplace=True)
ecapDF.columns = ecapDF.columns.astype(float)
# ecap detrending
'''ecapDetrender = ash.genDetrender(
    timeWindow=[35e-3, 39e-3], useMean=False)'''
ecapDetrender = ash.genDetrender(
    timeWindow=[ecapTWinStart, ecapTWinStop], useMean=True)
ecapDF = ecapDetrender(ecapDF, None)
#
#
ecapRauc = ash.rAUC(
    ecapDF,
    tStart=ecapTWinStart, tStop=ecapTWinStop).to_frame(name='rauc')
ecapRauc['kruskalStat'] = np.nan
ecapRauc['kruskalP'] = np.nan
for name, group in ecapRauc.groupby(['electrode', 'feature']):
    subGroups = [i['rauc'].to_numpy() for n, i in group.groupby('nominalCurrent')]
    try:
        stat, pval = stats.kruskal(*subGroups, nan_policy='omit')
        ecapRauc.loc[group.index, 'kruskalStat'] = stat
        ecapRauc.loc[group.index, 'kruskalP'] = pval
    except Exception:
        ecapRauc.loc[group.index, 'kruskalStat'] = 0
        ecapRauc.loc[group.index, 'kruskalP'] = 1
derivedAnnot = ['normalizedRAUC', 'standardizedRAUC']

ecapRauc.reset_index(inplace=True)
for annName in derivedAnnot:
    ecapRauc.loc[:, annName] = np.nan
#
# normalizationGrouper = ecapRauc.groupby(['feature'])
# or
# normalizationGrouper = [('all', ecapRauc), ]

for name, group in ecapRauc.groupby(['feature']):
    qScaler = QuantileTransformer(output_distribution='normal')
    qScaler.fit(
        ecapRauc.loc[group.index, 'rauc']
        .to_numpy()
        .reshape(-1, 1))
    ecapRauc.loc[group.index, 'rauc'] = (
        qScaler.transform(
            ecapRauc.loc[group.index, 'rauc']
            .to_numpy()
            .reshape(-1, 1)
        )
    )
for name, group in ecapRauc.groupby(['feature']):
    rauc = group['rauc'].copy()
    minParams = group.groupby([amplitudeFieldName, 'electrode']).mean()['rauc'].idxmin()
    minMask = (group[amplitudeFieldName] == minParams[0]) & (group['electrode'] == minParams[1])
    #
    scaler = StandardScaler()
    scaler.fit(group.loc[minMask, 'rauc'].to_numpy().reshape(-1, 1))
    standardizedRAUC = scaler.transform(group['rauc'].to_numpy().reshape(-1, 1))
    ecapRauc.loc[group.index, 'standardizedRAUC'] = standardizedRAUC
    #
    thisScaler = MinMaxScaler()
    thisScaler.fit(
        rauc.to_numpy().reshape(-1, 1))
    ecapRauc.loc[group.index, 'normalizedRAUC'] = (
        thisScaler.transform(rauc.to_numpy().reshape(-1, 1)))
#
ecapRaucWideDF = ecapRauc.set_index(keepCols)[whichRaucLFP].unstack(level='feature')
#
recCurve = pd.read_hdf(resultPathEMG, 'emgRAUC')
emgDF = pd.read_hdf(resultPathEMG, 'emgRAUC_raw')
plotOptsEMG = pd.read_hdf(resultPathEMG, 'emgRAUC_plotOpts')
emgNL = plotOptsEMG.loc[:, ['featureName', 'EMG Location']].set_index('featureName')['EMG Location']
emgNL = emgNL.apply(lambda x: ' '.join(re.findall('[A-Z][^A-Z]*', x.replace(' ', ''))))
emgPalette = plotOptsEMG.loc[:, ['featureName', 'color']].set_index('featureName')['color']
emgPaletteDesat = emgPalette.apply(sns.desaturate, args=(0.3, ))
#
featToSiteEMG = plotOptsEMG.loc[:, ['featureName', 'EMGSite']].set_index('EMGSite')['featureName']
for idxName in recCurve.index.names:
    if idxName not in keepCols:
        recCurve.index = recCurve.index.droplevel(idxName)
        emgDF.index = emgDF.index.droplevel(idxName)
#
#
emgCorrDF = pd.read_hdf(corrPathEMG, 'noiseCeil').to_frame(name='noiseCeil').reset_index()
#
emgCorrDF.loc[:, 'electrode'] = emgCorrDF['electrode'].apply(lambda x: x[1:])
emgCorrDF.loc[:, 'feature'] = emgCorrDF['feature'].apply(lambda x: x.replace('EmgEnv#0', ''))
emgCorrDF.loc[:, 'nominalCurrent'] = emgCorrDF['nominalCurrent'] * (-1)
emgCorrMaskLookup = emgCorrDF.set_index(stimulusConditionNames + ['feature'])['noiseCeil'] > emgCorrThreshold

#
recCurve.reset_index(inplace=True)
emgDF.reset_index(inplace=True)
# simplify electrode names
recCurve.loc[:, 'electrode'] = recCurve['electrode'].apply(lambda x: x[1:])
emgDF.loc[:, 'electrode'] = emgDF['electrode'].apply(lambda x: x[1:])
#
recCurve.loc[:, 'feature'] = recCurve['featureName'].to_numpy()
emgDF.loc[:, 'feature'] = recCurve['featureName'].to_numpy()
#
#
if RCCalcOpts['significantOnly']:
    recCurve = recCurve.query("(kruskalP < 1e-3)")
if RCCalcOpts['keepElectrodes'] is not None:
    keepDataMask = recCurve['electrode'].isin(RCCalcOpts['keepElectrodes'])
    recCurve = recCurve.loc[keepDataMask, :]
if RCCalcOpts['keepFeatures'] is not None:
    keepDataMask = recCurve['featureName'].isin(RCCalcOpts['keepFeatures'])
    recCurve = recCurve.loc[keepDataMask, :]
emgDF = emgDF.loc[emgDF.index.isin(recCurve.index), :]
hueOrderEMG = (
    featToSiteEMG.loc[featToSiteEMG.isin(recCurve['featureName'])]
        .sort_index().to_numpy())

recCurveCorrIdx = pd.MultiIndex.from_frame(recCurve.loc[:, stimulusConditionNames + ['feature']])
recCurve.loc[:, 'passesCorrMask'] = recCurveCorrIdx.map(emgCorrMaskLookup).to_numpy()
passingIndices = recCurve.loc[recCurve['passesCorrMask'], :].index

for name, group in recCurve.groupby(['feature']):
    qScaler = QuantileTransformer(output_distribution='normal')
    trainIndices = np.intersect1d(group.index, passingIndices)
    if trainIndices.size > 0:
        qScaler.fit(
            recCurve.loc[trainIndices, 'rauc']
            .to_numpy()
            .reshape(-1, 1))
        recCurve.loc[group.index, 'rauc'] = (
            qScaler.transform(
                recCurve.loc[group.index, 'rauc']
                .to_numpy()
                .reshape(-1, 1)
            )
        )
    else:
        recCurve.loc[group.index, 'rauc'] = 0.
#
recCurve.set_index(keepCols, inplace=True)
emgDF.set_index(keepCols, inplace=True)
emgDF.columns = emgDF.columns.astype(float)
recCurveWideDF = recCurve[whichRaucEMG].unstack(level='feature')
recCurveMaskDF = recCurve['passesCorrMask'].unstack(level='feature')
emgRCIndex = pd.MultiIndex.from_frame(recCurveWideDF.index.to_frame().reset_index(drop=True).loc[:, ['segment', 'originalIndex', 't']])
ecapRCIndex = pd.MultiIndex.from_frame(ecapRaucWideDF.index.to_frame().reset_index(drop=True).loc[:, ['segment', 'originalIndex', 't']])
commonIdx = np.intersect1d(emgRCIndex, ecapRCIndex)
# commonIdx = np.intersect1d(recCurveWideDF.index.to_numpy(), ecapRaucWideDF.index.to_numpy())
recCurveWideDF = recCurveWideDF.loc[emgRCIndex.isin(commonIdx), :]
ecapRaucWideDF = ecapRaucWideDF.loc[ecapRCIndex.isin(commonIdx), :]
recCurveMaskDF = recCurveMaskDF.loc[emgRCIndex.isin(commonIdx), :]
ecapRaucMaskDF = ecapRaucWideDF.notna()

presentAmplitudes = sorted(ecapRaucWideDF.index.get_level_values('nominalCurrent').unique())
if 0 not in presentAmplitudes:
    presentAmplitudes = sorted(presentAmplitudes + [0])
stimAmpPalette = pd.Series(
    sns.color_palette(
        "ch:1.6,-.3,dark=.1,light=0.7,reverse=1", len(presentAmplitudes)),
    index=presentAmplitudes)
stimAmpPaletteDesat = stimAmpPalette.apply(sns.desaturate, args=(0.3, ))
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
#
corrDFIndex = pd.MultiIndex.from_product(
    [recCurveWideDF.columns, ecapRaucWideDF.columns],
    names=['emg', 'lfp'])
corrDF = pd.DataFrame(np.nan, index=corrDFIndex, columns=['R', 'pval', 'valid'] + mapAnnotNames)
#
for emgName, lfpName in corrDF.index:
    rowIdx = (emgName, lfpName)
    finiteMask = (recCurveWideDF[emgName].notna().to_numpy() & ecapRaucWideDF[lfpName].notna().to_numpy())
    sizeMask = (recCurveMaskDF[emgName].to_numpy() & ecapRaucMaskDF[lfpName].to_numpy())
    thisMask = finiteMask & sizeMask
    if thisMask.sum() > 10:
        # more than 10 pairs of #s to calculate on
        '''corrDF.loc[rowIdx, 'R'] = np.corrcoef(
            recCurveWideDF.loc[thisMask, emgName],
            ecapRaucWideDF.loc[thisMask, lfpName])[0, 1]'''
        spr, pvalue = stats.spearmanr(
            recCurveWideDF.loc[thisMask, emgName],
            ecapRaucWideDF.loc[thisMask, lfpName]
            )
        corrDF.loc[rowIdx, 'R'] = spr
        corrDF.loc[rowIdx, 'pval'] = pvalue
        corrDF.loc[rowIdx, 'valid'] = True
    else:
        corrDF.loc[rowIdx, 'R'] = 0.
        corrDF.loc[rowIdx, 'pval'] = 1.
        corrDF.loc[rowIdx, 'valid'] = False

reject, pValsCorrected = pg.multicomp(corrDF.loc[corrDF['valid'].to_numpy(), 'pval'].to_list())
corrDF.loc[:, 'correctedPVal'] = 1.
corrDF.loc[corrDF['valid'].to_numpy(), 'correctedPVal'] = pValsCorrected
corrDF.loc[:, 'rejectH0'] = corrDF['correctedPVal'] < 1e-2
for annotName in mapAnnotNames:
    lookupSource = mapDF.loc[mapAMask, [annotName, 'topoName']].set_index('topoName')[annotName]
    corrDF.loc[:, annotName] = corrDF.index.get_level_values('lfp').map(lookupSource)

'''corrDF.loc[:, 'xIdx'], corrDF.loc[:, 'yIdx'] = ssplt.coordsToIndices(
    corrDF['xcoords'], corrDF['ycoords'],
    swapXY=True)'''
corrDF.loc[:, 'R2'] = corrDF['R'] ** 2
corrDF.loc[:, 'absR'] = corrDF['R'].abs()
corrDF.loc[:, 'xDummy'] = 1

plotDF = corrDF.reset_index().query('whichArray == "rostral"')

########################################################################
## plot illustrations of RAUC and R
########################################################################
zoomLevel = 1e-3
#
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
        'emgToLfpCorrelationExamples'))
maxEMGName, maxLFPName = plotDF.loc[plotDF[barVarName].idxmax(), ['emg', 'lfp']]
minEMGName, minLFPName = plotDF.loc[plotDF[barVarName].idxmin(), ['emg', 'lfp']]
exAmplitude = max(presentAmplitudes)

exLfpSingleTrial = (
    ecapDF
    .xs(maxLFPName, level='feature')
    .xs(exAmplitude, level=amplitudeFieldName, drop_level=False)
    .iloc[0, :])
exEmgSingleTrial = (
    emgDF
    .xs(maxEMGName, level='feature')
    .xs(exAmplitude, level=amplitudeFieldName, drop_level=False)
    .iloc[0, :])
if True:
    with PdfPages(pdfPath) as pdf:
        for exEMGName in recCurveWideDF.columns:
            for exLFPName in ecapRaucWideDF.columns:
                if not ((exEMGName, exLFPName) == (maxEMGName, maxLFPName) or (exEMGName, exLFPName) == (minEMGName, minLFPName)):
                    continue
                # pdb.set_trace()
                print('Plotting joint distributions of {} and {}'.format(exEMGName, exLFPName))
                finiteMask = (recCurveWideDF[exEMGName].notna().to_numpy() & ecapRaucWideDF[exLFPName].notna().to_numpy())
                sizeMask = (recCurveMaskDF[exEMGName].to_numpy() & ecapRaucMaskDF[exLFPName].to_numpy())
                thisMask = finiteMask & sizeMask
                if thisMask.sum() > 10:
                    g = sns.jointplot(
                        x=ecapRaucWideDF.loc[thisMask, exLFPName],
                        y=recCurveWideDF.loc[thisMask, exEMGName], kind='reg',
                        color=emgPalette[exEMGName],
                        xlim=ecapRaucWideDF.loc[thisMask, exLFPName].quantile([zoomLevel, 1-zoomLevel]).to_list(),
                        ylim=recCurveWideDF.loc[thisMask, exEMGName].quantile([zoomLevel, 1-zoomLevel]).to_list(),)
                    if (exEMGName, exLFPName) == (maxEMGName, maxLFPName):
                        # pdb.set_trace()
                        g.ax_joint.text(
                            ecapRaucWideDF.loc[exEmgSingleTrial.name, exLFPName],
                            recCurveWideDF.loc[exEmgSingleTrial.name, exEMGName],
                            '*', ha='center', va='center'
                        )
                    g.set_axis_labels(
                        '{} electrode {} RAUC (a.u.)'.format(
                            lfpNL.loc[exLFPName, 'whichArray'], int(lfpNL.loc[exLFPName, 'elecID'])),
                        '{} RAUC (a.u.)'.format(emgNL[exEMGName])
                        )
                    scoreMask = (plotDF['emg'] == exEMGName) & (plotDF['lfp'] == exLFPName)
                    thisScore = plotDF.loc[scoreMask, barVarName]
                    thisPVal = plotDF.loc[scoreMask, 'pval']
                    figTitle = g.fig.suptitle('{} = {:.3f}, p={:.3f}'.format(
                        barVarName, float(thisScore), float(thisPVal)))
                    g.fig.set_size_inches((10, 5))
                    pdf.savefig(
                        bbox_inches='tight', pad_inches=0.2,
                        bbox_extra_artists=[figTitle]
                        )
                    if (exEMGName, exLFPName) == (maxEMGName, maxLFPName) or (exEMGName, exLFPName) == (minEMGName, minLFPName):
                        if arguments['showFigures']:
                            plt.show()
                        else:
                            plt.close()
                    else:
                        plt.close()
#########
sns.set(
    context='notebook', style='white',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
fig, ax = plt.subplots(1, 2)
lfpAx, emgAx = ax
lfpAxTStart, lfpAxTStop = 1, 8
lfpMaskAx = (
        (ecapDF.columns >= lfpAxTStart * 1e-3) &
        (ecapDF.columns < lfpAxTStop * 1e-3))
lfpMask = (
        (ecapDF.columns >= ecapTWinStart) &
        (ecapDF.columns < ecapTWinStop))
lfpAx.fill_between(
    exLfpSingleTrial.index[lfpMask] * 1e3, exLfpSingleTrial[lfpMask], 0,
    edgecolor=stimAmpPalette[exAmplitude],
    facecolor=stimAmpPaletteDesat[exAmplitude],
    linewidth=0, alpha=0.5)
lfpAx.plot(
    exLfpSingleTrial.index[lfpMaskAx] * 1e3, exLfpSingleTrial[lfpMaskAx],
    color=stimAmpPalette[exAmplitude], lw=2)
lfpAx.set_xlabel('Time (msec)')
lfpAx.set_ylabel('{} LFP (uV)'.format(maxLFPName))
lfpAx.set_xlim([lfpAxTStart, lfpAxTStop])
lfpAx.set_title('LFP RAUC ({} to {} msec)'.format(int(ecapTWinStart * 1e3), int(ecapTWinStop * 1e3)))

emgAxTStart, emgAxTStop = 0, 80
emgMaskAx = (
        (emgDF.columns >= emgAxTStart * 1e-3) &
        (emgDF.columns < emgAxTStop * 1e-3))
emgMask = (
        (emgDF.columns >= emgTWinStart) &
        (emgDF.columns < emgTWinStop))
emgAx.fill_between(
    exEmgSingleTrial.index[emgMask] * 1e3, exEmgSingleTrial[emgMask], 0,
    edgecolor=emgPalette[maxEMGName],
    facecolor=emgPaletteDesat[maxEMGName],
    linewidth=0, alpha=0.5)
emgAx.plot(
    exEmgSingleTrial.index[emgMaskAx] * 1e3, exEmgSingleTrial[emgMaskAx],
    color=emgPalette[maxEMGName], lw=2)
emgAx.set_title('EMG RAUC ({} to {} msec)'.format(int(emgTWinStart * 1e3), int(emgTWinStop * 1e3)))
emgAx.set_xlim([emgAxTStart, emgAxTStop])
emgAx.set_xlabel('Time (msec)')
emgAx.set_ylabel('{} EMG (uV)'.format(emgNL[maxEMGName]))
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
        'raucBoundsIllustration'))
fig.set_size_inches((10, 5))
plt.savefig(
    pdfPath,
    bbox_inches='tight', pad_inches=0.2,
    # bbox_extra_artists=[leg, figYLabel, figXLabel]
    )
if arguments['showFigures']:
    plt.show()
else:
    plt.close()
########################################################################
## plot response magnitude correlation
########################################################################
sns.set(
    context='notebook', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
'''
    asp.genTitleAnnotator(
        template='{}', colNames=['lfp'],
        dropNaNCol='R2', shared=False),'''
'''
plotProcFuns = [
    asp.genGridAnnotator(
        xpos=1, ypos=1, template='{}', colNames=['lfp'],
        dropNaNCol=barVarName,
        textOpts={
            'verticalalignment': 'top',
            'horizontalalignment': 'right'
        }, shared=False),
    asp.genTicksToScale(
        lineOpts={'lw': 2}, shared=True,
        xUnitFactor=0, xUnits=False,
        yUnitFactor=1, yUnits='', limFracX=0., limFracY=.2,
        dropNaNCol=barVarName, scaleBarPosition=[-0.35, 0.4],
        dX=0., yTicks=[0]
        )
        ]'''
def axModFun(g, ro, co, hu, dataSubset):
    limFrac = 0.9
    emptySubset = (
        (dataSubset.empty) or
        (dataSubset[barVarName].isna().all()))
    if not hasattr(g.axes[ro, co], 'axWasChanged'):
        g.axes[ro, co].axWasChanged = True
        if not emptySubset:
                currYTicks = g.axes[ro, co].get_yticks()
                currYLim = g.axes[ro, co].get_ylim()
                print(currYLim[0])
                if len(currYTicks):
                    g.axes[ro, co].set_yticks([
                        np.round(currYLim[0] * limFrac, decimals=1),
                        0,
                        np.round(currYLim[-1] * limFrac, decimals=1)])
        else:
            g.axes[ro, co].set_xticks([])
            g.axes[ro, co].set_facecolor('w')
    return
#
def changePatchAlpha(g, ro, co, hu, dataSubset):
    emptySubset = (
        (dataSubset.empty) or
        (dataSubset[barVarName].isna().all()))
    if not emptySubset:
        # hu seems broken in this version of catplot, override it
        for hu, thisPatch in enumerate(g.axes[ro, co].patches):
            # pdb.set_trace() ##
            thisSubset = dataSubset.iloc[hu, :]
            thisPatch = g.axes[ro, co].patches[hu]
            thisPatch.set_edgecolor((0., 0., 0., 0.))
            if thisSubset['rejectH0']:
                xPos = thisPatch.get_x() + thisPatch.get_width() / 2
                yPos = thisPatch.get_y() + thisPatch.get_height()
                g.axes[ro, co].text(xPos, yPos, '*', ha='center', va='center')
            if (thisSubset['emg'] == maxEMGName) and (thisSubset['lfp'] == maxLFPName):
                thisPatch.set_edgecolor((0., 0., 0., 1.))
    return
#
plotProcFuns = [
    axModFun, changePatchAlpha
    ]
g = sns.catplot(
    row='ycoords', col='xcoords', y=barVarName, x='xDummy',
    data=plotDF, height=5, aspect=.7,
    kind='bar', hue='emg', palette=emgPalette.to_dict(),
    hue_order=sorted(plotDF['emg'].unique()),
    )
for (ro, co, hu), dataSubset in g.facet_data():
    print('ro = {}, co = {}, hu={}'.format(ro, co, hu))
    emptySubset = (
            (dataSubset.empty) or
            (dataSubset[barVarName].isna().all()))
    if len(plotProcFuns):
        for procFun in plotProcFuns:
            procFun(g, ro, co, hu, dataSubset)
leg = g._legend
titleOverrides = {'emg': 'Muscle Name'}
if leg is not None:
    for t in leg.texts:
        tContent = t.get_text()
        print(tContent)
        if tContent in emgNL.index:
            t.set_text(emgNL[tContent])
        elif tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
g.set_titles('')
g.set_xlabels('')
g.set_ylabels('')
g.set_xticklabels('')
g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
figYLabel = g.fig.supylabel('Spearman\'s R', x=0.1)
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
        'emgToLfpCorrelation'))
plt.savefig(
    pdfPath,
    bbox_inches='tight', pad_inches=0.2,
    bbox_extra_artists=[leg, figYLabel])
if arguments['showFigures']:
    plt.show()
else:
    plt.close()

########################################################################
## plot lfp rc
########################################################################
sns.set(
    context='notebook', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
plotEcapRC = ecapRauc.copy()
for annotName in mapAnnotNames:
    lookupSource = mapDF.loc[mapAMask, [annotName, 'topoName']].set_index('topoName')[annotName]
    plotEcapRC.loc[:, 'feature_' + annotName] = plotEcapRC['feature'].map(lookupSource)
    plotEcapRC.loc[:, 'electrode_' + annotName] = plotEcapRC['electrode'].map(lookupSource)
'''plotEcapRC.loc[:, 'feature_xIdx'], plotEcapRC.loc[:, 'feature_yIdx'] = ssplt.coordsToIndices(
    plotEcapRC['feature_xcoords'], plotEcapRC['feature_ycoords'],
    swapXY=True)
plotEcapRC.loc[:, 'electrode_xIdx'], plotEcapRC.loc[:, 'electrode_yIdx'] = ssplt.coordsToIndices(
    plotEcapRC['electrode_xcoords'], plotEcapRC['electrode_ycoords'],
    swapXY=True)'''
plotEcapRC = plotEcapRC.query('feature_whichArray == "rostral"')
if RCPlotOpts['keepElectrodes'] is not None:
    keepDataMask = plotEcapRC['electrode'].isin(RCPlotOpts['keepElectrodes'])
    plotEcapRC = plotEcapRC.loc[keepDataMask, :]
#
'''
    asp.genGridAnnotator(
        xpos=1, ypos=1, template='{}', colNames=['feature'],
        dropNaNCol=whichRaucLFP,
        textOpts={
            'verticalalignment': 'top',
            'horizontalalignment': 'right'
        }, shared=False),
'''
whichRaucLFP = 'normalizedRAUC'
def axModFun(g, ro, co, hu, dataSubset):
    emptySubset = (
        (dataSubset.empty) or
        (dataSubset[whichRaucLFP].isna().all()))
    if not hasattr(g.axes[ro, co], 'axWasChanged'):
        g.axes[ro, co].axWasChanged = True
        if emptySubset:
            g.axes[ro, co].set_xticks([])
            g.axes[ro, co].set_facecolor('w')
    return
plotProcFuns = [
    asp.genTicksToScale(
        lineOpts={'lw': 2}, shared=True,
        xUnitFactor=1, xUnits='uA',
        yUnitFactor=1, yUnits='', limFracX=0.2, limFracY=.2,
        dropNaNCol=whichRaucLFP, scaleBarPosition=[600., 0.2],
        # dX=0., yTicks=[0]
    ),
    axModFun]
g = sns.relplot(
    col='feature_xcoords',
    row='feature_ycoords',
    x=amplitudeFieldName,
    y=whichRaucLFP,
    hue='electrode_xcoords', palette=xPalette.to_dict(),
    # hue='electrode_ycoords', palette=yUnshiftedPalette.to_dict(),
    kind='line', data=plotEcapRC,
    height=5, aspect=.7, ci='sem', estimator='mean',
    facet_kws=dict(sharey=True, sharex=True), lw=2,
    )
for (ro, co, hu), dataSubset in g.facet_data():
    emptySubset = (
            (dataSubset.empty) or
            (dataSubset[whichRaucLFP].isna().all()))
    if len(plotProcFuns):
        for procFun in plotProcFuns:
            procFun(g, ro, co, hu, dataSubset)
leg = g._legend
titleOverrides = {
    'electrode_xcoords': 'Stimulation location\nw.r.t. midline\n(mm)'
    }
entryOverrides = mapDF.loc[:, ['xcoords', 'electrodeSide', 'whichArray']].drop_duplicates().set_index('xcoords')
if leg is not None:
    for t in leg.texts:
        tContent = t.get_text()
        if tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
        # elif tContent.replace('.', '', 1).isdigit():
        # e.g. is numeric
        else:
            try:
                tNumeric = float(tContent)
                t.set_text('{:.2f}'.format(tNumeric / 10))
                '''if tNumeric in entryOverrides.index:
                    t.set_text('{} {}'.format(
                        entryOverrides.loc[float(tContent), 'whichArray'],
                        entryOverrides.loc[float(tContent), 'electrodeSide']))'''
            except Exception:
                pass
g.set_titles('')
g.set_xlabels('')
g.set_ylabels('')
g.set_yticklabels('')
g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
figYLabel = g.fig.supylabel('LFP RAUC (a.u.)', x=0.1)
figXLabel = g.fig.supxlabel('Stimulation Amplitude (uA)', y=0.1)
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
        'lfpRAUC'))
plt.savefig(
    pdfPath,
    bbox_inches='tight', pad_inches=0.2,
    bbox_extra_artists=[leg, figYLabel, figXLabel]
    ) #
if arguments['showFigures']:
    plt.show()
else:
    plt.close()
########################################################################
## plot emg rc
########################################################################
#
whichRaucEMG = 'normalizedRAUC'
plotEmgRC = recCurve.reset_index()
for annotName in mapAnnotNames:
    lookupSource = mapDF.loc[mapAMask, [annotName, 'topoName']].set_index('topoName')[annotName]
    plotEmgRC.loc[:, 'electrode_' + annotName] = plotEmgRC['electrode'].map(lookupSource)
####
if RCPlotOpts['significantOnly']:
    plotEmgRC = plotEmgRC.query("(kruskalP < 1e-3)")
if RCPlotOpts['keepElectrodes'] is not None:
    keepDataMask = plotEmgRC['electrode'].isin(RCPlotOpts['keepElectrodes'])
    plotEmgRC = plotEmgRC.loc[keepDataMask, :]
if RCPlotOpts['keepFeatures'] is not None:
    keepDataMask = plotEmgRC['featureName'].isin(RCPlotOpts['keepFeatures'])
    plotEmgRC = plotEmgRC.loc[keepDataMask, :]
#####
colName = 'electrode_xcoords'

def axModFun(g, ro, co, hu, dataSubset):
    emptySubset = (
        (dataSubset.empty) or
        (dataSubset[whichRaucEMG].isna().all()))
    if not hasattr(g.axes[ro, co], 'axWasChanged'):
        g.axes[ro, co].axWasChanged = True
        if emptySubset:
            g.axes[ro, co].set_xticks([])
            g.axes[ro, co].set_facecolor('w')
        else:
            titleText = '{:0.2f} mm'.format(dataSubset[colName].iloc[0] / 10)
            g.axes[ro, co].set_title(titleText)
    return


plotProcFuns = [
    asp.genTicksToScale(
        lineOpts={'lw': 2}, shared=True,
        xUnitFactor=1, xUnits='uA',
        yUnitFactor=1, yUnits='', limFracX=0.2, limFracY=.2,
        dropNaNCol=whichRaucEMG, scaleBarPosition=[500., 0.6],
        # dX=0., yTicks=[0]
    ),
    axModFun]
colOrder = sorted(np.unique(plotEmgRC[colName]))
hueName = 'featureName'
colWrap = min(3, len(colOrder))
height, aspect = 5, 1.5

g = sns.relplot(
    data=plotEmgRC,
    col=colName,
    col_order=colOrder,
    # col_wrap=colWrap,
    # row='EMGSide',
    # x='normalizedAmplitude',
    x=amplitudeFieldName,
    y=whichRaucEMG,
    style='EMGSide', style_order=['Right', 'Left'],
    hue=hueName, hue_order=hueOrderEMG, palette=emgPalette.to_dict(),
    kind='line',
    height=height, aspect=aspect, ci='sem', estimator='mean',
    facet_kws=dict(sharey=True, sharex=False, legend_out=True), lw=2,
    )
# g.fig.set_size_inches(colWrap * height * aspect + 10, height + 2)
# plt.tight_layout(pad=.1)
for (ro, co, hu), dataSubset in g.facet_data():
    emptySubset = (
            (dataSubset.empty) or
            (dataSubset[whichRaucEMG].isna().all()))
    if len(plotProcFuns):
        for procFun in plotProcFuns:
            procFun(g, ro, co, hu, dataSubset)
leg = g._legend
titleOverrides = {
    'featureName': 'Muscle Name',
    'EMGSide': 'Side of Body'
    }
g.set_xlabels('')
g.set_ylabels('')
g.set_yticklabels('')
g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
figYLabel = g.fig.supylabel('EMG RAUC (a.u.)', x=0.1)
figXLabel = g.fig.supxlabel('Stimulation Amplitude (uA)', y=0.1)
figTitle = g.fig.suptitle('Stimulation location (mediolateral, w.r.t. midline')
if leg is not None:
    for t in leg.texts:
        tContent = t.get_text()
        if tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
        elif tContent in emgNL.index:
            t.set_text('{}'.format(emgNL[tContent]))
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
        'emgRAUC'))
plt.savefig(
    pdfPath,
    bbox_inches='tight', pad_inches=0.2,
    bbox_extra_artists=[leg, figYLabel, figXLabel]
    )
if arguments['showFigures']:
    plt.show()
else:
    plt.close()