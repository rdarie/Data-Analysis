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
import matplotlib.ticker as mpltk
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
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer

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
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 7
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV
####
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
emgCorrThreshold = 0.1
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
    qScaler = PowerTransformer()
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
emgPaletteDarker = emgPalette.apply(sns.set_hls_values, args=(None, .4, None))
#
featToSiteEMG = plotOptsEMG.loc[:, ['featureName', 'EMGSite']].set_index('EMGSite')['featureName']
for idxName in recCurve.index.names:
    if idxName not in keepCols:
        recCurve.index = recCurve.index.droplevel(idxName)
        emgDF.index = emgDF.index.droplevel(idxName)
#
emgCorrDF = pd.read_hdf(corrPathEMG, 'noiseCeil').to_frame(name='noiseCeil').reset_index()
#
emgCorrDF.loc[:, 'electrode'] = emgCorrDF['electrode'].apply(lambda x: x[1:])
emgCorrDF.loc[:, 'feature'] = emgCorrDF['feature'].apply(lambda x: x.replace('EmgEnv#0', '').replace('Emg#0', ''))
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

presentAmplitudes = sorted(ecapRaucWideDF.index.get_level_values(amplitudeFieldName).unique())
presentElectrodes = sorted(ecapRaucWideDF.index.get_level_values('electrode').unique())
presentRates = sorted(ecapRaucWideDF.index.get_level_values('RateInHz').unique())
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
mapDF.loc[:, 'shifted_xcoords'] = mapDF['xcoords'].copy()
electrodeMaxXDistance = mapDF['xcoords'].max() - mapDF['xcoords'].min()
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
uniqueX = np.unique(mapDF['xcoords'])
xUnshiftedPalette = pd.Series(
    sns.color_palette('crest', n_colors=uniqueX.size),
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
        .map(xUnshiftedPalette))
xPalette = xPalette.loc[~xPalette.index.duplicated()]
#
xPaletteDesat = xPalette.apply(sns.desaturate, args=(0.3, ))
#mapDF.loc[:, ['topoName', 'xcoords']]
lfpXPalette = (
    mapDF
        .loc[:, ['topoName', 'xcoords']]
        .set_index('topoName')['xcoords']
        .map(xUnshiftedPalette))
lfpXPalette = lfpXPalette.loc[~lfpXPalette.index.duplicated()]
lfpXPaletteDesat = lfpXPalette.apply(sns.desaturate, args=(0.3, ))
#
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
        # spr, pvalue = stats.pearsonr(
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
corrDF.loc[:, 'rejectH0'] = corrDF['correctedPVal'] < 5e-3
print('{} conditions reject H0 out of {}'.format(corrDF['rejectH0'].sum(), corrDF['rejectH0'].size))
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
if True:
    quantileZoomLevel = 1e-3
    #
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
            'emgToLfpCorrelationExamples'))
    maxEMGName, maxLFPName = plotDF.loc[plotDF[barVarName].idxmax(), ['emg', 'lfp']]
    minEMGName, minLFPName = plotDF.loc[plotDF[barVarName].idxmin(), ['emg', 'lfp']]
    exAmplitude = presentAmplitudes[-2]
    exElectrode = presentElectrodes[0]
    exRate = presentRates[0]
    exLfpSingleTrial = (
        ecapDF
        .xs(maxLFPName, level='feature')
        .xs(exAmplitude, level=amplitudeFieldName, drop_level=False)
        .xs(exRate, level='RateInHz', drop_level=False)
        .xs(exElectrode, level='electrode', drop_level=False)
        .iloc[0, :])
    exEmgSingleTrial = (
        emgDF
        .xs(maxEMGName, level='feature')
        .xs(exAmplitude, level=amplitudeFieldName, drop_level=False)
        .xs(exRate, level='RateInHz', drop_level=False)
        .xs(exElectrode, level='electrode', drop_level=False)
        .iloc[0, :])
    with PdfPages(pdfPath) as pdf:
        for exEMGName in recCurveWideDF.columns:
            for exLFPName in ecapRaucWideDF.columns:
                if not ((exEMGName, exLFPName) == (maxEMGName, maxLFPName) or (exEMGName, exLFPName) == (minEMGName, minLFPName)):
                    continue
                print('Plotting joint distributions of {} and {}'.format(exEMGName, exLFPName))
                finiteMask = (recCurveWideDF[exEMGName].notna().to_numpy() & ecapRaucWideDF[exLFPName].notna().to_numpy())
                sizeMask = (recCurveMaskDF[exEMGName].to_numpy() & ecapRaucMaskDF[exLFPName].to_numpy())
                thisMask = finiteMask & sizeMask
                if thisMask.sum() > 10:
                    fig, ax = plt.subplots(1, 1, figsize=(2.7, 1.5))
                    ax = sns.regplot(
                        x=ecapRaucWideDF.loc[thisMask, exLFPName],
                        y=recCurveWideDF.loc[thisMask, exEMGName],
                        color=emgPalette[exEMGName]
                        )
                    xLim = ecapRaucWideDF.loc[thisMask, exLFPName].quantile(
                        [quantileZoomLevel, 1 - quantileZoomLevel]).to_list()
                    ax.set_xlim(xLim)
                    yLim = recCurveWideDF.loc[thisMask, exEMGName].quantile(
                        [quantileZoomLevel, 1 - quantileZoomLevel]).to_list()
                    ax.set_ylim(yLim)
                    '''g = sns.JointGrid(
                        x=ecapRaucWideDF.loc[thisMask, exLFPName],
                        y=recCurveWideDF.loc[thisMask, exEMGName],
                        xlim=ecapRaucWideDF.loc[thisMask, exLFPName].quantile([quantileZoomLevel, 1-quantileZoomLevel]).to_list(),
                        ylim=recCurveWideDF.loc[thisMask, exEMGName].quantile([quantileZoomLevel, 1-quantileZoomLevel]).to_list(),
                        ratio=100, height=3.5, marginal_ticks=False
                        )
                    g.plot_joint(sns.regplot, color=emgPalette[exEMGName])'''
                    #######
                    '''axLW = snsRCParams['axes.linewidth']
                    ax.tick_params(length=axLW * 10, width=axLW, which='major', direction='in', reset=True)
                    ax.tick_params(length=axLW * 5, width=axLW, which='minor', direction='in', reset=True)'''
                    ax.xaxis.set_major_locator(mpltk.MaxNLocator(prune='both', nbins=3))
                    ax.xaxis.set_minor_locator(mpltk.AutoMinorLocator())
                    ax.yaxis.set_major_locator(mpltk.MaxNLocator(prune='both', nbins=3))
                    ax.yaxis.set_minor_locator(mpltk.AutoMinorLocator(n=5))
                    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
                        ax.spines[side].set_visible(True)
                        ax.spines[side].set_color(emgPalette[exEMGName])
                        # ax.spines[side].set_linewidth(axLW)
                    if (exEMGName, exLFPName) == (maxEMGName, maxLFPName):
                        ax.text(
                            ecapRaucWideDF.loc[exEmgSingleTrial.name, exLFPName],
                            recCurveWideDF.loc[exEmgSingleTrial.name, exEMGName],
                            '$\Delta$', ha='center', va='baseline'
                        )
                    ax.set_xlabel(
                        '{} electrode {}\nrAUC (a.u.)'.format(
                            lfpNL.loc[exLFPName, 'whichArray'], int(lfpNL.loc[exLFPName, 'elecID'])))
                    ax.set_ylabel(
                        '{}\nrAUC (a.u.)'.format(emgNL[exEMGName])
                        )
                    scoreMask = (plotDF['emg'] == exEMGName) & (plotDF['lfp'] == exLFPName)
                    thisScore = float(plotDF.loc[scoreMask, barVarName])
                    thisPVal = float(plotDF.loc[scoreMask, 'pval'])
                    '''figTitle = g.fig.suptitle('{} = {:.3f}, p={:.3f}'.format(
                        barVarName, float(thisScore), float(thisPVal)))'''
                    if thisPVal < 1e-6:
                        thisPText = 'p < 1e-6'
                    else:
                        thisPText = 'p = {:.2g}'.format(thisPVal)
                    ax.text(
                        .95, .95, "Spearman's R: {:.2g}; {}".format(thisScore, thisPText),
                        ha='right', va='top', transform=ax.transAxes)
                    if (exEMGName, exLFPName) == (maxEMGName, maxLFPName):
                        ax.text(
                            .025, .95, u'\u2b24', ha='left', va='top', transform=ax.transAxes,
                            color=emgPaletteDarker[exEMGName]
                        )
                    if (exEMGName, exLFPName) == (minEMGName, minLFPName):
                        ax.text(
                            .025, .95, r'$\blacksquare$', ha='left', va='top', transform=ax.transAxes,
                            color=emgPaletteDarker[exEMGName]
                        )
                    print('Applying tight layout on distributions {} and {}'.format(exEMGName, exLFPName))
                    fig_width, fig_height = fig.get_size_inches()
                    ax.text(
                        styleOpts['panel_heading.pad'], fig_height - styleOpts['panel_heading.pad'], 'A.', fontweight='bold',
                        ha='left', va='top',
                        fontsize=8, transform=fig.dpi_scale_trans)
                    fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                    pdf.savefig(bbox_inches='tight')
                    if (exEMGName, exLFPName) == (maxEMGName, maxLFPName) or (exEMGName, exLFPName) == (minEMGName, minLFPName):
                        if arguments['showFigures']:
                            plt.show()
                        else:
                            plt.close()
                    else:
                        plt.close()
#########
# plot example trials
#########
if True:
    exPlotKWArgs = {'lw': 0.25}
    fig, ax = plt.subplots(2, 1, figsize=(2.5, 1.5))
    lfpAx, emgAx = ax
    lfpAxTStart, lfpAxTStop = 0, 8
    lfpMaskAx = (
            (ecapDF.columns >= lfpAxTStart * 1e-3) &
            (ecapDF.columns < lfpAxTStop * 1e-3))
    lfpMask = (
            (ecapDF.columns >= ecapTWinStart) &
            (ecapDF.columns < ecapTWinStop))
    lfpAx.fill_between(
        exLfpSingleTrial.index[lfpMask] * 1e3, exLfpSingleTrial[lfpMask], 0,
        edgecolor=lfpXPalette[exElectrode],
        facecolor=lfpXPaletteDesat[exElectrode],
        linewidth=0, alpha=0.5)
    lfpLabel = '{} electrode {}'.format(lfpNL.loc[maxLFPName, 'whichArray'], int(lfpNL.loc[maxLFPName, 'elecID']))
    lfpLines, = lfpAx.plot(
        exLfpSingleTrial.index[lfpMaskAx] * 1e3, exLfpSingleTrial[lfpMaskAx],
        color=lfpXPalette[maxLFPName], label=lfpLabel, **exPlotKWArgs)
    # lfpAx.set_xlabel('Time (msec)')
    lfpAx.set_ylabel('ECAP (uV)'.format(maxLFPName))
    lfpAx.set_xlim([lfpAxTStart, lfpAxTStop])
    lfpLims = lfpAx.get_ylim()
    lfpAx.set_xlim([lfpAxTStart, lfpAxTStop])
    axLW = snsRCParams['axes.linewidth']
    lfpAx.tick_params(length=axLW * 10, width=axLW, which='major', direction='in', reset=True)
    lfpAx.tick_params(length=axLW * 5, width=axLW, which='minor', direction='in', reset=True)
    # axTickIncr = lambda x: 10 ** np.round(np.log10((x[1] - x[0]) / 5))
    #
    # lfpAx.yaxis.set_major_locator(mpltk.MultipleLocator(axTickIncr(lfpLims)))
    lfpAx.yaxis.set_major_locator(mpltk.MaxNLocator(3))
    lfpAx.yaxis.set_minor_locator(mpltk.AutoMinorLocator(2))
    # lfpAx.xaxis.set_major_locator(mpltk.MultipleLocator(axTickIncr([lfpAxTStart, lfpAxTStop])))
    lfpAx.yaxis.set_major_locator(mpltk.MaxNLocator(3))
    lfpAx.xaxis.set_minor_locator(mpltk.AutoMinorLocator(2))
    #
    '''for side in lfpAx.spines.keys():  # 'top', 'bottom', 'left', 'right'
        lfpAx.spines[side].set_color(lfpXPalette[maxLFPName])'''
    lfpAx.text(
        .025, .95, "$\searrow$", color=lfpXPalette[exElectrode],
        ha='left', va='top', transform=lfpAx.transAxes)
    leg = lfpAx.legend(handles=[lfpLines], loc='lower right')
    for l in leg.get_lines():
        l.set_lw(styleOpts['legend.lw'])
    fig_width, fig_height = fig.get_size_inches()
    lfpAx.text(
        styleOpts['panel_heading.pad'], fig_height - styleOpts['panel_heading.pad'], 'A.', fontweight='bold',
        ha='left', va='top',
        fontsize=8, transform=fig.dpi_scale_trans)
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
    emgLines, = emgAx.plot(
        exEmgSingleTrial.index[emgMaskAx] * 1e3, exEmgSingleTrial[emgMaskAx],
        color=emgPalette[maxEMGName], label=emgNL[maxEMGName], **exPlotKWArgs)
    # emgAx.set_title('EMG rAUC ({} to {} msec)'.format(int(emgTWinStart * 1e3), int(emgTWinStop * 1e3)))
    emgAx.set_xlim([emgAxTStart, emgAxTStop])
    emgAx.set_xlabel('Time (msec)')
    emgAx.set_ylabel('EMG (uV)'.format(emgNL[maxEMGName]))
    leg = emgAx.legend(handles=[emgLines], loc='lower right')
    for l in leg.get_lines():
        l.set_lw(styleOpts['legend.lw'])
    emgLims = emgAx.get_ylim()
    #
    emgAx.tick_params(length=axLW * 10, width=axLW, which='major', direction='in', reset=True)
    emgAx.tick_params(length=axLW * 5, width=axLW, which='minor', direction='in', reset=True)
    #
    # emgAx.yaxis.set_major_locator(mpltk.MultipleLocator(axTickIncr(emgLims)))
    emgAx.yaxis.set_major_locator(mpltk.MaxNLocator(3))
    emgAx.yaxis.set_minor_locator(mpltk.AutoMinorLocator(2))
    # emgAx.xaxis.set_major_locator(mpltk.MultipleLocator(axTickIncr([emgAxTStart, emgAxTStop])))
    emgAx.xaxis.set_major_locator(mpltk.MaxNLocator(5))
    emgAx.xaxis.set_minor_locator(mpltk.AutoMinorLocator(2))
    #
    '''for side in emgAx.spines.keys():  # 'top', 'bottom', 'left', 'right'
        emgAx.spines[side].set_color(emgPalette[exEMGName])'''
    emgAx.text(
        .025, .95, "$\searrow$", color=emgPaletteDarker[exEMGName],
        ha='left', va='top', transform=emgAx.transAxes)
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
            'raucBoundsIllustration'))
    print('Applying tight layout on bounds illustration')
    fig.tight_layout(pad=styleOpts['tight_layout.pad'])
    plt.savefig(
        pdfPath,
        bbox_inches='tight',
        # pad_inches=0.2,
        # bbox_extra_artists=[leg, figYLabel, figXLabel]
        )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
########################################################################
## plot response magnitude correlation
########################################################################
if True:
    sns.set_style(style='white')
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
        if not hasattr(g, 'anyAxWasChanged'):
            if not emptySubset:
                g.anyAxWasChanged = True
                currYLim = [yl for yl in g.axes[ro, co].get_ylim()]
                currYLim[1] += 0.1
                g.axes[ro, co].set_ylim(currYLim)
        if not hasattr(g.axes[ro, co], 'axWasChanged'):
            g.axes[ro, co].axWasChanged = True
            if g.axes[ro, co].get_subplotspec().is_first_row() and g.axes[ro, co].get_subplotspec().is_first_col():
                fig_width, fig_height = g.fig.get_size_inches()
                g.axes[ro, co].text(
                    styleOpts['panel_heading.pad'], fig_height - styleOpts['panel_heading.pad'], 'B.', fontweight='bold',
                    ha='left', va='top',
                    fontsize=8, transform=g.fig.dpi_scale_trans, zorder=100)
            if not emptySubset:
                if g.axes[ro, co].get_subplotspec().is_first_col():
                    g.axes[ro, co].spines['left'].set_visible(True)
                    # g.axes[ro, co].yaxis.set_major_locator(mpltk.MaxNLocator(5))
                    # g.axes[ro, co].yaxis.set_minor_locator(mpltk.AutoMinorLocator(2))
                    for side in ['top', 'right']:  # 'top', 'bottom', 'left', 'right'
                        g.axes[ro, co].spines[side].set_visible(False)
                else:
                    for side in ['top', 'left', 'right']:  # 'top', 'bottom', 'left', 'right'
                        g.axes[ro, co].spines[side].set_visible(False)
                '''currYTicks = g.axes[ro, co].get_yticks()
                currYLim = [yl for yl in g.axes[ro, co].get_ylim()]
                if len(currYTicks):
                    g.axes[ro, co].set_yticks([
                        np.round(currYLim[0] * limFrac, decimals=1),
                        0,
                        np.round(currYLim[-1] * limFrac, decimals=1)])'''
            else:
                g.axes[ro, co].set_xticks([])
                g.axes[ro, co].set_facecolor('w')
                for side in ['top', 'bottom', 'left', 'right']:  # 'top', 'bottom', 'left', 'right'
                    g.axes[ro, co].spines[side].set_visible(False)
        return
    #
    def changePatchAlpha(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset[barVarName].isna().all()))
        if not emptySubset:
            # hu seems broken in this version of catplot, override it
            resetXLims = [
                g.axes[ro, co].patches[0].get_x(),
                g.axes[ro, co].patches[-1].get_x() + g.axes[ro, co].patches[-1].get_width(),
            ]
            for hu, thisPatch in enumerate(g.axes[ro, co].patches):
                thisSubset = dataSubset.iloc[hu, :]
                thisPatch = g.axes[ro, co].patches[hu]
                # thisPatch.set_edgecolor((0., 0., 0., 0.))
                thisPatch.set_lw(0)
                xPos = thisPatch.get_x() + thisPatch.get_width() / 2
                yPos = thisPatch.get_y() + thisPatch.get_height()
                if "Right" in emgNL[thisSubset['emg']]:
                    thisPatch.set_hatch('//')
                if thisSubset['rejectH0']:
                    sigStar = g.axes[ro, co].text(
                        xPos, yPos, '*',
                        ha='center', va='bottom',
                        fontsize='xx-large', fontweight='bold')
                if (thisSubset['emg'] == maxEMGName) and (thisSubset['lfp'] == maxLFPName):
                    thisPatch.set_edgecolor(emgPaletteDarker[maxEMGName])
                    g.axes[ro, co].text(xPos, 0.1, u'\u2b24', ha='center', va='bottom', color=emgPaletteDarker[maxEMGName])
                if (thisSubset['emg'] == minEMGName) and (thisSubset['lfp'] == minLFPName):
                    thisPatch.set_edgecolor(emgPaletteDarker[minEMGName])
                    g.axes[ro, co].text(xPos, 0.1, r'$\blacksquare$', ha='center', va='bottom', color=emgPaletteDarker[minEMGName])
            g.axes[ro, co].set_xlim(resetXLims)
        return
    #
    plotProcFuns = [
        axModFun, changePatchAlpha
        ]

    targetTotalWidth = 5.9 # inches
    targetTotalHeight = 3.6 # inches
    width = targetTotalWidth / plotDF['xcoords'].unique().size
    height = targetTotalHeight / plotDF['ycoords'].unique().size
    aspect = width / height

    catPlotHeight, catPlotAspect = 1., 1.5
    catPlotWidth = catPlotHeight * catPlotAspect
    g = sns.catplot(
        row='ycoords', col='xcoords', y=barVarName, x='xDummy',
        data=plotDF, height=height, aspect=aspect,
        kind='bar', hue='emg', palette=emgPalette.to_dict(),
        hue_order=sorted(plotDF['emg'].unique()),
        )
    leg = g._legend
    titleOverrides = {'emg': 'EMG Recording\nSite'}

    if leg is not None:
        t = leg.get_title()
        tContent = t.get_text()
        if tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
        for tIdx, t in enumerate(leg.texts):
            tContent = t.get_text()
            if 'Right' in emgNL[tContent]:
                leg.get_patches()[tIdx].set_hatch('/' * 5)
            if tContent in emgNL.index:
                t.set_text(emgNL[tContent])
            elif tContent in titleOverrides:
                t.set_text(titleOverrides[tContent])
    g.set_titles('')
    g.set_xlabels('')
    g.set_ylabels('')
    g.set_xticklabels('')
    x0, y0, x1, y1 = g._tight_layout_rect
    figYLabel = g.fig.supylabel(
        'Spearman\'s R',
        x=x0, y=(y1 - y0)/2,
        ha='left', va='center')
    g.resize_legend(adjust_subtitles=True)
    for (ro, co, hu), dataSubset in g.facet_data():
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset[barVarName].isna().all()))
        if len(plotProcFuns):
            for procFun in plotProcFuns:
                procFun(g, ro, co, hu, dataSubset)
    yLabelH = (figYLabel.get_fontsize() / g.fig.dpi) / g.fig.get_size_inches()[0]
    g._tight_layout_rect[0] += yLabelH  # add room for figYLabel
    g.fig.subplots_adjust(
        left=g._tight_layout_rect[0])
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
            'emgToLfpCorrelation'))
    plt.savefig(
        pdfPath,
        bbox_inches='tight',
        # pad_inches=0.2,
        bbox_extra_artists=[leg, figYLabel])
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
########################################################################
## plot lfp rc
########################################################################
if True:
    plotEcapRC = ecapRauc.copy()
    for annotName in mapAnnotNames:
        lookupSource = mapDF.loc[mapAMask, [annotName, 'topoName']].set_index('topoName')[annotName]
        plotEcapRC.loc[:, 'feature_' + annotName] = plotEcapRC['feature'].map(lookupSource)
        plotEcapRC.loc[:, 'electrode_' + annotName] = plotEcapRC['electrode'].map(lookupSource)
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
        maskFromExample = (
            (dataSubset['feature'] == maxLFPName) &
            (dataSubset[amplitudeFieldName] == exLfpSingleTrial.name[4]) &
            (dataSubset['electrode'] == exLfpSingleTrial.name[3])
        )
        dataFromExample = dataSubset.loc[maskFromExample, :]
        if not dataFromExample.empty:
            thisColor = xPalette[float(dataFromExample['electrode_xcoords'].unique())]
            g.axes[ro, co].text(
                exLfpSingleTrial.name[4],
                dataFromExample[g._y_var].mean(),
                '$\searrow$', ha='right', va='baseline',
                # '$\Delta$', ha='center', va='center',
                color=thisColor
            )
            g.axes[ro, co].set_xlim([dataSubset[g._x_var].min(), dataSubset[g._x_var].max()])
            for side in g.axes[ro, co].spines.keys():  # 'top', 'bottom', 'left', 'right'
                g.axes[ro, co].spines[side].set_visible(True)
                g.axes[ro, co].spines[side].set_color(thisColor)
            g.axes[ro, co].axWasChanged = True
        if not hasattr(g.axes[ro, co], 'axWasChanged'):
            g.axes[ro, co].axWasChanged = True
            for side in g.axes[ro, co].spines.keys():  # 'top', 'bottom', 'left', 'right'
                g.axes[ro, co].spines[side].set_visible(False)
            if emptySubset:
                g.axes[ro, co].set_xticks([])
                g.axes[ro, co].set_facecolor('w')
            else:
                g.axes[ro, co].set_xlim([dataSubset[g._x_var].min(), dataSubset[g._x_var].max()])
        if g.axes[ro, co].get_subplotspec().is_first_row() and g.axes[ro, co].get_subplotspec().is_first_col():
            fig_width, fig_height = g.fig.get_size_inches()
            g.axes[ro, co].text(
                styleOpts['panel_heading.pad'], fig_height - styleOpts['panel_heading.pad'], 'B.', fontweight='bold',
                ha='left', va='top',
                fontsize=8, transform=g.fig.dpi_scale_trans, zorder=10)
        return
    plotProcFuns = [axModFun]
    '''plotProcFuns.append(asp.genTicksToScale(
        lineOpts={'lw': 1}, shared=True,
        xUnitFactor=1, xUnits='uA',
        yUnitFactor=1, yUnits='', limFracX=0.2, limFracY=.2,
        dropNaNCol=whichRaucLFP, scaleBarPosition=[presentAmplitudes[2], 0.3],
        # dX=0., yTicks=[0]
    ))'''
    ########
    # array is 5 rows by 6 columns
    # for a square figure, make width = l/6, height = l/5
    targetTotalWidth = 4.2 # inches
    targetTotalHeight = 3 # inches
    width = targetTotalWidth / plotEcapRC['feature_xcoords'].unique().size
    height = targetTotalHeight / plotEcapRC['feature_ycoords'].unique().size
    aspect = width / height
    g = sns.relplot(
        col='feature_xcoords',
        row='feature_ycoords',
        x=amplitudeFieldName,
        y=whichRaucLFP,
        hue='electrode_xcoords', palette=xPalette.to_dict(),
        # hue='electrode_ycoords', palette=yUnshiftedPalette.to_dict(),
        kind='line', data=plotEcapRC,
        height=height, aspect=aspect, errorbar='se', estimator='mean',
        facet_kws=dict(sharey=True, sharex=True),
        lw=.25,
        )
    leg = g._legend
    titleOverrides = {
        'electrode_xcoords': 'Stimulation location\nw.r.t. midline\n(mm)'
        }
    entryOverrides = mapDF.loc[:, ['xcoords', 'electrodeSide', 'whichArray']].drop_duplicates().set_index('xcoords')
    if leg is not None:
        t = leg.get_title()
        tContent = t.get_text()
        if tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
        for l in leg.get_lines():
            # l.set_lw(2 * l.get_lw())
            l.set_lw(styleOpts['legend.lw'])
        for t in leg.texts:
            tContent = t.get_text()
            if tContent in titleOverrides:
                t.set_text(titleOverrides[tContent])
            # elif tContent.replace('.', '', 1).isdigit():
            # e.g. is numeric
            else:
                try:
                    tNumeric = float(tContent) - electrodeMaxXDistance / 2
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
    g.set_xticklabels('')
    g.set_yticklabels('')
    #
    '''figYLabel = g.fig.supylabel(
        'ECAP rAUC (a.u.)', x=0.01, y=0.5,
        ha='left', va='center'
        )
    figXLabel = g.fig.supxlabel(
        'Stimulation amplitude (uA)', x=0.5, y=0.01,
        ha='center', va='bottom'
        )
    g._tight_layout_rect[0] += 1e-2 # add room for figXLabel
    g.fig.subplots_adjust(left=g._tight_layout_rect[0])
    '''
    #
    g.resize_legend(adjust_subtitles=True)
    for (ro, co, hu), dataSubset in g.facet_data():
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset[whichRaucLFP].isna().all()))
        if len(plotProcFuns):
            for procFun in plotProcFuns:
                procFun(g, ro, co, hu, dataSubset)
    print('Applying tight layout on lfp rc')
    textSizeX = (snsRCParams['font.size'] / g.fig.dpi) / g.fig.get_size_inches()[0]
    g._tight_layout_rect[0] += textSizeX
    textSizeY = (snsRCParams['font.size'] / g.fig.dpi) / g.fig.get_size_inches()[1]
    g._tight_layout_rect[3] -= textSizeY
    g.fig.subplots_adjust(
        left=g._tight_layout_rect[0],
        bottom=g._tight_layout_rect[1],
        right=g._tight_layout_rect[2],
        top=g._tight_layout_rect[3])
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
            'lfpRAUC'))
    plt.savefig(
        pdfPath,
        bbox_inches='tight',
        bbox_extra_artists=[
            leg,
            # figYLabel, figXLabel
            ]
        )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
########################################################################
## plot lfp rc closeup
########################################################################
if True:
    maskExampleFeature = (plotEcapRC['feature'] == maxLFPName)
    plotEcapRCExample = plotEcapRC.loc[maskExampleFeature, :].copy()
    def axModFun(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset[whichRaucLFP].isna().all()))
        maskFromExample = (
            (dataSubset['feature'] == maxLFPName) &
            (dataSubset[amplitudeFieldName] == exLfpSingleTrial.name[4]) &
            (dataSubset['electrode'] == exLfpSingleTrial.name[3])
            )
        dataFromExample = dataSubset.loc[maskFromExample, :]
        if not dataFromExample.empty:
            thisColor = xPalette[float(dataFromExample['electrode_xcoords'].unique())]
            g.axes[ro, co].text(
                exLfpSingleTrial.name[4],
                dataFromExample[g._y_var].mean(),
                '$\searrow$', ha='right', va='baseline',
                # '$\Delta$', ha='center', va='center',
                color=thisColor
            )
            if g.axes[ro, co].get_subplotspec().is_first_row() and g.axes[ro, co].get_subplotspec().is_first_col():
                fig_width, fig_height = g.fig.get_size_inches()
                g.axes[ro, co].text(
                    styleOpts['panel_heading.pad'], fig_height - styleOpts['panel_heading.pad'], 'C.', fontweight='bold',
                    ha='left', va='top',
                    fontsize=8, transform=g.fig.dpi_scale_trans, zorder=10)
            g.axes[ro, co].set_xlim([dataSubset[g._x_var].min(), dataSubset[g._x_var].max()])
            for side in g.axes[ro, co].spines.keys():  # 'top', 'bottom', 'left', 'right'
                g.axes[ro, co].spines[side].set_lw(axLW * 2)
                g.axes[ro, co].spines[side].set_visible(True)
                g.axes[ro, co].spines[side].set_color(lfpXPalette[exElectrode])
            g.axes[ro, co].tick_params(length=axLW * 10, width=axLW, which='major', direction='in', reset=True)
            g.axes[ro, co].tick_params(length=axLW * 5, width=axLW, which='minor', direction='in', reset=True)
            g.axes[ro, co].xaxis.set_major_locator(mpltk.MaxNLocator(prune='both', nbins=4))
            g.axes[ro, co].xaxis.set_minor_locator(mpltk.AutoMinorLocator())
            g.axes[ro, co].yaxis.set_major_locator(mpltk.MaxNLocator(prune='both', nbins=4))
            g.axes[ro, co].yaxis.set_minor_locator(mpltk.AutoMinorLocator())
            g.axes[ro, co].axWasChanged = True
        if not hasattr(g.axes[ro, co], 'axWasChanged'):
            g.axes[ro, co].axWasChanged = True
            for side in g.axes[ro, co].spines.keys():  # 'top', 'bottom', 'left', 'right'
                g.axes[ro, co].spines[side].set_visible(False)
            if emptySubset:
                g.axes[ro, co].set_xticks([])
                g.axes[ro, co].set_facecolor('w')
            else:
                g.axes[ro, co].set_xlim([dataSubset[g._x_var].min(), dataSubset[g._x_var].max()])
        return
    plotProcFuns = [axModFun]
    ########
    height = 1.5
    width = 1.5
    aspect = width / height
    g = sns.relplot(
        col='feature_xcoords',
        row='feature_ycoords',
        x=amplitudeFieldName,
        y=whichRaucLFP,
        hue='electrode_xcoords', palette=xPalette.to_dict(),
        # hue='electrode_ycoords', palette=yUnshiftedPalette.to_dict(),
        kind='line', data=plotEcapRCExample,
        height=height, aspect=aspect, errorbar='se', estimator='mean',
        facet_kws=dict(sharey=True, sharex=True),
        lw=.5,
        )
    leg = g._legend
    titleOverrides = {
        'electrode_xcoords': 'Stimulation location\nw.r.t. midline\n(mm)'
        }
    entryOverrides = mapDF.loc[:, ['xcoords', 'electrodeSide', 'whichArray']].drop_duplicates().set_index('xcoords')

    if leg is not None:
        t = leg.get_title()
        tContent = t.get_text()
        if tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
        for l in leg.get_lines():
            # l.set_lw(2 * l.get_lw())
            l.set_lw(styleOpts['legend.lw'])
        for t in leg.texts:
            tContent = t.get_text()
            if tContent in titleOverrides:
                t.set_text(titleOverrides[tContent])
            # elif tContent.replace('.', '', 1).isdigit():
            # e.g. is numeric
            else:
                try:
                    tNumeric = float(tContent) - electrodeMaxXDistance / 2
                    t.set_text('{:.2f}'.format(tNumeric / 10))
                    '''if tNumeric in entryOverrides.index:
                        t.set_text('{} {}'.format(
                            entryOverrides.loc[float(tContent), 'whichArray'],
                            entryOverrides.loc[float(tContent), 'electrodeSide']))'''
                except Exception:
                    pass
    g.set_titles('')
    g.set_xlabels('Stimulation amplitude (uA)')
    g.set_ylabels('ECAP rAUC (a.u.)')
    '''
    x0, y0, x1, y1 = g._tight_layout_rect
    figYLabel = g.fig.supylabel(
        'ECAP rAUC (a.u.)', x=x0, y=(y1 - y0) / 2,
        ha='left', va='center'
        )
    figXLabel = g.fig.supxlabel(
        'Stimulation amplitude (uA)', x=(x1 - x0) / 2, y=y0,
        ha='center', va='bottom'
        )'''
    g.resize_legend(adjust_subtitles=True)
    for (ro, co, hu), dataSubset in g.facet_data():
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset[whichRaucLFP].isna().all()))
        if len(plotProcFuns):
            for procFun in plotProcFuns:
                procFun(g, ro, co, hu, dataSubset)
    '''xLabelH = (figXLabel.get_fontsize() / g.fig.dpi) / g.fig.get_size_inches()[1]
    g._tight_layout_rect[1] += xLabelH # add room for figYLabel
    yLabelH = (figYLabel.get_fontsize() / g.fig.dpi) / g.fig.get_size_inches()[0]
    g._tight_layout_rect[0] += yLabelH # add room for figYLabel
    g.fig.subplots_adjust(
        left=g._tight_layout_rect[0],
        bottom=g._tight_layout_rect[1])'''
    textSizeX = (snsRCParams['font.size'] / g.fig.dpi) / g.fig.get_size_inches()[0]
    g._tight_layout_rect[0] += textSizeX
    textSizeY = (snsRCParams['font.size'] / g.fig.dpi) / g.fig.get_size_inches()[1]
    g._tight_layout_rect[3] -= textSizeY
    g.fig.subplots_adjust(
        left=g._tight_layout_rect[0],
        bottom=g._tight_layout_rect[1],
        right=g._tight_layout_rect[2],
        top=g._tight_layout_rect[3])
    print('Applying tight layout on lfp rc closeup')
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
            'lfpRAUC_zoom'))
    plt.savefig(
        pdfPath,
        bbox_inches='tight',
        bbox_extra_artists=[
            leg,
            # figYLabel, figXLabel
        ]
        ) #
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
########################################################################
## plot emg rc
########################################################################
if True:
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
        maskFromExample = (
            (dataSubset['feature'] == maxEMGName) &
            (dataSubset[amplitudeFieldName] == exEmgSingleTrial.name[4]) &
            (dataSubset['electrode'] == exEmgSingleTrial.name[3])
        )
        if g.axes[ro, co].get_subplotspec().is_first_row() and g.axes[ro, co].get_subplotspec().is_first_col():
            fig_width, fig_height = g.fig.get_size_inches()
            g.axes[ro, co].text(
                styleOpts['panel_heading.pad'], fig_height - styleOpts['panel_heading.pad'], 'E.', fontweight='bold',
                ha='left', va='top',
                fontsize=8, transform=g.fig.dpi_scale_trans, zorder=10)
        dataFromExample = dataSubset.loc[maskFromExample, :]
        if not dataFromExample.empty:
            g.axes[ro, co].text(
                exEmgSingleTrial.name[4],
                dataFromExample[g._y_var].mean(),
                '$\searrow$', ha='right', va='baseline',
                color=emgPaletteDarker[maxEMGName],
            )
        if not hasattr(g.axes[ro, co], 'axWasChanged'):
            g.axes[ro, co].axWasChanged = True
            if emptySubset:
                g.axes[ro, co].set_xticks([])
                g.axes[ro, co].set_facecolor('w')
            else:
                elecPos = dataSubset[colName].iloc[0] - electrodeMaxXDistance / 2
                thisElecName = dataSubset['electrode'].unique()[0]
                titleText = '{:0.2f} mm'.format(elecPos / 10)
                g.axes[ro, co].set_title(titleText)
                axLW = snsRCParams['axes.linewidth']
                g.axes[ro, co].set_xlim([dataSubset[g._x_var].min(), dataSubset[g._x_var].max()])
                g.axes[ro, co].tick_params(length=axLW * 10, width=axLW, which='major', direction='in', reset=True)
                g.axes[ro, co].tick_params(length=axLW * 5, width=axLW, which='minor', direction='in', reset=True)
                g.axes[ro, co].xaxis.set_major_locator(mpltk.MaxNLocator(prune='both', nbins=3))
                g.axes[ro, co].xaxis.set_minor_locator(mpltk.AutoMinorLocator())
                g.axes[ro, co].yaxis.set_major_locator(mpltk.MaxNLocator(prune='both', nbins=3))
                g.axes[ro, co].yaxis.set_minor_locator(mpltk.AutoMinorLocator())
                for side in g.axes[ro, co].spines.keys():  # 'top', 'bottom', 'left', 'right'
                    g.axes[ro, co].spines[side].set_visible(True)
                    g.axes[ro, co].spines[side].set_lw(axLW * 4)
                    g.axes[ro, co].spines[side].set_color(lfpXPalette[thisElecName])
        return

    plotProcFuns = []
    plotProcFuns.append(axModFun)
    '''plotProcFuns.append(
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=True,
            xUnitFactor=1, xUnits='uA',
            yUnitFactor=1, yUnits='', limFracX=0.2, limFracY=.2,
            dropNaNCol=whichRaucEMG, scaleBarPosition=[presentAmplitudes[2], 0.6],
            # dX=0., yTicks=[0]
        )
    )'''
    colOrder = sorted(np.unique(plotEmgRC[colName]))
    hueName = 'featureName'
    colWrap = min(3, len(colOrder))

    width = 2.2
    height = 1.8
    aspect = width / height
    width = height * aspect
    g = sns.relplot(
        data=plotEmgRC,
        col=colName,
        col_order=colOrder,
        x=amplitudeFieldName,
        y=whichRaucEMG,
        style='EMGSide', style_order=['Right', 'Left'],
        hue=hueName, hue_order=hueOrderEMG, palette=emgPalette.to_dict(),
        kind='line',
        height=height, aspect=aspect, errorbar='se', estimator='mean',
        facet_kws=dict(sharey=True, sharex=False, legend_out=True),
        lw=.5,
        )
    leg = g._legend
    titleOverrides = {
        'featureName': 'EMG Recording\nSite',
        'EMGSide': 'Side of Body'
        }
    g.set_xlabels('')
    g.set_ylabels('')
    # g.set_yticklabels('')
    if leg is not None:
        t = leg.get_title()
        tContent = t.get_text()
        if tContent in titleOverrides:
            t.set_text(titleOverrides[tContent])
        for t in leg.texts:
            tContent = t.get_text()
            if tContent in titleOverrides:
                t.set_text(titleOverrides[tContent])
            elif tContent in emgNL.index:
                t.set_text('{}'.format(emgNL[tContent]))
        for l in leg.get_lines():
            # l.set_lw(2 * l.get_lw())
            l.set_lw(styleOpts['legend.lw'])
    g.resize_legend(adjust_subtitles=True)
    for (ro, co, hu), dataSubset in g.facet_data():
        emptySubset = (
                (dataSubset.empty) or
                (dataSubset[whichRaucEMG].isna().all()))
        if len(plotProcFuns):
            for procFun in plotProcFuns:
                procFun(g, ro, co, hu, dataSubset)
    x0, y0, x1, y1 = g._tight_layout_rect
    figYLabel = g.fig.supylabel(
        'EMG rAUC (a.u.)', x=x0, y=(y1 - y0) / 2,
        ha='left', va='center'
        )
    figXLabel = g.fig.supxlabel(
        'Stimulation amplitude (uA)', x=(x1 - x0) / 2, y=y0,
        ha='center', va='bottom'
        )
    figTitle = g.fig.suptitle(
        'Stimulation location (mediolateral, w.r.t. midline)',
        x=(x1 - x0) / 2, y=y1,
        ha='center', va='top'
        )
    xLabelH = (figXLabel.get_fontsize() / g.fig.dpi) / g.fig.get_size_inches()[1]
    g._tight_layout_rect[1] += xLabelH # add room for figYLabel
    yLabelH = (figYLabel.get_fontsize() / g.fig.dpi) / g.fig.get_size_inches()[0]
    g._tight_layout_rect[0] += yLabelH # add room for figYLabel
    supTitleH = (figTitle.get_fontsize() / g.fig.dpi) / g.fig.get_size_inches()[1]
    g._tight_layout_rect[3] -= supTitleH # add room for figTitle
    g.fig.subplots_adjust(
        left=g._tight_layout_rect[0],
        bottom=g._tight_layout_rect[1],
        right=g._tight_layout_rect[2],
        top=g._tight_layout_rect[3])
    print('Applying tight layout on emg rc')
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdfPath = os.path.join(
        figureOutputFolder,
        prefix + '_{}_{}_{}.pdf'.format(
            arguments['emgBlockSuffix'], arguments['lfpBlockSuffix'],
            'emgRAUC'))
    plt.savefig(
        pdfPath,
        bbox_inches='tight',
        bbox_extra_artists=[leg, figTitle, figYLabel, figXLabel]
        )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()