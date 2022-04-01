"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --expList=expList                      which experimental day to analyze
    --selectionList=selectionList          which signal type to analyze
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
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierMask                    delete outlier trials? [default: False]
    --showFigures                          show plots interactively? [default: False]
    --plotThePieces                        show plots interactively? [default: False]
    --plotTheAverage                       show plots interactively? [default: False]
    --plotTopoEffectSize                  show plots interactively? [default: False]
"""
import logging, sys
logging.captureWarnings(True)
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pdb, traceback
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.preproc.ns5 as ns5
import os
from itertools import product
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
import pingouin as pg
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime as dt
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .25,
        'lines.markersize': 2.4,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .125,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        'axes.facecolor': 'w',
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
    'figure.titlesize': 7,
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 2e-1, # units of font size
    'panel_heading.pad': 0.
    }
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'maskOutlierBlocks': False, 'inputBlockSuffix': 'lfp_CAR_spectral_mahal_ledoit',
        'window': 'XL', 'blockIdx': '2', 'lazy': False, 'verbose': False, 'plotThePieces': True,
        'exp': 'exp202101271100', 'alignFolderName': 'motion', 'analysisName': 'hiRes', 'unitQuery': 'mahal',
        'alignQuery': 'starting', 'inputBlockPrefix': 'Block', 'invertOutlierMask': False, 'processAll': True,
        'showFigures': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''

basePalette = pd.Series(sns.color_palette('Paired'))
allAmpPalette = pd.Series(
    basePalette.apply(sns.utils.alter_color, h=-0.05).to_numpy()[:4],
    index=[
        'trialAmplitude', 'trialAmplitude_md',
        'trialAmplitude:trialRateInHz', 'trialAmplitude:trialRateInHz_md'])
allRelPalette = pd.Series(
    basePalette.apply(sns.utils.alter_color, h=0.05).to_numpy()[:6],
    index=['50.0', '50.0_md', '100.0', '100.0_md', '0.0', '0.0_md'])
# allRelPalette.loc['0.0'] = sns.utils.alter_color(allRelPalette.loc['50.0'], l=0.5)
# allRelPalette.loc['0.0_md'] = sns.utils.alter_color(allRelPalette.loc['50.0_md'], l=0.5)

'''
if True:
    fig, ax = plt.subplots(3, 1)
    basePalette = pd.Series(sns.color_palette('Paired'))
    sns.palplot(basePalette.apply(sns.utils.alter_color, h=-0.05).to_numpy()[:6], ax=ax[0])
    sns.palplot(basePalette.to_numpy()[:6], ax=ax[1])
    sns.palplot(basePalette.apply(sns.utils.alter_color, h=0.05).to_numpy()[:6], ax=ax[2])
    plt.show()'''
blockBaseName = arguments['inputBlockPrefix']
listOfExpNames = [x.strip() for x in arguments['expList'].split(',')]
listOfSelectionNames = [x.strip() for x in arguments['selectionList'].split(',')]
recCurveList = []
ampStatsDict = {}
relativeStatsDict = {}
relativeStatsNoStimDict = {}
ampStatsPerFBDict = {}
relativeStatsPerFBDict = {}
compoundAnnLookupList = []
featureInfoList = []
for expName in listOfExpNames:
    arguments['exp'] = expName
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName'])
    #
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName'])
    calcSubFolder = os.path.join(analysisSubFolder, 'dataframes')
    outlierTrials = ash.processOutlierTrials(
        scratchFolder, blockBaseName, **arguments)
    for inputBlockSuffix in listOfSelectionNames:
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc.h5'.format(
                inputBlockSuffix, arguments['window']))
        signalIsMahalDist = 'mahal' in inputBlockSuffix
        print('loading {}'.format(resultPath))
        if not os.path.exists(resultPath):
            print('WARNING: path does not exist\n{}'.format(resultPath))
            continue
        try:
            rawRecCurve = pd.read_hdf(resultPath, 'raw')
            thisRecCurveFeatureInfo = rawRecCurve.columns.to_frame().reset_index(drop=True)
            rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
            thisRecCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
            scaledRaucDF = pd.read_hdf(resultPath, 'boxcox')
            scaledRaucDF.columns = scaledRaucDF.columns.get_level_values('feature')
            thisRecCurve.loc[:, 'scaledRAUC'] = scaledRaucDF.stack().to_numpy()
            for iN in ['isOutlierTrial', 'outlierDeviation']:
                if iN in thisRecCurve.index.names:
                    thisRecCurve.index = thisRecCurve.index.droplevel(iN)
            # relativeRaucDF = pd.read_hdf(resultPath, 'relative')
            # relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
            # thisRecCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()
            # clippedRaucDF = pd.read_hdf(resultPath, 'clipped')
            # clippedRaucDF.columns = clippedRaucDF.columns.get_level_values('feature')
            # thisRecCurve.loc[:, 'clippedRAUC'] = clippedRaucDF.stack().to_numpy()
            #
            thisRecCurve.loc[:, 'freqBandName'] = thisRecCurve.index.get_level_values('feature').map(thisRecCurveFeatureInfo[['feature', 'freqBandName']].set_index('feature')['freqBandName'])
            thisRecCurve.set_index('freqBandName', append=True, inplace=True)
            ################################################################################################################
            ################################################################################################################
            recCurveTrialInfo = thisRecCurve.index.to_frame().reset_index(drop=True)
            recCurveTrialInfo.loc[:, 'kinematicCondition'] = recCurveTrialInfo['kinematicCondition'].apply(lambda x: x.replace('CCW_', 'CW_'))
            thisRecCurve.index = pd.MultiIndex.from_frame(recCurveTrialInfo)
            hackMask1 = recCurveTrialInfo['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']).any()
            if hackMask1:
                dropMask = recCurveTrialInfo['electrode'].isin(['-E09+E16', '-E00+E16']).to_numpy()
                thisRecCurve = thisRecCurve.loc[~dropMask, :]
            hackMask2 = recCurveTrialInfo['expName'].isin(['exp201902031100']).any()
            if hackMask2:
                dropMask = recCurveTrialInfo['electrode'].isin(['-E00+E16']).to_numpy()
                thisRecCurve = thisRecCurve.loc[~dropMask, :]
            ################################################################################################################
            ################################################################################################################
            thisRecCurve.loc[:, 'isMahalDist'] = signalIsMahalDist
            thisRecCurveFeatureInfo.loc[:, 'isMahalDist'] = signalIsMahalDist
            recCurveList.append(thisRecCurve)
            #####
            try:
                thisRecCurveFeatureInfo.loc[:, 'xIdx'], thisRecCurveFeatureInfo.loc[:, 'yIdx'] = ssplt.coordsToIndices(
                    thisRecCurveFeatureInfo['xCoords'], thisRecCurveFeatureInfo['yCoords'])
            except Exception:
                thisRecCurveFeatureInfo.loc[:, 'xIdx'] = 0.
                thisRecCurveFeatureInfo.loc[:, 'yIdx'] = 0.
                thisRecCurveFeatureInfo.loc[:, 'xCoords'] = 0.
                thisRecCurveFeatureInfo.loc[:, 'yCoords'] = 0.
            #
            ampStatsDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'amplitudeStats')
            ampStatsDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            relativeStatsDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'relativeStatsDF')
            relativeStatsDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            relativeStatsNoStimDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'noStimTTest')
            relativeStatsNoStimDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            ampStatsPerFBDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'amplitudeStatsPerFreqBand')
            ampStatsPerFBDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            relativeStatsPerFBDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'relativeStatsDFPerFreqBand')
            relativeStatsPerFBDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            compoundAnnLookupList.append(pd.read_hdf(resultPath, 'compoundAnnLookup'))
            featureInfoList.append(thisRecCurveFeatureInfo)
            ################################################################################################################
            ################################################################################################################
            lOfStatsDFs = [
                ampStatsDict[(expName, inputBlockSuffix)],
                relativeStatsDict[(expName, inputBlockSuffix)],
                # relativeStatsNoStimDict[(expName, inputBlockSuffix)],
                ampStatsPerFBDict[(expName, inputBlockSuffix)],
                relativeStatsPerFBDict[(expName, inputBlockSuffix)]
                ]
            for dfidx, df in enumerate(lOfStatsDFs):
                dfTI = df.index.to_frame().reset_index(drop=True)
                dfTI.loc[:, 'kinematicCondition'] = dfTI['kinematicCondition'].apply(lambda x: x.replace('CCW_', 'CW_'))
                lOfStatsDFs[dfidx].index = pd.MultiIndex.from_frame(dfTI)
            ################################################################################################################
            ################################################################################################################
        except Exception:
            traceback.print_exc()
compoundAnnLookupDF = pd.concat(compoundAnnLookupList).drop_duplicates()
recCurveFeatureInfo = pd.concat(featureInfoList).drop_duplicates()
#
recCurve = pd.concat(recCurveList)
del recCurveList
ampStatsDF = pd.concat(ampStatsDict, names=['expName', 'selectionName'])
ampStatsDF.drop(labels=['Intercept'], axis='index', level='names', inplace=True)
del ampStatsDict
relativeStatsDF = pd.concat(relativeStatsDict, names=['expName', 'selectionName'])
del relativeStatsDict
relativeStatsNoStimDF = pd.concat(relativeStatsNoStimDict, names=['expName', 'selectionName'])
del relativeStatsNoStimDict
relativeStatsPerFBDF = pd.concat(relativeStatsPerFBDict, names=['expName', 'selectionName'])
del relativeStatsPerFBDict
ampStatsPerFBDF = pd.concat(ampStatsPerFBDict, names=['expName', 'selectionName'])
del ampStatsPerFBDict

########################################################################################################################
########################################################################################################################
dfTI = ampStatsDF.index.to_frame().reset_index(drop=True)
hackMask3 = (
    dfTI['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']) &
    dfTI['electrode'].isin(['-E00+E16', '-E09+E16',])
    ).to_numpy()
if hackMask3.any():
    ampStatsDF = ampStatsDF.loc[~hackMask3, :]
dfTI = ampStatsDF.index.to_frame().reset_index(drop=True)
hackMask4 = (
    dfTI['expName'].isin(['exp201902031100']) &
    dfTI['electrode'].isin(['-E00+E16'])
    ).to_numpy()
if hackMask4.any():
    ampStatsDF = ampStatsDF.loc[~hackMask4, :]
dfTI = relativeStatsDF.index.to_frame().reset_index(drop=True)
hackMask5 = (
    dfTI['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']) &
    dfTI['stimCondition'].isin(['-E00+E16_50.0', '-E09+E16_50.0', '-E00+E16_100.0', '-E09+E16_100.0',])
    ).to_numpy()
if hackMask5.any():
    relativeStatsDF = relativeStatsDF.loc[~hackMask5, :]
dfTI = relativeStatsDF.index.to_frame().reset_index(drop=True)
hackMask6 = (
    dfTI['expName'].isin(['exp201902031100']) &
    dfTI['stimCondition'].isin(['-E00+E16_50.0', '-E00+E16_100.0',])
    ).to_numpy()
if hackMask6.any():
    relativeStatsDF = relativeStatsDF.loc[~hackMask6, :]
del dfTI
########################################################################################################################
########################################################################################################################

correctMultiCompHere = True
confidence_alpha = .05
if correctMultiCompHere:
    pvalsDict = {
        'amp': ampStatsDF.loc[:, ['pval']].reset_index(drop=True),
        'relative': relativeStatsDF.loc[:, ['pval']].reset_index(drop=True),
        'relative_no_stim': relativeStatsNoStimDF.loc[:, ['pval']].reset_index(drop=True),
        }
    pvalsCatDF = pd.concat(pvalsDict, names=['origin', 'originIndex'])
    reject, pval = pg.multicomp(pvalsCatDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    pvalsCatDF.loc[:, 'pval'] = pval
    pvalsCatDF.loc[:, 'reject'] = reject
    for cN in ['pval', 'reject']:
        ampStatsDF.loc[:, cN] = pvalsCatDF.xs('amp', level='origin')[cN].to_numpy()
        relativeStatsDF.loc[:, cN] = pvalsCatDF.xs('relative', level='origin')[cN].to_numpy()
        relativeStatsNoStimDF.loc[:, cN] = pvalsCatDF.xs('relative_no_stim', level='origin')[cN].to_numpy()
    reject, pval = pg.multicomp(ampStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    ampStatsPerFBDF.loc[:, 'pval'] = pval
    ampStatsPerFBDF.loc[:, 'reject'] = reject
    reject, pval = pg.multicomp(relativeStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    relativeStatsPerFBDF.loc[:, 'pval'] = pval
    relativeStatsPerFBDF.loc[:, 'reject'] = reject

#
countAmpStatsSignifDF = ampStatsDF.groupby(['expName', 'electrode', 'kinematicCondition']).sum()['reject']
print(countAmpStatsSignifDF)
countRelativeStatsSignifDF = relativeStatsDF.groupby(['expName', 'electrode', 'kinematicCondition']).sum()['reject']
print(countAmpStatsSignifDF)
#
freqBandOrderExtended = ['broadband'] + freqBandOrder
whichRAUC = 'rawRAUC'
# whichRAUC = 'normalizedRAUC'
# whichRAUC = 'scaledRAUC'
#
figureOutputFolder = os.path.join(
    remoteBasePath, 'figures', 'lfp_recruitment_across_exp')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)

countSummaryHtml = (
        countAmpStatsSignifDF.to_frame(name='significant_stim_effect').to_html() +
        countRelativeStatsSignifDF.to_frame('significant_stim_vs_no_stim').to_html())
pdfNameSuffix = 'RAUC_all'
countSummaryPath = os.path.join(
    figureOutputFolder, '{}_{}.html'.format(subjectName, pdfNameSuffix))
with open(countSummaryPath, 'w') as _f:
    _f.write(countSummaryHtml)

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(subjectName, pdfNameSuffix))
# lOfKinematicsToPlot = ['NA_NA']
lOfKinematicsToPlot = ['NA_NA', 'CW_outbound', 'CW_return']
# lOfKinematicsToPlot = ['CW_outbound', 'CCW_outbound']
# lOfKinematicsToPlot = ['CW_return', 'CCW_return']
# lOfKinematicsToPlot = ['NA_NA', 'CW_return']
# lOfKinematicsToPlot = ['NA_NA', 'CW_outbound', 'CW_return']
# lOfKinematicsToPlot = [
#     'NA_NA',
#     'CW_outbound', 'CCW_outbound',
#     'CW_return', 'CCW_return',
#     ]
plotRC = recCurve.reset_index()
spinalMapDF = spinalElectrodeMaps[subjectName].sort_values(['xCoords', 'yCoords'])
spinalElecCategoricalDtype = pd.CategoricalDtype(spinalMapDF.index.to_list(), ordered=True)
spinalMapDF.index = pd.Index(spinalMapDF.index, dtype=spinalElecCategoricalDtype)
plotRC.loc[:, 'electrode'] = plotRC['electrode'].astype(spinalElecCategoricalDtype)
keepCols = [
    'segment', 'originalIndex', 't',
    'feature', 'freqBandName', 'lag',
    'stimCondition', 'kinematicCondition'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)
######

for cN in relativeStatsDF.columns:
    if cN not in relativeStatsNoStimDF.columns:
        relativeStatsNoStimDF.loc[:, cN] = 0
relativeStatsDF.loc[:, 'T_abs'] = relativeStatsDF['T'].abs()
relativeStatsNoStimDF.loc[:, 'T_abs'] = relativeStatsNoStimDF['T'].abs()

nFeats = plotRC['feature'].unique().shape[0]
nFeatsToPlot = max(min(3, int(np.floor(nFeats/2))), 1)
keepTopIdx = (
    [i for i in range(nFeatsToPlot)] +
    [i for i in range(-1 * nFeatsToPlot, 0)]
    )
keepColsForPlot = []
rankMask = relativeStatsDF.index.get_level_values('stimCondition') != 'NA_0.0'
for freqBandName, relativeStatsThisFB in relativeStatsDF.loc[rankMask, :].groupby('freqBandName'):
    statsRankingDF = relativeStatsThisFB.groupby('feature').mean().sort_values('T', ascending=False)
    nFeatsToPlot = max(min(2, int(np.floor(statsRankingDF.shape[0]/2))), 1)
    keepTopIdx = (
        [i for i in range(nFeatsToPlot)] +
        [i for i in range(-1 * nFeatsToPlot, 0)]
        )
    keepColsForPlot += statsRankingDF.index[keepTopIdx].to_list()
#
print('Plotting select features:')
print(', '.join(["'{}#0'".format(fN) for fN in keepColsForPlot]))
plotRCPieces = plotRC.loc[plotRC['feature'].isin(keepColsForPlot), :].copy()
######
refGroupPieces = plotRCPieces.loc[plotRCPieces['stimCondition'] == 'NA_0.0', :]
testGroupPieces = plotRCPieces.loc[plotRCPieces['stimCondition'] != 'NA_0.0', :]
#
refGroup = plotRC.loc[plotRC['stimCondition'] == 'NA_0.0', :]
testGroup = plotRC.loc[plotRC['stimCondition'] != 'NA_0.0', :]

def genStatsAnnotator(ampDF, relDF, hN, hP):
    def statsAnnotator(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
        if not emptySubset:
            if not hasattr(g.axes[ro, co], 'starsAnnotated'):
                xLim = g.axes[ro, co].get_xlim()
                yLim = g.axes[ro, co].get_ylim()
                trans = transforms.blended_transform_factory(
                    g.axes[ro, co].transAxes, g.axes[ro, co].transData)
                # dx = (xLim[1] - xLim[0]) / 5
                # dy = (yLim[1] - yLim[0]) / 5
                for hn, group in dataSubset.groupby([hN]):
                    rn = group[g._row_var].unique()[0]
                    cn = group[g._col_var].unique()[0]
                    thisElectrode = compoundAnnLookupDF.loc[cn, 'electrode']
                    st = ampDF.xs(rn, level=g._row_var).xs(thisElectrode, level='electrode').xs(hn, level=hN).xs('trialAmplitude', level='names')
                    x = group[g._x_var].max()
                    y = group.groupby(g._x_var).mean().loc[x, g._y_var]
                    message = ''
                    if st['reject'].iloc[0]:
                        message += '*'
                    st = ampDF.xs(rn, level=g._row_var).xs(thisElectrode, level='electrode').xs(hn, level=hN).xs('trialRateInHz', level='names')
                    if st['reject'].iloc[0]:
                        message += '^'
                    rst = relDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hN)
                    # if rst['pval'].iloc[0] < 1 - confidence_alpha:
                    if rst['reject'].iloc[0]:
                        message += '+'
                    if len(message):
                        g.axes[ro, co].text(1., y, message, color=hP[hn], va='bottom', ha='left', transform=trans)
                if (ro == 0) and (co == 0):
                    g.axes[ro, co].text(
                        0.95, 0.95,
                        '\n'.join([
                            '+: highest amp. stim. vs baseline (pval < 1e-3)',
                            '*: amplitude vs auc (pval < 1e-3)',
                            '^: rate vs auc (pval < 1e-3)']),
                        va='top', ha='right', transform=g.axes[ro, co].transAxes)
                g.axes[ro, co].starsAnnotated = True
        return
    return statsAnnotator

def genNumSigAnnotator(pvDF, xOrder=None, hueVar=None, palette=None, fontOpts={}, width=0.8):
    def numSigAnnotator(g, ro, co, hu, dataSubset):
        if not hasattr(g.axes[ro, co], 'pvalsAnnotated'):
            trans = transforms.blended_transform_factory(
                g.axes[ro, co].transData, g.axes[ro, co].transAxes)
            hueOrder = palette.index.to_list()
            huePalette = palette.to_dict()
            offsets = np.linspace(0, width - width / len(hueOrder), len(hueOrder))
            offsets -= offsets.mean()
            # print(['offsets from {} to {}'.format(offsets[0], offsets[-1])])
            thisMask = pvDF.notna().all(axis='columns')
            if g._col_var is not None:
                thisMask = thisMask & (pvDF[g._col_var] == g.col_names[co])
            if g._row_var is not None:
                thisMask = thisMask & (pvDF[g._row_var] == g.row_names[ro])
            for xIdx, xLabel in enumerate(xOrder):
                xMask = (pvDF[g._x_var] == xLabel)
                for hIdx, hLabel in enumerate(hueOrder):
                    hMask = (pvDF[hueVar] == hLabel)
                    totalMask = (thisMask & xMask & hMask)
                    if totalMask.any():
                        thisEntry = pvDF.loc[totalMask, :]
                        try:
                            assert thisEntry.shape[0] == 1
                        except:
                            traceback.print_exc()
                        thisEntry = thisEntry.iloc[0, :]
                        if thisEntry['count'] == 0:
                            continue
                        message = '{}/{}'.format(thisEntry['pass'], thisEntry['count'])
                        x = xIdx + offsets[hIdx]
                        y = 0
                        g.axes[ro, co].text(
                            x, y, message,
                            transform=trans, color=huePalette[hLabel],
                            **fontOpts)
            g.axes[ro, co].pvalsAnnotated = True
        else:
            print('ro {} co {} g.axes[ro, co].pvalsAnnotated {}'.format(ro, co, g.axes[ro, co].pvalsAnnotated))
        return
    return numSigAnnotator

def genNumSigAnnotatorV2(
        pvDF, xOrder=None, hueVar=None,
        palette=None, fontOpts={}, width=0.8):
    def numSigAnnotator(g, theAx, rowVar, colVar):
        if not hasattr(theAx, 'pvalsAnnotated'):
            trans = transforms.blended_transform_factory(
                theAx.transData, theAx.transAxes)
            hueOrder = palette.index.to_list()
            huePalette = palette.to_dict()
            offsets = np.linspace(0, width - width / len(hueOrder), len(hueOrder))
            offsets -= offsets.mean()
            thisMask = pvDF.notna().all(axis='columns')
            if g._col_var is not None:
                thisMask = thisMask & (pvDF[g._col_var] == colVar)
            if g._row_var is not None:
                thisMask = thisMask & (pvDF[g._row_var] == rowVar)
            for xIdx, xLabel in enumerate(xOrder):
                xMask = (pvDF[g._x_var] == xLabel)
                for hIdx, hLabel in enumerate(hueOrder):
                    hMask = (pvDF[hueVar] == hLabel)
                    totalMask = (thisMask & xMask & hMask)
                    if totalMask.any():
                        thisEntry = pvDF.loc[totalMask, :]
                        assert thisEntry.shape[0] == 1
                        thisEntry = thisEntry.iloc[0, :]
                        if thisEntry['count'] == 0:
                            continue
                        message = '{}/{}'.format(thisEntry['pass'], thisEntry['count'])
                        x = xIdx + offsets[hIdx]
                        y = 0
                        theAx.text(
                            x, y, message,
                            transform=trans, color=huePalette[hLabel],
                            **fontOpts)
            theAx.pvalsAnnotated = True
        return
    return numSigAnnotator

print('Saving plots to {}'.format(pdfPath))
with PdfPages(pdfPath) as pdf:
    rowVar = 'feature'
    rowOrder = sorted(np.unique(plotRC[rowVar]))
    colVar = 'stimCondition'
    colOrder = plotRC.loc[:, ['trialRateInHz', 'electrode', 'stimCondition']].drop_duplicates().sort_values(by=['electrode', 'trialRateInHz'])['stimCondition'].to_list()
    # colOrder = np.unique(plotRC[colVar])
    colWrap = min(3, len(colOrder))
    hueName = 'kinematicCondition'
    # hueOrder = sorted(np.unique(plotRC[hueName]))
    hueOrder = ['NA_NA', 'CW_outbound', 'CW_return']
    pal = sns.color_palette("Set2")
    huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
    huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
    height, width = 3, 3
    aspect = width / height
    if arguments['plotThePieces']:
        ####
        widthRatios = [3] * np.unique(testGroupPieces[colVar]).shape[0] + [1]
        plotLimsMin = plotRCPieces.groupby(rowVar).min()[whichRAUC]
        plotLimsMax = plotRCPieces.groupby(rowVar).max()[whichRAUC]
        plotLimsRange = plotLimsMax - plotLimsMin
        plotLimsMin -= plotLimsRange / 100
        plotLimsMax += plotLimsRange / 100
        xJ = testGroupPieces[amplitudeFieldName].diff().dropna().abs().unique().mean() / 20
        g = sns.lmplot(
            col=colVar, col_order=colOrder,
            row=rowVar,
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            data=testGroupPieces,
            ci=95, n_boot=100,
            x_jitter=xJ,
            scatter_kws=dict(s=2.5),
            height=height, aspect=aspect,
            facet_kws=dict(
                sharey=False, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)),
            )
        # ampStatsDF
        plotProcFuns = [
            genStatsAnnotator(ampStatsDF, relativeStatsDF, hueName, huePalette),
            asp.genTitleChanger(prettyNameLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA_0.0':
                refMask = (refGroupPieces[rowVar] == row_val)
                if refMask.any():
                    refData = refGroupPieces.loc[refMask, :]
                else:
                    refData = refGroupPieces
                sns.boxplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC,
                    hue=hueName, hue_order=hueOrder, palette=huePaletteAlpha,
                    data=refData, saturation=0.25,
                    ax=ax, whis=np.inf, dodge=True)
                sns.stripplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC, palette=huePalette,
                    hue=hueName, hue_order=hueOrder, data=refData,
                    ax=ax, size=2.5, dodge=True)
                ax.set_xlabel('')
                ax.set_xticks([])
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        g.set_axis_labels('Stimulation amplitude (uA)', 'Normalized AUC')
        asp.reformatFacetGridLegend(
            g, titleOverrides={
                'kinematicCondition': 'Movement type'
            },
            contentOverrides={
                'NA_NA': 'No movement',
                'CW_outbound': 'Start of movement (extension)',
                'CW_return': 'Return to start (flexion)'
            },
            styleOpts=styleOpts)
        g.resize_legend(adjust_subtitles=True)
        for (rN, cN), ax in g.axes_dict.items():
            ax.set_ylim([plotLimsMin.loc[rN], plotLimsMax.loc[rN]])
        # g.axes[0, 0].set_ylim(plotLims)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0,)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
    if True:
        stripplotKWArgs = dict(
            dodge=True, size=2.5,
            rasterized=False,
            # jitter=0.8,
            linewidth=snsRCParams['lines.linewidth']
            )
        boxplotKWArgs = dict(
            dodge=True,
            whis=np.inf,
            linewidth=snsRCParams['lines.linewidth'],
            # saturation=0.2,
            )
        colVar = 'electrode'
        rowVar = 'kinematicCondition'
        ################################################################################################################
        plotAmpStatsDF = ampStatsDF.reset_index()
        del ampStatsDF
        plotAmpStatsDF.loc[:, 'electrode'] = plotAmpStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotAmpStatsDF.loc[:, 'namesAndMD'] = plotAmpStatsDF['names']
        plotAmpStatsDF.loc[plotAmpStatsDF['isMahalDist'], 'namesAndMD'] += '_md'
        ################################################################################################################
        dummyEntriesReject = plotAmpStatsDF.drop_duplicates(subset=['freqBandName', 'electrode', 'namesAndMD', 'kinematicCondition']).copy()
        dummyEntriesReject.loc[:, ['coef', 'coefStd', 'se', 'T', 'pval', 'r2', 'adj_r2', 'relimp', 'relimp_perc']] = np.nan
        dummyEntriesReject.loc[:, 'reject'] = True
        dummyEntriesDoNotReject = dummyEntriesReject.copy()
        dummyEntriesDoNotReject.loc[:, 'reject'] = False
        plotAmpStatsDF = pd.concat([plotAmpStatsDF, dummyEntriesReject, dummyEntriesDoNotReject], ignore_index=True)
        #######
        # thisMaskAmp = plotAmpStatsDF['kinematicCondition'] != 'NA_NA'
        # thisMaskAmp = pd.Series(True, index=plotAmpStatsDF.index)
        # thisMaskAmp = plotAmpStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)
        thisMaskAmp = plotAmpStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) & ~plotAmpStatsDF['isMahalDist']
        thisFreqBandOrder = [
            fN for fN in freqBandOrderExtended if fN in plotAmpStatsDF.loc[thisMaskAmp, 'freqBandName'].unique().tolist()]
        rowOrder = [rN for rN in lOfKinematicsToPlot if rN in plotAmpStatsDF['kinematicCondition'].to_list()]
        #
        thisPalette = allAmpPalette.loc[allAmpPalette.index.isin(plotAmpStatsDF.loc[thisMaskAmp, 'namesAndMD'])]
        colOrder = [eN for eN in plotAmpStatsDF[colVar].unique().tolist() if eN !='NA']
        height, width = 2, 2
        aspect = width / height
        #
        countDF = plotAmpStatsDF.loc[thisMaskAmp].dropna().groupby(['kinematicCondition', 'electrode', 'freqBandName', 'namesAndMD']).count()['reject']
        passDF = plotAmpStatsDF.loc[thisMaskAmp].dropna().groupby(['kinematicCondition', 'electrode', 'freqBandName', 'namesAndMD']).sum()['reject']
        pvalDF = pd.concat([countDF, passDF], axis='columns')
        pvalDF.columns = ['count', 'pass']
        for yVar in ['coefStd']:
            g = sns.catplot(
                y=yVar,
                x='freqBandName', order=thisFreqBandOrder,
                col=colVar, col_order=colOrder,
                row=rowVar, row_order=rowOrder,
                hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(), color='w',
                data=plotAmpStatsDF.loc[thisMaskAmp, :],  # & plotAmpStatsDF['reject']
                height=height, aspect=aspect,
                sharey=True, sharex=True,  margin_titles=True,
                kind='box', **boxplotKWArgs
                # kind='violin', inner=None, cut=1,
                )
            for name, ax in g.axes_dict.items():
                rowName, colName = name
                # non-significant are transparent
                subSetMask = thisMaskAmp & (plotAmpStatsDF[rowVar] == rowName) & (plotAmpStatsDF[colVar] == colName) & (~plotAmpStatsDF['reject'])#& plotAmpStatsDF[yVar].notna()
                if subSetMask.any():
                    sns.stripplot(
                        data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder,
                        hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        alpha=0.2, **stripplotKWArgs)
                # significant are opaque
                subSetMask = thisMaskAmp & (plotAmpStatsDF[rowVar] == rowName) & (plotAmpStatsDF[colVar] == colName) & plotAmpStatsDF['reject']#& plotAmpStatsDF[yVar].notna()
                if subSetMask.any():
                    sns.stripplot(
                        data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder, hue='namesAndMD',
                        hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        **stripplotKWArgs
                        )
                # ax.tick_params(axis='x', labelrotation=30)
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                    ax.set_xticklabels(
                        newXTickLabels, rotation=30, va='top', ha='right')
                ax.axhline(0, c='r', zorder=2.5)
                for xJ in range(0, len(thisFreqBandOrder), 2):
                    ax.axvspan(-0.45 + xJ, 0.45 + xJ, color="0.1", alpha=0.1, zorder=1.)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            plotProcFuns = [
                genNumSigAnnotator(
                    pvalDF.reset_index(),
                    xOrder=thisFreqBandOrder, hueVar='namesAndMD', palette=thisPalette,
                    fontOpts=dict(
                        va='bottom', ha='center',
                        fontsize=snsRCParams["font.size"],
                        fontweight='bold', rotation=45)),
                asp.genTitleChanger(prettyNameLookup)
                ]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.suptitle('Coefficient distribution for AUC regression')
            asp.reformatFacetGridLegend(
                g, titleOverrides={
                    'namesAndMD': 'Regressors'
                },
                contentOverrides={
                    'namesAndMD': 'Regressors',
                    'trialAmplitude': 'Stimulation amplitude',
                    'trialAmplitude:trialRateInHz': 'Stimulation rate interaction',
                    'trialAmplitude_md': 'Stimulation amplitude (Mahal. dist.)',
                    'trialAmplitude:trialRateInHz_md': 'Stimulation rate interaction (Mahal. dist.)',
                },
            styleOpts=styleOpts)
            g.set_axis_labels(prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            ########################
            pdf.savefig(
                bbox_inches='tight', pad_inches=0,
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ######
        plotRelativeStatsDF = relativeStatsDF.reset_index()
        plotStatsList = [plotRelativeStatsDF]
        for cN in plotRelativeStatsDF['kinematicCondition'].unique():
            if cN == 'NA_NA':
                continue
            noStimMask = (relativeStatsNoStimDF['A'] == cN) & (relativeStatsNoStimDF['B'] == 'NA_NA')
            if noStimMask.any():
                takeThese = relativeStatsNoStimDF.loc[noStimMask, :].reset_index()
                takeThese.loc[:, 'kinematicCondition'] = cN
                takeThese.loc[:, 'kinAndElecCondition'] = 'NA_{}'.format(cN)
                takeThese.loc[:, 'stimCondition'] = 'NA_0.0'
                takeThese.loc[:, 'electrode'] = 'NA'
                takeThese.loc[:, 'trialRateInHz'] = 0.
                # drop duplicates to avoid having multiple mahalanobis distances?
                # takeThese.drop_duplicates(subset=['feature'], inplace=True)
                relevantColumns = [cN for cN in plotRelativeStatsDF.columns if cN in takeThese.columns]
                try:
                    plotStatsList.append(takeThese.loc[:, relevantColumns].copy())
                except Exception:
                    traceback.print_exc()
        plotRelativeStatsDF = pd.concat(plotStatsList)
        plotRelativeStatsDF.loc[:, 'trialRateInHzStr'] = plotRelativeStatsDF['trialRateInHz'].apply(lambda x: '{}'.format(x))
        plotRelativeStatsDF.loc[:, 'electrode'] = plotRelativeStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isMahalDist'], 'trialRateInHzStr'] += '_md'
        #####
        dummyEntriesReject = plotRelativeStatsDF.drop_duplicates(subset=['freqBandName', 'electrode', 'trialRateInHz']).copy()
        dummyEntriesReject.loc[:, ['hedges', 'T', 'pval', 'cohen-d']] = np.nan
        dummyEntriesReject.loc[:, 'reject'] = True
        dummyEntriesDoNotReject = dummyEntriesReject.copy()
        dummyEntriesDoNotReject.loc[:, 'reject'] = False
        plotRelativeStatsDF = pd.concat([plotRelativeStatsDF, dummyEntriesReject, dummyEntriesDoNotReject], ignore_index=True)
        #####
        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(x['feature'].replace('#0', '')), axis='columns')
        ###
        # thisMaskRel = (plotRelativeStatsDF['reject']) & (plotRelativeStatsDF['kinematicCondition'] != 'NA_NA')
        # thisMaskRelStimOnly = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)) & (plotRelativeStatsDF['stimCondition'] != 'NA_0.0')
        # thisMaskRel = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot))
        thisMaskRel = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)) & ~plotRelativeStatsDF['isMahalDist']
        thisMaskRelStimOnly = thisMaskRel & (plotRelativeStatsDF['stimCondition'] != 'NA_0.0')
        thisMaskRelNoStim = thisMaskRel & (plotRelativeStatsDF['stimCondition'] == 'NA_0.0')
        # thisMaskRel = pd.Series(True, index=plotRelativeStatsDF.index)
        # thisMaskRel = (plotRelativeStatsDF['stimCondition'] != 'NA_0.0')
        thisFreqBandOrder = [
            fN for fN in freqBandOrderExtended if fN in plotRelativeStatsDF.loc[thisMaskRel, 'freqBandName'].unique().tolist()]
        ####
        thisFullPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[(plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)), 'trialRateInHzStr'])]
        thisStimPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelStimOnly, 'trialRateInHzStr'])]
        thisNoStimPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelNoStim, 'trialRateInHzStr'])]
        # thisPalette = pd.Series(sns.color_palette('Set2_r', 4), index=['50.0', '50.0_md', '100.0', '100.0_md'])
        colOrder = ['NA'] + [eN for eN in plotRelativeStatsDF[colVar].sort_values().unique().tolist() if eN !='NA']
        countDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby(['kinematicCondition', colVar, 'freqBandName', 'trialRateInHzStr']).count()['reject']
        passDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby(['kinematicCondition', colVar, 'freqBandName', 'trialRateInHzStr']).sum()['reject']
        pvalDF = pd.concat([countDF, passDF], axis='columns')
        pvalDF.columns = ['count', 'pass']
        widthRatios = [3] * np.unique(plotRelativeStatsDF[colVar]).shape[0]
        widthRatios[0] = 2
        # plotRelativeStatsDF.loc[plotRelativeStatsDF['electrode'] == 'NA', :]
        for yVar in ['hedges']:
            g = sns.catplot(
                y=yVar,
                x='freqBandName',
                order=thisFreqBandOrder,
                col=colVar, col_order=colOrder,
                row=rowVar, row_order=rowOrder,
                height=height, aspect=aspect,
                sharey=True, sharex=True, margin_titles=True,
                hue='trialRateInHzStr', hue_order=thisStimPalette.index.to_list(), palette=thisStimPalette.to_dict(),
                data=plotRelativeStatsDF.loc[thisMaskRelStimOnly, :],  # & plotRelativeStatsDF['reject']
                facet_kws=dict(
                    gridspec_kws=dict(width_ratios=widthRatios)),
                kind='box',
                # kind='violin', inner=None, cut=1, width=0.9,
                # saturation=0.2,
                **boxplotKWArgs
                )
            for name, ax in g.axes_dict.items():
                rowName, colName = name
                if colName == 'NA':
                    thisPalette = thisNoStimPalette
                    subSetMask = (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) & plotRelativeStatsDF[yVar].notna()
                    sns.boxplot(
                        data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder, hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        **boxplotKWArgs)
                else:
                    thisPalette = thisStimPalette
                # plot non-significant observations with transparency
                subSetMask = (
                        (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                        (~plotRelativeStatsDF['reject']) & plotRelativeStatsDF[yVar].notna() & thisMaskRel)
                if subSetMask.any():
                    sns.stripplot(
                        data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName', order=thisFreqBandOrder,
                        hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        alpha=0.2, **stripplotKWArgs)
                # plot significant observations fully opaque
                subSetMask = (
                        (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                        plotRelativeStatsDF['reject'] & plotRelativeStatsDF[yVar].notna() & thisMaskRel)
                if subSetMask.any():
                    sns.stripplot(
                        data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName', order=thisFreqBandOrder,
                        hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        **stripplotKWArgs
                        )
                nSigAnnotator = genNumSigAnnotatorV2(
                    pvalDF.reset_index(),
                    xOrder=thisFreqBandOrder, hueVar='trialRateInHzStr', palette=thisPalette,
                    fontOpts=dict(
                        va='bottom', ha='center',
                        fontsize=snsRCParams["font.size"],
                        fontweight='bold', rotation=45))
                nSigAnnotator(g, ax, rowName, colName)
                ax.axhline(0, c='r', zorder=2.5)
                # ax.tick_params(axis='x', labelrotation=30)
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                    ax.set_xticklabels(newXTickLabels, rotation=90, va='top', ha='right')
                for xJ in range(0, len(thisFreqBandOrder), 2):
                    ax.axvspan(-0.45 + xJ, 0.45 + xJ, color="0.1", alpha=0.1, zorder=1.)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            plotProcFuns = [
                asp.genTitleChanger(prettyNameLookup),
                ]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.suptitle('Effect size distribution for stim vs no-stim comparisons')
            asp.reformatFacetGridLegend(
                g, titleOverrides=prettyNameLookup,
                contentOverrides=prettyNameLookup,
            styleOpts=styleOpts)
            g.set_axis_labels(prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0,)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        #####################################################################
        #####################################################################
        # dummy plot for legends
        g = sns.catplot(
            y=yVar,
            x=colVar, order=colOrder,
            height=height, aspect=aspect,
            sharey=True, sharex=True, margin_titles=True,
            hue='trialRateInHzStr', hue_order=thisFullPalette.index.to_list(), palette=thisFullPalette.to_dict(),
            data=plotRelativeStatsDF, # & plotRelativeStatsDF['reject']
            kind='box',
            # kind='violin', inner=None, cut=1, width=0.9,
            # saturation=0.2,
            **boxplotKWArgs
            )
        g.suptitle('Dummy plot to get full legend from')
        asp.reformatFacetGridLegend(
            g, titleOverrides=prettyNameLookup,
            contentOverrides=prettyNameLookup,
        styleOpts=styleOpts)
        g.set_axis_labels(prettyNameLookup[colVar], prettyNameLookup[yVar])
        g.resize_legend(adjust_subtitles=True)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0,)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        ######## mahal dist plots
        allFilledMarkers = matplotlib.lines.Line2D.filled_markers
        thisMaskRel = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)) & plotRelativeStatsDF['isMahalDist']
        if thisMaskRel.any():
            thisPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRel, 'trialRateInHzStr'])]
            yVar = 'hedges'
            g = sns.catplot(
                y=yVar,
                x=colVar, order=colOrder,
                row=rowVar, row_order=rowOrder,
                height=height, aspect=aspect,
                sharey=True, sharex=True, margin_titles=True,
                hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                data=plotRelativeStatsDF.loc[thisMaskRel, :],  # & plotRelativeStatsDF['reject']
                kind='box',
                # kind='violin', inner=None, cut=1, width=0.9,
                # saturation=0.2,
                **boxplotKWArgs
                )
            for name, ax in g.axes_dict.items():
                rowName = name #, colName
                for freqBandIdx, freqBandName in enumerate(thisFreqBandOrder):
                    # pdb.set_trace()
                    # non-significant
                    subSetMask = (plotRelativeStatsDF[rowVar] == rowName) & (~plotRelativeStatsDF['reject']) & thisMaskRel & (plotRelativeStatsDF['freqBandName'] == freqBandName)# & plotRelativeStatsDF[yVar].notna()
                    if subSetMask.any():
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x=colVar, order=colOrder,
                            marker=allFilledMarkers[freqBandIdx],
                            hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                            alpha=0.2, **stripplotKWArgs)
                    # significant
                    subSetMask = (plotRelativeStatsDF[rowVar] == rowName) & plotRelativeStatsDF['reject'] & thisMaskRel & (plotRelativeStatsDF['freqBandName'] == freqBandName) # & plotRelativeStatsDF[yVar].notna()
                    if subSetMask.any():
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x=colVar, order=colOrder,
                            hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                            **stripplotKWArgs
                            )
                applyNumSigAnnotator = True
                if applyNumSigAnnotator:
                    nSigAnnotator = genNumSigAnnotatorV2(
                        pvalDF.reset_index(),
                        xOrder=thisFreqBandOrder, hueVar='trialRateInHzStr', palette=thisPalette,
                        fontOpts=dict(
                            va='bottom', ha='center',
                            fontsize=snsRCParams["font.size"],
                            fontweight='bold'))
                nSigAnnotator(g, ax, rowName, colName)
                ax.axhline(0, c='r', zorder=2.5)
                # ax.tick_params(axis='x', labelrotation=30)
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                    ax.set_xticklabels(newXTickLabels, rotation=45, va='top', ha='right')
                for xJ in range(0, len(thisFreqBandOrder), 2):
                    ax.axvspan(-0.45 + xJ, 0.45 + xJ, color="0.1", alpha=0.1, zorder=1.)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            plotProcFuns = [
                asp.genTitleChanger(prettyNameLookup),
                ]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.suptitle('Effect size distribution for stim vs no-stim comparisons (Mahal dist)')
            asp.reformatFacetGridLegend(
                g, titleOverrides=prettyNameLookup,
                contentOverrides=prettyNameLookup,
            styleOpts=styleOpts)
            g.set_axis_labels(prettyNameLookup[colVar], prettyNameLookup[yVar])
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0,)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    hasTopo = relativeStatsDF.groupby(['xCoords', 'yCoords']).ngroups > 1
    plotTDist = False
    if hasTopo: # histogram t stats
        rowVar = 'feature'
        rowOrder = sorted(np.unique(plotRC[rowVar]))
        colVar = 'stimCondition'
        colOrder = sorted(np.unique(plotRC[colVar]))
        colWrap = min(3, len(colOrder))
        hueName = 'kinematicCondition'
        # hueOrder = sorted(np.unique(plotRC[hueName]))
        hueOrder = ['NA_NA', 'CW_outbound', 'CW_return']
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
        height, width = 3, 3
        aspect = width / height
        ################################
        # TODO: I don't remember why I recalculate the relative stats table here
        plotRelativeStatsDF = relativeStatsDF.reset_index()
        plotRelativeStatsDF.loc[:, 'electrode'] = plotRelativeStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotRelativeStatsDF = plotRelativeStatsDF.loc[plotRelativeStatsDF['stimCondition'] != 'NA_0.0', :]
        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(
                x['feature'].replace('#0', '')), axis='columns')
        if plotTDist:
            g = sns.displot(
                x='T', hue='freqBandName',
                data=plotRelativeStatsDF,
                col=colVar,
                col_order=[
                    cn
                    for cn in colOrder
                    if cn in plotRelativeStatsDF[colVar].to_list()],
                row=hueName,
                row_order=[
                    cn
                    for cn in hueOrder
                    if cn in plotRelativeStatsDF[hueName].to_list()],
                kind='hist', element='step', stat='density',
                height=2 * height, aspect=aspect,
                facet_kws=dict(sharey=False)
                )
            for anName, ax in g.axes_dict.items():
                fillMin, fillMax = (
                    plotRelativeStatsDF['critical_T_min'].mean(),
                    plotRelativeStatsDF['critical_T_max'].mean()
                    )
                ax.axvspan(fillMin, fillMax, color='r', alpha=0.1, zorder=-100)
            g.suptitle('T-statistic distribution')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight', pad_inches=0,
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #############
        if arguments['plotTopoEffectSize']:
            statPalettes = [
                sns.diverging_palette(220, 20, as_cmap=True),
                sns.diverging_palette(145, 300, s=60, as_cmap=True)
                ]
            for name, statsThisFB in plotRelativeStatsDF.groupby(['freqBandName', 'isMahalDist']):
                freqBandName, isMahalDist = name
                print('plotting topo {}'.format(name))
                try:
                    statsThisFB = statsThisFB.loc[(statsThisFB['stimCondition'] != 'NA_0.0').to_numpy(), :]
                    numStimC = statsThisFB['stimCondition'].unique().size
                    numKinC = statsThisFB['kinematicCondition'].unique().size
                    for statIdx, statName in enumerate(['hedges', 'T']):
                        fig, ax = plt.subplots(
                            numKinC, numStimC + 1,
                            figsize = (6 * numKinC, 6 * numStimC + .6),
                            gridspec_kw={
                                'width_ratios': [10] * numStimC + [1],
                                'wspace': 0.1}
                            )
                        vMin, vMax = statsThisFB[statName].min(), statsThisFB[statName].max()
                        cBarKinIdx = int(numKinC / 2)
                        cBarStimIdx = int(numStimC)
                        for kinIdx, stimIdx in product(range(numKinC), range(numStimC)):
                            kinName = np.unique(statsThisFB['kinematicCondition'])[kinIdx]
                            stimName = np.unique(statsThisFB['stimCondition'])[stimIdx]
                            thisMask = (statsThisFB['kinematicCondition'] == kinName) & (statsThisFB['stimCondition'] == stimName)
                            ann2D = statsThisFB.loc[thisMask, :].pivot(index='yCoords', columns='xCoords', values='sigAnn')
                            stats2D = statsThisFB.loc[thisMask, :].pivot(index='yCoords', columns='xCoords', values=statName)
                            heatMapKWs = dict(
                                vmin=vMin, vmax=vMax, center=0.,  fmt='s',
                                linewidths=0, cmap=statPalettes[statIdx],
                                annot=ann2D,
                                annot_kws=dict(fontsize=snsRCParams["font.size"]),
                                xticklabels=False, yticklabels=False, square=True
                                )
                            if (kinIdx == cBarKinIdx) and (stimIdx == (cBarStimIdx - 1)):
                                heatMapKWs['cbar'] = True
                                heatMapKWs['cbar_ax'] = ax[cBarKinIdx, cBarStimIdx]
                            else:
                                heatMapKWs['cbar'] = False
                            sns.heatmap(
                                data=stats2D, ax=ax[kinIdx, stimIdx], ** heatMapKWs)
                            ax[kinIdx, stimIdx].set_title('{}, {}'.format(kinName, stimName))
                            ax[kinIdx, stimIdx].set_xlabel('')
                            ax[kinIdx, stimIdx].set_ylabel('')
                        for kinIdx in range(numKinC):
                            if kinIdx != cBarKinIdx:
                                ax[kinIdx, cBarStimIdx].set_xticks([])
                                ax[kinIdx, cBarStimIdx].set_yticks([])
                                sns.despine(
                                    fig=fig, ax=ax[kinIdx, cBarStimIdx],
                                    top=True, right=True, left=True, bottom=True,
                                    offset=None, trim=False)
                            else:
                                ax[kinIdx, cBarStimIdx].set_ylabel('{}'.format(statName))
                        figTitle = fig.suptitle('{} ({})'.format(statName, freqBandName))
                        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                        pdf.savefig(
                            bbox_inches='tight', pad_inches=0, bbox_extra_artists=[figTitle])
                        if arguments['showFigures']:
                            plt.show()
                        else:
                            plt.close()
                except Exception:
                    traceback.print_exc()
    if arguments['plotTheAverage']:
        colVar = 'stimCondition'
        colOrder = sorted(np.unique(plotRC[colVar]))
        hueName = 'kinematicCondition'
        hueOrder = sorted(np.unique(plotRC[hueName]))
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        #
        rowVar = 'freqBandName'
        rowOrder = sorted(np.unique(plotRC[rowVar]))
        colWrap = min(3, len(colOrder))
        height, width = 3, 3
        aspect = width / height
        widthRatios = [3] * np.unique(testGroup[colVar]).shape[0] + [1]
        '''
        g = sns.relplot(
            col=colVar, col_order=colOrder,
            row=rowVar,
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            data=testGroup,
            height=height, aspect=aspect,
            # kind='line',
            # errorbar='sd', estimator='mean', lw=1,
            kind='scatter',
            facet_kws=dict(
                sharey=True, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)),
            )'''
        plotLimsMin = plotRC.groupby(rowVar).min()[whichRAUC]
        plotLimsMax = plotRC.groupby(rowVar).max()[whichRAUC]
        plotLimsRange = plotLimsMax - plotLimsMin
        plotLimsMin -= plotLimsRange / 100
        plotLimsMax += plotLimsRange / 100
        xJ = testGroup[amplitudeFieldName].diff().dropna().abs().unique().mean() / 20
        g = sns.lmplot(
            col=colVar, col_order=colOrder,
            row=rowVar,
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            data=testGroup,
            ci=95, n_boot=100,
            x_jitter=xJ,
            scatter_kws=dict(s=2.5),
            height=height, aspect=aspect,
            facet_kws=dict(
                sharey=False, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)),
            )
        plotProcFuns = [
            genStatsAnnotator(ampStatsPerFBDF, relativeStatsPerFBDF, hueName, huePalette),
            asp.genTitleChanger(prettyNameLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA_0.0':
                refMask = (refGroup[rowVar] == row_val)
                if refMask.any():
                    refData = refGroup.loc[refMask, :]
                else:
                    refData = refGroup
                sns.violinplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC, palette=huePalette,
                    data=refData,
                    linewidth=0.,
                    cut=0, inner='box', saturation=0.25,
                    ax=ax)
                # sns.swarmplot(
                #     x=hueName, order=hueOrder,
                #     y=whichRAUC, palette=huePalette,
                #     hue=hueName, hue_order=hueOrder, data=refData,
                #     ax=ax, size=1.)
                ax.set_xlabel('')
                ax.set_xticks([])
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        g.set_axis_labels('Stimulation amplitude (uA)', 'Normalized AUC')
        asp.reformatFacetGridLegend(
            g, titleOverrides={
                'kinematicCondition': 'Movement type'
            },
            contentOverrides={
                'NA_NA': 'No movement',
                'CW_outbound': 'Start of movement (extension)',
                'CW_return': 'Return to start (flexion)'
            },
            styleOpts=styleOpts)
        g.resize_legend(adjust_subtitles=True)
        g.axes[0, 0].set_ylim(plotLims)
        pdf.savefig(
            bbox_inches='tight',
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
