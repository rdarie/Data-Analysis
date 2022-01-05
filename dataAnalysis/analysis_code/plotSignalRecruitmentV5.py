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
import pingouin as pg
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.preproc.ns5 as ns5
import os
from statannotations.Annotator import Annotator
from statannotations.stats.StatResult import StatResult
from statannotations.PValueFormat import PValueFormat
from itertools import product
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime as dt
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .75,
        'lines.markersize': 2.4,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 7,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
        'axes.facecolor': 'w',
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
    font_scale=.8, color_codes=True, rc=snsRCParams)
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

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName'])
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
calcSubFolder = os.path.join(analysisSubFolder, 'dataframes')
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'], 'lfp_recruitment')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
#
resultPath = os.path.join(
    calcSubFolder,
    blockBaseName + '{}_{}_rauc.h5'.format(
        inputBlockSuffix, arguments['window']))
print('loading {}'.format(resultPath))
outlierTrials = ash.processOutlierTrials(
    scratchFolder, blockBaseName, **arguments)
#  Overrides
limitPages = None
#  End Overrides
#
compoundAnnLookupDF = pd.read_hdf(resultPath, 'compoundAnnLookup')
rawRecCurve = pd.read_hdf(resultPath, 'raw')
recCurveFeatureInfo = rawRecCurve.columns.to_frame().reset_index(drop=True)
rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
recCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
scaledRaucDF = pd.read_hdf(resultPath, 'scaled')
scaledRaucDF.columns = scaledRaucDF.columns.get_level_values('feature')
recCurve.loc[:, 'scaledRAUC'] = scaledRaucDF.stack().to_numpy()
# relativeRaucDF = pd.read_hdf(resultPath, 'relative')
# relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
# recCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()
clippedRaucDF = pd.read_hdf(resultPath, 'clipped')
clippedRaucDF.columns = clippedRaucDF.columns.get_level_values('feature')
recCurve.loc[:, 'clippedRAUC'] = clippedRaucDF.stack().to_numpy()

whichRAUC = 'rawRAUC'
# whichRAUC = 'normalizedRAUC'
# whichRAUC = 'rauc'
# whichRAUC = 'scaledRAUC'

ampStatsDF = pd.read_hdf(resultPath, 'amplitudeStats')
ampStatsDF.drop(labels=['Intercept'], axis='index', level='names', inplace=True)
relativeStatsDF = pd.read_hdf(resultPath, 'relativeStatsDF')
ampStatsPerFBDF = pd.read_hdf(resultPath, 'amplitudeStatsPerFreqBand')
relativeStatsPerFBDF = pd.read_hdf(resultPath, 'relativeStatsDFPerFreqBand')
relativeStatsNoStimDF = pd.read_hdf(resultPath, 'noStimTTest')
#
correctMultiCompHere = True
confidence_alpha = .05
if correctMultiCompHere:
    pvalsDict = {
        'amp': ampStatsDF.loc[:, ['pval']].reset_index(drop=True),
        'relative': relativeStatsDF.loc[:, ['pval']].reset_index(drop=True),
        'relative_no_stim': relativeStatsNoStimDF.loc[:, ['pval']].reset_index(drop=True),
        }
    pvalsCatDF = pd.concat(pvalsDict, names=['origin', 'originIndex'])
    reject, pval = pg.multicomp(pvalsCatDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_by')
    pvalsCatDF.loc[:, 'pval'] = pval
    pvalsCatDF.loc[:, 'reject'] = reject
    for cN in ['pval', 'reject']:
        ampStatsDF.loc[:, cN] = pvalsCatDF.xs('amp', level='origin')[cN].to_numpy()
        relativeStatsDF.loc[:, cN] = pvalsCatDF.xs('relative', level='origin')[cN].to_numpy()
        relativeStatsNoStimDF.loc[:, cN] = pvalsCatDF.xs('relative_no_stim', level='origin')[cN].to_numpy()
    reject, pval = pg.multicomp(ampStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_by')
    ampStatsPerFBDF.loc[:, 'pval'] = pval
    ampStatsPerFBDF.loc[:, 'reject'] = reject
    reject, pval = pg.multicomp(relativeStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_by')
    relativeStatsPerFBDF.loc[:, 'pval'] = pval
    relativeStatsPerFBDF.loc[:, 'reject'] = reject

ampStatsDF.loc[:, 'isMahalDist'] = ('mahal' in inputBlockSuffix)
ampStatsDF.set_index('isMahalDist', append=True, inplace=True)
#
relativeStatsDF.loc[:, 'isMahalDist'] = ('mahal' in inputBlockSuffix)
relativeStatsDF.set_index('isMahalDist', append=True, inplace=True)
#
relativeStatsNoStimDF.loc[:, 'isMahalDist'] = ('mahal' in inputBlockSuffix)
relativeStatsNoStimDF.set_index('isMahalDist', append=True, inplace=True)
relativeStatsNoStimDF.loc[:, 'T_abs'] = relativeStatsNoStimDF['T'].abs()
for cN in relativeStatsDF.columns:
    if cN not in relativeStatsNoStimDF.columns:
        relativeStatsNoStimDF.loc[:, cN] = 0
#

allAmpPalette = pd.Series(sns.color_palette('Set3')[0:8:2], index=['trialAmplitude', 'trialAmplitude:trialRateInHz', 'trialAmplitude_md', 'trialAmplitude:trialRateInHz_md'])
allRelPalette = pd.Series(sns.color_palette('Set3')[1:12:2], index=['50.0', '100.0', '50.0_md', '100.0_md', '0.0', '0.0_md'])

pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '_{}{}_{}_{}.pdf'.format(
        expDateTimePathStr, inputBlockSuffix, arguments['window'],
        'RAUC'))

recCurve.loc[:, 'freqBandName'] = recCurve.index.get_level_values('feature').map(recCurveFeatureInfo[['feature', 'freqBandName']].set_index('feature')['freqBandName'])
recCurve.set_index('freqBandName', append=True, inplace=True)
plotRC = recCurve.reset_index()
kinematicOrderMaster = ['NA_NA', 'CCW_outbound', 'CW_outbound', 'CCW_return', 'CW_return']
kinematicOrder = [hN for hN in kinematicOrderMaster if hN in plotRC['kinematicCondition'].to_list()]
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
relativeStatsDF.loc[:, 'T_abs'] = relativeStatsDF['T'].abs()
nFeats = plotRC['feature'].unique().shape[0]
nFeatsToPlot = max(min(5, int(np.floor(nFeats/2))), 1)
keepTopIdx = (
    [i for i in range(nFeatsToPlot)] +
    [i for i in range(-1 * nFeatsToPlot, 0)]
    )
keepColsForPlot = []
rankMask = relativeStatsDF.index.get_level_values('stimCondition') != 'NA_0.0'
for freqBandName, relativeStatsThisFB in relativeStatsDF.loc[rankMask, :].groupby('freqBandName'):
    noStimStatsRankingDF = relativeStatsNoStimDF.xs(freqBandName, level='freqBandName').groupby('feature').mean()['T']
    statsRankingDF = relativeStatsThisFB.groupby('feature').mean()['T']
    #
    orderedStats = (noStimStatsRankingDF + statsRankingDF).sort_values(ascending=False)
    nFeatsToPlot = max(min(4, int(np.floor(orderedStats.shape[0]/2))), 1)
    keepTopIdx = (
        [i for i in range(nFeatsToPlot)]
        # [i for i in range(-1 * nFeatsToPlot, 0)]
        )
    keepColsForPlot += orderedStats.index[keepTopIdx].to_list()
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
#####
try:
    recCurveFeatureInfo.loc[:, 'xIdx'], recCurveFeatureInfo.loc[:, 'yIdx'] = ssplt.coordsToIndices(
        recCurveFeatureInfo['xCoords'], recCurveFeatureInfo['yCoords'])
except Exception:
    recCurveFeatureInfo.loc[:, 'xIdx'] = 0.
    recCurveFeatureInfo.loc[:, 'yIdx'] = 0.
    recCurveFeatureInfo.loc[:, 'xCoords'] = 0.
    recCurveFeatureInfo.loc[:, 'yCoords'] = 0.

def genStatsAnnotator(ampDF, relDF, hN, hP):
    def statsAnnotator(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
        if not emptySubset:
            if not hasattr(g.axes[ro, co], 'starsAnnotated'):
                if g._row_var is not None:
                    rocoSubset = g.data.loc[g.data[g._row_var] == g.row_names[ro], :]
                else:
                    rocoSubset = g.data
                if g._col_var is not None:
                    rocoSubset = rocoSubset.loc[rocoSubset[g._col_var] == g.col_names[co], :]
                pvFormatter = PValueFormat()
                pvFormatter.config(text_format='simple')
                xLim = g.axes[ro, co].get_xlim()
                yLim = g.axes[ro, co].get_ylim()
                trans = transforms.blended_transform_factory(
                    g.axes[ro, co].transAxes, g.axes[ro, co].transData)
                # dx = (xLim[1] - xLim[0]) / 5
                # dy = (yLim[1] - yLim[0]) / 5
                offsets = np.linspace(0, .8 - .8 / rocoSubset.groupby([hN]).ngroups, rocoSubset.groupby([hN]).ngroups)
                offsets -= offsets.mean()
                offsets += 0.5
                for hnIdx, (hn, group) in enumerate(rocoSubset.groupby([hN])):
                    rn = group[g._row_var].unique()[0]
                    cn = group[g._col_var].unique()[0]
                    thisElectrode = compoundAnnLookupDF.loc[cn, 'electrode']
                    x = group[g._x_var].max()
                    y = group.groupby(g._x_var).mean().loc[x, g._y_var]
                    messageList = []
                    ##
                    st = ampDF.xs(rn, level=g._row_var).xs(thisElectrode, level='electrode').xs(hn, level=hN).xs('trialAmplitude', level='names')
                    if st['reject'].iloc[0]:
                        thisPV = StatResult(
                            test_description='', test_short_name='', stat_str=r"$\beta_A$",
                            stat=st['coef'].iloc[0],
                            pval=st['pval'].iloc[0],
                            alpha=confidence_alpha)
                        messageList.append(r"$\beta_A$" + ' = {:0.2f} ({})'.format(st['coef'].iloc[0], pvFormatter.format_data(thisPV)))
                    ##
                    try:
                        st = ampDF.xs(rn, level=g._row_var).xs(thisElectrode, level='electrode').xs(hn, level=hN).xs('trialAmplitude:trialRateInHz', level='names')
                        if st['reject'].iloc[0]:
                            thisPV = StatResult(
                                test_description='', test_short_name='', stat_str=r"$\beta_R$",
                                stat=st['coef'].iloc[0],
                                pval=st['pval'].iloc[0],
                                alpha=confidence_alpha)
                            messageList.append(r"$\beta_R$" + ' = {:0.2f} ({})'.format(st['coef'].iloc[0], pvFormatter.format_data(thisPV)))
                    except Exception:
                        pass
                    ##
                    rst = relDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hN)
                    # if rst['pval'].iloc[0] < (1-confidence_alpha):
                    if rst['reject'].iloc[0]:
                        thisPV = StatResult(
                            test_description='', test_short_name='', stat_str=r"$g_{SM}$",
                            stat=rst['hedges'].iloc[0],
                            pval=rst['pval'].iloc[0],
                            alpha=confidence_alpha)
                        messageList.append(r"$g_{SM}$" + ' = {:0.2f} ({})'.format(rst['hedges'].iloc[0], pvFormatter.format_data(thisPV)))
                    if len(messageList):
                        message = '\n'.join(messageList)
                        '''g.axes[ro, co].text(
                            1., y, message, transform=trans,
                            color=hP[hn],
                            va='bottom', ha='left',
                            )'''
                        g.axes[ro, co].text(
                            0.05, offsets[hnIdx], message,
                            transform=g.axes[ro, co].transAxes,
                            va='center', ha='left', fontsize=snsRCParams["xtick.labelsize"],
                            bbox=dict(facecolor=hP[hn], alpha=0.1)
                            )
                '''if (ro == 0) and (co == 0):
                    g.axes[ro, co].text(
                        0.95, 0.95,
                        '\n'.join([
                            '+: highest amp. stim. vs baseline (p < 0.05)',
                            '*: amplitude vs auc (p < 0.05)',
                            '^: rate vs auc (p < 0.05)']),
                        va='top', ha='right', fontsize=4, transform=g.axes[ro, co].transAxes)'''
                g.axes[ro, co].starsAnnotated = True
        return
    return statsAnnotator

def genStatsAnnotatorV2(ampDF, hN, hP, hO):
    def statsAnnotator(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
        if not emptySubset:
            if not hasattr(g.axes[ro, co], 'starsAnnotated'):
                thesePvalAnns = []
                significantPairs = []
                if g._row_var is not None:
                    dataMask = (g.data[g._row_var] == g.row_names[ro])
                    statsThisFeature = ampDF.xs(g.row_names[ro], level=g._row_var)
                else:
                    dataMask = pd.Series(True, index=g.data.index)
                    statsThisFeature = ampDF.copy()
                if g._col_var is not None:
                    dataMask = dataMask & (g.data[g._col_var] == g.col_names[co])
                    if g._col_var == 'stimCondition':
                        thisElectrode = compoundAnnLookupDF.loc[g.col_names[co], 'electrode']
                        statsThisFeature = statsThisFeature.xs(thisElectrode, level='electrode')
                    else:
                        statsThisFeature = statsThisFeature.xs(g.col_names[co], level=g._col_var)
                statsThisFeature.reset_index(inplace=True)
                pvFormatter = PValueFormat()
                pvFormatter.config(text_format='simple')
                for hn, group in dataSubset.groupby([hN]):
                    tp = ((group[g._x_var].min(), hn), (group[g._x_var].max(), hn),)
                    hueMask = statsThisFeature[hN] == hn
                    hueMaskRate = hueMask & (statsThisFeature['names'] == 'trialAmplitude:trialRateInHz')
                    if hueMaskRate.any():
                        pass
                    hueMaskAmp = hueMask & (statsThisFeature['names'] == 'trialAmplitude')
                    if hueMaskAmp.any():
                        assert hueMaskAmp.sum() == 1
                        ampStatsThisHue = statsThisFeature.loc[hueMaskAmp, :].iloc[0]
                        thisPV = StatResult(
                            test_description='', test_short_name='', stat_str=r"$\beta_{A}$",
                            stat=ampStatsThisHue['coef'],
                            pval=ampStatsThisHue['pval'],
                            alpha=confidence_alpha)
                        if thisPV.pvalue < confidence_alpha:
                            significantPairs.append(tp)
                            thesePvalAnns.append(r"$\beta_{A}$" + ' = {:0.2f} ({})'.format(
                                ampStatsThisHue['coef'], pvFormatter.format_data(thisPV)))
                if len(significantPairs):
                    annotator = Annotator(
                        g.axes[ro, co], significantPairs,
                        plot='boxplot',
                        x=g._x_var, order=np.unique(g.data.loc[dataMask, g._x_var]),
                        y=g._y_var,
                        hue=hN, hue_order=[hh for hh in hO if hh in g.data.loc[dataMask, hN].to_list()],
                        palette=hP, data=g.data.loc[dataMask, :],
                        )
                    annotator.configure(test=None, test_short_name='LR')
                    annotator.annotate_custom_annotations(thesePvalAnns)
                g.axes[ro, co].starsAnnotated = True
        return
    return statsAnnotator

def genNumSigAnnotator(pvalDF, xOrder=None, hueVar=None, palette=None, fontOpts={}):
    def numSigAnnotator(g, ro, co, hu, dataSubset):
        if not hasattr(g.axes[ro, co], 'pvalsAnnotated'):
            trans = transforms.blended_transform_factory(
                g.axes[ro, co].transData, g.axes[ro, co].transAxes)
            hueOrder = palette.index.to_list()
            huePalette = palette.to_dict()
            offsets = np.linspace(0, .8 - .8 / len(hueOrder), len(hueOrder))
            offsets -= offsets.mean()
            thisMask = pvalDF.notna().all(axis='columns')
            if g._col_var is not None:
                thisMask = thisMask & (pvalDF[g._col_var] == g.col_names[co])
            if g._row_var is not None:
                thisMask = thisMask & (pvalDF[g._row_var] == g.row_names[ro])
            for xIdx, xLabel in enumerate(xOrder):
                xMask = (pvalDF[g._x_var] == xLabel)
                for hIdx, hLabel in enumerate(hueOrder):
                    hMask = (pvalDF[hueVar] == hLabel)
                    totalMask = (thisMask & xMask & hMask)
                    if totalMask.any():
                        thisEntry = pvalDF.loc[totalMask, :]
                        try:
                            assert thisEntry.shape[0] == 1
                        except:
                            traceback.print_exc()
                        thisEntry = thisEntry.iloc[0, :]
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

with PdfPages(pdfPath) as pdf:
    # plotLims = [0, plotRC[whichRAUC].quantile(1-1e-2)]
    '''plotLims = [
        plotRCPieces[whichRAUC].quantile(1e-2),
        plotRCPieces[whichRAUC].quantile(1-1e-2)]'''
    #
    plotLims = [
        plotRCPieces[whichRAUC].min(),
        plotRCPieces[whichRAUC].max()]
    rowName = 'feature'
    rowOrder = sorted(np.unique(plotRC[rowName]))
    colName = 'stimCondition'
    colOrder = [iV for iV in compoundAnnLookupDF.sort_values(['trialRateInHz', 'electrode']).index if iV in np.unique(plotRC[colName])]
    colWrap = min(3, len(colOrder))
    hueName = 'kinematicCondition'
    # hueOrder = sorted(np.unique(plotRC[hueName]))
    hueOrder = kinematicOrder
    pal = sns.color_palette("Set2")
    huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
    huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
    height, width = 2, 3
    aspect = width / height
    if arguments['plotThePieces']:
        widthRatios = [1] + [3] * np.unique(testGroupPieces[colName]).shape[0]
        plotLimsMin = plotRCPieces.groupby(rowName).min()[whichRAUC]
        plotLimsMax = plotRCPieces.groupby(rowName).max()[whichRAUC]
        plotLimsRange = plotLimsMax - plotLimsMin
        plotLimsMin -= plotLimsRange / 100
        plotLimsMax += plotLimsRange / 100
        xJ = testGroupPieces[amplitudeFieldName].diff().dropna().abs().unique().mean() / 20
        g = sns.lmplot(
            col=colName, col_order=colOrder,
            row=rowName,
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
            # genStatsAnnotatorV2(ampStatsDF, hueName, huePalette, hueOrder),
            asp.genTitleChanger(prettyNameLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        g.set_axis_labels('Stimulation amplitude (uA)', 'Normalized AUC')
        for ro in range(g.axes.shape[0]):
            for co in range(g.axes.shape[1]):
                if co > 0:
                    g.axes[ro, 0].get_shared_y_axes().join(g.axes[ro, 0], g.axes[ro, co])
                    g.axes[ro, co].set_ylabel(None)
                    g.axes[ro, co].set_yticklabels([])
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA_0.0':
                refMask = (refGroupPieces[rowName] == row_val)
                if refMask.any():
                    refData = refGroupPieces.loc[refMask, :]
                else:
                    refData = refGroupPieces
                sns.boxplot(
                    # x=hueName, order=hueOrder,
                    x=colName,
                    y=whichRAUC,
                    hue=hueName, hue_order=hueOrder, palette=huePaletteAlpha,
                    data=refData, # saturation=0.25,
                    ax=ax, whis=np.inf, dodge=True)
                sns.stripplot(
                    # x=hueName, order=hueOrder,
                    x=colName,
                    y=whichRAUC,
                    hue=hueName, hue_order=hueOrder, palette=huePalette, data=refData,
                    ax=ax, size=2.5, dodge=True)
                ###
                pairs = [("CW_outbound", "NA_NA"), ("CW_return", "NA_NA")]
                thesePvalAnns = []
                significantPairs = []
                statsThisFeature = relativeStatsNoStimDF.xs(row_val, level=rowName)
                pvFormatter = PValueFormat()
                pvFormatter.config(text_format='simple')
                for tp in pairs:
                    pairMask = (statsThisFeature['A'] == tp[0]) & (statsThisFeature['B'] == tp[1])
                    assert pairMask.sum() == 1
                    thisPV = StatResult(
                        test_description='', test_short_name='', stat_str='T',
                        stat=statsThisFeature.loc[pairMask, 'T'].iloc[0],
                        pval=statsThisFeature.loc[pairMask, 'pval'].iloc[0],
                        alpha=confidence_alpha)
                    if thisPV.pvalue < confidence_alpha:
                        significantPairs.append(tp)
                        thesePvalAnns.append(r"$g_M$" + ' = {:0.2f} ({})'.format(
                            statsThisFeature.loc[pairMask, 'hedges'].iloc[0], pvFormatter.format_data(thisPV)))
                # pdb.set_trace()
                if len(significantPairs):
                    # pairsDoubled = [
                    #     ((tp[0], tp[0]), (tp[1], tp[1]))
                    #     for tp in significantPairs
                    #     ]
                    pairsDoubled = [
                        ((col_val, tp[0]), (col_val, tp[1]))
                        for tp in significantPairs
                        ]
                    annotator = Annotator(
                        ax, pairsDoubled,
                        # x=hueName, order=hueOrder,
                        x=colName,
                        y=whichRAUC,
                        hue=hueName, hue_order=hueOrder, palette=huePalette, data=refData,
                        size=2.5, whis=np.inf, dodge=True)
                    annotator.configure(test=None, test_short_name='WT', line_height=0.01)
                    annotator.annotate_custom_annotations(thesePvalAnns)
                ###
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                ax.set_xticks([])
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
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
        # for (rN, cN), ax in g.axes_dict.items():
        #     ax.set_ylim([plotLimsMin.loc[rN], plotLimsMax.loc[rN]])
        # g.axes[0, 0].set_ylim(plotLims)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0, )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
    if True:
        ###
        plotAmpStatsDF = ampStatsDF.reset_index()
        plotAmpStatsDF.loc[:, 'namesAndMD'] = plotAmpStatsDF['names']
        plotAmpStatsDF.loc[plotAmpStatsDF['isMahalDist'], 'namesAndMD'] += '_md'
        #######
        dummyEntries = plotAmpStatsDF.drop_duplicates(subset=['freqBandName', 'electrode', 'names']).copy()
        dummyEntries.loc[:, ['coef', 'se', 'T', 'pval', 'r2', 'adj_r2', 'relimp', 'relimp_perc']] = np.nan
        dummyEntries.loc[:, 'reject'] = True
        plotAmpStatsDF = pd.concat([plotAmpStatsDF, dummyEntries], ignore_index=True)
        #######
        thisFreqBandOrder = [
            fN
            for fN in freqBandOrderExtended
            if fN in plotAmpStatsDF['freqBandName'].unique().tolist()]
        # thisMaskAmp = plotAmpStatsDF['kinematicCondition'] != 'NA_NA'
        thisMaskAmp = pd.Series(True, index=plotAmpStatsDF.index)
        # thisMaskAmp = plotAmpStatsDF['kinematicCondition'].isin(['CW_outbound', 'CCW_outbound'])
        thisPalette = allAmpPalette.loc[allAmpPalette.index.isin(plotAmpStatsDF['namesAndMD'])]
        colOrder = sorted(plotAmpStatsDF['electrode'].unique().tolist())
        height, width = 3, 5
        aspect = width / height
        #
        countDF = plotAmpStatsDF.loc[thisMaskAmp].dropna().groupby(['kinematicCondition', 'electrode', 'freqBandName', 'names']).count()['reject']
        passDF = plotAmpStatsDF.loc[thisMaskAmp].dropna().groupby(['kinematicCondition', 'electrode', 'freqBandName', 'names']).sum()['reject']
        pvalDF = pd.concat([countDF, passDF], axis='columns')
        pvalDF.columns = ['count', 'pass']
        for yVar in ['coef', 'relimp']:
            g = sns.catplot(
                y=yVar,
                x='freqBandName', order=thisFreqBandOrder,
                col='electrode', col_order=colOrder,
                row='kinematicCondition',
                hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(), color='w',
                data=plotAmpStatsDF.loc[thisMaskAmp, :],  # & plotAmpStatsDF['reject']
                height=height, aspect=aspect, margin_titles=True,
                # kind='box', whis=np.inf,
                kind='violin', inner=None, cut=0,
                saturation=0.2, linewidth=0.5
                )
            for name, ax in g.axes_dict.items():
                kinName, elecName = name
                subSetMask = thisMaskAmp & (plotAmpStatsDF['kinematicCondition'] == kinName) & (plotAmpStatsDF['electrode'] == elecName) & (~plotAmpStatsDF['reject']) & plotAmpStatsDF[yVar].notna()
                if subSetMask.any():
                    sns.stripplot(
                        data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder,
                        hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        dodge=True, linewidth=0., alpha=0.2)
                subSetMask = thisMaskAmp & (plotAmpStatsDF['kinematicCondition'] == kinName) & (plotAmpStatsDF['electrode'] == elecName) & plotAmpStatsDF['reject'] & plotAmpStatsDF[yVar].notna()
                if subSetMask.any():
                    sns.stripplot(
                        data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder, hue='namesAndMD',
                        hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        dodge=True, linewidth=0.5)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            plotProcFuns = [
                genNumSigAnnotator(
                    pvalDF.reset_index(),
                    xOrder=thisFreqBandOrder, hueVar='names', palette=thisPalette,
                    fontOpts=dict(va='bottom', ha='center', fontsize=6, fontweight='bold')),
                ]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.suptitle('Coefficient distribution for AUC regression')
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
        plotNoStimStatsList = [plotRelativeStatsDF]
        for cN in plotRelativeStatsDF['kinematicCondition'].unique():
            if cN == 'NA_NA':
                continue
            noStimMask = (relativeStatsNoStimDF['A'] == cN) & (relativeStatsNoStimDF['B'] == 'NA_NA')
            if noStimMask.any():
                takeThese = relativeStatsNoStimDF.loc[noStimMask, :].reset_index()
                takeThese.loc[:, 'kinematicCondition'] = cN
                takeThese.loc[:, 'stimCondition'] = 'NA_0.0'
                takeThese.loc[:, 'electrode'] = 'NA'
                takeThese.loc[:, 'trialRateInHz'] = 0.
                #
                plotNoStimStatsList.append(takeThese.loc[:, plotRelativeStatsDF.columns].copy())
            else:
                print('TODO: resolve case where noStimMask is not true because A and B names are flipped')
                pdb.set_trace()
        plotRelativeStatsDF = pd.concat(plotNoStimStatsList)
        plotRelativeStatsDF.loc[:, 'trialRateInHzStr'] = plotRelativeStatsDF['trialRateInHz'].apply(lambda x: '{}'.format(x))
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isMahalDist'], 'trialRateInHzStr'] += '_md'
        ######
        # dummyEntries = plotRelativeStatsDF.drop_duplicates(subset=['freqBandName', 'electrode', 'trialRateInHz']).copy()
        # dummyEntries.loc[:, ['hedges', 'T', 'pval', 'cohen-d']] = np.nan
        # dummyEntries.loc[:, 'reject'] = True
        # plotRelativeStatsDF = pd.concat([plotRelativeStatsDF, dummyEntries], ignore_index=True)
        #######
        thisFreqBandOrder = [fN for fN in freqBandOrderExtended if fN in plotRelativeStatsDF['freqBandName'].unique().tolist()]
        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(x['feature'].replace('#0', '')), axis='columns')
        # thisMaskRel = (plotRelativeStatsDF['reject']) & (plotRelativeStatsDF['kinematicCondition'] != 'NA_NA')
        # thisMaskRel = (plotRelativeStatsDF['kinematicCondition'] != 'NA_NA')
        thisMaskRel = pd.Series(True, index=plotRelativeStatsDF.index)
        thisPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF['trialRateInHzStr'])]
        # thisPalette = pd.Series(sns.color_palette('Set2_r', 4), index=['50.0', '50.0_md', '100.0', '100.0_md'])
        colOrder = sorted(plotRelativeStatsDF['electrode'].unique().tolist())
        # yVar = 'hedges'
        for yVar in ['hedges', 'cohen-d', 'glass']:
            countDF = plotRelativeStatsDF.loc[thisMaskRel].dropna().groupby(['kinematicCondition', 'electrode', 'freqBandName', 'trialRateInHzStr']).count()['reject']
            passDF = plotRelativeStatsDF.loc[thisMaskRel].dropna().groupby(['kinematicCondition', 'electrode', 'freqBandName', 'trialRateInHzStr']).sum()['reject']
            pvalDF = pd.concat([countDF, passDF], axis='columns')
            pvalDF.columns = ['count', 'pass']
            g = sns.catplot(
                y=yVar,
                x='freqBandName',
                order=thisFreqBandOrder,
                col='electrode', col_order=colOrder, row='kinematicCondition',
                height=height, aspect=aspect, margin_titles=True,
                hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                data=plotRelativeStatsDF.loc[thisMaskRel, :],  # & plotRelativeStatsDF['reject']
                # kind='box', whis=np.inf,
                kind='violin', inner=None, cut=0,
                saturation=0.2, linewidth=0.5)
            for name, ax in g.axes_dict.items():
                kinName, elecName = name
                subSetMask = (plotRelativeStatsDF['kinematicCondition'] == kinName) & (plotRelativeStatsDF['electrode'] == elecName) & (~plotRelativeStatsDF['reject']) & plotRelativeStatsDF[yVar].notna()
                if subSetMask.any():
                    sns.stripplot(
                        data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder, hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        dodge=True, alpha=0.2)
                subSetMask = (plotRelativeStatsDF['kinematicCondition'] == kinName) & (plotRelativeStatsDF['electrode'] == elecName) & plotRelativeStatsDF['reject'] & plotRelativeStatsDF[yVar].notna()
                if subSetMask.any():
                    sns.stripplot(
                        data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                        y=yVar, x='freqBandName',
                        order=thisFreqBandOrder, hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                        dodge=True, linewidth=0.5)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            plotProcFuns = [
                genNumSigAnnotator(
                    pvalDF.reset_index(),
                    xOrder=thisFreqBandOrder, hueVar='trialRateInHzStr', palette=thisPalette,
                    fontOpts=dict(va='bottom', ha='center', fontsize=4, fontweight='bold')),
                ]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.suptitle('Effect size distribution for stim vs no-stim comparisons')
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ####
    if relativeStatsDF.groupby(['xCoords', 'yCoords']).ngroups > 1: # histogram t stats
        rowName = 'feature'
        rowOrder = sorted(np.unique(plotRC[rowName]))
        colName = 'stimCondition'
        colOrder = sorted(np.unique(plotRC[colName]))
        colWrap = min(3, len(colOrder))
        hueName = 'kinematicCondition'
        hueOrder = kinematicOrder
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
        height, width = 3, 3
        aspect = width / height
        ################################
        plotRelativeStatsDF = plotRelativeStatsDF.loc[plotRelativeStatsDF['stimCondition'] != 'NA_0.0', :]
        g = sns.displot(
            x='T', hue='freqBandName',
            data=plotRelativeStatsDF,
            col=colName,
            col_order=[
                cn
                for cn in colOrder
                if cn in plotRelativeStatsDF[colName].to_list()],
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
            for freqBandName, statsThisFB in plotRelativeStatsDF.groupby(['freqBandName']):
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
                            ann2D = statsThisFB.loc[thisMask, :].dropna().pivot(index='yCoords', columns='xCoords', values='sigAnn')
                            stats2D = statsThisFB.loc[thisMask, :].dropna().pivot(index='yCoords', columns='xCoords', values=statName)
                            heatMapKWs = dict(
                                vmin=vMin, vmax=vMax, center=0.,  fmt='s',
                                linewidths=0, cmap=statPalettes[statIdx],
                                annot=ann2D, annot_kws=dict(fontsize=4.),
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
        colName = 'stimCondition'
        colOrder = sorted(np.unique(plotRC[colName]))
        hueName = 'kinematicCondition'
        hueOrder = kinematicOrder
        pal = sns.color_palette("Set1")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        #
        rowName = 'freqBandName'
        rowOrder = sorted(np.unique(plotRC[rowName]))
        colWrap = min(3, len(colOrder))
        height, width = 3, 3
        aspect = width / height
        widthRatios = [3] * np.unique(testGroup[colName]).shape[0] + [1]
        plotLimsMin = plotRC.groupby(rowName).min()[whichRAUC]
        plotLimsMax = plotRC.groupby(rowName).max()[whichRAUC]
        plotLimsRange = plotLimsMax - plotLimsMin
        plotLimsMin -= plotLimsRange / 100
        plotLimsMax += plotLimsRange / 100
        xJ = testGroup[amplitudeFieldName].diff().dropna().abs().unique().mean() / 20
        g = sns.lmplot(
            col=colName, col_order=colOrder,
            row=rowName,
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
                refMask = (refGroup[rowName] == row_val)
                if refMask.any():
                    refData = refGroup.loc[refMask, :]
                else:
                    refData = refGroup
                sns.boxplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC,
                    hue=hueName, hue_order=hueOrder, palette=huePaletteAlpha,
                    data=refData, saturation=0.25,
                    ax=ax, whis=np.inf)
                # sns.stripplot(
                #     x=hueName, order=hueOrder,
                #     y=whichRAUC, palette=huePalette,
                #     hue=hueName, hue_order=hueOrder, data=refData,
                #     ax=ax, size=2.5, dodge=True)
                ax.set_xlabel(None)
                ax.set_xticks([])
                # ax.set_yticklabels(['' for yL in ax.get_yticks()])
                ax.set_yticklabels([])
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        g.set_axis_labels('Stimulation amplitude (uA)', 'AUC')
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
            bbox_inches='tight', pad_inches=0,
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
