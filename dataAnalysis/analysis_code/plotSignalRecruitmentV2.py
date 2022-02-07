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
        'maskOutlierBlocks': False, 'analysisName': 'hiRes', 'window': 'XL', 'plotThePieces': True,
        'plotTopoEffectSize': True, 'lazy': False, 'unitQuery': 'mahal', 'inputBlockPrefix': 'Block',
        'plotTheAverage': False, 'blockIdx': '2', 'showFigures': False, 'alignQuery': 'starting',
        'invertOutlierMask': False, 'inputBlockSuffix': 'laplace_scaled', 'alignFolderName': 'motion',
        'exp': 'exp202101211100', 'verbose': False, 'processAll': True}
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
pedalDirCat = pd.CategoricalDtype(['NA', 'CW', 'CCW'], ordered=True)
pedalMoveCat = pd.CategoricalDtype(['NA', 'outbound', 'return'], ordered=True)
compoundAnnLookupDF.loc[:, 'pedalMovementCat'] = compoundAnnLookupDF['pedalMovementCat'].astype(pedalMoveCat)
compoundAnnLookupDF.loc[:, 'pedalDirection'] = compoundAnnLookupDF['pedalDirection'].astype(pedalDirCat)
spinalMapDF = spinalElectrodeMaps[subjectName].sort_values(['xCoords', 'yCoords'])
spinalElecCategoricalDtype = pd.CategoricalDtype(spinalMapDF.index.to_list(), ordered=True)
compoundAnnLookupDF.loc[:, 'electrode'] = compoundAnnLookupDF['electrode'].astype(spinalElecCategoricalDtype)
#
rawRecCurve = pd.read_hdf(resultPath, 'raw')
recCurveFeatureInfo = rawRecCurve.columns.to_frame().reset_index(drop=True)
rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
recCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
scaledRaucDF = pd.read_hdf(resultPath, 'boxcox')
scaledRaucDF.columns = scaledRaucDF.columns.get_level_values('feature')
recCurve.loc[:, 'scaledRAUC'] = scaledRaucDF.stack().to_numpy()
# relativeRaucDF = pd.read_hdf(resultPath, 'relative')
# relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
# recCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()
clippedRaucDF = pd.read_hdf(resultPath, 'raw_clipped')
clippedRaucDF.columns = clippedRaucDF.columns.get_level_values('feature')
recCurve.loc[:, 'clippedRAUC'] = clippedRaucDF.stack().to_numpy()

# whichRAUC = 'rawRAUC'
# whichRAUC = 'normalizedRAUC'
# whichRAUC = 'rauc'
whichRAUC = 'scaledRAUC'

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

allAmpPalette = pd.Series(
    sns.color_palette('Set3')[0:12:2],
    index=[
        'trialAmplitude', 'trialRateInHz', 'trialAmplitude:trialRateInHz',
        'trialAmplitude_md', 'trialRateInHz_md', 'trialAmplitude:trialRateInHz_md'])
allRelPalette = pd.Series(sns.color_palette('Set3')[1:12:2], index=['50.0', '100.0', '50.0_md', '100.0_md', '0.0', '0.0_md'])
ratePalette = pd.Series(sns.husl_palette(3), index=[0., 50., 100.])
#
stimConditionPalette = compoundAnnLookupDF['trialRateInHz'].dropna().map(ratePalette)
pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '_{}{}_{}_{}.pdf'.format(
        expDateTimePathStr, inputBlockSuffix, arguments['window'],
        'RAUC'))

recCurve.loc[:, 'freqBandName'] = recCurve.index.get_level_values('feature').map(recCurveFeatureInfo[['feature', 'freqBandName']].set_index('feature')['freqBandName'])
recCurve.set_index('freqBandName', append=True, inplace=True)
plotRC = recCurve.reset_index()
# plotRC = plotRC.loc[plotRC['kinematicCondition'].isin(['NA_NA', 'CCW_outbound', 'CW_outbound']), :]
kinematicOrderMaster = ['NA_NA', 'CCW_outbound', 'CW_outbound', 'CCW_return', 'CW_return']
kinematicOrder = [hN for hN in kinematicOrderMaster if hN in plotRC['kinematicCondition'].to_list()]
keepCols = [
    'segment', 'originalIndex', 't',
    'feature', 'freqBandName', 'lag',
    'stimCondition', 'kinematicCondition', 'kinAndElecCondition'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)
######
relativeStatsDF.loc[:, 'T_abs'] = relativeStatsDF['T'].abs()
ampStatsDF.loc[:, 'T_abs'] = ampStatsDF['T'].abs()
ampStatsDF.loc[:, 'coef_abs'] = ampStatsDF['coef'].abs()
# nFeats = plotRC['feature'].unique().shape[0]
# nFeatsToPlot = max(min(5, int(np.floor(nFeats/2))), 1)
# keepTopIdx = (
#     [i for i in range(nFeatsToPlot)] +
#     [i for i in range(-1 * nFeatsToPlot, 0)]
#     )
# ampStatsDF.groupby(['names', 'feature']).mean()['coef']
keepColsForPlot = []
rankMask = relativeStatsDF.index.get_level_values('stimCondition') != 'NA_0.0'
for freqBandName, relativeStatsThisFB in relativeStatsDF.loc[rankMask, :].groupby('freqBandName'):
    noStimStatsRankingDF = relativeStatsNoStimDF['T_abs'].xs(freqBandName, level='freqBandName').groupby('feature').max()
    ampStatsRankingDF = ampStatsDF['T_abs'].xs(freqBandName, level='freqBandName').groupby(['names', 'feature']).max()
    statsRankingDF = relativeStatsThisFB['T_abs'].groupby('feature').max()
    if True:
        orderedStats = (noStimStatsRankingDF + statsRankingDF + ampStatsRankingDF.xs('trialAmplitude:trialRateInHz', level='names')).sort_values(ascending=False)
    else:
        orderedStats = ampStatsRankingDF.xs('trialAmplitude:trialRateInHz', level='names').sort_values(ascending=False)
    nFeatsToPlot = max(min(2, int(np.floor(orderedStats.shape[0]/2))), 1)
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

def genRegressionResultsOverlay(
    statsResultsDF, rowVar=None, colVar=None, xOffset=None):
    def overlayRegressionResults(
            data=None,
            x=None, y=None,
            hue=None, hue_order=None, color=None,
            ci=90,
            *args, **kwargs):
        ax = plt.gca()
        theseStatsDF = statsResultsDF
        if rowVar is not None:
            rowNameList = data[rowVar].unique()
            assert len(rowNameList) == 1
            rowName = rowNameList[0]
            theseStatsDF = theseStatsDF.xs(rowName, level=rowVar)
        if colVar is not None:
            colNameList = data[colVar].unique()
            assert len(colNameList) == 1
            colName = colNameList[0]
            theseStatsDF = theseStatsDF.xs(colName, level=colVar)
        #
        if xOffset is not None:
            allOffsets = np.arange(len(hue_order)) - ((len(hue_order) - 1) / 2)
            thisOffset = allOffsets[hue_order.index(data[hue].unique()[0])] * 4 * xOffset
        else:
            thisOffset = 0.
        allHues = data[hue].unique()
        allRegressors = theseStatsDF.index.get_level_values('names')
        hueXInteraction = '{}:{}'.format(x, hue)
        for hueName in allHues:
            hueMask = (data[hueVar] == hueName)
            xx = np.linspace(data.loc[hueMask, x].min(), data.loc[hueMask, x].max(), 100)
            yy = np.zeros((xx.shape[0], ))
            nBoot = 10000
            normRng = np.random.default_rng()
            yyBoot = np.zeros((nBoot, xx.shape[0]))
            if 'Intercept' in allRegressors:
                interceptStats = theseStatsDF.xs('Intercept', level='names')
                assert interceptStats.shape[0] == 1
                interceptStats = interceptStats.iloc[0, :]
                yy += interceptStats['coef']
                # yyBoot += np.ones((xx.shape[0], )) * (interceptStats['coef'] + normRng.standard_normal(nBoot).reshape(-1, 1) * interceptStats['se'])
                yyBoot += np.ones((xx.shape[0], )) * interceptStats['coef']
            #
            if x in allRegressors:
                xStats = theseStatsDF.xs(x, level='names')
                assert xStats.shape[0] == 1
                xStats = xStats.iloc[0, :]
                yy += (xx - thisOffset) * xStats['coef']
                xCoefBoot = xStats['coef'] + normRng.standard_normal(nBoot).reshape(-1, 1) * xStats['se']
                yyBoot += (xx - thisOffset) * xCoefBoot
            if hue in allRegressors:
                hueStats = theseStatsDF.xs(hue, level='names')
                assert hueStats.shape[0] == 1
                hueStats = hueStats.iloc[0, :]
                yy += hueName * hueStats['coef']
                yyBoot += hueName * (hueStats['coef'] + normRng.standard_normal(nBoot).reshape(-1, 1) * hueStats['se'])
            if hueXInteraction in allRegressors:
                hueXInteractionStats = theseStatsDF.xs(hueXInteraction, level='names')
                assert hueXInteractionStats.shape[0] == 1
                hueXInteractionStats = hueXInteractionStats.iloc[0, :]
                yy += (xx - thisOffset) * hueName * hueXInteractionStats['coef']
                yyBoot += (xx - thisOffset) * hueName * (hueXInteractionStats['coef'] + normRng.standard_normal(nBoot).reshape(-1, 1) * hueXInteractionStats['se'])
                # yyBoot += (xx - thisOffset) * hueName * hueXInteractionStats['coef']
            yyUpper = np.percentile(yyBoot, 50 - ci/2, axis=0)
            yyLower = np.percentile(yyBoot, 50 + ci/2, axis=0)
            ax.fill_between(xx, yyLower, yyUpper, color=color, alpha=0.1, edgecolor='face', linewidth=1.)
            ax.plot(xx, yy, color=color, lw=1.5, ls='-')
            # print('color = {}'.format(color))
        return
    return overlayRegressionResults

def boxplotWrapper(
        data=None,
        x=None, order=None,
        y=None,
        hue=None, hue_order=None, palette=None,
        color=None, *args, **kwargs):
    ax = plt.gca()
    sns.boxplot(
        data=data, x=x, order=order, y=y,
        hue=hue, hue_order=hue_order, palette=palette,
        ax=ax,
        *args, **kwargs)
    return

def yTickLabelRemover(g, ro, co, hu, dataSubset):
    g.axes[ro, co].set_yticklabels([])
    return

def catplotWrapper(
        dataDF=None, statsDF=None,
        pdf=None, catKWArgs=None, facetKWArgs=None,
        catPlotKind=None, statAnnPlotType=None,
        pairsToAnnotate=None, renameLookup=None, gSuffix="",
        plotProcFuns=[], mapDFProcFuns=[]
        ):
    nHues = dataDF[catKWArgs['hue']].unique().shape[0]
    g = sns.catplot(
        data=dataDF,
        kind=catPlotKind, **catKWArgs,
        **facetKWArgs
        )
    for mpdf in mapDFProcFuns:
        mpdf_fun, mpdf_args, mpdf_kwargs = mpdf
        g.map_dataframe(mpdf_fun, *mpdf_args, **mpdf_kwargs)
    # ampStatsDF
    for (ro, co, hu), dataSubset in g.facet_data():
        if len(plotProcFuns):
            for procFun in plotProcFuns:
                procFun(g, ro, co, hu, dataSubset)
    g.set_axis_labels(
        renameLookup.pop(catKWArgs['x'], catKWArgs['x']),
        renameLookup.pop(catKWArgs['y'], catKWArgs['y']))
    if g.axes.shape[1] > 1:
        for ro in range(g.axes.shape[0]):
            for co in range(g.axes.shape[1]):
                if co != 0:
                    g.axes[ro, 0].get_shared_y_axes().join(g.axes[ro, 0], g.axes[ro, co])
                    g.axes[ro, co].set_ylabel(None)
                    g.axes[ro, co].set_yticklabels([])
    pvFormatter = PValueFormat()
    pvFormatter.config(text_format='simple')
    #
    annotatorPlotKWArgs = catKWArgs.copy()
    annotatorPlotKWArgs.update(facetKWArgs)
    popAnns = ['col', 'col_order', 'row', 'row_order']
    if nHues == 1:
        popAnns += ['hue', 'hue_order']
    for annN in popAnns:
        annotatorPlotKWArgs.pop(annN, None)
    annotator = Annotator(
        None, None)
    for (row_val, col_val), ax in g.axes_dict.items():
        #
        axTickLabels = [tl.get_text() for tl in ax.get_xticklabels()]
        if len(axTickLabels):
            newAxTickLabels = [
                (renameLookup[tl] if tl in renameLookup else tl)
                for tl in axTickLabels]
            ax.set_xticklabels(newAxTickLabels)
        axTickLabels = [tl.get_text() for tl in ax.get_yticklabels()]
        if len(axTickLabels):
            newAxTickLabels = [
                (renameLookup[tl] if tl in renameLookup else tl)
                for tl in axTickLabels]
            ax.set_yticklabels(newAxTickLabels)
        thesePvalAnns = []
        significantPairs = []
        if len(pairsToAnnotate):
            maskStatsThisFeature = (statsDF[catKWArgs['row']] == row_val) & (statsDF[catKWArgs['col']] == col_val)
            statsThisFeature = statsDF.loc[maskStatsThisFeature, :]
            thisAxMask = (dataDF[catKWArgs['col']] == col_val) & (dataDF[catKWArgs['row']] == row_val)
            dataThisAx = dataDF.loc[thisAxMask].copy()
            huesThisAx = dataThisAx[catKWArgs['hue']].unique()
        for tp in pairsToAnnotate:
            if nHues > 1:
                (x1, h1), (x2, h2) = tp
                assert h1 == h2
                pairMask = (statsThisFeature['A'] == x1) & (statsThisFeature['B'] == x2) & (statsThisFeature[catKWArgs['hue']] == h1)
            else:
                if isinstance(tp[0], tuple) or isinstance(tp[0], list):
                    # passed redundant information, remove hue
                    (x1, h1), (x2, h2) = tp
                    tp = (x1, x2)
                else:
                    x1, x2 = tp
                pairMask = (statsThisFeature['A'] == x1) & (statsThisFeature['B'] == x2)
            if pairMask.sum() == 0:
                continue
            else:
                assert pairMask.sum() == 1
            thisPV = StatResult(
                test_description='', test_short_name='', stat_str='T',
                stat=statsThisFeature.loc[pairMask, 'T'].iloc[0],
                pval=statsThisFeature.loc[pairMask, 'pval'].iloc[0],
                alpha=confidence_alpha)
            if thisPV.pvalue < confidence_alpha:
                significantPairs.append(tp)
                thesePvalAnns.append(gSuffix + ' = {:0.2f} ({})'.format(
                    statsThisFeature.loc[pairMask, 'hedges'].iloc[0], pvFormatter.format_data(thisPV)))
        if (len(significantPairs) > 0):
            if (nHues > 1) and (huesThisAx.shape[0] < nHues):
                for missingHue in catKWArgs['hue_order']:
                    if missingHue not in huesThisAx:
                        dummyMask = (dataDF[catKWArgs['hue']] == missingHue) & (dataDF[catKWArgs['row']] == row_val)
                        dummyData = dataDF.loc[dummyMask, :].copy()
                        dummyData.loc[:, catKWArgs['hue']] = missingHue
                        dataThisAx = pd.concat([
                            dummyData, dataThisAx]).sort_values([catKWArgs['hue'], catKWArgs['x']])
                #
            #
            annotatorPlotKWArgs['data'] = dataThisAx
            #
            annotator.new_plot(
                ax,
                pairs=significantPairs,
                plot=statAnnPlotType,
                **annotatorPlotKWArgs
            )
            annotator.configure(
                test=None, test_short_name='WT',
                )
            # print(significantPairs)
            # pdb.set_trace()
            annotator.annotate_custom_annotations(thesePvalAnns)
    asp.reformatFacetGridLegend(
        g, titleOverrides=renameLookup,
        contentOverrides={
            'NA_NA': 'No movement',
            'CW_outbound': 'Start of movement (extension)',
            'CW_return': 'Return to start (flexion)'
            },
        styleOpts=styleOpts)
    g.resize_legend(adjust_subtitles=True)
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0, )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
    return

with PdfPages(pdfPath) as pdf:
    plotLims = [
        plotRCPieces[whichRAUC].min(),
        plotRCPieces[whichRAUC].max()]
    if arguments['plotThePieces']:
        ##### nostim
        ################################################################################
        #
        noStimStatsForPlot = relativeStatsNoStimDF.copy().reset_index()
        pairsToAnnotate = [(("CW_outbound", "NA_0.0"), ("NA_NA", "NA_0.0"))]
        #
        rowVar = 'feature'
        rowOrder = [rN for rN in keepColsForPlot if rN in plotRCPieces[rowVar].unique()]
        colVar = 'electrode'
        colOrder = compoundAnnLookupDF.sort_values('trialRateInHz')['electrode'].dropna().unique().tolist()
        #
        xVar = 'kinematicCondition'
        xOrder = [xN for xN in compoundAnnLookupDF.sort_values(['pedalDirection', 'pedalMovementCat']).index if xN in refGroupPieces[xVar].unique()]
        hueVar = 'stimCondition'
        hueOrder = [
            iV
            for iV in compoundAnnLookupDF.sort_values(['trialRateInHz', 'electrode']).index
            if iV in np.unique(refGroupPieces[hueVar])]
        #
        huePalette = {hN: tuple(list(stimConditionPalette[hN]) + [1.]) for hN in hueOrder}
        huePaletteAlpha = {hN: tuple(list(hV)[:3] + [0.5]) for hN, hV in huePalette.items()}
        height, width = 2, 3
        aspect = width / height
        #
        violinKWArgs = dict(cut=0., split=False)
        boxKWArgs = dict(whis=np.inf)
        stripKWArgs = dict(size=2.5, dodge=True, jitter=0.2)
        facetKWArgs = stripKWArgs
        statAnnPlotType = 'boxplot'
        catPlotKind = 'strip'
        catKWArgs = dict(
            x=xVar, order=xOrder,
            y=whichRAUC,
            hue=hueVar, hue_order=hueOrder, palette=huePalette,
            col=colVar,
            col_order=[
                colName for colName in colOrder if colName in refGroupPieces[colVar].unique()],
            row=rowVar, row_order=rowOrder,
            height=height, aspect=aspect,
            sharey=True, sharex=True,
            facet_kws=dict(margin_titles=True),
        )
        renameLookup = prettyNameLookup.copy()
        renameLookup['NA_NA'] = 'No movement'
        plotProcFuns = [
            yTickLabelRemover,
            asp.genTitleChanger(renameLookup)]
        boxPalette = {hN: (1, 1, 1, 1) for hN, hV in huePalette.items()}
        mapDFProcFuns = [
            (boxplotWrapper, [], dict(
                x=xVar, order=xOrder, y=whichRAUC,
                hue=hueVar, hue_order=hueOrder,
                palette=boxPalette, zorder=1.9, whis=np.inf)),
            ]
        catplotWrapper(
            dataDF=refGroupPieces, statsDF=noStimStatsForPlot,
            pairsToAnnotate=pairsToAnnotate,
            pdf=pdf, catKWArgs=catKWArgs,
            catPlotKind=catPlotKind, statAnnPlotType=statAnnPlotType,
            facetKWArgs=facetKWArgs, renameLookup=renameLookup, gSuffix=r"$g_{M}$",
            plotProcFuns=plotProcFuns, mapDFProcFuns=mapDFProcFuns)
        ##### stim catplot
        ################################################################################
        #
        relativeStatsForPlot = relativeStatsDF.copy().reset_index()
        for cN in ['A', 'B', 'trialRateInHz']:
            relativeStatsForPlot.loc[:, cN] = relativeStatsForPlot[cN].apply(lambda x: '{}'.format(x))
        rowVar = 'feature'
        rowOrder = [rN for rN in keepColsForPlot if rN in plotRCPieces[rowVar].unique()]
        # rowOrder = sorted(np.unique(plotRCPieces[rowVar]))
        colWrap = min(3, len(colOrder))
        #
        height, width = 2, 3
        aspect = width / height
        noMoveStimPiecesDF = testGroupPieces.loc[testGroupPieces['kinematicCondition'] == 'NA_NA', :].copy()
        hueVar = 'trialRateInHz'
        hueOrder = [
            iV
            for iV in compoundAnnLookupDF['trialRateInHz'].dropna().drop_duplicates().sort_values()
            if iV in np.unique(noMoveStimPiecesDF[hueVar])]
        huePalette = {hN: tuple(list(ratePalette[hN]) + [1.]) for hN in hueOrder}
        # convert to strings to resolve annotations bug
        noMoveStimPiecesDF.loc[:, hueVar] = noMoveStimPiecesDF[hueVar].apply(lambda x: '{}'.format(x))
        hueOrder = ['{}'.format(hN) for hN in hueOrder]
        huePalette = {'{}'.format(hN): thisColor for hN, thisColor in huePalette.items()}
        #
        colVar = 'kinAndElecCondition'
        colOrder = [
            iV
            for iV in compoundAnnLookupDF.sort_values(['pedalMovementCat', 'pedalDirection', 'electrode']).index
            if iV in np.unique(noMoveStimPiecesDF[colVar])]
        xVar = amplitudeFieldName
        xOrder = sorted(noMoveStimPiecesDF[xVar].unique())
        noMoveStimPiecesDF.loc[:, xVar] = noMoveStimPiecesDF[xVar].apply(lambda x: '{}'.format(x))
        xOrder = ['{}'.format(xV) for xV in xOrder]
        catKWArgs.update(dict(
            x=xVar, order=xOrder,
            row=rowVar, row_order=rowOrder,
            height=height, aspect=aspect,
            col=colVar, col_order=colOrder,
            hue=hueVar, hue_order=hueOrder, palette=huePalette
            ))
        pairsToAnnotate = []
        for name, group in noMoveStimPiecesDF.groupby([colVar, hueVar]):
            colName, hueName = name
            minX, maxX = group[xVar].min(), group[xVar].max()
            thisPair = ((minX, hueName), (maxX, hueName))
            if thisPair not in pairsToAnnotate:
                pairsToAnnotate.append(thisPair)
        renameLookup = prettyNameLookup.copy()
        renameLookup['NA_NA'] = 'No movement'
        plotProcFuns = [
            asp.genTitleChanger(renameLookup)]
        boxPalette = {hN: (1, 1, 1, 1) for hN, hV in huePalette.items()}
        mapDFProcFuns = [
            (boxplotWrapper, [], dict(
                x=xVar, order=xOrder, y=whichRAUC,
                hue=hueVar, hue_order=hueOrder,
                palette=boxPalette, zorder=1.9, whis=np.inf)),
            ]
        catplotWrapper(
            dataDF=noMoveStimPiecesDF,
            statsDF=relativeStatsForPlot,
            pairsToAnnotate=pairsToAnnotate,
            pdf=pdf, catKWArgs=catKWArgs,
            catPlotKind=catPlotKind, statAnnPlotType=statAnnPlotType,
            facetKWArgs=facetKWArgs, renameLookup=renameLookup, gSuffix=r"$g_{S}$",
            plotProcFuns=plotProcFuns, mapDFProcFuns=mapDFProcFuns)
        #
        moveStimPiecesDF = testGroupPieces.loc[testGroupPieces['kinematicCondition'] != 'NA_NA', :]
        xVar = amplitudeFieldName
        xOrder = sorted(moveStimPiecesDF[xVar].unique())
        hueVar = 'trialRateInHz'
        hueOrder = [
            iV
            for iV in compoundAnnLookupDF['trialRateInHz'].dropna().drop_duplicates().sort_values()
            if iV in np.unique(moveStimPiecesDF[hueVar])]
        huePalette = {hN: tuple(list(ratePalette[hN]) + [1.]) for hN in hueOrder}
        # convert to strings to resolve annotations bug
        moveStimPiecesDF.loc[:, hueVar] = moveStimPiecesDF[hueVar].apply(lambda x: '{}'.format(x))
        hueOrder = ['{}'.format(hN) for hN in hueOrder]
        huePalette = {'{}'.format(hN): thisColor for hN, thisColor in huePalette.items()}
        moveStimPiecesDF.loc[:, xVar] = moveStimPiecesDF[xVar].apply(lambda x: '{}'.format(x))
        xOrder = ['{}'.format(xV) for xV in xOrder]
        colVar = 'kinAndElecCondition'
        colOrder = [
            iV
            for iV in compoundAnnLookupDF.sort_values(['pedalMovementCat', 'pedalDirection', 'electrode']).index
            if iV in np.unique(moveStimPiecesDF[colVar])]
        catKWArgs.update(dict(
            x=xVar, order=xOrder,
            row=rowVar, row_order=rowOrder,
            height=height, aspect=aspect,
            col=colVar, col_order=colOrder,
            hue=hueVar, hue_order=hueOrder, palette=huePalette
            ))
        pairsToAnnotate = []
        for name, group in moveStimPiecesDF.groupby([colVar, hueVar]):
            colName, hueName = name
            minX, maxX = group[xVar].min(), group[xVar].max()
            thisPair = ((minX, hueName), (maxX, hueName))
            if thisPair not in pairsToAnnotate:
                pairsToAnnotate.append(thisPair)
        renameLookup = prettyNameLookup.copy()
        renameLookup['NA_NA'] = 'No movement'
        plotProcFuns = [
            asp.genTitleChanger(renameLookup)]
        boxPalette = {hN: (1, 1, 1, 1) for hN, hV in huePalette.items()}
        mapDFProcFuns = [
            (boxplotWrapper, [], dict(
                x=xVar, order=xOrder, y=whichRAUC,
                hue=hueVar, hue_order=hueOrder,
                palette=boxPalette, zorder=1.9, whis=np.inf)),
            ]
        catplotWrapper(
            dataDF=moveStimPiecesDF, statsDF=relativeStatsForPlot,
            pairsToAnnotate=pairsToAnnotate,
            pdf=pdf, catKWArgs=catKWArgs,
            catPlotKind=catPlotKind, statAnnPlotType=statAnnPlotType,
            facetKWArgs=facetKWArgs, renameLookup=renameLookup, gSuffix=r"$g_{SM}$",
            plotProcFuns=plotProcFuns, mapDFProcFuns=mapDFProcFuns)
        #### dummy to get legend from
        xVar = amplitudeFieldName
        xOrder = sorted(plotRCPieces[xVar].unique())
        hueVar = 'trialRateInHz'
        hueOrder = [
            iV
            for iV in compoundAnnLookupDF['trialRateInHz'].dropna().drop_duplicates().sort_values()
            if iV in np.unique(plotRCPieces[hueVar])]
        huePalette = {hN: tuple(list(ratePalette[hN]) + [1.]) for hN in hueOrder}
        # convert to strings to resolve annotations bug
        plotRCPieces.loc[:, hueVar] = plotRCPieces[hueVar].apply(lambda x: '{}'.format(x))
        hueOrder = ['{}'.format(hN) for hN in hueOrder]
        huePalette = {'{}'.format(hN): thisColor for hN, thisColor in huePalette.items()}
        plotRCPieces.loc[:, xVar] = plotRCPieces[xVar].apply(lambda x: '{}'.format(x))
        xOrder = ['{}'.format(xV) for xV in xOrder]
        colVar = 'kinAndElecCondition'
        colOrder = [
            iV
            for iV in compoundAnnLookupDF.sort_values(['pedalMovementCat', 'pedalDirection', 'electrode']).index
            if iV in np.unique(plotRCPieces[colVar])]
        catKWArgs.update(dict(
            x=xVar, order=xOrder,
            row=rowVar, row_order=rowOrder,
            height=height, aspect=aspect,
            col=colVar, col_order=colOrder,
            hue=hueVar, hue_order=hueOrder, palette=huePalette
        ))
        catplotWrapper(
            dataDF=plotRCPieces, statsDF=None,
            pairsToAnnotate=[],
            pdf=pdf, catKWArgs=catKWArgs,
            catPlotKind=catPlotKind, statAnnPlotType=statAnnPlotType,
            facetKWArgs=facetKWArgs, renameLookup=renameLookup, gSuffix="",
            plotProcFuns=plotProcFuns)
        '''
        g = sns.catplot(ss
            data=testGroupPieces,
            kind=catPlotKind, **catKWArgs,
            **facetKWArgs
            )
        # ampStatsDF
        plotProcFuns = [
            asp.genTitleChanger(prettyNameLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        g.set_axis_labels('Stimulation amplitude (uA)', 'Normalized AUC')
        if g.axes.shape[1] > 1:
            for ro in range(g.axes.shape[0]):
                for co in range(g.axes.shape[1]):
                    if co != 0:
                        g.axes[ro, 0].get_shared_y_axes().join(g.axes[ro, 0], g.axes[ro, co])
                        g.axes[ro, co].set_ylabel(None)
                        g.axes[ro, co].set_yticklabels([])
        pvFormatter = PValueFormat()
        pvFormatter.config(text_format='simple')
        #
        annotatorPlotKWArgs = catKWArgs.copy()
        annotatorPlotKWArgs.update(facetKWArgs)
        for annN in ['col', 'col_order', 'row', 'row_order']:
            annotatorPlotKWArgs.pop(annN, None)
        annotator = Annotator(
            None, None)
        for (row_val, col_val), ax in g.axes_dict.items():
            thisAxMask = (testGroupPieces[colVar] == col_val) & (testGroupPieces[rowVar] == row_val)
            dataThisAx = testGroupPieces.loc[thisAxMask].copy()
            thesePvalAnns = []
            significantPairs = []
            thisElec, thisMT, thisDir = compoundAnnLookupDF.loc[col_val, ['electrode', 'pedalMovementCat', 'pedalDirection']].to_list()
            statsThisFeature = relativeStatsDF.xs(row_val, level=rowVar).xs(thisElec, level='electrode').xs('{}_{}'.format(thisDir, thisMT), level='kinematicCondition')
            nHues = statsThisFeature.groupby(hueVar).ngroups
            for thisRate, statsThisHue in statsThisFeature.groupby(hueVar):
                assert statsThisHue.shape[0] == 1
                thisPV = StatResult(
                    test_description='', test_short_name='', stat_str='T',
                    stat=statsThisHue['T'].iloc[0],
                    pval=statsThisHue['pval'].iloc[0],
                    alpha=confidence_alpha)
                if thisPV.pvalue < confidence_alpha:
                    thesePvalAnns.append(r"$g_{SM}$" + ' = {:0.2f} ({})'.format(
                        statsThisHue['hedges'].iloc[0], pvFormatter.format_data(thisPV)))
                    significantPairs.append(
                        [
                            (minAmp, thisRate), (maxAmp, thisRate)]
                        )
                    print('added pair {}'.format(significantPairs[-1]))
            if (len(significantPairs) > 0):
                for missingHue in hueOrder:
                    if missingHue not in huesThisAx:
                        dummyMask = (testGroupPieces[hueVar] == missingHue) & (testGroupPieces[rowVar] == row_val)
                        dummyData = testGroupPieces.loc[dummyMask, :].copy()
                        dummyData.loc[:, hueVar] = missingHue
                        dataThisAx = pd.concat([
                            dummyData, dataThisAx]).sort_values([hueVar, amplitudeFieldName])
                annotatorPlotKWArgs['data'] = dataThisAx
                annotator.new_plot(
                    ax, pairs=significantPairs, 
                    plot=statAnnPlotType,
                    **annotatorPlotKWArgs
                    )
                annotator.configure(
                    test=None, test_short_name='WT',
                    )
                annotator.annotate_custom_annotations(thesePvalAnns)
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
        '''
        ##### stim lmplot
        ################################################################################
        #
        rowVar = 'feature'
        rowOrder = [rN for rN in keepColsForPlot if rN in plotRCPieces[rowVar].unique()]
        # rowOrder = sorted(np.unique(plotRCPieces[rowVar]))
        colVar = 'kinAndElecCondition'
        colOrder = [
            iV
            for iV in compoundAnnLookupDF.sort_values(['pedalMovementCat', 'pedalDirection', 'electrode']).index
            if iV in np.unique(testGroupPieces[colVar])]
        hueVar = 'trialRateInHz'
        hueOrder = [
            iV
            for iV in compoundAnnLookupDF['trialRateInHz'].dropna().drop_duplicates().sort_values()
            if iV in np.unique(testGroupPieces[hueVar])]
        huePalette = {hN: ratePalette[hN] for hN in hueOrder}
        huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
        ##
        height, width = 2, 3
        aspect = width / height
        xJitter = testGroupPieces[xVar].diff().dropna().abs().unique().mean() / 20
        regPlotKWArgs = dict(
            x_jitter=xJitter,
            fit_reg=False
        )
        lmKWArgs = dict(
            col=colVar,
            col_order=colOrder,
            row=rowVar, row_order=rowOrder,
            x=xVar, y=whichRAUC,
            hue=hueVar, hue_order=hueOrder, palette=huePalette,
            height=height, aspect=aspect,
            facet_kws=dict(
                margin_titles=True,
                sharey=False, sharex=True,
                # gridspec_kws=dict(width_ratios=widthRatios)
                ),
            )
        jitteredPieces = testGroupPieces.copy()
        dodge = True
        if dodge:
            offsets = np.arange(len(hueOrder)) - ((len(hueOrder) - 1) / 2)
            for hueGrpIndex, thisOffset in enumerate(offsets):
                hueMask = (testGroupPieces[hueVar] == hueOrder[hueGrpIndex])
                jitteredPieces.loc[hueMask, xVar] = jitteredPieces.loc[hueMask, xVar] + 4 * xJitter * thisOffset
        g = sns.lmplot(
            data=jitteredPieces,
            **lmKWArgs, **regPlotKWArgs
            )
        #
        freshAmpStatsDF = pd.read_hdf(resultPath, 'amplitudeStats')
        g.map_dataframe(
            genRegressionResultsOverlay(
                freshAmpStatsDF, rowVar=rowVar, colVar=colVar, xOffset=xJitter),
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueVar, hue_order=hueOrder)
        plotProcFuns = [
            asp.genTitleChanger(prettyNameLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        g.set_axis_labels('Stimulation amplitude (uA)', 'Normalized AUC')
        if g.axes.shape[1] > 1:
            for ro in range(g.axes.shape[0]):
                for co in range(g.axes.shape[1]):
                    if co != 0:
                        g.axes[ro, 0].get_shared_y_axes().join(g.axes[ro, 0], g.axes[ro, co])
                        g.axes[ro, co].set_ylabel(None)
                        g.axes[ro, co].set_yticklabels([])
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
    #
    rowVar = 'feature'
    rowOrder = sorted(np.unique(plotRCPieces[rowVar]))
    colVar = 'electrode'
    colOrder = compoundAnnLookupDF.sort_values('trialRateInHz')['electrode'].dropna().unique().tolist()
    colWrap = min(3, len(colOrder))
    hueVar = 'kinematicCondition'
    # hueOrder = sorted(np.unique(plotRCPieces[hueVar]))
    hueOrder = kinematicOrder
    pal = sns.color_palette("Set2")
    huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
    huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
    height, width = 3, 1.5
    aspect = width / height
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
        for yVar in ['coefStd', 'coef']:
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
                saturation=0.8, linewidth=0.5
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
                takeThese.loc[:, 'kinAndElecCondition'] = '{}_{}'.format('NA', cN)
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
    if False and (relativeStatsDF.groupby(['xCoords', 'yCoords']).ngroups > 1): # histogram t stats
        rowVar = 'feature'
        rowOrder = sorted(np.unique(plotRCPieces[rowVar]))
        colVar = 'stimCondition'
        colOrder = sorted(np.unique(plotRCPieces[colVar]))
        colWrap = min(3, len(colOrder))
        hueVar = 'kinematicCondition'
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
            col=colVar,
            col_order=[
                cn
                for cn in colOrder
                if cn in plotRelativeStatsDF[colVar].to_list()],
            row=hueVar,
            row_order=[
                cn
                for cn in hueOrder
                if cn in plotRelativeStatsDF[hueVar].to_list()],
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
    if False and arguments['plotTheAverage']:
        colVar = 'stimCondition'
        colOrder = sorted(np.unique(plotRC[colVar]))
        hueVar = 'kinematicCondition'
        hueOrder = kinematicOrder
        pal = sns.color_palette("Set1")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        #
        rowVar = 'freqBandName'
        rowOrder = sorted(np.unique(plotRC[rowVar]))
        colWrap = min(3, len(colOrder))
        height, width = 3, 3
        aspect = width / height
        widthRatios = [3] * np.unique(testGroup[colVar]).shape[0] + [1]
        plotLimsMin = plotRC.groupby(rowVar).min()[whichRAUC]
        plotLimsMax = plotRC.groupby(rowVar).max()[whichRAUC]
        plotLimsRange = plotLimsMax - plotLimsMin
        plotLimsMin -= plotLimsRange / 100
        plotLimsMax += plotLimsRange / 100
        xJitter = testGroup[amplitudeFieldName].diff().dropna().abs().unique().mean() / 20
        g = sns.lmplot(
            col=colVar, col_order=colOrder,
            row=rowVar,
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueVar, hue_order=hueOrder, palette=huePalette,
            data=testGroup,
            ci=95, n_boot=100,
            x_jitter=xJitter,
            scatter_kws=dict(s=2.5),
            height=height, aspect=aspect,
            facet_kws=dict(
                sharey=False, sharex=False,
                margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)),
            )
        plotProcFuns = [
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
                sns.boxplot(
                    x=hueVar, order=hueOrder,
                    y=whichRAUC,
                    hue=hueVar, hue_order=hueOrder, palette=huePaletteAlpha,
                    data=refData, saturation=0.25,
                    ax=ax, whis=np.inf)
                # sns.stripplot(
                #     x=hueVar, order=hueOrder,
                #     y=whichRAUC, palette=huePalette,
                #     hue=hueVar, hue_order=hueOrder, data=refData,
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
