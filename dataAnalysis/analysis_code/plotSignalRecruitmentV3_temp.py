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
        'lines.linewidth': 1,
        'lines.markersize': 1.2,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
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
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
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
rawRecCurve = pd.read_hdf(resultPath, 'raw')
recCurveFeatureInfo = rawRecCurve.columns.to_frame().reset_index(drop=True)
rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
recCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
rauc = pd.read_hdf(resultPath, 'scaled')
rauc.columns = rauc.columns.get_level_values('feature')
recCurve.loc[:, 'rauc'] = rauc.stack().to_numpy()
relativeRaucDF = pd.read_hdf(resultPath, 'relative')
relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
recCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()

whichRAUC = 'normalizedRAUC'
# whichRAUC = 'rauc'

ampStatsDF = pd.read_hdf(resultPath, 'amplitudeStats')
relativeStatsDF = pd.read_hdf(resultPath, 'relativeStatsDF')
ampStatsPerFBDF = pd.read_hdf(resultPath, 'amplitudeStatsPerFreqBand')
relativeStatsPerFBDF = pd.read_hdf(resultPath, 'relativeStatsDFPerFreqBand')
pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '{}_{}_{}.pdf'.format(
        inputBlockSuffix, arguments['window'],
        'RAUC'))


recCurve.loc[:, 'freqBandName'] = recCurve.index.get_level_values('feature').map(recCurveFeatureInfo[['feature', 'freqBandName']].set_index('feature')['freqBandName'])
recCurve.set_index('freqBandName', append=True, inplace=True)
plotRC = recCurve.reset_index()
keepCols = ['segment', 'originalIndex', 't', 'feature', 'freqBandName', 'lag', 'kinematicCondition'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)
######
relativeStatsDF.loc[:, 'T_abs'] = relativeStatsDF['T'].abs()
nFeats = plotRC['feature'].unique().shape[0]
nFeatsToPlot = max(min(2, int(np.floor(nFeats/2))), 1)
keepTopIdx = (
    [i for i in range(nFeatsToPlot)] +
    [i for i in range(-1 * nFeatsToPlot, 0)]
    )
# [i for i in range(-1 * (1 + 1), 0)]
# [i for i in range(nFeatsToPlot)]
statsRankingDF = relativeStatsDF.groupby('feature').mean().sort_values('T_abs')
statsRankingPrintout = relativeStatsDF.groupby('feature').mean().sort_values('T_abs', ascending=False).head(16)
print(
    'top 16 features, ordered by absolute T statistic of maxp amp vs baseline:\n{}'.format(
        statsRankingPrintout))
print(', '.join(["'{}#0'".format(fN) for fN in statsRankingPrintout.index]))
# pdb.set_trace()
keepCols = (
    pd.Series(
        statsRankingDF
        .iloc[keepTopIdx, :]
        .index.get_level_values('feature'))
    .drop_duplicates()
    )
# pdb.set_trace() # relativeStatsDF.groupby('kinematicCondition').mean()
refGroup = plotRC.loc[plotRC['electrode'] == 'NA', :]
testGroup = plotRC.loc[plotRC['electrode'] != 'NA', :]
#
print('Plotting select features:\n{}'.format(statsRankingDF.loc[keepCols, :]))
plotRCPieces = plotRC.loc[plotRC['feature'].isin(keepCols), :].copy()
######
refGroupPieces = plotRCPieces.loc[plotRCPieces['electrode'] == 'NA', :]
testGroupPieces = plotRCPieces.loc[plotRCPieces['electrode'] != 'NA', :]
#
try:
    recCurveFeatureInfo.loc[:, 'xIdx'], recCurveFeatureInfo.loc[:, 'yIdx'] = ssplt.coordsToIndices(
        recCurveFeatureInfo['xCoords'], recCurveFeatureInfo['yCoords'])
except Exception:
    traceback.print_exc()
    recCurveFeatureInfo.loc[:, 'xIdx'] = 0.
    recCurveFeatureInfo.loc[:, 'yIdx'] = 0.
    recCurveFeatureInfo.loc[:, 'xCoords'] = 0.
    recCurveFeatureInfo.loc[:, 'yCoords'] = 0.
for cN in ['xCoords', 'yCoords', 'xIdx', 'yIdx', 'freqBandName']:
    statsRankingDF.loc[:, cN] = statsRankingDF.index.map(recCurveFeatureInfo[['feature', cN]].set_index('feature')[cN])
########
# plotRC = plotRC.loc[plotRC['feature'].str.contains('_all'), :]

alpha = 1e-3
def genStatsAnnotator(ampDF, relDF, hN, hP):
    def statsAnnotator(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
        if not emptySubset:
            if not hasattr(g.axes[ro, co], 'starsAnnotated'):
                xLim = g.axes[ro, co].get_xlim()
                yLim = g.axes[ro, co].get_ylim()
                dx = (xLim[1] - xLim[0]) / 20
                dy = (yLim[1] - yLim[0]) / 25
                for hn, group in dataSubset.groupby([hN]):
                    rn = group[g._row_var].unique()[0]
                    cn = group[g._col_var].unique()[0]
                    st = ampDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hN).xs('trialAmplitude', level='names')
                    x = group[g._x_var].max()
                    y = group.groupby(g._x_var).mean().loc[x, g._y_var]
                    if st['pval'].iloc[0] < alpha:
                        g.axes[ro, co].text(x + dx, y + dy, '*', color=hP[hn], va='bottom', ha='left')
                    rst = relDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hN)
                    if rst['p-val'].iloc[0] < alpha:
                        g.axes[ro, co].text(x + dx, y + dy, '+', color=hP[hn], va='bottom', ha='right')
                if (ro == 0) and (co == 0):
                    g.axes[ro, co].text(
                        0.95, 0.95,
                        '+: highest amp. stim. vs baseline (p-val < 1e-3)\n*: linear regression of amplitude vs auc (p-val < 1e-3)',
                        va='top', ha='right', transform=g.axes[ro, co].transAxes)
                g.axes[ro, co].starsAnnotated = True
        return
    return statsAnnotator
#
titleLabelLookup = {
    'electrode = + E16 - E5': 'Stimulation (+ E16 - E5)',
    'electrode = NA': 'No stimulation',
    'feature = mahal_ledoit_all': 'Mahalanobis distance (all bands)',
    'feature = mahal_ledoit_alpha': 'Mahalanobis distance (alpha band)',
    'feature = mahal_ledoit_beta': 'Mahalanobis distance (beta band)',
    'feature = mahal_ledoit_gamma': 'Mahalanobis distance (gamma band)',
    'feature = mahal_ledoit_higamma': 'Mahalanobis distance (high gamma band)',
    'feature = mahal_ledoit_spb': 'Mahalanobis distance (spike power band)',
    }
if True:
    # plotLims = [0, plotRC[whichRAUC].quantile(1-1e-2)]
    plotLims = [
        plotRCPieces[whichRAUC].quantile(1e-3),
        plotRCPieces[whichRAUC].quantile(1 - 1e-3)]
    if arguments['plotThePieces']:
        colName = 'electrode'
        colOrder = sorted(np.unique(plotRC[colName]))
        hueName = 'kinematicCondition'
        hueOrder = sorted(np.unique(plotRC[hueName]))
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        #
        rowName = 'feature'
        rowOrder = sorted(np.unique(plotRC[rowName]))
        colWrap = min(3, len(colOrder))
        height, width = 3, 3
        aspect = width / height
        alpha = 1e-3
        ####
        widthRatios = [2] * np.unique(testGroupPieces[colName]).shape[0] + [1]
        g = sns.relplot(
            col=colName,
            col_order=colOrder,
            row=rowName,
            x=amplitudeFieldName,
            y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            kind='line', data=testGroupPieces,
            height=height, aspect=aspect, errorbar='sd', estimator='mean',
            facet_kws=dict(
                sharey=True, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)), lw=1,
            )
        plotProcFuns = [genStatsAnnotator(ampStatsDF, relativeStatsDF, hueName, huePalette), asp.genTitleChanger(titleLabelLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA':
                refMask = (refGroupPieces[rowName] == row_val)
                if refMask.any():
                    refData = refGroupPieces.loc[refMask, :]
                else:
                    refData = refGroupPieces
                sns.violinplot(
                    x=hueName,
                    y=whichRAUC, palette=huePalette,
                    order=hueOrder, data=refData,
                    cut=0, inner=None, saturation=0.5,
                    ax=ax)
                sns.swarmplot(
                    x=hueName, hue=hueName,
                    y=whichRAUC, palette=huePalette,
                    order=hueOrder, hue_order=hueOrder, data=refData,
                    size=1.5, ax=ax)
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
        # pdf.savefig(
        #     bbox_inches='tight',
        #     )
        plt.show()
        # if arguments['showFigures']:
        #     plt.show()
        # else:
        #     plt.close()
        ################################
        plotRelativeStatsDF = relativeStatsDF.reset_index()
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
            kind='hist', element='step',
            height=2 * height, aspect=aspect
            )
        g.suptitle('T-statistic distribution')
        pdf.savefig(
            bbox_inches='tight',
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
    if True: # topo view t statistics
        for name, statsThisFB in relativeStatsDF.groupby(['freqBandName', 'electrode', 'kinematicCondition']): break
            freqBandName, electrode, kinematicCondition = name
            try:
                stats2D = statsThisFB.reset_index().pivot(index='yCoords', columns='xCoords', values='T')
                fig, ax = plt.subplots(
                1, 2, figsize=(3, 2),
                gridspec_kw={
                    'width_ratios': [10, 1],
                    'wspace': 0.1})
                ax = sns.heatmap(
                    data=stats2D,
                    cmap=sns.color_palette("vlag", as_cmap=True),
                    center=0.,
                    linewidths=0, ax=ax[0], cbar_ax=ax[1])
                figTitle = fig.suptitle('T-statistic (freq. band: {})'.format(freqBandName))
                fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                pdf.savefig(
                    bbox_inches='tight', pad_inches=0, bbox_extra_artists=[figTitle])
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            except Exception:
                pass
    if arguments['plotTheAverage']:
        colName = 'electrode'
        colOrder = sorted(np.unique(plotRC[colName]))
        hueName = 'kinematicCondition'
        hueOrder = sorted(np.unique(plotRC[hueName]))
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        #
        rowName = 'freqBandName'
        rowOrder = sorted(np.unique(plotRC[rowName]))
        colWrap = min(3, len(colOrder))
        height, width = 3, 3
        aspect = width / height
        alpha = 1e-3
        widthRatios = [4] * np.unique(testGroupPieces[colName]).shape[0] + [1]
        g = sns.relplot(
            col=colName,
            col_order=colOrder,
            row=rowName,
            x=amplitudeFieldName,
            y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            kind='line', data=testGroup,
            height=height, aspect=aspect, errorbar='sd', estimator='mean',
            facet_kws=dict(
                sharey=True, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)), lw=1,
        )
        plotProcFuns = [genStatsAnnotator(ampStatsPerFBDF, relativeStatsPerFBDF, hueName, huePalette), asp.genTitleChanger(titleLabelLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA':
                refMask = (refGroup[rowName] == row_val)
                if refMask.any():
                    refData = refGroup.loc[refMask, :]
                else:
                    refData = refGroup
                sns.violinplot(
                    x=amplitudeFieldName,
                    y=whichRAUC, palette=huePalette,
                    hue=hueName, hue_order=hueOrder, data=refData,
                    cut=0, inner='box',
                    ax=ax)
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
