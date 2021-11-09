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
        'lines.markersize': 2.4,
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

sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})

blockBaseName = arguments['inputBlockPrefix']
listOfExpNames = [x.strip() for x in arguments['expList'].split(',')]
listOfSelectionNames = [x.strip() for x in arguments['selectionList'].split(',')]
recCurveList = []
ampStatsDict = {}
relativeStatsDict = {}
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
        rawRecCurve = pd.read_hdf(resultPath, 'raw')
        thisRecCurveFeatureInfo = rawRecCurve.columns.to_frame().reset_index(drop=True)
        rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
        thisRecCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
        scaledRaucDF = pd.read_hdf(resultPath, 'scaled')
        scaledRaucDF.columns = scaledRaucDF.columns.get_level_values('feature')
        thisRecCurve.loc[:, 'scaledRAUC'] = scaledRaucDF.stack().to_numpy()
        # relativeRaucDF = pd.read_hdf(resultPath, 'relative')
        # relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
        # thisRecCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()
        clippedRaucDF = pd.read_hdf(resultPath, 'clipped')
        clippedRaucDF.columns = clippedRaucDF.columns.get_level_values('feature')
        thisRecCurve.loc[:, 'clippedRAUC'] = clippedRaucDF.stack().to_numpy()
        #
        thisRecCurve.loc[:, 'freqBandName'] = thisRecCurve.index.get_level_values('feature').map(thisRecCurveFeatureInfo[['feature', 'freqBandName']].set_index('feature')['freqBandName'])
        thisRecCurve.set_index('freqBandName', append=True, inplace=True)
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
        ampStatsPerFBDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'amplitudeStatsPerFreqBand')
        ampStatsPerFBDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
        relativeStatsPerFBDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'relativeStatsDFPerFreqBand')
        relativeStatsPerFBDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
        compoundAnnLookupList.append(pd.read_hdf(resultPath, 'compoundAnnLookup'))
        featureInfoList.append(thisRecCurveFeatureInfo)
#
compoundAnnLookupDF = pd.concat(compoundAnnLookupList).drop_duplicates()
recCurveFeatureInfo = pd.concat(featureInfoList).drop_duplicates()
#
recCurve = pd.concat(recCurveList)
del recCurveList
ampStatsDF = pd.concat(ampStatsDict, names=['expName', 'selectionName'])
del ampStatsDict
relativeStatsDF = pd.concat(relativeStatsDict, names=['expName', 'selectionName'])
#
# for cN in ['electrode', 'trialRateInHz']:
#     relativeStatsDF.loc[:, cN] = relativeStatsDF.index.get_level_values('stimCondition').map(compoundAnnLookupDF[cN])
#     relativeStatsDF.set_index(cN, append=True, inplace=True)
#
del relativeStatsDict
relativeStatsPerFBDF = pd.concat(relativeStatsPerFBDict, names=['expName', 'selectionName'])
del relativeStatsPerFBDict
ampStatsPerFBDF = pd.concat(ampStatsPerFBDict, names=['expName', 'selectionName'])
del ampStatsPerFBDict
whichRAUC = 'rawRAUC'
# whichRAUC = 'normalizedRAUC'
# whichRAUC = 'scaledRAUC'
#
figureOutputFolder = os.path.join(
    remoteBasePath, 'figures', 'lfp_recruitment')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)

pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '_{}_{}_{}_{}.pdf'.format(
        expDateTimePathStr, inputBlockSuffix, arguments['window'],
        'RAUC'))
plotRC = recCurve.reset_index()
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

confidence_alpha = 1e-3
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
                    # if rst['pval'].iloc[0] < confidence_alpha:
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
with PdfPages(pdfPath) as pdf:
    # plotLims = [0, plotRC[whichRAUC].quantile(1-1e-2)]
    '''plotLims = [
        plotRCPieces[whichRAUC].quantile(1e-2),
        plotRCPieces[whichRAUC].quantile(1-1e-2)]'''
    #
    plotLims = [
        plotRCPieces[whichRAUC].min(),
        plotRCPieces[whichRAUC].max()]
    colName = 'stimCondition'
    colOrder = sorted(np.unique(plotRC[colName]))
    hueName = 'kinematicCondition'
    hueOrder = sorted(np.unique(plotRC[hueName]))
    pal = sns.color_palette("Set2")
    huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
    huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
    rowName = 'feature'
    rowOrder = sorted(np.unique(plotRC[rowName]))
    colWrap = min(3, len(colOrder))
    height, width = 3, 3
    aspect = width / height
    if arguments['plotThePieces']:
        ####
        widthRatios = [3] * np.unique(testGroupPieces[colName]).shape[0] + [1]
        '''g = sns.relplot(
            col=colName, col_order=colOrder,
            row=rowName,
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            data=testGroupPieces,
            height=height, aspect=aspect,
            # kind='line',
            # errorbar='sd', estimator='mean', lw=1,
            kind='scatter',
            facet_kws=dict(
                sharey=False, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)),
            )'''
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
            asp.genTitleChanger(titleLabelLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA_0.0':
                refMask = (refGroupPieces[rowName] == row_val)
                if refMask.any():
                    refData = refGroupPieces.loc[refMask, :]
                else:
                    refData = refGroupPieces
                sns.violinplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC, palette=huePaletteAlpha,
                    data=refData, linewidth=0.,
                    cut=0, inner=None, # saturation=0.25,
                    ax=ax)
                sns.swarmplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC, palette=huePalette,
                    hue=hueName, hue_order=hueOrder, data=refData,
                    ax=ax, size=2.5)
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
        pdf.savefig(bbox_inches='tight', )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
    if relativeStatsDF.groupby(['xCoords', 'yCoords']).ngroups > 1: # histogram t stats
        ################################
        plotRelativeStatsDF = relativeStatsDF.reset_index()
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
            bbox_inches='tight',
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        #############
        plotAmpStatsDF = ampStatsDF.reset_index()
        plotAmpStatsDF.loc[:, 'namesAndMD'] = plotAmpStatsDF['names']
        plotAmpStatsDF.loc[plotAmpStatsDF['isMahalDist'], 'namesAndMD'] += '_md'
        thisFreqBandOrder = [fN for fN in freqBandOrderExtended if fN in plotAmpStatsDF['freqBandName'].unique().tolist()]
        # thisMaskAmp = (plotAmpStatsDF['reject']) & (plotAmpStatsDF['kinematicCondition'] != 'NA_NA') & (plotAmpStatsDF['names'] != 'Intercept')
        thisMaskAmp = (plotAmpStatsDF['kinematicCondition'] != 'NA_NA') & (plotAmpStatsDF['names'] != 'Intercept')
        # thisMaskAmp = plotAmpStatsDF['kinematicCondition'] != 'NA_NA'+
        thisPalette = pd.Series(sns.color_palette('Set1', 4), index=['trialAmplitude', 'trialAmplitude_md', 'trialAmplitude:trialRateInHz', 'trialAmplitude:trialRateInHz_md'])
        g = sns.catplot(
            y='coef',
            x='freqBandName',
            order=thisFreqBandOrder,
            row='kinematicCondition', col='electrode',
            hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(), color='w',
            data=plotAmpStatsDF.loc[thisMaskAmp, :],  #
            kind='box', whis=np.inf, saturation=0.)
        for name, ax in g.axes_dict.items():
            kinName, elecName = name
            subSetMask = thisMaskAmp & (plotAmpStatsDF['kinematicCondition'] == kinName) & (plotAmpStatsDF['electrode'] == elecName) & (~plotAmpStatsDF['reject'])
            sns.stripplot(
                data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                y='coef', x='freqBandName',
                order=thisFreqBandOrder, hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                dodge=True, linewidth=0.5, alpha=0.2)
            subSetMask = thisMaskAmp & (plotAmpStatsDF['kinematicCondition'] == kinName) & (plotAmpStatsDF['electrode'] == elecName) & plotAmpStatsDF['reject']
            sns.stripplot(
                data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                y='coef', x='freqBandName',
                order=thisFreqBandOrder, hue='namesAndMD', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                dodge=True, linewidth=0.5)
            ax.get_legend().remove()
        g.suptitle('Coefficient distribution for AUC regression')
        pdf.savefig(
            bbox_inches='tight',
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        ######
        plotRelativeStatsDF = relativeStatsDF.reset_index()
        plotRelativeStatsDF.loc[:, 'trialRateInHzStr'] = plotRelativeStatsDF['trialRateInHz'].apply(lambda x: '{}'.format(x))
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isMahalDist'], 'trialRateInHzStr'] += '_md'
        thisFreqBandOrder = [fN for fN in freqBandOrderExtended if fN in plotRelativeStatsDF['freqBandName'].unique().tolist()]
        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(x['feature'].replace('#0', '')), axis='columns')
        # thisMaskRel = (plotRelativeStatsDF['reject']) & (plotRelativeStatsDF['kinematicCondition'] != 'NA_NA')
        thisMaskRel = (plotRelativeStatsDF['kinematicCondition'] != 'NA_NA')
        thisPalette = pd.Series(sns.color_palette('Set1', 4), index=['50.0', '50.0_md', '100.0', '100.0_md'])
        g = sns.catplot(
            y='hedges',
            x='freqBandName',
            order=thisFreqBandOrder,
            row='kinematicCondition', col='electrode',
            hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
            data=plotRelativeStatsDF.loc[thisMaskRel, :],  #
            kind='box', whis=np.inf, saturation=0.)
        for name, ax in g.axes_dict.items():
            kinName, elecName = name
            subSetMask = thisMaskRel & (plotRelativeStatsDF['kinematicCondition'] == kinName) & (plotRelativeStatsDF['electrode'] == elecName) & (~plotRelativeStatsDF['reject'])
            sns.stripplot(
                data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                y='hedges', x='freqBandName',
                order=thisFreqBandOrder, hue='trialRateInHzStr',
                hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                dodge=True, alpha=0.2)
            subSetMask = thisMaskRel & (plotRelativeStatsDF['kinematicCondition'] == kinName) & (plotRelativeStatsDF['electrode'] == elecName) & plotRelativeStatsDF['reject']
            sns.stripplot(
                data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                y='hedges', x='freqBandName',
                order=thisFreqBandOrder, hue='trialRateInHzStr',
                hue_order=thisPalette.index.to_list(), palette=thisPalette.to_dict(),
                dodge=True, linewidth=0.5)
            ax.get_legend().remove()
        g.suptitle('Effect size distribution for stim vs no-stim comparisons')
        pdf.savefig(
            bbox_inches='tight',
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        ####
        statPalettes = [
            sns.diverging_palette(220, 20, as_cmap=True),
            sns.diverging_palette(145, 300, s=60, as_cmap=True)
            ]
        for name, statsThisFB in plotRelativeStatsDF.groupby(['freqBandName', 'isMahalDist']):
            freqBandName, isMahalDist = name
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
        hueOrder = sorted(np.unique(plotRC[hueName]))
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        #
        rowName = 'freqBandName'
        rowOrder = sorted(np.unique(plotRC[rowName]))
        colWrap = min(3, len(colOrder))
        height, width = 3, 3
        aspect = width / height
        confidence_alpha = 1e-2
        widthRatios = [4] * np.unique(testGroup[colName]).shape[0] + [1]
        '''
        g = sns.relplot(
            col=colName, col_order=colOrder,
            row=rowName,
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
            asp.genTitleChanger(titleLabelLookup)]
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
                sns.violinplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC, palette=huePalette,
                    data=refData, linewidth=0.,
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
