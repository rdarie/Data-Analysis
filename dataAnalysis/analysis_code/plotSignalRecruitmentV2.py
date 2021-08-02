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
"""
import logging
logging.captureWarnings(True)
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
import os
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler

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

rawRecCurve = pd.read_hdf(resultPath, 'raw')
rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
recCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
rauc = pd.read_hdf(resultPath, 'scaled')
rauc.columns = rauc.columns.get_level_values('feature')
recCurve.loc[:, 'rauc'] = rauc.stack().to_numpy()
relativeRaucDF = pd.read_hdf(resultPath, 'relative')
relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
recCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()


ampStatsDF = pd.read_hdf(resultPath, 'amplitudeStats')
relativeStatsDF = pd.read_hdf(resultPath, 'relativeStatsDF')
pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '{}_{}_{}.pdf'.format(
        inputBlockSuffix, arguments['window'],
        'RAUC'))

plotRC = recCurve.reset_index()
keepCols = ['segment', 'originalIndex', 'feature', 'lag', 'kinematicCondition'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)
refGroup = plotRC.loc[plotRC['electrode'] == 'NA', :]
testGroup = plotRC.loc[plotRC['electrode'] != 'NA', :]

averageRaucDF = relativeRaucDF.mean(axis='columns').to_frame(name='normalizedRAUC').reset_index()
averageRaucDF.loc[:, 'feature'] = 'averageFeature'
refGroupAverage = averageRaucDF.loc[averageRaucDF['electrode'] == 'NA', :]
testGroupAverage = averageRaucDF.loc[averageRaucDF['electrode'] != 'NA', :]

colName = 'electrode'
colOrder = sorted(np.unique(plotRC[colName]))
hueName = 'kinematicCondition'
hueOrder = sorted(np.unique(plotRC[hueName]))
pal = sns.color_palette("Set2")
huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
rowName = 'feature'
rowOrder = sorted(np.unique(plotRC[rowName]))
colWrap = min(3, len(colOrder))
height, width = 3, 3
aspect = width / height
whichRAUC = 'normalizedRAUC'
alpha = 1e-3
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
            for hn, group in dataSubset.groupby([hueName]):
                rn = group[g._row_var].unique()[0]
                cn = group[g._col_var].unique()[0]
                st = ampStatsDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hueName).xs('trialAmplitude', level='names')
                x = group[g._x_var].max()
                y = group.groupby(g._x_var).mean().loc[x, g._y_var]
                if st['pval'].iloc[0] < alpha:
                    g.axes[ro, co].text(x + dx, y + dy, '*', color=huePalette[hn], va='bottom', ha='left')
                rst = relativeStatsDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hueName)
                if rst['p-val'].iloc[0] < alpha:
                    g.axes[ro, co].text(x + dx, y + dy, '+', color=huePalette[hn], va='bottom', ha='right')
            g.axes[ro, co].starsAnnotated = True
    return

#
with PdfPages(pdfPath) as pdf:
    plotLims = plotRC[whichRAUC].quantile([0, 1-5e-3])
    if arguments['plotThePieces']:
        g = sns.relplot(
            col=colName,
            col_order=colOrder,
            row=rowName,
            x=amplitudeFieldName,
            y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            kind='line', data=testGroup,
            height=height, aspect=aspect, errorbar='sd', estimator='mean',
            facet_kws=dict(sharey=True, sharex=False, margin_titles=True), lw=1,
            )
        plotProcFuns = [statsAnnotator]
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
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        g.axes[0, 0].set_ylim(plotLims)
        pdf.savefig(
            bbox_inches='tight',
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        ################################
    g = sns.relplot(
        col=colName,
        col_order=colOrder,
        row=rowName,
        x=amplitudeFieldName,
        y=whichRAUC,
        hue=hueName, hue_order=hueOrder, palette=huePalette,
        kind='line', data=testGroupAverage,
        height=height, aspect=aspect, errorbar='sd', estimator='mean',
        facet_kws=dict(sharey=True, sharex=False, margin_titles=True), lw=1,
        )
    plotProcFuns = []
    for (ro, co, hu), dataSubset in g.facet_data():
        if len(plotProcFuns):
            for procFun in plotProcFuns:
                procFun(g, ro, co, hu, dataSubset)
    for (row_val, col_val), ax in g.axes_dict.items():
        if col_val == 'NA':
            refMask = (refGroupAverage[rowName] == row_val)
            if refMask.any():
                refData = refGroupAverage.loc[refMask, :]
            else:
                refData = refGroupAverage
            sns.violinplot(
                x=amplitudeFieldName,
                y=whichRAUC, palette=huePalette,
                hue=hueName, hue_order=hueOrder, data=refData,
                cut=0, inner='box',
                ax=ax)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    g.axes[0, 0].set_ylim(plotLims)
    g.suptitle('Averaged across all signals')
    pdf.savefig(
        bbox_inches='tight',
        )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()