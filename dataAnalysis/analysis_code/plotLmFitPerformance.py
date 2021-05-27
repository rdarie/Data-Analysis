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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import seaborn as sns
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
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
ecapPath = os.path.join(
    calcSubFolder, 'lmfit',
    prefix + '_{}_{}_lmfit_signals_CAR.parquet'.format(
        arguments['inputBlockSuffix'], arguments['window']))
print('plotLmFitPerformance loading {}...'.format(ecapPath))
rawEcapDF = pd.read_parquet(ecapPath, engine='fastparquet')
rawEcapDF.loc[:, 'nominalCurrent'] = rawEcapDF['nominalCurrent'] * (-1)
# simplify electrode names
rawEcapDF.loc[:, 'electrode'] = rawEcapDF['electrode'].apply(lambda x: x[1:])
#
if RCPlotOpts['rejectFeatures'] is not None:
    rejectDataMask = rawEcapDF['feature'].isin(RCPlotOpts['rejectFeatures'])
    rawEcapDF = rawEcapDF.loc[~rejectDataMask, :]
#
removeStimOnRec = True
if removeStimOnRec:
    ecapRmMask = (rawEcapDF['electrode'] == rawEcapDF['feature'])
    rawEcapDF.drop(index=rawEcapDF.index[ecapRmMask], inplace=True)
#
###############################################################################
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
# name lookup
lfpNL = mapDF.loc[mapAMask, :].set_index('topoName')
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
###############################################################################
#
trialMetaNames = [
    'segment', 'originalIndex', 't',
    'RateInHz',
    'electrode', amplitudeFieldName]
annotNames = ['xcoords', 'ycoords', 'whichArray']
featureMetaNames = annotNames + ['regrID', 'feature']
keepCols = trialMetaNames + ['feature']
for colName in keepCols:
    if colName not in rawEcapDF.columns:
        rawEcapDF.loc[:, colName] = 0.
plotDF = rawEcapDF.copy()
plotDF.loc[:, 'columnLabel'] = 'NA'
plotDF.loc[plotDF['regrID'].isin(['target', 'exp_']), 'columnLabel'] = 'targets'
plotDF.loc[plotDF['regrID'].isin(['exp_resid_CAR', 'exp_resid_mean']), 'columnLabel'] = 'CAR'
plotDF.loc[plotDF['regrID'].isin(['target_CAR', 'target_mean']), 'columnLabel'] = 'target_CAR'
plotDF.loc[plotDF['regrID'].isin(['exp_resid']), 'columnLabel'] = 'components'
plotDF.drop(index=plotDF.index[plotDF['columnLabel'] == 'NA'], inplace=True)
#
plotDF.loc[:, 'rowLabel'] = (
    plotDF['electrode'].astype(str) +
    ': ' +
    plotDF[amplitudeFieldName].astype(str))
relplotKWArgs.pop('palette', None)
###########################
timeScales = ['10', '40']
plotDF.set_index(['regrID', 'feature', 'columnLabel', 'rowLabel'] + annotNames + trialMetaNames, inplace=True)
plotDF.columns = plotDF.columns.astype(float)
relplotKWArgs['height'] = 4
relplotKWArgs['aspect'] = 1.5
colOrder = ['targets', 'components', 'CAR']
plotProcFuns = [asp.genYLimSetter(quantileLims=0.99, forceLims=True)]
if True:
    for timeScale in timeScales:
        pdfPath = os.path.join(
            figureOutputFolder,
            prefix + '_{}_{}_lmfit_{}_msec.pdf'.format(
                arguments['inputBlockSuffix'], arguments['window'], timeScale))
        pageCount = 0
        with PdfPages(pdfPath) as pdf:
            # for pageName, group in tqdm(plotDF.groupby('feature')):
            for pageName, group in tqdm(plotDF.groupby(['rowLabel', 'columnLabel'])):
                print('Plotting {} at time scale {}'.format(pageName, timeScale))
                plotGroup = group.stack().to_frame(name='signal').reset_index()
                if True:
                    # group by rowLabel, columnLabel
                    g = sns.relplot(
                        # data=plotGroup,
                        data=plotGroup.query('(bin < {}e-3) & (bin >= 1.3e-3)'.format(timeScale)),
                        x='bin', y='signal',
                        hue='regrID',
                        row='ycoords', col='xcoords',
                        palette='Set1', lw=2,
                        **relplotKWArgs)
                else:
                    g = sns.relplot(
                        # data=plotGroup,
                        data=plotGroup.query('(bin < {}e-3) & (bin >= 1.3e-3)'.format(timeScale)),
                        x='bin', y='signal',
                        hue='regrID',
                        row='rowLabel', col='columnLabel',
                        facet_kws=dict(sharey=False), palette='Set1',
                        lw=2, col_order=colOrder,
                        **relplotKWArgs)
                for (ro, co, hu), dataSubset in g.facet_data():
                    emptySubset = (
                            (dataSubset.empty) or
                            (dataSubset['signal'].isna().all()))
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                pageTitle = g.fig.suptitle(pageName)
                saveLegendOpts = {
                        'bbox_extra_artists': [pageTitle]}
                # contrived way of pushing legend outside without
                # resizing the figure
                allLegends = [
                    a.get_legend()
                    for a in g.axes.flat
                    if a.get_legend() is not None] + [g._legend]
                if len(allLegends):
                    # bb = matplotlib.transforms.Bbox([[-1.01, 0.01], [-0.01, 1.01]])
                    # allLegends[0].set_bbox_to_anchor(bb)
                    saveLegendOpts.update({
                        'bbox_extra_artists': [pageTitle] + allLegends})
                g.fig.set_size_inches(
                    g._ncol * relplotKWArgs['height'] * relplotKWArgs['aspect'] + 10,
                    g._nrow * relplotKWArgs['height'] + 2)
                pdf.savefig(bbox_inches='tight', pad_inches=0, **saveLegendOpts)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                pageCount += 1
                if limitPages is not None:
                    if pageCount > limitPages:
                        break
paletteLookup = {
    'exp_resid': 'Purples',
    'exp_resid_CAR': 'Reds',
    'exp_resid_mean': 'Blues',
    #
    'target': 'Greens',
    'target_CAR': 'Yellows',
    'target_mean': 'Blues',
    }
if False:
    for timeScale in timeScales:
        pdfPath = os.path.join(
            figureOutputFolder,
            prefix + '_{}_{}_lmfit_by_amplitude_{}_msec.pdf'.format(
                arguments['inputBlockSuffix'], arguments['window'], timeScale))
        pageCount = 0
        maskForThisPlot = plotDF.index.get_level_values('regrID').isin(paletteLookup.keys())
        with PdfPages(pdfPath) as pdf:
            for pageName, group in tqdm(plotDF.loc[maskForThisPlot, :].groupby(['electrode', 'regrID'])):
                regrName = pageName[1]
                print('Plotting {} at time scale {}'.format(pageName, timeScale))
                plotGroup = group.stack().to_frame(name='signal').reset_index()
                g = sns.relplot(
                    # data=plotGroup,
                    data=plotGroup.query('(bin < {}e-3) & (bin >= 1e-3)'.format(timeScale)),
                    x='bin', y='signal',
                    hue=amplitudeFieldName, palette=paletteLookup[regrName],
                    row='ycoords', col='xcoords', lw=2,
                    **relplotKWArgs)
                pageTitle = g.fig.suptitle(pageName)
                saveLegendOpts = {
                        'bbox_extra_artists': [pageTitle]}
                # contrived way of pushing legend outside without
                # resizing the figure
                allLegends = [
                    a.get_legend()
                    for a in g.axes.flat
                    if a.get_legend() is not None] + [g._legend]
                if len(allLegends):
                    # bb = matplotlib.transforms.Bbox([[-1.01, 0.01], [-0.01, 1.01]])
                    # allLegends[0].set_bbox_to_anchor(bb)
                    saveLegendOpts.update({
                        'bbox_extra_artists': [pageTitle] + allLegends})
                g.fig.set_size_inches(
                    g._ncol * relplotKWArgs['height'] * relplotKWArgs['aspect'] + 10,
                    g._nrow * relplotKWArgs['height'] + 2)
                for (ro, co, hu), dataSubset in g.facet_data():
                    emptySubset = (
                            (dataSubset.empty) or
                            (dataSubset['signal'].isna().all()))
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                pdf.savefig(bbox_inches='tight', pad_inches=0, **saveLegendOpts)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                pageCount += 1
                if limitPages is not None:
                    if pageCount > limitPages:
                        break
