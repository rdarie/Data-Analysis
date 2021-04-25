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
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierMask                    delete outlier trials? [default: False]
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
from namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
sns.set(
    context='notebook', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName']
    )
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['inputBlockSuffix'], arguments['window']))
print('loading {}'.format(resultPath))
outlierTrials = ash.processOutlierTrials(
    scratchPath, prefix, **arguments)
#  Overrides
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
recCurve = pd.read_hdf(resultPath, 'meanRAUC')
plotOpts = pd.read_hdf(resultPath, 'meanRAUC_plotOpts')
pdfPath = os.path.join(
    figureOutputFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockSuffix'], arguments['window'],
        'meanRAUC'))

plotEmgRC = recCurve.copy().reset_index()
keepCols = [
    'segment', 'originalIndex', 't', 'RateInHz',
    'electrode', 'nominalCurrent', 'feature', 'lag']
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in []]
if RCPlotOpts['significantOnly']:
    plotEmgRC = plotEmgRC.query("(kruskalP < 1e-3)")
#
if RCPlotOpts['keepElectrodes'] is not None:
    keepDataMask = emgRC['electrode'].isin(RCPlotOpts['keepElectrodes'])
    plotEmgRC = plotEmgRC.loc[keepDataMask, :]
#
if RCPlotOpts['keepFeatures'] is not None:
    keepDataMask = plotEmgRC['featureName'].isin(RCPlotOpts['keepFeatures'])
    plotEmgRC = plotEmgRC.loc[keepDataMask, :]
#
emgPalette = (
    plotOpts
        .loc[:, ['featureName', 'color']]
        .set_index('featureName')['color']
        .to_dict())

colOrder = sorted(np.unique(plotEmgRC['electrode']))
hueName = 'featureName'
featToSite = plotOpts.loc[:, ['featureName', 'EMGSite']].set_index('EMGSite')['featureName']
hueOrder = (
    featToSite.loc[featToSite.isin(plotEmgRC['featureName'])]
    .sort_index().to_numpy())
g = sns.relplot(
    col='electrode',
    col_order=colOrder,
    col_wrap=min(3, len(colOrder)),
    # row='EMGSide',
    # x='normalizedAmplitude',
    x=amplitudeFieldName,
    y='normalizedRAUC',
    style='EMGSide', style_order=['Right', 'Left'],
    hue=hueName, hue_order=hueOrder,
    kind='line', data=plotEmgRC.query('RateInHz > 50'),
    palette=emgPalette,
    height=5, aspect=1.5, ci='sem', estimator='mean',
    facet_kws=dict(sharey=True, sharex=False), lw=2,
    )
#
plt.savefig(pdfPath)
plt.show()
