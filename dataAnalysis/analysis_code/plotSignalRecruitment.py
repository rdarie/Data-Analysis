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
import dataAnalysis.helperFunctions.helper_functions_new as hf
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
recCurve = pd.read_hdf(resultPath, 'RAUC')
# plotOpts = pd.read_hdf(resultPath, 'RAUC_plotOpts')
pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '_{}_{}_{}.pdf'.format(
        inputBlockSuffix, arguments['window'],
        '_RAUC'))
plotRC = recCurve.reset_index()
keepCols = [
    'segment', 'originalIndex', 'feature', 'lag'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)

#
'''emgPalette = (
    plotOpts
        .loc[:, ['featureName', 'color']]
        .set_index('featureName')['color']
        .to_dict())
mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
mapDF.loc[:, 'whichArray'] = mapDF['elecName'].apply(lambda x: x[:-1])
mapDF.loc[mapDF['whichArray'] == 'rostral', 'xcoords'] += mapDF['xcoords'].max() * 2
mapDF.loc[:, 'channelRepetition'] = mapDF['label'].apply(lambda x: x.split('_')[-1])
mapDF.loc[:, 'topoName'] = mapDF['label'].apply(lambda x: x[:-2])
mapAMask = (mapDF['channelRepetition'] == 'a').to_numpy()'''
#
# plotRC.loc[:, 'electrode'] = plotRC['electrode'].apply(lambda x: x[1:])
# plotRC.loc[:, 'feature'] = plotRC['feature'].apply(lambda x: x[:-4])
#

'''if RCPlotOpts['significantOnly']:
    plotRC = plotRC.query("(kruskalP < 1e-3)")
#
if RCPlotOpts['keepElectrodes'] is not None:
    keepDataMask = plotRC['electrode'].isin(RCPlotOpts['keepElectrodes'])
    plotRC = plotRC.loc[keepDataMask, :]
#
if RCPlotOpts['keepFeatures'] is not None:
    keepDataMask = plotRC['featureName'].isin(RCPlotOpts['keepFeatures'])
    plotRC = plotRC.loc[keepDataMask, :]'''

'''annotNames = ['xcoords', 'ycoords', 'whichArray']
for annotName in annotNames:
    lookupSource = mapDF.loc[mapAMask, [annotName, 'topoName']].set_index('topoName')[annotName]
    plotRC.loc[:, 'electrode_' + annotName] = plotRC['electrode'].map(lookupSource)
'''
pdb.set_trace()
# colName = 'electrode_xcoords'
colName = 'electrode'
colOrder = sorted(np.unique(plotRC[colName]))
hueName = 'feature'
hueOrder = sorted(np.unique(plotRC[hueName]))
rowName = 'pedalMovementCat'
rowOrder = sorted(np.unique(plotRC[rowName]))
colWrap = min(3, len(colOrder))
height, aspect = 5, 3
# pdb.set_trace()
g = sns.relplot(
    col=colName,
    col_order=colOrder,
    row=rowName,
    # x='normalizedAmplitude',
    x=amplitudeFieldName,
    y='normalizedRAUC',
    hue=hueName, hue_order=hueOrder,
    kind='line', data=plotRC,
    # palette=emgPalette,
    height=height, aspect=aspect, ci='sem', estimator='mean',
    facet_kws=dict(sharey=True, sharex=False, legend_out=True), lw=3,
    )
'''g = sns.catplot(
    col=colName,
    col_order=colOrder,
    col_wrap=colWrap,
    # row='EMGSide',
    # x='normalizedAmplitude',
    x=amplitudeFieldName,
    y='normalizedRAUC',
    hue=hueName, hue_order=hueOrder,
    kind='box', data=plotRC,
    # palette=emgPalette,
    height=height, aspect=aspect,
    facet_kws=dict(sharey=True, sharex=False, legend_out=True),
    )'''
g.fig.set_size_inches(len(rowOrder) * height * aspect + 10, height + 2)
plt.tight_layout(pad=.1)
plt.savefig(pdfPath)
if arguments['showFigures']:
    plt.show()
else:
    plt.close()