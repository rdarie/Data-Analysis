"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from stats test
    --hueName=hueName                      break down by hue  [default: amplitudeCat]
    --hueControl=hueControl                hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=hueControl              styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from stats test [default: control]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import seaborn as sns
import matplotlib.pyplot as plt
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
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")

rowColOpts = asp.processRowColArguments(arguments)

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(resultPath))
pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_{}_{}.pdf'.format(
        arguments['inputBlockName'], arguments['window'],
        'meanRAUC'))
#  Overrides
limitPages = None
#  End Overrides
RecCurve = pd.read_hdf(resultPath, 'meanRAUC')

plotRC = RecCurve.reset_index()
plotRC['normalizedRAUC'] = np.nan
plotRC['featureName'] = np.nan
plotRC['EMGSide'] = np.nan
plotRC['EMGSite'] = np.nan
plotRC['EMGFeature'] = False
for name, group in plotRC.groupby('feature'):
    plotRC.loc[group.index, 'normalizedRAUC'] = (
        MinMaxScaler()
        .fit_transform(
            group['rauc'].to_numpy().reshape(-1, 1)))
    featName = name.replace('#0', '')
    for key in openEphysChanNames.keys():
        if key == featName:
            featName = openEphysChanNames[key]
            plotRC.loc[group.index, 'featureName'] = featName
            plotRC.loc[group.index, 'EMGSite'] = featName.split(' ')[1]
            plotRC.loc[group.index, 'EMGSide'] = featName.split(' ')[0]
            plotRC.loc[group.index, 'EMGFeature'] = True
plotRC.loc[plotRC['EMGFeature'], 'EMG Location'] = (
    plotRC.loc[plotRC['EMGFeature'], 'EMGSide'] + ' ' +
    plotRC.loc[plotRC['EMGFeature'], 'EMGSite']
    )
plotRC = plotRC.query("EMGFeature & (kruskalP < 1e-3)")
g = sns.relplot(
    col='electrode', col_wrap=5, col_order=np.unique(plotRC['electrode']),
    x='amplitude', y='normalizedRAUC',
    style='EMGSide', style_order=['Right', 'Left', 'Central'],
    hue='EMGSite', hue_order=np.unique(plotRC['EMGSite']),
    kind='line', data=plotRC,
    height=5, aspect=1.5, ci='sd', estimator='mean',
    )
# for (ro, co, hu), dataSubset in g.facet_data():
#     break
plt.savefig(pdfPath)
