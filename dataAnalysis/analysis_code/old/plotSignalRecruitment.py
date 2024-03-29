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
        'window': 'XL', 'blockIdx': '2', 'lazy': False, 'verbose': False,
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
    scratchFolder, arguments['analysisName']
    )
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName']
    )
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
if False:
    recCurve = pd.read_hdf(resultPath, 'RAUC')
else:
    rawRecCurve = pd.read_hdf(resultPath, 'raw')
    rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
    recCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
    rauc = pd.read_hdf(resultPath, 'scaled')
    rauc.columns = rauc.columns.get_level_values('feature')
    recCurve.loc[:, 'rauc'] = rauc.stack().to_numpy()
    normalizedRAUC = pd.read_hdf(resultPath, 'normalized')
    normalizedRAUC.columns = normalizedRAUC.columns.get_level_values('feature')
    recCurve.loc[:, 'normalizedRAUC'] = normalizedRAUC.stack().to_numpy()

# plotOpts = pd.read_hdf(resultPath, 'RAUC_plotOpts')
pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '{}_{}_{}.pdf'.format(
        inputBlockSuffix, arguments['window'],
        'RAUC'))
plotRC = recCurve.reset_index()
keepCols = [
    'segment', 'originalIndex', 'feature', 'lag'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)

# colName = 'electrode_xcoords'
colName = 'electrode'
colOrder = sorted(np.unique(plotRC[colName]))
hueName = 'feature'
hueOrder = sorted(np.unique(plotRC[hueName]))
rowName = 'pedalMovementCat'
rowOrder = sorted(np.unique(plotRC[rowName]))
colWrap = min(3, len(colOrder))
whichRAUC = 'normalizedRAUC'
height, width = 3, 3
aspect = width / height
with PdfPages(pdfPath) as pdf:
    plotLims = plotRC[whichRAUC].quantile([1e-2, 1-1e-2])
    for name, group in plotRC.groupby('trialRateInHz'):
        nAmps = group[amplitudeFieldName].unique().size
        if nAmps > 1:
            g = sns.relplot(
                col=colName,
                col_order=colOrder,
                row=rowName,
                # x='normalizedAmplitude',
                x=amplitudeFieldName,
                y=whichRAUC,
                hue=hueName, hue_order=hueOrder,
                kind='line', data=group,
                # palette=emgPalette,
                height=height, aspect=aspect, errorbar='sd', estimator='mean',
                facet_kws=dict(sharey=True, sharex=False, margin_titles=True), lw=1,
                )
        else:
            g = sns.catplot(
                col=colName,
                col_order=colOrder,
                row=rowName,
                # row='EMGSide',
                # x='normalizedAmplitude',
                x=amplitudeFieldName,
                y=whichRAUC,
                hue=hueName, hue_order=hueOrder,
                kind='box', data=group,
                # palette=emgPalette,
                height=height, aspect=aspect,
                facet_kws=dict(sharey=True, sharex=False, margin_titles=True),
                )
        g.axes[0, 0].set_ylim(plotLims)
        g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
        g.tight_layout(pad=0.)
        figTitle = g.fig.suptitle('rate = {}'.format(name))
        leg = g._legend
        pdf.savefig(
            bbox_inches='tight',
            bbox_extra_artists=[leg, figTitle]
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
#
pdfPath = os.path.join(
    figureOutputFolder,
    blockBaseName + '{}_{}_{}.pdf'.format(
        inputBlockSuffix, arguments['window'],
        'RAUC_distributions'))
with PdfPages(pdfPath) as pdf:
    g = sns.displot(
        data=plotRC,
        x='rawRAUC', hue='feature', kind='hist', element='step'
        )
    figTitle = g.fig.suptitle('raw')
    # g.axes[0, 0].set_xlim(plotRC['rawRAUC'].quantile([1e-6, 1-1e-2]))
    pdf.savefig(
        bbox_inches='tight',
        bbox_extra_artists=[figTitle]
        )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
    g = sns.displot(
        data=plotRC,
        x='rauc', hue='feature', kind='hist', element='step'
        )
    figTitle = g.fig.suptitle('after power transformation')
    pdf.savefig(
        bbox_inches='tight',
        bbox_extra_artists=[figTitle]
        )
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
