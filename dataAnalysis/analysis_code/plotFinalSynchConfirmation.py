"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --lazy                                 load from raw, or regular? [default: False]
    --showFigures                          load from raw, or regular? [default: False]
    --debugging                            load from raw, or regular? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps) [default: rig]
    --selectionName2=selectionName2        filename for resulting estimator (cross-validated n_comps) [default: laplace]
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
    --plotSuffix=plotSuffix                filename for resulting estimator (cross-validated n_comps)
    --enableOverrides                      modify default plot opts? [default: False]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list?
    --alignQuery=alignQuery                how to restrict trials
    --individualTraces                     mean+sem or individual traces? [default: False]
    --noStim                               mean+sem or individual traces? [default: False]
    --limitPages=limitPages                mean+sem or individual traces?
    --overlayStats                         overlay ANOVA significance stars? [default: False]
    --recalcStats                          overlay ANOVA significance stars? [default: False]
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from stats test
    --hueName=hueName                      break down by hue  [default: amplitude]
    --hueControl=hueControl                hues to exclude from stats test
    --sizeName=sizeName                    break down by hue  [default: RateInHz]
    --sizeControl=sizeControl              hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=styleControl            styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from stats test [default: control]
    --winStart=winStart                    start of window [default: -200]
    --winStop=winStop                      end of window [default: 400]
"""

import logging
logging.captureWarnings(True)
import matplotlib, os, sys
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
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os, traceback
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
from copy import deepcopy
import dataAnalysis.custom_transformers.tdr as tdr
from dask.distributed import Client, LocalCluster
import dataAnalysis.preproc.ns5 as ns5
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import gc
import dill as pickle
from docopt import docopt
for arg in sys.argv:
    print(arg)
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 4,
        "axes.labelsize": 7,
        "axes.titlesize": 9,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
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
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV


print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''

arguments = {
    'rowControl': None, 'limitPages': None, 'alignQuery': None, 'colControl': 'control', 'processAll': True,
    'styleControl': None, 'selectionName': 'rig', 'alignFolderName': 'motion', 'unitQuery': None,
    'sizeName': 'RateInHz', 'enableOverrides': True, 'analysisName': 'hiRes', 'window': 'XL',
    'winStart': '-200', 'verbose': '1', 'lazy': False, 'exp': 'exp201901261000', 'showFigures': False,
    'winStop': '400', 'sizeControl': None, 'overlayStats': False, 'colName': 'electrode', 'individualTraces': False,
    'datasetName': 'Block_XL_df_pa', 'styleName': 'RateInHz', 'noStim': False, 'recalcStats': False,
    'hueName': 'amplitude', 'plotSuffix': 'final_synch_confirmation', 'debugging': False,
    'rowName': 'pedalDirection', 'blockIdx': '2', 'estimatorName': None, 'hueControl': None}
os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

if __name__ == '__main__':
    idxSl = pd.IndexSlice
    styleOpts = {
        'legend.lw': 2,
        'tight_layout.pad': 3e-1, # units of font size
        'panel_heading.pad': 0.
        }
    arguments['verbose'] = int(arguments['verbose'])
    #
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    from dataAnalysis.analysis_code.new_plot_options import *
    #
    arguments['verbose'] = int(arguments['verbose'])
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    #
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
    pdfName = '{}_{}'.format(
        datasetName, selectionName)
    if arguments['plotSuffix'] is not None:
        plotSuffix = '_{}'.format(arguments['plotSuffix'])
    else:
        plotSuffix = ''
    pdfPath = os.path.join(figureOutputFolder, '{}{}.pdf'.format(pdfName, plotSuffix))
    #
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    #
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
        iteratorsBySegment = loadingMeta['iteratorsBySegment']
        cvIterator = iteratorsBySegment[0]
        # cv_kwargs = loadingMeta['cv_kwargs']
        if 'normalizeDataset' in loadingMeta:
            normalizeDataset = loadingMeta['normalizeDataset']
            unNormalizeDataset = loadingMeta['unNormalizeDataset']
            normalizationParams = loadingMeta['normalizationParams']
        else:
            normalizeDataset = None
    #
    rigDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    lfpDF = pd.read_hdf(datasetPath, '/{}/data'.format(arguments['selectionName2']))
    #
    lfpDF.index = rigDF.index
    #
    # rigTrialInfo = rigDF.index.to_frame().reset_index(drop=True)
    # lfpTrialInfo = lfpDF.index.to_frame().reset_index(drop=True)
    #
    dataDF = pd.concat([rigDF, lfpDF], axis='columns')
    del rigDF, lfpDF
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
    #
    alignQuery = ash.processAlignQueryArgs(
        namedQueries, alignQuery=arguments['alignQuery'])
    if alignQuery is not None:
        keepIndices = trialInfo.query(alignQuery, engine='python').index
        keepMask = trialInfo.index.isin(keepIndices)
        if not keepMask.any():
            raise(Exception('query {} did not produce any results'.format(alignQuery)))
        else:
            dataDF = dataDF.loc[keepMask, :]
            trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    tMask = pd.Series(True, index=trialInfo.index)
    if arguments['winStop'] is not None:
        tMask = tMask & (trialInfo['bin'] < (float(arguments['winStop']) * 1e-3))
    if arguments['winStart'] is not None:
        tMask = tMask & (trialInfo['bin'] >= (float(arguments['winStart']) * 1e-3))
    if not tMask.all():
        dataDF = dataDF.iloc[tMask.to_numpy(), :]
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        del tMask
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz', ],
        'kinematicCondition': ['pedalDirection', 'pedalSizeCat', 'pedalMovementCat'],
        'kinematicConditionNoSize': ['pedalDirection', 'pedalMovementCat']
        }
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        trialInfo.loc[:, canName] = compoundAnn
    dataDF.index = pd.MultiIndex.from_frame(trialInfo)
    ######
    prf.print_memory_usage('just loaded data, plotting')
    try:
        dt = loadingMeta['iteratorOpts']['forceBinInterval']
    except:
        dt = rasterOpts['binOpts'][arguments['analysisName']]['binInterval']
    trialInfo.loc[:, 'trialStartT'] = trialInfo['t']
    trialInfo.loc[:, 't'] = np.arange(trialInfo.shape[0]) * dt
    ######
    confPlotWinSize = 15.  # seconds
    plotRounds = trialInfo['t'].apply(lambda x: np.floor(x / confPlotWinSize))
    rigChans = [
        cN
        for cN in [
            'amplitude_raster', 'amplitude', 'position', 'utah_rawAverage_0', 'utah_artifact_0',
            'utah_csd_11', 'utah_csd_18', 'utah_csd_81', 'utah_csd_88']
        if cN in featureInfo['feature'].to_list()]
    insChans = [
        cN
        for cN in featureInfo['feature'].to_list()
        if ('ins_td' in cN) or ('ins_accinertia' in cN)]
    rigChans += insChans
    with PdfPages(pdfPath) as pdf:
        for pr in tqdm(plotRounds.unique()):
            plotMask = (plotRounds == pr).to_numpy()
            fig, ax = plt.subplots(1, 1, figsize=(50, 3))
            extraArtists = []
            for cIdx, cN in enumerate(rigChans):
                plotTrace = dataDF.xs(cN, level='feature', axis='columns').iloc[plotMask, :].to_numpy()
                plotTrace = MinMaxScaler().fit_transform(plotTrace).flatten() + cIdx * .75
                thisT = trialInfo.loc[plotMask, 't'].to_numpy()
                ax.plot(thisT, plotTrace, label=cN, alpha=0.5)
                traceLabel = ax.text(np.min(thisT), cIdx * .75 + 0.5, '{}'.format(cN), va='center', ha='right')
                extraArtists.append(traceLabel)
                ax.set_xlim([thisT.min(), thisT.max()])
            if 'amplitude_raster' in featureInfo['feature'].to_list():
                cN = 'amplitude_raster'
                plotTrace = dataDF.xs(cN, level='feature', axis='columns').iloc[plotMask, :].to_numpy()
                thisMask = (plotTrace.flatten() > 0)
                thisT = trialInfo.loc[plotMask, 't'].to_numpy()
                plotTrace = MinMaxScaler().fit_transform(plotTrace).flatten()
                ax.scatter(
                    thisT[thisMask], plotTrace[thisMask],
                    marker='+', c='k', s=3,
                    label='stim pulses')
            for name, group in trialInfo.loc[plotMask, :].groupby('trialUID'):
                ax.axvline(group['t'].min(), c='k')
                if group['bin'].min() < 0:
                    ax.axvspan(
                        group['t'].min() - group['bin'].min(),
                        group['t'].max(), alpha=0.1,
                        color='red', zorder=-10)
                metaInfo = group.iloc[0, :]
                ps = (
                    '{:.0f}  deg.'.format(100 * metaInfo['pedalSize'])
                    if (not isinstance(metaInfo['pedalSize'], str))
                    else '{}'.format(metaInfo['pedalSize']))
                captionText = (
                    't(start) = {:.2f} sec\n'.format(metaInfo['trialStartT']) +
                    'segment = {}\n'.format(metaInfo['segment']) +
                    'stim electrode = {}\n'.format(metaInfo['electrode']) +
                    'stim amplitude = {} uA\n'.format(metaInfo['trialAmplitude']) +
                    'stim rate = {} Hz\n'.format(metaInfo['trialRateInHz']) +
                    'sessionID = {}\n'.format(metaInfo['expName']) +
                    'movement size = {}\n'.format(ps)
                    )
                axCaption = ax.text(group['t'].min(), ax.get_ylim()[1], captionText, ha='left', va='bottom')
                extraArtists.append(axCaption)
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            # ax.legend(loc='lower left')
            ax.set_xlabel('Time (s)')
            ax.set_yticks([])
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            # extraArtists.append(ax.get_legend())
            figSaveOpts = dict(
                bbox_extra_artists=tuple(extraArtists),
                bbox_inches='tight')
            pdf.savefig(**figSaveOpts)
            plt.close()