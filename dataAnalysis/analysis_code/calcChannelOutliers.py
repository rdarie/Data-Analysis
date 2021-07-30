"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                    which experimental day to analyze
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --processAll                                 process entire experimental day? [default: False]
    --lazy                                       load from raw, or regular? [default: False]
    --saveResults                                load from raw, or regular? [default: False]
    --useCachedMahalanobis                       load previous covariance matrix? [default: False]
    --inputBlockSuffix=inputBlockSuffix          which trig_ block to pull [default: pca]
    --verbose                                    print diagnostics? [default: False]
    --plotting                                   plot results?
    --window=window                              process with short window? [default: long]
    --unitQuery=unitQuery                        how to restrict channels if not supplying a list? [default: all]
    --alignQuery=alignQuery                      query what the units will be aligned to? [default: all]
    --selector=selector                          filename if using a unit selector
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --amplitudeFieldName=amplitudeFieldName      what is the amplitude named? [default: nominalCurrent]
    --sqrtTransform                              for firing rates, whether to take the sqrt to stabilize variance [default: False]
    --arrayName=arrayName                        name of electrode array? (for map file) [default: utah]
"""
import logging
logging.captureWarnings(True)
import matplotlib, os, sys
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
import pdb, traceback
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
# import pingouin as pg
import quantities as pq
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
from sklearn.preprocessing import StandardScaler, RobustScaler
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)

# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'processAll': False, 'useCachedMahalanobis': False, 'inputBlockSuffix': 'lfp',
        'amplitudeFieldName': 'amplitude', 'analysisName': 'hiRes', 'saveResults': True,
        'alignQuery': 'starting', 'exp': 'exp202101271100', 'sqrtTransform': False, 'blockIdx': '3',
        'plotting': True, 'window': 'XL', 'verbose': True, 'alignFolderName': 'motion', 'lazy': False,
        'selector': None, 'unitQuery': 'lfp', 'arrayName': 'utah'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'outlierTrials')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)

calcSubFolder = os.path.join(
    scratchFolder, 'outlierTrials', arguments['alignFolderName'])
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = 'Block'
else:
    prefix = ns5FileName

triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockSuffix'], arguments['window']))

resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_outliers.h5'.format(
        arguments['window']))

outlierLogPath = os.path.join(
    figureFolder,
    prefix + '_{}_outlierTrials.txt'.format(arguments['window']))
if os.path.exists(outlierLogPath):
    os.remove(outlierLogPath)

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, removeFuzzyName=False,
    decimate=1,
    metaDataToCategories=False,
    getMetaData=essentialMetadataFields,
    transposeToColumns='feature', concatOn='columns',
    verbose=False, procFun=None))
#
print(
    "'outlierDetectOptions' in locals(): {}"
    .format('outlierDetectOptions' in locals()))
#
if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC') or (blockExperimentType == 'isi'):
    # has stim but no motion
    stimulusConditionNames = stimConditionNames
elif blockExperimentType == 'proprio-motionOnly':
    # has motion but no stim
    stimulusConditionNames = motionConditionNames
else:
    stimulusConditionNames = stimConditionNames + motionConditionNames
print('Block type {}; using the following stimulus condition breakdown:'.format(blockExperimentType))
print('\n'.join(['    {}'.format(scn) for scn in stimulusConditionNames]))
if 'outlierDetectOptions' in locals():
    targetEpochSize = outlierDetectOptions['targetEpochSize']
    twoTailed = outlierDetectOptions['twoTailed']
    alignedAsigsKWargs['windowSize'] = outlierDetectOptions['windowSize']
    devQuantile = outlierDetectOptions.pop('devQuantile', 0.95)
    qThresh = outlierDetectOptions.pop('qThresh', 1e-6)
else:
    targetEpochSize = 1e-3
    twoTailed = False
    alignedAsigsKWargs['windowSize'] = (-100e-3, 400e-3)
    devQuantile = 0.95
    qThresh = 1e-6


alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)

if __name__ == "__main__":
    if 'mapDF' not in locals():
        electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
        mapExt = electrodeMapPath.split('.')[-1]
        if mapExt == 'cmp':
            mapDF = prb_meta.cmpToDF(electrodeMapPath)
        elif mapExt == 'map':
            mapDF = prb_meta.mapToDF(electrodeMapPath)
    print('loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    dataDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    # fix order of magnitude
    dataDF = dataDF.astype(float)
    ordMag = np.floor(np.log10(dataDF.abs().mean().mean()))
    if ordMag < 0:
        dataDF = dataDF * 10 ** (-ordMag)
    featureInfo = dataDF.columns.to_frame().reset_index(drop=True)
    dataChanNames = featureInfo['feature'].apply(lambda x: x.replace('#0', ''))
    banksLookup = mapDF.loc[:, ['label', 'bank']].set_index(['label'])['bank']
    featureInfo.loc[:, 'bank'] = dataChanNames.map(banksLookup)
    plotDF = dataDF.reset_index(drop=True).T.reset_index(drop=True).T
    plotDF.columns.name = 'flatFeature'
    plotDF.index.name = 'flatIndex'
    plotDF = plotDF.stack().reset_index(name='signal')
    plotDF.loc[:, 'bank'] = plotDF['flatFeature'].map(featureInfo['bank'])
    plotDF.loc[:, 'feature'] = plotDF['flatFeature'].map(featureInfo['feature'])
    signalRangesFigPath = os.path.join(
        figureOutputFolder,
        prefix + '_outlier_channels.pdf')
    h = 18
    w = 3
    aspect = w / h
    g = sns.catplot(
        col='bank', x='signal', y='feature',
        data=plotDF, orient='h', kind='violin', ci='sd',
        linewidth=0.5, cut=0,
        sharex=False, sharey=False, height=h, aspect=aspect
        )
    g.tight_layout()
    g.fig.savefig(
        signalRangesFigPath, bbox_inches='tight')
    plt.close()