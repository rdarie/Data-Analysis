"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --blockIdx=blockIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --plotting                                plot out the correlation matrix? [default: True]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --inputBlockName=inputBlockName           filename for inputs [default: fr]
    --maskOutlierBlocks                       delete outlier trials? [default: False]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                   query what the units will be aligned to? [default: midPeak]
    --alignFolderName=alignFolderName         append a name to the resulting blocks? [default: motion]
    --selector=selector                       filename if using a unit selector
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneOut, PredefinedSplit
from sklearn.utils import shuffle
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler, StandardScaler
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
import seaborn as sns
from math import factorial
from tqdm import tqdm
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("dark")
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(0, 300e-3),
    transposeToColumns='bin', concatOn='index',
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchPath, prefix, **arguments)
from sklearn.preprocessing import scale, robust_scale


def classEffect(xDF, effectLabel=None):
    tempDF = xDF.stack().to_frame(name='signal')
    epochs = pd.cut(
        tempDF.index.get_level_values('bin'),
        bins=10)
    tempDF.set_index(
        pd.Index(epochs, name='epoch'),
        append=True, inplace=True)
    testGroups = []
    for name, group in tempDF.groupby(effectLabel):
        testGroups.append(group.to_numpy())
    stat, pval = stats.kruskal(*testGroups)
    log_pval = np.log10(pval)
    if np.isneginf(log_pval):
        log_pval = np.nan
    log_stat = np.log10(stat)
    if np.isneginf(log_stat):
        log_stat = np.nan
    return stat, pval, log_pval, log_stat

testVar = 'feature'
groupBy = ['electrode', 'nominalCurrent']
effectLabel = ['epoch']
resultNames = ['stat', 'pval', 'log_pval', 'log_stat']

recalc = True
if recalc:
    print('loading {}'.format(triggeredPath))
    dataReader, dataBlock = preproc.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    dataDF = preproc.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    resDF = ash.applyFunGrouped(
        dataDF,
        groupBy, testVar,
        fun=classEffect, funArgs=[],
        funKWargs=dict(effectLabel=effectLabel),
        resultNames=resultNames, plotting=False)
    for rN in resultNames:
        resDF[rN].to_hdf(resultPath, rN, format='fixed')
else:
    resDF = {
        'stat': None,
        'p': None,
        }
# pdb.set_trace()
mask = pd.DataFrame(
    0,
    index=resDF['pval'].index,
    columns=resDF['pval'].columns)
dropColumns = [
    'LBicepsBrachiiEmgEnv#0', 'RBicepsBrachiiEmgEnv#0',
    # 'RPeroneusLongusEmgEnv#0', 'RSemitendinosusEmgEnv#0',
    # 'RVastusLateralisEmgEnv#0'
    ]
for cN in dropColumns:
    mask.loc[:, cN] = True
dropElectrodes = [
    '-rostralZ_e17+rostralZ_e21', '-caudalZ_e17+caudalZ_e21'
    ]
dropIndex = mask.index.get_level_values('electrode').isin(dropElectrodes)
mask.loc[dropIndex, :] = True

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    pdfName = os.path.join(figureOutputFolder, 'epoch_kruskal.pdf')
    with PdfPages(pdfName) as pdf:
        vB = {'log_stat': {}, 'log_pval': {'vmin': -30}}
        for cN in ['log_stat', 'log_pval']:
            fig, ax = plt.subplots(figsize=(12, 12))
            sns.heatmap(
                resDF[cN], mask=mask, ax=ax,
                **vB[cN],
                cmap=sns.cubehelix_palette(
                    reverse=True, as_cmap=True))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
            ax.set_title(cN)
            plt.tight_layout()
            pdf.savefig()
            plt.show()
        # for pageIdx, (pageName, pageGroup) in enumerate(tqdm(noiseCeil.groupby('electrode'))):