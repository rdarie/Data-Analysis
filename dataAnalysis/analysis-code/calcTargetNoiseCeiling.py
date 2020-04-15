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
from sklearn.model_selection import LeaveOneOut, PredefinedSplit
from sklearn.utils import shuffle
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
    alignSubFolder, prefix, **arguments)
from sklearn.preprocessing import scale, robust_scale
print('loading {}'.format(triggeredPath))

dataReader, dataBlock = preproc.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_outliers.h5'.format(
        arguments['window']))
dataDF = preproc.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)

def noiseCeil(
        x, xBounds=None,
        plotting=False, iterMethod='loo',
        corrMethod='pearson',
        maxIter=1e6):
    if xBounds is not None:
        maskX = (x.columns > xBounds[0]) & (x.columns < xBounds[1])
    else:
        maskX = np.ones_like(x.columns).astype(np.bool)
    #
    if iterMethod == 'loo':
        loo = LeaveOneOut()
        allCorr = pd.Series(index=x.index)
        iterator = loo.split(x)
        for idx, (train_index, test_index) in enumerate(tqdm(iterator)):
            refX = x.iloc[train_index, maskX].mean()
            testX = pd.Series(
                x.iloc[test_index, maskX].to_numpy().squeeze(),
                index=refX.index)
            allCorr.iloc[test_index] = refX.corr(
                testX, method=corrMethod)
            if idx > maxIter:
                break
        allCorr.dropna(inplace=True)
    elif iterMethod == 'half':
        nSamp = x.shape[0]
        nChoose = int(x.shape[0] / 2)
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        allCorr = pd.Series(index=range(maxIter))
        for idx in tqdm(range(maxIter)):
            testX = shuffle(x, n_samples=nChoose)
            refX = x.loc[~x.index.isin(testX.index), :]
            allCorr.iloc[idx] = refX.mean().corr(
                testX.mean(), method=corrMethod)
    print('\n')
    return allCorr.mean(), allCorr.std()

testVar = 'feature'
groupBy = ['electrode', 'nominalCurrent']
resultNames = ['noiseCeil', 'noiseCeilStd']

resDF = ash.applyFunGrouped(
    dataDF,
    groupBy, testVar,
    fun=noiseCeil, funArgs=[],
    funKWargs=dict(
        plotting=arguments['plotting'],
        iterMethod='loo', maxIter=1e3),
    resultNames=resultNames, plotting=False)
pdb.set_trace()
for rN in resultNames:
    resDF[rN].to_hdf(resultPath, rN, format='fixed')
mask = pd.DataFrame(
    0,
    index=resDF['noiseCeil'].index,
    columns=resDF['noiseCeil'].columns)
for cN in ['LBicepsBrachiiEmgEnv#0', 'RBicepsBrachiiEmgEnv#0']:
    mask.loc[:, cN] = True
dropIndex = mask.index.get_level_values('electrode').isin([
    '-rostralZ_e17+rostralZ_e21', '-caudalZ_e17+caudalZ_e21'])
mask.loc[dropIndex, :] = True
if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    pdfName = os.path.join(figureOutputFolder, 'noise_ceil_loo.pdf')
    with PdfPages(pdfName) as pdf:
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(resDF['noiseCeil'], mask=mask, ax=ax, center=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.show()
        # for pageIdx, (pageName, pageGroup) in enumerate(tqdm(resDF['noiseCeil'].groupby('electrode'))):
