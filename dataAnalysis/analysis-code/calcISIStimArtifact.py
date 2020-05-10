"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --saveResults                          load from raw, or regular? [default: False]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --outputBlockName=outputBlockName      which trig_ block to pull [default: pca_clean]
    --verbose                              print diagnostics? [default: False]
    --plotting                             plot results?
    --window=window                        process with short window? [default: long]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: all]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: all]
    --selector=selector                    filename if using a unit selector
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
"""

import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
import pingouin as pg
from tqdm import tqdm
import pandas as pd
import numpy as np
import quantities as pq
from copy import deepcopy
from docopt import docopt
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
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
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
outputPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['outputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, removeFuzzyName=False,
    decimate=1, procFun=None,
    # windowSize=(-1e-3, 2e-3),
    metaDataToCategories=False,
    transposeToColumns='bin', concatOn='index',
    verbose=False))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)

dataDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
dataDF.columns = dataDF.columns.astype(np.float)
procDF = dataDF.copy()

# pdb.set_trace()

groupBy = [
    'electrode', 'nominalCurrent',
    'feature', 'firstPW', 'secondPW', 'stimCat']
tau = 0.03e-3


def calcStimArtifact(DF, t, tOffset, vMag, tau):
    artDF = pd.DataFrame(0., index=DF.index, columns=DF.columns)
    tMask = (t - tOffset) >= 0
    # pdb.set_trace()
    for rowIdx in artDF.index:
        artDF.loc[rowIdx, tMask] = (
            vMag[rowIdx] * np.exp(-(t[tMask] - tOffset) / tau))
    return artDF


for name, group in tqdm(dataDF.groupby(groupBy)):
    blankingDur = (name[3] + name[4]) * 1e-6 + 5 * 30e3 ** (-1)
    t = group.columns
    if name[5] == 'stimOn':
        blankMask = (t >= 0) & (t <= 0 + blankingDur)
    # remove saturated readings
    satLimit = 6e3
    lastBlankIdx = np.flatnonzero(blankMask)[-1]
    while (group.iloc[:, lastBlankIdx+1].abs() > satLimit).any():
        lastBlankIdx += 1
    blankMask[np.flatnonzero(blankMask)[-1]:lastBlankIdx+1] = True
    firstBlankIdx = np.flatnonzero(blankMask)[0]
    lastBlankIdx = np.flatnonzero(blankMask)[-1]
    # deltaV = (group.iloc[:, lastBlankIdx+1] - group.iloc[:, firstBlankIdx-1])
    deltaV = (group.iloc[:, lastBlankIdx+1])
    artifactDF = calcStimArtifact(group, t, t[lastBlankIdx+1], deltaV, tau)
    diagnosticPlots = False
    if diagnosticPlots:
        illustrateIdx = 1
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].plot(t, group.iloc[illustrateIdx, :], '.-')
        ax[0].plot(t[blankMask], group.iloc[illustrateIdx, :][blankMask])
        ax[0].plot(t, artifactDF.iloc[illustrateIdx, :])
    procDF.loc[group.index, :] = procDF.loc[group.index, :] - artifactDF
    procDF.loc[group.index, blankMask] = np.nan
    if diagnosticPlots:
        plotFixed = procDF.loc[group.index, :].iloc[illustrateIdx, :]
        ax[1].plot(t, plotFixed)
        ax[0].set_title('{} during stim on {}'.format(name[2], name[0]))
        plt.show()
#
procDF.interpolate(method='akima', axis=1, inplace=True)
del dataDF
#
masterBlock = ns5.alignedAsigDFtoSpikeTrain(procDF, dataBlock)
if arguments['lazy']:
    dataReader.file.close()
masterBlock = ns5.purgeNixAnn(masterBlock)
if os.path.exists(outputPath):
    os.remove(outputPath)
writer = ns5.NixIO(filename=outputPath)
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
