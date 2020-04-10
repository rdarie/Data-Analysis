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
import pandas as pd
import numpy as np
from copy import deepcopy
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
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
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_outliers.h5'.format(
        arguments['window']))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, removeFuzzyName=False, decimate=10,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=False, verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, inputBlockName='fr', **arguments)

dataDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)

outlierLogPath = os.path.join(
    figureFolder,
    prefix + '_{}_outlierTrials.txt'.format(arguments['window']))
if os.path.exists(outlierLogPath):
    os.remove(outlierLogPath)

def findOutliers(
        mahalDistDF, qThresh=None, sdThresh=None,
        nDim=1, multiplier=1):
    if sdThresh is None:
        if qThresh is None:
            qThresh = 1 - 1e-12
        sdThresh = multiplier * chi2.interval(qThresh, nDim)[1]
    seg = (
        mahalDistDF
        .index
        .get_level_values('segment')
        .unique())
    t = (
        mahalDistDF
        .index
        .get_level_values('t')
        .unique())
    # print('Outlier thresh is {}'.format(sdThresh))
    averageDeviation = (mahalDistDF).quantile(q=0.9)[0]
    #averageDeviation = (mahalDistDF).max()[0]
    tooMuch = (averageDeviation >= sdThresh)
    if tooMuch:
        try:
            summaryMessage = [
                'segment {} time {}\n'.format(seg[0], t[0]),
                'average deviation {} > {}\n'.format(
                    averageDeviation, sdThresh)]
            print(summaryMessage)
            with open(outlierLogPath, 'a') as f:
                f.writelines(summaryMessage)
        except Exception:
            pass
    # else:
    #     print(
    #         'segment {} time {} average deviation {} < {}'.format(
    #             seg[0], t[0], averageDeviation, sdThresh))
    return averageDeviation, tooMuch, seg[0], t[0]


def applyMad(ser):
    if np.median(ser) == 0:
        return np.abs(zscore(ser))
    else:
        return np.abs(ser - np.median(ser)) / pg.mad(ser)

testVar = None
groupBy = ['segment', 'originalIndex', 't']
resultNames = [
    'averageDeviation', 'rejectBlock', 'seg', 't']

print('working with {} samples'.format(dataDF.shape[0]))
randSample = slice(None, None, 5)
#
if arguments['verbose']:
    print('Calculating covariance matrix...')
supportFraction = 0.99
covMat = (
    MinCovDet()
    .fit(dataDF.to_numpy()[randSample, :]))
###
frMahalDist = pd.DataFrame(
    covMat.mahalanobis(dataDF.to_numpy()),
    index=dataDF.index, columns=['mahalDist'])
outlierTrials = ash.applyFunGrouped(
    frMahalDist,
    groupBy, testVar,
    fun=findOutliers, funArgs=[],
    funKWargs=dict(multiplier=2, nDim=len(dataDF.columns)),
    resultNames=resultNames,
    plotting=False)
pdb.set_trace()
if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(2, 1)
    sns.distplot(
        (rigMahalDist),
        #outlierTrialsRig['averageDeviation'],
        ax=ax[0], kde=False)
    ax[0].set_xlabel('Rig')
    # sns.distplot(
    #     (frMahalDist),
    #     #outlierTrialsFr['averageDeviation'],
    #     ax=ax[1], kde=False)
    # ax[1].set_xlabel('Fr')
    plt.show(block=False)
if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(2, 1)
    sns.boxplot(
        outlierTrialsRig['averageDeviation'],
        ax=ax[0])
    ax[0].set_xlabel('Rig')
    #sns.boxplot(
    #    outlierTrialsFr['averageDeviation'],
    #    ax=ax[1])
    #ax[1].set_xlabel('Fr')
    plt.show(block=False)
if arguments['plotting']:
    fig, ax = plt.subplots(2, 1, sharex=True)
    bla = (frMahalDist.xs(992, level='originalIndex').xs(3, level='segment'))
    ax[0].plot(
        bla.index.get_level_values('bin').to_numpy(),
        bla.to_numpy())
    bla = (dataDF.xs(992, level='originalIndex').xs(3, level='segment'))
    ax[1].plot(
        bla.index.get_level_values('bin').to_numpy(),
        bla.to_numpy())
    plt.show()
# outlierTrialsFr['averageDeviation'].xs(3, level='segment').sort_values('all')
# outlierTrialsRig['averageDeviation'].xs(3, level='segment').sort_values('all')
# outlierTrialsFr['averageDeviation'].xs(992, level='originalIndex').xs(3, level='segment')
outlierTrials['averageDeviation'] = (
    outlierTrialsRig['averageDeviation'] +
    outlierTrialsFr['averageDeviation'])
outlierTrials['rejectBlock'] = (
    (outlierTrialsRig['rejectBlock']).astype(np.bool) |
    (outlierTrialsFr['rejectBlock']).astype(np.bool))
if arguments['saveResults']:
    outlierTrials['averageDeviation'].to_hdf(
        resultPath, 'averageDeviation', format='fixed')
    outlierTrials['rejectBlock'].to_hdf(
        resultPath, 'rejectBlock', format='fixed')