"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --saveResults                          load from raw, or regular? [default: False]
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
frPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        'fr', arguments['window']))
rigPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        'rig', arguments['window']))
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
#
if arguments['verbose']:
    print('Loading dataBlock: {}'.format(frPath))
frReader, frBlock = ns5.blockFromPath(
    frPath, lazy=arguments['lazy'])
if arguments['verbose']:
    print('Loading alignedAsigs: {}'.format(frPath))
frDF = ns5.alignedAsigsToDF(
    frBlock, **alignedAsigsKWargs)
#
rigKWargs = deepcopy(alignedAsigsKWargs)
arguments.pop('selector')
rigKWargs['unitNames'], rigKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, inputBlockName='rig', **arguments)
if arguments['verbose']:
    print('Loading dataBlock: {}'.format(rigPath))
rigReader, rigBlock = ns5.blockFromPath(
    rigPath, lazy=arguments['lazy'])
if arguments['verbose']:
    print('Loading alignedAsigs: {}'.format(rigPath))
rigDF = ns5.alignedAsigsToDF(
    rigBlock, **rigKWargs)
rigDF.fillna(0, inplace=True)

rigColumnSelect = [
    i for i in rigDF.columns
    if (
        ('_Right_x' in i[0]) or ('_Right_y' in i[0]) or
        ('_Right_z' in i[0]) or
        ('ins_' in i[0])
        )]

outlierLogPath = os.path.join(
    figureFolder,
    prefix + '_{}_outlierBlocks.txt'.format(arguments['window']))
if os.path.exists(outlierLogPath):
    os.remove(outlierLogPath)


def findOutliers(
        frDF, qThresh=None, countThresh=10, plotting=False):
    if qThresh is None:
        qThresh = 1 - 1e-6
    sdThresh = chi2.interval(qThresh, 1)
    sdThreshMV = chi2.interval(qThresh, len(frDF.columns))
    #
    averageDeviation = np.sqrt((frDF.quantile(q=0.9) ** 2).sum())
    # if not (np.isfinite(averageDeviation)):
    highFeatures = (frDF.quantile(q=0.9) > sdThresh[1])
    nOutliers = highFeatures.sum()
    outlierLabels = ' '.join([
        i[0][:-2]
        for i in highFeatures.index[highFeatures].to_list()])
    # tooMuch = (nOutliers >= countThresh)
    tooMuch = (averageDeviation >= sdThreshMV[1])
    seg = (
        frDF
        .index
        .get_level_values('segment')
        .unique())
    t = (
        frDF
        .index
        .get_level_values('t')
        .unique())
    # if (nOutliers >= countThresh) / 2:
    if tooMuch:
        try:
            summaryMessage = [
                'segment {} time {}\n'.format(seg[0], t[0]),
                (outlierLabels + '\n'),
                'average deviation, {} > {}\n'.format(averageDeviation, sdThreshMV[1]),
                'Found {} outlier channels\n'.format(nOutliers)]
            print(summaryMessage)
            with open(outlierLogPath, 'a') as f:
                f.writelines(summaryMessage)
        except Exception:
            pass
        if plotting:
            import matplotlib.pyplot as plt
            import seaborn as sns
            bins = np.linspace(sdThresh[0], sdThresh[1], 200)
            #  fig, ax = plt.subplots(len(frDF.columns), 1, sharex=True)
            fig, ax = plt.subplots(1, 1, sharex=True)
            for cIdx, cName in enumerate(frDF.columns):
                sns.distplot(
                    frDF[cName], kde=False, ax=ax,
                    bins=bins, label=cName[0])
            plt.suptitle('limits are {}'.format(sdThresh))
            plt.show()
    return nOutliers, averageDeviation, tooMuch, outlierLabels, seg[0], t[0]


def findOutliers2(
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

print('working with {} samples'.format(frDF.shape[0]))
# randSample = swr(rigDF.shape[0], 100000)
randSample = slice(None, None, 5)
#
testQThresh = 1 - 1e-6
print('For rig, sd tresh is {}'.format(chi2.interval((testQThresh), len(rigColumnSelect))))
print('For fr, sd tresh is {}'.format(chi2.interval((testQThresh), len(frDF.columns))))
print('For df 1 sd tresh is {}'.format(chi2.interval((testQThresh), 1)))
# np.sqrt(frMahalDist.xs(798, level='originalIndex').xs(1, level='segment'))
if arguments['verbose']:
    print('Calculating covariance matrix...')
supportFraction = 0.99
rigSub = (rigDF.loc[:, rigColumnSelect].to_numpy())
rigCov = (
    MinCovDet(support_fraction=supportFraction)
    .fit(rigSub[randSample, :]))
rigMahalDist = pd.DataFrame(
    rigCov.mahalanobis(rigSub),
    index=rigDF.index, columns=['mahalDist'])
outlierBlocksRig = ash.applyFunGrouped(
    rigMahalDist,
    groupBy, testVar,
    fun=findOutliers2, funArgs=[],
    funKWargs=dict(multiplier=200, nDim=len(rigColumnSelect)),
    resultNames=resultNames,
    plotting=False)
#
if arguments['verbose']:
    print('Calculating covariance matrix...')
frCov = (
    MinCovDet()
    .fit(np.sqrt(frDF).to_numpy()[randSample, :]))
###
frMahalDist = pd.DataFrame(
    frCov.mahalanobis(np.sqrt(frDF).to_numpy()),
    index=frDF.index, columns=['mahalDist'])
outlierBlocksFr = ash.applyFunGrouped(
    frMahalDist,
    groupBy, testVar,
    fun=findOutliers2, funArgs=[],
    funKWargs=dict(multiplier=2, nDim=len(frDF.columns)),
    resultNames=resultNames,
    plotting=False)
#outlierBlocksFr['rejectBlock'].sum()
outlierBlocks = {
    k: pd.DataFrame(
        np.nan,
        index=outlierBlocksFr[k].index,
        columns=outlierBlocksFr[k].columns)
    for k in resultNames}

if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(2, 1)
    sns.distplot(
        (rigMahalDist),
        #outlierBlocksRig['averageDeviation'],
        ax=ax[0], kde=False)
    ax[0].set_xlabel('Rig')
    # sns.distplot(
    #     (frMahalDist),
    #     #outlierBlocksFr['averageDeviation'],
    #     ax=ax[1], kde=False)
    # ax[1].set_xlabel('Fr')
    plt.show(block=False)
if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(2, 1)
    sns.boxplot(
        outlierBlocksRig['averageDeviation'],
        ax=ax[0])
    ax[0].set_xlabel('Rig')
    #sns.boxplot(
    #    outlierBlocksFr['averageDeviation'],
    #    ax=ax[1])
    #ax[1].set_xlabel('Fr')
    plt.show(block=False)
if arguments['plotting']:
    fig, ax = plt.subplots(2, 1, sharex=True)
    bla = (frMahalDist.xs(992, level='originalIndex').xs(3, level='segment'))
    ax[0].plot(
        bla.index.get_level_values('bin').to_numpy(),
        bla.to_numpy())
    bla = (frDF.xs(992, level='originalIndex').xs(3, level='segment'))
    ax[1].plot(
        bla.index.get_level_values('bin').to_numpy(),
        bla.to_numpy())
    plt.show()
# outlierBlocksFr['averageDeviation'].xs(3, level='segment').sort_values('all')
# outlierBlocksRig['averageDeviation'].xs(3, level='segment').sort_values('all')
# outlierBlocksFr['averageDeviation'].xs(992, level='originalIndex').xs(3, level='segment')
outlierBlocks['averageDeviation'] = (
    outlierBlocksRig['averageDeviation'] +
    outlierBlocksFr['averageDeviation'])
outlierBlocks['rejectBlock'] = (
    (outlierBlocksRig['rejectBlock']).astype(np.bool) |
    (outlierBlocksFr['rejectBlock']).astype(np.bool))
if arguments['saveResults']:
    outlierBlocks['averageDeviation'].to_hdf(
        resultPath, 'averageDeviation', format='fixed')
    outlierBlocks['rejectBlock'].to_hdf(
        resultPath, 'rejectBlock', format='fixed')