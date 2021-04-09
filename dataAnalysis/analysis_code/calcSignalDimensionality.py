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
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --iteratorSuffix=iteratorSuffix        filename for cross_val iterator
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
    --selector=selector                    filename if using a unit selector
    --loadFromFrames                       load data from pre-saved dataframes?
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from docopt import docopt

idxSl = pd.IndexSlice
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
dataFramesFolder = os.path.join(alignSubFolder, 'dataframes')

if not arguments['loadFromFrames']:
    triggeredPath = os.path.join(
        alignSubFolder,
        blockBaseName + '{}_{}.nix'.format(
            inputBlockSuffix, arguments['window']))

fullEstimatorName = '{}_{}_{}_{}'.format(
    blockBaseName,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])

estimatorSubFolder = os.path.join(
    analysisSubFolder, 'estimators'
    )
if not os.path.exists(estimatorSubFolder):
    os.makedirs(estimatorSubFolder)

estimatorPath = os.path.join(
    estimatorSubFolder,
    arguments['estimatorName'] + '.joblib')
    # fullEstimatorName + '.joblib')

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
#
cvIteratorSubfolder = os.path.join(
    scratchFolder, 'testTrainSplits',
    arguments['alignFolderName'])
if arguments['iteratorSuffix'] is not None:
    iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
else:
    iteratorSuffix = ''
iteratorPath = os.path.join(
    cvIteratorSubfolder,
    '{}_{}_{}{}_cvIterators.pickle'.format(
        blockBaseName,
        arguments['window'],
        arguments['alignQuery'],
        iteratorSuffix))
with open(iteratorPath, 'rb') as f:
    loadingMeta = pickle.load(f)
    iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
    cv_kwargs = loadingMeta.pop('cv_kwargs')
#
if not arguments['loadFromFrames']:
    alignedAsigsKWargs = loadingMeta.pop('alignedAsigsKWargs')
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **arguments)
    alignedAsigsKWargs['verbose'] = arguments['verbose']
    if arguments['verbose']:
        prf.print_memory_usage('loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    nSeg = len(dataBlock.segments)


def compute_scores(
        X, estimator,
        nComponentsToTest,
        cv, estimatorKWArgs={},
        verbose=False):
    scores = {}
    for n in nComponentsToTest:
        if verbose:
            print('evaluating with {} components'.format(n))
        instance = estimator(**estimatorKWArgs)
        instance.n_components = n
        scores[n] = cross_validate(instance, X, cv=cv)
    return scores


def shrunk_cov_score(
        X, cvIterator):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(
        ShrunkCovariance(),
        {'shrinkage': shrinkages}, cv=cvIterator)
    cv.fit(X)
    cvResDF = pd.DataFrame(cv.cv_results_)
    return cvResDF['mean_test_score'].max()


def calc_lw_score(X, cv):
    return cross_val_score(LedoitWolf(), X, cv=cv)

listOfDataFrames = []
saveUnitNames = None
if not arguments['loadFromFrames']:
    for segIdx in range(nSeg):
        if arguments['verbose']:
            prf.print_memory_usage('fitting on segment {}'.format(segIdx))
        # pdb.set_trace()
        if 'listOfROIMasks' in loadingMeta:
            alignedAsigsKWargs.update({'finalIndexMask': loadingMeta['listOfROIMasks'][segIdx]})
        thisDF = ns5.alignedAsigsToDF(
            dataBlock,
            whichSegments=[segIdx],
            **alignedAsigsKWargs)
        # if 'listOfExampleIndexes' in loadingMeta:
        #     assert np.all(dataDF.index == loadingMeta['listOfExampleIndexes'][segIdx])
        if saveUnitNames is None:
            saveUnitNames = [cN[0] for cN in thisDF.columns]
        listOfDataFrames.append(thisDF)
    if arguments['lazy']:
        dataReader.file.close()
else:    # loading frames
    experimentsToAssemble = loadingMeta.pop('experimentsToAssemble')
    currBlockNum = 0
    for expName, lOfBlocks in experimentsToAssemble.items():
        thisScratchFolder = os.path.join(scratchPath, expName)
        analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
            arguments, thisScratchFolder)
        thisDFFolder = os.path.join(alignSubFolder, 'dataframes')
        for bIdx in lOfBlocks:
            theseArgs = arguments.copy()
            theseArgs['blockIdx'] = '{}'.format(bIdx)
            theseArgs['processAll'] = False
            thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
            dFPath = os.path.join(
                thisDFFolder,
                '{}_{}_{}_df{}.h5'.format(
                    thisBlockBaseName,
                    arguments['window'],
                    arguments['alignQuery'],
                    iteratorSuffix))
            thisDF = pd.read_hdf(dFPath, arguments['unitQuery'])
            # newSegLevel = [currBlockNum for i in range(thisDF.shape[0])]
            thisDF.index = thisDF.index.set_levels([currBlockNum], level='segment')
            listOfDataFrames.append(thisDF)
            currBlockNum += 1


dataDF = pd.concat(listOfDataFrames)
trialInfo = dataDF.index.to_frame().reset_index(drop=True)
cvIterator = iteratorsBySegment[0]
workIdx = cvIterator.work
workingDataDF = dataDF.iloc[workIdx, :]
prf.print_memory_usage('just loaded data, fitting')
nFeatures = dataDF.columns.shape[0]
nCompsToTest = range(1, nFeatures + 1)
scores = compute_scores(
    dataDF, PCA,
    nCompsToTest, cv=cvIterator, verbose=True,
    estimatorKWArgs=dict(svd_solver='full'))
scoresDF = pd.concat(
    {nc: pd.DataFrame(scr) for nc, scr in scores.items()},
    names=['nComponents', 'fold'])
#
pcaMle = PCA(svd_solver='full', n_components='mle')
pcaMle.fit(workingDataDF)
n_components_pca_mle = pcaMle.n_components_
pcaFull = PCA(svd_solver='full')
pcaFull.fit(workingDataDF)
#
lWScores = calc_lw_score(dataDF, cv=cvIterator)
lWForPlot = pd.concat({
    nc: pd.DataFrame({'test_score': lWScores})
    for nc in nCompsToTest}, names=['nComponents', 'fold'])
scoresForPlot = pd.concat(
    {
        'PCA': scoresDF.loc[:, ['test_score']],
        'ledoitWolfMLE': lWForPlot},
    names=['estimator']).reset_index()
#
if arguments['plotting']:
    figureOutputPath = os.path.join(
            figureOutputFolder,
            '{}_{}_{}_dimensionality.pdf'.format(
                blockBaseName,
                arguments['window'], arguments['estimatorName']))
    with PdfPages(figureOutputPath) as pdf:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        sns.lineplot(
            data=scoresForPlot,
            x='nComponents', y='test_score',
            hue='estimator', ci='sem', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        meanScoreMLE = scoresDF.loc[idxSl[n_components_pca_mle, :], 'test_score'].mean()
        line, = ax.plot(n_components_pca_mle, meanScoreMLE, 'g*', label='num. components from MLE')
        handles.append(line)
        labels.append('num. components from MLE')
        ax.legend(handles, labels)
        ax.set_xlabel('number of components')
        ax.set_ylabel('average log-likelihood')
        fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        cumExplVariance = pd.Series(
            np.cumsum(pcaFull.explained_variance_ratio_),
            index=nCompsToTest)
        ax.plot(cumExplVariance)
        ax.plot(n_components_pca_mle, cumExplVariance.loc[n_components_pca_mle], '*')
        ax.set_ylim((0, 1))
        ax.set_xlabel('number of components')
        ax.set_ylabel('explained variance')
        fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        plt.close()
del dataDF
gc.collect()
#
prf.print_memory_usage('Done fitting')

jb.dump(pcaFull, estimatorPath)

alignedAsigsKWargs['unitNames'] = saveUnitNames
alignedAsigsKWargs['unitQuery'] = None
alignedAsigsKWargs.pop('dataQuery', None)
# pdb.set_trace()
estimatorMetadata = {
    'trainingDataPath': os.path.basename(iteratorPath),
    'path': os.path.basename(estimatorPath),
    'name': arguments['estimatorName'],
    'inputBlockSuffix': inputBlockSuffix,
    'blockBaseName': blockBaseName,
    'inputFeatures': saveUnitNames,
    'alignedAsigsKWargs': alignedAsigsKWargs
    }

with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
    pickle.dump(estimatorMetadata, f)
