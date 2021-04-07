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
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --rhsBlockSuffix=rhsBlockSuffix        which trig_ block to pull [default: pca]
    --rhsBlockPrefix=rhsBlockPrefix        which trig_ block to pull [default: Block]
    --unitQueryRhs=unitQueryRhs            how to restrict channels? [default: fr_sqrt]
    --lhsBlockSuffix=lhsBlockSuffix        which trig_ block to pull [default: pca]
    --lhsBlockPrefix=lhsBlockPrefix        which trig_ block to pull [default: Block]
    --unitQueryLhs=unitQueryLhs            how to restrict channels? [default: fr_sqrt]
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
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
idxSl = pd.IndexSlice

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'lhsBlockPrefix': 'Block', 'blockIdx': '2', 'estimatorName': 'pls',
        'loadFromFrames': True, 'plotting': True, 'selector': None, 'winStart': '200',
        'alignFolderName': 'motion', 'processAll': True, 'unitQueryLhs': 'lfp',
        'window': 'L', 'verbose': False, 'rhsBlockPrefix': 'Block',
        'unitQueryRhs': 'pedalPosition', 'lhsBlockSuffix': 'lfp', 'winStop': '400',
        'iteratorSuffix': 'a', 'analysisName': 'default', 'lazy': False, 'exp': 'exp202101201100',
        'alignQuery': 'starting', 'rhsBlockSuffix': 'rig'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
calcSubFolder = os.path.join(alignSubFolder, 'pls')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder)
dataFramesFolder = os.path.join(alignSubFolder, 'dataframes')

if arguments['processAll']:
    rhsBlockBaseName = arguments['rhsBlockPrefix']
    lhsBlockBaseName = arguments['lhsBlockPrefix']
else:
    rhsBlockBaseName = '{}{:0>3}'.format(
        arguments['rhsBlockPrefix'], arguments['blockIdx'])
    lhsBlockBaseName = '{}{:0>3}'.format(
        arguments['lhsBlockPrefix'], arguments['blockIdx'])

fullEstimatorName = '{}_to_{}_{}_{}_{}'.format(
    arguments['unitQueryLhs'], arguments['unitQueryRhs'],
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])

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
        rhsBlockBaseName,
        arguments['window'],
        arguments['alignQuery'],
        iteratorSuffix))
with open(iteratorPath, 'rb') as f:
    loadingMeta = pickle.load(f)

# rhs loading paths
if arguments['rhsBlockSuffix'] is not None:
    rhsBlockSuffix = '_{}'.format(arguments['rhsBlockSuffix'])
else:
    rhsBlockSuffix = ''
triggeredRhsPath = os.path.join(
    alignSubFolder,
    rhsBlockBaseName + '{}_{}.nix'.format(
        rhsBlockSuffix, arguments['window']))

if arguments['lhsBlockSuffix'] is not None:
    lhsBlockSuffix = '_{}'.format(arguments['lhsBlockSuffix'])
else:
    lhsBlockSuffix = ''

triggeredLhsPath = os.path.join(
    alignSubFolder,
    lhsBlockBaseName + '{}_{}.nix'.format(
        lhsBlockSuffix, arguments['window']))
#
#
iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
cv_kwargs = loadingMeta.pop('cv_kwargs')
#
def compute_scores(
        X, y, estimator,
        nComponentsToTest,
        estimatorKWArgs={},
        crossvalKWArgs={},
        verbose=False):
    scores = {}
    for n in nComponentsToTest:
        if verbose:
            print('evaluating with {} components'.format(n))
        instance = estimator(**estimatorKWArgs)
        instance.n_components = n
        scores[n] = cross_validate(instance, X, y, **crossvalKWArgs)
    return scores

lOfRhsDF = []
lOfLhsDF = []
if not arguments['loadFromFrames']:
    alignedAsigsKWargs = loadingMeta.pop('alignedAsigsKWargs')
    loadRhsKWArgs = alignedAsigsKWargs.copy()
    rhsArgs = arguments.copy()
    rhsArgs['unitQuery'] = rhsArgs['unitQueryRhs']
    loadRhsKWArgs['unitNames'], loadRhsKWArgs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **rhsArgs)
    rhsArgs['verbose'] = rhsArgs['verbose']
    #
    loadLhsKWArgs = alignedAsigsKWargs.copy()
    lhsArgs = arguments.copy()
    lhsArgs['unitQuery'] = lhsArgs['unitQueryLhs']
    loadLhsKWArgs['unitNames'], loadLhsKWArgs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **lhsArgs)
    lhsArgs['verbose'] = lhsArgs['verbose']
    #
    if arguments['verbose']:
        prf.print_memory_usage('before load data')
        print('loading {}'.format(triggeredRhsPath))
    dataRhsReader, dataRhsBlock = ns5.blockFromPath(
        triggeredRhsPath, lazy=arguments['lazy'])
    if triggeredLhsPath == triggeredRhsPath:
        dataLhsReader, dataLhsBlock = dataRhsReader, dataRhsBlock
    else:
        print('loading {}'.format(triggeredLhsPath))
        dataLhsReader, dataLhsBlock = ns5.blockFromPath(
            triggeredLhsPath, lazy=arguments['lazy'])
    nSeg = len(dataRhsBlock.segments)
    for segIdx in range(nSeg):
        if arguments['verbose']:
            prf.print_memory_usage('fitting on segment {}'.format(segIdx))
        if 'listOfROIMasks' in loadingMeta:
            loadLhsKWArgs.update({'finalIndexMask': loadingMeta['listOfROIMasks'][segIdx]})
            loadRhsKWArgs.update({'finalIndexMask': loadingMeta['listOfROIMasks'][segIdx]})
        lhsDF = ns5.alignedAsigsToDF(
            dataLhsBlock,
            whichSegments=[segIdx],
            **loadLhsKWArgs)
        if arguments['verbose']:
            prf.print_memory_usage('loaded LHS')
        rhsDF = ns5.alignedAsigsToDF(
            dataRhsBlock,
            whichSegments=[segIdx],
            **loadRhsKWArgs)
        if arguments['verbose']:
            prf.print_memory_usage('loaded RHS')
        assert lhsDF.shape[0] == rhsDF.shape[0]
        lOfRhsDF.append(rhsDF)
        lOfLhsDF.append(lhsDF)
    if arguments['lazy']:
        dataRhsReader.file.close()
        if not (triggeredLhsPath == triggeredRhsPath):
            dataLhsReader.file.close()
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
            theseArgs['inputBlockSuffix'] = theseArgs['rhsBlockSuffix']
            theseArgs['inputBlockPrefix'] = theseArgs['rhsBlockPrefix']
            thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
            dFPath = os.path.join(
                thisDFFolder,
                '{}_{}_{}_df{}.h5'.format(
                    thisBlockBaseName,
                    arguments['window'],
                    arguments['alignQuery'],
                    iteratorSuffix))
            thisRhsDF = pd.read_hdf(dFPath, arguments['unitQueryRhs'])
            thisRhsDF.index = thisRhsDF.index.set_levels([currBlockNum], level='segment')
            lOfRhsDF.append(thisRhsDF)
            thisLhsDF = pd.read_hdf(dFPath, arguments['unitQueryLhs'])
            thisLhsDF.index = thisLhsDF.index.set_levels([currBlockNum], level='segment')
            lOfLhsDF.append(thisLhsDF)
            currBlockNum += 1
lhsDF = pd.concat(lOfLhsDF)
rhsDF = pd.concat(lOfRhsDF)
#####
cvIterator = iteratorsBySegment[0]
workIdx = cvIterator.work
workingLhsDF = lhsDF.iloc[workIdx, :]
workingRhsDF = rhsDF.iloc[workIdx, :]
nFeatures = lhsDF.columns.shape[0]
nTargets = rhsDF.columns.shape[0]
nCompsToTest = range(1, min(nTargets, nFeatures) + 1)
scores = compute_scores(
    lhsDF, rhsDF, PLSRegression,
    nCompsToTest, verbose=True,
    crossvalKWArgs=dict(
        cv=cvIterator, scoring='r2',
        return_train_score=True),
    # estimatorKWArgs=dict(svd_solver='full')
    )

scoresDF = pd.concat(
    {nc: pd.DataFrame(scr) for nc, scr in scores.items()},
    names=['nComponents', 'fold'])
'''
if arguments['plotting']:
    figureOutputPath = os.path.join(
            figureOutputFolder,
            '{}_{}_{}_xxxx.pdf'.format(
                rhsBlockBaseName,
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
del lhsDF, rhsDF
gc.collect()
#
prf.print_memory_usage('Done fitting')

jb.dump(pcaFull, estimatorPath)

alignedAsigsKWargs['unitNames'] = saveUnitNames
alignedAsigsKWargs['unitQuery'] = None
alignedAsigsKWargs.pop('dataQuery')
estimatorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(estimatorPath),
    'name': arguments['estimatorName'],
    'inputBlockSuffix': inputBlockSuffix,
    'blockBaseName': blockBaseName,
    'inputFeatures': saveUnitNames,
    'alignedAsigsKWargs': alignedAsigsKWargs
    }

with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
    pickle.dump(
        estimatorMetadata, f)
'''