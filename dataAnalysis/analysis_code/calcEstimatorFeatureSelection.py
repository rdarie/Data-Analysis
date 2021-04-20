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
    --plotting                             make plots? [default: False]
    --showFigures                          show plots? [default: False]
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
from sklearn.linear_model import LinearRegression
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
        'iteratorSuffix': 'a', 'alignFolderName': 'motion',
        'processAll': True, 'exp': 'exp202101201100', 'analysisName': 'default',
        'blockIdx': '2', 'rhsBlockPrefix': 'Block', 'verbose': False,
        'lhsBlockSuffix': 'lfp_CAR_spectral', 'unitQueryLhs': 'lfp_CAR_spectral',
        'rhsBlockSuffix': 'rig', 'unitQueryRhs': 'jointAngle',
        'loadFromFrames': True, 'estimatorName': 'ols_lfp_CAR_ja',
        'alignQuery': 'starting', 'winStop': '400', 'window': 'L', 'selector': None, 'winStart': '200',
        'plotting': True, 'lazy': False, 'lhsBlockPrefix': 'Block',
        'showFigures': True}
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

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder,
        arguments['analysisName'], arguments['alignFolderName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
#
cvIteratorSubfolder = os.path.join(
    alignSubFolder, 'testTrainSplits')
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

fullEstimatorName = '{}_{}_to_{}{}_{}_{}'.format(
    arguments['estimatorName'],
    arguments['unitQueryLhs'], arguments['unitQueryRhs'],
    iteratorSuffix,
    arguments['window'],
    arguments['alignQuery'])
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
iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
cv_kwargs = loadingMeta['cv_kwargs'].copy()
#
def compute_scores(
        X, y, estimator,
        estimatorKWArgs={},
        crossvalKWArgs={},
        verbose=False):
    instance = estimator(**estimatorKWArgs)
    scores = cross_validate(instance, X, y, **crossvalKWArgs)
    return scores

lOfRhsDF = []
lOfLhsDF = []
if not arguments['loadFromFrames']:
    alignedAsigsKWargs = loadingMeta['alignedAsigsKWargs'].copy()
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
    experimentsToAssemble = loadingMeta['experimentsToAssemble'].copy()
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

allScores = []
lhGroupNames = ['freqBandName']
for groupName, lhGroup in lhsDF.groupby(lhGroupNames, axis='columns'):
    scores = {}
    # pdb.set_trace()
    for columnTuple in rhsDF.columns:
        targetName = columnTuple[0]
        scores[targetName] = compute_scores(
            lhGroup, rhsDF.loc[:, columnTuple], LinearRegression,
            verbose=True,
            crossvalKWArgs=dict(
                cv=cvIterator, scoring='r2',
                return_estimator=True,
                return_train_score=True),
            # estimatorKWArgs=dict(svd_solver='full')
            )
    scoresDF = pd.concat(
        {nc: pd.DataFrame(scr) for nc, scr in scores.items()},
        names=['target', 'fold'])
    if not isinstance(groupName, list):
        attrNameList = [groupName]
    else:
        attrNameList = groupName
    for i, lhAttr in enumerate(lhGroupNames):
        scoresDF.loc[:, lhAttr] = attrNameList[i]
    allScores.append(scoresDF)
allScoresDF = pd.concat(allScores)
allScoresDF.set_index(lhGroupNames, inplace=True, append=True)

if arguments['plotting']:
    figureOutputPath = os.path.join(
            figureOutputFolder,
            '{}_{}_{}_r2.pdf'.format(
                rhsBlockBaseName,
                arguments['window'], arguments['estimatorName']))
    scoresForPlot = pd.concat(
        {'test': allScoresDF['test_score'], 'train': allScoresDF['train_score']},
        names=['evalType']).to_frame(name='score').reset_index()
    with PdfPages(figureOutputPath) as pdf:
        # fig, ax = plt.subplots()
        # fig.set_size_inches(12, 8)
        g = sns.catplot(
            data=scoresForPlot, hue='evalType',
            col='target',
            x='freqBandName', y='score',
            kind='box')
        g.fig.suptitle('R^2')
        for ax in g.axes.flat:
            ax.set_xlabel('regression target')
            ax.set_ylabel('R2 of ordinary least squares fit')
        g.fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
del lhsDF, rhsDF
gc.collect()
#
prf.print_memory_usage('Done fitting')
estimatorsSubFolder = os.path.join(
    alignSubFolder, 'estimators')
if not os.path.exists(estimatorsSubFolder):
    os.makedirs(estimatorsSubFolder)
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
if os.path.exists(estimatorPath):
    os.remove(estimatorPath)
allScoresDF.to_hdf(estimatorPath, 'cv')
loadingMeta['arguments'] = arguments.copy()
loadingMeta['lhGroupNames'] = lhGroupNames
with open(estimatorPath.replace('.h5', '_meta.pickle'), 'wb') as f:
    pickle.dump(loadingMeta, f)