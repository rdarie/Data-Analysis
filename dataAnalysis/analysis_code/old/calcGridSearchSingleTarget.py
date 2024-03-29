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
    --debugging                            restrict datasets for debugging? [default: False]
    --showFigures                          show plots? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --datasetNameRhs=datasetNameRhs        which trig_ block to pull [default: Block]
    --selectionNameRhs=selectionNameRhs    how to restrict channels? [default: fr_sqrt]
    --datasetNameLhs=datasetNameLhs        which trig_ block to pull [default: Block]
    --selectionNameLhs=selectionNameLhs    how to restrict channels? [default: fr_sqrt]
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
    --selector=selector                    filename if using a unit selector
"""
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'DISPLAY' in os.environ:
    matplotlib.use('QT5Agg')   # generate postscript output
else:
    matplotlib.use('PS')   # generate postscript output
from dask.distributed import Client
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
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
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import sys
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt

idxSl = pd.IndexSlice
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True)

for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
pdb.set_trace()
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'exp': 'exp202101281100', 'datasetNameRhs': 'Block_XL_df_c', 'estimatorName': 'enr',
        'winStop': '400', 'datasetNameLhs': 'Block_XL_df_c', 'analysisName': 'default',
        'plotting': True, 'winStart': '200', 'verbose': '2', 'showFigures': False, 'blockIdx': '2',
        'debugging': False, 'lazy': False, 'alignQuery': 'midPeak', 'processAll': True,
        'selectionNameLhs': 'pedalState', 'window': 'long', 'selector': None,
        'alignFolderName': 'motion', 'selectionNameRhs': 'lfp_CAR'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


if __name__ == '__main__':
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], 'pls')
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    #
    rhsDatasetPath = os.path.join(
        dataFramesFolder,
        arguments['datasetNameRhs'] + '.h5'
        )
    assert os.path.exists(rhsDatasetPath)
    lhsDatasetPath = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '.h5'
        )
    assert os.path.exists(lhsDatasetPath)
    fullEstimatorName = '{}_{}_{}'.format(
        arguments['estimatorName'], arguments['datasetNameLhs'], arguments['selectionNameLhs'])
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    loadingMetaPathLhs = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '_' + arguments['selectionNameLhs'] + '_meta.pickle'
        )
    #
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
    #
    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    workIdx = cvIterator.work
    ###
    estimatorClass = ElasticNet
    estimatorKWArgs = dict()
    gridSearchKWArgs = dict(
        max_iter=10000,
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1.],
        cv=cvIterator)
    '''if arguments['debugging']:
        gridSearchKWArgs['l1_ratio'] = [.1, .7, .99, 1.]
        gridSearchKWArgs['n_alphas'] = 100'''
    #
    '''estimatorClass = SGDRegressor
    estimatorKWArgs = dict(
        max_iter=2000
        )
    gridSearchKWArgs = dict(
        param_grid=dict(
            l1_ratio=[.1, .75, 1.],
            alpha=np.logspace(-4, 1, 4),
            loss=['epsilon_insensitive'],
            ),
        cv=cvIterator,
        scoring='r2'
        )'''
    crossvalKWArgs = dict(
        cv=cvIterator, scoring='r2',
        return_estimator=True,
        return_train_score=True)
    joblibBackendArgs = dict(
        backend='dask'
        )
    lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
    lhsMasks = pd.read_hdf(lhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameLhs']))
    #
    pipelineNameRhs = '{}_{}_{}'.format(
        'pca', arguments['datasetNameRhs'], arguments['selectionNameRhs'])
    pipelinePathRhs = os.path.join(
        estimatorsSubFolder,
        pipelineNameRhs + '.h5'
        )
    pipelineMetaDataPathRhs = os.path.join(
        estimatorsSubFolder,
        pipelineNameRhs + '_meta.pickle'
        )
    workingLhsDF = lhsDF.iloc[workIdx, :]
    workingRhsDF = rhsDF.iloc[workIdx, :]
    nFeatures = lhsDF.columns.shape[0]
    nTargets = rhsDF.columns.shape[0]
    #
    allScores = []
    lhGroupNames = lhsMasks.index.names
    if 'backend' in joblibBackendArgs:
        if joblibBackendArgs['backend'] == 'dask':
            daskClient = Client()
    allGridSearchDict = {}
    # pdb.set_trace()
    for idx, (maskIdx, lhsMask) in enumerate(lhsMasks.iterrows()):
        maskParams = {k: v for k, v in zip(lhsMask.index.names, maskIdx)}
        ####
        if arguments['debugging']:
            if maskParams['freqBandName'] not in ['beta', 'gamma', 'higamma']:
                # if maskParams['lag'] not in [0]:
                continue
        ###
        scores = {}
        gridSearcherDict = {}
        lhGroup = lhsDF.loc[:, lhsMask]
        for columnTuple in rhsDF.columns:
            targetName = columnTuple[0]
            ####
            if arguments['debugging']:
                if targetName not in ['position#0', 'forceMagnitude#0', 'velocity#0', 'velocity_abs#0']:
                    continue
            ###
            print('Fitting {} to {}...'.format(lhsMask.name[-1], targetName))
            scores[targetName], gridSearcherDict[targetName] = tdr.gridSearchHyperparameters(
                lhGroup, rhsDF.loc[:, columnTuple], estimatorClass,
                # verbose=int(arguments['verbose']),
                gridSearchKWArgs=gridSearchKWArgs,
                crossvalKWArgs=crossvalKWArgs,
                estimatorKWArgs=estimatorKWArgs,
                joblibBackendArgs=joblibBackendArgs
                )
        scoresDF = pd.concat(
            {nc: pd.DataFrame(scr) for nc, scr in scores.items()},
            names=['target', 'fold'])
        # pdb.set_trace()
        allGridSearchDict[lhsMask.name] = gridSearcherDict
        '''if not isinstance(groupName, list):
            attrNameList = [groupName]
        else:
            attrNameList = groupName'''
        for i, lhAttr in enumerate(maskIdx):
            lhKey = lhGroupNames[i]
            scoresDF.loc[:, lhKey] = lhAttr
        # scoresDF.loc[:, 'lhsComponents'] = '{}'.format(maskParams)
        allScores.append(scoresDF)
    allScoresDF = pd.concat(allScores)
    allScoresDF.set_index(
        lhGroupNames,
        inplace=True, append=True)
    #
    prf.print_memory_usage('Done fitting')
    if os.path.exists(estimatorPath):
        os.remove(estimatorPath)
    allScoresDF.to_hdf(estimatorPath, 'cv')
    '''loadingMeta['arguments'] = arguments.copy()
    loadingMeta['lhGroupNames'] = lhGroupNames
    loadingMeta['lhsNormalizationParams'] = lhsNormalizationParams
    loadingMeta['rhsNormalizationParams'] = rhsNormalizationParams
    with open(estimatorPath.replace('.h5', '_meta.pickle'), 'wb') as f:
        pickle.dump(loadingMeta, f)'''
    #
    if arguments['plotting']:
        figureOutputPath = os.path.join(
                figureOutputFolder,
                '{}_r2.pdf'.format(fullEstimatorName))
        scoresForPlot = pd.concat(
            {'test': allScoresDF['test_score'], 'train': allScoresDF['train_score']},
            names=['evalType']).to_frame(name='score').reset_index()
        lastFoldIdx = scoresForPlot['fold'].max()
        validationMask = (
            (scoresForPlot['fold'] == lastFoldIdx) &
            (scoresForPlot['evalType'] == 'test'))
        scoresForPlot.loc[validationMask, 'evalType'] = 'validation'
        workingMask = (
            (scoresForPlot['fold'] == lastFoldIdx) &
            (scoresForPlot['evalType'] == 'train'))
        scoresForPlot.loc[workingMask, 'evalType'] = 'work'
        colWrap = np.ceil(np.sqrt(scoresForPlot['maskName'].unique().size)).astype(int)
        with PdfPages(figureOutputPath) as pdf:
            # fig, ax = plt.subplots()
            # fig.set_size_inches(12, 8)
            g = sns.catplot(
                data=scoresForPlot, hue='evalType',
                col='maskName', col_wrap=colWrap,
                x='target', y='score',
                kind='box')
            g.fig.suptitle('R^2')
            newYLims = scoresForPlot['score'].quantile([0.25, 1 - 1e-3]).to_list()
            for ax in g.axes.flat:
                ax.set_xlabel('regression target')
                ax.set_ylabel('R2 of ordinary least squares fit')
                ax.set_ylim(newYLims)
            g.fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    del lhsDF, rhsDF
    # gc.collect()
