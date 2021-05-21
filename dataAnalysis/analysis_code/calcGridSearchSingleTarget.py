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
from dask.distributed import Client
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
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.svm import LinearSVR
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


if __name__ == '__main__':
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
    '''cvIteratorSubfolder = os.path.join(
        alignSubFolder, 'testTrainSplits')'''
    if arguments['iteratorSuffix'] is not None:
        iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
    else:
        iteratorSuffix = ''
    #
    datasetName = '{}_to_{}{}_{}_{}'.format(
        arguments['unitQueryRhs'], arguments['unitQueryLhs'],
        iteratorSuffix,
        arguments['window'],
        arguments['alignQuery'])
    rhsDatasetName = '{}{}_{}_{}'.format(
        arguments['unitQueryRhs'],
        iteratorSuffix,
        arguments['window'],
        arguments['alignQuery'])
    lhsDatasetName = '{}{}_{}_{}'.format(
        arguments['unitQueryLhs'],
        iteratorSuffix,
        arguments['window'],
        arguments['alignQuery'])
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], datasetName)
    estimatorsSubFolder = os.path.join(
        alignSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    rhsDatasetPath = os.path.join(
        dataFramesFolder,
        rhsDatasetName + '.h5'
        )
    assert os.path.exists(rhsDatasetPath)
    lhsDatasetPath = os.path.join(
        dataFramesFolder,
        lhsDatasetName + '.h5'
        )
    assert os.path.exists(lhsDatasetPath)
    #
    '''iteratorPath = os.path.join(
        cvIteratorSubfolder,
        '{}_{}_{}{}_cvIterators.pickle'.format(
            rhsBlockBaseName,
            arguments['window'],
            arguments['alignQuery'],
            iteratorSuffix))'''
    iteratorPath = rhsDatasetPath.replace('.h5', '_meta.pickle')
    #
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
    #
    if arguments['lhsBlockSuffix'] is not None:
        lhsBlockSuffix = '_{}'.format(arguments['lhsBlockSuffix'])
    else:
        lhsBlockSuffix = ''
    #
    triggeredLhsPath = os.path.join(
        alignSubFolder,
        lhsBlockBaseName + '{}_{}.nix'.format(
            lhsBlockSuffix, arguments['window']))
    #
    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    cv_kwargs = loadingMeta['cv_kwargs'].copy()
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
    lhsDF = pd.read_hdf(lhsDatasetPath, arguments['unitQueryLhs'])
    rhsDF = pd.read_hdf(rhsDatasetPath, arguments['unitQueryRhs'])
    lhsMasks = pd.read_hdf(lhsDatasetPath, arguments['unitQueryLhs'] + '_featureMasks')
    #
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
