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
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             load from raw, or regular? [default: False]
    --showFigures                          load from raw, or regular? [default: False]
    --debugging                            load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
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
import os, traceback
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

analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder,
        arguments['analysisName'], arguments['alignFolderName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
#
'''fullEstimatorName = '{}_{}_to_{}{}_{}_{}'.format(
    arguments['estimatorName'],
    arguments['unitQueryLhs'], arguments['unitQueryRhs'],
    iteratorSuffix,
    arguments['window'],
    arguments['alignQuery'])'''
datasetName = arguments['datasetName']
# pdb.set_trace()
fullEstimatorName = '{}_{}'.format(
    arguments['estimatorName'], arguments['datasetName'])
#
estimatorsSubFolder = os.path.join(
    alignSubFolder, 'estimators')
if not os.path.exists(estimatorsSubFolder):
    os.makedirs(estimatorsSubFolder)
dataFramesFolder = os.path.join(alignSubFolder, 'dataframes')
datasetPath = os.path.join(
    dataFramesFolder,
    datasetName + '.h5'
    )
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
#
with open(datasetPath.replace('.h5', '_meta.pickle'), 'rb') as _f:
    loadingMeta = pickle.load(_f)
    iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
    cv_kwargs = loadingMeta.pop('cv_kwargs')
for argName in ['plotting', 'showFigures', 'debugging']:
    loadingMeta['arguments'].pop(argName, None)
arguments.update(loadingMeta['arguments'])


def compute_scores(
        X, estimator,
        nComponentsToTest,
        cv, estimatorKWArgs={},
        return_train_score=True, return_estimator=True,
        verbose=False):
    scores = {}
    for n in nComponentsToTest:
        if verbose:
            print('compute_scores() evaluating with {} components'.format(n))
        instance = estimator(**estimatorKWArgs)
        instance.n_components = n
        scores[n] = cross_validate(
            instance, X.to_numpy(), cv=cv,
            return_train_score=return_train_score,
            return_estimator=return_estimator)
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


def calc_lw_score(
        X, cv):
    result = cross_val_score(
        LedoitWolf(), X.to_numpy(), cv=cv)
    return result


dataDF = pd.read_hdf(datasetPath, datasetName)
# only use zero lag targets    
dataDF = dataDF.xs(0, level='lag', axis='columns')
trialInfo = dataDF.index.to_frame().reset_index(drop=True)
cvIterator = iteratorsBySegment[0]
workIdx = cvIterator.work
workingDataDF = dataDF.iloc[workIdx, :]
prf.print_memory_usage('just loaded data, fitting')
nFeatures = dataDF.columns.shape[0]
#
if arguments['verbose']:
    print('Fitting mle estimator to working dataset...')
pcaMle = PCA(svd_solver='full', n_components='mle')
pcaMle.fit(workingDataDF)
n_components_pca_mle = pcaMle.n_components_
if arguments['verbose']:
    print('Fitting all-dimensional estimator to working dataset...')
pcaFull = PCA(svd_solver='full')
pcaFull.fit(workingDataDF)
#
nCompsToTest = range(1, nFeatures + 1)
if arguments['debugging']:
    nCompsToTest = range(1, 200)
    n_components_pca_mle = min(n_components_pca_mle, 199)
scores = compute_scores(
    dataDF, PCA,
    nCompsToTest, cv=cvIterator, verbose=True,
    estimatorKWArgs=dict(svd_solver='auto'))
scoresDF = pd.concat(
    {nc: pd.DataFrame(scr) for nc, scr in scores.items()},
    names=['nComponents', 'fold'])
#
nCompsMaxMLE = scoresDF.groupby(['nComponents']).mean()['test_score'].idxmax()
maxMLE = scoresDF.xs(nCompsMaxMLE, level='nComponents', axis='index').mean()['test_score']
cumExplVariance = pd.Series(
    np.cumsum(pcaFull.explained_variance_ratio_[:len(nCompsToTest)]),
    index=nCompsToTest)
######
varCutoff = 0.95
nCompsCutoff = cumExplVariance.index[cumExplVariance > varCutoff].min()
######
#
if arguments['verbose']:
    print('Calculating ledoit-wolf # of components...')
lWScores = calc_lw_score(dataDF, cv=cvIterator)
lWForPlot = pd.concat({
    nc: pd.DataFrame({'test_score': lWScores})
    for nc in nCompsToTest}, names=['nComponents', 'fold'])
scoresForPlot = pd.concat(
    {
        'PCA_test': scoresDF.loc[:, ['test_score']],
        'PCA_train': scoresDF.loc[:, ['train_score']],
        'ledoitWolfMLE': lWForPlot},
    names=['estimator']).reset_index()
pdb.set_trace()
if arguments['plotting']:
    figureOutputPath = os.path.join(
            figureOutputFolder,
            '{}_{}_dimensionality.pdf'.format(
                arguments['estimatorName'], datasetName))
    with PdfPages(figureOutputPath) as pdf:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        sns.lineplot(
            data=scoresForPlot,
            x='nComponents', y='test_score',
            hue='estimator', ci='sem', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        try:
            meanScoreMLE = scoresDF.loc[idxSl[n_components_pca_mle, :], 'test_score'].mean()
            textDescr = 'num. components from MLE'
            lineMLE, = ax.plot(n_components_pca_mle, meanScoreMLE, 'g*', label=textDescr)
            labels.append(textDescr)
            handles.append(lineMLE)
        except Exception:
            traceback.print_exc()
            pass
        try:
            textDescr = 'num. components that maximize likelihood'
            lineMax, = ax.plot(nCompsMaxMLE, maxMLE, 'r*', label=textDescr)
            handles.append(lineMax)
            labels.append(textDescr)
        except Exception:
            traceback.print_exc()
            pass
        ax.legend(handles, labels)
        ax.set_xlabel('number of components')
        ax.set_ylabel('average log-likelihood')
        fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        ax.plot(cumExplVariance)
        ax.plot(
            n_components_pca_mle,
            cumExplVariance.loc[n_components_pca_mle],
            '*')
        ax.set_ylim((0, 1))
        ax.set_xlabel('number of components')
        ax.set_ylabel('explained variance')
        fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
del dataDF
gc.collect()
#
prf.print_memory_usage('Done fitting')

jb.dump(pcaFull, estimatorPath)

'''
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
    '''
