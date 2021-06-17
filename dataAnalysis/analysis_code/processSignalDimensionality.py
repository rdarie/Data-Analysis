"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose=verbose                        print diagnostics? [default: 0]
    --debugging                              print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName            filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os, sys
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
from numpy.random import default_rng
rng = default_rng()
idxSl = pd.IndexSlice
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=0.5, color_codes=True)
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# pdb.set_trace()
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'processAll': True, 'debugging': False, 'blockIdx': '2', 'estimatorName': 'pca', 'analysisName': 'default',
        'showFigures': False, 'selectionName': 'lfp_CAR_spectral', 'alignFolderName': 'motion', 'plotting': True,
        'verbose': '1', 'datasetName': 'Block_XL_df_a', 'exp': 'exp202101281100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder,
        arguments['analysisName'], 'dimensionality')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
#
datasetName = arguments['datasetName']
selectionName = arguments['selectionName']
estimatorName = arguments['estimatorName']
fullEstimatorName = '{}_{}_{}'.format(
    estimatorName, datasetName, selectionName)
#
estimatorsSubFolder = os.path.join(
    analysisSubFolder, 'estimators')
dataFramesFolder = os.path.join(
    analysisSubFolder, 'dataframes')
datasetPath = os.path.join(
    dataFramesFolder,
    datasetName + '.h5'
    )
loadingMetaPath = os.path.join(
    dataFramesFolder,
    datasetName + '_{}'.format(selectionName) + '_meta.pickle'
    )
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
estimatorMetaDataPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '_meta.pickle'
    )
#
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'], 'dimensionality')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder)
with open(loadingMetaPath, 'rb') as _f:
    loadingMeta = pickle.load(_f)
    iteratorsBySegment = loadingMeta['iteratorsBySegment']
    cv_kwargs = loadingMeta['cv_kwargs']
    for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
        loadingMeta['arguments'].pop(argName, None)
arguments.update(loadingMeta['arguments'])
cvIterator = iteratorsBySegment[0]
outputSelectionName = '{}_{}'.format(
    selectionName, estimatorName)
# 
validationFeaturesDF = pd.read_hdf(datasetPath, '/{}/data'.format(outputSelectionName))
featureInfo = validationFeaturesDF.columns.to_frame().reset_index(drop=True)
#
scoresDF = pd.read_hdf(estimatorPath, 'full_scores')
estimators = pd.read_hdf(estimatorPath, 'full_estimators')
scoresDF.loc[:, 'estimator'] = estimators
bestEstimators = pd.read_hdf(estimatorPath, 'cv_estimators')
dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
featureColumnFields = dataDF.columns.names
featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
removeAllColumn = False
if removeAllColumn:
    featureMasks = featureMasks.loc[~ featureMasks.all(axis='columns'), :]
workIdx = cvIterator.work
workingDataDF = dataDF.iloc[workIdx, :]
#
lastFoldIdx = cvIterator.get_n_splits()
lOfFeatures = []
dictOfRec = {}
dictOfEVS0 = {}
dictOfCovMats0 = {}
for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
    maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
    trfName = '{}_{}'.format(estimatorName, maskParams['freqBandName'])
    dataGroup = dataDF.loc[:, featureMask]
    nFeatures = dataGroup.columns.shape[0]
    # theseFeatureColumns = featureInfo.loc[featureInfo['freqBandName'] == maskParams['freqBandName'], :]
    lOfFeaturesPerFold = []
    lOfRecPerFold = []  # reconstructions
    dictOfEVS1 = {}
    dictOfCovMats1 = {}
    for foldIdx, estimatorsGroup in estimators.groupby('fold'):
        if foldIdx < lastFoldIdx:
            trainIdx, testIdx = cvIterator.folds[foldIdx]
            trialTypes = ['train', 'test']
        else:
            trainIdx, testIdx = cvIterator.work, cvIterator.validation
            trialTypes = ['work', 'validation']
        foldTestDF = dataGroup.iloc[testIdx, :]
        foldTrainDF = dataGroup.iloc[trainIdx, :]
        thisEstimator = estimators.loc[idxSl[maskParams['freqBandName'], foldIdx]]
        bestEst = bestEstimators.loc[idxSl[maskParams['freqBandName'], foldIdx]]
        dictOfCovMats1[foldIdx] = pd.DataFrame(
            thisEstimator.get_covariance(), index=dataGroup.columns.get_level_values('feature'),
            columns=dataGroup.columns.get_level_values('feature'),
            )
        featureColumns = pd.DataFrame(
            np.nan,
            index=range(thisEstimator.n_components),
            columns=featureColumnFields)
        for fcn in featureColumnFields:
            if fcn == 'feature':
                featureColumns.loc[:, fcn] = [
                    '{}{:0>3d}#0'.format(trfName, nc)
                    for nc in range(1, thisEstimator.n_components + 1)]
            elif fcn == 'lag':
                featureColumns.loc[:, fcn] = 0
            else:
                featureColumns.loc[:, fcn] = maskParams[fcn]
        featTestDF = pd.DataFrame(
            thisEstimator.transform(foldTestDF), index=foldTestDF.index,
            columns=pd.MultiIndex.from_frame(featureColumns))
        featTestDF.loc[:, 'fold'] = foldIdx
        featTestDF.loc[:, 'trialType'] = trialTypes[1]
        featTestDF.set_index(['fold', 'trialType'], append=True, inplace=True)
        lOfFeaturesPerFold.append(featTestDF)
        recTestDF = pd.DataFrame(
            np.tile(thisEstimator.mean_, (featTestDF.shape[0], 1)),
            index=foldTestDF.index, columns=foldTestDF.columns)
        lOfEVS2 = []
        for compIdx in range(thisEstimator.components_.shape[0]):
            eVS = (explained_variance_score(foldTestDF, recTestDF))
            recTestDF += (
                    thisEstimator.components_[compIdx, :].reshape((-1, 1)) *
                    featTestDF.iloc[:, compIdx].to_numpy()).T
            eVSSrs = pd.Series([eVS])
            eVSSrs.index = pd.MultiIndex.from_tuples([(compIdx, trialTypes[1],)], names=['component', 'trialType'])
            lOfEVS2.append(eVSSrs)
        '''recTestDF = pd.DataFrame(
            np.dot(featTestDF.to_numpy(), thisEstimator.components_) + thisEstimator.mean_,
            index=foldTestDF.index, columns=foldTestDF.columns)'''
        recTestDF.loc[:, 'fold'] = foldIdx
        recTestDF.loc[:, 'trialType'] = trialTypes[1]
        recTestDF.set_index(['fold', 'trialType'], append=True, inplace=True)
        lOfRecPerFold.append(recTestDF)
        featTrainDF = pd.DataFrame(
            thisEstimator.transform(foldTrainDF), index=foldTrainDF.index,
            columns=pd.MultiIndex.from_frame(featureColumns))
        featTrainDF.loc[:, 'fold'] = foldIdx
        featTrainDF.loc[:, 'trialType'] = trialTypes[0]
        featTrainDF.set_index(['fold', 'trialType'], append=True, inplace=True)
        lOfFeaturesPerFold.append(featTrainDF)
        recTrainDF = pd.DataFrame(
            np.tile(thisEstimator.mean_, (featTrainDF.shape[0], 1)),
            index=foldTrainDF.index, columns=foldTrainDF.columns)
        for compIdx in range(thisEstimator.components_.shape[0]):
            eVS = (explained_variance_score(foldTrainDF, recTrainDF))
            recTrainDF += (thisEstimator.components_[compIdx, :].reshape((-1, 1)) * featTrainDF.iloc[:, compIdx].to_numpy()).T
            eVSSrs = pd.Series([eVS])
            eVSSrs.index = pd.MultiIndex.from_tuples(
                [(compIdx, trialTypes[0],)], names=['component', 'trialType'])
            lOfEVS2.append(eVSSrs)
        '''# equivalent to:
        recTrainDF = pd.DataFrame(
            np.dot(featTrainDF.to_numpy(), thisEstimator.components_) + thisEstimator.mean_,
            index=foldTrainDF.index, columns=foldTrainDF.columns)'''
        recTrainDF.loc[:, 'fold'] = foldIdx
        recTrainDF.loc[:, 'trialType'] = trialTypes[0]
        recTrainDF.set_index(['fold', 'trialType'], append=True, inplace=True)
        lOfRecPerFold.append(recTrainDF)
        dictOfEVS1[foldIdx] = pd.concat(lOfEVS2)
    dictOfCovMats0[maskParams['freqBandName']] = pd.concat(dictOfCovMats1, names=['fold'])
    dictOfEVS0[maskParams['freqBandName']] = pd.concat(dictOfEVS1, names=['fold', 'component', 'trialType'])
    lOfFeatures.append(pd.concat(lOfFeaturesPerFold))
    dictOfRec[maskParams['freqBandName']] = pd.concat(lOfRecPerFold)
featuresDF = pd.concat(lOfFeatures, axis='columns')
featuresDF.to_hdf(estimatorPath, 'full_features')
del lOfFeatures
eVSDF = pd.concat(dictOfEVS0, names=['freqBandName', 'fold', 'component', 'trialType'])
eVSDF.to_hdf(estimatorPath, 'full_explained_variance')
del dictOfEVS0, dictOfEVS1
covMatDF = pd.concat(dictOfCovMats0, names=['freqBandName', 'fold', 'feature'])
covMatDF.to_hdf(estimatorPath, 'cv_covariance_matrices')
del dictOfCovMats0, dictOfCovMats1
recsNoGT = pd.concat(dictOfRec, names=['freqBandName'])
recsNoGT.to_hdf(estimatorPath, 'reconstructed')
del dictOfRec
#
pdfPath = os.path.join(
    figureOutputFolder, '{}_covariance_matrix_heatmap.pdf'.format(
        fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    fig, ax = plt.subplots()
    ax = sns.heatmap(thisEstimator.get_covariance())
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
# subselect features
nFeats = recsNoGT.groupby('feature', axis='columns').ngroups
if nFeats > 10:
    plotFeatIdxes = rng.choice(
        nFeats, size=10, replace=False)
else:
    plotFeatIdxes = np.arange(nFeats)
plotFeatNames = recsNoGT.columns.get_level_values('feature')[plotFeatIdxes]
#
dataDF.loc[:, 'fold'] = 0
dataDF.loc[:, 'trialType'] = 'ground_truth'
dataDF.set_index(['fold', 'trialType'], append=True, inplace=True)
#
pdfPath = os.path.join(
    figureOutputFolder, '{}_reconstructed_signals.pdf'.format(
        fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
        # if maskParams['freqBandName'] == 'all':
        #     continue
        maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
        dataGroup = dataDF.loc[:, featureMask]
        for featName, group in dataGroup.groupby('feature', axis='columns'):
            if featName not in plotFeatNames:
                continue
            idxMask = recsNoGT.index.get_level_values('freqBandName') == maskParams['freqBandName']
            colMask = recsNoGT.columns.isin(group.columns)
            recDF = recsNoGT.loc[idxMask, colMask].copy()
            recDF.columns = recDF.columns.get_level_values('feature')
            # pdb.set_trace()
            predStack = recDF.stack(recDF.columns.names).to_frame(name='signal').reset_index()
            GT = group.copy()
            GT.columns = GT.columns.get_level_values('feature')
            GTStack = GT.stack(GT.columns.names).to_frame(name='signal').reset_index()
            print('making plot of {}'.format(featName))
            predStack = pd.concat([predStack.loc[:, GTStack.columns], GTStack]).reset_index(drop=True)
            predStack.loc[:, 'trialNum'] = np.nan
            for tIdx, (trialIdx, trialGroup) in enumerate(predStack.groupby(['segment', 'originalIndex', 't'])):
                predStack.loc[trialGroup.index, 'trialNum'] = tIdx
            chooseIndices = rng.choice(predStack['trialNum'].unique(), 25)
            plotPredStack = predStack.loc[predStack['trialNum'].isin(chooseIndices), :]
            # pdb.set_trace()
            g = sns.relplot(
                col='trialNum', col_wrap=5,
                hue='trialType', style='trialType',
                # style='expName',
                x='bin', y='signal', data=plotPredStack,
                kind='line', alpha=0.5, lw=0.5, errorbar='se')
            g.fig.set_size_inches((12, 8))
            g.fig.suptitle('{}'.format(featName))
            g.tight_layout(pad=.3)
            pdf.savefig(bbox_inches='tight')
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

pdfPath = os.path.join(
    figureOutputFolder, '{}_explained_variance.pdf'.format(
        fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    for name, group in eVSDF.groupby('freqBandName'):
        plotDF = group.to_frame(name='signal').reset_index()
        g = sns.relplot(
            data=plotDF, x='component', hue='trialType',
            y='signal', kind='line', alpha=0.5, lw=0.5, errorbar='se')
        g.fig.set_size_inches((12, 8))
        g.fig.suptitle('{}'.format(name))
        g.resize_legend(adjust_subtitles=True)
        g.tight_layout(pad=.3)
        pdf.savefig(bbox_inches='tight')
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
