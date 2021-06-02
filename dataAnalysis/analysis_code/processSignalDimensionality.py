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
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=0.5, color_codes=True)
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
idxSl = pd.IndexSlice

for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# pdb.set_trace()
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
    'analysisName': 'default', 'showFigures': True, 'exp': 'exp202101201100', 'processAll': True,
    'verbose': False, 'plotting': True, 'fullEstimatorName': 'ols_lfp_CAR_spectral_to_jointAngle_a_L_starting',
    'alignFolderName': 'motion', 'blockIdx': '2', 'correctFreqBandName': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    scratchPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings'
    scratchFolder = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202101201100-Rupert'
    figureFolder = '/gpfs/data/dborton/rdarie/Neural Recordings/processed/202101201100-Rupert/figures'
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
selectionName = arguments['selectionName']
estimatorName = arguments['estimatorName']
fullEstimatorName = '{}_{}'.format(
    estimatorName, arguments['datasetName'])
#
estimatorsSubFolder = os.path.join(
    alignSubFolder, 'estimators')
dataFramesFolder = os.path.join(
    alignSubFolder, 'dataframes')
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
validationFeaturesDF = pd.read_hdf(datasetPath, outputSelectionName)
featureInfo = validationFeaturesDF.columns.to_frame().reset_index(drop=True)
#
scoresDF = pd.read_hdf(estimatorPath, 'cv')
estimators = pd.read_hdf(estimatorPath, 'cv_estimators')
dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
removeAllColumn = True
if removeAllColumn:
    featureMasks = featureMasks.loc[~ featureMasks.all(axis='columns'), :]
workIdx = cvIterator.work
workingDataDF = dataDF.iloc[workIdx, :]
#
lastFoldIdx = cvIterator.get_n_splits()
lOfFeatures = []
lOfRec = []
for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
    maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
    dataGroup = dataDF.loc[:, featureMask]
    nFeatures = dataGroup.columns.shape[0]
    theseFeatureColumns = featureInfo.loc[featureInfo['freqBandName'] == maskParams['freqBandName'], :]
    lOfFeaturesPerFold = []
    lOfRecPerFold = [] #reconstructions
    for foldIdx, estimatorsGroup in estimators.groupby('fold'):
        if foldIdx < lastFoldIdx:
            trainIdx, testIdx = cvIterator.folds[foldIdx]
            evalTypes = ['train', 'test']
        else:
            trainIdx, testIdx = cvIterator.work, cvIterator.validation
            evalTypes = ['work', 'validation']
        foldTestDF = dataGroup.iloc[testIdx, :]
        foldTrainDF = dataGroup.iloc[trainIdx, :]
        thisEstimator = estimators.loc[idxSl[maskParams['freqBandName'], foldIdx]]
        #
        featTestDF = pd.DataFrame(
            thisEstimator.transform(foldTestDF), index=foldTestDF.index,
            columns=pd.MultiIndex.from_frame(theseFeatureColumns))
        featTestDF.loc[:, 'fold'] = foldIdx
        featTestDF.loc[:, 'evalType'] = evalTypes[1]
        featTestDF.set_index(['fold', 'evalType'], append=True, inplace=True)
        lOfFeaturesPerFold.append(featTestDF)
        # pdb.set_trace()
        recTestDF = pd.DataFrame(
            np.dot(featTestDF.to_numpy(), thisEstimator.components_) * np.sqrt(thisEstimator.noise_variance_) + thisEstimator.mean_,
            index=foldTestDF.index, columns=foldTestDF.columns)
        recTestDF.loc[:, 'fold'] = foldIdx
        recTestDF.loc[:, 'evalType'] = evalTypes[1]
        recTestDF.set_index(['fold', 'evalType'], append=True, inplace=True)
        lOfRecPerFold.append(recTestDF)
        featTrainDF = pd.DataFrame(
            thisEstimator.transform(foldTrainDF), index=foldTrainDF.index,
            columns=pd.MultiIndex.from_frame(theseFeatureColumns))
        featTrainDF.loc[:, 'fold'] = foldIdx
        featTrainDF.loc[:, 'evalType'] = evalTypes[0]
        featTrainDF.set_index(['fold', 'evalType'], append=True, inplace=True)
        lOfFeaturesPerFold.append(featTrainDF)
        recTrainDF = pd.DataFrame(
            np.dot(featTrainDF.to_numpy(), thisEstimator.components_) + thisEstimator.mean_,
            index=foldTrainDF.index, columns=foldTrainDF.columns)
        recTrainDF.loc[:, 'fold'] = foldIdx
        recTrainDF.loc[:, 'evalType'] = evalTypes[0]
        recTrainDF.set_index(['fold', 'evalType'], append=True, inplace=True)
        lOfRecPerFold.append(recTrainDF)
    lOfFeatures.append(pd.concat(lOfFeaturesPerFold))
    lOfRec.append(pd.concat(lOfRecPerFold))
featuresDF = pd.concat(lOfFeatures, axis='columns')
#
dataDF.loc[:, 'fold'] = 0
dataDF.loc[:, 'evalType'] = 'ground_truth'
dataDF.set_index(['fold', 'evalType'], append=True, inplace=True)
#
recDF = pd.concat([
    pd.concat(lOfRec, axis='columns'),
    dataDF])
recDF.columns = recDF.columns.get_level_values('feature')
#
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisName'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder)
# pdb.set_trace()
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}_covariance_matrix_heatmap.pdf'.format(
        datasetName, fullEstimatorName))
with PdfPages(pdfPath) as pdf:
    fig, ax = plt.subplots()
    ax = sns.heatmap(thisEstimator.get_covariance())
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}_reconstructed_signals.pdf'.format(
        datasetName, fullEstimatorName))
#
with PdfPages(pdfPath) as pdf:
    for name, group in recDF.groupby('feature', axis='columns'):
        print('making plot of {}'.format(name))
        # pdb.set_trace()
        predStack = group.stack(group.columns.names).to_frame(name='signal').reset_index()
        predStack.loc[:, 'trialNum'] = np.nan
        for tIdx, (trialIdx, trialGroup) in enumerate(predStack.groupby(['segment', 'originalIndex', 't'])):
            predStack.loc[trialGroup.index, 'trialNum'] = tIdx
        chooseIndices = np.random.choice(predStack['trialNum'].unique(), 25)
        plotPredStack = predStack.loc[predStack['trialNum'].isin(chooseIndices), :]
        # pdb.set_trace()
        g = sns.relplot(
            col='trialNum', col_wrap=5,
            hue='evalType', style='expName',
            x='bin', y='signal', data=plotPredStack, kind='line', errorbar='se')
        g.fig.set_size_inches((12, 8))
        g.fig.suptitle('{}'.format(name))
        g.fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
