"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose                                print diagnostics? [default: False]
    --debugging                              print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
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
    font_scale=1.5, color_codes=True)
import os
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
fullEstimatorName = '{}_{}'.format(
    arguments['estimatorName'], arguments['datasetName'])
#
estimatorsSubFolder = os.path.join(
    alignSubFolder, 'estimators')
if not os.path.exists(estimatorsSubFolder):
    os.makedirs(estimatorsSubFolder)
datasetPath = os.path.join(
    estimatorsSubFolder,
    datasetName + '.h5'
    )
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
scoresDF = pd.read_hdf(estimatorPath, 'cv')
with open(datasetPath.replace('.h5', '_meta.pickle'), 'rb') as _f:
    loadingMeta = pickle.load(_f)
arguments.update(loadingMeta['arguments'])
#
cvIteratorSubfolder = os.path.join(
    alignSubFolder, 'testTrainSplits')
if arguments['iteratorSuffix'] is not None:
    iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
else:
    iteratorSuffix = ''
#
if arguments['processAll']:
    rhsBlockBaseName = arguments['rhsBlockPrefix']
    lhsBlockBaseName = arguments['lhsBlockPrefix']
else:
    rhsBlockBaseName = '{}{:0>3}'.format(
        arguments['rhsBlockPrefix'], arguments['blockIdx'])
    lhsBlockBaseName = '{}{:0>3}'.format(
        arguments['lhsBlockPrefix'], arguments['blockIdx'])
# rhs loading paths
if arguments['rhsBlockSuffix'] is not None:
    rhsBlockSuffix = '_{}'.format(arguments['rhsBlockSuffix'])
else:
    rhsBlockSuffix = ''
if arguments['lhsBlockSuffix'] is not None:
    lhsBlockSuffix = '_{}'.format(arguments['lhsBlockSuffix'])
else:
    lhsBlockSuffix = ''
#
iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
cv_kwargs = loadingMeta['cv_kwargs'].copy()

'''lhGroupNames = loadingMeta['lhGroupNames']
lOfRhsDF = []
lOfLhsDF = []
lOfLhsMasks = []
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
        # only use zero lag targets    
        thisRhsDF = thisRhsDF.xs(0, level='lag', axis='columns')
        thisRhsDF.loc[:, 'expName'] = expName
        thisRhsDF.set_index('expName', inplace=True, append=True)
        #
        lOfRhsDF.append(thisRhsDF)
        thisLhsDF = pd.read_hdf(dFPath, arguments['unitQueryLhs'])
        thisLhsDF.index = thisLhsDF.index.set_levels([currBlockNum], level='segment')
        thisLhsDF.loc[:, 'expName'] = expName
        thisLhsDF.set_index('expName', inplace=True, append=True)
        lOfLhsDF.append(thisLhsDF)
        thisLhsMask = pd.read_hdf(dFPath, arguments['unitQueryLhs'] + '_featureMasks')
        lOfLhsMasks.append(thisLhsMask)
        currBlockNum += 1
lhsDF = pd.concat(lOfLhsDF)
rhsDF = pd.concat(lOfRhsDF)
del lOfRhsDF, lOfLhsDF, thisRhsDF, thisLhsDF

## Normalize lhs
lhsNormalizationParams = loadingMeta['lhsNormalizationParams']
if 'spectral' in arguments['unitQueryLhs']:
    for lhnDict in lhsNormalizationParams[0]:
        featName = lhnDict['feature']
        featMask = lhsDF.columns.get_level_values('feature') == featName
        expName = lhnDict['expName']
        expMask = lhsDF.index.get_level_values('expName') == expName
        print('Pre-normalizing {}, {}'.format(expName, featName))
        meanLevel = lhnDict['mu']
        lhsDF.loc[expMask, featMask] = np.sqrt(lhsDF.loc[expMask, featMask] / meanLevel)
    for lhnDict in lhsNormalizationParams[1]:
        featName = lhnDict['feature']
        featMask = lhsDF.columns.get_level_values('feature') == featName
        print('Final normalizing {}'.format(featName))
        lhsDF.loc[:, featMask] = (lhsDF.loc[:, featMask] - lhnDict['mu']) / lhnDict['sigma']
#
## Normalize rhs
rhsNormalizationParams = loadingMeta['rhsNormalizationParams']
for rhnDict in rhsNormalizationParams[0]:
    featName = rhnDict['feature']
    featMask = rhsDF.columns.get_level_values('feature') == featName
    print('Final normalizing {}'.format(featName))
    rhsDF.loc[:, featMask] = (rhsDF.loc[:, featMask] - rhnDict['mu']) / rhnDict['sigma']'''

lhsDF = pd.read_hdf(datasetPath, 'lhsDF')
rhsDF = pd.read_hdf(datasetPath, 'rhsDF')
lhsMasks = pd.read_hdf(datasetPath, 'lhsFeatureMasks')
#
cvIterator = iteratorsBySegment[0]
workIdx = cvIterator.work
workingLhsDF = lhsDF.iloc[workIdx, :]
workingRhsDF = rhsDF.iloc[workIdx, :]
nFeatures = lhsDF.columns.shape[0]
nTargets = rhsDF.columns.shape[0]
#
lhsFeatureInfo = lhsDF.columns.to_frame().reset_index(drop=True)
workingTrialInfo = workingRhsDF.index.to_frame().reset_index(drop=True)
allPredictionsList = []
# lhsMasks = lOfLhsMasks[0]
lhGroupNames = lhsMasks.index.names
for idx, (attrNameList, lhsMask) in enumerate(lhsMasks.iterrows()):
    maskParams = {k: v for k, v in zip(lhsMask.index.names, attrNameList)}
    # for groupName, scoresGroup in scoresDF.groupby(lhGroupNames):
    '''if not isinstance(groupName, list):
        attrNameList = [groupName]
    else:
        attrNameList = groupName'''
    # lhsMask = np.ones(lhsDF.shape[1], dtype=bool)
    '''scoresMask = np.ones(scoresDF.shape[1], dtype=bool)
    pdb.set_trace()
    for attrIdx, attrName in enumerate(attrNameList):
        attrKey = lhGroupNames[attrIdx]
        # lhsMask = lhsMask & (lhsFeatureInfo[attrKey] == attrName).to_numpy()
        scoresMask = scoresMask & (lhsFeatureInfo[attrKey] == attrName).to_numpy()'''
    lhsGroup = lhsDF.loc[:, lhsMask]
    # scoresGroup = scoresDF.loc[:, scoresMask]
    predListPerTarget = []
    for targetName, rhsGroup in rhsDF.groupby('feature', axis='columns'):
        predListPerFold = []
        for foldIdx, (trainIdx, testIdx) in enumerate(cvIterator.folds):
            foldLHS = lhsGroup.iloc[trainIdx, :]
            foldRHS = rhsGroup.iloc[trainIdx, :]
            '''foldMaskScores = (
                (scoresGroup.index.get_level_values('fold') == foldIdx) &
                (scoresGroup.index.get_level_values('target') == targetName))
            assert scoresGroup.loc[foldMaskScores, 'estimator'].size == 1'''
            #
            theseIndices = (targetName, foldIdx) + attrNameList
            '''theseIndicesTemp = [ti for ti in ((targetName, foldIdx) + attrNameList)] + ['{}'.format(maskParams)]
            theseIndices = tuple(ti for ti in theseIndicesTemp)'''
            #
            if theseIndices not in scoresDF.index:
                continue
            thisEstimator = scoresDF.loc[theseIndices, 'estimator']
            # pdb.set_trace()
            foldPrediction = pd.DataFrame(
                thisEstimator.predict(foldLHS.to_numpy()), index=foldRHS.index,
                columns=foldRHS.columns)
            tempIndexFrame = foldPrediction.index.to_frame().reset_index(drop=True)
            for attrIdx, attrName in enumerate(attrNameList):
                attrKey = lhGroupNames[attrIdx]
                tempIndexFrame.loc[:, attrKey] = attrName
            foldPrediction.index = pd.MultiIndex.from_frame(tempIndexFrame)
            predListPerFold.append(foldPrediction)
        if len(predListPerFold):
            targetPredictions = pd.concat(predListPerFold)
            predListPerTarget.append(targetPredictions)
    if len(predListPerTarget):
        groupPredictions = pd.concat(predListPerTarget, axis='columns')
        allPredictionsList.append(groupPredictions)
#
# target values do not have meaningful attributes from the predictor group
predictedDF = pd.concat(allPredictionsList)
predictedDF.to_hdf(estimatorPath, 'predictions')
