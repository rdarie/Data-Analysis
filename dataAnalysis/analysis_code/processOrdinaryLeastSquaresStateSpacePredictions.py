"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --oneTrialOnly                           show plots? [default: False]
    --verbose=verbose                        print diagnostics?
    --debugging                              print diagnostics? [default: False]
    --memoryEfficientLoad                    print diagnostics? [default: False]
    --forceReprocess                         print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
    --eraMethod=eraMethod                    append a name to the resulting blocks? [default: ERA]
"""

import logging
logging.captureWarnings(True)
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from dask.distributed import Client, LocalCluster
import os, traceback
from tqdm import tqdm
import contextlib
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_rd', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'pls_select_scaled', 'forceReprocess': True,
        'blockIdx': '2', 'exp': 'exp202101271100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.plotting.aligned_signal_plots as asp
from dataAnalysis.custom_transformers.tdr import reconstructionR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
from datetime import datetime
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn_pandas import gen_features, DataFrameMapper
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer, r2_score
import joblib as jb
import patsy
from sklego.preprocessing import PatsyTransformer
import dill as pickle
pickle.settings['recurse'] = True
import gc, sys
from copy import deepcopy
import colorsys
for arg in sys.argv:
    print(arg)
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .2,
        'lines.markersize': .4,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 9
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV


if __name__ == '__main__':
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')

    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'], 'regression')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    #
    datasetName = arguments['datasetName']
    fullEstimatorName = '{}_{}'.format(
        arguments['estimatorName'], arguments['datasetName'])
    #
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    dataFramesFolder = os.path.join(
        analysisSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    estimatorMetaDataPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_meta.pickle'
        )
    with open(estimatorMetaDataPath, 'rb') as _f:
        estimatorMeta = pickle.load(_f)
    #
    loadingMetaPath = estimatorMeta['loadingMetaPath']
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        iteratorOpts = loadingMeta['iteratorOpts']
        binInterval = iteratorOpts['forceBinInterval'] if (iteratorOpts['forceBinInterval'] is not None) else rasterOpts['binInterval']
    #
    transferFuncPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '_{}_tf.h5'.format(arguments['eraMethod'])
        )

    for hIdx, histOpts in enumerate(addEndogHistoryTerms):
        locals().update({'enhto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    for hIdx, histOpts in enumerate(addExogHistoryTerms):
        locals().update({'exhto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
    thisEnv = patsy.EvalEnvironment.capture()

    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    lastFoldIdx = cvIterator.n_splits
    #
    selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
    selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
    #
    lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
    lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
    allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets').xs(arguments['estimatorName'], level='regressorName')
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    ##
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    modelsToTestDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'modelsToTest')
    #
    stimConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'stimConditionLookup')
    kinConditionLookup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'kinConditionLookup')
    ################ collect estimators and scores
    scoresStack = pd.read_hdf(estimatorPath, 'processedScores')
    #
    # Index(['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'foldType',
    #    'trialType', 'electrode', 'trialAmplitude', 'trialRateInHz',
    #    'pedalMovementCat', 'pedalDirection', 'pedalSizeCat', 'bin', 'trialUID',
    #    'conditionUID','designAsLabel', 'fullDesign', 'fullDesignAsLabel'
    trialInfo = lhsDF.index.to_frame().reset_index(drop=True)
    # list of non-redundant indices
    notRedundantTrialInfo = ['trialUID', 'conditionUID'] ################
    redundantTrialInfo = ['electrode', 'trialAmplitude', 'trialRateInHz', 'pedalMovementCat', 'pedalDirection', 'pedalSizeCat']
    trialMetadataLookup = trialInfo.drop_duplicates(notRedundantTrialInfo).set_index(notRedundantTrialInfo).loc[:, redundantTrialInfo]
    notRedundantModelInfo = ['lhsMaskIdx', 'rhsMaskIdx'] ################
    redundantModelInfo = ['designAsLabel', 'fullDesign', 'fullDesignAsLabel', 'design']
    modelMetadataLookup = scoresStack.drop_duplicates(notRedundantModelInfo).set_index(notRedundantModelInfo).loc[:, redundantModelInfo]
    for cN in lhsMasksInfo.columns:
        if cN not in modelMetadataLookup:
            modelMetadataLookup.loc[:, cN] = modelMetadataLookup.index.get_level_values('lhsMaskIdx').map(lhsMasksInfo[cN])
    for cN in rhsMasksInfo.columns:
        if cN not in modelMetadataLookup:
            modelMetadataLookup.loc[:, cN] = modelMetadataLookup.index.get_level_values('rhsMaskIdx').map(rhsMasksInfo[cN])
    notRedundantOtherInfo = ['bin', 'target', 'fold']
    infoOnlyInPredDF = ['trialType', 'foldType', ]
    predIndexNames = notRedundantTrialInfo + notRedundantModelInfo + notRedundantOtherInfo + infoOnlyInPredDF
    #
    modelMetadataLookup.to_hdf(transferFuncPath, 'modelMetadataLookup')
    trialMetadataLookup.to_hdf(transferFuncPath, 'trialMetadataLookup')
    #
    memoryEfficientLoad = True
    if memoryEfficientLoad:
        predDF = None
        R2Per = None
        ccDF = None
        inputDrivenDF = None
        oskDF = None
    else:
        # predDF = None
        predList = []
        R2PerList = []
        ccList = []
        inputDrivenList = []
        oskList = []
    #
    if processSlurmTaskCount is not None:
        slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
        allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
        lhsTargetMask = allTargetsDF.index.get_level_values('lhsMaskIdx').isin(lhsMasksOfInterest['plotPredictions'])
        jobsToLoad = allTargetsDF.loc[lhsTargetMask, 'parentProcess'].unique()
        #
        for modelIdx in range(processSlurmTaskCount):
            thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(modelIdx))
            if os.path.exists(thisEstimatorPath):
                prf.print_memory_usage('Loading predictions from {}'.format(thisEstimatorPath))
                with pd.HDFStore(thisEstimatorPath) as store:
                    try:
                        thisPred = pd.read_hdf(store, 'predictions')
                        if memoryEfficientLoad:
                            if predDF is None:
                                predDF = thisPred
                            else:
                                predDF = predDF.append(thisPred)
                        else:
                            predList.append(thisPred)
                    except Exception:
                        print('#' * 30)
                        traceback.print_exc()
                        print('#' * 30)
            # load transfer funs if they exist
            thisTFPath = transferFuncPath.replace('_tf.h5', '_{}_tf.h5'.format(modelIdx))
            if os.path.exists(thisTFPath):
                prf.print_memory_usage('Loading state space predictions from {}'.format(thisTFPath))
                thisOSK = pd.read_hdf(thisTFPath, 'oneStepKalman')
                thisInputDriven = pd.read_hdf(thisTFPath, 'inputDriven')
                if memoryEfficientLoad:
                    if inputDrivenDF is None:
                        inputDrivenDF = thisInputDriven
                        oskDF = thisOSK
                    else:
                        inputDrivenDF = inputDrivenDF.append(thisInputDriven)
                        oskDF = oskDF.append(thisOSK)
                else:
                    inputDrivenList.append(thisInputDriven)
                    oskList.append(thisOSK)
    del thisInputDriven, thisOSK, thisPred
    gc.collect()
    prf.print_memory_usage('concatenating predictions from .h5 array')
    if not memoryEfficientLoad:
        predDF = pd.concat(predList, copy=False)
        del predList
        inputDrivenDF = pd.concat(inputDrivenList, copy=False)
        del inputDrivenList
        oskDF = pd.concat(oskList, copy=False)
        del oskList
    ###
    predDF = predDF.droplevel([lN for lN in predDF.index.names if lN not in predIndexNames])
    inputDrivenDF = inputDrivenDF.droplevel([lN for lN in inputDrivenDF.index.names if lN not in predIndexNames])
    oskDF = oskDF.droplevel([lN for lN in oskDF.index.names if lN not in predIndexNames])
    ####
    assert (inputDrivenDF.index == oskDF.index).all()
    idTrialInfo = inputDrivenDF.index.to_frame().reset_index(drop=True)
    predTrialInfo = predDF.index.to_frame().reset_index(drop=True)
    foldsUID = idTrialInfo.apply(lambda x: (x['fold'], x['trialUID'], ), axis='columns')
    typeLookup = predTrialInfo.loc[:, ['fold', 'foldType', 'trialType', 'trialUID']].drop_duplicates().set_index(['fold', 'trialUID'])
    idTrialInfo.loc[:, 'foldType'] = foldsUID.map(typeLookup['foldType'])
    idTrialInfo.loc[:, 'trialType'] = foldsUID.map(typeLookup['trialType'])
    ammendedIndex = pd.MultiIndex.from_frame(idTrialInfo)
    inputDrivenDF.index = ammendedIndex
    oskDF.index = ammendedIndex
    #
    predDF = predDF.reorder_levels(predIndexNames)
    inputDrivenDF = inputDrivenDF.stack().to_frame(name='inputDriven')
    inputDrivenDF = inputDrivenDF.reorder_levels(predIndexNames)
    oskDF = oskDF.stack().to_frame(name='oneStepKalman')
    oskDF = oskDF.reorder_levels(predIndexNames)
    #
    stateSpaceScoresDict = {}
    dataDict = {}
    indicesIterBy = ['lhsMaskIdx', 'rhsMaskIdx', 'fold', 'target', 'foldType', 'trialType']

    showProgBar = True
    if showProgBar:
        progBarCtxt = tqdm(total=predDF.groupby(indicesIterBy).ngroups, mininterval=30., maxinterval=120.)
    else:
        progBarCtxt = contextlib.nullcontext()
    with progBarCtxt as pbar:   
        for name, group in predDF.groupby(indicesIterBy, sort=False):
            lhsMaskIdx, rhsMaskIdx, fold, target, foldType, trialType = name
            try:
                dataDF = pd.concat([
                    predDF.xs(name, level=indicesIterBy),
                    inputDrivenDF.xs(name, level=indicesIterBy),
                    oskDF.xs(name, level=indicesIterBy), ], axis='columns').dropna()
                #
                allCorr = pd.concat(
                    {
                        False: dataDF.corr(), 
                        True: dataDF.groupby('bin').mean().corr()},
                    names=['isTrialAveraged', 'term'])
                allCorr.columns.name = 'term'
                stateSpaceScoresDict[name] = allCorr
                #
                dataDict[name] = dataDF
                if showProgBar:
                    pbar.update(1)
            except Exception:
                traceback.print_exc()
                continue
    del inputDrivenDF, oskDF, predDF
    ######
    prf.print_memory_usage('done concatenating predictions from .h5 array')
    pdb.set_trace()
    ##
    ssScores = pd.concat(stateSpaceScoresDict, names=indicesIterBy)
    del stateSpaceScoresDict
    ssScores.to_hdf(transferFuncPath, 'stateSpaceScores')
    predDF = pd.concat(dataDict, names=indicesIterBy)
    del dataDict
    predDF.to_hdf(transferFuncPath, 'stateSpacePredictions')
    ##
    gc.collect()
    print('\n' + '#' * 50 + '\n{}\nCompleted.\n'.format(__file__) + '#' * 50 + '\n')