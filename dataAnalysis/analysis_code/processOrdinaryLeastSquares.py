"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose=verbose                        print diagnostics?
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
import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import colorsys
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklego.preprocessing import PatsyTransformer
from dataAnalysis.analysis_code.regression_parameters import *
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
import patsy
from itertools import product
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
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
    'figure.titlesize': 7
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_ra', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr_ta',
        'blockIdx': '2', 'exp': 'exp202101271100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    scratchPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings'
    scratchFolder = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202101201100-Rupert'
    figureFolder = '/gpfs/data/dborton/rdarie/Neural Recordings/processed/202101201100-Rupert/figures'
    
'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


def makeImpulseLike(df, categoricalCols=[], categoricalIndex=[]):
    for _, oneTrialDF in df.groupby('trialUID'):
        break
    impulseList = []
    fillIndexCols = [cN for cN in oneTrialDF.index.names if cN not in ['bin', 'trialUID', 'conditionUID']]
    fillCols = [cN for cN in oneTrialDF.columns if cN not in categoricalCols]
    if not len(categoricalCols):
        grouper = [('all', df),]
    else:
        grouper = df.groupby(categoricalCols)
    uid = 0
    for elecName, _ in grouper:
        thisImpulse = oneTrialDF.copy()
        thisTI = thisImpulse.index.to_frame().reset_index(drop=True)
        thisTI.loc[:, fillIndexCols] = 'NA'
        thisTI.loc[:, 'trialUID'] = uid
        uid += 1
        if len(categoricalCols):
            if len(categoricalCols) == 1:
                thisImpulse.loc[:, categoricalCols[0]] = elecName
                thisTI.loc[:, categoricalIndex[0]] = elecName
            else:
                for idx, cN in enumerate(categoricalCols):
                    thisImpulse.loc[:, cN] = elecName[idx]
                    thisTI.loc[:, categoricalIndex[idx]] = elecName[idx]
        #
        tBins = thisImpulse.index.get_level_values('bin')
        zeroBin = np.min(np.abs(tBins))
        thisImpulse.loc[:, fillCols] = 0.
        thisImpulse.loc[tBins == zeroBin, fillCols] = 1.
        thisImpulse.index = pd.MultiIndex.from_frame(thisTI)
        impulseList.append(thisImpulse)
    impulseDF = pd.concat(impulseList)
    return impulseDF


analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'regression')
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
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
#

for hIdx, histOpts in enumerate(addHistoryTerms):
    locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
thisEnv = patsy.EvalEnvironment.capture()

iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
# cv_kwargs = loadingMeta['cv_kwargs'].copy()
cvIterator = iteratorsBySegment[0]
lastFoldIdx = cvIterator.n_splits
#
selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
#

#
if estimatorMeta['pipelinePathRhs'] is not None:
    transformedSelectionNameRhs = '{}_{}'.format(
        selectionNameRhs, estimatorMeta['arguments']['transformerNameRhs'])
    transformedRhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(transformedSelectionNameRhs))
    pipelineScoresRhsDF = pd.read_hdf(estimatorMeta['pipelinePathRhs'], 'work')
    workingPipelinesRhs = pipelineScoresRhsDF['estimator']
else:
    workingPipelinesRhs = None
#
if estimatorMeta['pipelinePathLhs'] is not None:
    pipelineScoresLhsDF = pd.read_hdf(estimatorMeta['pipelinePathLhs'], 'work')
    workingPipelinesLhs = pipelineScoresLhsDF['estimator']
else:
    workingPipelinesLhs = None
#
estimatorsDF = pd.read_hdf(estimatorPath, 'cv_estimators')
scoresDF = pd.read_hdf(estimatorPath, 'cv_scores')
lhsDF = pd.read_hdf(estimatorMeta['lhsDatasetPath'], '/{}/data'.format(selectionNameLhs))
rhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(selectionNameRhs))
lhsMasks = pd.read_hdf(estimatorMeta['lhsDatasetPath'], '/{}/featureMasks'.format(selectionNameLhs))
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
#
trialInfoLhs = lhsDF.index.to_frame().reset_index(drop=True)
trialInfoRhs = rhsDF.index.to_frame().reset_index(drop=True)
checkSameMeta = stimulusConditionNames + ['bin', 'trialUID', 'conditionUID']
assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
trialInfo = trialInfoLhs

stimCondition = pd.Series(np.nan, index=trialInfo.index)
stimOrder = []
for name, group in trialInfo.groupby(['electrode', 'trialRateInHz']):
    stimCondition.loc[group.index] = '{} {}'.format(*name)
    stimOrder.append('{} {}'.format(*name))
trialInfo.loc[:, 'stimCondition'] = stimCondition
stimConditionLookup = (
    trialInfo
        .loc[:, ['electrode', 'trialRateInHz', 'stimCondition']]
        .drop_duplicates()
        .set_index(['electrode', 'trialRateInHz'])['stimCondition'])
kinCondition = pd.Series(np.nan, index=trialInfo.index)
kinOrder = []
for name, group in trialInfo.groupby(['pedalMovementCat', 'pedalDirection']):
    kinCondition.loc[group.index] = '{} {}'.format(*name)
    kinOrder.append('{} {}'.format(*name))
trialInfo.loc[:, 'kinCondition'] = kinCondition
kinConditionLookup = (
    trialInfo
        .loc[:, ['pedalMovementCat', 'pedalDirection', 'kinCondition']]
        .drop_duplicates()
        .set_index(['pedalMovementCat', 'pedalDirection'])['kinCondition'])
#
lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
#
lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormula', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(lambda x: ' + '.join(x), axis='columns')
#
rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
#
with pd.HDFStore(estimatorPath) as store:
    if 'coefficients' in store:
        coefDF = pd.read_hdf(store, 'coefficients')
        loadedCoefs = True
    else:
        loadedCoefs = False
    if 'predictions' in store:
        predDF = pd.read_hdf(store, 'predictions')
        loadedPreds = True
    else:
        loadedPreds = False
    #
    if 'designMatrix' in store:
        allDesignDF = pd.read_hdf(store, 'designMatrix')
        loadedDesigns = True
    else:
        loadedDesigns = False
    if 'histDesignMatrix' in store:
        histDesignDF = pd.read_hdf(store, 'histDesignMatrix')
        loadedHistDesigns = True
    else:
        loadedHistDesigns = False
    if 'sourcePalette' in store:
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        loadedPlotOpts = True
    else:
        loadedPlotOpts = False
    iRsExist = (
        ('impulseResponsePerTerm' in store) &
        ('impulseResponsePerFactor' in store)
        )
    if iRsExist:
        iRPerTerm = pd.read_hdf(store, 'impulseResponsePerTerm')
        iRPerFactor = pd.read_hdf(store, 'impulseResponsePerFactor')
        stimConditionLookupIR = pd.read_hdf(store, 'impulseResponseStimConditionLookup')
        kinConditionLookupIR = pd.read_hdf(store, 'impulseResponseKinConditionLookup')
        loadedIR = True
    else:
        loadedIR = False
# prep rhs dataframes
rhsPipelineAveragerDict = {}
rhGroupDict = {}
histDesignInfoDict = {}
histImpulseDict = {}
histSourceTermDict = {}
for rhsMaskIdx in range(rhsMasks.shape[0]):
    print('\n    On rhsRow {}\n'.format(rhsMaskIdx))
    rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
    rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
    rhGroup = rhsDF.loc[:, rhsMask].copy()
    # transform to PCs
    if workingPipelinesRhs is not None:
        transformPipelineRhs = workingPipelinesRhs.xs(rhsMaskParams['freqBandName'], level='freqBandName').iloc[0]
        rhsPipelineMinusAverager = Pipeline(transformPipelineRhs.steps[1:])
        rhsPipelineAveragerDict[rhsMaskIdx] = transformPipelineRhs.named_steps['averager']
        rhTransformedColumns = transformedRhsDF.columns[
            transformedRhsDF.columns.get_level_values('freqBandName') == rhsMaskParams['freqBandName']]
        rhGroup = pd.DataFrame(
            rhsPipelineMinusAverager.transform(rhGroup),
            index=rhsDF.index, columns=rhTransformedColumns)
    else:
        rhsPipelineAveragerDict[rhsMaskIdx] = Pipeline[('averager', tdr.DataFramePassThrough(),)]
    rhGroup.columns = rhGroup.columns.get_level_values('feature')
    #####################
    rhGroup = rhGroup.iloc[:, :3]
    ####################
    rhGroupDict[rhsMaskIdx] = rhGroup
    ####
    # if arguments['debugging']:
    #     if rhsMaskParams['freqBandName'] not in ['beta', 'gamma', 'higamma', 'all']:
    #         # if maskParams['lag'] not in [0]:
    #         continue
    ####
    # for ensTemplate, selfTemplate in lOfEnsembleTemplates:
    for ensTemplate in lOfHistTemplates:
        if ensTemplate is not None:
            histSourceTermDict.update({ensTemplate.format(cN): cN for cN in rhGroup.columns})
            ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
            ensFormula += ' - 1'
            print('Evaluating ensemble history {}'.format(ensFormula))
            ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
            exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
            ensDesignMatrix = ensPt.fit_transform(exampleRhGroup)
            ensDesignInfo = ensDesignMatrix.design_info
            print(ensDesignInfo.term_names)
            print('\n')
            # ensDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
            histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
            #
            impulseDF = makeImpulseLike(exampleRhGroup)
            impulseDM = ensPt.transform(impulseDF)
            ensImpulseDesignDF = (
                pd.DataFrame(
                    impulseDM,
                    index=impulseDF.index,
                    columns=ensDesignInfo.column_names))
            ensImpulseDesignDF.columns.name = 'factor'
            # ensImpulseDict[(rhsMaskIdx, ensTemplate)] = ensImpulseDesignDF
            histImpulseDict[(rhsMaskIdx, ensTemplate)] = ensImpulseDesignDF
#
designInfoDict0 = {}
impulseDict = {}
for lhsMaskIdx in range(lhsMasks.shape[0]):
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    lhGroup = lhsDF.loc[:, lhsMask]
    #
    lhGroup.columns = lhGroup.columns.get_level_values('feature')
    designFormula = lhsMask.name[lhsMasks.index.names.index('designFormula')]
    print('Evaluating exog history {}'.format(designFormula))
    #
    if designFormula not in designInfoDict0:
        pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
        exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
        designMatrix = pt.fit_transform(exampleLhGroup)
        designInfo = designMatrix.design_info
        designInfoDict0[designFormula] = designInfo
        #
        impulseDF = makeImpulseLike(
            exampleLhGroup, categoricalCols=['e'], categoricalIndex=['electrode'])
        impulseDM = pt.transform(impulseDF)
        impulseDesignDF = (
            pd.DataFrame(
                impulseDM,
                index=impulseDF.index,
                columns=designInfo.column_names))
        impulseDesignDF.columns.name = 'factor'
        impulseDict[designFormula] = impulseDesignDF

designInfoDF = pd.Series(designInfoDict0).to_frame(name='designInfo')
designInfoDF.index.name = 'design'
# designInfoDF = pd.DataFrame(
#     [value for key, value in designInfoDict0.items()],
#     columns=['designInfo'])
# designInfoDF.index = pd.MultiIndex.from_tuples([key for key, value in designInfoDict0.items()], names=['design'])

histDesignInfoDF = pd.DataFrame(
    [value for key, value in histDesignInfoDict.items()],
    columns=['designInfo'])
histDesignInfoDF.index = pd.MultiIndex.from_tuples([key for key, value in histDesignInfoDict.items()], names=['rhsMaskIdx', 'ensTemplate'])

reProcessPredsCoefs = not (loadedCoefs and loadedPreds and loadedDesigns)
if reProcessPredsCoefs:
    coefDict0 = {}
    predDict0 = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        # lhGroup = lhsDF.loc[:, lhsMask]
        #
        # lhGroup.columns = lhGroup.columns.get_level_values('feature')
        designFormula = lhsMask.name[lhsMasks.index.names.index('designFormula')]
        designInfo = designInfoDict0[designFormula]
        # designDF = allDesignDF.xs(lhsMaskIdx, level='lhsMaskIdx').xs(designFormula, level='design').loc[:, designInfo.column_names]
        designDF = allDesignDF.xs(designFormula, level='design').loc[:, designInfo.column_names]
        #
        # add ensemble to designDF?
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        for rhsMaskIdx in range(rhsMasks.shape[0]):
            print('\n    On rhsRow {}\n'.format(rhsMaskIdx))
            #  for lhsMaskIdx, rhsMaskIdx in product(list(range(lhsMasks.shape[0])), list(range(rhsMasks.shape[0]))):
            #
            rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
            ####
            rhGroup = rhGroupDict[rhsMaskIdx]
            ####
            if ensTemplate is not None:
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
            if selfTemplate is not None:
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
            ####
            for targetName in rhGroup.columns:
                # add targetDF to designDF?
                if ensTemplate is not None:
                    thisEnsDesign = (
                        histDesignDF
                            .xs(rhsMaskIdx, level='rhsMaskIdx', axis='columns')
                            .xs(ensTemplate, level='ensTemplate', axis='columns'))
                    ensHistList = [
                        thisEnsDesign.iloc[:, sl]
                        for key, sl in ensDesignInfo.term_name_slices.items()
                        if key != ensTemplate.format(targetName)]
                    ensTermNames = [
                        tN
                        for tN in ensDesignInfo.term_names
                        if tN != ensTemplate.format(targetName)]
                else:
                    ensHistList = []
                    ensTermNames = []
                #
                if selfTemplate is not None:
                    thisSelfDesign = (
                        histDesignDF
                            .xs(rhsMaskIdx, level='rhsMaskIdx', axis='columns')
                            .xs(selfTemplate, level='ensTemplate', axis='columns'))
                    selfHistList = [
                        thisSelfDesign.iloc[:, sl]
                        for key, sl in selfDesignInfo.term_name_slices.items()
                        if key == selfTemplate.format(targetName)]
                    selfTermNames = [
                        tN
                        for tN in selfDesignInfo.term_names
                        if tN == selfTemplate.format(targetName)]
                else:
                    selfHistList = []
                    selfTermNames = []
                #
                fullDesignList = [designDF] + ensHistList + selfHistList
                fullDesignDF = pd.concat(fullDesignList, axis='columns')
                for foldIdx in range(cvIterator.n_splits + 1):
                    targetDF = rhGroup.loc[:, [targetName]]
                    estimatorIdx = (lhsMaskIdx, rhsMaskIdx, targetName, foldIdx)
                    print('estimator: {}'.format(estimatorIdx))
                    print('in dataframe: {}'.format(estimatorIdx in estimatorsDF.index))
                    if not estimatorIdx in estimatorsDF.index:
                        continue
                    if foldIdx == cvIterator.n_splits:
                        # work and validation folds
                        trainIdx, testIdx = cvIterator.workIterator.split(rhGroup)[0]
                        trainStr, testStr = 'work', 'validation'
                        foldType = 'validation'
                    else:
                        trainIdx, testIdx = cvIterator.raw_folds[foldIdx]
                        trainStr, testStr = 'train', 'test'
                        foldType = 'train'
                    estimator = estimatorsDF.loc[estimatorIdx]
                    coefs = pd.Series(
                        estimator.regressor_.named_steps['regressor'].coef_, index=fullDesignDF.columns)
                    coefDict0[(lhsMaskIdx, designFormula, rhsMaskIdx, targetName, foldIdx)] = coefs
                    estPreprocessorLhs = Pipeline(estimator.regressor_.steps[:-1])
                    estPreprocessorRhs = estimator.transformer_
                    predictionPerComponent = pd.concat({
                        trainStr: estPreprocessorLhs.transform(fullDesignDF.iloc[trainIdx, :]) * coefs,
                        testStr: estPreprocessorLhs.transform(fullDesignDF.iloc[testIdx, :]) * coefs
                        }, names=['trialType'])
                    predictionSrs = predictionPerComponent.sum(axis='columns')
                    # sanity check
                    #############################
                    indicesThisFold = np.concatenate([trainIdx, testIdx])
                    predictionsNormalWay = np.concatenate([
                        estimator.predict(fullDesignDF.iloc[trainIdx, :]),
                        estimator.predict(fullDesignDF.iloc[testIdx, :])
                        ])
                    mismatch = predictionSrs - predictionsNormalWay.reshape(-1)
                    print('max mismatch is {}'.format(mismatch.abs().max()))
                    assert (mismatch.abs().max() < 1e-3)
                    termNames = designInfo.term_names + ensTermNames + selfTermNames
                    predictionPerSource = pd.DataFrame(
                        np.nan, index=predictionPerComponent.index,
                        columns=termNames)
                    for termName, termSlice in designInfo.term_name_slices.items():
                        factorNames = designInfo.column_names[termSlice]
                        predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                    if ensTemplate is not None:
                        for termName in ensTermNames:
                            factorNames = ensDesignInfo.column_names[ensDesignInfo.term_name_slices[termName]]
                            predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                    if selfTemplate is not None:
                        for termName in selfTermNames:
                            factorNames = selfDesignInfo.column_names[selfDesignInfo.term_name_slices[termName]]
                            predictionPerSource.loc[:, termName] = predictionPerComponent.loc[:, factorNames].sum(axis='columns')
                    #
                    predictionPerSource.loc[:, 'prediction'] = predictionSrs
                    predictionPerSource.loc[:, 'ground_truth'] = np.concatenate([
                        estPreprocessorRhs.transform(targetDF.iloc[trainIdx, :]),
                        estPreprocessorRhs.transform(targetDF.iloc[testIdx, :])
                        ])
                    predDict0[(lhsMaskIdx, designFormula, rhsMaskIdx, targetName, foldIdx, foldType)] = predictionPerSource
    #
    predDF = pd.concat(predDict0, names=['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'foldType'])
    predDF.columns.name = 'term'
    coefDF = pd.concat(coefDict0, names=['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'factor'])
    coefDF.to_hdf(estimatorPath, 'coefficients')
    predDF.to_hdf(estimatorPath, 'predictions')

if not loadedPlotOpts:
    termPalette = pd.concat({
        'exog': pd.Series(np.unique(np.concatenate([di.term_names for di in designInfoDF['designInfo']]))),
        'endog': pd.Series(np.unique(np.concatenate([di.term_names for di in histDesignInfoDF['designInfo']]))),
        'other': pd.Series(['prediction', 'ground_truth']),
        }, names=['type', 'index']).to_frame(name='term')
    sourceTermLookup = pd.concat({
        'exog': pd.Series(sourceTermDict).to_frame(name='source'),
        'endog': pd.Series(histSourceTermDict).to_frame(name='source'),
        'other': pd.Series(['prediction', 'ground_truth'], index=['prediction', 'ground_truth']).to_frame(name='source'),}, names=['type', 'term'])
    #
    primaryPalette = pd.DataFrame(sns.color_palette('deep', 10), columns=['r', 'g', 'b'])
    rgb = pd.DataFrame(
        primaryPalette.iloc[[0, 3, 8, 7], :].to_numpy(),
        columns=['r', 'g', 'b'], index=['v', 'a', 'r', 'ens'])
    hls = rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']), axis='columns')
    hls.loc['a*r', :] = hls.loc[['a', 'a', 'r'], :].mean()
    hls.loc['v*r', :] = hls.loc[['v', 'r', 'r'], :].mean()
    hls.loc['v*a', :] = hls.loc[['v', 'a', 'a'], :].mean()
    hls.loc['v*a*r', :] = hls.loc[['v', 'a', 'a', 'r'], :].mean()
    hls.loc['v*a*r', 's'] = hls.loc['v*a*r', 's'] / 2
    for rhsMaskIdx, rhGroup in rhGroupDict.items():
        lumVals = np.linspace(0.3, 0.7, rhGroup.shape[1])
        for cIdx, cN in enumerate(rhGroup.columns):
            hls.loc[cN, :] = hls.loc['ens', :]
            hls.loc[cN, 'l'] = lumVals[cIdx]
    hls.loc['prediction', :] = hls.loc['ens', :]
    hls.loc['prediction', 'l'] = 0.1
    hls.loc['ground_truth', :] = hls.loc['prediction', :]
    primarySourcePalette = hls.apply(lambda x: pd.Series(colorsys.hls_to_rgb(*x), index=['r', 'g', 'b']), axis='columns')
    sourcePalette = primarySourcePalette.apply(lambda x: tuple(x), axis='columns')
    #
    factorPaletteDict = {}
    endoFactors = []
    for designFormula, row in designInfoDF.iterrows():
        for tN, factorIdx in row['designInfo'].term_name_slices.items():
            thisSrs = pd.Series({fN: tN for fN in row['designInfo'].column_names[factorIdx]})
            thisSrs.name = 'term'
            thisSrs.index.name = 'factor'
            endoFactors.append(thisSrs)
    factorPaletteDict['endo'] = pd.concat(endoFactors).to_frame(name='term').reset_index().drop_duplicates(subset='factor')
    exoFactors = []
    for (rhsMaskIdx, ensTemplate), row in histDesignInfoDF.iterrows():
        for tN, factorIdx in row['designInfo'].term_name_slices.items():
            thisSrs = pd.Series({fN: tN for fN in row['designInfo'].column_names[factorIdx]})
            thisSrs.name = 'term'
            thisSrs.index.name = 'factor'
            exoFactors.append(thisSrs)
    factorPaletteDict['exo'] = pd.concat(exoFactors).to_frame(name='term').reset_index().drop_duplicates(subset='factor')
    factorPalette = pd.concat(factorPaletteDict, names=['type', 'index'])
    ############# workaround inconsistent use of whitespace with patsy
    sourceTermLookup.reset_index(inplace=True)
    sourceTermLookup.loc[:, 'term'] = sourceTermLookup['term'].apply(lambda x: x.replace(' ', ''))
    sourceTermLookup.set_index(['type', 'term'], inplace=True)
    ######
    termPalette.loc[:, 'termNoWS'] = termPalette['term'].apply(lambda x: x.replace(' ', ''))
    termPalette.loc[:, 'source'] = termPalette['termNoWS'].map(sourceTermLookup.reset_index(level='type')['source'])
    termPalette = termPalette.sort_values('source', kind='mergesort').sort_index(kind='mergesort')
    #
    factorPalette.loc[:, 'termNoWS'] = factorPalette['term'].apply(lambda x: x.replace(' ', ''))
    factorPalette.loc[:, 'source'] = factorPalette['termNoWS'].map(sourceTermLookup.reset_index(level='type')['source'])
    factorPalette = factorPalette.sort_values('source', kind='mergesort').sort_index(kind='mergesort')
    ############
    termPalette.loc[:, 'color'] = termPalette['source'].map(sourcePalette)
    factorPalette.loc[:, 'color'] = factorPalette['source'].map(sourcePalette)
    #
    trialTypeOrder = ['train', 'work', 'test', 'validation']
    trialTypePalette = pd.Series(
        sns.color_palette('Paired', 12)[::-1][:len(trialTypeOrder)],
        index=trialTypeOrder)
    #
    sourceTermLookup.to_hdf(estimatorPath, 'sourceTermLookup')
    sourcePalette.to_hdf(estimatorPath, 'sourcePalette')
    termPalette.to_hdf(estimatorPath, 'termPalette')
    factorPalette.to_hdf(estimatorPath, 'factorPalette')
    trialTypePalette.to_hdf(estimatorPath, 'trialTypePalette')

if not loadedIR:
    iRPerFactorDict0 = {}
    iRPerTermDict0 = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        designInfo = designInfoDict0[designFormula]
        theseEstimators = estimatorsDF.xs(lhsMaskIdx, level='lhsMaskIdx')
        designDF = impulseDict[designFormula]
        termNames = designInfo.term_names
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        #
        iRPerFactorDict1 = {}
        iRPerTermDict1 = {}
        for (rhsMaskIdx, targetName, fold), estimatorSrs in theseEstimators.groupby(['rhsMaskIdx', 'target', 'fold']):
            estimator = estimatorSrs.iloc[0]
            coefs = coefDF.loc[idxSl[lhsMaskIdx, designFormula, rhsMaskIdx, targetName, fold, :]]
            coefs.index = coefs.index.get_level_values('factor')
            columnsInDesign = [cN for cN in coefs.index if cN in designDF.columns]
            columnsNotInDesign = [cN for cN in coefs.index if cN not in designDF.columns]
            thisIR = designDF * coefs.loc[columnsInDesign]
            outputIR = thisIR.copy()
            #
            if ensTemplate is not None:
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
                ensTermNames = [
                    key
                    for key in ensDesignInfo.term_names
                    if key != ensTemplate.format(targetName)]
                ensFactorNames = np.concatenate([
                    np.atleast_1d(ensDesignInfo.column_names[sl])
                    for key, sl in ensDesignInfo.term_name_slices.items()
                    if key != ensTemplate.format(targetName)])
                thisEnsDesignDF = histImpulseDict[(rhsMaskIdx, ensTemplate)].loc[:, ensFactorNames]
                columnsInDesign = [cN for cN in coefs.index if cN in ensFactorNames]
                ensIR = thisEnsDesignDF * coefs.loc[columnsInDesign]
                for cN in ensFactorNames:
                    outputIR.loc[:, cN] = np.nan
            else:
                ensTermNames = []
            #
            if selfTemplate is not None:
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
                selfTermNames = [
                    key
                    for key in selfDesignInfo.term_names
                    if key == selfTemplate.format(targetName)]
                selfFactorNames = np.concatenate([
                    np.atleast_1d(selfDesignInfo.column_names[sl])
                    for key, sl in selfDesignInfo.term_name_slices.items()
                    if key == selfTemplate.format(targetName)])
                thisSelfDesignDF = histImpulseDict[(rhsMaskIdx, selfTemplate)].loc[:, selfFactorNames]
                columnsInDesign = [cN for cN in coefs.index if cN in selfFactorNames]
                selfIR = thisSelfDesignDF * coefs.loc[columnsInDesign]
                for cN in selfFactorNames:
                    outputIR.loc[:, cN] = np.nan
            else:
                selfTermNames = []
            ## !! reorder columns consistently
            ipsList = []
            for trialUID, group in thisIR.groupby('trialUID'):
                iRPerSource = pd.DataFrame(
                    np.nan, index=group.index,
                    columns=termNames + ensTermNames + selfTermNames)
                for termName, termSlice in designInfo.term_name_slices.items():
                    factorNames = designInfo.column_names[termSlice]
                    iRPerSource.loc[:, termName] = group.loc[:, factorNames].sum(axis='columns').to_numpy()
                if ensTemplate is not None:
                    outputIR.loc[group.index, ensFactorNames] = ensIR.to_numpy()
                    for termName in ensTermNames:
                        factorNames = ensFactorNames[ensDesignInfo.term_name_slices[termName]]
                        iRPerSource.loc[:, termName] = ensIR.loc[:, factorNames].sum(axis='columns').to_numpy()
                if selfTemplate is not None:
                    outputIR.loc[group.index, selfFactorNames] = selfIR.to_numpy()
                    for termName in selfTermNames:
                        factorNames = selfDesignInfo.column_names[selfDesignInfo.term_name_slices[termName]]
                        iRPerSource.loc[:, termName] = selfIR.loc[:, factorNames].sum(axis='columns').to_numpy()
                ipsList.append(iRPerSource)
            iRPerFactorDict1[(rhsMaskIdx, targetName, fold)] = outputIR
            iRPerTermDict1[(rhsMaskIdx, targetName, fold)] = pd.concat(ipsList)
        iRPerFactorDict0[(lhsMaskIdx, designFormula)] = pd.concat(iRPerFactorDict1, names=['rhsMaskIdx', 'target', 'fold'])
        iRPerTermDict0[(lhsMaskIdx, designFormula)] = pd.concat(iRPerTermDict1, names=['rhsMaskIdx', 'target', 'fold'])
    #
    #
    iRPerFactor = pd.concat(iRPerFactorDict0, names=['lhsMaskIdx', 'design'])
    iRPerFactor.columns.name = 'factor'
    iRPerTerm = pd.concat(iRPerTermDict0, names=['lhsMaskIdx', 'design'])
    iRPerTerm.columns.name = 'term'
    #
    trialInfoIR = iRPerTerm.index.to_frame().reset_index(drop=True)
    stimConditionIR = pd.Series(np.nan, index=trialInfoIR.index)
    stimOrderIR = []
    for name, group in trialInfoIR.groupby(['electrode', 'trialRateInHz']):
        stimConditionIR.loc[group.index] = '{} {}'.format(*name)
        stimOrderIR.append('{} {}'.format(*name))
    trialInfoIR.loc[:, 'stimCondition'] = stimConditionIR
    stimConditionLookupIR = (
        trialInfoIR
            .loc[:, ['electrode', 'trialRateInHz', 'stimCondition']]
            .drop_duplicates()
            .set_index(['electrode', 'trialRateInHz'])['stimCondition'])
    kinConditionIR = pd.Series(np.nan, index=trialInfoIR.index)
    kinOrderIR = []
    for name, group in trialInfoIR.groupby(['pedalMovementCat', 'pedalDirection']):
        kinConditionIR.loc[group.index] = '{} {}'.format(*name)
        kinOrderIR.append('{} {}'.format(*name))
    trialInfoIR.loc[:, 'kinCondition'] = kinConditionIR
    kinConditionLookupIR = (
        trialInfoIR
            .loc[:, ['pedalMovementCat', 'pedalDirection', 'kinCondition']]
            .drop_duplicates()
            .set_index(['pedalMovementCat', 'pedalDirection'])['kinCondition'])
    iRPerTerm.index = pd.MultiIndex.from_frame(trialInfoIR)
    #
    iRPerTerm.to_hdf(estimatorPath, 'impulseResponsePerTerm')
    iRPerFactor.to_hdf(estimatorPath, 'impulseResponsePerFactor')
    stimConditionLookupIR.to_hdf(estimatorPath, 'impulseResponseStimConditionLookup')
    kinConditionLookupIR.to_hdf(estimatorPath, 'impulseResponseKinConditionLookup')
#
groupPagesBy = ['rhsMaskIdx', 'lhsMaskIdx', 'target']
groupSubPagesBy = ['trialType', 'foldType', 'electrode']
scoresStack = pd.concat({
        'test': scoresDF['test_score'],
        'train': scoresDF['train_score']},
    names=['trialType']
    ).to_frame(name='score').reset_index()
#
lastFoldMask = (scoresStack['fold'] == cvIterator.n_splits)
trainMask = (scoresStack['trialType'] == 'train')
testMask = (scoresStack['trialType'] == 'test')
#
scoresStack.loc[:, 'foldType'] = ''
scoresStack.loc[(trainMask & lastFoldMask), 'foldType'] = 'work'
scoresStack.loc[(trainMask & (~lastFoldMask)), 'foldType'] = 'train'
scoresStack.loc[(testMask & lastFoldMask), 'foldType'] = 'validation'
scoresStack.loc[(testMask & (~lastFoldMask)), 'foldType'] = 'test'
scoresStack.loc[:, 'dummyX'] = 0
scoresStack.loc[:, 'design'] = scoresStack['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'designFormula'])
scoresStack.loc[:, 'designAsLabel'] = scoresStack['design'].apply(lambda x: x.replace(' + ', ' +\n'))
scoresStack.loc[:, 'fullDesign'] = scoresStack['lhsMaskIdx'].apply(lambda x: lhsMasksInfo.loc[x, 'fullFormulaDescr'])
scoresStack.loc[:, 'fullDesignAsLabel'] = scoresStack['fullDesign'].apply(lambda x: x.replace(' + ', ' +\n'))
#
llDict1 = {}
for scoreName, targetScores in scoresStack.groupby(['lhsMaskIdx', 'target', 'fold']):
    lhsMaskIdx, targetName, fold = scoreName
    designFormula = lhsMasksInfo.loc[lhsMaskIdx, 'designFormula']
    estimator = estimatorsDF.loc[idxSl[lhsMaskIdx, :, targetName, fold]].iloc[0]
    regressor = estimator.regressor_.steps[-1][1]
    thesePred = predDF.xs(targetName, level='target').xs(lhsMaskIdx, level='lhsMaskIdx').xs(fold, level='fold')
    llDict2 = {}
    for name, predGroup in thesePred.groupby(['electrode', 'trialType']):
        llDict3 = dict()
        llDict3['llSat'] = regressor.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
        nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
        llDict3['llNull'] = regressor.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
        llDict3['llFull'] = regressor.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
        llDict2[name] = pd.Series(llDict3)
    for trialType, predGroup in thesePred.groupby('trialType'):
        llDict3 = dict()
        llDict3['llSat'] = regressor.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
        nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
        llDict3['llNull'] = regressor.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
        llDict3['llFull'] = regressor.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
        llDict2[('all', trialType)] = pd.Series(llDict3)
    llDict1[(lhsMaskIdx, designFormula, targetName, fold)] = pd.concat(llDict2, names=['electrode', 'trialType', 'llType'])
llDF = pd.concat(llDict1, names=['lhsMaskIdx', 'design', 'target', 'fold', 'electrode', 'trialType', 'llType']).to_frame(name='ll')
R2Per = llDF['ll'].groupby(['lhsMaskIdx', 'design', 'target', 'electrode', 'fold', 'trialType']).apply(getR2).to_frame(name='score')

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'partial_scores'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    for modelToTest in modelsToTest:
        testDesign = modelToTest['testDesign']
        refDesign = modelToTest['refDesign']
        if 'captionStr' in modelToTest:
            titleText = modelToTest['captionStr']
        else:
            titleText = 'partial R2 scores for {} compared to {}'.format(testDesign, refDesign)
        pR2 = partialR2(llDF['ll'], refDesign=refDesign, testDesign=testDesign)
        #
        plotScores = pR2.reset_index().rename(columns={'ll': 'score'})
        #
        plotScores.loc[:, 'xDummy'] = 0
        #
        thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['trialType'])]
        g = sns.catplot(
            data=plotScores, kind='box',
            y='score', x='target',
            col='electrode', hue='trialType',
            hue_order=thisPalette.index.to_list(),
            palette=thisPalette.to_dict(),
            height=height, aspect=aspect,
            sharey=False
            )
        g.set_xticklabels(rotation=30, ha='right')
        g.set_titles(template="data subset: {col_name}")
        g.suptitle(titleText)
        # g.axes.flat[0].set_ylim(allScoreQuantiles)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(
            bbox_inches='tight',
        )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'r2'))
with PdfPages(pdfPath) as pdf:
    for rhsMaskIdx, plotScores in scoresStack.groupby(['rhsMaskIdx']):
        rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
        thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['foldType'])]
        g = sns.catplot(
            data=plotScores, hue='foldType',
            x='fullDesignAsLabel', y='score', col='target',
            hue_order=thisPalette.index.to_list(),
            palette=thisPalette.to_dict(),
            kind='box')
        g.suptitle('R2 (freqBand: {})'.format(rhsMasksInfo.iloc[rhsMaskIdx, :]['freqBandName']))
        g.set_xticklabels(rotation=-30, ha='left')
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'impulse_responses'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    for (lhsMaskIdx, designFormula, targetName), thisIRPerTerm in iRPerTerm.groupby(['lhsMaskIdx', 'design', 'target']):
        designInfo = designInfoDF.loc[designFormula, 'designInfo']
        plotDF = thisIRPerTerm.stack().to_frame(name='signal').reset_index()
        kinOrder = kinConditionLookupIR.loc[kinConditionLookupIR.isin(plotDF['kinCondition'])].to_list()
        stimOrder = stimConditionLookupIR.loc[stimConditionLookupIR.isin(plotDF['stimCondition'])].to_list()
        thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
        g = sns.relplot(
            row='kinCondition', row_order=kinOrder,
            col='stimCondition', col_order=stimOrder,
            x='bin', y='signal', hue='term',
            hue_order=thisTermPalette['term'].to_list(),
            palette=thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color'].to_dict(),
            kind='line', errorbar='se', data=plotDF,
            )
        g.set_axis_labels("Time (sec)", 'contribution to {}'.format(targetName))
        g.suptitle('Impulse responses (per term) for model {}'.format(designFormula))
        asp.reformatFacetGridLegend(
            g, titleOverrides={},
            contentOverrides=termPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict(),
            styleOpts=styleOpts)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()

height, width = 2, 2
trialTypeToPlot = 'test'
aspect = width / height
commonOpts = dict(
    )
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'reconstructions'))
with PdfPages(pdfPath) as pdf:
    for name0, predGroup0 in predDF.groupby(groupPagesBy):
        nmLk0 = {key: value for key, value in zip(groupPagesBy, name0)} # name lookup
        nmLk0['design'] = lhsMasksInfo.loc[nmLk0['lhsMaskIdx'], 'designFormula']
        scoreMasks = [
            scoresStack[cN] == nmLk0[cN]
            for cN in groupPagesBy]
        plotScores = scoresStack.loc[np.logical_and.reduce(scoreMasks), :]
        thisPalette = trialTypePalette.loc[trialTypePalette.index.isin(plotScores['foldType'])]
        g = sns.catplot(
            data=plotScores, hue='foldType',
            x='fullDesignAsLabel', y='score',
            hue_order=thisPalette.index.to_list(),
            palette=thisPalette.to_dict(),
            kind='box')
        g.set_xticklabels(rotation=-30, ha='left')
        g.suptitle('R^2 for target {target}'.format(**nmLk0))
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        for name1, predGroup1 in predGroup0.groupby(groupSubPagesBy):
            nmLk1 = {key: value for key, value in zip(groupSubPagesBy, name1)} # name lookup
            nmLk0.update(nmLk1)
            if nmLk0['trialType'] != trialTypeToPlot:
                continue
            plotDF = predGroup1.stack().to_frame(name='signal').reset_index()
            plotDF.loc[:, 'predType'] = 'component'
            plotDF.loc[plotDF['term'] == 'ground_truth', 'predType'] = 'ground_truth'
            plotDF.loc[plotDF['term'] == 'prediction', 'predType'] = 'prediction'
            plotDF.loc[:, 'kinCondition'] = plotDF.loc[:, ['pedalMovementCat', 'pedalDirection']].apply(lambda x: tuple(x), axis='columns').map(kinConditionLookup)
            plotDF.loc[:, 'stimCondition'] = plotDF.loc[:, ['electrode', 'trialRateInHz']].apply(lambda x: tuple(x), axis='columns').map(stimConditionLookup)
            kinOrder = kinConditionLookup.loc[kinConditionLookup.isin(plotDF['kinCondition'])].to_list()
            stimOrder = stimConditionLookup.loc[stimConditionLookup.isin(plotDF['stimCondition'])].to_list()
            thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
            theseColors = thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color'].to_dict()
            g = sns.relplot(
                data=plotDF,
                col='trialAmplitude', row='kinCondition',
                row_order=kinOrder,
                x='bin', y='signal', hue='term',
                height=height, aspect=aspect, palette=theseColors,
                kind='line', errorbar='sd',
                size='predType', sizes={
                    'component': .5,
                    'prediction': 1.,
                    'ground_truth': 1.,
                    },
                style='predType', dashes={
                    'component': (3, 1),
                    'prediction': (2, 1),
                    'ground_truth': (8, 0),
                    },
                style_order=['component', 'prediction', 'ground_truth'],
                facet_kws=dict(margin_titles=True),
                )
            g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
            titleText = 'model {design}\n{target}, electrode {electrode} ({trialType})'.format(
                **nmLk0)
            print('Saving plot of {}...'.format(titleText))
            g.suptitle(titleText)
            asp.reformatFacetGridLegend(
                g, titleOverrides={},
                contentOverrides=termPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict(),
                styleOpts=styleOpts)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(
                bbox_inches='tight',
                # bbox_extra_artists=[figTitle, g.legend]
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

'''
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'regressors'))
with PdfPages(pdfPath) as pdf:
    for (lhsMaskIdx, designFormula), row in designInfoDF.iterrows():
        #
        designInfo = row['designInfo']
        designDF = allDesignDF.xs(lhsMaskIdx, level='lhsMaskIdx').xs(designFormula, level='design').loc[:, designInfo.column_names]
        factorNames = designDF.columns.to_frame().reset_index(drop=True)
        factorNames.loc[:, 'term'] = np.nan
        termDF = pd.DataFrame(
            np.nan, index=designDF.index,
            columns=designInfo.term_names)
        termDF.columns.name = 'term'
        for termName, termSlice in designInfo.term_name_slices.items():
            termDF.loc[:, termName] = designDF.iloc[:, termSlice].sum(axis='columns')
            factorNames.iloc[termSlice, 1] = termName
        #
        trialInfo = termDF.index.to_frame().reset_index(drop=True)
        stimCondition = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(['electrode', 'trialRateInHz']):
            stimCondition.loc[group.index] = '{} {}'.format(*name)
        trialInfo.loc[:, 'stimCondition'] = stimCondition
        kinCondition = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(['pedalMovementCat', 'pedalDirection']):
            kinCondition.loc[group.index] = '{} {}'.format(*name)
        trialInfo.loc[:, 'kinCondition'] = kinCondition
        #
        #
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMask.index.names, lhsMask.name)}
        plotDF = lhsDF.loc[:, lhsMask].copy()
        plotDF.columns = plotDF.columns.get_level_values('feature')
        plotDF.drop(columns=['e'], inplace=True)
        plotDF.index = pd.MultiIndex.from_frame(trialInfo)
        plotDF = plotDF.stack().to_frame(name='signal')
        plotDF.reset_index(inplace=True)
        g = sns.relplot(
            row='kinCondition', col='stimCondition',
            x='bin', y='signal', hue='feature',
            kind='line', errorbar='se', data=plotDF
            )
        g.fig.suptitle('Features for model {}'.format(designFormula))
        leg = g._legend
        if leg is not None:
            t = leg.get_title()
            tContent = t.get_text()
            # if tContent in titleOverrides:
            #     t.set_text(titleOverrides[tContent])
            # for t in leg.texts:
            #     tContent = t.get_text()
            #     if tContent in titleOverrides:
            #         t.set_text(titleOverrides[tContent])
            #     elif tContent in emgNL.index:
            #         t.set_text('{}'.format(emgNL[tContent]))
            for l in leg.get_lines():
                # l.set_lw(2 * l.get_lw())
                l.set_lw(styleOpts['legend.lw'])
        g.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        #
        designDF.index = pd.MultiIndex.from_frame(trialInfo)
        designDF = designDF.stack().to_frame(name='signal')
        designDF.reset_index(inplace=True)
        factorPalette = factorNames.set_index('factor')['term'].map(termPalette)
        g = sns.relplot(
            row='kinCondition', col='stimCondition',
            x='bin', y='signal', hue='factor',
            kind='line', errorbar='se', data=designDF,
            palette=factorPalette.to_dict()
            )
        g.fig.suptitle('Terms for model {}'.format(designFormula))
        leg = g._legend
        if leg is not None:
            t = leg.get_title()
            tContent = t.get_text()
            # if tContent in titleOverrides:
            #     t.set_text(titleOverrides[tContent])
            # for t in leg.texts:
            #     tContent = t.get_text()
            #     if tContent in titleOverrides:
            #         t.set_text(titleOverrides[tContent])
            #     elif tContent in emgNL.index:
            #         t.set_text('{}'.format(emgNL[tContent]))
            for l in leg.get_lines():
                # l.set_lw(2 * l.get_lw())
                l.set_lw(styleOpts['legend.lw'])
        g.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        #
        termDF.index = pd.MultiIndex.from_frame(trialInfo)
        termDF = termDF.stack().to_frame(name='signal')
        termDF.reset_index(inplace=True)
        g = sns.relplot(
            row='kinCondition', col='stimCondition',
            x='bin', y='signal', hue='term',
            kind='line', errorbar='se', data=termDF,
            palette=termPalette.to_dict()
            )
        g.fig.suptitle('Terms for model {}'.format(designFormula))
        leg = g._legend
        if leg is not None:
            t = leg.get_title()
            tContent = t.get_text()
            # if tContent in titleOverrides:
            #     t.set_text(titleOverrides[tContent])
            # for t in leg.texts:
            #     tContent = t.get_text()
            #     if tContent in titleOverrides:
            #         t.set_text(titleOverrides[tContent])
            #     elif tContent in emgNL.index:
            #         t.set_text('{}'.format(emgNL[tContent]))
            for l in leg.get_lines():
                # l.set_lw(2 * l.get_lw())
                l.set_lw(styleOpts['legend.lw'])
        g.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
'''

'''
from scipy import signal
import control as ctrl
histOpts = hto0
### sanity check impulse responses
rawProbeTermName = 'pca_ta_all001'
probeTermName = 'rcb({}, **hto0)'.format(rawProbeTermName)
for (lhsMaskIdx, designFormula, rhsMaskIdx, targetName, fold), thisIRPerTerm in iRPerTerm.groupby(['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold']):
    if fold == cvIterator.n_splits:
        continue
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    lhGroup = lhsDF.loc[:, lhsMask]
    #
    lhGroup.columns = lhGroup.columns.get_level_values('feature')
    #
    designInfo = designInfoDict0[(lhsMaskIdx, designFormula)]
    termNames = designInfo.term_names
    ensTemplate = lhsMaskParams['ensembleTemplate']
    selfTemplate = lhsMaskParams['selfTemplate']
    if ensTemplate is not None:
        ensDesignInfo = ensDesignInfoDict[(rhsMaskIdx, ensTemplate)]
        ensDesignDF = ensImpulseDict[(rhsMaskIdx, ensTemplate)]
        ensTermNames = [
            key
            for key, sl in ensDesignInfo.term_name_slices.items()
            if key != ensTemplate.format(targetName)]
    else:
        ensTermNames = []
    if selfTemplate is not None:
        selfDesignInfo = selfDesignInfoDict[(rhsMaskIdx, selfTemplate)]
        selfDesignDF = selfImpulseDict[(rhsMaskIdx, selfTemplate)]
        selfTermNames = [
            key
            for key, sl in selfDesignInfo.term_name_slices.items()
            if key == selfTemplate.format(targetName)]
    else:
        selfTermNames = []
    estimatorIdx = (lhsMaskIdx, rhsMaskIdx, targetName, fold)
    if not estimatorIdx in estimatorsDF.index:
        continue
    estimator = estimatorsDF.loc[estimatorIdx]
    if rawProbeTermName in lhsDF.columns.get_level_values('feature'):
        estPreprocessorLhs = Pipeline(estimator.regressor_.steps[:-1])
        preprocdLhs = estPreprocessorLhs.transform(lhsDF.iloc[cvIterator.folds[fold][0], :])
        preprocdLhs.columns = preprocdLhs.columns.get_level_values('feature')
        dFToConvolve = preprocdLhs
    else:
        estPreprocessorRhs = estimator.transformer_
        preprocdRhs = estPreprocessorRhs.transform(rhGroupDict[rhsMaskIdx].iloc[cvIterator.folds[fold][0], :])
        preprocdRhs.columns = preprocdRhs.columns.get_level_values('feature')
        dFToConvolve = preprocdRhs
    #
    thisIRPerTerm = thisIRPerTerm.dropna(axis='columns')
    #
    fig, ax = plt.subplots(2, 1)
    for elecName, iRGroup in thisIRPerTerm.groupby('electrode'):
        for moveCat, dataGroup0 in dFToConvolve.xs(elecName, level='electrode', drop_level=False).groupby('pedalMovementCat'):
            for trialUID, dataGroup in dataGroup0.groupby('trialUID'):
                break
            try:
                tBins = dataGroup.index.get_level_values('bin')
                ax[0].plot(tBins, dataGroup[rawProbeTermName], label='input {}'.format(rawProbeTermName))
                ax[0].plot(tBins, iRGroup[probeTermName], '*-', label='kernel corresponding to {}'.format(rawProbeTermName))
                ax[0].legend()
                ax[0].set_title('target {}, elec {}, {}'.format(targetName, elecName, moveCat))
                predDFIdx = idxSl[lhsMaskIdx, designFormula, rhsMaskIdx, targetName, fold, 'train', 'train', elecName, :, :, :, :, :, :, trialUID]
                ax[1].plot(tBins, predDF.loc[predDFIdx, probeTermName], 'c', label='predicted', lw=5)
                #
                kernelTMask = (tBins >= 0) & (tBins <= histOpts['historyLen'])
                b = iRGroup.loc[kernelTMask, probeTermName].to_numpy()
                a = np.zeros(b.shape)
                a[0] = 1
                #
                filtered = signal.lfilter(b, a, dataGroup[rawProbeTermName])
                ax[1].plot(tBins, filtered, 'b--', label='filtered', lw=2)
                ax[1].legend()
                plt.show()
                # sys = signal.dlti(b, a, dt=binInterval)
                # trFun = ctrl.tf(b, a, dt=binInterval)
                # ctrl.tf2ss(trFun)
            except:
                traceback.print_exc()
                continue
            fig, ax = plt.subplots(2, 1)
        break
    break
from sympy.matrices import Matrix, eye, zeros
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.abc import z
from sympy import simplify, cancel, ratsimp, degree
iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold', 'electrode']
for name, iRGroup0 in iRPerTerm.groupby(iRGroupNames):
    ##
    iRGroup0.dropna(inplace=True, axis='columns')
    ##
    lhsMaskIdx, designFormula, rhsMaskIdx, fold, electrode = name
    if lhsMaskIdx != 1:
        continue
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    #
    designInfo = designInfoDict0[(lhsMaskIdx, designFormula)]
    termNames = designInfo.term_names
    signalNames = iRGroup0.index.get_level_values('target').unique().to_list()
    ensTemplate = lhsMaskParams['ensembleTemplate']
    if ensTemplate is not None:
        ensDesignInfo = ensDesignInfoDict[(rhsMaskIdx, ensTemplate)]
    if selfTemplate is not None:
        selfDesignInfo = selfDesignInfoDict[(rhsMaskIdx, selfTemplate)]
    selfTemplate = lhsMaskParams['selfTemplate']
    iRGroup0.rename(columns={ensTemplate.format(key): key for key in signalNames}, inplace=True)
    #
    #
    Zs = pd.DataFrame(np.nan, index=signalNames, columns=signalNames + termNames)
    MN = zeros(*Zs.shape)
    for targetName, iRGroup in iRGroup0.groupby('target'):
        if ensTemplate is not None:
            ensTermNames = [
                key
                for key, sl in ensDesignInfo.term_name_slices.items()
                if key != ensTemplate.format(targetName)]
        else:
            ensTermNames = []
        if selfTemplate is not None:
            selfTermNames = [
                key
                for key, sl in selfDesignInfo.term_name_slices.items()
                if key == selfTemplate.format(targetName)]
        else:
            selfTermNames = []
        tBins = iRGroup.index.get_level_values('bin')
        kernelTMask = (tBins >= 0) & (tBins <= histOpts['historyLen'])
        for termName in iRGroup.columns:
            kernel = iRGroup.loc[kernelTMask, termName]
            b = kernel.to_numpy()
            a = np.zeros(b.shape)
            a[0] = 1.
            rowIdx = Zs.index.to_list().index(targetName)
            colIdx = Zs.columns.to_list().index(termName)
            for order, coefficient in enumerate(b):
                MN[rowIdx, colIdx] += coefficient * z ** (-(order + 1))
            # trFun = ctrl.tf(b, a, dt=binInterval)
            # Zs.loc[targetName, termName] = trFun / ctrl.tf('z', dt=binInterval)
            # sns.heatmap(sS.A, annot=True, annot_kws={'fontsize': 2})
    M = MN[:, :len(signalNames)]
    N = MN[:, len(signalNames):]
    G = (z * eye(len(signalNames)) - M).inv() * N
    tf_num = []
    tf_den = []
    for rowIdx, targetName in enumerate(Zs.index):
        tf_num.append([])
        tf_den.append([])
        for colIdx, termName in enumerate(signalNames):
            num, den = ratsimp(G[rowIdx, colIdx]).as_numer_denom()
            print('{}, {}'.format(num, den))
            if num == 0:
                tf_num[rowIdx].append(np.asarray([0]))
            else:
                tf_num[rowIdx].append(np.asarray(num.as_poly().all_coeffs()))
            if den == 1:
                tf_den[rowIdx].append(np.asarray([1]))
            else:
                tf_den[rowIdx].append(np.asarray(den.as_poly().all_coeffs()))
    trFun = ctrl.tf(tf_num, tf_den, dt=binInterval)
    break
'''
####