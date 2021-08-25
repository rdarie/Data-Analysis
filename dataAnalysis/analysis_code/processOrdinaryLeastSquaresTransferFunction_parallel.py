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
    --forceReprocess                         print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
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
from datetime import datetime
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.plotting.aligned_signal_plots as asp
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
from scipy.stats import zscore, mode
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from dataAnalysis.analysis_code.regression_parameters import *
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import patsy
from sklego.preprocessing import PatsyTransformer
import dill as pickle
pickle.settings['recurse'] = True
import gc
from itertools import product
from docopt import docopt
import colorsys
idxSl = pd.IndexSlice
#
print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'showFigures': False, 'analysisName': 'hiRes', 'exp': 'exp202101281100', 'datasetName': 'Block_XL_df_ra',
        'debugging': False, 'blockIdx': '2', 'alignFolderName': 'motion', 'estimatorName': 'enr_fa_ta',
        'plotting': True, 'verbose': '1', 'processAll': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
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
'''if estimatorMeta['pipelinePathRhs'] is not None:
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
    workingPipelinesLhs = None'''
#
lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets')
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
#
lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
lhsMasksInfo.loc[:, 'designFormulaShortHand'] = lhsMasksInfo['designFormula'].apply(lambda x: formulasShortHand[x])
lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormulaShortHand', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(lambda x: ' + '.join(x), axis='columns')
#
rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)
#
iRPerTerm = pd.read_hdf(estimatorPath, 'impulseResponsePerTerm')
iRPerFactor = pd.read_hdf(estimatorPath, 'impulseResponsePerFactor')
sourcePalette = pd.read_hdf(estimatorPath, 'sourcePalette')
termPalette = pd.read_hdf(estimatorPath, 'termPalette')
factorPalette = pd.read_hdf(estimatorPath, 'factorPalette')
trialTypePalette = pd.read_hdf(estimatorPath, 'trialTypePalette')
####
# prep rhs dataframes
histDesignInfoDict = {}
histImpulseDict = {}
histSourceTermDict = {}
for rhsMaskIdx in range(rhsMasks.shape[0]):
    prf.print_memory_usage('\n Prepping RHS dataframes (rhsRow {})'.format(rhsMaskIdx))
    rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
    rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
    rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
    for ensTemplate in lOfHistTemplates:
        if ensTemplate != 'NULL':
            histSourceTermDict.update({ensTemplate.format(cN): cN for cN in rhGroup.columns})
            ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
            ensFormula += ' - 1'
            print('Generating endog design info for {}'.format(ensFormula))
            ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
            exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
            ensPt.fit(exampleRhGroup)
            ensDesignMatrix = ensPt.transform(exampleRhGroup)
            ensDesignInfo = ensDesignMatrix.design_info
            histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
            #
            impulseDF = tdr.makeImpulseLike(exampleRhGroup)
            impulseDM = ensPt.transform(impulseDF)
            ensImpulseDesignDF = (
                pd.DataFrame(
                    impulseDM,
                    index=impulseDF.index,
                    columns=ensDesignInfo.column_names))
            ensImpulseDesignDF.columns.name = 'factor'
            histImpulseDict[(rhsMaskIdx, ensTemplate)] = ensImpulseDesignDF
#
designInfoDict = {}
impulseDict = {}
for lhsMaskIdx in range(lhsMasks.shape[0]):
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    designFormula = lhsMaskParams['designFormula']
    #
    if designFormula != 'NULL':
        if (designFormula not in designInfoDict) and (designIsLinear[designFormula]):
            print('Generating exog design info for {}'.format(designFormula))
            formulaIdx = lOfDesignFormulas.index(designFormula)
            lhGroup = lhsDF.loc[:, lhsMask]
            pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
            exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
            designMatrix = pt.fit_transform(exampleLhGroup)
            designInfo = designMatrix.design_info
            designInfoDict[designFormula] = designInfo
            #
            impulseDF = tdr.makeImpulseLike(
                exampleLhGroup, categoricalCols=['e'], categoricalIndex=['electrode'])
            impulseDM = pt.transform(impulseDF)
            impulseDesignDF = (
                pd.DataFrame(
                    impulseDM,
                    index=impulseDF.index,
                    columns=designInfo.column_names))
            impulseDesignDF.columns.name = 'factor'
            impulseDict[designFormula] = impulseDesignDF
designInfoDF = pd.Series(designInfoDict).to_frame(name='designInfo')
designInfoDF.index.name = 'design'
histDesignInfoDF = pd.DataFrame(
    [value for key, value in histDesignInfoDict.items()],
    columns=['designInfo'])
histDesignInfoDF.index = pd.MultiIndex.from_tuples(
    [key for key, value in histDesignInfoDict.items()],
    names=['rhsMaskIdx', 'ensTemplate'])
#
if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
    slurmTaskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
else:
    slurmTaskID = 0
if os.getenv('SLURM_ARRAY_TASK_COUNT') is not None:
    slurmTaskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
else:
    slurmTaskCount = 1
if os.getenv('SLURM_ARRAY_TASK_MIN') is not None:
    slurmTaskMin = int(os.getenv('SLURM_ARRAY_TASK_MIN'))
else:
    slurmTaskMin = 0
#  ##################################################################
#  slurmTaskID = 23
#  estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
#  slurmTaskCount = 57
#  slurmTaskMin = 0
#  ##################################################################
iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold']
eps = np.spacing(1)
eigenValueDict = {}
ADict = {}
BDict = {}
KDict = {}
CDict = {}
DDict = {}
HDict = {}
#################################################################
for modelIdx, (name, iRGroup0) in enumerate(iRPerTerm.groupby(iRGroupNames)):
    # print('{}, {}'.format(modelIdx, modelIdx % slurmTaskCount))
    if (modelIdx % slurmTaskCount) != slurmTaskID:
        continue
    iRWorkingCopy = iRGroup0.copy()
    iRWorkingCopy.dropna(inplace=True, axis='columns')
    ##
    lhsMaskIdx, designFormula, rhsMaskIdx, fold = name
    if not designIsLinear[designFormula]:
        continue
    if not (lhsMaskIdx in [0, 9, 10, 28, 29, 47, 48]):
        continue
    print('Calculating state space coefficients for {}\n    {}\n'.format(
        lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr'], name))
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    #
    exogNames0 = iRWorkingCopy.columns.map(termPalette.xs('exog', level='type').loc[:, ['source', 'term']].set_index('term')['source']).to_series()
    exogNames = exogNames0.dropna().to_list()
    if designFormula != 'NULL':
        # designInfo = designInfoDict[designFormula]
        # exogNames = designInfo.term_names
        histLens = [designHistOptsDict[designFormula]['historyLen']]
    else:
        histLens = []
    ensTemplate = lhsMaskParams['ensembleTemplate']
    selfTemplate = lhsMaskParams['selfTemplate']
    if ensTemplate != 'NULL':
        # ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
        histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
    if selfTemplate != 'NULL':
        # selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
        histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
    iRWorkingCopy.rename(columns=termPalette.loc[:, ['source', 'term']].set_index('term')['source'], inplace=True)
    # if either ensemble or self are NULL, fill with zeros:
    endogNames = iRWorkingCopy.index.get_level_values('target').unique().to_list()
    for sN in endogNames:
        if sN not in iRWorkingCopy.columns:
            iRWorkingCopy.loc[:, sN] = 0.
    trialInfo = iRWorkingCopy.index.to_frame().reset_index(drop=True)
    ##
    #######################
    # Apply ERA
    Phi = pd.concat([iRWorkingCopy.reset_index(drop=True), trialInfo.loc[:, ['bin', 'target']]], axis='columns')
    Phi.columns.name = 'feature'
    kernelMask = (Phi['bin'] >= 0) & (Phi['bin'] <= max(histLens))
    Phi = Phi.loc[kernelMask, ['bin', 'target'] + endogNames + exogNames]
    Phi.loc[Phi['bin'] == 0, endogNames] = 0. # enforce that y[t] = F(y[t-p]), p positive
    Phi.sort_values(by=['bin', 'target'], kind='mergesort', inplace=True)
    Phi.set_index(['bin', 'target'], inplace=True)
    Phi = Phi.stack().unstack('target').T
    #
    nLags = int(10 * max(histLens) / binInterval)
    #
    A, B, K, C, D, H, (fig, ax,) = tdr.ERA(
        Phi, maxNDim=None,
        endogNames=endogNames, exogNames=exogNames, nLags=nLags,
        plotting=False)
    ADict[name] = A
    BDict[name] = B
    KDict[name] = K
    CDict[name] = C
    DDict[name] = D
    HDict[name] = H
    # sS = ctrl.ss(A, B, C, D, dt=binInterval)
    theseEigValsList = []
    for nd in range(A.shape[0], 1, -10):
        w, v = np.linalg.eig(A.iloc[:nd, :nd])
        w = pd.Series(w)
        logW = np.log(w)
        theseEigVals = pd.concat({
            'complex': w,
            'logComplex': logW,
            'magnitude': w.abs(),
            'phase': w.apply(np.angle),
            'real': w.apply(np.real),
            'r': logW.apply(np.real).apply(np.exp),
            'delta': logW.apply(np.imag),
            'imag': w.apply(np.imag)}, axis='columns')
        theseEigVals.loc[:, 'stateNDim'] = nd
        theseEigVals.loc[:, 'nDimIsMax'] = (nd == A.shape[0])
        theseEigVals.index.name = 'eigenvalueIndex'
        theseEigVals.set_index('stateNDim', append=True, inplace=True)
        #
        theseEigVals.loc[:, 'tau'] = 2 * np.pi * binInterval / theseEigVals['delta']
        theseEigVals.loc[:, 'chi'] = - (theseEigVals['r'].apply(np.log) ** (-1)) * binInterval
        theseEigVals.loc[:, 'isOscillatory'] = (theseEigVals['imag'].abs() > 1e-6) | (theseEigVals['real'] < 0)
        theseEigVals.loc[:, 'isStable'] = theseEigVals['magnitude'] <= 1
        #
        theseEigVals.loc[:, 'eigValType'] = ''
        theseEigVals.loc[theseEigVals['isOscillatory'] & theseEigVals['isStable'], 'eigValType'] = 'oscillatory decay'
        theseEigVals.loc[~theseEigVals['isOscillatory'] & theseEigVals['isStable'], 'eigValType'] = 'pure decay'
        theseEigVals.loc[theseEigVals['isOscillatory'] & ~theseEigVals['isStable'], 'eigValType'] = 'oscillatory growth'
        theseEigVals.loc[~theseEigVals['isOscillatory'] & ~theseEigVals['isStable'], 'eigValType'] = 'pure growth'
        theseEigValsList.append(theseEigVals)
    if len(theseEigValsList):
        eigenValueDict[name] = pd.concat(theseEigValsList)
#
if len(eigenValueDict) > 0:
    eigDF = pd.concat(eigenValueDict, names=iRGroupNames)
    eigDF.to_hdf(estimatorPath, 'eigenvalues')
allA = pd.concat(ADict, names=iRGroupNames)
allA.to_hdf(estimatorPath, 'A')
allB = pd.concat(BDict, names=iRGroupNames)
allB.to_hdf(estimatorPath, 'B')
allK = pd.concat(KDict, names=iRGroupNames)
allK.to_hdf(estimatorPath, 'K')
allC = pd.concat(CDict, names=iRGroupNames)
allC.to_hdf(estimatorPath, 'C')
allD = pd.concat(DDict, names=iRGroupNames)
allD.to_hdf(estimatorPath, 'D')
allH = pd.concat(HDict, names=iRGroupNames)
allH.to_hdf(estimatorPath, 'H')
####