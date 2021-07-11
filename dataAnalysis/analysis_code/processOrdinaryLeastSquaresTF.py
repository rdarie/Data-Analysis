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
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
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
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_d', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr2_ta',
        'blockIdx': '2', 'exp': 'exp202101281100'}
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
#
lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
#
binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
#
with pd.HDFStore(estimatorPath) as store:
    if 'plotOpts' in store:
        termPalette = pd.read_hdf(store, 'plotOpts')
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
        loadedIR = True
    else:
        loadedIR = False

# prep rhs dataframes
rhsPipelineAveragerDict = {}
rhGroupDict = {}
selfDesignInfoDict = {}
selfImpulseDict = {}
ensDesignInfoDict = {}
ensImpulseDict = {}
selfImpulseDict = {}
for rhsMaskIdx in range(rhsMasks.shape[0]):
    #
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
    for ensTemplate in lOfEnsembleTemplates:
        if ensTemplate is None:
            continue
        ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
        ensFormula += ' - 1'
        print('Evaluating ensemble history {}'.format(ensFormula))
        ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
        exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
        ensDesignMatrix = ensPt.fit_transform(exampleRhGroup)
        ensDesignInfo = ensDesignMatrix.design_info
        print(ensDesignInfo.term_names)
        print('\n')
        ensDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
    for selfTemplate in lOfSelfTemplates:
        if selfTemplate is None:
            continue
        selfFormula = ' + '.join([selfTemplate.format(cN) for cN in rhGroup.columns])
        selfFormula += ' - 1'
        print('Evaluating self history {}'.format(selfFormula))
        selfPt = PatsyTransformer(selfFormula, eval_env=thisEnv, return_type="matrix")
        exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
        selfDesignMatrix = selfPt.fit_transform(exampleRhGroup)
        selfDesignInfo = selfDesignMatrix.design_info
        selfDesignInfoDict[(rhsMaskIdx, selfTemplate)] = selfDesignInfo
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
    #
    pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
    exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
    designMatrix = pt.fit_transform(exampleLhGroup)
    designInfo = designMatrix.design_info
    designInfoDict0[(lhsMaskIdx, designFormula)] = designInfo
#
designInfoDF = pd.DataFrame(
    [value for key, value in designInfoDict0.items()],
    columns=['designInfo'])
designInfoDF.index = pd.MultiIndex.from_tuples([key for key, value in designInfoDict0.items()], names=['lhsMaskIdx', 'design'])

from scipy import signal
import control as ctrl
histOpts = hto0
### sanity check impulse responses
rawProbeTermName = 'pca_ta_all001'
probeTermName = 'rcb({}, **hto0)'.format(rawProbeTermName)

from sympy.matrices import Matrix, eye, zeros
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.abc import z
from sympy import simplify, cancel, ratsimp, degree
iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold', 'electrode']
##
eps = np.spacing(1)
##
for name, iRGroup0 in iRPerTerm.groupby(iRGroupNames):
    iRWorkingCopy = iRGroup0.copy()
    ##
    iRWorkingCopy.dropna(inplace=True, axis='columns')
    ##
    lhsMaskIdx, designFormula, rhsMaskIdx, fold, electrode = name
    if lhsMaskIdx != 1:
        continue
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    #
    designInfo = designInfoDict0[(lhsMaskIdx, designFormula)]
    termNames = designInfo.term_names
    signalNames = iRWorkingCopy.index.get_level_values('target').unique().to_list()
    ensTemplate = lhsMaskParams['ensembleTemplate']
    if ensTemplate is not None:
        ensDesignInfo = ensDesignInfoDict[(rhsMaskIdx, ensTemplate)]
    if selfTemplate is not None:
        selfDesignInfo = selfDesignInfoDict[(rhsMaskIdx, selfTemplate)]
    selfTemplate = lhsMaskParams['selfTemplate']
    iRWorkingCopy.rename(columns={ensTemplate.format(key): key for key in signalNames}, inplace=True)
    #
    #
    Zs = pd.DataFrame(np.nan, index=signalNames, columns=signalNames + termNames)
    MN = zeros(*Zs.shape)
    for targetName, iRGroup in iRWorkingCopy.groupby('target'):
        '''if ensTemplate is not None:
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
            selfTermNames = []'''
        tBins = iRGroup.index.get_level_values('bin')
        kernelTMask = (tBins >= 0) & (tBins <= histOpts['historyLen'])
        # sns.heatmap(iRGroup.loc[kernelTMask, :])
        # sns.displot(iRGroup.loc[kernelTMask, :].abs().to_numpy().flatten())
        for termName in iRGroup.columns:
            kernel = iRGroup.loc[kernelTMask, termName]
            b = kernel.to_numpy()
            rowIdx = Zs.index.to_list().index(targetName)
            colIdx = Zs.columns.to_list().index(termName)
            for order, coefficient in enumerate(b):
                if np.abs(coefficient) > eps:
                    MN[rowIdx, colIdx] += coefficient * z ** (-(order + 1))
            # trFun = ctrl.tf(b, a, dt=binInterval)
            # Zs.loc[targetName, termName] = trFun / ctrl.tf('z', dt=binInterval)
            # sns.heatmap(sS.A, annot=True, annot_kws={'fontsize': 2})
    M = MN[:, :len(signalNames)]
    N = MN[:, len(signalNames):]
    G = (z * eye(len(signalNames)) - M).inv() * N
    break
    tf_num = []
    tf_den = []
    for rowIdx, targetName in enumerate(Zs.index):
        tf_num.append([])
        tf_den.append([])
        for colIdx, termName in enumerate(signalNames):
            num, den = cancel(G[rowIdx, colIdx]).as_numer_denom()
            print('\n{}\n------\n{}\n'.format(num, den))
            try:
                rawNum = np.asarray(num.as_poly().all_coeffs(), dtype=float)
            except:
                if num.is_constant():
                    rawNum = np.asarray([num], dtype=float)
            tf_num[rowIdx].append(rawNum)
            try:
                rawDen = np.asarray(den.as_poly().all_coeffs(), dtype=float)
            except:
                if den.is_constant():
                    rawDen = np.asarray([den], dtype=float)
            tf_den[rowIdx].append(rawDen)
    trFun = ctrl.tf(tf_num, tf_den, dt=binInterval)
    sS = ctrl.ss(trFun)
    #
    '''A = Matrix(sS.A)
    B = Matrix(sS.B)
    C = Matrix(sS.C)
    D = Matrix(sS.D)
    GRec = C * (z * eye(*A.shape) - A).inv() * B + D
    #
    for rowIdx, targetName in enumerate(Zs.index):
        for colIdx, termName in enumerate(signalNames):
            recNum, recDen = cancel(GRec[rowIdx, colIdx]).as_numer_denom()
            print('\n{}\n------\n{}\n'.format(recNum, recDen))
            break
        break'''
    sSMin = ctrl.minreal(sS)
    # sns.heatmap(sS.A)
    break
####