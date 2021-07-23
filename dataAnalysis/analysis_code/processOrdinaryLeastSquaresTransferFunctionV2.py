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
import logging
logging.captureWarnings(True)
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
from scipy.stats import zscore, mode
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
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr_pca_ta',
        'blockIdx': '2', 'exp': 'exp202101271100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    
'''

def optimalSVDThreshold(Y):
    beta = Y.shape[0] / Y.shape[1]
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return omega

def ERAMittnik(N, maxNDim=None, endogNames=None, exogNames=None, plotting=False):
    tBins = np.unique(N.columns.get_level_values('bin'))
    nRegressors = len(endogNames + exogNames)
    H0 = pd.concat([N.shift(-nRegressors * it, axis='columns').fillna(0) for it in range(tBins.size)])
    H1 = H0.shift(-nRegressors, axis='columns').fillna(0)
    u, s, vh = np.linalg.svd(H0, full_matrices=False)
    if maxNDim is None:
        optThresh = optimalSVDThreshold(H0) * np.median(s)
        maxNDim = (s > optThresh).sum()
    stateSpaceNDim = min(maxNDim, u.shape[0])
    u = u[:, :stateSpaceNDim]
    s = s[:stateSpaceNDim]
    vh = vh[:stateSpaceNDim, :]
    O = u @ np.diag(np.sqrt(s))
    R = np.diag(np.sqrt(s)) @ vh
    F = np.diag(np.sqrt(s) ** -1) @ u.T @ H1 @ vh.T @ np.diag(np.sqrt(s) ** -1)
    C = pd.DataFrame(O[:len(endogNames), :], index=endogNames)
    K = pd.DataFrame(R[:, :len(endogNames)], columns=endogNames)
    G = pd.DataFrame(R[:, len(endogNames):len(endogNames) + len(exogNames)], columns=exogNames)
    A = pd.DataFrame(F + K @ C)
    A.columns.name = 'state'
    A.index.name = 'state'
    D = pd.DataFrame(N.xs(0, level='bin', axis='columns').loc[:, exogNames], index=endogNames, columns=exogNames)
    B = pd.DataFrame(G + K @ D, columns=exogNames)
    B.index.name = 'state'
    if not plotting:
        return A, B, C, D
    else:
        fig, ax = plt.subplots()
        ax.plot(s)
        ax.set_ylabel('eigenvalue of H0')
        ax.axvline(maxNDim)
        return A, B, C, D, fig, ax

def ERA(
        Phi, maxNDim=None, endogNames=None, exogNames=None,
        plotting=False, nLags=None, method='ERA'):
    tBins = np.unique(Phi.columns.get_level_values('bin'))
    bins2Index = {bN: bIdx for bIdx, bN in enumerate(tBins)}
    Phi.rename(columns=bins2Index, inplace=True)
    # following notation from vicario 2014 (thesis)
    p = Phi.columns.get_level_values('bin').max() # order of the varx model
    Phi.loc[:, Phi.columns.get_level_values('bin') == p] = 0
    q = len(endogNames)
    m = len(exogNames)
    N = nLags
    Phi1 = {}
    Phi2 = {}
    Phi1[0] = Phi.xs(0, level='bin', axis='columns').loc[:, exogNames] # D
    D = Phi1[0]
    for j in range(1, N+1):
        if j <= p:
            Phi1[j] = Phi.xs(j, level='bin', axis='columns').loc[:, exogNames]
            Phi2[j] = Phi.xs(j, level='bin', axis='columns').loc[:, endogNames]
        else:
            Phi1[j] = Phi.xs(p, level='bin', axis='columns').loc[:, exogNames] * 0
            Phi2[j] = Phi.xs(p, level='bin', axis='columns').loc[:, endogNames] * 0
    # matrix form from Juang 1992 (p. 324)
    # lPhiMat * Psi_sDF = rPhiMat_s
    # lPhiMat * Psi_gDF = rPhiMat_g
    lPhiMatLastRowList = [
        Phi2[k].reset_index(drop=True) * (-1) for k in range(N-1, 0, -1)
    ] + [pd.DataFrame(np.eye(q))]
    lPhiMatLastRow = pd.concat(lPhiMatLastRowList, axis='columns', ignore_index=True)
    lPhiMat = pd.concat([
        lPhiMatLastRow.shift(k * q * (-1), axis='columns').fillna(0)
        for k in range(N-1, -1, -1)
        ])
    lPhiMatInv = np.linalg.inv(lPhiMat)
    #
    if D.any().any():
        rPhiMat_s = pd.concat({
            k: Phi1[k] + Phi2[k].to_numpy() @ D.to_numpy()
            for k in range(1, N+1)
            }, names=['bin'])
    else:
        rPhiMat_s = pd.concat({
            k: Phi1[k]
            for k in range(1, N+1)
            }, names=['bin'])
    Psi_sDF = lPhiMatInv @ rPhiMat_s
    Psi_sDF.index = rPhiMat_s.index
    Psi_sDF = Psi_sDF.unstack(level='bin')
    Psi_sDF.sort_index(axis='columns', level='bin', sort_remaining=False, kind='mergesort', inplace=True)
    #
    rPhiMat_g = pd.concat({
        k: Phi2[k]
        for k in range(1, N+1)
        }, names=['bin'])
    Psi_gDF = lPhiMatInv @ rPhiMat_g
    Psi_gDF.index = rPhiMat_g.index
    Psi_gDF = Psi_gDF.unstack(level='bin')
    Psi_gDF.sort_index(axis='columns', level='bin', sort_remaining=False, kind='mergesort', inplace=True)
    '''
    # recursive substitution way
    Psi_s = {}
    Psi_g = {}
    Psi_s[0] = Phi1[0]  # also D
    for j in range(1, nLags):
        print('j = {}'.format(j))
        if j <= p:
            Phi1[j] = Phi.xs(j, level='bin', axis='columns').loc[:, exogNames]
            Phi2[j] = Phi.xs(j, level='bin', axis='columns').loc[:, endogNames]
        else:
            Phi1[j] = Phi.xs(p, level='bin', axis='columns').loc[:, exogNames] * 0
            Phi2[j] = Phi.xs(p, level='bin', axis='columns').loc[:, endogNames] * 0
        Psi_s[j] = Phi1[j]
        Psi_g[j] = Phi2[j]
        #
        print('\tnorm(Phi1) = {}'.format(np.linalg.norm(Phi1[j])))
        print('\tnorm(Phi2) = {}'.format(np.linalg.norm(Phi2[j])))
        for h in range(1, j+1):
            # print('\th = {}'.format(h))
            Psi_s[j] += Phi2[h].to_numpy() @ Psi_s[j-h].to_numpy()
            if h < j:
                Psi_g[j] += Phi2[h].to_numpy() @ Psi_g[j-h].to_numpy()
    #
    Psi_sDF = pd.concat(Psi_s, axis='columns', names=['bin'])
    Psi_gDF = pd.concat(Psi_g, axis='columns', names=['bin'])'''
    if plotting:
        dt = mode(np.diff(tBins))[0][0]
        fig, ax = plt.subplots(3, 1)
        plotS = Psi_sDF.stack('feature')
        plotG = Psi_gDF.stack('feature')
        for rowName, row in plotS.iterrows():
            ax[0].plot(row.index * dt, row, label=rowName)
        for rowName, row in plotG.iterrows():
            ax[1].plot(row.index * dt, row, label=rowName)
        ax[0].set_title('System Markov parameters (Psi_s)')
        ax[0].axvline(p * dt)
        ax[0].legend()
        ax[1].set_title('Gain Markov parameters (Psi_g)')
        ax[1].axvline(p * dt)
        ax[1].legend()
        ax[1].set_xlabel('Time (sec)')
    else:
        fig, ax = None, None
    PsiDF = pd.concat([Psi_gDF, Psi_sDF], axis='columns')
    PsiDF.sort_index(
        axis='columns', level=['bin'],
        kind='mergesort', sort_remaining=False, inplace=True)
    PsiDF.loc[:, PsiDF.columns.get_level_values('bin') == N] = 0.
    ####
    HComps = []
    PsiBins = PsiDF.columns.get_level_values('bin')
    for hIdx in range(1, int(N / 2) + 1):
        HMask = (PsiBins >= hIdx) & (PsiBins < (hIdx + int(N / 2) - 1))
        HComps.append(PsiDF.loc[:, HMask].to_numpy())
    H0 = np.concatenate(HComps[:-1])
    H1 = np.concatenate(HComps[1:])
    if method == 'ERA':
        svdTargetH = H0
    elif method == 'ERADC':
        HH0 = H0 @ H0.T
        HH1 = H1 @ H0.T
        svdTargetH = HH0
    u, s, vh = np.linalg.svd(svdTargetH, full_matrices=False)
    if maxNDim is None:
        optThresh = optimalSVDThreshold(svdTargetH) * np.median(s[:int(N / 2)])
        maxNDim = (s > optThresh).sum()
    stateSpaceNDim = min(maxNDim, u.shape[0])
    if plotting:
        ax[2].plot(s)
        ax[2].set_title('singular values of Hankel matrix (ERA)')
        ax[2].set_ylabel('s')
        ax[2].axvline(stateSpaceNDim)
        ax[2].set_xlabel('Count')
    #
    u = u[:, :stateSpaceNDim]
    s = s[:stateSpaceNDim]
    vh = vh[:stateSpaceNDim, :]
    ###
    O = u @ np.diag(np.sqrt(s))
    RColMask = PsiDF.columns.get_level_values('bin').isin(range(len(HComps)))
    if method == 'ERA':
        R = pd.DataFrame(np.diag(np.sqrt(s)) @ vh, index=None, columns=PsiDF.loc[:, RColMask].columns)
        ARaw = np.diag(np.sqrt(s) ** -1) @ u.T @ H1 @ vh.T @ np.diag(np.sqrt(s) ** -1)
    elif method == 'ERADC':
        R = pd.DataFrame(np.diag(np.sqrt(s)) @ u.T @ H0, index=None, columns=PsiDF.loc[:, RColMask].columns)
        ARaw = np.diag(np.sqrt(s) ** -1) @ u.T @ HH1 @ vh.T @ np.diag(np.sqrt(s) ** -1)
    ###
    A = pd.DataFrame(ARaw)
    A.columns.name = 'state'
    A.index.name = 'state'
    C = pd.DataFrame(O[:len(endogNames), :], index=endogNames)
    BK = R.xs(1, level='bin', axis='columns')
    B = BK.loc[:, exogNames]
    B.index.name = 'state'
    K = BK.loc[:, endogNames]
    K.index.name = 'state'
    return A, B, K, C, D, (fig, ax,)

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
with pd.HDFStore(estimatorPath) as store:
    if 'termPalette' in store:
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
        loadedIR = True
    else:
        loadedIR = False
'''
# prep rhs dataframes
histDesignInfoDict = {}
for rhsMaskIdx in range(rhsMasks.shape[0]):
    #
    print('\n    On rhsRow {}\n'.format(rhsMaskIdx))
    rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
    rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
    rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
    for ensTemplate in lOfHistTemplates:
        if ensTemplate is not None:
            ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
            ensFormula += ' - 1'
            print('Evaluating ensemble history {}'.format(ensFormula))
            ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
            exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
            ensDesignMatrix = ensPt.fit_transform(exampleRhGroup)
            ensDesignInfo = ensDesignMatrix.design_info
            print(ensDesignInfo.term_names)
            print('\n')
            histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
#
designInfoDict = {}
for lhsMaskIdx in range(lhsMasks.shape[0]):
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    designFormula = lhsMaskParams['designFormula']
    #
    if designFormula != 'NULL':
        if designFormula not in designInfoDict:
            print('Evaluating exog regressors {}'.format(designFormula))
            formulaIdx = lOfDesignFormulas.index(designFormula)
            lhGroup = lhsDF.loc[:, lhsMask]
            pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
            exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
            designMatrix = pt.fit_transform(exampleLhGroup)
            designInfo = designMatrix.design_info
            designInfoDict[designFormula] = designInfo
designInfoDF = pd.Series(designInfoDict).to_frame(name='designInfo')
designInfoDF.index.name = 'design'
histDesignInfoDF = pd.DataFrame(
    [value for key, value in histDesignInfoDict.items()],
    columns=['designInfo'])
histDesignInfoDF.index = pd.MultiIndex.from_tuples([key for key, value in histDesignInfoDict.items()], names=['rhsMaskIdx', 'ensTemplate'])'''

iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold']
eps = np.spacing(1)
eigenValueDict = {}
ADict = {}
BDict = {}
KDict = {}
CDict = {}
DDict = {}
if arguments['plotting']:
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'OKID_ERA'))
    cm = PdfPages(pdfPath)
else:
    import contextlib
    cm = contextlib.nullcontext()
with cm as pdf:
    for name, iRGroup0 in iRPerTerm.groupby(iRGroupNames):
        iRWorkingCopy = iRGroup0.copy()
        iRWorkingCopy.dropna(inplace=True, axis='columns')
        ##
        lhsMaskIdx, designFormula, rhsMaskIdx, fold = name
        if not designIsLinear[designFormula]:
            continue
        if not (fold == lastFoldIdx):
            continue
        print('Calculating state space coefficients for {}'.format(lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr']))
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
        nLags = int(5 * max(histLens) / binInterval)
        #
        A, B, K, C, D, (fig, ax,) = ERA(
            Phi, maxNDim=None,
            endogNames=endogNames, exogNames=exogNames, nLags=nLags,
            plotting=arguments['plotting'])
        ADict[name] = A
        BDict[name] = B
        KDict[name] = K
        CDict[name] = C
        DDict[name] = D
        #
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
        eigenValueDict[name] = pd.concat(theseEigValsList)
        if arguments['plotting']:
            fig.suptitle('{}'.format(name))
            fig.tight_layout()
            pdf.savefig(
                bbox_inches='tight',
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
#
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

eigValTypes = ['oscillatory decay', 'pure decay', 'oscillatory growth', 'pure growth']
eigValColors = sns.color_palette('Set2')
eigValPaletteDict = {}
eigValColorAlpha = 0.5
for eIx, eType in enumerate(eigValTypes):
    eigValPaletteDict[eType] = tuple([col for col in eigValColors[eIx]] + [eigValColorAlpha])
eigValPalette = pd.Series(eigValPaletteDict)
eigValPalette.index.name = 'eigValType'
eigValPalette.name = 'color'
eigValPalette.to_hdf(estimatorPath, 'eigValPalette')
##############################################################################################################
##############################################################################################################
##############################################################################################################
if False:
    # sanity check that signal dynamics are independent of regressors
    meanA = allA.groupby(['rhsMaskIdx', 'state']).mean()
    stdA = allA.groupby(['rhsMaskIdx', 'state']).std()
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(meanA.reset_index(drop=True), ax=ax[0])
    sns.heatmap(stdA.reset_index(drop=True), ax=ax[1])
    #

    plotEig = pd.concat({
        'magnitude': eigDF.apply(lambda x: np.absolute(x)),
        'phase': eigDF.apply(lambda x: np.angle(x)),
        'real': eigDF.apply(lambda x: x.real),
        'imag': eigDF.apply(lambda x: x.imag)}, axis='columns')
    plotEig.reset_index(inplace=True)
    g = sns.relplot(
        row='lhsMaskIdx', col='rhsMaskIdx',
        x='real', y='imag', hue='electrode', style='electrode',
        kind='scatter', data=plotEig)