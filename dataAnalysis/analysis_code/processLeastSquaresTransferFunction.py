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
from datetime import datetime
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.plotting.aligned_signal_plots as asp
# from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore, mode
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))
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
import colorsys, sys, time
import control as ctrl
import contextlib
idxSl = pd.IndexSlice
#
print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
for arg in sys.argv:
    print(arg)

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


def calcTransferFunctionFromLeastSquares():
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
    lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
    allTargetsDF = (
        pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets')
        .xs(arguments['estimatorName'], level='regressorName'))
    rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
    # rhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(selectionNameRhs))
    rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
    lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
    #
    iRPerTerm = pd.read_hdf(estimatorPath, 'impulseResponsePerTerm')
    iRPerFactor = pd.read_hdf(estimatorPath, 'impulseResponsePerFactor')
    sourcePalette = pd.read_hdf(estimatorPath, 'sourcePalette')
    termPalette = pd.read_hdf(estimatorPath, 'termPalette')
    factorPalette = pd.read_hdf(estimatorPath, 'factorPalette')
    trialTypePalette = pd.read_hdf(estimatorPath, 'trialTypePalette')

    lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
    pt = PatsyTransformer('vx + vy + e:(a + r) - 1', return_type="dataframe")
    uDF = pt.fit_transform(lhsDF)
    uDF.rename(
        columns={
            'e[{}]:{}'.format(eName, sVar): '[{}]{}'.format(eName, sVar)
            for eName, sVar in product(lhsDF['e'].unique(), ['a', 'r'])},
        inplace=True)
    inputSets = {
        'velocity': ['vx', 'vy']
    }
    for elecName, uGroup in uDF.groupby('electrode'):
        inputSets[elecName] = [cN for cN in uDF.columns if elecName in cN]
    ####
    # prep rhs dataframes
    '''
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
        '''
    ###
    iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold']
    nTFsToProcess = iRPerTerm.groupby(iRGroupNames).ngroups
    print('There are {} transfer functions to process (nTFsToProcess)'.format(nTFsToProcess))
    #######
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        slurmTaskID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        estimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(slurmTaskID))
        transferFuncPath = os.path.join(
            estimatorsSubFolder,
            fullEstimatorName + '_{}_{}_tf.h5'.format(arguments['eraMethod'], slurmTaskID)
            )
    else:
        slurmTaskID = 0
        transferFuncPath = os.path.join(
            estimatorsSubFolder,
            fullEstimatorName + '_{}_tf.h5'.format(arguments['eraMethod'])
            )
    print('This is job {} (slurmTaskID)'.format(slurmTaskID))
    # if os.getenv('SLURM_ARRAY_TASK_COUNT') is not None:
    #     slurmTaskCount = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    # else:
    #     slurmTaskCount = 1
    # print('There are {} slurm jobs (slurmTaskCount)'.format(slurmTaskCount))
    slurmTaskCount = processSlurmTaskCount
    if os.getenv('SLURM_ARRAY_TASK_MIN') is not None:
        slurmTaskMin = int(os.getenv('SLURM_ARRAY_TASK_MIN'))
    else:
        slurmTaskMin = 0
    #
    #  ##################################################################
    if arguments['debugging']:
        slurmTaskID = 12
        transferFuncPath = os.path.join(
            estimatorsSubFolder,
            fullEstimatorName + '_{}_{}_tf.h5'.format(arguments['eraMethod'], slurmTaskID)
            )
        slurmTaskCount = processSlurmTaskCount
        slurmTaskMin = 0
    #  ##################################################################
    #
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], 'regression', 'transfer_functions')
        # if (slurmTaskID != slurmTaskMin):
        #     # give the first process a chance to make the folder
        #     time.sleep(60)
        if not os.path.exists(figureOutputFolder):
            try:
                os.makedirs(figureOutputFolder, exist_ok=False)
            except Exception:
                # another process beat this one to the directory creation
                print('process {}: figure output folder did not exist, but then existed by the time I tried to make it!'.format(slurmTaskID))
        pdfPath = os.path.join(
            figureOutputFolder, '{}_{}_{}_transfer_functions.pdf'.format(fullEstimatorName, arguments['eraMethod'], slurmTaskID)
            )
        cm = PdfPages(pdfPath)
    else:
        cm = contextlib.nullcontext()

    slurmGroupSize = int(np.ceil(nTFsToProcess / slurmTaskCount))
    # slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / slurmTaskCount))
    print('Each job will process {} tasks (slurmGroupSize)'.format(slurmGroupSize))
    #######
    eps = np.spacing(1)
    eigenValueDict = {}
    ADict = {}
    BDict = {}
    KDict = {}
    CDict = {}
    DDict = {}
    HDict = {}
    PsiDict = {}
    inputDrivenDict = {}
    oneStepKalmanDict = {}
    untruncatedHEigenValsDict = {}
    ctrbObsvRanksDict = {}
    #
    #################################################################
    with cm as pdf:
        for modelIdx, (name, iRGroup0) in enumerate(iRPerTerm.groupby(iRGroupNames)):
            # print('slurmTaskID ({}), modelIdx ({}), modelIdx // slurmGroupSize ({})'.format(slurmTaskID, modelIdx, modelIdx // slurmGroupSize))
            if (modelIdx // slurmGroupSize) != slurmTaskID:
                continue
            lhsMaskIdx, designFormula, rhsMaskIdx, fold = name
            rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            vDF = pd.concat([rhGroup, uDF], axis='columns')
            #
            iRWorkingCopy = iRGroup0.copy()
            # iRWorkingCopy.dropna(axis='columns')
            exogNames0 = iRWorkingCopy.columns[~iRWorkingCopy.isna().all()].map(
                termPalette.xs('exog', level='type').loc[:, ['source', 'term']].set_index('term')['source']).to_series()
            exogNames = exogNames0.dropna().to_list()
            iRWorkingCopy.rename(columns=termPalette.loc[:, ['source', 'term']].set_index('term')['source'], inplace=True)
            # print('lhsMaskIdx, designFormula, rhsMaskIdx, fold = {}'.format(name))
            if not designIsLinear[designFormula]:
                print('Skipping because designIsLinear[designFormula] = {}'.format(designIsLinear[designFormula]))
                continue
            if not (lhsMaskIdx in lhsMasksOfInterest['varVsEnsemble']):
                print('not (lhsMaskIdx in lhsMasksOfInterest)')
                continue
            print('Calculating state space coefficients for\n{}\n    {}\n'.format(
                lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr'],
                {k: v for k, v in zip(iRGroupNames, name)}
                ))
            lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
            lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
            #
            # thisPredDF = pd.read_hdf(estimatorPath, 'predictions')
            #
            if designFormula != 'NULL':
                thisHistLen = designHistOptsDict[designFormula]['historyLen'] + designHistOptsDict[designFormula]['timeDelay']
                histLens = [thisHistLen]
            else:
                histLens = []
            ensTemplate = lhsMaskParams['ensembleTemplate']
            selfTemplate = lhsMaskParams['selfTemplate']
            if ensTemplate != 'NULL':
                thisHistLen = templateHistOptsDict[ensTemplate]['historyLen'] + templateHistOptsDict[ensTemplate]['timeDelay']
                histLens.append(thisHistLen)
            if selfTemplate != 'NULL': #
                thisHistLen = templateHistOptsDict[selfTemplate]['historyLen'] + templateHistOptsDict[selfTemplate]['timeDelay']
                histLens.append(thisHistLen)
            # if either ensemble or self are NULL, fill with zeros:
            endogNames = iRWorkingCopy.index.get_level_values('target').unique().to_list()
            for sN in endogNames:
                if iRWorkingCopy[sN].isna().all():
                    iRWorkingCopy.loc[:, sN] = 0.
            # fill with zeros if term is not accounted for in this model
            iRWorkingCopy = iRWorkingCopy.loc[:, ~iRWorkingCopy.isna().all()]
            ##
            #################################################################
            # Apply ERA
            kernelMask = (
                    (iRWorkingCopy.index.get_level_values('bin') >= 0) &
                    (iRWorkingCopy.index.get_level_values('bin') <= max(histLens))
                )
            iRWorkingCopy = iRWorkingCopy.loc[kernelMask, :]
            trialInfo = iRWorkingCopy.index.to_frame().reset_index(drop=True)
            Phi = pd.concat([
                    iRWorkingCopy.reset_index(drop=True),
                    trialInfo.loc[:, ['bin', 'target']]],
                axis='columns')
            Phi.columns.name = 'feature'
            Phi = Phi.loc[:, ['bin', 'target'] + endogNames + exogNames]
            Phi.fillna(0, inplace=True)
            ###
            try:
                assert (Phi.loc[Phi['bin'] == 0, endogNames].abs() < np.spacing(1)).all().all()
            except Exception:
                traceback.print_exc()
                print('This occured during\nlhsMaskIdx {},\ndesignFormula {},\nrhsMaskIdx {},\nfold {}\n'.format(*name))
            Phi.loc[Phi['bin'] == 0, endogNames] = 0. # enforce that y[t] = F(y[t-p]), p positive
            ###
            Phi.sort_values(
                by=['bin', 'target'], kind='mergesort', inplace=True)
            Phi.set_index(['bin', 'target'], inplace=True)
            Phi = Phi.stack().unstack('target').T
            # sns.heatmap(Phi, mask=Phi > 0); plt.show()
            nLags = int(10 * max(histLens) / binInterval)
            print('nLags = {}'.format(nLags))
            #
            A, B, K, C, D, H, PsiDF, reconPsiDF, hEigenVals, (fig, ax,) = tdr.ERA(
                Phi, maxNDim=None, method=arguments['eraMethod'],
                endogNames=endogNames, exogNames=exogNames, nLags=nLags,
                plotting=arguments['plotting'], verbose=2,
                checkApproximations=True,
                noEndogTerms=((ensTemplate == 'NULL') & (selfTemplate == 'NULL'))
                )
            ##################
            theseRanks = []
            O = ctrl.obsv(A, C)
            rankHere = np.linalg.matrix_rank(O.T, tol=1e-12)
            print('ndim = {} rank(obsv(A, C)) = {}'.format(A.shape[0], rankHere))
            theseRanks.append(['observability', 'full', A.shape[0], rankHere])
            for setName, colList in inputSets.items():
                presentCols = [cN for cN in colList if cN in B.columns]
                if len(presentCols):
                    R = ctrl.ctrb(A, B.loc[:, presentCols])
                    rankHere = np.linalg.matrix_rank(R, tol=1e-12)
                    print('ndim = {} rank(ctrb(A, B[{}])) = {}'.format(A.shape[0], setName, rankHere))
                    theseRanks.append(['controlability', setName, A.shape[0], rankHere])
            ctrbObsvRanksDF = pd.DataFrame(theseRanks, columns=['matrix', 'inputSubset', 'nDim', 'rank'])
            ########################################
            # print('Calculating input driven signal')
            inputDrivenDF = pd.DataFrame(0, index=lhsDF.index, columns=PsiDF.index)
            oneStepKalmanDF = pd.DataFrame(0, index=lhsDF.index, columns=PsiDF.index)
            showProgBar = True
            if showProgBar:
                progBarCtxt = tqdm(total=reconPsiDF.groupby('sampleBin', axis='columns').ngroups, mininterval=30., maxinterval=120.)
            else:
                progBarCtxt = contextlib.nullcontext()
            with progBarCtxt as pbar:
                for binIdx, psiThisLag in reconPsiDF.groupby(level='sampleBin', axis='columns', group_keys=False, sort=False):
                    if designFormula != 'NULL':  # these systems have input drive
                        exogPsi = psiThisLag.loc[:, idxSl[uDF.columns, binIdx]].T
                        exogPsi.index = exogPsi.index.droplevel('sampleBin')
                        thisPiece = tdr.inputDrivenDynamics(
                            psi=exogPsi, u=uDF, lag=binIdx,
                            groupVar='trialUID', joblibBackendArgs=dict(backend='loky'),
                            runInParallel=True, verbose=False)
                        inputDrivenDF += thisPiece
                    # pdb.set_trace()
                    reconPsiThisLag = psiThisLag.copy().T
                    reconPsiThisLag.index = reconPsiThisLag.index.droplevel('sampleBin')
                    thisReconPiece = tdr.inputDrivenDynamics(
                        psi=reconPsiThisLag, u=vDF.loc[:, reconPsiThisLag.index], lag=binIdx,
                        groupVar='trialUID', joblibBackendArgs=dict(backend='loky'),
                        runInParallel=True, verbose=False)
                    oneStepKalmanDF += thisReconPiece
                    if showProgBar:
                        pbar.update(1)
            burnInThresh = inputDrivenDF.index.get_level_values('bin').min() + burnInPeriod
            burnInMask = inputDrivenDF.index.get_level_values('bin') >= burnInThresh
            inputDrivenDF = inputDrivenDF.loc[burnInMask, :]
            oneStepKalmanDF = oneStepKalmanDF.loc[burnInMask, :]
            ########################################
            if arguments['plotting']:
                fig.tight_layout()
                figTitle = fig.suptitle(
                    'State space coefficients for\n{}\n    {}\n'.format(
                        lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr'],
                        {k: v for k, v in zip(iRGroupNames, name)}))
                pdf.savefig(
                    bbox_inches='tight',
                    bbox_extra_artists=(figTitle,)
                    )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                if designFormula != 'NULL':
                    plotInputDriven = inputDrivenDF.iloc[:, 0].to_frame(name='signal').reset_index()
                    plotInputDriven.loc[:, 'stimCondition'] = plotInputDriven.apply(lambda coln: '{}_{}'.format(coln['electrode'], coln['trialRateInHz']), axis='columns')
                    plotInputDriven.loc[:, 'kinematicCondition'] = plotInputDriven.apply(lambda coln: '{}_{}'.format(coln['pedalMovementCat'], coln['pedalDirection']), axis='columns')
                    g = sns.relplot(
                        x='bin', y='signal',
                        col='stimCondition', row='kinematicCondition',
                        data=plotInputDriven,
                        kind='line', errorbar='se', hue='trialAmplitude',
                        facet_kws=dict(margin_titles=True)
                        )
                    g.tight_layout()
                    g.set_titles({'col_template': '{col_name}', 'row_template': '{row_name}'})
                    figTitle = g.suptitle(
                        'Input driven dynamics for\n{}\n    {}\n'.format(
                            lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr'],
                            {k: v for k, v in zip(iRGroupNames, name)}))
                    pdf.savefig(
                        bbox_inches='tight',
                        )
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
            ADict[name] = A
            BDict[name] = B
            KDict[name] = K
            CDict[name] = C
            DDict[name] = D
            HDict[name] = H
            PsiDict[name] = PsiDF
            inputDrivenDict[name] = inputDrivenDF
            oneStepKalmanDict[name] = oneStepKalmanDF
            untruncatedHEigenValsDict[name] = hEigenVals
            ctrbObsvRanksDict[name] = ctrbObsvRanksDF
            theseEigValsList = []
            for nd in range(A.shape[0], 1, max(1, int(A.shape[0] / 5)) * (-1)):
                print('    Calculating eigenvalues for {} dimensional state space'.format(nd))
                w, v = np.linalg.eig(A.iloc[:nd, :nd])
                w = pd.Series(w)
                theseEigVals = pd.concat({
                    'complex': w,
                    # 'logComplex': logW,
                    #
                    'magnitude': w.abs(),
                    'phase': w.apply(np.angle),
                    #
                    'real': w.apply(np.real),
                    'imag': w.apply(np.imag),
                    # 'r': logW.apply(np.real).apply(np.exp),
                    # 'delta': logW.apply(np.imag),
                    }, axis='columns')
                theseEigVals.loc[:, 'stateNDim'] = nd
                theseEigVals.loc[:, 'nDimIsMax'] = (nd == A.shape[0])
                theseEigVals.index.name = 'eigenvalueIndex'
                theseEigVals.set_index('stateNDim', append=True, inplace=True)
                #
                # theseEigVals.loc[:, 'tau'] = 2 * np.pi * binInterval / theseEigVals['delta']
                # theseEigVals.loc[:, 'chi'] = - (theseEigVals['r'].apply(np.log) ** (-1)) * binInterval
                # theseEigVals.loc[:, 'isOscillatory'] = (theseEigVals['imag'].abs() > 1e-6) | (theseEigVals['real'] < 0)
                #
                theseEigVals.loc[:, 'tau'] = (-1) * binInterval / (theseEigVals['magnitude'].apply(np.log)) ## seconds
                theseEigVals.loc[:, 'chi'] = (2 * np.pi * binInterval) / theseEigVals['phase'].abs()
                #
                theseEigVals.loc[:, 'isStable'] = theseEigVals['magnitude'] <= 1
                theseEigVals.loc[:, 'isOscillatory'] = (theseEigVals['phase'].abs() > np.spacing(100))
                theseEigVals.loc[:, 'eigValType'] = ''
                theseEigVals.loc[theseEigVals['isOscillatory'] & theseEigVals['isStable'], 'eigValType'] = 'oscillatory decay'
                theseEigVals.loc[~theseEigVals['isOscillatory'] & theseEigVals['isStable'], 'eigValType'] = 'pure decay'
                theseEigVals.loc[theseEigVals['isOscillatory'] & ~theseEigVals['isStable'], 'eigValType'] = 'oscillatory growth'
                theseEigVals.loc[~theseEigVals['isOscillatory'] & ~theseEigVals['isStable'], 'eigValType'] = 'pure growth'
                theseEigValsList.append(theseEigVals)
            if len(theseEigValsList):
                eigenValueDict[name] = pd.concat(theseEigValsList)
    #
    if len(ADict):
        if len(eigenValueDict) > 0:
            eigDF = pd.concat(eigenValueDict, names=iRGroupNames)
            eigDF.to_hdf(transferFuncPath, 'eigenvalues')
        allA = pd.concat(ADict, names=iRGroupNames)
        allA.eraMethod = arguments['eraMethod']
        allA.to_hdf(transferFuncPath, 'A')
        allB = pd.concat(BDict, names=iRGroupNames)
        allB.eraMethod = arguments['eraMethod']
        allB.to_hdf(transferFuncPath, 'B')
        allK = pd.concat(KDict, names=iRGroupNames)
        allK.eraMethod = arguments['eraMethod']
        allK.to_hdf(transferFuncPath, 'K')
        allC = pd.concat(CDict, names=iRGroupNames)
        allC.eraMethod = arguments['eraMethod']
        allC.to_hdf(transferFuncPath, 'C')
        allD = pd.concat(DDict, names=iRGroupNames)
        allD.eraMethod = arguments['eraMethod']
        allD.to_hdf(transferFuncPath, 'D')
        allH = pd.concat(HDict, names=iRGroupNames)
        allH.to_hdf(transferFuncPath, 'H')
        allPsi = pd.concat(PsiDict, names=iRGroupNames)
        allPsi.to_hdf(transferFuncPath, 'Psi')
        allInputDrivenDF = pd.concat(inputDrivenDict, names=iRGroupNames)
        allInputDrivenDF.to_hdf(transferFuncPath, 'inputDriven')
        #
        allOneStepKalmanDF = pd.concat(oneStepKalmanDict, names=iRGroupNames)
        allOneStepKalmanDF.to_hdf(transferFuncPath, 'oneStepKalman')
        #
        allFullEigenValsDF = pd.concat(untruncatedHEigenValsDict, names=iRGroupNames)
        allFullEigenValsDF.to_hdf(transferFuncPath, 'untruncatedHEigenVals')
        #
        allctrbObsvRanksDF = pd.concat(ctrbObsvRanksDict, names=iRGroupNames)
        allctrbObsvRanksDF.to_hdf(transferFuncPath, 'ctrbObsvRanks')
    else:
        print('No tasks run on this job!!')
        if arguments['plotting']:
            os.remove(pdfPath)
    print('\n' + '#' * 50 + '\n{} Complete.\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
    return

if __name__ == "__main__":
    runProfiler = False
    if runProfiler:
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=calcTransferFunctionFromLeastSquares,
            modulesToProfile=[ash, tdr, hf],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix, outputUnits=1e-3)
    else:
        calcTransferFunctionFromLeastSquares()