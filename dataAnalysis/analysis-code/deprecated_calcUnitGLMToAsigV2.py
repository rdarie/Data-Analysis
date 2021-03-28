#!/users/rdarie/anaconda/nda/bin/python
"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --blockIdx=blockIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --debugging                               print diagnostics? [default: False]
    --dryRun                                  print diagnostics? [default: False]
    --attemptMPI                              whether to try to load MPI [default: False]
    --plotting                                plot out the correlation matrix? [default: True]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName         append a name to the resulting blocks? [default: motion]
    --inputBlockName=inputBlockName           filename for inputs [default: raster]
    --secondaryBlockName=secondaryBlockName   filename for secondary inputs [default: rig]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: raster]
    --alignQuery=alignQuery                   query what the units will be aligned to? [default: midPeak]
    --estimatorName=estimatorName             filename for resulting estimator [default: tdr]
    --selector=selector                       filename if using a unit selector
    --maskOutlierBlocks                       delete outlier trials? [default: False]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Qt5Agg')   # generate postscript output by default
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")
import pdb, traceback
from patsy import (
    ModelDesc, EvalEnvironment, Term, Sum, Treatment, INTERCEPT,
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices,
    build_design_matrices)
import os, glob
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import numpy as np
import pandas as pd
from scipy import stats
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
import dataAnalysis.preproc.ns5 as ns5
from dataAnalysis.custom_transformers.tdr import *
import statsmodels.api as sm
# import MLencoding as spykesml
# from pyglmnet import GLM
# import warnings
# warnings.filterwarnings('error')
import joblib as jb
import dill as pickle
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

try:
    if not arguments['attemptMPI']:
        raise(Exception('MPI aborted by cmd line argument'))
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    HAS_MPI = True
except Exception:
    traceback.print_exc()
    RANK = 0
    SIZE = 1
    HAS_MPI = False


def calcUnitRegressionToAsig():
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
        )
    if not os.path.exists(analysisSubFolder):
        os.makedirs(analysisSubFolder, exist_ok=True)
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName']
        )
    if not os.path.exists(alignSubFolder):
        os.makedirs(alignSubFolder, exist_ok=True)
    if arguments['processAll']:
        prefix = assembledName
    else:
        prefix = ns5FileName
    triggeredPath = os.path.join(
        alignSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['inputBlockName'], arguments['window']))
    regressorPath = os.path.join(
        alignSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['secondaryBlockName'], arguments['window']))
    fullEstimatorName = '{}_{}_{}_{}'.format(
        prefix,
        arguments['estimatorName'],
        arguments['window'],
        arguments['alignQuery'])
    estimatorSubFolder = os.path.join(
        alignSubFolder, fullEstimatorName)
    if not os.path.exists(estimatorSubFolder):
        os.makedirs(estimatorSubFolder, exist_ok=True)
    estimatorFiguresFolder = os.path.join(
        GLMFiguresFolder, fullEstimatorName)
    if not os.path.exists(estimatorFiguresFolder):
        os.makedirs(estimatorFiguresFolder, exist_ok=True)
    subsampleOpts = glmOptsLookup[arguments['estimatorName']]
    alignedAsigsKWargs.update(dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False,
        transposeToColumns='feature', concatOn='columns',
        removeFuzzyName=False,
        getMetaData=[
            'RateInHz', 'activeGroup', 'amplitude', 'amplitudeCat',
            'bin', 'electrode', 'pedalDirection', 'pedalMetaCat',
            'pedalMovementCat', 'pedalMovementDuration',
            'pedalSize', 'pedalSizeCat', 'pedalVelocityCat',
            'program', 'segment', 't'],
        **subsampleOpts,
        metaDataToCategories=False,
        verbose=False, procFun=None))
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, alignSubFolder, **arguments)
    alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
        prefix, **arguments)
    #
    featuresMetaDataPath = os.path.join(
        estimatorSubFolder, 'features_meta.pickle')
    targetH5Path = os.path.join(
        estimatorSubFolder, prefix + '_{}_{}.h5'.format(
            arguments['inputBlockName'], arguments['window']))
    regressorH5Path = os.path.join(
        estimatorSubFolder, prefix + '_{}_{}.h5'.format(
            arguments['secondaryBlockName'], arguments['window']))
    if(RANK == 0):
        # find existing variants
        variantMetaDataPath = os.path.join(
            estimatorSubFolder, 'variant_metadata.xlsx')
    #
    kList = ['GT_Right', 'K_Right', 'M_Right']
    angleList = ['{}_angle'.format(i) for i in kList]
    velList = ['{}_angular_velocity'.format(i) for i in kList]
    accList = ['{}_angular_acceleration'.format(i) for i in kList]
    pList = [
        'program0_amplitude', 'program1_amplitude',
        'program2_amplitude', 'program3_amplitude'
        ]
    #
    featureLoadArgs = alignedAsigsKWargs.copy()
    featureLoadArgs['unitNames'] = sorted([
        'RateInHz#0',
        'amplitude#0'] +
        [k + '#0' for k in angleList] +
        [k + '#0' for k in velList] +
        [k + '#0' for k in accList] +
        [p + '#0' for p in pList]
        )
    featureLoadArgs['unitQuery'] = None
    targetLoadArgs = alignedAsigsKWargs.copy()
    #
    cv_kwargs = dict(
        n_splits=7,
        shuffle=True,
        stratifyFactors=['RateInHz', 'program', 'amplitudeCat'],
        continuousFactors=['segment', 'originalIndex'])
    #
    ensembleHistoryLen = .25
    covariateHistoryLen = .15
    winSize = subsampleOpts['rollingWindow'] if subsampleOpts['rollingWindow'] is not None else 1
    halfWinSize = int(np.ceil(winSize/2))
    countPerBin = rasterOpts['binInterval'] * winSize
    binSize = rasterOpts['binInterval'] * subsampleOpts['decimate']
    #
    plotTheBases = (arguments['debugging'] and arguments['plotting'])
    showNow = arguments['debugging']
    covariateBasisTerms = {
        'nb': 21,
        'dt': rasterOpts['binInterval'],
        'endpoints': [-covariateHistoryLen, covariateHistoryLen]
        }
    cBasis = makeRaisedCosBasis(**covariateBasisTerms)
    if plotTheBases:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(cBasis)
        ax[1].set_title('raised cos basis')
    cBasis = (
        cBasis.rolling(winSize, center=True, win_type='gaussian')
        .mean(std=halfWinSize).iloc[halfWinSize:-halfWinSize:subsampleOpts['decimate']]
        .fillna(method='ffill', axis=0)
        .fillna(method='bfill', axis=0))
    cBasis.to_excel(variantMetaDataPath, sheet_name='covariateBasis')
    if plotTheBases:
        ax[1].plot(cBasis)
        ax[1].set_xlabel('Time (sec)')
        if showNow:
            plt.show()
        else:
            plt.close()

    def applyCBasis(vec, cbCol):
        return np.convolve(
            vec.to_numpy(),
            cBasis[cbCol].to_numpy(),
            mode='same')

    addHistoryTerms = {
        'nb': 5,
        'dt': rasterOpts['binInterval'],
        'b': 0.001,
        'zflag': False
        }
    # make sure history terms don't bleed into current bins
    historyEdge = raisedCosBoundary(
        b=addHistoryTerms['b'], DT=ensembleHistoryLen,
        minX=max(rasterOpts['binInterval'] * winSize / 2, 1e-3),
        nb=addHistoryTerms['nb'], plotting=plotTheBases
        )
    addHistoryTerms['endpoints'] = [historyEdge[0], historyEdge[0] + ensembleHistoryLen]
    if addHistoryTerms:
        ihbasisDF, orthobasisDF = makeLogRaisedCosBasis(**addHistoryTerms)
        iht = np.array(ihbasisDF.index)
        ihbasis = ihbasisDF.to_numpy()
        orthobas = orthobasisDF.to_numpy()
        # addHistoryTerms['nb'] = orthobas.shape[1]
        if plotTheBases:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(iht, ihbasis)
            ax[0].set_title('raised log cos basis')
            ax[1].plot(iht, orthobas)
            ax[1].set_title('orthogonalized log cos basis')
            ax[1].set_xlabel('Time (sec)')
            if arguments['debugging']:
                plt.show()
            else:
                plt.close()
        with pd.ExcelWriter(variantMetaDataPath, mode='a') as writer:
            ihbasisDF.to_excel(writer, sheet_name='ensembleBasis')
    # statistical model specific options
    # for statsmodels wrapper
    #  modelArguments = dict(
    #      modelKWargs={
    #          'sm_class': sm.GLM,
    #          'family': sm.families.Poisson(),
    #          'alpha': None, 'L1_wt': None,
    #          'refit': None,
    #          # 'alpha': 1e-3, 'L1_wt': .1,
    #          # 'refit': False,
    #          'maxiter': 250, 'disp': True
    #          },
    #      model=SMWrapper
    #  )
    #  gridParams = {
    #      'alpha': np.logspace(
    #          np.log(0.05), np.log(0.0001),
    #          3, base=np.exp(1))
    #      }
    modelArguments = dict(
        modelKWargs=dict(
            distr='softplus', alpha=0.25, reg_lambda=1e-4,
            fit_intercept=False,
            # verbose=arguments['debugging'],
            verbose=arguments['verbose'],
            solver='L-BFGS-B',
            # solver='cdfast',
            max_iter=100, tol=1e-5, learning_rate=2e-1,
            score_metric='pseudo_R2', track_convergence=True),
        model=pyglmnetWrapper
        )
    # if arguments['debugging']:
    #     modelArguments['modelKWargs']['max_iter'] = 20
    gridParams = [
        {
            'reg_lambda': np.round(
                np.logspace(
                    np.log(5e-2), np.log(1e-4),
                    7, base=np.exp(1)),
                decimals=5)}
        ]
    # elastic_net L1 weight, maxiter and regularization params
    # chosen based on Benjamin et al 2017
    if(RANK == 0):
        # find existing variants
        variantMetaDataPath = os.path.join(
            estimatorSubFolder, 'variant_metadata.pickle')
        varFolders = sorted(
            glob.glob(os.path.join(estimatorSubFolder, 'var_*')))
        existingVarSubFolders = [os.path.basename(i) for i in varFolders]
        if len(existingVarSubFolders):
            assert os.path.exists(variantMetaDataPath), 'Missing saved model info!'
            with open(variantMetaDataPath, 'rb') as f:
                allModelOpts = pickle.load(f)
        else:
            timeLags = [
                [i for i in cBasis.columns if i > 0],
                [i for i in cBasis.columns if i < 0],
                [i for i in cBasis.columns],
                [cBasis.columns[0]],
                [cBasis.columns[-1]],
                [cBasis.columns[int(covariateBasisTerms['nb']/2)]],
            ]
            stimSplineFuns = ['bs({}, df=3)', '{}']
            if arguments['verbose'] and (RANK == 0):
                print('timeLags = {}'.format(timeLags))
            allModelOpts = []
            # orderList = [False, True]
            # orderList = [True, False]
            for eht in [False, True]:
                for st in [True, False]:
                    for stdi in [False, True]:
                        for kt in [False, True]:
                            for acct in [False, True]:
                                if not any([eht, kt, st]):
                                    continue
                                if stdi and (not st):
                                    continue
                                if acct and (not kt):
                                    continue
                                if eht and (not stdi):
                                    continue
                                if eht and (not acct):
                                    continue
                                if not any([kt, st]):
                                    useTimeLags = [timeLags[-1]]
                                else:
                                    useTimeLags = [i for i in timeLags]
                                #
                                if st:
                                    allSplineFun = [i for i in stimSplineFuns]
                                else:
                                    allSplineFun = [stimSplineFuns[0]]
                                for tl in useTimeLags:
                                    for sp in allSplineFun:
                                        allModelOpts += [
                                            {
                                                'kinLag': tl,
                                                'stimLag': tl,
                                                'ensembleHistoryTerms': eht,
                                                'stimTerms': st,
                                                'angleTerms': kt,
                                                'velTerms': kt,
                                                'accTerms': acct,
                                                'stimSplineFun': sp,
                                                'addStimDiff': stdi,
                                                'addIntercept': True}
                                            ]
            with open(variantMetaDataPath, 'wb') as f:
                pickle.dump(
                    allModelOpts, f)
    else:
        allModelOpts = None
        existingVarSubFolders = None
    pdb.set_trace()
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        allModelOpts = COMM.bcast(allModelOpts, root=0)
        existingVarSubFolders = COMM.bcast(existingVarSubFolders, root=0)
        modelArguments['modelKWargs']['verbose'] = False
    # check if we need to recalculate saved features
    if (RANK == 0):
        if os.path.exists(featuresMetaDataPath):
            with open(featuresMetaDataPath, 'rb') as f:
                featuresMetaData = pickle.load(f)
            sameFeatures = (featuresMetaData['featureLoadArgs'] == featureLoadArgs)
            sameTargets = (featuresMetaData['targetLoadArgs'] == targetLoadArgs)
            if sameFeatures and sameTargets:
                reloadFeatures = False
            else:
                reloadFeatures = True
        else:
            reloadFeatures = True
            featuresMetaData = {
                'targetLoadArgs': targetLoadArgs,
                'featureLoadArgs': featureLoadArgs
            }
            if addHistoryTerms:
                featuresMetaData['ihbasis'] = ihbasis
                featuresMetaData['iht'] = iht
                featuresMetaData['addHistoryTerms'] = addHistoryTerms
    else:
        reloadFeatures = None
        featuresMetaData = None
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        reloadFeatures = COMM.bcast(reloadFeatures, root=0)
        featuresMetaData = COMM.bcast(featuresMetaData, root=0)
    if reloadFeatures and (RANK == 0):
        print('Loading {}'.format(regressorPath))
        regressorReader, regressorBlock = ns5.blockFromPath(
            regressorPath, lazy=arguments['lazy'])
        featuresDF = ns5.alignedAsigsToDF(
            regressorBlock, **featureLoadArgs)
        #
        featuresDF.columns = [
            '{}'.format(i[0]).replace('#0', '').replace('#', '_')
            for i in featuresDF.columns]
        # if addStimDiff:
        #     def pFun(x):
        #         return x.diff(axis='columns').fillna(0)
        #     diffLoadArgs = featureLoadArgs.copy()
        #     diffLoadArgs['procFun'] = pFun
        #     diffLoadArgs['unitNames'] = [p + '#0' for p in pList]
        #     if arguments['verbose'] and (RANK == 0):
        #         print('Calculating diff terms')
        #     diffDF = ns5.alignedAsigsToDF(
        #         regressorBlock, **diffLoadArgs)
        #     #
        #     diffDF.columns = [
        #         i[0].replace('amplitude#0', 'dAmpDt')
        #         for i in diffDF.columns]
        #     featuresDF = pd.concat(
        #         [featuresDF, diffDF], axis='columns')
        # sanity check diff terms
        # if arguments['plotting']:
        #     for _, g in featuresDF.query('program==0').groupby(['segment', 't']): break
        #     firstTrialIdx = g.index
        #     firstTrialT = (
        #         featuresDF.loc[firstTrialIdx, :].index
        #         .get_level_values('bin'))
        #     plt.plot(firstTrialT, diffDF.loc[firstTrialIdx, 'program0_dAmpDt'].to_numpy())
        #     plt.plot(firstTrialT, featuresDF.loc[firstTrialIdx, 'program0_amplitude'].to_numpy())
        #     if arguments['debugging']:
        #         plt.show()
        #     else:
        #         plt.close()
        print('Loading {}'.format(triggeredPath))
        #
        # if arguments['debugging']:
        #     yColDbList = ['elec10#0_raster#0', 'elec75#1_raster#0']
        #     alignedAsigsKWargs['unitNames'] = yColDbList
        #     alignedAsigsKWargs['unitQuery'] = None
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        targetDF = ns5.alignedAsigsToDF(
            dataBlock, **targetLoadArgs)
        targetDF.columns = [
            '{}'.format(i[0]).replace('_raster#0', '').replace('#', '_')
            for i in targetDF.columns]
        if addHistoryTerms:
            historyLoadArgs = targetLoadArgs.copy()
            historyTerms = []
            for colIdx in range(addHistoryTerms['nb']):
                if arguments['verbose'] and (RANK == 0):
                    print('Calculating history term {}'.format(colIdx + 1))
                thisBasis = ihbasis[:, colIdx]
                # thisBasis = orthobas[:, colIdx]
                def shiftConvolve(c):
                    origSize = c.size
                    halfSize = int(np.ceil(thisBasis.size/2))
                    fullConv = pd.Series(np.convolve(
                        c,
                        thisBasis, mode='full')).shift(halfSize).fillna(0)
                    return fullConv.iloc[halfSize:].iloc[:origSize].to_numpy()
                def pFun(x):
                    x.iloc[:, :] = np.apply_along_axis(
                        shiftConvolve, 1, x.to_numpy())
                    return x
                #  #  Test the convolution
                #  if True:
                #      testTarget = pd.Series((firstTrialT ** 0) - 1)
                #      testTarget = pd.Series(targetDF.loc[firstTrialIdx, 'elec10_0'])
                #      testTarget[firstTrialT == 0] = 1
                #      testConv = pd.Series(np.convolve(testTarget.to_numpy(), thisBasis, mode='same'))
                #      testConvShift = pd.Series(shiftConvolve(testTarget))
                #      plt.plot(firstTrialT, testTarget, label='target')
                #      plt.plot(firstTrialT, testConv, label='normal conv')
                #      plt.plot(firstTrialT, testConvShift, label='shifted conv')
                #      plt.legend()
                #      plt.show()
                historyLoadArgs['procFun'] = pFun
                historyDF = ns5.alignedAsigsToDF(
                    dataBlock, **historyLoadArgs)
                #
                historyDF.columns = [
                    '{}_cos_{}'.format(
                        i[0].replace('_raster#0', '').replace('#', '_'),
                        colIdx)
                    for i in historyDF.columns]
                if plotTheBases:
                    for _, g in featuresDF.query('program==0').groupby(['segment', 't']): break
                    firstTrialIdx = g.index
                    firstTrialT = (
                        featuresDF.loc[firstTrialIdx, :].index
                        .get_level_values('bin'))
                    plt.plot(
                        firstTrialT,
                        historyDF
                        .loc[firstTrialIdx, 'elec10_0_cos_{}'.format(colIdx)]
                        .to_numpy())
                historyTerms.append(historyDF.copy())
                # if arguments['debugging']:
                #     addHistoryTerms['nb'] = 1
                #     break
            # sanity check diff terms
            if plotTheBases:
                plt.plot(
                    firstTrialT,
                    targetDF.loc[firstTrialIdx, 'elec10_0'].to_numpy())
                if arguments['debugging']:
                    plt.show()
                else:
                    plt.close()
            featuresDF = pd.concat(
                [featuresDF] + historyTerms, axis='columns')
        dropMask = np.logical_or.reduce([
            targetDF.isna().T.any(),
            featuresDF.isna().T.any()])
        dropIndex = targetDF.index[dropMask]
        targetDF.drop(index=dropIndex, inplace=True)
        featuresDF.drop(index=dropIndex, inplace=True)
        #  Standardize units (count/bin)
        targetDF = targetDF * countPerBin
        # columnInfo = (
        #     featuresDF
        #     .columns.to_frame()
        #     .reset_index(drop=True))
        ############################################################
        # scale each feature independently
        ############################################################
        targetScalers = []
        featureScalers = []
        for featName in featuresDF.columns:
            if ('amplitude' in featName) or ('Hz' in featName) or ('dAmpDt' in featName):
                featureScalers.append((MaxAbsScaler(), [featName]))
            else:
                featureScalers.append((StandardScaler(), [featName]))
        if addHistoryTerms:
            for hTBaseName in targetDF.columns:
                matchingNames = [
                    i
                    for i in featuresDF.columns
                    if hTBaseName in i
                    ]
                if len(matchingNames):
                    featureScalers.append((StandardScaler(), matchingNames))
        ############################################################
        # featuresDF[('program0_dAmpDt', 0)].abs().max()
        featuresDF = applyScalersGroupedNoLag(featuresDF, featureScalers)
        targetDF = applyScalersGroupedNoLag(targetDF, targetScalers) 
        #
        featuresMetaData['featureScalers'] = featureScalers
        featuresMetaData['targetScalers'] = targetScalers
        with open(featuresMetaDataPath, 'wb') as f:
            pickle.dump(featuresMetaData, f)
        #
        featuresDF.to_hdf(regressorH5Path, 'feature')
        targetDF.to_hdf(targetH5Path, 'target')
        if arguments['plotting']:
            for _, g in featuresDF.query('program==0').groupby(['segment', 't']): break
            firstTrialIdx = g.index
            firstTrialT = (
                featuresDF.loc[firstTrialIdx, :].index
                .get_level_values('bin'))
            pdfPath = os.path.join(GLMFiguresFolder, 'check_features.pdf')
            plt.plot(
                firstTrialT,
                targetDF.loc[firstTrialIdx, 'elec10_0'].to_numpy(),
                label='target')
            if addHistoryTerms:
                for colIdx in range(addHistoryTerms['nb']):
                    cosBaseLabel = 'elec10_0_cos_{}'.format(colIdx)
                    plt.plot(
                        firstTrialT,
                        featuresDF
                        .loc[firstTrialIdx, cosBaseLabel]
                        .to_numpy(), label=cosBaseLabel)
            plt.plot(
                firstTrialT,
                featuresDF.loc[firstTrialIdx, 'program0_dAmpDt'].to_numpy(),
                label='program0_dAmpDt')
            plt.plot(
                firstTrialT,
                featuresDF.loc[firstTrialIdx, 'program0_amplitude'].to_numpy(),
                label='program0_amplitude')
            plt.legend()
            plt.savefig(
                pdfPath,
                bbox_extra_artists=(plt.gca().get_legend(),),
                bbox_inches='tight')
            if arguments['debugging']:
                plt.show()
            else:
                plt.close()
    if HAS_MPI:
        # wait for RANK 0 to create the h5 files
        COMM.Barrier()
    if not (reloadFeatures and (RANK == 0)):
        featuresDF = pd.read_hdf(regressorH5Path, 'feature')
        targetDF = pd.read_hdf(targetH5Path, 'target')
    if (RANK == 0):
        prelimCV = trialAwareStratifiedKFold(**cv_kwargs).split(featuresDF)
        (train, test) = prelimCV[0]
        cv_kwargs['n_splits'] -= 1
        cv = trialAwareStratifiedKFold(**cv_kwargs)
    else:
        train = None
        test = None
        cv = None
    if HAS_MPI:
        # wait for RANK 0 to create the h5 files
        COMM.Barrier()
        train = COMM.bcast(train, root=0)
        test = COMM.bcast(test, root=0)
        cv = COMM.bcast(cv, root=0)
    #
    yTrain = targetDF.iloc[train, :].copy()
    yTest = targetDF.iloc[test, :].copy()
    cv_folds = cv.split(yTrain)
    #
    if HAS_MPI:
        # !! OverflowError: cannot serialize a bytes object larger than 4 GiB
        COMM.Barrier()  # sync MPI threads, wait for 0
        # subselect this RANK's units
        yColIdx = range(RANK, yTrain.shape[1], SIZE)
        print(
            'Process {}, running regression for units {}'
            .format(RANK, yColIdx))
        yTrain = yTrain.iloc[:, yColIdx].copy()
        yTest = yTest.iloc[:, yColIdx].copy()
    #
    # if arguments['debugging']:
    #     yColDbList = ['elec10_0', 'elec75_1']
    #     yTrain = yTrain.loc[:, yColDbList].copy()
    #     yTest = yTest.loc[:, yColDbList].copy()
    #
    del targetDF
    #
    for modelIdx, modelOpts in enumerate(allModelOpts):
        varBaseName = 'var_{:03d}'.format(modelIdx)
        variantSubFolder = os.path.join(
            estimatorSubFolder,
            varBaseName
            )
        if len(existingVarSubFolders):
            if varBaseName in existingVarSubFolders:
                continue
        if arguments['verbose'] and (RANK == 0):
            print('Running model variant {}'.format(varBaseName))
        #
        estimatorPath = os.path.join(
            variantSubFolder, 'estimator.joblib')
        estimatorMetadataPath = os.path.join(
            variantSubFolder, 'estimator_metadata.pickle')
        if HAS_MPI:
            estimatorPath = estimatorPath.replace(
                'estimator', 'estimator_{:03d}'.format(RANK))
        #
        if RANK == 0:
            estimatorMetadata = {
                'targetPath': targetH5Path,
                'regressorPath': regressorH5Path,
                'path': os.path.basename(estimatorPath),
                'name': arguments['estimatorName'],
                'trainIdx': train,
                'testIdx': test,
                'alignedAsigsKWargs': alignedAsigsKWargs,
                'cBasis': cBasis,
                }
        #
        xTrainList = []
        xTestList = []
        if modelOpts['addIntercept']:
            xTrainList.append(dmatrix(
                ModelDesc([], [Term([])]),
                featuresDF.iloc[train, :],
                return_type='dataframe'))
            xTestList.append(dmatrix(
                ModelDesc([], [Term([])]),
                featuresDF.iloc[test, :],
                return_type='dataframe'))
        if modelOpts['angleTerms']:
            kinDescList = []
            kinDescList += [
                Term([LookupFactor('{}'.format(k))])
                for k in angleList
                ]
            if modelOpts['velTerms']:
                kinDescList += [
                    Term([LookupFactor('{}'.format(k))])
                    for k in velList
                    ]
            if modelOpts['accTerms']:
                kinDescList += [
                    Term([LookupFactor('{}'.format(k))])
                    for k in accList
                    ]
            #
            kinModelDesc = ModelDesc([], kinDescList)
            # TODO ensure this works
            if RANK == 0:
                estimatorMetadata['kinModelDesc'] = kinModelDesc.describe()
            kinX = dmatrix(
                kinModelDesc,
                featuresDF,
                return_type='dataframe')
            kinDescListFinal = [
                Term([EvalFactor('applyCBasis(Q("{}"), {:.3f})'.format(k.name(), l))])
                for k, l in product(kinDescList, modelOpts['kinLag'])
                ]
            xTrainList.append(dmatrix(
                ModelDesc([], kinDescListFinal),
                kinX.iloc[train, :],
                return_type='dataframe'))
            xTestList.append(dmatrix(
                ModelDesc([], kinDescListFinal),
                kinX.iloc[test, :],
                return_type='dataframe'))
        if modelOpts['stimTerms']:
            def f1(p):
                return EvalFactor('bs({}, df=3)'.format(p))
            def f2(p):
                return LookupFactor('{}'.format(p))
            if 'bs' in modelOpts['stimSplineFun']:
                stimSplineFun = lambda p: EvalFactor(modelOpts['stimSplineFun'].format(p))
            else:
                stimSplineFun = lambda p: LookupFactor(modelOpts['stimSplineFun'].format(p))
            stimDescList = []
            stimDescList += [
                Term([stimSplineFun(p)])
                for p in pList
                ]
            stimDescList += [
                Term([stimSplineFun(p), LookupFactor('RateInHz')])
                for p in pList
                ]
            stimModelDesc = ModelDesc([], stimDescList)
            # TODO ensure saving the modelDesc works
            if RANK == 0:
                estimatorMetadata['stimModelDesc'] = stimModelDesc.describe()
            stimXTrain = dmatrix(
                stimModelDesc,
                featuresDF.iloc[train, :],
                return_type='dataframe')
            stimXTest = build_design_matrices(
                [stimXTrain.design_info], featuresDF.iloc[test, :],
                return_type='dataframe')[0]
            stimDescListFinal = [
                Term([EvalFactor('applyCBasis(Q("{}"), {:.3f})'.format(p, l))])
                for p, l in product(stimXTrain.columns, modelOpts['stimLag'])
                ]
            if modelOpts['addStimDiff']:
                deriv = lambda x: pd.Series(x).diff().fillna(0).to_numpy()
                stimDescListFinal += [
                    Term([EvalFactor('deriv(applyCBasis(Q("{}"), {:.3f}))'.format(p, l))])
                    for p, l in product(stimXTrain.columns, modelOpts['stimLag'])
                    ]
            stimXTrainFinal = dmatrix(
                ModelDesc([], stimDescListFinal),
                stimXTrain,
                return_type='dataframe')
            xTrainList.append(stimXTrainFinal)
            stimXTestFinal = dmatrix(
                ModelDesc([], stimDescListFinal),
                stimXTest,
                return_type='dataframe')
            xTestList.append(stimXTestFinal)
            
        if modelOpts['ensembleHistoryTerms']:
            histDescList = [
                Term([LookupFactor(histTerm)])
                for histTerm in featuresDF.columns if '_cos' in histTerm
                ]
            #
            xTrainList.append(dmatrix(
                ModelDesc([], histDescList),
                featuresDF.iloc[train, :],
                return_type='dataframe'))
            xTestList.append(dmatrix(
                ModelDesc([], histDescList),
                featuresDF.iloc[test, :],
                return_type='dataframe'))
        #
        xTrain = pd.concat(xTrainList, axis='columns')
        xTest = pd.concat(xTestList, axis='columns')
        #
        if RANK == 0:
            estimatorMetadata.update({
                'inputFeatures': yTrain.columns.to_list(),
                'modelFeatures': xTest.columns.to_list()
                })
            if not os.path.exists(variantSubFolder):
                os.makedirs(variantSubFolder, exist_ok=True)
            if not arguments['debugging']:
                with open(estimatorMetadataPath, 'wb') as f:
                    pickle.dump(
                        estimatorMetadata, f)
        #
        if arguments['plotting'] and (RANK == 0) and (not HAS_MPI):
            with sns.plotting_context(context='notebook', font_scale=.75):
                # plotDF = stimXTestFinal
                # plotDF = stimXTest
                plotDF = xTest
                fig, ax = plt.subplots(9, 1, sharex=True, figsize=(20, 15))
                for _, g in plotDF.query('program==0').groupby(['segment', 't']): break
                firstTrialIdx = g.index
                firstTrialT = (
                    plotDF.loc[firstTrialIdx, :].index
                    .get_level_values('bin'))
                pdfPath = os.path.join(estimatorFiguresFolder, 'check_xy_test_{}.pdf'.format(varBaseName))
                ax[0].plot(
                    firstTrialT,
                    yTest.loc[firstTrialIdx, 'elec10_0'].to_numpy(),
                    label='target')
                if modelOpts['stimTerms']:
                    maNames = [
                        i
                        for i in plotDF.columns
                        if ('RateInHz' in i) and not ('deriv' in i) and ('program0' in i)]
                    for sta1 in maNames:
                        ax[1].plot(
                            firstTrialT,
                            plotDF.loc[firstTrialIdx, sta1].to_numpy(),
                            label=sta1)
                    mdNames = [
                        i
                        for i in plotDF.columns
                        if ('RateInHz' in i) and ('deriv' in i) and ('program0' in i)]
                    for sta2 in mdNames:
                        ax[2].plot(
                            firstTrialT,
                            plotDF.loc[firstTrialIdx, sta2].to_numpy(),
                            label=sta2)
                    saNames = [
                        i
                        for i in plotDF.columns
                        if not ('RateInHz' in i) and not ('deriv' in i) and ('program0' in i)]
                    for sta3 in saNames:
                        ax[3].plot(
                            firstTrialT,
                            plotDF.loc[firstTrialIdx, sta3].to_numpy(),
                            label=sta3)
                    sdNames = [
                        i
                        for i in plotDF.columns
                        if not ('RateInHz' in i) and ('deriv' in i) and ('program0' in i)]
                    for sta4 in sdNames:
                        ax[4].plot(
                            firstTrialT,
                            plotDF.loc[firstTrialIdx, sta4].to_numpy(),
                            label=sta4)
                if modelOpts['angleTerms']:
                    ktNames = [
                        i
                        for i in xTest.columns
                        if ('_angle' in i)]
                    for sta1 in ktNames:
                        ax[5].plot(
                            firstTrialT,
                            xTest.loc[firstTrialIdx, sta1].to_numpy(),
                            label=sta1)
                    ktNames = [
                        i
                        for i in xTest.columns
                        if ('_angular_velocity' in i)]
                    for sta1 in ktNames:
                        ax[6].plot(
                            firstTrialT,
                            xTest.loc[firstTrialIdx, sta1].to_numpy(),
                            label=sta1)
                    ktNames = [
                        i
                        for i in xTest.columns
                        if ('_angular_acceleration' in i)]
                    for sta1 in ktNames:
                        ax[7].plot(
                            firstTrialT,
                            xTest.loc[firstTrialIdx, sta1].to_numpy(),
                            label=sta1)
                if modelOpts['ensembleHistoryTerms']:
                    for colIdx in range(addHistoryTerms['nb']):
                        cosBaseLabel = 'elec10_0_cos_{}'.format(colIdx)
                        ax[8].plot(
                            firstTrialT,
                            xTest
                            .loc[firstTrialIdx, cosBaseLabel]
                            .to_numpy(), label=cosBaseLabel)
                for thisAx in ax:
                    thisAx.legend()
                figSaveOpts = dict(
                    bbox_extra_artists=(thisAx.get_legend() for thisAx in ax),
                    bbox_inches='tight')
                plt.savefig(pdfPath, **figSaveOpts)
                if arguments['debugging']:
                    plt.show()
                else:
                    plt.close()
        #
        snr = SingleNeuronRegression(
            xTrain=xTrain, yTrain=yTrain,
            xTest=xTest, yTest=yTest,
            cv_folds=cv_folds,
            **modelArguments,
            verbose=arguments['verbose'],
            plotting=arguments['plotting']
            )
        #
        if not arguments['dryRun']:
            snr.apply_gridSearchCV(gridParams)
        else:
            pass
        # tolFindParams = {
        #     'max_iter': 5000,
        #     'tol': 1e-6}
        # snr.dispatchParams(tolFindParams)
        if arguments['debugging']:
            from ttictoc import tic, toc
            tic()
        #
        if not arguments['dryRun']:
            snr.fit()
        #
        if arguments['debugging']:
            toc()
            pdb.set_trace()
        #
        if not arguments['debugging']:
            snr.clear_data()
        if (not arguments['debugging']) and (not arguments['dryRun']):
            jb.dump(snr, estimatorPath)
        if HAS_MPI:
            COMM.Barrier()  # sync MPI threads, wait for 0
    return


if __name__ == "__main__":
    runProfiler = False
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        if arguments['lazy']:
            nameSuffix = 'lazy'
        else:
            nameSuffix = 'not_lazy'
        prf.profileFunction(
            topFun=calcUnitRegressionToAsig,
            modulesToProfile=[ash, ns5, TargetedDimensionalityReduction],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        calcUnitRegressionToAsig()
