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
from collections.abc import Iterable
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
#
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if arguments['debugging']:
    matplotlib.use('Qt5Agg')   # interactive
else:
    matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")
#
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
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName']
        )
    if arguments['processAll']:
        prefix = assembledName
    else:
        prefix = ns5FileName
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
    #
    subsampleOpts = glmOptsLookup[arguments['estimatorName']]['subsampleOpts']
    ensembleHistoryLen = glmOptsLookup['ensembleHistoryLen']
    covariateHistoryLen = glmOptsLookup['covariateHistoryLen']
    winSize = subsampleOpts['rollingWindow'] if subsampleOpts['rollingWindow'] is not None else 1
    halfWinSize = int(np.ceil(winSize/2))
    countPerBin = rasterOpts['binInterval'] * winSize
    binSize = rasterOpts['binInterval'] * subsampleOpts['decimate']
    sourceOpts = [
        # {
        #     'experimentName': '201901271000-Proprio',
        #     'analysisName': 'default',
        #     'alignFolderName': 'stim',
        #     'prefix': 'Block005',
        #     'alignQuery': 'stimOn'
        #     },
        {
            'experimentName': '201901271000-Proprio',
            'analysisName': 'default',
            'alignFolderName': 'motion',
            'prefix': '',
            'alignQuery': 'midPeak'
            }
        ]
    alignedAsigsKWargs.update(dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False,
        transposeToColumns='feature', concatOn='columns',
        removeFuzzyName=False,
        getMetaData=[
            'RateInHz', 'activeGroup', 'amplitude', 'amplitudeCat',
            'bin', 'electrode', 'pedalDirection', 'pedalMetaCat',
            'pedalMovementCat', 'stimCat', 'pedalMovementDuration',
            'pedalSize', 'pedalSizeCat', 'pedalVelocityCat',
            'program', 'segment', 't'],
        **subsampleOpts,
        metaDataToCategories=False,
        verbose=False, procFun=None))
    #
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
        namedQueries, **arguments)
    uN, uQ = ash.processUnitQueryArgs(
        namedQueries, alignSubFolder, **arguments)
    alignedAsigsKWargs['unitNames'] = uN
    alignedAsigsKWargs['unitQuery'] = uQ
    #
    alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
        alignSubFolder, prefix, **arguments)
    #
    featuresMetaDataPath = os.path.join(
        estimatorSubFolder, 'features_meta.pickle')
    targetH5Path = os.path.join(
        estimatorSubFolder, prefix + '_{}_{}.h5'.format(
            arguments['inputBlockName'], arguments['window']))
    regressorH5Path = os.path.join(
        estimatorSubFolder, prefix + '_{}_{}.h5'.format(
            arguments['secondaryBlockName'], arguments['window']))
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
        stratifyFactors=['RateInHz', 'program', 'amplitude'],
        continuousFactors=['segment', 'originalIndex'])
    #
    arguments['plotting'] = arguments['plotting'] and (not HAS_MPI)
    if(RANK == 0):
        # find existing variants
        variantMetaDataPath = os.path.join(
            estimatorSubFolder, 'variant_metadata.xlsx')
        if not os.path.exists(variantMetaDataPath):
            covariateBasisTerms = {
                # 'nb': glmOptsLookup['nCovariateBasisTerms'],
                'spacing': glmOptsLookup[arguments['estimatorName']]['covariateSpacing'],
                'dt': rasterOpts['binInterval'],
                'endpoints': [-covariateHistoryLen, covariateHistoryLen]
                }
            cBasis = makeRaisedCosBasis(**covariateBasisTerms)
            if arguments['plotting']:
                pdfPath = os.path.join(
                    estimatorFiguresFolder, 'covariate_basis.pdf')
                print('saving {}'.format(pdfPath))
                fig, ax = plt.subplots(2, 1, sharex=True)
                ax[0].plot(cBasis)
                ax[0].set_xlabel('Time (sec)')
                ax[0].set_ylabel('(a.u.)')
                ax[1].set_title('raised cos basis')
            cBasis = (
                cBasis.rolling(winSize, center=True, win_type='gaussian')
                .mean(std=halfWinSize).iloc[::subsampleOpts['decimate'], ::2]
                .fillna(0, axis=0).fillna(0, axis=1))
            cBasis.to_excel(variantMetaDataPath, sheet_name='covariateBasis')
            #
            if arguments['plotting']:
                ax[1].plot(cBasis)
                ax[1].set_xlabel('Time (sec)')
                ax[1].set_ylabel('(a.u.)')
                ax[1].set_title('downsampled raised cos basis')
                plt.savefig(pdfPath)
                plt.savefig(pdfPath.replace('.pdf', '.png'))
                if arguments['debugging']:
                    plt.show()
                else:
                    plt.close()
            addHistoryTerms = {
                'nb': glmOptsLookup['nHistoryBasisTerms'],
                'dt': rasterOpts['binInterval'],
                'b': 0.001,
                'zflag': False
                }
            # make sure history terms don't bleed into current bins
            historyEdge = raisedCosBoundary(
                b=addHistoryTerms['b'], DT=ensembleHistoryLen,
                minX=max(rasterOpts['binInterval'] * winSize / 2, 1e-3),
                nb=addHistoryTerms['nb'], plotting=arguments['debugging']
                )
            addHistoryTerms['endpoints'] = [
                historyEdge[0], historyEdge[0] + ensembleHistoryLen]
            if addHistoryTerms:
                ihbasisDF, orthobasisDF = makeLogRaisedCosBasis(
                    **addHistoryTerms)
                iht = np.array(ihbasisDF.index)
                ihbasis = ihbasisDF.to_numpy()
                orthobas = orthobasisDF.to_numpy()
                # addHistoryTerms['nb'] = orthobas.shape[1]
                if arguments['plotting']:
                    pdfPath = os.path.join(
                        estimatorFiguresFolder, 'history_basis.pdf'
                        )
                    fig, ax = plt.subplots(2, 1, sharex=True)
                    ax[0].plot(iht, ihbasis)
                    ax[0].set_title('raised log cos basis')
                    ax[1].plot(iht, orthobas)
                    ax[1].set_title('orthogonalized log cos basis')
                    ax[1].set_xlabel('Time (sec)')
                    plt.savefig(pdfPath)
                    plt.savefig(pdfPath.replace('.pdf', '.png'))
                    if arguments['debugging']:
                        plt.show()
                    else:
                        plt.close()
                with pd.ExcelWriter(variantMetaDataPath, mode='a') as writer:
                    ihbasisDF.to_excel(writer, sheet_name='ensembleBasis')
                raise(Exception('Please specify variants now.'))
        else:
            cBasis = pd.read_excel(
                variantMetaDataPath, sheet_name='covariateBasis', index_col=0)
            ihbasisDF = pd.read_excel(
                variantMetaDataPath, sheet_name='ensembleBasis', index_col=0)
            iht = np.array(ihbasisDF.index)
            ihbasis = ihbasisDF.to_numpy()
    else:
        cBasis = None
        iht = None
        ihbasis = None
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        cBasis = COMM.bcast(cBasis, root=0)
        iht = COMM.bcast(iht, root=0)
        ihbasis = COMM.bcast(ihbasis, root=0)

    def applyCBasis(vec, cbCol):
        return np.convolve(
            vec.to_numpy(),
            cBasis[cbCol].to_numpy(),
            mode='same')
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
        model=pyglmnetWrapper,
        )
    # if arguments['debugging']:
    #     modelArguments['modelKWargs']['max_iter'] = 20
    gridParams = [
        {
            'reg_lambda': np.round(
                np.logspace(
                    np.log(5e-2), np.log(1e-5),
                    7, base=np.exp(1)),
                decimals=5)}
        ]
    # elastic_net L1 weight, maxiter and regularization params
    # chosen based on Benjamin et al 2017
    if(RANK == 0):
        # find existing variants
        varFolders = sorted(
            glob.glob(os.path.join(estimatorSubFolder, 'var_*')))
        existingVarSubFolders = [os.path.basename(i) for i in varFolders]
        varMetaData = pd.read_excel(variantMetaDataPath, sheet_name='variantInfo')
    else:
        varMetaData = None
        existingVarSubFolders = None
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        varMetaData = COMM.bcast(varMetaData, root=0)
        existingVarSubFolders = COMM.bcast(existingVarSubFolders, root=0)
        modelArguments['modelKWargs']['verbose'] = False
    # check if we need to recalculate saved features
    if (RANK == 0):
        if os.path.exists(featuresMetaDataPath):
            with open(featuresMetaDataPath, 'rb') as f:
                featuresMetaData = pickle.load(f)
            sameFeatures = True
            for k, v in featureLoadArgs.items():
                if isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                    if not all(featuresMetaData['featureLoadArgs'][k] == v):
                        sameFeatures = False
                else:
                    if not (featuresMetaData['featureLoadArgs'][k] == v):
                        sameFeatures = False
            sameTargets = True
            for k, v in targetLoadArgs.items():
                if isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                    if not all(featuresMetaData['targetLoadArgs'][k] == v):
                        sameTargets = False
                else:
                    if not (featuresMetaData['targetLoadArgs'][k] == v):
                        sameTargets = False
            # sameFeatures = (featuresMetaData['featureLoadArgs'] == featureLoadArgs)
            # sameTargets = (featuresMetaData['targetLoadArgs'] == targetLoadArgs)
            sameSources = (featuresMetaData['sourceOpts'] == sourceOpts)
            if sameFeatures and sameTargets and sameSources:
                reloadFeatures = False
            else:
                reloadFeatures = True
        else:
            reloadFeatures = True
            featuresMetaData = {
                'targetLoadArgs': targetLoadArgs,
                'featureLoadArgs': featureLoadArgs,
                'sourceOpts': sourceOpts
            }
            featuresMetaData['ihbasis'] = ihbasis
            featuresMetaData['iht'] = iht
    else:
        reloadFeatures = None
        featuresMetaData = None
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        reloadFeatures = COMM.bcast(reloadFeatures, root=0)
        featuresMetaData = COMM.bcast(featuresMetaData, root=0)
    if reloadFeatures and (RANK == 0):
        featuresDFList = []
        targetDFList = []
        for sourceOpt in sourceOpts:
            sourceScratchFolder = os.path.join(
                scratchPath, sourceOpt['experimentName'])
            sourceAnalysisSubFolder = os.path.join(
                sourceScratchFolder, sourceOpt['analysisName']
                )
            sourceAlignSubFolder = os.path.join(
                sourceAnalysisSubFolder, sourceOpt['alignFolderName']
                )
            targetLoadArgs['dataQuery'] = ash.processAlignQueryArgs(
                namedQueries, alignQuery=sourceOpt['alignQuery'])
            targetLoadArgs['outlierTrials'] = ash.processOutlierTrials(
                sourceAlignSubFolder, sourceOpt['prefix'], **arguments)
            featureLoadArgs['dataQuery'] = ash.processAlignQueryArgs(
                namedQueries, alignQuery=sourceOpt['alignQuery'])
            featureLoadArgs['outlierTrials'] = ash.processOutlierTrials(
                sourceAlignSubFolder, sourceOpt['prefix'], **arguments)
            triggeredPath = os.path.join(
                sourceAlignSubFolder,
                sourceOpt['prefix'] + '_{}_{}.nix'.format(
                    arguments['inputBlockName'], arguments['window']))
            regressorPath = os.path.join(
                sourceAlignSubFolder,
                sourceOpt['prefix'] + '_{}_{}.nix'.format(
                    arguments['secondaryBlockName'], arguments['window']))
            print('Loading {}'.format(regressorPath))
            regressorReader, regressorBlock = ns5.blockFromPath(
                regressorPath, lazy=arguments['lazy'])
            partialFeaturesDF = ns5.alignedAsigsToDF(
                regressorBlock, **featureLoadArgs)
            #
            partialFeaturesDF.columns = [
                '{}'.format(i[0]).replace('#0', '').replace('#', '_')
                for i in partialFeaturesDF.columns]
            print('Loading {}'.format(triggeredPath))
            dataReader, dataBlock = ns5.blockFromPath(
                triggeredPath, lazy=arguments['lazy'])
            partialTargetDF = ns5.alignedAsigsToDF(
                dataBlock, **targetLoadArgs)
            partialTargetDF.columns = [
                '{}'.format(i[0]).replace('_raster#0', '').replace('#', '_')
                for i in partialTargetDF.columns]
            #
            historyLoadArgs = targetLoadArgs.copy()
            historyTerms = []
            for colIdx in range(ihbasis.shape[1]):
                if arguments['verbose'] and (RANK == 0):
                    print('Calculating history term {}'.format(colIdx + 1))
                thisBasis = ihbasis[:, colIdx]
                # thisBasis = orthobas[:, colIdx]
                def pFun(x, st=None):
                    x.iloc[:, :] = np.apply_along_axis(
                        np.convolve, 1, x.to_numpy(),
                        thisBasis, mode='same')
                    return x
                historyLoadArgs['procFun'] = pFun
                historyDF = ns5.alignedAsigsToDF(
                    dataBlock, **historyLoadArgs)
                #
                historyDF.columns = [
                    '{}_cos_{}'.format(
                        i[0].replace('_raster#0', '').replace('#', '_'),
                        colIdx)
                    for i in historyDF.columns]
                if arguments['plotting']:
                    for _, g in partialFeaturesDF.query('program==0').groupby(['segment', 't']): break
                    firstBlockIdx = g.index
                    firstBlockT = (
                        partialFeaturesDF.loc[firstBlockIdx, :].index
                        .get_level_values('bin'))
                    plt.plot(
                        firstBlockT,
                        historyDF
                        .loc[firstBlockIdx, 'elec10_0_cos_{}'.format(colIdx)]
                        .to_numpy())
                historyTerms.append(historyDF.copy())
                # if debugging, only calculate first history basis term
                if arguments['debugging']:
                    break
            # sanity check diff terms
            if arguments['plotting']:
                plt.plot(
                    firstBlockT,
                    partialTargetDF.loc[firstBlockIdx, 'elec10_0'].to_numpy())
                if arguments['debugging']:
                    plt.show()
                else:
                    plt.close()
            partialFeaturesDF = pd.concat(
                [partialFeaturesDF] + historyTerms, axis='columns')
            dropMask = np.logical_or.reduce([
                partialTargetDF.isna().T.any(),
                partialFeaturesDF.isna().T.any()])
            dropIndex = partialTargetDF.index[dropMask]
            partialTargetDF.drop(index=dropIndex, inplace=True)
            partialFeaturesDF.drop(index=dropIndex, inplace=True)
            #  Standardize units (count/bin)
            partialTargetDF = partialTargetDF * countPerBin
            featuresDFList.append(partialFeaturesDF)
            targetDFList.append(partialTargetDF)
        # pdb.set_trace()
        featuresDF = pd.concat(featuresDFList)
        targetDF = pd.concat(targetDFList)
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
        try:
            for hTBaseName in targetDF.columns:
                matchingNames = [
                    i
                    for i in featuresDF.columns
                    if hTBaseName in i
                    ]
                if len(matchingNames):
                    featureScalers.append((StandardScaler(), matchingNames))
        except Exception:
            traceback.print_exc()
            raise(Exception('fix me!'))
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
    if HAS_MPI:
        # wait for RANK 0 to create the h5 files
        COMM.Barrier()
    if not (reloadFeatures and (RANK == 0)):
        featuresDF = pd.read_hdf(regressorH5Path, 'feature')
        targetDF = pd.read_hdf(targetH5Path, 'target')
    if (RANK == 0):
        # restrict features to relevant portions depending on trial duration
        pdb.set_trace()
        #
        prelimCV = trialAwareStratifiedKFold(**cv_kwargs).split(featuresDF)
        (train, test) = prelimCV[0]
        cv_kwargs['n_splits'] -= 1
        cv = trialAwareStratifiedKFold(**cv_kwargs)
        cv_folds = cv.split(targetDF.iloc[train, :])
    else:
        train = None
        test = None
        cv = None
        cv_folds = None
    if HAS_MPI:
        # wait for RANK 0 to create the h5 files
        COMM.Barrier()
        train = COMM.bcast(train, root=0)
        test = COMM.bcast(test, root=0)
        cv = COMM.bcast(cv, root=0)
        cv_folds = COMM.bcast(cv_folds, root=0)
    #
    # yTrain = targetDF.iloc[train, :].copy()
    # yTest = targetDF.iloc[test, :].copy()
    # #
    # if HAS_MPI:
    #     # !! OverflowError: cannot serialize a bytes object larger than 4 GiB
    #     COMM.Barrier()  # sync MPI threads, wait for 0
    #     # subselect this RANK's units
    # yColIdx = range(RANK, yTrain.shape[1], SIZE)
    print(
        'Process {}, running regression for units {}'
        .format(RANK, yColIdx))
    yTrain = yTrain.iloc[train, yColIdx].copy()
    yTest = yTest.iloc[test, yColIdx].copy()
    #
    # if arguments['debugging']:
    #     yColDbList = ['elec10_0', 'elec75_1']
    #     yTrain = yTrain.loc[:, yColDbList].copy()
    #     yTest = yTest.loc[:, yColDbList].copy()
    del targetDF
    #
    for modelIdx, modelOpts in varMetaData.iterrows():
        varBaseName = 'var_{:03d}'.format(modelIdx)
        #
        if len(existingVarSubFolders):
            if varBaseName in existingVarSubFolders:
                print(' variant {} already exists! skipping...'.format(varBaseName))
                continue
        variantSubFolder = os.path.join(
            estimatorSubFolder,
            varBaseName
            )
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
                'cv_folds': cv_folds,
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
        #
        kinDescList = []
        if modelOpts['angle']:
            kinDescList += [
                Term([LookupFactor('{}'.format(k))])
                for k in angleList
                ]
        if modelOpts['angularVelocity']:
            kinDescList += [
                Term([LookupFactor('{}'.format(k))])
                for k in velList
                ]
        if modelOpts['angularAcceleration']:
            kinDescList += [
                Term([LookupFactor('{}'.format(k))])
                for k in accList
                ]
        if len(kinDescList):
            kinModelDesc = ModelDesc([], kinDescList)
            if RANK == 0:
                estimatorMetadata['kinModelDesc'] = kinModelDesc.describe()
            kinX = dmatrix(
                kinModelDesc,
                featuresDF,
                return_type='dataframe')
            kinLags = [
                np.float(i)
                for i in cBasis.columns
                if (
                    (np.float(i) >= modelOpts['minKinLag']) and
                    (np.float(i) <= modelOpts['maxKinLag']))]
            kinDescListFinal = [
                Term([EvalFactor('applyCBasis(Q("{}"), {:.3f})'.format(k.name(), l))])
                for k, l in product(kinDescList, kinLags)
                ]
            kinModelDescFinal = ModelDesc([], kinDescListFinal)
            if RANK == 0:
                estimatorMetadata['kinModelDescFinal'] = (
                    kinModelDescFinal.describe())
            kinXTrain = dmatrix(
                kinModelDescFinal,
                kinX.iloc[train, :],
                return_type='dataframe')
            kinXTest = build_design_matrices(
                [kinXTrain.design_info], featuresDF.iloc[test, :],
                return_type='dataframe')[0]
            xTrainList.append(kinXTrain)
            xTestList.append(kinXTest)
        stimDescList = []
        #
        presentPrograms = [p for p in pList if modelOpts[p]]
        if len(presentPrograms):
            if modelOpts['separateRateTerm']:
                stimDescList += [Term([LookupFactor('RateInHz')])]
            stimDescList += [
                Term([eval(modelOpts['stimAmpSpline'].format(p))])
                for p in presentPrograms
                ]
            if modelOpts['rateInteraction']:
                stimDescList += [
                    Term([
                        eval(modelOpts['stimAmpSpline'].format(p)),
                        LookupFactor('RateInHz')])
                    for p in presentPrograms
                    ]
            stimModelDesc = ModelDesc([], stimDescList)
            if RANK == 0:
                estimatorMetadata['stimModelDesc'] = stimModelDesc.describe()
            stimXTrain = dmatrix(
                stimModelDesc,
                featuresDF.iloc[train, :],
                return_type='dataframe')
            stimXTest = build_design_matrices(
                [stimXTrain.design_info], featuresDF.iloc[test, :],
                return_type='dataframe')[0]
            stimLags = [
                np.float(i)
                for i in cBasis.columns
                if (
                    (np.float(i) >= modelOpts['minStimLag']) and
                    (np.float(i) <= modelOpts['maxStimLag']))]
            stimDescListFinal = []
            if modelOpts['stimAmplitude']:
                stimDescListFinal += [
                    Term([EvalFactor('applyCBasis(Q("{}"), {:.3f})'.format(p, l))])
                    for p, l in product(stimXTrain.columns, stimLags)
                    ]
            if modelOpts['stimVelocity']:
                deriv = lambda x: pd.Series(x).diff().fillna(0).to_numpy()
                stimDescListFinal += [
                    Term([EvalFactor('deriv(applyCBasis(Q("{}"), {:.3f}))'.format(p, l))])
                    for p, l in product(stimXTrain.columns, stimLags)
                    ]
            stimModelDescFinal = ModelDesc([], stimDescListFinal)
            if RANK == 0:
                estimatorMetadata['stimModelDescFinal'] = stimModelDescFinal.describe()
            stimXTrainFinal = dmatrix(
                stimModelDescFinal,
                stimXTrain,
                return_type='dataframe')
            xTrainList.append(stimXTrainFinal)
            stimXTestFinal = dmatrix(
                stimModelDescFinal,
                stimXTest,
                return_type='dataframe')
            xTestList.append(stimXTestFinal)
        if modelOpts['ensembleTerms']:
            histDescList = [
                Term([LookupFactor(histTerm)])
                for histTerm in featuresDF.columns if '_cos' in histTerm
                ]
            #
            histDesc = ModelDesc([], histDescList)
            if RANK == 0:
                estimatorMetadata['histDesc'] = histDesc.describe()
            xTrainList.append(dmatrix(
                histDesc,
                featuresDF.iloc[train, :],
                return_type='dataframe'))
            xTestList.append(dmatrix(
                histDesc,
                featuresDF.iloc[test, :],
                return_type='dataframe'))
        #
        xTrain = pd.concat(xTrainList, axis='columns')
        xTest = pd.concat(xTestList, axis='columns')
        #
        if (RANK == 0) and (not arguments['dryRun']):
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
        if HAS_MPI:
            COMM.Barrier()
            # sync MPI threads, wait for 0
        if arguments['plotting'] and (RANK == 0) and (not HAS_MPI):
            with sns.plotting_context(context='notebook', font_scale=.75):
                cPalette = sns.color_palette("Set1")
                # plotDF = stimXTestFinal
                # plotDF = stimXTest
                plotDF = xTest
                fig, ax = plt.subplots(8, 1, sharex=True, figsize=(20, 15))
                if len(presentPrograms):
                    exampleProgramName = presentPrograms[0].replace('_amplitude', '')
                    exProgNum = int(exampleProgramName[-1])
                else:
                    exProgNum = 0
                plotQuer = 'program=={}'.format(exProgNum)
                for _, g in plotDF.query(plotQuer).groupby(['segment', 't']): break
                firstBlockIdx = g.index
                firstBlockT = (
                    plotDF.loc[firstBlockIdx, :].index
                    .get_level_values('bin'))
                pdfPath = os.path.join(
                    estimatorFiguresFolder,
                    'check_xy_test_{}.pdf'.format(varBaseName))
                ax[0].plot(
                    firstBlockT,
                    yTest.loc[firstBlockIdx, :].iloc[:, -1].to_numpy(),
                    c='k', lw=2,
                    label='target')
                if len(presentPrograms):
                    if modelOpts['rateInteraction'] and modelOpts['stimAmplitude']:
                        maNames = [
                            i
                            for i in plotDF.columns
                            if ('RateInHz' in i) and not ('deriv' in i) and (exampleProgramName in i)]
                        origTrace = (
                            featuresDF.iloc[test, :].loc[
                                firstBlockIdx,
                                presentPrograms[0]].to_numpy() *
                            featuresDF.iloc[test, :].loc[
                                firstBlockIdx,
                                'RateInHz'].to_numpy())
                        origAx = ax[1].twinx()
                        origAx.plot(
                            firstBlockT,
                            origTrace, lw=2,
                            c='k', label=presentPrograms[0] + ':Rate')
                        origAx.legend()
                        for nm in maNames:
                            if 'bs' in nm:
                                bsIdx = int(nm[nm.find('df=')+6])
                                cl = cPalette[bsIdx]
                            else:
                                cl = cPalette[0]
                            ax[1].plot(
                                firstBlockT,
                                plotDF.loc[firstBlockIdx, nm].to_numpy(),
                                c=cl, alpha=0.3)
                    if modelOpts['rateInteraction'] and modelOpts['stimVelocity']:
                        mdNames = [
                            i
                            for i in plotDF.columns
                            if ('RateInHz' in i) and ('deriv' in i) and (exampleProgramName in i)]
                        origTrace = deriv(
                            featuresDF.iloc[test, :].loc[
                                firstBlockIdx,
                                presentPrograms[0]])
                        origAx = ax[2].twinx()
                        origAx.plot(
                            firstBlockT,
                            origTrace, lw=2,
                            c='k', label='deriv(' + presentPrograms[0] + '):Rate')
                        origAx.legend()
                        for nm in mdNames:
                            if 'bs' in nm:
                                bsIdx = int(nm[nm.find('df=')+6])
                                cl = cPalette[bsIdx]
                            else:
                                cl = cPalette[0]
                            ax[2].plot(
                                firstBlockT,
                                plotDF.loc[firstBlockIdx, nm].to_numpy(),
                                c=cl, alpha=0.3)
                    if modelOpts['stimAmplitude']:
                        saNames = [
                            i
                            for i in plotDF.columns
                            if not ('RateInHz' in i) and not ('deriv' in i) and (exampleProgramName in i)]
                        origTrace = (
                            featuresDF.iloc[test, :].loc[
                                firstBlockIdx,
                                presentPrograms[0]].to_numpy())
                        origAx = ax[3].twinx()
                        origAx.plot(
                            firstBlockT,
                            origTrace, lw=2,
                            c='k', label=presentPrograms[0])
                        origAx.legend()
                        for nm in saNames:
                            if 'bs' in nm:
                                bsIdx = int(nm[nm.find('df=')+6])
                                cl = cPalette[bsIdx]
                            else:
                                cl = cPalette[0]
                            ax[3].plot(
                                firstBlockT,
                                plotDF.loc[firstBlockIdx, nm].to_numpy(),
                                c=cl, alpha=0.3)
                    if modelOpts['stimVelocity']:
                        sdNames = [
                            i
                            for i in plotDF.columns
                            if not ('RateInHz' in i) and ('deriv' in i) and (exampleProgramName in i)]
                        origTrace = deriv(
                            featuresDF.iloc[test, :].loc[
                                firstBlockIdx,
                                presentPrograms[0]].to_numpy())
                        origAx = ax[4].twinx()
                        origAx.plot(
                            firstBlockT,
                            origTrace, lw=2,
                            c='k', label='deriv(' + presentPrograms[0])
                        origAx.legend()
                        for nm in sdNames:
                            if 'bs' in nm:
                                bsIdx = int(nm[nm.find('df=')+6])
                                cl = cPalette[bsIdx]
                            else:
                                cl = cPalette[0]
                            ax[4].plot(
                                firstBlockT,
                                plotDF.loc[firstBlockIdx, nm].to_numpy(),
                                c=cl, alpha=0.3)
                cPalette = sns.color_palette("Set2", n_colors=8)
                if modelOpts['angle']:
                    origAx = ax[5].twinx()
                    origTrace = kinX.iloc[test, :]
                    for kIdx, nm in enumerate(kList):
                        origAx.plot(
                            firstBlockT,
                            kinX.iloc[test, :].loc[
                                firstBlockIdx,
                                nm + '_angle'].to_numpy(),
                            c=cPalette[kIdx],
                            label=nm + '_angle')
                    origAx.legend()
                    ktNames = [
                        i
                        for i in xTest.columns
                        if ('_angle' in i)]
                    for nIdx, nm in enumerate(ktNames):
                        if 'GT' in nm:
                            cl = cPalette[0]
                        elif 'K' in nm:
                            cl = cPalette[1]
                        elif 'M' in nm:
                            cl = cPalette[2]
                        ax[5].plot(
                            firstBlockT,
                            xTest.loc[firstBlockIdx, nm].to_numpy(),
                            c=cl, alpha=0.3)
                if modelOpts['angularVelocity']:
                    origAx = ax[6].twinx()
                    origTrace = kinX.iloc[test, :]
                    for kIdx, nm in enumerate(kList):
                        origAx.plot(
                            firstBlockT,
                            kinX.iloc[test, :].loc[
                                firstBlockIdx,
                                nm + '_angular_velocity'].to_numpy(),
                            c=cPalette[kIdx],
                            label=nm + '_angular_velocity')
                    origAx.legend()
                    ktNames = [
                        i
                        for i in xTest.columns
                        if ('_angular_velocity' in i)]
                    for nIdx, nm in enumerate(ktNames):
                        if 'GT' in nm:
                            cl = cPalette[0]
                        elif 'K' in nm:
                            cl = cPalette[1]
                        elif 'M' in nm:
                            cl = cPalette[2]
                        ax[6].plot(
                            firstBlockT,
                            xTest.loc[firstBlockIdx, nm].to_numpy(),
                            c=cl, alpha=0.3)
                if modelOpts['angularAcceleration']:
                    origAx = ax[7].twinx()
                    origTrace = kinX.iloc[test, :]
                    for kIdx, nm in enumerate(kList):
                        origAx.plot(
                            firstBlockT,
                            kinX.iloc[test, :].loc[
                                firstBlockIdx,
                                nm + '_angular_acceleration'].to_numpy(),
                            label=nm + '_angular_acceleration',
                            c=cPalette[kIdx])
                    origAx.legend()
                    ktNames = [
                        i
                        for i in xTest.columns
                        if ('_angular_acceleration' in i)]
                    for nIdx, nm in enumerate(ktNames):
                        if 'GT' in nm:
                            cl = cPalette[0]
                        elif 'K' in nm:
                            cl = cPalette[1]
                        elif 'M' in nm:
                            cl = cPalette[2]
                        ax[7].plot(
                            firstBlockT,
                            xTest.loc[firstBlockIdx, nm].to_numpy(),
                            # label=nm,
                            c=cl, alpha=0.3)
                if modelOpts['ensembleTerms']:
                    for colIdx in range(ihbasis.shape[1]):
                        cosBaseLabel = 'cos_{}'.format(colIdx)
                        ax[0].plot(
                            firstBlockT,
                            xTest
                            .loc[firstBlockIdx, :].iloc[:, -(colIdx + 1)]
                            .to_numpy(), c='k', alpha=0.3, label=cosBaseLabel)
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
            snr.GridSearchCV(gridParams)
            snr.cross_validate()
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
            print('{:.3f} seconds elapsed'.format(toc()))
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
