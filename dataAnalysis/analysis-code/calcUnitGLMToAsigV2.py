#!/users/rdarie/anaconda/nda/bin/python
"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --trialIdx=trialIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --debugging                               print diagnostics? [default: False]
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
    --maskOutlierTrials                       delete outlier trials? [default: False]
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
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices)
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
    int(arguments['trialIdx']), arguments['exp'])
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
    #
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
    #
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
        **glmOptsLookup[arguments['estimatorName']],
        metaDataToCategories=False,
        verbose=False, procFun=None))
    #
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, alignSubFolder, **arguments)
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
    cv_kwargs = dict(
        n_splits=7,
        shuffle=True,
        stratifyFactors=['RateInHz', 'program', 'amplitudeCat'],
        continuousFactors=['segment', 'originalIndex'])
    addHistoryTerms = {
        'nb': 7,
        'dt': rasterOpts['binInterval'],
        'b': 0.001,
        'zflag': False
        }
    # make sure history terms don't bleed into current bins
    historyLen = .3
    historyEdge = raisedCosBoundary(
        b=addHistoryTerms['b'], DT=historyLen,
        minX=max(rasterOpts['binInterval'] * alignedAsigsKWargs['decimate'] / 2, 1e-3),
        nb=addHistoryTerms['nb'], plotting=(arguments['debugging'] and arguments['plotting'])
        )
    addHistoryTerms['endpoints'] = [historyEdge[0], historyEdge[0] + historyLen]
    if addHistoryTerms:
        iht, orthobas, ihbasis = makeRaisedCosBasis(**addHistoryTerms)
        addHistoryTerms['nb'] = orthobas.shape[1]
        # pdb.set_trace()
        plotTheBases = arguments['debugging'] and arguments['plotting']
        if plotTheBases:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(iht, ihbasis)
            ax[0].set_title('raised cos basis')
            ax[1].plot(iht, orthobas)
            ax[1].set_title('orthogonalized basis')
            plt.show()
    #
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
            distr='softplus', alpha=0.1, reg_lambda=1e-4,
            fit_intercept=False,
            # verbose=arguments['debugging'],
            verbose=arguments['verbose'],
            solver='L-BFGS-B',
            # solver='cdfast',
            max_iter=100, tol=1e-5, learning_rate=2e-1,
            score_metric='pseudo_R2', track_convergence=True),
        model=pyglmnetWrapper
        )
    kList = [
        'GT_Right_angle',
        'K_Right_angle',
        'M_Right_angle',
        'GT_Right_angular_velocity',
        'K_Right_angular_velocity',
        'M_Right_angular_velocity',
        'GT_Right_angular_acceleration',
        'K_Right_angular_acceleration',
        'M_Right_angular_acceleration'
        ]
    pList = [
        'program0_amplitude', 'program1_amplitude',
        'program2_amplitude', 'program3_amplitude'
        ]
    #
    countPerBin = rasterOpts['binInterval'] * alignedAsigsKWargs['decimate']
    if(RANK == 0):
        # find existing variants
        variantMetaDataPath = os.path.join(
            estimatorSubFolder, 'variant_metadata.pickle')
        varFolders = sorted(glob.glob(os.path.join(estimatorSubFolder, 'var_*')))
        existingVarSubFolders = [os.path.basename(i) for i in varFolders]
        if len(existingVarSubFolders):
            assert os.path.exists(variantMetaDataPath), 'Missing saved model info!'
            with open(variantMetaDataPath, 'rb') as f:
                allModelOpts = pickle.load(f)
        else:
            timeLags = [
                int(i / countPerBin)
                for i in [0, 75e-3, 150e-3, -150e-3]
            ]
            if arguments['verbose'] and (RANK == 0):
                print('timeLags = {}'.format(timeLags))
            allModelOpts = []
            for eht in [True, False]:
                for kt in [True, False]:
                    for st in [True, False]:
                        if not any([eht, kt, st]):
                            continue
                        if kt:
                            allKinLags = timeLags
                        else:
                            allKinLags = [0]
                        if st:
                            allStimLags = timeLags
                        else:
                            allStimLags = [0]
                        allModelOpts += [
                            {
                                'kinLag': allKinLags,
                                'stimLag': allStimLags,
                                'ensembleHistoryTerms': eht,
                                'stimTerms': st,
                                'kinematicTerms': kt}
                            ]
                        if not any([kt, st]):
                            continue
                        for ktl in allKinLags:
                            for stl in allStimLags:
                                allModelOpts += [
                                    {
                                        'kinLag': [ktl],
                                        'stimLag': [stl],
                                        'ensembleHistoryTerms': eht,
                                        'stimTerms': st,
                                        'kinematicTerms': kt}
                                    ]
            with open(variantMetaDataPath, 'wb') as f:
                pickle.dump(
                    allModelOpts, f)
    else:
        allModelOpts = None
        existingVarSubFolders = None
    #
    if HAS_MPI:
        COMM.Barrier()  # sync MPI threads, wait for 0
        allModelOpts = COMM.bcast(allModelOpts, root=0)
        existingVarSubFolders = COMM.bcast(existingVarSubFolders, root=0)
        modelArguments['modelKWargs']['verbose'] = False
    # if arguments['debugging']:
    #     modelArguments['modelKWargs']['max_iter'] = 20
    gridParams = [
        {
            'reg_lambda': np.logspace(
                np.log(5e-2), np.log(1e-4),
                7, base=np.exp(1))}
        ]
    # elastic_net L1 weight, maxiter and regularization params
    # chosen based on Benjamin et al 2017
    #
    featureNames = sorted([
        'RateInHz#0',
        'amplitude#0',
        'program0_amplitude#0',
        'program1_amplitude#0',
        'program2_amplitude#0',
        'program3_amplitude#0',
        'GT_Right_angle#0',
        'K_Right_angle#0',
        'M_Right_angle#0',
        'GT_Right_angular_velocity#0',
        'K_Right_angular_velocity#0',
        'M_Right_angular_velocity#0',
        'GT_Right_angular_acceleration#0',
        'K_Right_angular_acceleration#0',
        'M_Right_angular_acceleration#0',
        ])
    featureLoadArgs = alignedAsigsKWargs.copy()
    featureLoadArgs['unitNames'] = featureNames
    featureLoadArgs['unitQuery'] = None
    targetLoadArgs = alignedAsigsKWargs.copy()
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
        print('Loading {}'.format(triggeredPath))
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        targetDF = ns5.alignedAsigsToDF(
            dataBlock, **targetLoadArgs)
        if addHistoryTerms:
            historyLoadArgs = targetLoadArgs.copy()
            historyTerms = []
            for colIdx in range(addHistoryTerms['nb']):
                if arguments['verbose'] and (RANK == 0):
                    print('Calculating history term {}'.format(colIdx + 1))
                # thisBasis = ihbasis[:, colIdx] / np.sum(ihbasis[:, colIdx])
                thisBasis = orthobas[:, colIdx]
                #
                def pFun(x):
                    x.iloc[:, :] = np.apply_along_axis(
                        np.convolve, 1, x.to_numpy(),
                        thisBasis, mode='same')
                    return x
                historyLoadArgs['procFun'] = pFun
                historyDF = ns5.alignedAsigsToDF(
                    dataBlock, **historyLoadArgs)
                #
                historyBins = historyDF.index.get_level_values('bin')
                targetBins = targetDF.index.get_level_values('bin')
                binOverlap = historyBins.isin(targetBins)
                assert binOverlap.all()
                historyDF.rename(
                    columns={
                        i: i.replace('raster', 'cos_{}'.format(colIdx)).replace('#', '_')
                        for i in historyDF.columns.get_level_values('feature')
                        },
                    inplace=True)
                # if arguments['debugging'] and arguments['plotting']:
                if False:
                    # decFactor = targetLoadArgs['decimate']
                    plt.plot(historyDF.iloc[binOverlap, 0].to_numpy())
                    plt.plot(targetDF.iloc[:, 0].to_numpy())
                    plt.show()
                historyTerms.append(historyDF.iloc[binOverlap, :].copy())
        print('Loading {}'.format(regressorPath))
        regressorReader, regressorBlock = ns5.blockFromPath(
            regressorPath, lazy=arguments['lazy'])
        featuresDF = ns5.alignedAsigsToDF(
            regressorBlock, **featureLoadArgs)
        if addHistoryTerms:
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
        columnInfo = (
            featuresDF
            .columns.to_frame()
            .reset_index(drop=True))
        targetScalers = []
        featureScalers = []
        ############################################################
        # scale each feature independently
        ############################################################
        for featName in columnInfo['feature'].unique():
            if ('amplitude' in featName) or ('Hz' in featName):
                featureScalers.append((MaxAbsScaler(), [featName]))
            else:
                featureScalers.append((StandardScaler(), [featName]))
        ############################################################
        featuresDF = applyScalersGrouped(featuresDF, featureScalers)
        targetDF = applyScalersGrouped(targetDF, targetScalers)
        #
        featuresDF.columns = [
            '{}'.format(i[0]).replace('#0', '')
            for i in featuresDF.columns]
        targetDF.columns = [
            '{}'.format(i[0]).replace('_raster#0', '')
            for i in targetDF.columns]
        #
        with open(featuresMetaDataPath, 'wb') as f:
            pickle.dump(featuresMetaData, f)
        featuresDF.to_hdf(regressorH5Path, 'feature')
        targetDF.to_hdf(targetH5Path, 'target')
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
    if arguments['debugging']:
        yColDbList = ['elec10#0_p000', 'elec75#1_p000']
        yTrain = yTrain.loc[:, yColDbList].copy()
        yTest = yTest.loc[:, yColDbList].copy()
    #
    del targetDF
    # pdb.set_trace()
    # bla1 = dmatrix(kinematicDescStr(kList, allLagOpts[0]['kinLag']), featuresDF.iloc[train, :], return_type='dataframe')
    # bla2 = dmatrix(kinematicDescStr(kList, allLagOpts[-1]['kinLag']), featuresDF.iloc[train, :], return_type='dataframe')
    # plt.plot(bla1)
    # plt.plot(bla2)
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
        descStrList = []
        if modelOpts['kinematicTerms']:
            descStrList.append(kinematicDescStr(kList, modelOpts['kinLag']))
        if modelOpts['stimTerms']:
            descStrList.append(stimDescStr(pList, modelOpts['stimLag']))
        if modelOpts['ensembleHistoryTerms']:
            historyDescStr = ' + '.join([
                i
                for i in featuresDF.columns if '_cos' in i
                ])
            descStrList.append(historyDescStr)
        #
        descStr = ' + '.join(descStrList)
        #
        estimatorPath = os.path.join(
            variantSubFolder, 'estimator.joblib')
        estimatorMetadataPath = os.path.join(
            variantSubFolder, 'estimator_metadata.pickle')
        if HAS_MPI:
            estimatorPath = estimatorPath.replace(
                'estimator', 'estimator_{:03d}'.format(RANK))
        if RANK == 0:
            if not os.path.exists(variantSubFolder):
                os.makedirs(variantSubFolder, exist_ok=True)
            estimatorMetadata = {
                'targetPath': targetH5Path,
                'regressorPath': regressorH5Path,
                'path': os.path.basename(estimatorPath),
                'name': arguments['estimatorName'],
                'inputFeatures': yTrain.columns.to_list(),
                'modelDescStr': descStr,
                'trainIdx': train,
                'testIdx': test,
                'alignedAsigsKWargs': alignedAsigsKWargs,
                }
            if not arguments['debugging']:
                with open(estimatorMetadataPath, 'wb') as f:
                    pickle.dump(
                        estimatorMetadata, f)
        #
        desc = ModelDesc.from_formula(descStr)
        xTrain = dmatrix(
            desc,
            featuresDF.iloc[train, :],
            return_type='dataframe')
        xTest = dmatrix(
            desc,
            featuresDF.iloc[test, :],
            return_type='dataframe')
        #
        snr = SingleNeuronRegression(
            xTrain=xTrain, yTrain=yTrain,
            xTest=xTest, yTest=yTest,
            cv_folds=cv_folds,
            **modelArguments,
            verbose=arguments['verbose'],
            plotting=arguments['plotting']
            )
        if not arguments['debugging']:
            snr.apply_gridSearchCV(gridParams)
        else:
            pass
        #
        # tolFindParams = {
        #     'max_iter': 5000,
        #     'tol': 1e-6}
        # snr.dispatchParams(tolFindParams)
        if arguments['debugging']:
            from ttictoc import tic, toc
            tic()
        #
        snr.fit()
        #
        if arguments['debugging']:
            toc()
            pdb.set_trace()
        #
        if not arguments['debugging']:
            snr.clear_data()
        if not arguments['debugging']:
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
