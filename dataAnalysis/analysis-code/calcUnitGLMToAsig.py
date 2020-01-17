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
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import numpy as np
import pandas as pd
from scipy import stats
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
import dataAnalysis.preproc.ns5 as ns5
from dataAnalysis.custom_transformers.tdr import (
    SMWrapper, pyglmnetWrapper, trialAwareStratifiedKFold,
    applyScalersGrouped, SingleNeuronRegression)
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
import scipy.linalg


def raisedCos(x, c, dc):
    argCos = (x-c)*np.pi/dc/2
    argCos[argCos > np.pi] = np.pi
    argCos[argCos < -np.pi] = -np.pi
    return (np.cos(argCos) + 1) / 2


def makeRaisedCosBasis(
        nb, dt, endpoints, b=0.01, zflag=0):
    """
        Make nonlinearly stretched basis consisting of raised cosines
        Inputs:  nb = # of basis vectors
                 dt = time bin separation for representing basis
                 endpoints = 2-vector containg [1st_peak  last_peak], the peak 
                         (i.e. center) of the last raised cosine basis vectors
                 b = offset for nonlinear stretching of x axis:  y = log(x+b) 
                     (larger b -> more nearly linear stretching)
                 zflag = flag for making (if = 1) finest-timescale basis
                         vector constant below its peak
        
         Outputs:  iht = time lattice on which basis is defined
                   ihbas = orthogonalized basis
                   ihbasis = basis itself
        
         Example call
         iht, ihbas, ihbasis = makeRaisedCosBasis(10, .01, [0, 10], .1);
    """
    eps = 1e-20
    nlin = lambda x: np.log(x + eps)
    invnl = lambda x: np.exp(x) - eps
    assert b > 0
    if isinstance(endpoints, list):
        endpoints = np.array(endpoints)
    #
    yrnge = nlin(endpoints + b)
    db = np.diff(yrnge)/(nb-1)      # spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db, db)  # centers for basis vectors
    mxt = invnl(yrnge[1]+2*db) - b  # maximum time bin
    iht = np.arange(0, mxt, dt)
    nt = iht.size
    #
    repIht = np.vstack([nlin(iht+b) for i in range(nb)]).transpose()
    repCtrs = np.vstack([ctrs for i in range(nt)])
    ihbasis = raisedCos(repIht, repCtrs, db)
    if zflag:
        tMask = iht < endpoints[0]
        ihbasis[tMask, 0] = 1
    orthobas = scipy.linalg.orth(ihbasis)
    return iht, orthobas, ihbasis


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
    # lags in units of the _analyze file sample rate
    lagsInvestigated = [50, 100, 250]
    lagsInvestigated += [(-1) * i for i in lagsInvestigated]
    lagsInvestigated = [0] + sorted(lagsInvestigated)
    if DEBUGGING:
        lagsInvestigated = [0]
        alignedAsigsKWargs['unitNames'] = [
            'elec10#0_raster#0', 'elec75#1_raster#0']
    addLags = {
        # 'position#0': lagsInvestigated,
        # 'velocity#0': lagsInvestigated,
        # 'position_x#0': lagsInvestigated,
        # 'velocity_x#0': lagsInvestigated,
        # 'position_y#0': lagsInvestigated,
        # 'velocity_y#0': lagsInvestigated,
        'RateInHz#0': lagsInvestigated,
        'amplitude#0': lagsInvestigated,
        'program0_amplitude#0': lagsInvestigated,
        'program1_amplitude#0': lagsInvestigated,
        'program2_amplitude#0': lagsInvestigated,
        'program3_amplitude#0': lagsInvestigated,
        #
        # 'program0_ACR#0': lagsInvestigated,
        # 'program1_ACR#0': lagsInvestigated,
        # 'program2_ACR#0': lagsInvestigated,
        # 'program3_ACR#0': lagsInvestigated,
        #
        'GT_Right_angle#0': lagsInvestigated,
        'K_Right_angle#0': lagsInvestigated,
        'M_Right_angle#0': lagsInvestigated,
        'GT_Right_angular_velocity#0': lagsInvestigated,
        'K_Right_angular_velocity#0': lagsInvestigated,
        'M_Right_angular_velocity#0': lagsInvestigated,
        'GT_Right_angular_acceleration#0': lagsInvestigated,
        'K_Right_angular_acceleration#0': lagsInvestigated,
        'M_Right_angular_acceleration#0': lagsInvestigated,
        }
    addHistoryTerms = {
        'nb': 7,
        'dt': rasterOpts['binInterval'],
        'endpoints': [0, .250],
        'b': 0.01}

    featureNames = sorted([i for i in addLags.keys()])
    featureLoadArgs = alignedAsigsKWargs.copy()
    featureLoadArgs['unitNames'] = featureNames
    featureLoadArgs['unitQuery'] = None
    featureLoadArgs['addLags'] = addLags
    targetLoadArgs = alignedAsigsKWargs.copy()
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
        n_splits=5,
        stratifyFactors=['RateInHz', 'program', 'amplitudeCat'],
        continuousFactors=['segment', 'originalIndex'])
    #
    if RANK == 0:
        if os.path.exists(featuresMetaDataPath):
            with open(featuresMetaDataPath, 'rb') as f:
                featuresMetaData = pickle.load(f)
            sameFeatures = (featuresMetaData['featureLoadArgs'] == featureLoadArgs)
            sameTargets = (featuresMetaData['targetLoadArgs'] == targetLoadArgs)
            if sameFeatures and sameTargets:
                reloadFeatures = False
                featuresDF = pd.read_hdf(regressorH5Path, 'feature')
                targetDF = pd.read_hdf(targetH5Path, 'target')
            else:
                reloadFeatures = True
        else:
            reloadFeatures = True
        if reloadFeatures:
            print('Loading {}'.format(triggeredPath))
            dataReader, dataBlock = ns5.blockFromPath(
                triggeredPath, lazy=arguments['lazy'])
            targetDF = ns5.alignedAsigsToDF(
                dataBlock, **targetLoadArgs)
            if addHistoryTerms:
                iht, orthobas, ihbasis = makeRaisedCosBasis(**addHistoryTerms)
                historyLoadArgs = targetLoadArgs.copy()
                historyLoadArgs['rollingWindow'] = None
                historyLoadArgs['decimate'] = 1
                for colIdx in range(addHistoryTerms['nb']):
                    thisBasis = ihbasis[:, colIdx]
                    #
                    def pFun(x):
                        x.iloc[:, :] = np.apply_along_axis(
                            np.convolve, 1, x.to_numpy(),
                            thisBasis / np.sum(thisBasis), mode='same')
                        return x
                    historyLoadArgs['procFun'] = pFun
                    historyDF = ns5.alignedAsigsToDF(
                        dataBlock, **historyLoadArgs)
                    #
                    historyBins = historyDF.index.get_level_values('bin')
                    targetBins = targetDF.index.get_level_values('bin')
                    binOverlap = historyBins.isin(targetBins)
                    if True:
                        # decFactor = targetLoadArgs['decimate']
                        pdb.set_trace()
                        plt.plot(historyDF.iloc[binOverlap, 0].to_numpy())
                        plt.plot(targetDF.iloc[:, 0].to_numpy())
                        plt.show()
            #
            print('Loading {}'.format(regressorPath))
            regressorReader, regressorBlock = ns5.blockFromPath(
                regressorPath, lazy=arguments['lazy'])
            featuresDF = ns5.alignedAsigsToDF(
                regressorBlock, **featureLoadArgs)
            #
            dropMask = np.logical_or.reduce([
                targetDF.isna().T.any(),
                featuresDF.isna().T.any()])
            dropIndex = targetDF.index[dropMask]
            targetDF.drop(index=dropIndex, inplace=True)
            featuresDF.drop(index=dropIndex, inplace=True)
            #  Standardize units (count/window)
            #  targetDF = pd.DataFrame(targetDF, dtype=np.int)
            columnInfo = (
                featuresDF
                .columns.to_frame()
                .reset_index(drop=True))
            #
            targetScalers = []
            featureScalers = []
            ########################
            # group features by units and scale together
            ########################
            #
            # featureScalers.append([
            #     (MinMaxScaler(), ['position_x#0', 'position_y#0']),
            #     (MinMaxScaler(), ['velocity_x#0', 'velocity_y#0'])
            #     ])
            # featureScalers.append(
            #     (
            #         MaxAbsScaler(),
            #         [
            #             i
            #             for i in columnInfo['feature'].unique()
            #             if 'Hz' in i]))
            # featureScalers.append(
            #     (
            #         MaxAbsScaler(),
            #         [
            #             i
            #             for i in columnInfo['feature'].unique()
            #             if 'amplitude' in i]))
            # featureScalers.append(
            #     (
            #         StandardScaler(),
            #         [
            #             i
            #             for i in columnInfo['feature'].unique()
            #             if 'angle' in i]))
            # featureScalers.append(
            #     (
            #         StandardScaler(),
            #         [
            #             i
            #             for i in columnInfo['feature'].unique()
            #             if 'angular_velocity' in i]))
            # featureScalers.append(
            #     (
            #         MinMaxScaler(),
            #         [i for i in columnInfo['feature'].unique() if 'ACR' in i]))
            #
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
                '{}_{:+04d}'.format(*i).replace('#0', '').replace('+', 'p').replace('-', 'm')
                for i in featuresDF.columns]
            targetDF.columns = [
                '{}_{:+04d}'.format(*i).replace('_raster#0', '').replace('+', 'p').replace('-', 'm')
                for i in targetDF.columns]
            #
            featuresMetaData = {
                'targetLoadArgs': targetLoadArgs,
                'featureLoadArgs': featureLoadArgs
            }
            with open(featuresMetaDataPath, 'wb') as f:
                pickle.dump(featuresMetaData, f)
            featuresDF.to_hdf(regressorH5Path, 'feature')
            targetDF.to_hdf(targetH5Path, 'target')
        #
        prelimCV = trialAwareStratifiedKFold(**cv_kwargs).split(featuresDF)
        (train, test) = prelimCV[0]
        yTrain = targetDF.iloc[train, :]
        yTest = targetDF.iloc[test, :]
        #
        cv_kwargs['n_splits'] -= 1
        cv = trialAwareStratifiedKFold(**cv_kwargs)
        cv_folds = cv.split(yTrain)
        #
    else:
        featuresDF = None
        yTrain = None
        yTest = None
        cv_folds = None
        train = None
        test = None
    #
    if HAS_MPI:
        # !! OverflowError: cannot serialize a bytes object larger than 4 GiB
        COMM.Barrier()  # sync MPI threads, wait for 0
        featuresDF = COMM.bcast(featuresDF, root=0)
        yTrain = COMM.bcast(yTrain, root=0)
        yTest = COMM.bcast(yTest, root=0)
        cv_folds = COMM.bcast(cv_folds, root=0)
        test = COMM.bcast(test, root=0)
        train = COMM.bcast(train, root=0)
        # subselect this RANK's units
        yColIdx = range(RANK, yTrain.shape[1], SIZE)
        print(
            'Process {}, running regression for units {}'
            .format(RANK, yColIdx))
        yTrain = yTrain.iloc[:, yColIdx]
        yTest = yTest.iloc[:, yColIdx]
    #
    # statistical model specific options
    # for statsmodels wrapper
    modelArguments = dict(
        modelKWargs={
            'sm_class': sm.GLM,
            'family': sm.families.Poisson(),
            'alpha': None, 'L1_wt': None,
            'refit': None,
            # 'alpha': 1e-3, 'L1_wt': .1,
            # 'refit': False,
            'maxiter': 250, 'disp': True
            },
        model=SMWrapper
    )
    gridParams = {
        'alpha': np.logspace(
            np.log(0.05), np.log(0.0001),
            7, base=np.exp(1))
        }
    modelArguments = dict(
        modelKWargs=dict(
            distr='softplus', alpha=0.1, reg_lambda=1e-3,
            fit_intercept=False,
            # verbose='DEBUG',
            verbose=False,
            # solver='batch-gradient',
            solver='cdfast',
            max_iter=10000, tol=1e-06,
            score_metric='pseudo_R2', track_convergence=True),
        model=pyglmnetWrapper
    )
    gridParams = [
        {
            'reg_lambda': np.logspace(
                np.log(5e-2), np.log(1e-4),
                7, base=np.exp(1))}
        ]
    # elastic_net L1 weight, maxiter and regularization params
    # chosen based on Benjamin et al 2017
    #
    def formatLag(name, lag):
        return '{}_{:+04d}'.format(name, lag).replace('+', 'p').replace('-', 'm')
    # 
    allModelDescStr = []
    for lag in lagsInvestigated:
        kinematicDescStr = ' + '.join([
            formatLag('GT_Right_angle', lag),
            formatLag('K_Right_angle', lag),
            formatLag('M_Right_angle', lag),
            formatLag('GT_Right_angular_velocity', lag),
            formatLag('K_Right_angular_velocity', lag),
            formatLag('M_Right_angular_velocity', lag),
            formatLag('GT_Right_angular_acceleration', lag),
            formatLag('K_Right_angular_acceleration', lag),
            formatLag('M_Right_angular_acceleration', lag)
        ])
        stimDescStr = ' + '.join(
            [
                '{}'.format(formatLag('RateInHz', lag)) + ' * ' + i
                for i in [
                    formatLag('program0_amplitude', lag), formatLag('program1_amplitude', lag),
                    formatLag('program2_amplitude', lag), formatLag('program3_amplitude', lag)
                    ]
            ]
            )
        descStr = ' + '.join([kinematicDescStr, stimDescStr])
        allModelDescStr.append(descStr)
    #
    for modelIdx, descStr in enumerate(allModelDescStr):
        desc = ModelDesc.from_formula(descStr)
        xTrain = dmatrix(
            desc,
            featuresDF.iloc[train, :],
            return_type='dataframe')
        xTest = dmatrix(
            desc,
            featuresDF.iloc[test, :],
            return_type='dataframe')
        if DEBUGGING:
            fig, ax = plt.subplots()
            featureBins = np.linspace(-3, 3, 100)
            for featName in xTrain.columns:
                print('{}: {}'.format(featName, xTrain[featName].unique()))
                sns.distplot(
                    xTrain[featName], bins=featureBins,
                    ax=ax, label=featName, kde=False)
            plt.legend()
            plt.show()
        snr = SingleNeuronRegression(
            xTrain=xTrain, yTrain=yTrain,
            xTest=xTest, yTest=yTest,
            cv_folds=cv_folds,
            **modelArguments,
            verbose=arguments['verbose'],
            plotting=arguments['plotting']
            )
        #
        if not DEBUGGING:
            snr.apply_gridSearchCV(gridParams)
        else:
            pass
        #
        # tolFindParams = {
        #     'max_iter': 50000,
        #     'tol': 1e-10}
        # snr.dispatchParams(tolFindParams)
        snr.fit()
        if not DEBUGGING:
            snr.clear_data()
        else:
            for unitName in snr.regressionList.keys():
                reg = snr.regressionList[unitName]['reg']
                print(snr.regressionList[unitName]['validationScore'])
        #
        variantSubFolder = os.path.join(
            estimatorSubFolder,
            'var_{:03d}'.format(modelIdx)
            )
        if not os.path.exists(variantSubFolder):
            os.makedirs(variantSubFolder, exist_ok=True)
        estimatorPath = os.path.join(
            variantSubFolder, 'estimator.joblib')
        estimatorMetadataPath = os.path.join(
            variantSubFolder, 'estimator_metadata.pickle')
        if RANK == 0:
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
            if not DEBUGGING:
                with open(estimatorMetadataPath, 'wb') as f:
                    pickle.dump(
                        estimatorMetadata, f)
        if HAS_MPI:
            estimatorPath = estimatorPath.replace(
                'estimator', 'estimator_{:03d}'.format(RANK))
        else:
            if arguments['plotting']:
                snr.plot_xy()
        if not DEBUGGING:
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
