"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                 which experimental day to analyze
    --trialIdx=trialIdx                       which trial to analyze [default: 1]
    --processAll                              process entire experimental day? [default: False]
    --lazy                                    load from raw, or regular? [default: False]
    --verbose                                 print diagnostics? [default: False]
    --plotting                                plot out the correlation matrix? [default: True]
    --analysisName=analysisName               append a name to the resulting blocks? [default: default]
    --inputBlockName=inputBlockName           filename for inputs [default: raster]
    --secondaryBlockName=secondaryBlockName   filename for secondary inputs [default: rig]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: raster]
    --alignQuery=alignQuery                   query what the units will be aligned to? [default: midPeak]
    --estimatorName=estimatorName             filename for resulting estimator [default: tdr]
    --selector=selector                       filename if using a unit selector
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
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import numpy as np
import pandas as pd
from scipy import stats
from docopt import docopt
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
import dataAnalysis.preproc.ns5 as ns5
from dataAnalysis.custom_transformers.tdr import SMWrapper, TargetedDimensionalityReduction, SingleNeuronRegression
import statsmodels.api as sm
import MLencoding as spykesml
# from pyglmnet import GLM
import joblib as jb
import dill as pickle
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


def calcUnitRegressionToAsig():
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
        )
    if not os.path.exists(analysisSubFolder):
        os.makedirs(analysisSubFolder, exist_ok=True)
    #
    if arguments['processAll']:
        prefix = experimentName
    else:
        prefix = ns5FileName
    triggeredPath = os.path.join(
        analysisSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['inputBlockName'], arguments['window']))
    regressorPath = os.path.join(
        analysisSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['secondaryBlockName'], arguments['window']))
    resultPath = os.path.join(
        analysisSubFolder,
        prefix + '_{}_{}_calc.h5'.format(
            arguments['inputBlockName'], arguments['window']))
    fullEstimatorName = '{}_{}_{}_{}'.format(
        prefix,
        arguments['estimatorName'],
        arguments['window'],
        arguments['alignQuery'])
    estimatorPath = os.path.join(
        analysisSubFolder,
        fullEstimatorName + '.joblib')
    #
    rollingWindow = 30
    spkConversionFactor = (rollingWindow * 1e-3)
    alignedAsigsKWargs.update(dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False,
        transposeToColumns='feature', concatOn='columns',
        removeFuzzyName=True,
        getMetaData=[
            'RateInHz', 'activeGroup', 'amplitude', 'amplitudeCat',
            'bin', 'electrode', 'pedalDirection', 'pedalMetaCat',
            'pedalMovementCat', 'pedalMovementDuration',
            'pedalSize', 'pedalSizeCat', 'pedalVelocityCat',
            'program', 'segment', 't'],
        decimate=10,
        metaDataToCategories=False,
        verbose=False, procFun=None))
    #
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, analysisSubFolder, **arguments)
    # lags in units of the _analyze file sample rate
    addLags = {
        'position#0': list(range(-200, 210, 20)),
        'velocity#0': list(range(-200, 210, 20)),
        'program0_amplitude#0': list(range(-200, 210, 20)),
        'program1_amplitude#0': list(range(-200, 210, 20)),
        'program2_amplitude#0': list(range(-200, 210, 20)),
        'program3_amplitude#0': list(range(-200, 210, 20))
        }
    featureNames = sorted([
        'position#0', 'velocity#0',
        'program0_amplitude#0', 'program1_amplitude#0',
        'program2_amplitude#0', 'program3_amplitude#0'
        ])
    featureLoadArgs = alignedAsigsKWargs.copy()
    featureLoadArgs['unitNames'] = featureNames
    featureLoadArgs['unitQuery'] = None
    featureLoadArgs['addLags'] = addLags
    # featureLoadArgs['rollingWindow'] = rollingWindow
    #
    targetLoadArgs = alignedAsigsKWargs.copy()
    targetLoadArgs['rollingWindow'] = rollingWindow
    #
    reloadFeatures = True
    featuresMetaDataPath = estimatorPath.replace(
        '.joblib', '_features_meta.pickle')
    regressorH5Path = os.path.join(
        analysisSubFolder,
        prefix + '_{}_{}_{}.h5'.format(
            arguments['estimatorName'],
            arguments['secondaryBlockName'], arguments['window']))
    targetH5Path = os.path.join(
        analysisSubFolder,
        prefix + '_{}_{}_{}.h5'.format(
            arguments['estimatorName'],
            arguments['inputBlockName'], arguments['window']))
    #
    if os.path.exists(featuresMetaDataPath):
        with open(featuresMetaDataPath, 'rb') as f:
            featuresMetaData = pickle.load(f)
        sameFeatures = (featuresMetaData['featureLoadArgs'] == featureLoadArgs)
        sameTargets = (featuresMetaData['targetLoadArgs'] == targetLoadArgs)
        if sameFeatures and sameTargets:
            reloadFeatures = False
            featuresDF = pd.read_hdf(regressorH5Path, 'feature')
            targetDF = pd.read_hdf(targetH5Path, 'target')
    #
    ACRModel = True  # activation charge rate
    IARModel = True  # independent amplitude and rate
    if reloadFeatures:
        regressorReader, regressorBlock = ns5.blockFromPath(
            regressorPath, lazy=arguments['lazy'])
        featuresDF = ns5.alignedAsigsToDF(
            regressorBlock, **featureLoadArgs)
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        #
        targetDF = ns5.alignedAsigsToDF(
            dataBlock,
            **targetLoadArgs)
        #
        dropIndex = featuresDF.index[featuresDF.isna().T.any()]
        targetDF.drop(index=dropIndex, inplace=True)
        featuresDF.drop(index=dropIndex, inplace=True)
        #
        targetDF = pd.DataFrame(targetDF * spkConversionFactor, dtype=np.int)
        # derive regressors from saved traces (should move further upstream)
        progAmpNames = ['program{}_amplitude#0'.format(pNum) for pNum in range(4)]
        progAmpLookup = {'program{}_amplitude#0'.format(pNum): pNum for pNum in range(4)}
        dropColumns = []
        metaData = featuresDF.index.copy()
        # activeProgram = metaData.get_level_values('program').to_numpy()
        # trialAmplitude = metaData.get_level_values('amplitude').to_numpy()
        stimRate = metaData.get_level_values('RateInHz').to_numpy()
        uniqueRates = np.unique(stimRate[stimRate > 0])
        featuresDF.reset_index(inplace=True, drop=True)
        try:
            for name in featuresDF.columns:
                featureName = name[0]
                lag = name[1]
                if featureName in progAmpNames:
                    thisProgram = progAmpLookup[featureName]
                    if ACRModel:
                        acrName = 'p{}_ACR#0'.format(thisProgram)
                        featuresDF.loc[:, (acrName, lag)] = (
                            stimRate *
                            (featuresDF.loc[:, name])
                            )
                    if IARModel:
                        for rate in uniqueRates:
                            iarName = 'p{}_{}Hz#0'.format(thisProgram, int(rate))
                            featuresDF.loc[:, (iarName, lag)] = (
                                (stimRate == rate) *
                                (featuresDF.loc[:, name])
                                )
                    dropColumns.append(name)
                elif featureName == 'position#0':
                    featuresDF.loc[:, ('position_x#0', lag)] = ((
                        np.cos(
                            featuresDF.loc[:, ('position#0', lag)] *
                            100 * 2 * np.pi / 360))
                        .to_numpy())
                    featuresDF.sort_index(axis='columns', inplace=True)
                    featuresDF.loc[:, ('position_y#0', lag)] = ((
                        np.sin(
                            featuresDF.loc[:, ('position#0', lag)] *
                            100 * 2 * np.pi / 360))
                        .to_numpy())
                    featuresDF.sort_index(axis='columns', inplace=True)
                    dropColumns.append(name)
            for name in featuresDF.columns:
                # velocities require position
                featureName = name[0]
                lag = name[1]
                if featureName == 'velocity#0':
                    featuresDF.loc[:, ('velocity_x#0', lag)] = ((
                        featuresDF.loc[:, ('position_y#0', lag)] *
                        (-1) *
                        (featuresDF.loc[:, ('velocity#0', lag)] * 3e2))
                        .to_numpy())
                    featuresDF.sort_index(axis='columns', inplace=True)
                    featuresDF.loc[:, ('velocity_y#0', lag)] = ((
                        featuresDF.loc[:, ('position_x#0', lag)] *
                        (featuresDF.loc[:, ('velocity#0', lag)] * 3e2))
                        .to_numpy())
                    featuresDF.sort_index(axis='columns', inplace=True)
                    dropColumns.append(name)
        except Exception:
            traceback.print_exc()
            pdb.set_trace()
        featuresDF.drop(columns=dropColumns, inplace=True)
        featuresDF.columns = featuresDF.columns.remove_unused_levels()
        featuresDF.index = metaData
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
    targetScalers = []
    featureScalers = [
        (MinMaxScaler(), ['position_x#0', 'position_y#0']),
        (MinMaxScaler(), ['velocity_x#0', 'velocity_y#0'])
        ]
    columnInfo = (
        featuresDF
        .columns.to_frame()
        .reset_index(drop=True))
    #
    if ACRModel:
        featureScalers.append(
            (
                MinMaxScaler(),
                [i for i in columnInfo['feature'].unique() if 'ACR' in i]))
    if IARModel:
        featureScalers.append(
            (
                MinMaxScaler(),
                [i for i in columnInfo['feature'].unique() if 'Hz' in i]))
    # import warnings
    # warnings.filterwarnings('error')
    nPCAComponents = None
    conditionNames = [
        'RateInHz', 'program', 'amplitudeCat',
        ]
    #
    addInterceptToTDR = True
    modelKWargs = {
        'sm_class': sm.GLM,
        'sm_kwargs': {'family': sm.families.Poisson()},
        # 'regAlpha': 1e-4,
        'regAlpha': None,
        'regL1Wt': 0.5, 'regRefit': True,
        }
    # generate combinations of feature masks to select lags
    ILModel = False  # independent lags model
    SLModel = False  # single sensory lag model
    SLAltModel = True  # single sensory lag, 3 sets (kin only, kin+acr, kin+iar)
    if ILModel:
        regressorGroups = {
            'kinematic': (
                (columnInfo['feature'].str.contains('position')) |
                (columnInfo['feature'].str.contains('velocity'))),
            'stimulation': (
                (columnInfo['feature'].str.contains('ACR')) |
                (columnInfo['feature'].str.contains('Hz')))
            }
    if SLModel:
        regressorGroups = {
            'all': (
                (columnInfo['feature'].str.contains('position')) |
                (columnInfo['feature'].str.contains('program')) |
                (columnInfo['feature'].str.contains('ACR')) |
                (columnInfo['feature'].str.contains('Hz')))
            }
    if SLAltModel:
        regressorGroups = {
            'kinematic': (
                (columnInfo['feature'].str.contains('position')) |
                (columnInfo['feature'].str.contains('velocity'))),
            'kinandacr': (
                (columnInfo['feature'].str.contains('position')) |
                (columnInfo['feature'].str.contains('velocity')) |
                (columnInfo['feature'].str.contains('ACR'))),
            'kinandiar': (
                (columnInfo['feature'].str.contains('position')) |
                (columnInfo['feature'].str.contains('velocity')) |
                (columnInfo['feature'].str.contains('Hz'))),
            'acr': (
                (columnInfo['feature'].str.contains('ACR'))),
            'iar': (
                (columnInfo['feature'].str.contains('Hz')))
            }
    #
    featureSubsets = {k: [] for k in regressorGroups.keys()}
    for key, maskArray in regressorGroups.items():
        lags = columnInfo.loc[maskArray, 'lag'].unique()
        for lag in lags:
            subMask = columnInfo.loc[:, 'lag'] == lag
            subMask = maskArray & subMask
            theseIndexes = subMask.index[subMask].to_list()
            featureSubsets[key].append(theseIndexes)
    #
    featureSubsetsList = []
    if ILModel or SLModel:
        for indexSet in product(*featureSubsets.values()):
            fullSet = sorted(sum(indexSet, []))
            if addInterceptToTDR:
                fullSet += [-1]
            featureSubsetsList.append(fullSet)
    if SLAltModel:
        featureSubsetsList = sorted(sum(featureSubsets.values(), []))
        if addInterceptToTDR:
            featureSubsetsList = [i + [-1] for i in featureSubsetsList]
    gridParams = {
        # 'regAlpha': np.round(
        #     np.logspace(
        #         np.log(0.05), np.log(0.0001), 5, base=np.exp(1)),
        #     decimals=6),
        'featureMask': featureSubsetsList
    }
    snr = SingleNeuronRegression(
        featuresDF=featuresDF,
        targetDF=targetDF,
        featureScalers=featureScalers, targetScalers=targetScalers,
        addIntercept=addInterceptToTDR,
        model=SMWrapper,
        modelKWargs=modelKWargs, cv=3, conditionNames=conditionNames,
        verbose=arguments['verbose'], plotting=arguments['plotting'])
    snr.apply_gridSearchCV(gridParams)
    snr.fit()
    estimatorMetadata = {
        'trainingDataPath': os.path.basename(triggeredPath),
        'path': os.path.basename(estimatorPath),
        'name': arguments['estimatorName'],
        'inputFeatures': targetDF.columns.to_list(),
        'alignedAsigsKWargs': alignedAsigsKWargs,
        }
    with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
        pickle.dump(
            estimatorMetadata, f)
    
    if arguments['plotting']:
        snr.plot_xy()
    #
    # snr.clear_data()
    jb.dump(snr, estimatorPath)
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
