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
matplotlib.use('Qt5Agg')   # generate postscript output by default
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
from dataAnalysis.custom_transformers.tdr import TargetedDimensionalityReduction
import statsmodels.api as sm
import joblib as jb
import dill as pickle
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
        decimate=25,
        metaDataToCategories=False,
        verbose=False, procFun=None))
    #
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **arguments)
    #
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    regressorReader, regressorBlock = ns5.blockFromPath(
        regressorPath, lazy=arguments['lazy'])
    #
    targetLoadArgs = alignedAsigsKWargs.copy()
    #
    targetDF = ns5.alignedAsigsToDF(
        dataBlock,
        **targetLoadArgs)
    #
    featureLoadArgs = alignedAsigsKWargs.copy()
    unitNames = ['position#0', 'program0_amplitude#0', 'program1_amplitude#0', 'program2_amplitude#0', 'program3_amplitude#0', 'velocity#0']
    featureLoadArgs['unitNames'] = unitNames
    featureLoadArgs['unitQuery'] = None
    featuresDF = ns5.alignedAsigsToDF(
        regressorBlock, **featureLoadArgs)
    # posBinResolution = 0.025 # *100 in degrees
    # nPosBins = int(np.ceil((featuresDF['position#0'].max() - featuresDF['position#0'].min()) / posBinResolution))
    # 
    # featuresDF['positionBin'] = pd.cut(featuresDF['position#0'], bins=nPosBins, labels=False).to_numpy()
    # targetDF['positionBin'] = featuresDF['positionBin'].to_numpy()
    featuresDF['positionBin'] = 0
    targetDF['positionBin'] = 0
    featuresDF.set_index('positionBin', drop=True, append=True, inplace=True)
    targetDF.set_index('positionBin', drop=True, append=True, inplace=True)
    #
    metaDF = featuresDF.index.to_frame().reset_index(drop=True)
    # # independent model
    # for pNum in range(4):
    #     featuresDF['program{}_RateInHz#0'.format(pNum)] = metaDF['RateInHz'].to_numpy() * (featuresDF['program{}_amplitude#0'.format(pNum)] ** 0)
    # featuresDF['accel#0'] = featuresDF['velocity#0'].diff().fillna(0).to_numpy()
    # ACR model
    for pNum in range(4):
        featuresDF['program{}_ACR#0'.format(pNum)] = (
            metaDF['RateInHz'].to_numpy() *
            (featuresDF['program{}_amplitude#0'.format(pNum)])
            )
    featuresDF['position_x#0'] = np.cos(featuresDF['position#0'] * 100 * 2 * np.pi / 360)
    featuresDF['position_y#0'] = np.sin(featuresDF['position#0'] * 100 * 2 * np.pi / 360)
    featuresDF['velocity_x#0'] = featuresDF['position_y#0'] * (-1) * (featuresDF['velocity#0'] * 3e2)
    featuresDF['velocity_y#0'] = featuresDF['position_x#0'] * (featuresDF['velocity#0'] * 3e2)
    # pdb.set_trace()
    featuresDF.drop(
        columns=[
            'program{}_amplitude#0'.format(pNum)
            for pNum in range(4)] + ['position#0', 'velocity#0'],
        inplace=True)
    featuresDF.sort_index(axis='columns', kind='mergesort', inplace=True)
    featureScalers = [
        (MinMaxScaler(), ['position_x#0', 'position_y#0']),
        (MinMaxScaler(), ['velocity_x#0', 'velocity_y#0']),
        (MinMaxScaler(), ['program{}_ACR#0'.format(pNum) for pNum in range(4)]),
        ]
    targetScalers = []
    import warnings
    warnings.filterwarnings('error')
    #
    addLags = {
        'position_x#0': [5],
        'position_y#0': [5],
        'velocity_x#0': [5],
        'velocity_y#0': [5],
        'program0_ACR#0': [5],
        'program1_ACR#0': [5],
        'program2_ACR#0': [5],
        'program3_ACR#0': [5]
        }
    # pdb.set_trace()
    nPCAComponents = 12
    conditionNames = [
        'RateInHz', 'program', 'amplitudeCat',
        'pedalVelocityCat', 'positionBin']
    #
    addInterceptToTDR = True
    decimateTDR = 5
    rollingWindow = 10
    tdr = TargetedDimensionalityReduction(
        featuresDF=featuresDF,
        targetDF=pd.DataFrame(targetDF, dtype=np.int),
        model=sm.GLM,
        modelKWargs={'family': sm.families.Poisson()},
        featureScalers=featureScalers, targetScalers=targetScalers,
        addLags=addLags, decimate=decimateTDR, rollingWindow=rollingWindow,
        timeAxisName='positionBin',
        addIntercept=addInterceptToTDR,
        nPCAComponents=nPCAComponents, conditionNames=conditionNames,
        verbose=arguments['verbose'], plotting=arguments['plotting'])
    tdr.fit()
    tdr.clear_data()
    jb.dump(tdr, estimatorPath)
    estimatorMetadata = {
        'trainingDataPath': os.path.basename(triggeredPath),
        'path': os.path.basename(estimatorPath),
        'name': arguments['estimatorName'],
        'inputFeatures': targetDF.columns.to_list(),
        'outputFeatures': tdr.regressorNames,
        'alignedAsigsKWargs': alignedAsigsKWargs,
        }
    with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
        pickle.dump(
            estimatorMetadata, f)
    # pdb.set_trace()
    if arguments['plotting']:
        '''
        targetGroupProjected = pd.DataFrame(
            tdr.transform(targetDF),
            index=targetDF.index,
            columns=['TDR_{}'.format(i) for i in tdr.regressorNames])
        targetGroupProjected.columns.name = 'feature'
        targetGroupProjected.reset_index(inplace=True)
        fig, ax = plt.subplots()
        sns.lineplot(
            x='bin', y='TDR_velocity_x#0', hue='pedalSizeCat',
            data=targetGroupProjected, ci='sd', ax=ax)
        plt.show()
        '''
        pdb.set_trace()
        rsquared = pd.DataFrame([{'unit': i['unit'], 'rsquared': i['pseudorsquared']} for i in tdr.regressionList])
        unitName = rsquared.loc[rsquared['rsquared'].idxmax(), 'unit']
        regressionEntry = [i for i in tdr.regressionList if i['unit'] == unitName][0]
        #
        prediction = regressionEntry['reg'].predict()
        y = targetDF[unitName].to_numpy()[::decimateTDR]
        plt.plot(y/max(y), label='original')
        plt.plot(prediction, label='prediction')
        plt.title('{}: pR^2 = {}'.format(unitName, rsquared.loc[rsquared['rsquared'].idxmax(), 'rsquared']))
        plt.xlabel('samples')
        plt.ylabel('normalized (spk/s)')
        plt.legend()
        plt.show()
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
