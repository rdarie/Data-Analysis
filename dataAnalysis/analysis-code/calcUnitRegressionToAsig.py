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
    --inputBlockName=inputBlockName           filename for inputs [default: fr]
    --secondaryBlockName=secondaryBlockName   filename for secondary inputs [default: rig]
    --window=window                           process with short window? [default: long]
    --unitQuery=unitQuery                     how to restrict channels if not supplying a list? [default: fr]
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

import pdb
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

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
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
    duplicateControlsByProgram=True,
    makeControlProgram=False,
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=True, getMetaData=True, decimate=25,
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
#  
#  def zScoreAcross(x):
#      #  !This might need to be reworked if it zscores each segment individually
#      x.iloc[:, :] = stats.zscore(x, axis=None)
#      return x
#  
#  
#  targetLoadArgs['procFun'] = zScoreAcross
#  
targetDF = ns5.alignedAsigsToDF(
    dataBlock,
    **targetLoadArgs)
# targetDF.columns = [i[:-2] for i in targetDF.columns]
targetDF = targetDF.apply(stats.zscore, raw=True)
featureLoadArgs = alignedAsigsKWargs.copy()
unitNames = ['position#0', 'amplitude#0', 'velocity#0']
featureLoadArgs['unitNames'] = unitNames
featureLoadArgs['unitQuery'] = None
featuresDF = ns5.alignedAsigsToDF(
    regressorBlock, **featureLoadArgs)
# featuresDF.columns = [i[:-2] for i in featuresDF.columns]
nPosBins = int(np.ceil((featuresDF['position#0'].max() - featuresDF['position#0'].min()) / 0.05))
featuresDF['positionBin'] = pd.cut(featuresDF['position#0'], bins=nPosBins, labels=False).to_numpy()
targetDF['positionBin'] = featuresDF['positionBin'].to_numpy()
featuresDF.set_index('positionBin', drop=True, append=True, inplace=True)
targetDF.set_index('positionBin', drop=True, append=True, inplace=True)
#
metaDF = featuresDF.index.to_frame().reset_index(drop=True)
featuresDF['RateInHz#0'] = metaDF['RateInHz'].to_numpy() * featuresDF['amplitude#0'] ** 0
featuresDF['accel#0'] = featuresDF['velocity#0'].diff().fillna(0).to_numpy()
uniquePrograms = np.unique(metaDF['program'])
plotting = True
nPCAComponents = 20
conditionNames = [
    'RateInHz', 'program', 'amplitudeCat',
    'pedalSizeCat', 'pedalVelocityCat', 'positionBin']
for progName, idxVals in metaDF.groupby('program'):
    groupMask = metaDF.index.isin(idxVals.index)
    targetGroup = targetDF.loc[groupMask, :]
    featureGroup = featuresDF.loc[groupMask, :]
    # pdb.set_trace()
    tdr = TargetedDimensionalityReduction(
        timeAxisName='positionBin',
        nPCAComponents=nPCAComponents, conditionNames=conditionNames,
        featuresDF=featureGroup, targetDF=targetGroup, plotting=False)
    tdr.fit()
    targetGroupProjected = pd.DataFrame(
        tdr.transform(targetGroup),
        index=targetGroup.index,
        columns=['TDR_{}'.format(i) for i in tdr.regressorNames])
    targetGroupProjected.columns.name = 'feature'
    targetGroupProjected.reset_index(inplace=True)
    if plotting:
        fig, ax = plt.subplots()
        sns.lineplot(
            x='bin', y='TDR_velocity#0', hue='pedalSizeCat',
            data=targetGroupProjected, ci='sd', ax=ax)
        plt.show()
    thisEstimatorPath = estimatorPath.replace(
        '.joblib', '_prog{}.joblib'.format(int(progName)))
    jb.dump(tdr, thisEstimatorPath)
    estimatorMetadata = {
        'trainingDataPath': os.path.basename(triggeredPath),
        'path': os.path.basename(thisEstimatorPath),
        'name': arguments['estimatorName'],
        'inputFeatures': targetDF.columns.to_list(),
        'outputFeatures': tdr.regressorNames,
        'alignedAsigsKWargs': alignedAsigsKWargs,
        }
    with open(thisEstimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
        pickle.dump(
            estimatorMetadata, f)
