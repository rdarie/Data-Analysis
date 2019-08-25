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
import statsmodels.api as sm

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
#
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
targetDF.columns = [i[:-2] for i in targetDF.columns]
targetDF = targetDF.apply(stats.zscore, raw=True)
featureLoadArgs = alignedAsigsKWargs.copy()
unitNames = ['position#0', 'amplitude#0', 'velocity#0']
featureLoadArgs['unitNames'] = unitNames
featureLoadArgs['unitQuery'] = None
featuresDF = ns5.alignedAsigsToDF(
    regressorBlock, **featureLoadArgs)
featuresDF.columns = [i[:-2] for i in featuresDF.columns]

nPosBins = int(np.ceil((featuresDF['position'].max() - featuresDF['position'].min()) / 0.05))
featuresDF['positionBin'] = pd.cut(featuresDF['position'], bins=nPosBins, labels=False).to_numpy()
targetDF['positionBin'] = featuresDF['positionBin'].to_numpy()
featuresDF.set_index('positionBin', drop=True, append=True, inplace=True)
targetDF.set_index('positionBin', drop=True, append=True, inplace=True)
#
metaDF = featuresDF.index.to_frame().reset_index(drop=True)
featuresDF['RateInHz'] = metaDF['RateInHz'].to_numpy() * featuresDF['amplitude'] ** 0
featuresDF['accel'] = featuresDF['velocity'].diff().fillna(0).to_numpy()
uniquePrograms = np.unique(metaDF['program'])

from sklearn.base import TransformerMixin


class TargetedDimensionalityReduction(TransformerMixin):
    def __init__(
            self, timeAxis=None,
            featuresDF=None, targetDF=None):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[1] for _ in X]


uniquePositionBins = np.unique(metaDF['positionBin'])
uniqueUnits = targetDF.columns.to_numpy()
betaIndex = pd.MultiIndex.from_product([uniqueUnits, uniquePositionBins], names=['unit', 'positionBin'])
regressorNames = np.concatenate([['intercept'], featuresDF.columns.to_numpy()])
betas = {
    i: pd.DataFrame(np.nan, index=betaIndex, columns=regressorNames)
    for i in uniquePrograms}
#
plotting = True
if plotting:
    ax = sns.distplot(featuresDF['position'])
    plt.show()
regressionList = []
for name, yIndex in metaDF.groupby(['program', 'positionBin']):
    programName = name[0]
    posBinName = name[1]
    groupMask = metaDF.index.isin(yIndex.index)
    x = featuresDF.loc[groupMask, :].to_numpy()
    x2 = sm.add_constant(x, has_constant='add')
    for colName in targetDF:
        y = targetDF.loc[groupMask, colName].to_numpy()
        reg = sm.OLS(y, x2).fit()
        # print(reg.summary())
        yhat = reg.predict(x2)
        # predictionDF.loc[groupMask, colName] = yhat
        # print('reg.params is nan: {}'.format(np.isnan(reg.params).any()))
        betas[programName].loc[(colName, posBinName), :] = reg.params
        regressionList.append(
            {
                'program': programName, 'positionBin': posBinName,
                'unit': colName, 'reg': reg
                }
        )
    betas[programName].dropna(inplace=True)
    betas[programName].columns.name = 'taskVariable'

#
from sklearn.decomposition import PCA
nComponents = 20
pca = PCA(n_components=nComponents)
conditionNames = [
    'RateInHz', 'program', 'amplitudeCat',
    'pedalSizeCat', 'pedalVelocityCat', 'positionBin']
conditionAverages = targetDF.groupby(conditionNames).agg('mean')
pca.fit(conditionAverages.to_numpy())
# check that "denoising" works as expected
checkDenoising = False
if checkDenoising:
    denoisedCondAve = pd.DataFrame(
        pca.inverse_transform(pca.transform(conditionAverages)),
        index=conditionAverages.index, columns=conditionAverages.columns)
    D = np.zeros((conditionAverages.shape[1], conditionAverages.shape[1]))
    for idx in range(nComponents):
        D += np.dot(
            np.atleast_2d(pca.components_[idx, :]).transpose(),
            np.atleast_2d(pca.components_[idx, :]))
    compareDenoisedCA = pd.DataFrame(
        np.dot(D, conditionAverages.transpose()).transpose(),
        index=conditionAverages.index, columns=conditionAverages.columns)
    plt.plot(compareDenoisedCA[exampleNeuronName].to_numpy(), linestyle='-', label='D*CA')
    plt.plot(denoisedCondAve[exampleNeuronName].to_numpy(), linestyle='--', label='inv_trans(trans(CA))')
    plt.legend()
    plt.show()
transposedBetas = {i: betas[i].unstack(level='positionBin').transpose() for i in uniquePrograms}
denoisedBetas = {i: pd.DataFrame(pca.inverse_transform(pca.transform(transposedBetas[i])), index=transposedBetas[i].index, columns=transposedBetas[i].columns) for i in uniquePrograms}
#
pdb.set_trace()
for progName, targetGroup in targetDF.groupby('program'):
    if progName == 999:
        continue
    maxBins = []
    for name, group in denoisedBetas[progName].groupby('taskVariable'):
        maxBins.append((group ** 2).sum(axis='columns').idxmax())
    betaMax = transposedBetas[progName].loc[maxBins, :]
    q, r = np.linalg.qr(betaMax.transpose())
    conditionNames = [
        'RateInHz', 'amplitudeCat',
        'pedalSizeCat', 'pedalVelocityCat', 'bin']
    #  conditionAveragesT = targetGroup.groupby(conditionNames).agg('mean')
    #  conditionAveragesProjected = pd.DataFrame(
    #      np.dot(q.transpose(), conditionAveragesT.transpose()).transpose(),
    #      index=conditionAveragesT.index, columns=['TDR_{}'.format(i) for i in regressorNames])
    #  conditionAveragesProjected.columns.name = 'feature'
    #  conditionAveragesProjected.reset_index(inplace=True)
    targetGroupProjected = pd.DataFrame(
        np.dot(q.transpose(), targetGroup.transpose()).transpose(),
        index=targetGroup.index, columns=['TDR_{}'.format(i) for i in regressorNames])
    targetGroupProjected.columns.name = 'feature'
    targetGroupProjected.reset_index(inplace=True)
    if plotting:
        fig, ax = plt.subplots()
        sns.lineplot(
            x='bin', y='TDR_velocity', hue='pedalSizeCat',
            data=targetGroupProjected, ax=ax)
        plt.show()
    break

predictionDF = pd.DataFrame(np.nan, index=targetDF.index, columns=targetDF.columns)

if plotting:
    fullTargetDF = targetDF.copy()
    fullTargetDF['position'] = featuresDF.loc[:, 'position'].to_numpy()
    fullTargetDF.reset_index(inplace=True)
    # fullTargetDF.sort_values(by=['position'], inplace=True, kind='mergesort')
    fullPredictionDF = predictionDF.copy()
    fullPredictionDF['position'] = featuresDF.loc[:, 'position'].to_numpy()
    fullPredictionDF.reset_index(inplace=True)
    # fullPredictionDF.sort_values(by=['position'], inplace=True, kind='mergesort')
    exampleNeuronName = 'elec75#1_fr_sqrt'
    fig, ax = plt.subplots()
    sns.lineplot(
        x='bin', y=exampleNeuronName,
        data=fullTargetDF, ax=ax, ci='sem')
    sns.lineplot(
        x='bin', y=exampleNeuronName,
        data=fullPredictionDF, ax=ax, ci='sem')
    plt.show()

writeOut = True
if not writeOut:
    raise(Exception('Overriding writeout step'))