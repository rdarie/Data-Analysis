"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             make plots? [default: False]
    --debugging                            restrict datasets for debugging? [default: False]
    --showFigures                          show plots? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --datasetNameRhs=datasetNameRhs        which trig_ block to pull [default: Block]
    --selectionNameRhs=selectionNameRhs    how to restrict channels? [default: fr_sqrt]
    --datasetNameLhs=datasetNameLhs        which trig_ block to pull [default: Block]
    --selectionNameLhs=selectionNameLhs    how to restrict channels? [default: fr_sqrt]
    --estimatorName=estimatorName          filename for resulting estimator (cross-validated n_comps)
    --selector=selector                    filename if using a unit selector
"""
import matplotlib
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from dask.distributed import Client
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
import vg
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklego.preprocessing import PatsyTransformer
from sklearn_pandas import DataFrameMapper
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import sys
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
from itertools import product
from scipy import stats
from bokeh.models import ColorBar, ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
from mayavi import mlab

for arg in sys.argv:
    print(arg)

idxSl = pd.IndexSlice
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'alignQuery': 'midPeak', 'datasetNameRhs': 'Block_XL_df_c', 'blockIdx': '2', 'window': 'long',
        'debugging': False, 'exp': 'exp202101281100', 'winStop': '400', 'selectionNameRhs': 'lfp_CAR',
        'estimatorName': 'enr', 'selectionNameLhs': 'rig', 'lazy': False, 'processAll': True, 'plotting': True,
        'winStart': '200', 'selector': None, 'datasetNameLhs': 'Block_XL_df_c', 'analysisName': 'default',
        'alignFolderName': 'motion', 'verbose': '2', 'showFigures': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


if __name__ == '__main__':
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    estimatorsSubFolder = os.path.join(
        analysisSubFolder, 'estimators')
    if not os.path.exists(estimatorsSubFolder):
        os.makedirs(estimatorsSubFolder)
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], 'pls')
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    #
    rhsDatasetPath = os.path.join(
        dataFramesFolder,
        arguments['datasetNameRhs'] + '.h5'
        )
    assert os.path.exists(rhsDatasetPath)
    lhsDatasetPath = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '.h5'
        )
    assert os.path.exists(lhsDatasetPath)
    fullEstimatorName = '{}_{}_{}'.format(
        arguments['estimatorName'], arguments['datasetNameLhs'], arguments['selectionNameLhs'])
    estimatorPath = os.path.join(
        estimatorsSubFolder,
        fullEstimatorName + '.h5'
        )
    loadingMetaPathLhs = os.path.join(
        dataFramesFolder,
        arguments['datasetNameLhs'] + '_' + arguments['selectionNameLhs'] + '_meta.pickle'
        )
    #
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
    #
    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    workIdx = cvIterator.work
    ###
    estimatorClass = ElasticNet
    estimatorKWArgs = dict()
    gridSearchKWArgs = dict(
        max_iter=10000,
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1.],
        cv=cvIterator)
    '''if arguments['debugging']:
        gridSearchKWArgs['l1_ratio'] = [.1, .7, .99, 1.]
        gridSearchKWArgs['n_alphas'] = 100'''
    #
    '''estimatorClass = SGDRegressor
    estimatorKWArgs = dict(
        max_iter=2000
        )
    gridSearchKWArgs = dict(
        param_grid=dict(
            l1_ratio=[.1, .75, 1.],
            alpha=np.logspace(-4, 1, 4),
            loss=['epsilon_insensitive'],
            ),
        cv=cvIterator,
        scoring='r2'
        )'''
    crossvalKWArgs = dict(
        cv=cvIterator, scoring='r2',
        return_estimator=True,
        return_train_score=True)
    lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
    lhsMasks = pd.read_hdf(lhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameLhs']))
    #
    workingLhsDF = lhsDF.iloc[workIdx, :]
    workingRhsDF = rhsDF.iloc[workIdx, :]
    nFeatures = lhsDF.columns.shape[0]
    nTargets = rhsDF.columns.shape[0]
    #
    lhGroupNames = lhsMasks.index.names
    trialInfo = lhsDF.index.to_frame().reset_index(drop=True)
    infoPerTrial = trialInfo.drop_duplicates(['segment', 'originalIndex', 't'])
    ########################################################
    trialInfo.drop_duplicates(['segment', 'originalIndex', 't']).groupby(stimulusConditionNames).count().iloc[:, 0]
    trialInfo.drop_duplicates(['segment', 'originalIndex', 't']).groupby(['electrode', 'pedalMovementCat']).count().iloc[:, 0]
    conditions = {
        'pedalDirection': np.asarray(['CW']),
        'pedalMovementCat': np.asarray(['outbound', 'return']),
        'RateInHz': np.asarray([50, 100]),
        'amplitude': np.linspace(200, 800, 5),
        'electrode': np.asarray(['+ E16 - E9'])
        }
    electrodeProportion = 0.25
    electrodeNAProportion = 0.25
    kinematicNAProportion = 0.5
    baseNTrials = 80
    kinWindow = (0, 800e-3)
    kinWindowJitter = 50e-3
    stimWindow = (-50e-3, 700e-3)
    stimWindowJitter = 50e-3
    velocityLookup = {
        ('outbound', 'CW'): 1.,
        ('return', 'CW'): -1.,
        ('outbound', 'CCW'): -1.,
        ('return', 'CCW'): 1.,
        ('NA', 'NA'): 0
        }
    ########################
    protoIndex = []
    stimConfigs = pd.MultiIndex.from_product(
        [conditions['electrode'], conditions['RateInHz'], conditions['amplitude']], names=['electrode', 'RateInHz', 'amplitude'])
    kinConfigs = pd.MultiIndex.from_product(
        [conditions['pedalDirection'], conditions['pedalMovementCat']], names=['pedalDirection', 'pedalMovementCat'])
    for stimConfig in stimConfigs:
        for kinConfig in kinConfigs:
            protoIndex += [stimConfig + kinConfig] * int(baseNTrials * electrodeProportion)
    for stimConfig in stimConfigs:
        NAKinConfig = ('NA', 'NA')
        protoIndex += [stimConfig + NAKinConfig] * int(baseNTrials * electrodeProportion)
    for kinConfig in kinConfigs:
        NAStimConfig = ('NA', 0., 0.)
        protoIndex += [NAStimConfig + kinConfig] * int(baseNTrials)
    NAConfig = ('NA', 0., 0., 'NA', 'NA')
    protoIndex += [NAConfig] * int(baseNTrials)
    toyInfoPerTrial = pd.DataFrame(protoIndex, columns=['electrode', 'RateInHz', 'amplitude', 'pedalDirection', 'pedalMovementCat'])
    toyInfoPerTrial.loc[:, 'bin'] = 0.
    toyInfoPerTrial.loc[:, 'segment'] = 0.
    toyInfoPerTrial.loc[:, 'originalIndex'] = np.arange(toyInfoPerTrial.shape[0])
    toyInfoPerTrial.loc[:, 't'] = toyInfoPerTrial['originalIndex'] * 10.
    for cN in infoPerTrial.columns:
        if cN not in toyInfoPerTrial.columns:
            toyInfoPerTrial.loc[:, cN] = infoPerTrial[cN].iloc[0]
    toyLhsFeatures = ['velocity', 'amplitude', 'RateInHz']
    toyLhsColumns = pd.MultiIndex.from_tuples([
        (col, 0,) + ('NA',) * 4
        for col in toyLhsFeatures
        ], names=lhsDF.columns.names)
    bins = np.unique(lhsDF.index.get_level_values('bin'))
    rng = np.random.default_rng()
    toyLhsList = []
    for rowIdx, row in toyInfoPerTrial.iterrows():
        thisIdx = pd.MultiIndex.from_tuples([tuple(row)], names=row.index)
        vel = pd.DataFrame(0., index=thisIdx, columns=bins)
        vel.columns.name = 'bin'
        thisKW = (kinWindow[0] + rng.random() * kinWindowJitter, kinWindow[1] + rng.random() * kinWindowJitter)
        thisSW = (stimWindow[0] + rng.random() * stimWindowJitter, stimWindow[1] + rng.random() * stimWindowJitter)
        kBinMask = (bins >= thisKW[0]) & (bins < thisKW[1])
        vel.loc[:, kBinMask] = velocityLookup[(row['pedalMovementCat'], row['pedalDirection'])]
        amp = pd.DataFrame(0., index=thisIdx, columns=bins)
        amp.columns.name = 'bin'
        rate = pd.DataFrame(0., index=thisIdx, columns=bins)
        rate.columns.name = 'bin'
        if row['electrode'] != 'NA':
            stimMask = np.asarray((bins >= thisSW[0]) & (bins < thisSW[1]), dtype=float)
            amp.loc[:, :] = stimMask * row['amplitude']
            rate.loc[:, :] = stimMask * row['RateInHz']
        toyLhsList.append(
            pd.concat([
                vel.stack('bin').to_frame(name='velocity'),
                amp.stack('bin').to_frame(name='amplitude'),
                rate.stack('bin').to_frame(name='RateInHz'),
                ], axis='columns'))
    toyLhsDF = pd.concat(toyLhsList)
    splitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'])
    iteratorKWArgs = dict(
        n_splits=7,
        splitterClass=tdr.trialAwareStratifiedKFold, splitterKWArgs=splitterKWArgs,
        samplerKWArgs=dict(random_state=None, test_size=None, ),
        prelimSplitterClass=tdr.trialAwareStratifiedKFold, prelimSplitterKWArgs=splitterKWArgs,
        resamplerClass=None, resamplerKWArgs={},
        )
    cvIterator = tdr.trainTestValidationSplitter(
        dataDF=toyLhsDF, **iteratorKWArgs)
    iteratorInfo = {
        'iteratorKWArgs': iteratorKWArgs,
        'cvIterator': cvIterator
        }
    colsToScale = ['amplitude', 'RateInHz']
    toyLhsDF.reset_index(level=['electrode', 'pedalMovementCat'], inplace=True)
    toyTrialInfo = toyLhsDF.index.to_frame().reset_index(drop=True)
    toyTrialInfo.index.name = 'trial'
    toyLhsDF.reset_index(inplace=True, drop=True)
    toyLhsDF.index.name = 'trial'
    lOfTransformers = [
        (['amplitude'], MinMaxScaler(feature_range=(0, 1)),),
        (['RateInHz'], MinMaxScaler(feature_range=(0, .5)),)
        ]
    for cN in toyLhsDF.columns:
        if cN not in colsToScale:
            lOfTransformers.append(([cN], None,))
    lhsScaler = DataFrameMapper(
        lOfTransformers, input_df=True, df_out=True
        )
    scaledLhsDF = lhsScaler.fit_transform(toyLhsDF)
    designFormula = "velocity + electrode:(amplitude/RateInHz)"
    pt = PatsyTransformer(designFormula, return_type="matrix")
    designMatrix = pt.fit_transform(scaledLhsDF)
    designInfo = designMatrix.design_info
    designDF = (pd.DataFrame(designMatrix, index=toyLhsDF.index, columns=designInfo.column_names))
    ################################
    nDim = 3
    nDimLatent = 2
    #####
    kinDirection = vg.rotate(
        vg.basis.x, vg.basis.z, 10)
    stimDirection = vg.rotate(
        kinDirection, vg.basis.z, 90)
    #####
    mu = np.asarray([1., 2., 3.])
    phi, theta, tau = 30, 70, 20
    Wrot = np.concatenate(
        [
            vg.rotate(vg.rotate(vg.rotate(vg.basis.x, vg.basis.x, phi), vg.basis.y, theta), vg.basis.z, tau).reshape(-1, 1),
            vg.rotate(vg.rotate(vg.rotate(vg.basis.y, vg.basis.x, phi), vg.basis.y, theta), vg.basis.z, tau).reshape(-1, 1)],
        axis=1
        )
    var = np.asarray([2, 5])
    W = Wrot * var
    S = np.eye(nDim)
    #
    gtCoeffs = pd.Series({
        'Intercept': 0.,
        'velocity': 1.,
        'electrode[+ E16 - E9]:amplitude': 1.,
        'electrode[NA]:amplitude': 0.,
        'electrode[+ E16 - E9]:amplitude:RateInHz': 0.,
        'electrode[NA]:amplitude:RateInHz': 0.
        })

    projectionLookup = {
        'Intercept': kinDirection,
        'velocity': kinDirection,
        'electrode:amplitude': stimDirection,
        'electrode:amplitude:RateInHz': stimDirection
        }
    latentNoise = rng.normal(0, 1, size=(toyLhsDF.shape[0], nDimLatent))
    magnitudes = designDF * gtCoeffs
    latentRhsDF = pd.DataFrame(0, index=toyLhsDF.index, columns=range(nDimLatent))
    for axIdx in latentRhsDF.columns:
        for termName, termSlice in designInfo.term_name_slices.items():
            termValues = magnitudes.iloc[:, termSlice].to_numpy() * projectionLookup[termName][axIdx]
            latentRhsDF.loc[:, axIdx] += termValues.sum(axis=1)
    latentRhsDF += latentNoise
    latentRhsDF.columns.name = 'feature'

    latentPlotDF = pd.concat([latentRhsDF, toyLhsDF], axis='columns')
    latentPlotDF.columns = latentPlotDF.columns.astype(str)

    noiseDistr = stats.multivariate_normal(mean=mu, cov=S)
    noiseTerm = noiseDistr.rvs(latentRhsDF.shape[0])
    rhsDF = pd.DataFrame(
        latentRhsDF.to_numpy() @ W.T + noiseTerm,
        index=latentRhsDF.index,
        columns=['{}'.format(cN) for cN in range(nDim)])
    rhsPlotDF = pd.concat([rhsDF, toyLhsDF], axis='columns')
    pdb.set_trace()
    mlab.points3d(rhsPlotDF['0'], rhsPlotDF['1'], rhsPlotDF['2'])