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
    --iteratorSuffix=iteratorSuffix        filename if using a unit selector
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
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
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
from scipy.spatial.transform import Rotation as Rot
for arg in sys.argv:
    print(arg)

idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
    }
mplRCParams = {
    'figure.titlesize': 7
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

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
            arguments['analysisName'], 'dimensionality')
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
        'electrode': np.asarray(['+ E16 - E9', '+ E16 - E5'])
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
    rng = np.random.default_rng()
    shuffledIndices = np.arange(toyInfoPerTrial.shape[0])
    rng.shuffle(shuffledIndices)
    toyInfoPerTrial = toyInfoPerTrial.iloc[shuffledIndices, :]
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
    toyLhsList = []
    toyInfoPerTrial.drop(columns=['bin'], inplace=True)
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
    saveIndex = toyLhsDF.index.copy()
    toyTrialInfo = toyLhsDF.index.to_frame().reset_index(drop=True)
    lOfTransformers = [
        (['amplitude'], MinMaxScaler(feature_range=(0., 1)),),
        (['RateInHz'], MinMaxScaler(feature_range=(0., .5)),)
        ]
    for cN in toyLhsDF.columns:
        if cN not in colsToScale:
            lOfTransformers.append(([cN], None,))
    lhsScaler = DataFrameMapper(
        lOfTransformers, input_df=True, df_out=True
        )
    scaledLhsDF = lhsScaler.fit_transform(toyLhsDF)
    #
    scaledLhsDF.reset_index(level=['electrode', 'pedalMovementCat'], inplace=True)
    scaledLhsDF.reset_index(inplace=True, drop=True)
    scaledLhsDF.index.name = 'trial'
    #
    designFormula = "velocity + electrode:(amplitude/RateInHz)"
    pt = PatsyTransformer(designFormula, return_type="matrix")
    designMatrix = pt.fit_transform(scaledLhsDF)
    designInfo = designMatrix.design_info
    designDF = (
        pd.DataFrame(
            designMatrix,
            index=toyLhsDF.index,
            columns=designInfo.column_names))
    ################################
    iteratorSuffix = arguments['iteratorSuffix']
    if iteratorSuffix == 'a':
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 90)
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        var = np.diag([2, 7, 0])
        # W = wRot @ var
        S = np.eye(nDim) * .5e-2
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 3.,
            #
            'electrode[+ E16 - E5]:amplitude': 4.,
            'electrode[+ E16 - E9]:amplitude': 4.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [60., 5., 0.],
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        projectionLookup = {
            'Intercept': kinDirection,
            'velocity': kinDirection,
            'electrode:amplitude': stimDirection,
            'electrode:amplitude:RateInHz': stimDirection
            }
    elif iteratorSuffix == 'b':
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 10)
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        var = np.diag([2, 7, 0])
        # W = wRot @ var
        S = np.eye(nDim) * .5e-2
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 3.,
            #
            'electrode[+ E16 - E5]:amplitude': 4.,
            'electrode[+ E16 - E9]:amplitude': 4.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [80., 40., 0.],
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        projectionLookup = {
            'Intercept': kinDirection,
            'velocity': kinDirection,
            'electrode:amplitude': stimDirection,
            'electrode:amplitude:RateInHz': stimDirection
            }
    elif iteratorSuffix == 'c':
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 10)
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        var = np.diag([2, 7, 0])
        # W = wRot @ var
        S = np.eye(nDim) * .5e-2
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 3.,
            #
            'electrode[+ E16 - E5]:amplitude': 4.,
            'electrode[+ E16 - E9]:amplitude': 4.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [0., 0., 0.],
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        projectionLookup = {
            'Intercept': kinDirection,
            'velocity': kinDirection,
            'electrode:amplitude': stimDirection,
            'electrode:amplitude:RateInHz': stimDirection
            }
    magnitudes = (designDF * gtCoeffs).loc[:, designDF.columns]
    # sanity check
    sanityCheckThis = False
    if sanityCheckThis:
        fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        for cN in magnitudes.columns:
            ax[0].plot(magnitudes[cN], label=cN)
        ax[0].legend()
    latentRhsDF = pd.DataFrame(
        0, index=scaledLhsDF.index,
        columns=['latent{}'.format(cN) for cN in range(nDimLatent)])
    termMagnitudes = pd.DataFrame(
        0, index=scaledLhsDF.index, columns=designInfo.term_names
        )
    for termName, termSlice in designInfo.term_name_slices.items():
        termMagnitudes.loc[:, termName] = magnitudes.iloc[:, termSlice].sum(axis='columns').to_numpy()
    # sanity check
    if sanityCheckThis:
        for cN in termMagnitudes.columns:
            ax[1].plot(termMagnitudes[cN], label=cN)
        ax[1].legend()
        plt.show()
    for termName, termSlice in designInfo.term_name_slices.items():
        termValues = termMagnitudes[termName].to_numpy().reshape(-1, 1) * projectionLookup[termName]
        latentRhsDF.loc[:, :] += termValues[:, :nDimLatent]
    latentNoise = rng.normal(0, 1., size=(scaledLhsDF.shape[0], nDimLatent))
    latentRhsDF += latentNoise
    latentRhsDF.loc[:, :] = StandardScaler().fit_transform(latentRhsDF)
    latentRhsDF.columns.name = 'feature'
    #
    latentPlotDF = pd.concat([latentRhsDF, scaledLhsDF], axis='columns')
    latentPlotDF.columns = latentPlotDF.columns.astype(str)
    #
    toyRhsDF = pd.DataFrame(
        0.,
        index=latentRhsDF.index,
        columns=['data{}'.format(cN) for cN in range(nDim)])
    for name, group in scaledLhsDF.groupby(['electrode', 'amplitude']):
        elecName, amp = name
        print('elecName {}, amp {}'.format(elecName, amp))
        factorName = 'electrode[{}]:amplitude'.format(elecName)
        nPoints = group.index.shape[0]
        extraRotMat = Rot.from_euler('XYZ', [beta * amp for beta in rotCoeffs[factorName]], degrees=True).as_matrix()
        augL = np.concatenate([latentRhsDF.loc[group.index, :].to_numpy(), np.zeros([nPoints, 1])], axis=1).T
        toyRhsDF.loc[group.index, :] += (wRot @ extraRotMat @ var @ augL).T
    noiseDistr = stats.multivariate_normal(mean=mu, cov=S)
    noiseTerm = noiseDistr.rvs(latentRhsDF.shape[0])
    toyRhsDF += noiseTerm
    toyRhsDF += mu
    rhsPlotDF = pd.concat([toyRhsDF, scaledLhsDF], axis='columns')
    totalBoundsLatent = np.abs(
        (latentPlotDF.loc[:, latentRhsDF.columns].quantile(5e-3).min(),
        latentPlotDF.loc[:, latentRhsDF.columns].quantile(1 - 5e-3).max(),)).max() * np.asarray([-1, 1])
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(
        latentPlotDF.iloc[:, 0], latentPlotDF.iloc[:, 1],
        c=latentPlotDF['amplitude'], cmap='viridis', alpha=0.1, linewidths=0, rasterized=True)
    ax.set_xlim(totalBoundsLatent)
    ax.set_ylim(totalBoundsLatent)
    ax.axis('square')
    pdfPath = os.path.join(
        figureOutputFolder, 'synthetic_dataset_latent_{}.pdf'.format(iteratorSuffix)
    )
    fig.tight_layout()
    plt.savefig(pdfPath)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
    ####
    totalBounds = np.abs(
        (
            rhsPlotDF.loc[:, toyRhsDF.columns].quantile(5e-3).min(),
            rhsPlotDF.loc[:, toyRhsDF.columns].quantile(1 - 5e-3).max(),)).max() * np.asarray([-1, 1])
    fig = plt.figure()
    fig.set_size_inches((12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        rhsPlotDF.iloc[:, 0], rhsPlotDF.iloc[:, 1], rhsPlotDF.iloc[:, 2],
        c=rhsPlotDF['amplitude'], cmap='viridis', alpha=.1, linewidths=0, rasterized=True)
    ax.set_xlim(totalBounds)
    ax.set_ylim(totalBounds)
    ax.set_zlim(totalBounds)
    ax.view_init(elev=7., azim=-5.)
    pdfPath = os.path.join(
        figureOutputFolder, 'synthetic_dataset_{}.pdf'.format(iteratorSuffix)
    )
    fig.tight_layout()
    plt.savefig(pdfPath)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
    ####
    outputLoadingMeta = loadingMeta.copy()
    outputLoadingMeta['iteratorsBySegment'] = [cvIterator]
    outputLoadingMeta['cv_kwargs'] = iteratorKWArgs
    outputLoadingMeta.pop('normalizationParams')
    outputLoadingMeta.pop('normalizeDataset')
    outputLoadingMeta.pop('unNormalizeDataset')
    outputLoadingMeta.pop('arguments')
    datasetName='Synthetic_{}_df_{}'.format(loadingMeta['arguments']['window'], iteratorSuffix)
    #####
    featureColumns = pd.DataFrame(
        np.nan,
        index=range(toyRhsDF.shape[1]),
        columns=rhsDF.columns.names)
    for fcn in rhsDF.columns.names:
        if fcn == 'feature':
            featureColumns.loc[:, fcn] = toyRhsDF.columns
        elif fcn == 'lag':
            featureColumns.loc[:, fcn] = 0
        else:
            featureColumns.loc[:, fcn] = ns5.metaFillerLookup[fcn]
    toyRhsDF.columns = pd.MultiIndex.from_frame(featureColumns)
    toyRhsDF.index = saveIndex
    allGroupIdx = pd.MultiIndex.from_tuples(
        [tuple('all' for fgn in rhsDF.columns.names)],
        names=rhsDF.columns.names)
    allMask = pd.Series(True, index=toyRhsDF.columns).to_frame()
    allMask.columns = allGroupIdx
    maskDF = allMask.T
    maskParams = [
        {k: v for k, v in zip(maskDF.index.names, idxItem)}
        for idxItem in maskDF.index
        ]
    maskParamsStr = [
        '{}'.format(idxItem).replace("'", '')
        for idxItem in maskParams]
    maskDF.loc[:, 'maskName'] = maskParamsStr
    maskDF.set_index('maskName', append=True, inplace=True)
    hf.exportNormalizedDataFrame(
        dataDF=toyRhsDF, loadingMeta=outputLoadingMeta.copy(), featureInfoMask=maskDF,
        arguments=arguments, selectionName=arguments['selectionNameRhs'],
        dataFramesFolder=dataFramesFolder, datasetName=datasetName,
        )
    ###########
    featureColumns = pd.DataFrame(
        np.nan,
        index=range(toyLhsDF.shape[1]),
        columns=lhsDF.columns.names)
    for fcn in lhsDF.columns.names:
        if fcn == 'feature':
            featureColumns.loc[:, fcn] = toyLhsDF.columns
        elif fcn == 'lag':
            featureColumns.loc[:, fcn] = 0
        else:
            featureColumns.loc[:, fcn] = ns5.metaFillerLookup[fcn]
    toyLhsDF.columns = pd.MultiIndex.from_frame(featureColumns)
    toyLhsDF.index = saveIndex
    allGroupIdx = pd.MultiIndex.from_tuples(
        [tuple('all' for fgn in lhsDF.columns.names)],
        names=lhsDF.columns.names)
    allMask = pd.Series(True, index=toyLhsDF.columns).to_frame()
    allMask.columns = allGroupIdx
    maskDF = allMask.T
    maskParams = [
        {k: v for k, v in zip(maskDF.index.names, idxItem)}
        for idxItem in maskDF.index
        ]
    maskParamsStr = [
        '{}'.format(idxItem).replace("'", '')
        for idxItem in maskParams]
    maskDF.loc[:, 'maskName'] = maskParamsStr
    maskDF.set_index('maskName', append=True, inplace=True)
    hf.exportNormalizedDataFrame(
        dataDF=toyLhsDF, loadingMeta=outputLoadingMeta.copy(), featureInfoMask=maskDF,
        arguments=arguments, selectionName=arguments['selectionNameLhs'],
        dataFramesFolder=dataFramesFolder, datasetName=datasetName,
        )
    '''
    def exportNormalized(
        dataDF=None, selectionName=None,
        dataFramesFolder=None, datasetName=None,
        ):
        # save, copied from assemble dataframes
        finalDF = dataDF.copy()
        #  #### end of data loading stuff
        if 'spectral' in selectionName:
            normalizationParams = [[], []]
            for expName, dataGroup in dataDF.groupby('expName'):
                for featName, subGroup in dataGroup.groupby('feature', axis='columns'):
                    print('calculating pre-normalization params, exp: {}, feature: {}'.format(expName, featName))
                    meanLevel = np.nanmean(subGroup.xs(0, level='lag', axis='columns').to_numpy())
                    normalizationParams[0].append({
                        'expName': expName,
                        'feature': featName,
                        'mu': meanLevel,
                    })
                    # finalDF.loc[subGroup.index, subGroup.columns] = dataDF.loc[subGroup.index, subGroup.columns] - meanLevel
                    finalDF.loc[subGroup.index, subGroup.columns] = np.sqrt(dataDF.loc[subGroup.index, subGroup.columns] / meanLevel)
            intermediateDF = finalDF.copy()
            for featName, dataGroup in intermediateDF.groupby('feature', axis='columns'):
                print('calculating final normalization params, feature: {}'.format(featName))
                refData = dataGroup.xs(0, level='lag', axis='columns').to_numpy()
                mu = np.nanmean(refData)
                sigma = np.nanstd(refData)
                normalizationParams[1].append({
                    'feature': featName,
                    'mu': mu,
                    'sigma': sigma
                })
                #
                finalDF.loc[:, dataGroup.columns] = (intermediateDF[dataGroup.columns] - mu) / sigma
            #
            def normalizeDataset(inputDF, params):
                outputDF = inputDF.copy()
                for preParams in params[0]:
                    expMask = inputDF.index.get_level_values('expName') == preParams['expName']
                    featMask = inputDF.columns.get_level_values('feature') == preParams['feature']
                    if expMask.any() and featMask.any():
                        print('pre-normalizing exp {}: feature {}'.format(preParams['expName'], preParams['feature']))
                        # outputDF.loc[expMask, featMask] = inputDF.loc[expMask, featMask] - preParams['mu']
                        outputDF.loc[expMask, featMask] = np.sqrt(inputDF.loc[expMask, featMask] / preParams['mu'])
                intermediateDF = outputDF.copy()
                for postParams in params[1]:
                    featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                    if featMask.any():
                        print('final normalizing feature {}'.format(postParams['feature']))
                        outputDF.loc[:, featMask] = (intermediateDF.loc[:, featMask] - postParams['mu']) / postParams['sigma']
                return outputDF
            #
            def unNormalizeDataset(inputDF, params):
                outputDF = inputDF.copy()
                for postParams in params[1]:
                    featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                    if featMask.any():
                        print('pre un-normalizing feature {}'.format(postParams['feature']))
                        outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] * postParams['sigma']) + postParams['mu']
                intermediateDF = outputDF.copy()
                for preParams in params[0]:
                    expMask = inputDF.index.get_level_values('expName') == preParams['expName']
                    featMask = inputDF.columns.get_level_values('feature') == preParams['feature']
                    if expMask.any() and featMask.any():
                        print('final un-normalizing exp {}: feature {}'.format(preParams['expName'], preParams['feature']))
                        # outputDF.loc[expMask, featMask] = intermediateDF.loc[expMask, featMask] + preParams['mu']
                        outputDF.loc[expMask, featMask] = intermediateDF.loc[expMask, featMask] ** 2 * preParams['mu']
                return outputDF
            #
            # finalDF = normalizeDataset(finalDF, normalizationParams)
        else:
            # normal time domain data
            normalizationParams = [[]]
            for featName, dataGroup in dataDF.groupby('feature', axis='columns'):
                refData = dataGroup.xs(0, level='lag', axis='columns').to_numpy()
                print('calculating normalization params for {}'.format(featName))
                mu = np.nanmean(refData)
                sigma = np.nanstd(refData)
                print('mu = {} sigma = {}'.format(mu, sigma))
                normalizationParams[0].append({
                    'feature': featName,
                    'mu': mu,
                    'sigma': sigma
                })
                finalDF.loc[:, dataGroup.columns] = (dataDF[dataGroup.columns] - mu) / sigma
            #
            def normalizeDataset(inputDF, params):
                outputDF = inputDF.copy()
                for postParams in params[0]:
                    featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                    if featMask.any():
                        print('normalizing feature {}'.format(postParams['feature']))
                        print('mu = {} sigma = {}'.format(postParams['mu'], postParams['sigma']))
                        outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] - postParams['mu']) / postParams['sigma']
                return outputDF
            #
            def unNormalizeDataset(inputDF, params):
                outputDF = inputDF.copy()
                for postParams in params[0]:
                    featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                    if featMask.any():
                        print('un-normalizing feature {}'.format(postParams['feature']))
                        print('mu = {} sigma = {}'.format(postParams['mu'], postParams['sigma']))
                        outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] * postParams['sigma']) + postParams['mu']
                return outputDF
            #
            # finalDF = normalizeDataset(finalDF, normalizationParams)
        datasetPath = os.path.join(
            dataFramesFolder,
            datasetName + '.h5'
            )
        print('saving {} to {}'.format(selectionName, datasetPath))
        finalDF.to_hdf(datasetPath, '/{}/data'.format(selectionName), mode='a')
        thisMask.to_hdf(datasetPath, '/{}/featureMasks'.format(selectionName), mode='a')
        #
        loadingMetaPath = os.path.join(
            dataFramesFolder,
            datasetName + '_' + selectionName + '_meta.pickle'
            )
        if os.path.exists(loadingMetaPath):
            os.remove(loadingMetaPath)
        loadingMeta['arguments'] = arguments.copy()
        loadingMeta['normalizationParams'] = normalizationParams
        loadingMeta['normalizeDataset'] = normalizeDataset
        loadingMeta['unNormalizeDataset'] = unNormalizeDataset
        with open(loadingMetaPath, 'wb') as f:
            pickle.dump(loadingMeta, f)'''