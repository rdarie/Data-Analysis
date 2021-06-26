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

import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'DISPLAY' in os.environ:
    matplotlib.use('QT5Agg')   # generate postscript output
else:
    matplotlib.use('PS')   # generate postscript output
from mpl_toolkits.mplot3d import proj3d
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
    'figure.titlesize': 7,
    'lines.markersize': 12
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
    lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
    lhsMasks = pd.read_hdf(lhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameLhs']))
    #
    origBins = np.unique(lhsDF.index.get_level_values('bin'))
    # bins = origBins
    #
    trialInfo = lhsDF.index.to_frame().reset_index(drop=True)
    infoPerTrial = trialInfo.drop_duplicates(['segment', 'originalIndex', 't'])
    print('Loaded experimental data; ')
    print('Experimental data trial counts, grouped by {}'.format(stimulusConditionNames))
    print(trialInfo.drop_duplicates(['segment', 'originalIndex', 't']).groupby(stimulusConditionNames).count().iloc[:, 0])
    print('Experimental data trial counts, grouped by {}'.format(['electrode', 'pedalMovementCat']))
    print(trialInfo.drop_duplicates(['segment', 'originalIndex', 't']).groupby(['electrode', 'pedalMovementCat']).count().iloc[:, 0])
    ########################################################
    # trial type generation
    #######################################################
    conditions = {
        'pedalDirection': np.asarray(['CW']),
        'pedalMovementCat': np.asarray(['outbound', 'return']),
        'RateInHz': np.asarray([50, 100]),
        'amplitude': np.linspace(200, 800, 5),
        'electrode': np.asarray(['+ E16 - E9', '+ E16 - E5'])
        }
    electrodeRatio = 3.
    naRatio = 1.
    baseNTrials = 100
    bins = np.arange(-100e-3, 510e-3, 10e-3)
    kinWindow = (0., 500e-3)
    kinWindowJitter = 50e-3
    stimWindow = (0., 500e-3)
    stimWindowJitter = 100e-3
    velocityLookup = {
        ('outbound', 'CW'): 1.,
        ('return', 'CW'): -1.,
        ('outbound', 'CCW'): -1.,
        ('return', 'CCW'): 1.,
        ('NA', 'NA'): 0
        }
    ########################
    nAmpRate = conditions['RateInHz'].size * conditions['amplitude'].size
    protoIndex = []
    stimConfigs = pd.MultiIndex.from_product(
        [conditions['electrode'], conditions['RateInHz'], conditions['amplitude']], names=['electrode', 'RateInHz', 'amplitude'])
    kinConfigs = pd.MultiIndex.from_product(
        [conditions['pedalDirection'], conditions['pedalMovementCat']], names=['pedalDirection', 'pedalMovementCat'])
    # add stim movement
    for stimConfig in stimConfigs:
        for kinConfig in kinConfigs:
            protoIndex += [stimConfig + kinConfig] * int(np.ceil(baseNTrials * electrodeRatio / nAmpRate))
    # add stim no-movement
    for stimConfig in stimConfigs:
        NAKinConfig = ('NA', 'NA')
        protoIndex += [stimConfig + NAKinConfig] * int(np.ceil(baseNTrials * electrodeRatio / nAmpRate))
    # add no-stim movement
    for kinConfig in kinConfigs:
        NAStimConfig = ('NA', 0., 0.)
        protoIndex += [NAStimConfig + kinConfig] * int(baseNTrials)
    # add no-stim no-movement
    NAConfig = ('NA', 0., 0., 'NA', 'NA')
    protoIndex += [NAConfig] * int(np.ceil(baseNTrials * naRatio))
    ##
    toyInfoPerTrial = pd.DataFrame(
        protoIndex,
        columns=['electrode', 'RateInHz', 'amplitude', 'pedalDirection', 'pedalMovementCat'])
    rng = np.random.default_rng(seed=42)
    shuffledIndices = np.arange(toyInfoPerTrial.shape[0])
    rng.shuffle(shuffledIndices)
    toyInfoPerTrial = toyInfoPerTrial.iloc[shuffledIndices, :]
    toyInfoPerTrial.loc[:, 'segment'] = 0.
    toyInfoPerTrial.loc[:, 'originalIndex'] = np.arange(toyInfoPerTrial.shape[0])
    toyInfoPerTrial.loc[:, 'trialUID'] = np.arange(toyInfoPerTrial.shape[0])
    toyInfoPerTrial.loc[:, 't'] = toyInfoPerTrial['originalIndex'] * 10.
    for cN in infoPerTrial.columns:
        if cN not in toyInfoPerTrial.columns:
            toyInfoPerTrial.loc[:, cN] = infoPerTrial[cN].iloc[0]
    ##
    print('Created synthetic data trial types; ')
    print('Synthetic data trial counts, grouped by {}'.format(stimulusConditionNames))
    print(toyInfoPerTrial.drop_duplicates(['segment', 'originalIndex', 't']).groupby(stimulusConditionNames).count().iloc[:, 0])
    print('Synthetic data trial counts, grouped by {}'.format(['electrode', 'pedalMovementCat']))
    print(toyInfoPerTrial.drop_duplicates(['segment', 'originalIndex', 't']).groupby(['electrode', 'pedalMovementCat']).count().iloc[:, 0])
    ##
    '''toyLhsFeatures = ['velocity', 'amplitude', 'RateInHz']
    toyLhsColumns = pd.MultiIndex.from_tuples([
        (col, 0,) + ('NA',) * 4
        for col in toyLhsFeatures
        ], names=lhsDF.columns.names)'''
    # bins = np.arange(min(origBins), max(origBins) + 20e-3, 20e-3)
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
    colsToScale = ['amplitude', 'RateInHz']
    saveIndex = toyLhsDF.index.copy()
    # pdb.set_trace()
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
    iteratorSuffix = arguments['iteratorSuffix']
    if iteratorSuffix == 'a':
        restrictMask = (scaledLhsDF['electrode'] == '+ E16 - E9').to_numpy()
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 0)  # parallel
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [80., 40., 0.],  # out of plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        # W = wRot @ var
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 2.,
            #
            'electrode[+ E16 - E5]:amplitude': 5.,
            'electrode[+ E16 - E9]:amplitude': 5.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
    elif iteratorSuffix == 'b':
        restrictMask = (scaledLhsDF['electrode'] == '+ E16 - E9').to_numpy()
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 90)  # perpendicular
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [80., 40., 0.],  # out of plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        # W = wRot @ var
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 2.,
            #
            'electrode[+ E16 - E5]:amplitude': 5.,
            'electrode[+ E16 - E9]:amplitude': 5.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
    elif iteratorSuffix == 'c':
        restrictMask = (scaledLhsDF['electrode'] == '+ E16 - E9').to_numpy()
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 0)  # parallel
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [0., 0., 0.],  # in-plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        # W = wRot @ var
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 2.,
            #
            'electrode[+ E16 - E5]:amplitude': 5.,
            'electrode[+ E16 - E9]:amplitude': 5.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
    elif iteratorSuffix == 'd':
        restrictMask = (scaledLhsDF['electrode'] == '+ E16 - E9').to_numpy()
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 90)  # perpendicular
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [0., 0., 0.],  # in-plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 2.,
            #
            'electrode[+ E16 - E5]:amplitude': 5.,
            'electrode[+ E16 - E9]:amplitude': 5.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
    elif iteratorSuffix == 'e':
        restrictMask = (scaledLhsDF['electrode'] == 'NA').to_numpy()
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 0)  # parallel
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [0., 0., 0.],  # in-plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('xyz', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        # W = wRot @ var
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 4.,
            #
            'electrode[+ E16 - E5]:amplitude': 0.,
            'electrode[+ E16 - E9]:amplitude': 0.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 0.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 0.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
    elif iteratorSuffix == 'f':
        restrictMask = (
            (scaledLhsDF['electrode'] == 'NA') &
            (scaledLhsDF['pedalMovementCat'] != 'NA')).to_numpy()
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 0)  # parallel
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [0., 0., 0.],  # in-plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('xyz', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 0.,
            #
            'electrode[+ E16 - E5]:amplitude': 0.,
            'electrode[+ E16 - E9]:amplitude': 0.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 0.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 0.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
    elif iteratorSuffix == 'g':
        restrictMask = np.ones(scaledLhsDF['electrode'].shape, dtype=bool)
        ################################################################
        nDim = 3
        nDimLatent = 2
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 90)  # perpendicular
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [40., 20., 0.],  # out of plane
            'electrode[+ E16 - E5]:amplitude': [20., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
        })
        #####
        mu = np.asarray([2., 3., 1.])
        phi, theta, psi = 30, 10, 20
        r = Rot.from_euler('XYZ', [phi, theta, psi], degrees=True)
        wRot = r.as_matrix()
        explainedVar = np.diag([5, 3, 0])
        # W = wRot @ var
        S = np.eye(nDim) * 2.
        #
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 2.,
            #
            'electrode[+ E16 - E5]:amplitude': 4.,
            'electrode[+ E16 - E9]:amplitude': 4.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 1.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 1.,
            'electrode[NA]:amplitude:RateInHz': 0.
        })
    #
    ################################################################
    ################################################################
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
    # latentRhsDF.loc[:, :] = StandardScaler().fit_transform(latentRhsDF)
    latentRhsDF.columns.name = 'feature'
    # adjust explained_var to account for std introduced by the regressors
    '''for idx in range(nDimLatent):
        thisStd = latentRhsDF.iloc[:, idx].std()
        latentRhsDF.iloc[:, idx] = latentRhsDF.iloc[:, idx] / thisStd
        # explainedVar[idx] = explainedVar[idx] / thisStd'''
    #
    electrodeInfluence = termMagnitudes['electrode:amplitude'] + termMagnitudes['electrode:amplitude:RateInHz']
    defaultMS = matplotlib.rcParams['lines.markersize'] ** 2
    movementInfluence = pd.Series(
        MinMaxScaler(feature_range=(defaultMS, defaultMS * 2))
        .fit_transform(
            termMagnitudes['velocity']
            .to_numpy().reshape(-1, 1))
        .flatten(), index=termMagnitudes.index)
    #
    latentPlotDF = pd.concat([latentRhsDF, scaledLhsDF], axis='columns')
    latentPlotDF.columns = latentPlotDF.columns.astype(str)
    latentPlotDF.loc[:, 'restrictMask'] = restrictMask
    latentPlotDF.loc[:, 'electrodeInfluence'] = electrodeInfluence
    latentPlotDF.loc[:, 'movementInfluence'] = movementInfluence
    latentPlotDF.loc[:, 'limbState'] = toyLhsDF['velocity'].map({-1: 'flexion', 0:'rest', 1:'extension'}).to_numpy()
    latentPlotDF.loc[:, 'limbState x electrode'] = ' '
    for name, group in latentPlotDF.groupby(['limbState', 'electrode']):
        latentPlotDF.loc[group.index, 'limbState x electrode'] = '{} {}'.format(*name)
    toyRhsDF = pd.DataFrame(
        0.,
        index=latentRhsDF.index,
        columns=['data{}'.format(cN) for cN in range(nDim)])
    for name, group in scaledLhsDF.groupby(['electrode', 'amplitude']):
        elecName, amp = name
        print('Embedding latent data into higher D, electrode {}, ampl {}'.format(*name))
        print('elecName {}, amp {}'.format(elecName, amp))
        factorName = 'electrode[{}]:amplitude'.format(elecName)
        nPoints = group.index.shape[0]
        extraRotMat = Rot.from_euler('XYZ', [beta for beta in rotCoeffs[factorName]], degrees=True).as_matrix()
        augL = np.concatenate([latentRhsDF.loc[group.index, :].to_numpy(), np.zeros([nPoints, 1])], axis=1).T
        toyRhsDF.loc[group.index, :] += (wRot @ extraRotMat @ explainedVar @ augL).T
    #
    noiseDistr = stats.multivariate_normal(mean=mu, cov=S)
    noiseTerm = noiseDistr.rvs(latentRhsDF.shape[0])
    toyRhsDF += noiseTerm
    #
    rhsPlotDF = pd.concat([toyRhsDF, scaledLhsDF], axis='columns')
    #
    rhsPlotDF.loc[:, 'restrictMask'] = restrictMask
    rhsPlotDF.loc[:, 'electrodeInfluence'] = electrodeInfluence
    rhsPlotDF.loc[:, 'movementInfluence'] = movementInfluence
    rhsPlotDF.loc[:, 'limbState'] = latentPlotDF['limbState']
    rhsPlotDF.loc[:, 'limbState x electrode'] = ' '
    for name, group in rhsPlotDF.groupby(['limbState', 'electrode']):
        rhsPlotDF.loc[group.index, 'limbState x electrode'] = '{} {}'.format(*name)
    markerStyles = ['o', 'd', 's']
    msDict = {key: markerStyles[idx] for idx, key in enumerate(latentPlotDF['velocity'].unique())}
    maskOpts = {
        0: dict(alpha=0.2, linewidths=0),
        1: dict(alpha=0.3, linewidths=1)
        }
    lOfMasksForBreakdown = [
        {
            'mask': ((toyTrialInfo['electrode'] == 'NA') & (toyTrialInfo['pedalMovementCat'] == 'NA')).to_numpy(),
            'label': 'At rest, no stim'},
        {
            'mask': (toyTrialInfo['electrode'] == 'NA').to_numpy(),
            'label': 'Movement, no stim'},
        {
            'mask': toyTrialInfo['electrode'].isin(['NA', '+ E16 - E5']).to_numpy(),
            'label': 'Movement, stim 1'},
        {
            'mask': toyTrialInfo['electrode'].isin(['NA', '+ E16 - E9']).to_numpy(),
            'label': 'Movement, stim 2'},
    ]
    pdfPath = os.path.join(
        figureOutputFolder, 'synthetic_dataset_latent_{}.pdf'.format(iteratorSuffix)
        )
        
    with PdfPages(pdfPath) as pdf:
        extentLatent = (
            latentPlotDF.loc[:, latentRhsDF.columns].quantile(1 - 1e-3) -
            latentPlotDF.loc[:, latentRhsDF.columns].quantile(1e-3)).max()
        fig, ax = plt.subplots(figsize=(12, 12))
        for name, group in latentPlotDF.groupby('velocity'):
            ax.scatter(
                group.iloc[:, 0], group.iloc[:, 1], cmap='viridis',
                c=group['electrodeInfluence'],
                # s=group['movementInfluence'],
                marker=msDict[name], rasterized=True, **maskOpts[0])
        xMid = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
        ax.set_xlim([xMid - extentLatent/2, xMid + extentLatent/2])
        yMid = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
        ax.set_ylim([yMid - extentLatent/2, yMid + extentLatent/2])
        fig.tight_layout()
        pdf.savefig()
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.kdeplot(
            x='latent0', y='latent1', hue='limbState x electrode',
            levels=20, fill=True, thresh=.1, palette='Set1',
            common_norm=False, alpha=.5,
            data=latentPlotDF, ax=ax)
        ax.set_xlim([xMid - extentLatent/2, xMid + extentLatent/2])
        ax.set_ylim([yMid - extentLatent/2, yMid + extentLatent/2])
        fig.tight_layout()
        pdf.savefig()
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        if iteratorSuffix in ['a', 'b']:
            for maskDict in lOfMasksForBreakdown:
                thisMask = maskDict['mask']
                fig, ax = plt.subplots(figsize=(12, 12))
                for name, group in latentPlotDF.loc[thisMask, :].groupby('velocity'):
                    ax.scatter(
                        group['latent0'], group['latent1'], cmap='viridis',
                        c=group['electrodeInfluence'],
                        # s=group['movementInfluence'],
                        marker=msDict[name], rasterized=True, **maskOpts[0])
                ax.set_xlim([xMid - extentLatent/2, xMid + extentLatent/2])
                ax.set_ylim([yMid - extentLatent/2, yMid + extentLatent/2])
                fig.suptitle(maskDict['label'])
                fig.tight_layout()
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
    ####
    pdfPath = os.path.join(
        figureOutputFolder, 'synthetic_dataset_{}.pdf'.format(iteratorSuffix)
        )
    with PdfPages(pdfPath) as pdf:
        extent = (
            rhsPlotDF.loc[:, toyRhsDF.columns].quantile(1 - 1e-3) -
            rhsPlotDF.loc[:, toyRhsDF.columns].quantile(1e-3)).max()
        fig = plt.figure()
        fig.set_size_inches((12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.set_proj_type('ortho')
        for name, group in rhsPlotDF.groupby('velocity'):
            ax.scatter(
                group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2], cmap='plasma',
                c=group['electrodeInfluence'],
                # s=group['movementInfluence'],
                marker=msDict[name], rasterized=True, **maskOpts[0])
        #
        xMid = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
        ax.set_xlim([xMid - extent/2, xMid + extent/2])
        #
        yMid = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
        ax.set_ylim([yMid - extent/2, yMid + extent/2])
        #
        zMid = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
        ax.set_xlim([zMid - extent/2, zMid + extent/2])
        #
        ax.view_init(azim=-120., elev=30.)
        fig.tight_layout()
        pdf.savefig()
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        #
        rhsPlotDF.loc[:, 'x2'], rhsPlotDF.loc[:, 'y2'], _ = proj3d.proj_transform(
            rhsPlotDF.iloc[:, 0], rhsPlotDF.iloc[:, 1], rhsPlotDF.iloc[:, 2], ax.get_proj())
        #####
        ####
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.kdeplot(
            x='x2', y='y2', hue='limbState x electrode',
            levels=20, fill=True, thresh=.1, palette='Set2',
            common_norm=False, linewidth=0., alpha=.5,
            data=rhsPlotDF, ax=ax)
        fig.tight_layout()
        pdf.savefig()
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        if iteratorSuffix in ['a', 'b']:
            tempCols = [cN for cN in toyRhsDF.columns] + ['limbState x electrode']
            rhsPlotDFStack = pd.DataFrame(
                rhsPlotDF.loc[:, tempCols].to_numpy(),
                columns=tempCols,
                index=pd.MultiIndex.from_frame(toyTrialInfo.loc[:, ['electrode', 'bin', 'pedalMovementCat', 'amplitude']])
                )
            rhsPlotDFStack = rhsPlotDFStack.set_index('limbState x electrode', append=True)
            rhsPlotDFStack.columns.name = 'feature'
            rhsPlotDFStack = rhsPlotDFStack.stack().to_frame(name='signal').reset_index()
            g = sns.relplot(
                row='feature', col='limbState x electrode',
                x='bin', y='signal', hue='amplitude',
                data=rhsPlotDFStack, palette='plasma',
                errorbar='se', kind='line',
                height=2, aspect=3
                )
            g.tight_layout()
            pdf.savefig()
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            g = sns.displot(
                row='feature', col='limbState x electrode',
                y='signal', hue='amplitude',
                data=rhsPlotDFStack, palette='plasma',
                kind='kde', common_norm=False,
                height=2, aspect=3
                )
            g.tight_layout()
            pdf.savefig()
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            for maskDict in lOfMasksForBreakdown:
                fig = plt.figure()
                fig.set_size_inches((12, 12))
                ax = fig.add_subplot(projection='3d')
                ax.set_proj_type('ortho')
                thisMask = maskDict['mask']
                for name, group in rhsPlotDF.loc[thisMask, :].groupby('velocity'):
                    ax.scatter(
                        group['data0'], group['data1'], group['data2'], cmap='plasma',
                        c=group['electrodeInfluence'],
                        # s=group['movementInfluence'],
                        marker=msDict[name], rasterized=True, **maskOpts[0])
                #
                ax.set_xlim([xMid - extent/2, xMid + extent/2])
                ax.set_ylim([yMid - extent/2, yMid + extent/2])
                ax.set_xlim([zMid - extent/2, zMid + extent/2])
                #
                ax.view_init(azim=-120., elev=30.)
                fig.suptitle(maskDict['label'])
                fig.tight_layout()
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
    ####
    outputDatasetName = 'Synthetic_{}_df_{}'.format(loadingMeta['arguments']['window'], iteratorSuffix)
    outputLoadingMeta = loadingMeta.copy()
    outputLoadingMetaPath = os.path.join(
        dataFramesFolder,
        outputDatasetName + '_' + arguments['selectionNameRhs'] + '_meta.pickle'
        )
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
        dataDF=toyLhsDF.loc[restrictMask, :], **iteratorKWArgs)
    #
    outputLoadingMeta['iteratorsBySegment'] = [cvIterator]
    outputLoadingMeta['cv_kwargs'] = iteratorKWArgs
    outputLoadingMeta.pop('normalizationParams')
    outputLoadingMeta.pop('normalizeDataset')
    outputLoadingMeta.pop('unNormalizeDataset')
    outputLoadingMeta.pop('arguments')
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
        dataDF=toyRhsDF.loc[restrictMask, :], loadingMeta=outputLoadingMeta.copy(), featureInfoMask=maskDF,
        arguments=arguments, selectionName=arguments['selectionNameRhs'],
        dataFramesFolder=dataFramesFolder, datasetName=outputDatasetName,
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
        dataDF=toyLhsDF.loc[restrictMask, :], loadingMeta=outputLoadingMeta.copy(), featureInfoMask=maskDF,
        arguments=arguments, selectionName=arguments['selectionNameLhs'],
        dataFramesFolder=dataFramesFolder, datasetName=outputDatasetName,
        )
