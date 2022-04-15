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
    --preScale                             restrict datasets for debugging? [default: False]
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
from mpl_toolkits.mplot3d import proj3d, axes3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.transforms as transforms
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
# import dataAnalysis.helperFunctions.profiling as prf
# import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
# from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
import vg
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklego.preprocessing import PatsyTransformer
from sklearn_pandas import DataFrameMapper
# from sklearn.svm import LinearSVR
# from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
# import joblib as jb
import dill as pickle
import sys
# import gc
from copy import copy
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
# from itertools import product
from scipy import stats
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
for arg in sys.argv:
    print(arg)

idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .25,
        'lines.markersize': 2.5,
        'patch.linewidth': .25, # snsRCParams
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
        "legend.frameon": True,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.subplot.left': 0.01,
    'figure.subplot.right': 0.99,
    'figure.subplot.bottom': 0.01,
    'figure.subplot.top': 0.99,
    'figure.titlesize': 7,
    'axes.titlepad': 3.5,
    'axes.labelpad': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 2e-1,  # units of font size
    'panel_heading.pad': 0.
    }

plt.style.use('seaborn-white')
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV
#
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
    defaultSamplerKWArgs = dict(random_state=42, test_size=0.5)
    defaultPrelimSamplerKWArgs = dict(random_state=42, test_size=0.1)
    # args for tdr.
    defaultSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=defaultSamplerKWArgs)
    defaultPrelimSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=defaultPrelimSamplerKWArgs)
    splitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'])
    cvKWArgs = dict(
        n_splits=3,
        splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
        prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
        resamplerClass=None, resamplerKWArgs={},
        )
    iteratorOpts = {
        'ensembleHistoryLen': .30,
        'covariateHistoryLen': .50,
        'nHistoryBasisTerms': 1,
        'nCovariateBasisTerms': 1,
        'forceBinInterval': 5e-3,
        'minBinCount': 5,
        'calcTimeROI': True,
        'controlProportion': None,
        'cvKWArgs': cvKWArgs,
        'timeROIOpts': {
            'alignQuery': 'startingOrStimOn',
            'winStart': -200e-3,
            'winStop': 400e-3
        },
        'timeROIOpts_control': {
            'alignQuery': None,
            'winStart': -800e-3,
            'winStop': -200.
        }
    }
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
    # print('Loaded experimental data; ')
    # print('Experimental data trial counts, grouped by {}'.format(stimulusConditionNames))
    # print(trialInfo.drop_duplicates(['segment', 'originalIndex', 't']).groupby(stimulusConditionNames).count().iloc[:, 0])
    # print('Experimental data trial counts, grouped by {}'.format(['electrode', 'pedalMovementCat']))
    # print(trialInfo.drop_duplicates(['segment', 'originalIndex', 't']).groupby(['electrode', 'pedalMovementCat']).count().iloc[:, 0])
    ########################################################
    # trial type generation
    #######################################################
    conditions = {
        'pedalDirection': np.asarray(['CW']),
        'pedalMovementCat': np.asarray(['outbound', 'return']),
        'pedalSizeCat': np.asarray(['L']),
        'trialRateInHz': np.asarray([50, 100]),
        'trialAmplitude': np.linspace(200, 800, 10),
        'electrode': np.asarray(['+ E16 - E9', '+ E16 - E5'])
        }
    colorMaps = {
        # coldish
        'covMat': 'crest_r',
        'points': 'deep',
        # warmish
        'distMat': 'flare_r',
        'distHist': 'muted'
    }
    iteratorDescriptions = pd.Series({
        'ca': 'B',
        'cb': 'M',
        'ccs': 'S',
        'ccm': 'C',
        'ra': 'A'
    })
    prettyNameLookup.update({
        'data{}'.format(i): 'LFP Ch. #{}'.format(i)
        for i in range(3)
        })
    prettyNameLookup.update({
        'latent{}'.format(i): 'Latent feature #{}'.format(i)
        for i in range(3)
        })
    prettyNameLookup.update({
        'amplitude': 'Stim.\namplitude',
        'velocity': 'Pedal\nangular\nvelocity',
        'mahalanobis': 'Mh. dist.',
    })
    pickingColors = False
    singlesPalette = pd.DataFrame(
        [
            sns.color_palette(colorMaps['points'])[ii]
            for ii in [7, 3, 0, 2, 5]],
        columns=['r', 'g', 'b'],
        index=iteratorDescriptions.to_list())
    if pickingColors:
        sns.palplot(singlesPalette.apply(
            lambda x: tuple(x), axis='columns'))
        palAx = plt.gca()
        for tIdx, tN in enumerate(singlesPalette.index):
            palAx.text(tIdx, .5, '{}'.format(tN), fontsize=10)
    # sns.palplot(sns.color_palette('Set3')); plt.show()
    covPatternPalette = {
        'baseline NA': singlesPalette.loc['B', :].to_list(),
        'baseline + E16 - E5': sns.utils.alter_color(
            singlesPalette.loc['S', :].to_list(), l=-.3),
        'baseline + E16 - E9': sns.utils.alter_color(
            singlesPalette.loc['S', :].to_list(), l=.3),
        'moving NA': singlesPalette.loc['M', :].to_list(),
        'moving + E16 - E5': sns.utils.alter_color(
            singlesPalette.loc['C', :].to_list(), l=-.3),
        'moving + E16 - E9': sns.utils.alter_color(
            singlesPalette.loc['C', :].to_list(), l=.3),
        }
    prettyNameLookup.update({
        'baseline NA': 'Baseline',
        'baseline + E16 - E5': 'Stim.-only (electrode X)',
        'baseline + E16 - E9': 'Stim.-only (electrode Y)',
        'moving NA': 'Movement-only',
        'moving + E16 - E5': 'Stim.-movement (electrode X)',
        'moving + E16 - E9': 'Stim.-movement (electrode Y)',
        })
    limbStateElectrodeMarkerDict = {
        'baseline NA': r'$\clubsuit$',
        'baseline + E16 - E5': 'o',
        'baseline + E16 - E9': 'd',
        'moving NA': 's',
        'moving + E16 - E5': '*',
        'moving + E16 - E9': 'D',
        }
    limbStateElectrodeNames = [
        'baseline NA', 'moving NA', 'baseline + E16 - E5', 'baseline + E16 - E9',
        'moving + E16 - E5', 'moving + E16 - E9'
    ]
    #
    defaultAxisOrientation = dict(azim=75., elev=20.)
    electrodeRatio = 3.
    naRatio = 1.
    baseNTrials = 50
    #
    #
    nDim = 3
    rotationOffset = 0 * np.asarray([1., 1., 1.]).reshape(3, 1)
    #
    latentMu = 0. * np.asarray([1., 1., 1.])
    latentNoiseStd = 1.
    latentCovariance = np.eye(nDim) * (latentNoiseStd ** 2)
    #
    mu = np.asarray([0., 0., 0.])
    residualStd = 0.
    residualCovariance = np.eye(nDim) * residualStd ** 2
    #
    eulerOrder = 'XYZ'
    phi, theta, psi = 0., 30., 45.
    r = Rot.from_euler(eulerOrder, [phi, theta, psi], degrees=True)
    wRot = r.as_matrix()
    explainedVar = np.diag([12, 3, 1]) ** 2
    #
    bins = np.arange(-100e-3, 510e-3, iteratorOpts['forceBinInterval'])
    kinWindow = (0., 500e-3)
    kinWindowJitter = 50e-3
    kinWindowRamp = 150e-3
    stimWindow = (0., 500e-3)
    stimWindowJitter = 100e-3
    stimWindowRamp = 50e-3
    velocityLookup = {
        ('outbound', 'CW'): 1.,
        ('return', 'CW'): -1.,
        ('outbound', 'CCW'): -1.,
        ('return', 'CCW'): 1.,
        ('NA', 'NA'): 0
        }
    ampJitter = 100
    ampPaletteStr = "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    mahalPaletteStr = "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1"
    lfpColor = np.asarray(sns.color_palette('Set3')[4])
    mhDistColor = np.asarray(sns.color_palette('Set3')[6])
    markerStyles = ['o', 'd', 's']
    scatterOpts = dict(alpha=0.5, linewidths=0, s=.8 ** 2)
    ellipsoidPlotOpts = dict(
        rstride=3, cstride=3,
        alpha=0.1, linewidths=0, antialiased=True)
    #
    legendData = pd.Series([])
    legendData.loc['pcaDir0'] = Line2D(
        [0], [0], linestyle='-', linewidth=1, color='c',
        label='Principle direction #1')
    legendData.loc['pcaDir1'] = Line2D(
        [0], [0], linestyle='-', linewidth=1, color='m',
        label='Principle direction #2')
    legendData.loc['pcaDir2'] = Line2D(
        [0], [0], linestyle='-', linewidth=1, color='y',
        label='Principle direction #3')
    legendPcaDirs = ['pcaDir0', 'pcaDir1', 'pcaDir2']
    #
    legendData.loc['data'] = Line2D(
        [0], [0], linestyle='-', linewidth=2, color=lfpColor,
        label='LFP Ch.')
    legendData.loc['mahalanobis'] = Line2D(
        [0], [0], linestyle='-', linewidth=2, color=mhDistColor,
        label='Mh. Dist.')
    #
    legendData.loc['blank'] = Patch(alpha=0, linewidth=0, label=' ')
    #
    for key, color in covPatternPalette.items():
        legendData[key] = Patch(
            facecolor=color, label=prettyNameLookup[key])

    legendData.loc['ellipsoid'] = Patch(
        alpha=0, linewidth=0, label='Covariance ellipsoid')
    ########################
    nAmpRate = conditions['trialRateInHz'].size * conditions['trialAmplitude'].size
    protoIndex = []
    stimConfigs = pd.MultiIndex.from_product(
        [
            conditions['electrode'], conditions['trialAmplitude'], conditions['trialRateInHz'],],
        names=['electrode', 'trialAmplitude', 'trialRateInHz',])
    kinConfigs = pd.MultiIndex.from_product(
        [
            conditions['pedalMovementCat'], conditions['pedalDirection'], conditions['pedalSizeCat'],],
        names=['pedalMovementCat', 'pedalDirection', 'pedalSizeCat'])
    # add stim movement
    for stimConfig in stimConfigs:
        for kinConfig in kinConfigs:
            protoIndex += [stimConfig + kinConfig] * int(np.ceil(baseNTrials * electrodeRatio / nAmpRate))
    # add stim no-movement
    for stimConfig in stimConfigs:
        NAKinConfig = ('NA', 'NA', 'NA')
        protoIndex += [stimConfig + NAKinConfig] * int(np.ceil(baseNTrials * electrodeRatio / nAmpRate))
    # add no-stim movement
    for kinConfig in kinConfigs:
        NAStimConfig = ('NA', 0., 0.)
        protoIndex += [NAStimConfig + kinConfig] * int(baseNTrials)
    # add no-stim no-movement
    NAConfig = ('NA', 0., 0., 'NA', 'NA', 'NA')
    protoIndex += [NAConfig] * int(np.ceil(baseNTrials * naRatio))
    ##
    toyInfoPerTrial = pd.DataFrame(
        protoIndex, columns=stimulusConditionNames)
    rng = np.random.default_rng(seed=42)
    shuffledIndices = np.arange(toyInfoPerTrial.shape[0])
    rng.shuffle(shuffledIndices)
    toyInfoPerTrial = toyInfoPerTrial.iloc[shuffledIndices, :]
    toyInfoPerTrial.loc[:, 'segment'] = 0.
    toyInfoPerTrial.loc[:, 'originalIndex'] = np.arange(toyInfoPerTrial.shape[0])
    toyInfoPerTrial.loc[:, 'trialUID'] = np.arange(toyInfoPerTrial.shape[0])
    toyInfoPerTrial.loc[:, 't'] = toyInfoPerTrial['originalIndex'] * 10.
    #
    conditionUID = pd.Series(np.nan, index=toyInfoPerTrial.index)
    for name, group in toyInfoPerTrial.groupby(stimulusConditionNames):
        for uid, (_, subGroup) in enumerate(group.groupby('trialUID')):
            conditionUID.loc[subGroup.index] = uid
    toyInfoPerTrial.loc[:, 'conditionUID'] = conditionUID
    #
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

    def velRamp(t, kw, rt, pk):
        if t < kw[0]:
            return 0
        elif (t >= kw[0]) and (t < (kw[0] + rt)):
            return pk * (t - kw[0]) / rt
        elif (t >= (kw[0] + rt)) and (t < (kw[1] - rt)):
            return pk
        elif (t >= (kw[1] - rt)) and (t < kw[1]):
            return pk * (1 - (t - kw[1] + rt) / rt)
        else:
            return 0

    toyLhsList = []
    toyInfoPerTrial.drop(columns=['bin'], inplace=True)
    for rowIdx, row in toyInfoPerTrial.iterrows():
        thisIdx = pd.MultiIndex.from_tuples([tuple(row)], names=row.index)
        thisKW = (
            kinWindow[0] + rng.random() * kinWindowJitter,
            kinWindow[1] + rng.random() * kinWindowJitter)
        thisSW = (
            stimWindow[0] + rng.random() * stimWindowJitter,
            stimWindow[1] + rng.random() * stimWindowJitter)
        ##
        maxVel = velocityLookup[(row['pedalMovementCat'], row['pedalDirection'])]
        maxVel = maxVel + np.sign(maxVel) * (0.2 + 0.2 * rng.random())
        vel = pd.DataFrame(
            np.asarray([
                velRamp(
                    tb, thisKW,
                    kinWindowRamp, maxVel)
                for tb in bins]).reshape(1, -1),
            index=thisIdx, columns=bins)
        vel = vel.abs()
        vel.columns.name = 'bin'
        amp = pd.DataFrame(0., index=thisIdx, columns=bins)
        rate = pd.DataFrame(0., index=thisIdx, columns=bins)
        rate.columns.name = 'bin'
        if row['electrode'] != 'NA':
            stimMask = np.asarray((bins >= thisSW[0]) & (bins < thisSW[1]), dtype=float)
            # amp.loc[:, :] = stimMask * row['trialAmplitude']
            rate.loc[:, :] = stimMask * row['trialRateInHz']
            maxAmp = row['trialAmplitude']
            maxAmp = maxAmp + ampJitter * (0.2 + 0.8 * rng.random())
            amp = pd.DataFrame(
                np.asarray([
                    velRamp(
                        tb, thisSW,
                        stimWindowRamp, maxAmp)
                    for tb in bins]).reshape(1, -1),
                index=thisIdx, columns=bins)
        amp.columns.name = 'bin'
        toyLhsList.append(
            pd.concat([
                vel.stack('bin').to_frame(name='velocity'),
                amp.stack('bin').to_frame(name='amplitude'),
                rate.stack('bin').to_frame(name='RateInHz'),
                ], axis='columns'))
    toyLhsDF = pd.concat(toyLhsList)
    colsToScale = ['velocity', 'amplitude', 'RateInHz']
    saveIndex = toyLhsDF.index.copy()
    toyTrialInfo = toyLhsDF.index.to_frame().reset_index(drop=True)
    lOfTransformers = [
        (['velocity'], MinMaxScaler(feature_range=(0., 1.)),),
        (['amplitude'], MinMaxScaler(feature_range=(0., 1)),),
        (['RateInHz'], MinMaxScaler(feature_range=(0., .5)),)
        ]
    for cN in toyLhsDF.columns:
        if cN not in colsToScale:
            lOfTransformers.append(([cN], None,))
    lhsScaler = DataFrameMapper(lOfTransformers, input_df=True, df_out=True)
    scaledLhsDF = lhsScaler.fit_transform(toyLhsDF)
    scaledLhsDF.reset_index(
        level=['electrode', 'pedalMovementCat'], inplace=True)
    scaledLhsDF.reset_index(inplace=True, drop=True)
    scaledLhsDF.index.name = 'trial'
    #
    designFormula = "velocity + electrode:(amplitude/RateInHz)"
    pt = PatsyTransformer(designFormula, return_type="matrix")
    designMatrix = pt.fit_transform(scaledLhsDF)
    designInfo = designMatrix.design_info
    designDF = (
        pd.DataFrame(
            designMatrix, index=toyLhsDF.index,
            columns=designInfo.column_names))
    #

    iteratorSuffix = arguments['iteratorSuffix']
    if iteratorSuffix == 'a':
        restrictMask = (scaledLhsDF['electrode'] == '+ E16 - E9').to_numpy()
        ################################################################
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 0)  # parallel
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [80., 40., 0.],
            # out of plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
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
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        stimDirection = vg.rotate(
            kinDirection, vg.basis.z, 90)  # perpendicular
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [80., 40., 0.],
            # out of plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],
            # paralel
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        #####
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
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 2.,
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
        #####
        kinDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 90)
        stimDirection = vg.rotate(
            vg.basis.x, vg.basis.z, 0)
        rotCoeffs = pd.Series({
            'electrode[+ E16 - E9]:amplitude': [0., 90., 0.],  # out of plane
            'electrode[+ E16 - E5]:amplitude': [0., 0., 0.],  # in plane
            'electrode[NA]:amplitude': [0., 0., 0.],
            })
        gtCoeffs = pd.Series({
            'Intercept': 0.,
            'velocity': 10.,
            #
            'electrode[+ E16 - E5]:amplitude': 5.,
            'electrode[+ E16 - E9]:amplitude': 5.,
            'electrode[NA]:amplitude': 0.,
            #
            'electrode[+ E16 - E9]:amplitude:RateInHz': 0.,
            'electrode[+ E16 - E5]:amplitude:RateInHz': 0.,
            'electrode[NA]:amplitude:RateInHz': 0.
            })
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
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for cN in magnitudes.columns:
            ax[0].plot(magnitudes[cN], label=cN)
        ax[0].legend()

    latentNoiseDistr = stats.multivariate_normal(
        mean=latentMu, cov=latentCovariance)
    latentNoise = latentNoiseDistr.rvs(scaledLhsDF.shape[0])
    # (latentNoise - latentNoise.mean(axis=0)) + latentMu
    latentRhsDF = pd.DataFrame(
        latentNoise, index=scaledLhsDF.index,
        columns=['latent{}'.format(cN) for cN in range(nDim)])
    latentRhsDF.columns.name = 'feature'
    termMagnitudes = pd.DataFrame(
        0, index=scaledLhsDF.index, columns=designInfo.term_names
        )
    #
    for termName, termSlice in designInfo.term_name_slices.items():
        termMagnitudes.loc[:, termName] = magnitudes.iloc[:, termSlice].sum(axis='columns').to_numpy()
        if sanityCheckThis:
            for cN in termMagnitudes.columns:
                ax[1].plot(termMagnitudes[cN], label=cN)
            ax[1].legend()
            plt.show()
        termValues = pd.DataFrame(
            termMagnitudes[termName].to_numpy().reshape(-1, 1) * projectionLookup[termName],
            columns=['data{}'.format(cN) for cN in range(nDim)], index=latentRhsDF.index
            )
        latentRhsDF.loc[:, :] += termValues.to_numpy()

    # TODO: adjust explained_var to account for std introduced by the regressors
    embeddedDF = (np.sqrt(explainedVar) @ latentRhsDF.to_numpy().T)
    toyRhsDF = pd.DataFrame(
        (wRot @ (embeddedDF - rotationOffset) + rotationOffset).T,
        index=latentRhsDF.index,
        columns=['data{}'.format(cN) for cN in range(nDim)])
    electrodeInfluence = (
            termMagnitudes['electrode:amplitude'] +
            termMagnitudes['electrode:amplitude:RateInHz'])
    defaultMS = matplotlib.rcParams['lines.markersize'] ** 2
    movementInfluence = pd.Series(
        MinMaxScaler(feature_range=(defaultMS, defaultMS * 2))
        .fit_transform(
            termMagnitudes['velocity']
            .to_numpy().reshape(-1, 1))
        .flatten(), index=termMagnitudes.index)
    latentPlotDF = pd.concat([latentRhsDF, scaledLhsDF], axis='columns')
    latentPlotDF.columns = latentPlotDF.columns.astype(str)
    latentPlotDF.loc[:, 'restrictMask'] = restrictMask
    latentPlotDF.loc[:, 'electrodeInfluence'] = electrodeInfluence
    latentPlotDF.loc[:, 'movementInfluence'] = movementInfluence
    latentPlotDF.loc[:, 'moving'] = (
            np.sign(toyLhsDF['velocity']) *
            (toyLhsDF['velocity'].abs() > 0.25)).to_numpy()
    latentPlotDF.loc[:, 'limbState'] = latentPlotDF['moving'].map(
        {-1: 'moving', 0: 'baseline', 1: 'moving'}).to_numpy()
    latentPlotDF.loc[:, 'activeElectrode'] = latentPlotDF['electrode'].copy()
    latentPlotDF.loc[latentPlotDF['amplitude'] == latentPlotDF['amplitude'].min(), 'activeElectrode'] = 'NA'
    latentPlotDF.loc[:, 'limbState x activeElectrode'] = latentPlotDF.apply(
        lambda x: '{} {}'.format(x['limbState'], x['activeElectrode']), axis='columns')
    latentPlotDF.loc[:, 'pedalMovementCat x electrode'] = latentPlotDF.apply(
        lambda x: '{} {}'.format(x['pedalMovementCat'], x['electrode']), axis='columns')
    #
    for name, group in latentPlotDF.groupby('limbState x activeElectrode'):
        extraRotMat = np.eye(nDim)
        elecName = group['activeElectrode'].unique()[0]
        elecFactorName = 'electrode[{}]:amplitude'.format(elecName)
        if (elecFactorName in rotCoeffs):
            extraRotMat = Rot.from_euler(
                eulerOrder, [beta for beta in rotCoeffs[elecFactorName]],
                degrees=True).as_matrix()
            print('applying rotation for {}\nextraRotMat\n{}'.format(elecFactorName, extraRotMat))
            rotateMask = toyRhsDF.index.isin(group.index)
            tempToyData = toyRhsDF.loc[rotateMask, :].T.to_numpy()
            rotatedToyData = extraRotMat @ (tempToyData - rotationOffset) + rotationOffset
            toyRhsDF.loc[rotateMask, :] = rotatedToyData.T
    #
    if not np.allclose(np.zeros((nDim, nDim)), residualCovariance):
        noiseDistr = stats.multivariate_normal(
            mean=mu, cov=residualCovariance)
        noiseTerm = noiseDistr.rvs(toyRhsDF.shape[0])
        toyRhsDF += noiseTerm
        # toyRhsDF += ((noiseTerm - noiseTerm.mean(axis=0)) + mu)
    rhsPlotDF = pd.concat([toyRhsDF, scaledLhsDF, toyTrialInfo['bin']], axis='columns')
    rhsPlotDF.loc[:, 'restrictMask'] = restrictMask
    rhsPlotDF.loc[:, 'electrodeInfluence'] = electrodeInfluence
    rhsPlotDF.loc[:, 'movementInfluence'] = movementInfluence
    for cN in [
            'limbState', 'activeElectrode',
            'limbState x activeElectrode',
            'pedalMovementCat x electrode']:
        rhsPlotDF.loc[:, cN] = latentPlotDF[cN]
        toyTrialInfo.loc[:, cN] = latentPlotDF[cN]
    baseline_cov = LedoitWolf().fit(toyRhsDF.loc[toyTrialInfo['limbState x activeElectrode'] == 'baseline NA', :])
    rhsPlotDF.loc[:, 'mahalanobis'] = np.sqrt(baseline_cov.mahalanobis(toyRhsDF))
    lOfMasksForBreakdown = [
        {
            'mask': toyTrialInfo['electrode'].isin(['NA']).to_numpy() & toyTrialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Baseline',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': toyTrialInfo['electrode'].isin(['NA']).to_numpy() & toyTrialInfo['pedalMovementCat'].isin(['outbound', 'return']).to_numpy(),
            'label': 'Movement-only',
            'ellipsoidsToPlot': ['baseline NA', 'moving NA']},
        {
            'mask': toyTrialInfo['limbState x activeElectrode'].isin(['baseline + E16 - E5', 'baseline NA']).to_numpy(),
            'label': 'Stim.-only (electrode X)',
            'ellipsoidsToPlot': ['baseline NA', 'baseline + E16 - E5']},
        {
            'mask': toyTrialInfo['limbState x activeElectrode'].isin(['baseline + E16 - E9', 'baseline NA']).to_numpy(),
            'label': 'Stim.-only (electrode Y)',
            'ellipsoidsToPlot': ['baseline NA', 'baseline + E16 - E9']},
        {
            'mask': toyTrialInfo['limbState x activeElectrode'].isin(['baseline + E16 - E9', 'baseline + E16 - E5', 'baseline NA', 'moving NA']).to_numpy(),
            'label': 'Stim.-only (electrodes X, Y)',
            'ellipsoidsToPlot': ['baseline NA', 'moving NA', 'baseline + E16 - E9', 'baseline + E16 - E5']},
        ]
    lOfMasksForAnim = [
        {
            'mask': toyTrialInfo['electrode'].isin(['NA']).to_numpy() & toyTrialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Baseline',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': toyTrialInfo['electrode'].isin(['NA']).to_numpy() & toyTrialInfo['pedalMovementCat'].isin(['outbound']).to_numpy(),
            'label': 'Movement-only',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': toyTrialInfo['electrode'].isin(['+ E16 - E5']).to_numpy() & toyTrialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Stim.-only (electrode X)',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': toyTrialInfo['electrode'].isin(['+ E16 - E9']).to_numpy() & toyTrialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Stim.-only (electrode Y)',
            'ellipsoidsToPlot': ['baseline NA']},
        ]
    lOfMasksByLimbStateActiveElectrode = [
        {
            'mask': (toyTrialInfo['limbState x activeElectrode'] == lsae).to_numpy(),
            'label': lsae,
            }
        for lsae in toyTrialInfo['limbState x activeElectrode'].unique()
        ]
    #
    uEll = np.linspace(0, 2 * np.pi, 100)
    vEll = np.linspace(0, np.pi, 100)
    xEll = np.outer(np.cos(uEll), np.sin(vEll))
    yEll = np.outer(np.sin(uEll), np.sin(vEll))
    zEll = np.outer(np.ones_like(uEll), np.cos(vEll))
    unitSphere = np.stack((xEll, yEll, zEll), 0).reshape(3, -1)
    ellipsoidEmpiricalDict = {}
    ellipsoidTheoryDict = {}
    #
    ellipsoidScaleDict = {
        'baseline NA': .3,
        'baseline + E16 - E5': .3,
        'baseline + E16 - E9': .3,
        'moving NA': .3,
        'moving + E16 - E5': .3,
        'moving + E16 - E9': .3,
        }
    empCovMats = {}
    #
    for name, group in latentPlotDF.groupby('limbState x activeElectrode'):
        emp_cov = LedoitWolf().fit(toyRhsDF.loc[group.index, :])
        ellipsoidEmpiricalDict[name] = (
            (
                (ellipsoidScaleDict[name] * (emp_cov.covariance_) + 5. * np.eye(nDim)) @ unitSphere +
                emp_cov.location_.reshape(3, 1))
            .reshape(3, *xEll.shape))
        # pdb.set_trace()
        u, s, v = np.linalg.svd(emp_cov.covariance_)
        emp_cov.covariance_eigenvalues = s
        emp_cov.covariance_u = u
        empCovMats[name] = emp_cov
        # add in some isotropic covariance to make the blobs contain the points better
        #####
        theoretical_loc = mu.copy().reshape(3, 1)
        regressorStretch = np.zeros((nDim, nDim))
        extraDisplacement = latentMu.reshape(3, 1) * 0  # centroid moves because of regressors
        #
        stretchForRegresssors = True
        elecName = toyTrialInfo.loc[group.index, 'activeElectrode'].unique()[0]
        extraRotMat = np.eye(nDim)
        elecFactorName = 'electrode[{}]:amplitude'.format(elecName)
        elecRateFactorName = 'electrode[{}]:amplitude:RateInHz'.format(elecName)
        if elecFactorName in rotCoeffs:
            extraRotMat = Rot.from_euler(eulerOrder, [beta for beta in rotCoeffs[elecFactorName]], degrees=True).as_matrix()
        if (elecName != 'NA') and stretchForRegresssors:
            # the ellipsoid will get stretched because of regressors..
            regressorStretch += np.diag(projectionLookup['electrode:amplitude']) * ((gtCoeffs[elecFactorName]) ** 2 * 5e-2)
            extraDisplacement += (projectionLookup['electrode:amplitude'] * gtCoeffs[elecFactorName] / 2).reshape(3, 1)
            regressorStretch += np.diag(projectionLookup['electrode:amplitude:RateInHz']) * ((gtCoeffs[elecRateFactorName]) ** 2 * 5e-2)
            extraDisplacement += (projectionLookup['electrode:amplitude:RateInHz'] * gtCoeffs[elecRateFactorName] / 2).reshape(3, 1)
        if ('baseline' not in name) and stretchForRegresssors:
            regressorStretch += np.diag(projectionLookup['velocity']) * ((gtCoeffs['velocity']) ** 2 * 5e-2)
            extraDisplacement += (projectionLookup['velocity'] * gtCoeffs['velocity'] / 2).reshape(3, 1)
        # print('Ellipsoid: limbState x activeElectrode: {}\nwRot =\n{}\nextraRotMat=\n{}'.format(name, wRot, extraRotMat))
        # transformation in the latent space
        newScale = 3 * np.sqrt(latentCovariance + regressorStretch + residualCovariance)
        thisEllipsoid = newScale @ copy(unitSphere) + latentMu.reshape(3, 1) + extraDisplacement
        # embed in the higher dimension
        thisEllipsoid = np.sqrt(explainedVar) @ thisEllipsoid
        thisTransf = extraRotMat @ (wRot @ (np.eye(nDim)))
        thisEllipsoid = thisTransf @ (thisEllipsoid - rotationOffset) + rotationOffset + theoretical_loc
        '''
            tempToyData = toyRhsDF.loc[rotateMask, :].T.to_numpy()
            rotatedToyData = extraRotMat @ (tempToyData - rotationOffset) + rotationOffset'''
        # add in some isotropic covariance to make the blobs contain the points better
        ellipsoidTheoryDict[name] = thisEllipsoid.reshape(3, *xEll.shape)
    #
    '''
    ellipsoidDict = {
        'baseline NA': ellipsoidEmpiricalDict['baseline NA'],
        'baseline + E16 - E5': ellipsoidEmpiricalDict['baseline + E16 - E5'],
        'baseline + E16 - E9': ellipsoidEmpiricalDict['baseline + E16 - E9'],
        'moving NA': ellipsoidEmpiricalDict['moving NA'],
        'moving + E16 - E5': ellipsoidEmpiricalDict['moving + E16 - E5'],
        'moving + E16 - E9': ellipsoidEmpiricalDict['moving + E16 - E9'],
        }
    '''
    ellipsoidDict = ellipsoidTheoryDict
    #
    def drawEllipsoidAxes(theAx, theName):
        x0, y0, z0 = empCovMats[theName].location_
        sx, sy, sz = np.sqrt(empCovMats[theName].covariance_eigenvalues) * 2
        covarEigPlotOpts = dict(zorder=2.1, lw=.5)
        #
        dx, dy, dz = sx * empCovMats[theName].covariance_u[:, 0]
        theAx.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color='c', **covarEigPlotOpts)
        dx, dy, dz = sy * empCovMats[theName].covariance_u[:, 1]
        theAx.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color='m', **covarEigPlotOpts)
        dx, dy, dz = sz * empCovMats[theName].covariance_u[:, 2]
        theAx.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color='y', **covarEigPlotOpts)
        return
    makeKDEPlot = False
    makeLatentPDF = False
    if makeLatentPDF:
        pdfPath = os.path.join(
            figureOutputFolder,
            'synthetic_dataset_latent_{}.pdf'.format(iteratorSuffix)
            )
        with PdfPages(pdfPath) as pdf:
            extentLatent = (
                latentPlotDF.loc[:, latentRhsDF.columns].quantile(1 - 1e-2) -
                latentPlotDF.loc[:, latentRhsDF.columns].quantile(1e-2))
            #
            xExtentLatent, yExtentLatent = extentLatent['latent0'], extentLatent['latent1']
            midPointsLatent = (
                latentPlotDF.loc[:, latentRhsDF.columns].quantile(1 - 1e-2) +
                latentPlotDF.loc[:, latentRhsDF.columns].quantile(1e-2)) / 2
            #
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                latentPlotDF.iloc[:, 0], latentPlotDF.iloc[:, 1],
                cmap='viridis', c=latentPlotDF['electrodeInfluence'],
                # s=group['movementInfluence'],
                rasterized=True, **scatterOpts)
            ax.set_xlim([
                midPointsLatent['latent0'] - xExtentLatent,
                midPointsLatent['latent0'] + xExtentLatent])
            ax.set_ylim([
                midPointsLatent['latent1'] - yExtentLatent,
                midPointsLatent['latent1'] + yExtentLatent])
            fig.tight_layout()
            pdf.savefig()
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            if makeKDEPlot:
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.kdeplot(
                    x='latent0', y='latent1', hue='limbState x activeElectrode',
                    levels=20, fill=True, thresh=.1, palette='Set1',
                    common_norm=False, alpha=.5,
                    data=latentPlotDF, ax=ax)
                ax.set_xlim([
                    midPointsLatent['latent0'] - xExtentLatent, midPointsLatent['latent0'] + xExtentLatent])
                ax.set_ylim([
                    midPointsLatent['latent1'] - yExtentLatent, midPointsLatent['latent1'] + yExtentLatent])
                fig.tight_layout()
                pdf.savefig()
                #
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            if iteratorSuffix in ['a', 'b', 'g']:
                for maskDict in lOfMasksForBreakdown:
                    # for maskDict in lOfMasksForBreakdown: lOfMasksByLimbStateActiveElectrode
                    thisMask = maskDict['mask']
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(
                        latentPlotDF.loc[thisMask, 'latent0'],
                        latentPlotDF.loc[thisMask, 'latent1'], cmap='viridis',
                        c=latentPlotDF.loc[thisMask, 'electrodeInfluence'],
                        # s=group['movementInfluence'],
                        rasterized=True, **scatterOpts)
                    ax.set_xlim([
                        midPointsLatent['latent0'] - xExtentLatent,
                        midPointsLatent['latent0'] + xExtentLatent])
                    ax.set_ylim([
                        midPointsLatent['latent1'] - yExtentLatent,
                        midPointsLatent['latent1'] + yExtentLatent])
                    # figTitle = fig.suptitle(maskDict['label'], y=0.9)
                    fig.tight_layout()
                    pdf.savefig(
                        bbox_inches='tight', pad_inches=0,
                        # bbox_extra_artists=[figTitle]
                        )
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
    ####
    pdfPath = os.path.join(
        figureOutputFolder,
        'synthetic_dataset_{}.pdf'.format(iteratorSuffix))
    extent = (
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + colsToScale + ['mahalanobis']].quantile(1 - 1e-2) -
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + colsToScale + ['mahalanobis']].quantile(1e-2))
    # xExtent, yExtent, zExtent = extent['data0'], extent['data1'], extent['data2']
    masterExtent = extent.loc[toyRhsDF.columns.to_list()].max() * 0.6
    xExtent, yExtent, zExtent = masterExtent, masterExtent, masterExtent
    midPoints = (
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + colsToScale + ['mahalanobis']].quantile(1 - 1e-2) +
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + colsToScale + ['mahalanobis']].quantile(1e-2)) / 2
    #
    def formatTheAxes(
            theAx, customMidPoints=None, customExtents=None,
            customLabels=None,
            showOrigin=False, zoomFactor=1.):
        #
        theAx.view_init(**defaultAxisOrientation)
        #
        if customMidPoints is not None:
            xMid, yMid, zMid = customMidPoints
        else:
            xMid, yMid, zMid = midPoints['data0'], midPoints['data1'], midPoints['data2']
        #
        if customExtents is not None:
            xExt, yExt, zExt = customExtents
        else:
            xExt, yExt, zExt = xExtent, yExtent, zExtent
        #
        if customLabels is not None:
            xLab, yLab, zLab = customLabels
        else:
            xLab, yLab, zLab = 'LFP Ch. #1 (a.u.)', 'LFP Ch. #2 (a.u.)', 'LFP Ch. #3 (a.u.)'
        #
        theAx.set_xlim3d([xMid - xExt / zoomFactor, xMid + xExt / zoomFactor])
        theAx.set_ylim3d([yMid - yExt / zoomFactor, yMid + yExt / zoomFactor])
        theAx.set_zlim3d([zMid - zExt / zoomFactor, zMid + zExt / zoomFactor])
        #
        theAx.set_xticks([xMid - xExt / zoomFactor / 2, xMid, xMid + xExt / zoomFactor / 2])
        theAx.set_yticks([yMid - yExt / zoomFactor / 2, yMid, yMid + yExt / zoomFactor / 2])
        theAx.set_zticks([zMid - zExt / zoomFactor / 2, zMid, zMid + zExt / zoomFactor / 2])
        #
        theAx.set_xticklabels([])
        theAx.set_yticklabels([])
        theAx.set_zticklabels([])
        #
        theAx.set_xlabel(xLab, labelpad=-15, va='top', ha='center')
        theAx.set_ylabel(yLab, labelpad=-15, va='top', ha='center')
        theAx.set_zlabel(zLab, labelpad=-15, va='top', ha='center')
        #
        theAx.w_xaxis.set_pane_color([1., 1., 1.])
        theAx.w_yaxis.set_pane_color([1., 1., 1.])
        theAx.w_zaxis.set_pane_color([1., 1., 1.])
        #
        if showOrigin:
            x0, y0, z0 = baseline_cov.location_
            dx, dy, dz = xExt / zoomFactor / 4, yExt / zoomFactor / 4, zExt / zoomFactor / 4
            coordOriginPlotOpts = dict(zorder=2.5)
            theAx.plot([x0, x0 + dx], [y0, y0], [z0, z0], color='b', **coordOriginPlotOpts)
            theAx.plot([x0, x0], [y0, y0 + dy], [z0, z0], color='r', **coordOriginPlotOpts)
            theAx.plot([x0, x0], [y0, y0], [z0, z0 + dz], color='g', **coordOriginPlotOpts)
        return

    with PdfPages(pdfPath) as pdf:
        plotAllScatter = False
        if plotAllScatter:
            fig = plt.figure()
            fig.set_size_inches((6, 6))
            ax = fig.add_subplot(projection='3d')
            ax.set_proj_type('ortho')
            ax.scatter(
                rhsPlotDF.iloc[:, 0], rhsPlotDF.iloc[:, 1], rhsPlotDF.iloc[:, 2],
                # cmap=sns.color_palette(ampPaletteStr, as_cmap=True),
                # c=rhsPlotDF.loc[thisMask, 'electrodeInfluence'],
                #
                # cmap=covPatternPalette,
                # c=rhsPlotDF.loc[thisMask, 'limbState x activeElectrode'],
                #
                c=lfpColor,
                # s=group['movementInfluence'],
                rasterized=True,
                **scatterOpts)
            formatTheAxes(ax)
            fig.tight_layout()
            pdf.savefig()
            if arguments['debugging'] or arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #
            rhsPlotDF.loc[:, 'x2'], rhsPlotDF.loc[:, 'y2'], _ = proj3d.proj_transform(
                rhsPlotDF.iloc[:, 0], rhsPlotDF.iloc[:, 1], rhsPlotDF.iloc[:, 2], ax.get_proj())
        if makeKDEPlot and plotAllScatter:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.kdeplot(
                x='x2', y='y2', hue='limbState x activeElectrode',
                levels=20, fill=True, thresh=.1, palette='Set2',
                common_norm=False, linewidth=0., alpha=.5,
                data=rhsPlotDF, ax=ax)
            fig.tight_layout()
            pdf.savefig()
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        ########
        if iteratorSuffix in ['a', 'b', 'g']:
            plotRelPlots = True
            if plotRelPlots:
                tempCols = [cN for cN in toyRhsDF.columns] + [
                    'limbState x activeElectrode', 'pedalMovementCat x electrode']
                rhsPlotDFStack = pd.DataFrame(
                    rhsPlotDF.loc[:, tempCols].to_numpy(),
                    columns=tempCols,
                    index=pd.MultiIndex.from_frame(toyTrialInfo.loc[
                        :, [
                            'electrode', 'bin', 'pedalMovementCat', 'trialAmplitude']])
                        )
                rhsPlotDFStack = rhsPlotDFStack.set_index('limbState x activeElectrode', append=True)
                rhsPlotDFStack = rhsPlotDFStack.set_index('pedalMovementCat x electrode', append=True)
                rhsPlotDFStack.columns.name = 'feature'
                rhsPlotDFStack = rhsPlotDFStack.stack().to_frame(name='signal').reset_index()
                g = sns.relplot(
                    row='feature', col='pedalMovementCat x electrode',
                    x='bin', y='signal', hue='trialAmplitude',
                    data=rhsPlotDFStack,
                    palette="ch:0.6,-.3,dark=.1,light=0.7,reverse=1",
                    errorbar='se', kind='line',
                    facet_kws=dict(margin_titles=True),
                    height=3, aspect=1
                    )
                # plotProcFuns = [
                #     asp.genAxisLabelOverride(
                #         xTemplate=None, yTemplate=None,
                #         titleTemplate=None, rowTitleTemplate=None, colTitleTemplate=None,
                #         prettyNameLookup=None,
                #         colKeys=None, dropNaNCol='segment')
                #     ]
                # for (ro, co, hu), dataSubset in g.facet_data():
                #     if len(plotProcFuns):
                #         for procFun in plotProcFuns:
                #             procFun(g, ro, co, hu, dataSubset)
                g.tight_layout()
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            plotDisPlots = False
            if plotDisPlots:
                g = sns.displot(
                    row='feature', col='pedalMovementCat x electrode',
                    y='signal', hue='trialAmplitude',
                    data=rhsPlotDFStack,
                    palette="ch:0.6,-.3,dark=.1,light=0.7,reverse=1",
                    kind='kde', common_norm=False,
                    height=3, aspect=1
                    )
                g.tight_layout()
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                #
            plotRegressorRelPlots = False
            if plotRegressorRelPlots:
                tempCols = [cN for cN in toyLhsDF.columns] + [
                    'limbState x activeElectrode', 'pedalMovementCat x electrode']
                lhsPlotDFStack = pd.DataFrame(
                    rhsPlotDF.loc[:, tempCols].to_numpy(),
                    columns=tempCols,
                    index=pd.MultiIndex.from_frame(toyTrialInfo.loc[
                        :, [
                            'electrode', 'bin', 'pedalMovementCat', 'trialAmplitude']])
                    )
                lhsPlotDFStack = lhsPlotDFStack.set_index('limbState x activeElectrode', append=True)
                lhsPlotDFStack = lhsPlotDFStack.set_index('pedalMovementCat x electrode', append=True)
                lhsPlotDFStack.columns.name = 'feature'
                lhsPlotDFStack = lhsPlotDFStack.stack().to_frame(name='signal').reset_index()
                ###
                g = sns.relplot(
                    row='feature', col='pedalMovementCat x electrode',
                    x='bin', y='signal', hue='trialAmplitude',
                    data=lhsPlotDFStack, palette='viridis',
                    errorbar='se', kind='line', height=3, aspect=1
                    )
                g.tight_layout()
                #
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                ###
            plotCovMats = True
            if plotCovMats:
                nCols = int(np.ceil(np.sqrt(len(limbStateElectrodeNames))))
                nRows = int(np.floor(len(limbStateElectrodeNames) / nCols))
                print('Heatmap; nRows = {}, nCols = {}'.format(nRows, nCols))
                gs = GridSpec(
                    nRows, nCols + 1,
                    width_ratios=[2 for i in range(nCols)] + [.1],
                    height_ratios=[1 for i in range(nRows)]
                    )
                fig = plt.figure()
                fig.set_size_inches((2 * nCols + .1, 2 * nRows))
                covMatIndex = [prettyNameLookup[cN] for cN in toyRhsDF.columns]
                # pdb.set_trace()
                vMax = max([ecm.covariance_.max() for _, ecm in empCovMats.items()])
                vMin = max([ecm.covariance_.min() for _, ecm in empCovMats.items()])
                for axIdx, name in enumerate(limbStateElectrodeNames):
                    # print('axIdx = {}, (axIdx % nRows) = {}, (axIdx // nRows) = {}'.format(
                    #     axIdx, axIdx % nRows, axIdx // nRows))
                    thisAx = fig.add_subplot(gs[axIdx % nRows, axIdx // nRows])
                    covMatDF = pd.DataFrame(
                        empCovMats[name].covariance_,
                        index=covMatIndex, columns=covMatIndex)
                    if axIdx == 0:
                        cbAx = fig.add_subplot(gs[:, nCols])
                        cbarOpts = dict(cbar=True, cbar_ax=cbAx)
                    else:
                        cbarOpts = dict(cbar=False)
                    sns.heatmap(
                        covMatDF, ax=thisAx,
                        cmap='crest_r', vmin=vMin, vmax=vMax,
                        **cbarOpts)
                    thisAx.set_title(prettyNameLookup[name])
                figTitle = fig.suptitle('Covariance matrices')
                fig.tight_layout()
                pdf.savefig(
                    bbox_inches='tight', pad_inches=0,
                    bbox_extra_artists=[figTitle])
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            plotMaskedScatters = True
            markersByLimbStateElectrode = False
            if plotMaskedScatters:
                for maskDict in lOfMasksForBreakdown:
                    # for maskDict in lOfMasksByLimbStateActiveElectrode:
                    # for maskDict in lOfMasksForBreakdown:
                    # lOfMasksForAnim
                    with matplotlib.rc_context({'axes.titlepad': -20}):
                        fig = plt.figure()
                        fig.set_size_inches((6, 6))
                        ax = fig.add_subplot(projection='3d')
                        ax.set_proj_type('ortho')
                        #
                        thisMask = maskDict['mask']
                        if 'ellipsoidsToPlot' in maskDict:
                            ellipsoidsToPlot = maskDict['ellipsoidsToPlot']
                        else:
                            ellipsoidsToPlot = rhsPlotDF.loc[thisMask, 'limbState x activeElectrode'].unique()
                        for name in ellipsoidsToPlot:
                            ax.plot_surface(
                                *ellipsoidDict[name],
                                color=sns.utils.alter_color(covPatternPalette[name], l=.3),
                                **ellipsoidPlotOpts
                                )
                            drawEllipsoidAxes(ax, name)
                        if markersByLimbStateElectrode:
                            for name in rhsPlotDF.loc[thisMask, 'limbState x activeElectrode'].unique():
                                nameMask = rhsPlotDF['limbState x activeElectrode'] == name
                                ax.scatter(
                                    rhsPlotDF.loc[(thisMask & nameMask), 'data0'],
                                    rhsPlotDF.loc[(thisMask & nameMask), 'data1'],
                                    rhsPlotDF.loc[(thisMask & nameMask), 'data2'],
                                    ##
                                    # c=rhsPlotDF.loc[(thisMask & nameMask), 'electrodeInfluence'],
                                    # cmap=sns.color_palette(ampPaletteStr, as_cmap=True),
                                    c=covPatternPalette[name],
                                    ##
                                    marker=limbStateElectrodeMarkerDict[name],
                                    # s=group['movementInfluence'],
                                    rasterized=True, **scatterOpts)
                        else:
                            ax.scatter(
                                rhsPlotDF.loc[thisMask, 'data0'],
                                rhsPlotDF.loc[thisMask, 'data1'],
                                rhsPlotDF.loc[thisMask, 'data2'],
                                #
                                # cmap=sns.color_palette(ampPaletteStr, as_cmap=True),
                                # c=rhsPlotDF.loc[thisMask, 'electrodeInfluence'],
                                #
                                c=[covPatternPalette[lse] for lse in rhsPlotDF.loc[thisMask, 'limbState x activeElectrode']],
                                #
                                # c=lfpColor,
                                # s=group['movementInfluence'],
                                rasterized=True, **scatterOpts)
                        theseLegendEntries = legendData.loc[['ellipsoid'] + ellipsoidsToPlot + ['blank'] + legendPcaDirs]
                        ax.legend(handles=theseLegendEntries.to_list(), loc='lower right')
                        # fig.suptitle(
                        #     maskDict['label'] + '\n\n' + '\n'.join(rhsPlotDF.loc[thisMask, 'limbState x activeElectrode'].unique()))
                        # figTitle = fig.suptitle(maskDict['label'], y=0.9)
                        ##
                        # zF = 1.8 if (maskDict['label'] == 'Baseline') else 1.
                        # formatTheAxes(ax, zoomFactor=zF, showOrigin=False)
                        ##
                        formatTheAxes(ax)
                        fig.tight_layout()
                        pdf.savefig(
                            bbox_inches='tight', pad_inches=0,
                            # bbox_extra_artists=[figTitle]
                            )
                        if arguments['showFigures']:
                            plt.show()
                        else:
                            plt.close()
    ############################################
    makeAnimations = True
    if makeAnimations:
        fps = int(iteratorOpts['forceBinInterval'] ** (-1))
        tailLength = 3
        aniLineKws = {'linestyle': '-', 'linewidth': 1.5}
        cometLineKws = {'linestyle': '-', 'linewidth': 1.}
        mhLineKws = {'linestyle': '-', 'linewidth': 2.}
        aniMarkerKws = {'marker': 'o', 'markersize': 2, 'linewidth': 0}
        whatToPlot = ['data0', 'data1', 'data2', 'velocity', 'amplitude', 'mahalanobis']
        #
        cMap = sns.color_palette(ampPaletteStr, as_cmap=True)
        cMapScaleFun = lambda am: cMap.colors.shape[0] * min(1, max(0, (am - rhsPlotDF['amplitude'].min()) / (rhsPlotDF['amplitude'].max() - rhsPlotDF['amplitude'].min())))
        cMapMahal = sns.color_palette(mahalPaletteStr, as_cmap=True)
        cMapMahalScaleFun = lambda am: cMapMahal.colors.shape[0] * min(1, max(0, (am - rhsPlotDF['mahalanobis'].min()) / (
                    rhsPlotDF['mahalanobis'].max() - rhsPlotDF['mahalanobis'].min())))
        # cMap(cMapScaleFun(0.2))
        for maskIdx, maskDict in enumerate(lOfMasksForAnim):
            if maskIdx not in [mi for mi in range(len(lOfMasksForAnim))]:
                continue
            print('Starting to generate animation for {}'.format(maskDict['label']))
            aniPath = os.path.join(
                figureOutputFolder, 'synthetic_dataset_{}_mask_{}.mp4'.format(iteratorSuffix, maskIdx))
            #
            fig = plt.figure()
            nRows = len(whatToPlot)
            nCols = 2
            inchesPerRow = 2.
            fig.set_size_inches((inchesPerRow * nCols * nRows, inchesPerRow * nRows))
            gs = GridSpec(
                nRows, nCols,
                width_ratios=[1 for i in range(nCols)],
                height_ratios=[1 for i in range(nRows)],
                hspace=0.05, wspace=0.05
                )
            ax3d = fig.add_subplot(gs[:, 0], projection='3d')
            ax3d.set_proj_type('ortho')
            ax2d = [fig.add_subplot(gs[i, 1]) for i in range(nRows)]
            #
            chooseSomeTrials = toyTrialInfo.loc[maskDict['mask'] & (toyTrialInfo['trialAmplitude'] == toyTrialInfo.loc[maskDict['mask'], 'trialAmplitude'].max()), 'trialUID'].unique()[:3]
            thisMask = maskDict['mask'] & toyTrialInfo['trialUID'].isin(chooseSomeTrials)
            dataDF = rhsPlotDF.loc[thisMask, :].copy()
            dataFor3D = dataDF.loc[:, ['data0', 'data1', 'data2']].T.to_numpy()
            dataDF.loc[:, 'trialUID'] = toyTrialInfo.loc[thisMask, 'trialUID']
            nFrames = min(500, dataDF.shape[0])
            progBar = tqdm(total=nFrames, mininterval=30., maxinterval=120.)
            for name in maskDict['ellipsoidsToPlot']:
                ax3d.plot_surface(
                    *ellipsoidDict[name],
                    color=sns.utils.alter_color(covPatternPalette[name], l=.5),
                    **ellipsoidPlotOpts)
                drawEllipsoidAxes(ax3d, name)
            #
            # fig.suptitle(figTitleStr)
            #
            cometTailLines = [None for i in range(tailLength)]
            for idx in range(tailLength):
                cometTailLines[idx] = ax3d.plot(
                    [], [], [], zorder=2.4,
                    **cometLineKws)[0]
                cometTailLines[idx].set_color([0, 0, 0, 0])
            #
            cometHead = ax3d.plot(
                [], [], [], zorder=2.5,
                **aniMarkerKws)[0]
            mhLine = ax3d.plot(
                [], [], [],
                color='g', zorder=2.3,
                **mhLineKws)[0]
            #
            for axIdx, thisAx in enumerate(ax2d):
                thisAx.fullTrace = dataDF[whatToPlot[axIdx]]
                thisLine, = thisAx.plot([], [], **aniLineKws)
                thisAx.plotLine = thisLine
                newXLims = (dataDF['bin'].min(), dataDF['bin'].max())
                print('newXLims = {}'.format(newXLims))
                thisAx.set_xlim(newXLims)
                thisAx.set_ylabel(prettyNameLookup[whatToPlot[axIdx]])
                thisAx.set_yticklabels([])
                thisAx.set_ylim(
                    rhsPlotDF[whatToPlot[axIdx]].min() - extent[whatToPlot[axIdx]] * 1e-2,
                    rhsPlotDF[whatToPlot[axIdx]].max() + extent[whatToPlot[axIdx]] * 1e-2)
                if axIdx == (len(ax2d) - 1):
                    thisAx.set_xlabel('Time (sec)')
                else:
                    thisAx.set_xticklabels([])
            #
            # figTitleStr = maskDict['label']
            # ax3d.set_title(figTitleStr)
            # zF = 1.9 if name == 'baseline NA' else 1.2
            # formatTheAxes(ax3d, zoomFactor=zF)
            #
            customMidPoints = (
                  rhsPlotDF.loc[maskDict['mask'], toyRhsDF.columns].quantile(1 - 1e-2) +
                  rhsPlotDF.loc[maskDict['mask'], toyRhsDF.columns].quantile(1e-2)) / 2
            customExtents = (
                    rhsPlotDF.loc[maskDict['mask'], toyRhsDF.columns].quantile(1 - 1e-2) -
                    rhsPlotDF.loc[maskDict['mask'], toyRhsDF.columns].quantile(1e-2))
            formatTheAxes(
                ax,
                # customMidPoints=[
                #     customMidPoints['data0'],
                #     customMidPoints['data1'],
                #     customMidPoints['data2']],
                # customExtents=[
                #     customExtents.max(),
                #     customExtents.max(),
                #     customExtents.max()],
                zoomFactor=0.8,  # zoom out
                )
            sns.despine(fig=fig)
            theseLegendEntries = legendData.loc[
                ['ellipsoid'] + maskDict['ellipsoidsToPlot'] + ['blank'] +
                legendPcaDirs +
                ['blank'] + ['data', 'mahalanobis']]
            ax3d.legend(handles=theseLegendEntries.to_list(), loc='lower right')
            #
            def animate(idx):
                colorVal = dataDF['amplitude'].iloc[idx]
                rgbaColor = np.asarray(cMap(cMapScaleFun(colorVal)))
                #
                currBin = dataDF['bin'].iloc[idx]
                currTrialUID = dataDF['trialUID'].iloc[idx]
                #
                currMHDist = dataDF['mahalanobis'].iloc[idx]
                rgbaColorMahal = np.asarray(cMapMahal(cMapMahalScaleFun(currMHDist)))
                tMask = ((dataDF['trialUID'] == currTrialUID) & (dataDF['bin'] <= currBin)).to_numpy()
                #
                for axIdx, thisAx in enumerate(ax2d):
                    thisAx.plotLine.set_data(dataDF.loc[tMask, 'bin'], thisAx.fullTrace.loc[tMask])
                    if whatToPlot[axIdx] in ['amplitude']:
                        thisAx.plotLine.set_color(rgbaColor)
                    elif whatToPlot[axIdx] in ['mahalanobis']:
                        thisAx.plotLine.set_color(rgbaColorMahal)
                    elif 'data' in whatToPlot[axIdx]:
                        thisAx.plotLine.set_color(lfpColor)
                    else:
                        thisAx.plotLine.set_color('k')
                #
                mhData = np.stack([dataFor3D[0:3, idx], baseline_cov.location_], axis=1)
                mhLine.set_data(mhData[0:2, :])
                mhLine.set_3d_properties(mhData[2, :])
                mhLine.set_color(rgbaColorMahal)
                #
                cometHead.set_data(dataFor3D[0:2, idx])
                cometHead.set_3d_properties(dataFor3D[2, idx])
                cometHead.set_color(lfpColor)
                cometHead.set_markersize(aniMarkerKws['markersize'])
                #
                if idx >= tailLength:
                    for ptIdx in range(tailLength):
                        colorVal = dataDF['amplitude'].iloc[idx - ptIdx]
                        # rgbaColor = np.asarray(cMap(cMapScaleFun(colorVal)))
                        rgbaColor = np.asarray(lfpColor.tolist() + [1.])
                        rgbaColor[3] = (tailLength - ptIdx) / tailLength
                        #
                        cometTailLines[ptIdx].set_data(dataFor3D[0:2, idx - ptIdx - 1:idx - ptIdx + 1])
                        cometTailLines[ptIdx].set_3d_properties(dataFor3D[2, idx - ptIdx - 1:idx - ptIdx + 1])
                        cometTailLines[ptIdx].set_color(rgbaColor)
                if progBar is not None:
                    progBar.update(1)
                return

            ani = animation.FuncAnimation(
                fig, animate, frames=nFrames,
                interval=int(1e3 / fps), blit=False)
            saveToFile = False
            if saveToFile:
                slowFactor = 2
                writer = FFMpegWriter(
                    fps=slowFactor * int(fps), metadata=dict(artist='Me'), bitrate=3600)
                ani.save(aniPath, writer=writer)
            showNow = True
            if showNow:
                plt.show()
            else:
                plt.close()
            print('made animation for {}'.format(maskDict['label']))
    ############################################
    savingResults = False
    if savingResults:
        outputDatasetName = 'Synthetic_{}_df_{}'.format(
            loadingMeta['arguments']['window'], iteratorSuffix)
        outputLoadingMeta = loadingMeta.copy()
        outputLoadingMetaPath = os.path.join(
            dataFramesFolder,
            outputDatasetName + '_' + arguments['selectionNameRhs'] + '_meta.pickle'
            )
        cvIterator = tdr.trainTestValidationSplitter(
            dataDF=toyLhsDF.loc[restrictMask, :], **cvKWArgs)
        #
        outputLoadingMeta['iteratorsBySegment'] = [cvIterator]
        outputLoadingMeta['iteratorOpts'] = iteratorOpts
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
            for idxItem in maskDF.index]
        maskParamsStr = [
            '{}'.format(idxItem).replace("'", '')
            for idxItem in maskParams]
        maskDF.loc[:, 'maskName'] = maskParamsStr
        maskDF.set_index('maskName', append=True, inplace=True)
        rhsOutputLoadingMeta = outputLoadingMeta.copy()
        rhsOutputLoadingMeta['arguments'] = arguments
        rhsOutputLoadingMeta['arguments']['selectionName'] = arguments['selectionNameRhs']
        hf.exportNormalizedDataFrame(
            dataDF=toyRhsDF.loc[restrictMask, :], loadingMeta=rhsOutputLoadingMeta, featureInfoMask=maskDF,
            # arguments=arguments, selectionName=arguments['selectionNameRhs'],
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
        lhsOutputLoadingMeta = outputLoadingMeta.copy()
        lhsOutputLoadingMeta['arguments'] = arguments
        lhsOutputLoadingMeta['arguments']['selectionName'] = arguments['selectionNameLhs']
        hf.exportNormalizedDataFrame(
            dataDF=toyLhsDF.loc[restrictMask, :], loadingMeta=lhsOutputLoadingMeta, featureInfoMask=maskDF,
            # arguments=arguments, selectionName=arguments['selectionNameLhs'],
            dataFramesFolder=dataFramesFolder, datasetName=outputDatasetName,
            )