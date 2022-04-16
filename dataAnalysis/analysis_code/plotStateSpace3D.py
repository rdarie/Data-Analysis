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
    --lowNoise                             reduce noise level? [default: False]
    --forceField                           plot some quivers [default: False]
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
    --transformerNameLhs=transformerNameLhs    how to restrict channels?
    --transformerNameRhs=transformerNameRhs    how to restrict channels?
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
from sklearn.decomposition import PCA, IncrementalPCA
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
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 9,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
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
    designMatrixDatasetName = '{}_{}_{}_{}_regression_design_matrices'.format(
        arguments['datasetNameLhs'], arguments['selectionNameLhs'], arguments['selectionNameRhs'],
        arguments['transformerNameRhs'])
    designMatrixPath = os.path.join(
        dataFramesFolder,
        designMatrixDatasetName + '.h5'
        )
    with open(loadingMetaPathLhs, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        iteratorOpts = loadingMeta['iteratorOpts']
    lhsDF = pd.read_hdf(designMatrixPath, 'lhsDF')
    # lhsDF = pd.read_hdf(lhsDatasetPath, '/{}/data'.format(arguments['selectionNameLhs']))
    rhsDF = pd.read_hdf(rhsDatasetPath, '/{}/data'.format(arguments['selectionNameRhs']))
    lhsMasks = pd.read_hdf(lhsDatasetPath, '/{}/featureMasks'.format(arguments['selectionNameLhs']))
    #
    trialInfo = lhsDF.index.to_frame().reset_index(drop=True)
    #
    lhsDF.reset_index(drop=True, inplace=True)
    lhsDF.columns = lhsDF.columns.get_level_values('feature')
    rhsDF.reset_index(drop=True, inplace=True)
    rhsDF.columns = rhsDF.columns.get_level_values('feature')
    lhsDF.loc[:, 'velocity'] = np.sqrt(lhsDF['vx'] ** 2 + lhsDF['vy'] ** 2)
    lhsDF.rename({'a': 'amplitude', 'r': 'RateInHz'}, axis='columns', inplace=True)
    #
    trialInfo.loc[:, 'moving'] = (lhsDF['velocity'] > 0.25).to_numpy()
    trialInfo.loc[:, 'limbState'] = lhsDF.apply(lambda x: 'moving' if x['velocity'] > 0.25 else 'baseline', axis='columns')
    trialInfo.loc[:, 'activeElectrode'] = trialInfo['electrode'].copy()
    trialInfo.loc[lhsDF['amplitude'] == lhsDF['amplitude'].min(), 'activeElectrode'] = 'NA'
    trialInfo.loc[:, 'limbState x activeElectrode'] = trialInfo.apply(
        lambda x: '{} {}'.format(x['limbState'], x['activeElectrode']), axis='columns')
    trialInfo.loc[:, 'pedalMovementCat x electrode'] = trialInfo.apply(
        lambda x: '{} {}'.format(x['pedalMovementCat'], x['electrode']), axis='columns')
    nDim = 3
    pca = PCA(n_components=nDim, svd_solver='full')
    toyRhsDF = pd.DataFrame(
        pca.fit_transform(rhsDF), index=rhsDF.index,
        columns=['data{}'.format(cN) for cN in range(nDim)]
        )
    tInfoCols = [
        'electrode', 'pedalMovementCat',
        'RateInHz', 'bin', 'limbState', 'activeElectrode',
        'limbState x activeElectrode', 'pedalMovementCat x electrode',]
    rhsPlotDF = pd.concat([toyRhsDF, trialInfo.loc[:, tInfoCols], lhsDF.loc[:, ['velocity', 'amplitude', 'RateInHz']]], axis='columns')
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
        'data{}'.format(i): 'LFP ch. #{}'.format(i)
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
        'baseline -E05+E16': sns.utils.alter_color(
            singlesPalette.loc['S', :].to_list(), l=-.3),
        'baseline -E09+E16': sns.utils.alter_color(
            singlesPalette.loc['S', :].to_list(), l=.3),
        'moving NA': singlesPalette.loc['M', :].to_list(),
        'moving -E05+E16': sns.utils.alter_color(
            singlesPalette.loc['C', :].to_list(), l=-.3),
        'moving -E09+E16': sns.utils.alter_color(
            singlesPalette.loc['C', :].to_list(), l=.3),
        }
    prettyNameLookup.update({
        'baseline NA': 'Baseline',
        'baseline -E05+E16': 'Stim.-only (E05)',
        'baseline -E09+E16': 'Stim.-only (E09)',
        'moving NA': 'Movement-only',
        'moving -E05+E16': 'Stim.-movement (E05)',
        'moving -E09+E16': 'Stim.-movement (E09)',
        })
    limbStateElectrodeMarkerDict = {
        'baseline NA': r'$\clubsuit$',
        'baseline -E05+E16': 'o',
        'baseline -E09+E16': 'd',
        'moving NA': 's',
        'moving -E05+E16': '*',
        'moving -E09+E16': 'D',
        }
    limbStateElectrodeNames = [
        'baseline NA', 'moving NA', 'baseline -E05+E16', 'baseline -E09+E16',
        'moving -E05+E16', 'moving -E09+E16'
        ]
    defaultAxisOrientation = dict(azim=75., elev=20.)

    ampPaletteStr = "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    mahalPaletteStr = "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1"
    lfpColor = np.asarray(sns.color_palette('Set3')[4])
    mhDistColor = np.asarray(sns.color_palette('Set3')[6])
    markerStyles = ['o', 'd', 's']
    scatterOpts = dict(
        alpha=0.5, linewidths=0, s=.8 ** 2,
        rasterized=True)
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
        label='LFP state')
    legendData.loc['mahalanobis'] = Line2D(
        [0], [0], linestyle='-', linewidth=2, color=mhDistColor,
        label='Mh. dist.')
    #
    legendData.loc['blank'] = Patch(alpha=0, linewidth=0, label=' ')
    #
    for key, color in covPatternPalette.items():
        legendData[key] = Patch(
            facecolor=color, label=prettyNameLookup[key])
    #
    legendData.loc['ellipsoid'] = Patch(
        alpha=0, linewidth=0, label='Covariance ellipsoid')
    maskForBaseline = trialInfo['limbState x activeElectrode'] == 'baseline NA'
    pdb.set_trace()
    baseline_cov = LedoitWolf().fit(
        toyRhsDF.loc[maskForBaseline, :])
    rhsPlotDF.loc[:, 'mahalanobis'] = np.sqrt(baseline_cov.mahalanobis(toyRhsDF))
    if arguments['lowNoise']:
        rhsPlotDF.loc[:, 'mahalanobis'] = rhsPlotDF.loc[:, 'mahalanobis'] * 0.
    lOfMasksForBreakdown = [
        {
            'mask': trialInfo['electrode'].isin(['NA']).to_numpy() & trialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Baseline',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': trialInfo['electrode'].isin(['NA']).to_numpy() & trialInfo['pedalMovementCat'].isin(['outbound', 'return']).to_numpy(),
            'label': 'Movement-only',
            'ellipsoidsToPlot': ['baseline NA', 'moving NA']},
        {
            'mask': trialInfo['limbState x activeElectrode'].isin(['baseline -E05+E16', 'baseline NA']).to_numpy(),
            'label': 'Stim.-only (E05)',
            'ellipsoidsToPlot': ['baseline NA', 'baseline -E05+E16']},
        {
            'mask': trialInfo['limbState x activeElectrode'].isin(['baseline -E09+E16', 'baseline NA']).to_numpy(),
            'label': 'Stim.-only (E09)',
            'ellipsoidsToPlot': ['baseline NA', 'baseline -E09+E16']},
        {
            'mask': trialInfo['limbState x activeElectrode'].isin(['baseline -E09+E16', 'baseline -E05+E16', 'baseline NA', 'moving NA']).to_numpy(),
            'label': 'Stim.-only (electrodes X, Y)',
            'ellipsoidsToPlot': ['baseline NA', 'moving NA', 'baseline -E09+E16', 'baseline -E05+E16']},
        ]
    lOfMasksForAnim = [
        {
            'mask': trialInfo['electrode'].isin(['NA']).to_numpy() & trialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Baseline',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': trialInfo['electrode'].isin(['NA']).to_numpy() & trialInfo['pedalMovementCat'].isin(['outbound']).to_numpy(),
            'label': 'Movement-only',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': trialInfo['electrode'].isin(['-E05+E16']).to_numpy() & trialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Stim.-only (E05)',
            'ellipsoidsToPlot': ['baseline NA']},
        {
            'mask': trialInfo['electrode'].isin(['-E09+E16']).to_numpy() & trialInfo['pedalMovementCat'].isin(['NA']).to_numpy(),
            'label': 'Stim.-only (E09)',
            'ellipsoidsToPlot': ['baseline NA']},
        ]
    lOfMasksByLimbStateActiveElectrode = [
        {
            'mask': (trialInfo['limbState x activeElectrode'] == lsae).to_numpy(),
            'label': lsae,
            }
        for lsae in trialInfo['limbState x activeElectrode'].unique()
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
        'baseline -E05+E16': .3,
        'baseline -E09+E16': .3,
        'moving NA': .3,
        'moving -E05+E16': .3,
        'moving -E09+E16': .3,
        }
    empCovMats = {}
    #
    for name, group in rhsPlotDF.groupby('limbState x activeElectrode'):
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

    ellipsoidDict = ellipsoidEmpiricalDict
    #
    def drawEllipsoidAxes(theAx, theName):
        x0, y0, z0 = empCovMats[theName].location_
        sx, sy, sz = np.sqrt(empCovMats[theName].covariance_eigenvalues) * 2
        covarEigPlotOpts = dict(zorder=2.1, linewidth=.75)
        #
        dx, dy, dz = sx * empCovMats[theName].covariance_u[:, 0]
        theAx.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color='c', **covarEigPlotOpts)
        dx, dy, dz = sy * empCovMats[theName].covariance_u[:, 1]
        theAx.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color='m', **covarEigPlotOpts)
        dx, dy, dz = sz * empCovMats[theName].covariance_u[:, 2]
        theAx.plot([x0, x0 + dx], [y0, y0 + dy], [z0, z0 + dz], color='y', **covarEigPlotOpts)
        return

    extent = (
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + ['velocity', 'amplitude', 'RateInHz', 'mahalanobis']].quantile(1 - 1e-2) -
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + ['velocity', 'amplitude', 'RateInHz', 'mahalanobis']].quantile(1e-2))
    # xExtent, yExtent, zExtent = extent['data0'], extent['data1'], extent['data2']
    masterExtent = extent.loc[toyRhsDF.columns.to_list()].max() * 0.5
    xExtent, yExtent, zExtent = masterExtent, masterExtent, masterExtent
    midPoints = (
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + ['velocity', 'amplitude', 'RateInHz', 'mahalanobis']].quantile(1 - 1e-2) +
        rhsPlotDF.loc[:, toyRhsDF.columns.to_list() + ['velocity', 'amplitude', 'RateInHz', 'mahalanobis']].quantile(1e-2)) / 2
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
            xLab, yLab, zLab = 'LFP ch. #1 (a.u.)', 'LFP ch. #2 (a.u.)', 'LFP ch. #3 (a.u.)'
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

    pdfPath = os.path.join(
        figureOutputFolder,
        'plots3d_{}{}.pdf'.format(iteratorSuffix, specialSuffix))
    with PdfPages(pdfPath) as pdf:
        plotAllScatter = False
        if plotAllScatter:
            fig = plt.figure()
            fig.set_size_inches((4, 4))
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
                **scatterOpts)
            formatTheAxes(ax)
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
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
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig()
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        ########
        if iteratorSuffix in ['a', 'b', 'g']:
            plotRelPlots = False
            if plotRelPlots:
                tempCols = [cN for cN in toyRhsDF.columns] + [
                    'limbState x activeElectrode', 'pedalMovementCat x electrode']
                rhsPlotDFStack = pd.DataFrame(
                    rhsPlotDF.loc[:, tempCols].to_numpy(),
                    columns=tempCols,
                    index=pd.MultiIndex.from_frame(trialInfo.loc[
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
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
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
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
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
                    index=pd.MultiIndex.from_frame(trialInfo.loc[
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
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                #
                pdf.savefig()
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                ###
            plotCovMats = False
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
                fig.tight_layout(pad=styleOpts['tight_layout.pad'])
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
                            if not arguments['lowNoise']:
                                drawEllipsoidAxes(ax, name)
                            if arguments['forceField']:
                                drawForceField(ax)
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
                                    **scatterOpts)
                        else:
                            ax.scatter(
                                rhsPlotDF.loc[thisMask, 'data0'],
                                rhsPlotDF.loc[thisMask, 'data1'],
                                rhsPlotDF.loc[thisMask, 'data2'],
                                # cmap=sns.color_palette(ampPaletteStr, as_cmap=True),
                                # c=rhsPlotDF.loc[thisMask, 'electrodeInfluence'],
                                c=[covPatternPalette[lse] for lse in rhsPlotDF.loc[thisMask, 'limbState x activeElectrode']],
                                # c=lfpColor, s=group['movementInfluence'],
                                **scatterOpts)
                        theseLegendEntries = legendData.loc[
                            ['ellipsoid'] + ellipsoidsToPlot +
                            ['blank'] + legendPcaDirs]
                        ax.legend(
                            handles=theseLegendEntries.to_list(),
                            loc='center right')
                        # fig.suptitle(
                        #     maskDict['label'] + '\n\n' + '\n'.join(rhsPlotDF.loc[thisMask, 'limbState x activeElectrode'].unique()))
                        # figTitle = fig.suptitle(maskDict['label'], y=0.9)
                        ##
                        # zF = 1.8 if (maskDict['label'] == 'Baseline') else 1.
                        # formatTheAxes(ax, zoomFactor=zF, showOrigin=False)
                        ##
                        formatTheAxes(ax, zoomFactor=1.2)
                        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                        pdf.savefig(
                            bbox_inches='tight', pad_inches=0, # bbox_extra_artists=[figTitle]
                            )
                        if arguments['showFigures']:
                            plt.show()
                        else:
                            plt.close()
    ############################################
    makeAnimations = True
    if makeAnimations:
        slowFactor = 10
        fps = int(iteratorOpts['forceBinInterval'] ** (-1) / slowFactor)
        tailLength = 3
        aniLineKws = {'linestyle': '-', 'linewidth': 1.5}
        cometLineKws = {'linestyle': '-', 'linewidth': 1.}
        mhLineKws = {'linestyle': '-', 'linewidth': 2.}
        aniMarkerKws = {'marker': 'o', 'markersize': 2, 'linewidth': 0}
        whatToPlot = ['data0', 'data1', 'data2', 'amplitude']
        if not arguments['lowNoise']:
            whatToPlot += ['mahalanobis']
        cMap = sns.color_palette(ampPaletteStr, as_cmap=True)
        cMapScaleFun = lambda am: cMap.colors.shape[0] * min(1, max(0, (am - rhsPlotDF['amplitude'].min()) / (rhsPlotDF['amplitude'].max() - rhsPlotDF['amplitude'].min())))
        cMapMahal = sns.color_palette(mahalPaletteStr, as_cmap=True)
        cMapMahalScaleFun = lambda am: cMapMahal.colors.shape[0] * min(1, max(0, (am - rhsPlotDF['mahalanobis'].min()) / (
                    rhsPlotDF['mahalanobis'].max() - rhsPlotDF['mahalanobis'].min())))
        for maskIdx, maskDict in enumerate(lOfMasksForAnim):
            if maskIdx not in [mi for mi in range(len(lOfMasksForAnim))]:
                continue
            print('Starting to generate animation for {}'.format(maskDict['label']))
            aniPath = os.path.join(
                figureOutputFolder,
                'plots3d_{}_mask_{}{}.mp4'.format(
                    iteratorSuffix, maskIdx, specialSuffix))
            fig = plt.figure()
            nRows = len(whatToPlot)
            nCols = 2
            inchesPerRow = .8
            fig.set_size_inches((inchesPerRow * nCols * nRows, inchesPerRow * nRows))
            gs = GridSpec(
                nRows, nCols,
                width_ratios=[3, 2],
                height_ratios=[1 for i in range(nRows)],
                hspace=0.02, wspace=0.1
                )
            ax3d = fig.add_subplot(gs[:, 0], projection='3d')
            ax3d.set_proj_type('ortho')
            ax2d = [fig.add_subplot(gs[i, 1]) for i in range(nRows)]
            #
            chooseSomeTrials = trialInfo.loc[maskDict['mask'] & (trialInfo['trialAmplitude'] == trialInfo.loc[maskDict['mask'], 'trialAmplitude'].max()), 'trialUID'].unique()[:3]
            thisMask = maskDict['mask'] & trialInfo['trialUID'].isin(chooseSomeTrials)
            dataDF = rhsPlotDF.loc[thisMask, :].copy()
            dataFor3D = dataDF.loc[:, ['data0', 'data1', 'data2']].T.to_numpy()
            dataDF.loc[:, 'trialUID'] = trialInfo.loc[thisMask, 'trialUID']
            nFrames = min(500, dataDF.shape[0])
            progBar = tqdm(total=nFrames, mininterval=30., maxinterval=120.)

            customMidPoints = (
                  toyRhsDF.loc[maskDict['mask'], :].quantile(1 - 1e-2) +
                  toyRhsDF.loc[maskDict['mask'], :].quantile(1e-2)) / 2
            customExtents = (
                    toyRhsDF.loc[maskDict['mask'], :].quantile(1 - 1e-2) -
                    toyRhsDF.loc[maskDict['mask'], :].quantile(1e-2))

            for name in maskDict['ellipsoidsToPlot']:
                ax3d.plot_surface(
                    *ellipsoidDict[name],
                    color=sns.utils.alter_color(covPatternPalette[name], l=.5),
                    **ellipsoidPlotOpts)
                if not arguments['lowNoise']:
                    drawEllipsoidAxes(ax3d, name)
                if arguments['forceField']:
                    drawForceField(
                        ax3d,
                        customMidPoints=[
                            customMidPoints['data0'],
                            customMidPoints['data1'],
                            customMidPoints['data2']],
                        customExtents=[
                            customExtents.max(),
                            customExtents.max(),
                            customExtents.max()],
                            )
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
            # fig.suptitle(figTitleStr)
            # figTitleStr = maskDict['label']
            # ax3d.set_title(figTitleStr)
            # zF = 1.9 if name == 'baseline NA' else 1.2
            # formatTheAxes(ax3d, zoomFactor=zF)
            #
            formatTheAxes(
                ax3d,
                customMidPoints=[
                    customMidPoints['data0'],
                    customMidPoints['data1'],
                    customMidPoints['data2']],
                customExtents=[
                    customExtents.max() * 0.5,
                    customExtents.max() * 0.5,
                    customExtents.max() * 0.5],
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
                if not arguments['lowNoise']:
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
                        rgbaColor[3] = (tailLength - ptIdx) / (2 * tailLength)
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
                writer = FFMpegWriter(
                    fps=int(fps), metadata=dict(artist='Me'), bitrate=7200)
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