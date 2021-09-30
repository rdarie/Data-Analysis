"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose=verbose                        print diagnostics?
    --debugging                              print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
"""

import logging
logging.captureWarnings(True)
import matplotlib, os
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
from datetime import datetime
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
from scipy.stats import mode
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklego.preprocessing import PatsyTransformer

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
exec('from dataAnalysis.analysis_code.regression_parameters_{} import *'.format(arguments['datasetName'].split('_')[-1]))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
from scipy.spatial import distance
import dill as pickle
import gc
import patsy
from itertools import product
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions

import control as ctrl
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 5,
        "axes.labelsize": 7,
        "axes.titlesize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "legend.title_fontsize": 7,
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
    'figure.titlesize': 7,
    'font.family': "Nimbus Sans",
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1,  # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV


print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')


# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_ra', 'plotting': True,
        'showFigures': False, 'alignFolderName': 'motion', 'processAll': True,
        'verbose': '1', 'debugging': False, 'estimatorName': 'enr_fa_ta',
        'blockIdx': '2', 'exp': 'exp202101281100'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    
'''

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'regression')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder)
#
datasetName = arguments['datasetName']
fullEstimatorName = '{}_{}'.format(
    arguments['estimatorName'], arguments['datasetName'])
#
estimatorsSubFolder = os.path.join(
    analysisSubFolder, 'estimators')
dataFramesFolder = os.path.join(
    analysisSubFolder, 'dataframes')
datasetPath = os.path.join(
    dataFramesFolder,
    datasetName + '.h5'
    )
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
estimatorMetaDataPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '_meta.pickle'
    )
with open(estimatorMetaDataPath, 'rb') as _f:
    estimatorMeta = pickle.load(_f)
#
loadingMetaPath = estimatorMeta['loadingMetaPath']
with open(loadingMetaPath, 'rb') as _f:
    loadingMeta = pickle.load(_f)
    iteratorOpts = loadingMeta['iteratorOpts']
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
#
for hIdx, histOpts in enumerate(addHistoryTerms):
    locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
thisEnv = patsy.EvalEnvironment.capture()

iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
# cv_kwargs = loadingMeta['cv_kwargs'].copy()
cvIterator = iteratorsBySegment[0]
lastFoldIdx = cvIterator.n_splits
#
selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
#
lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
lhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsMasksInfo')
###
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
rhsMasksInfo = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhsMasksInfo')
#
if processSlurmTaskCountPLS is not None:
    ################ collect estimators and scores
    AList = []
    BList = []
    CList = []
    DList = []
    HList = []
    eigList = []
    for workerIdx in range(processSlurmTaskCountPLS):
        thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(workerIdx))
        try:
            AList.append(pd.read_hdf(thisEstimatorPath, 'A'))
            BList.append(pd.read_hdf(thisEstimatorPath, 'B'))
            CList.append(pd.read_hdf(thisEstimatorPath, 'C'))
            DList.append(pd.read_hdf(thisEstimatorPath, 'D'))
            HList.append(pd.read_hdf(thisEstimatorPath, 'H'))
            eigList.append(pd.read_hdf(thisEstimatorPath, 'eigenvalues'))
            print('oaded state transition froomn {}'.format(thisEstimatorPath))
        except Exception:
            traceback.print_exc()
    ADF = pd.concat(AList)
    BDF = pd.concat(BList)
    CDF = pd.concat(CList)
    DDF = pd.concat(DList)
    HDF = pd.concat(HList)
    eigDF = pd.concat(eigList)
else:
    with pd.HDFStore(estimatorPath) as store:
        ADF = pd.read_hdf(store, 'A')
        BDF = pd.read_hdf(store, 'B')
        CDF = pd.read_hdf(store, 'C')
        DDF = pd.read_hdf(store, 'D')
        eigDF = pd.read_hdf(store, 'eigenvalues')
        eigValPalette = pd.read_hdf(store, 'eigValPalette')
#
eigValTypes = ['oscillatory decay', 'pure decay', 'oscillatory growth', 'pure growth']
eigValColors = sns.color_palette('Set2')
eigValPaletteDict = {}
eigValColorAlpha = 0.5
for eIx, eType in enumerate(eigValTypes):
    eigValPaletteDict[eType] = tuple([col for col in eigValColors[eIx]] + [eigValColorAlpha])
eigValPalette = pd.Series(eigValPaletteDict)
eigValPalette.index.name = 'eigValType'
eigValPalette.name = 'color'
eigValPalette.to_hdf(estimatorPath, 'eigValPalette')
# check eigenvalue persistence over reduction of the state space dimensionality
eigDF.loc[:, 'freqBandName'] = eigDF.reset_index()['rhsMaskIdx'].map(rhsMasksInfo['freqBandName']).to_numpy()
eigDF.loc[:, 'designFormula'] = eigDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['designFormula']).to_numpy()
eigDF.loc[:, 'designFormulaLabel'] = eigDF.reset_index()['designFormula'].apply(lambda x: x.replace(' + ', ' +\n')).to_numpy()
eigDF.loc[:, 'fullFormula'] = eigDF.reset_index()['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr']).to_numpy()
eigDF.loc[:, 'fullFormulaLabel'] = eigDF.reset_index()['fullFormula'].apply(lambda x: x.replace(' + ', ' +\n')).to_numpy()
########################################
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalue_reduction'))
if False:
    with PdfPages(pdfPath) as pdf:
        for name, thisPlotEig in eigDF.groupby(['fullFormula', 'freqBandName']):
            height, width = 3, 3
            aspect = width / height
            g = sns.relplot(
                col='stateNDim', col_wrap=3,
                x='real', y='imag', hue='eigValType',
                height=height, aspect=aspect,
                facet_kws={'margin_titles': True},
                palette=eigValPalette.to_dict(),
                kind='scatter', data=thisPlotEig.reset_index(), rasterized=True, edgecolor=None)
            g.suptitle('design: {} freqBand: {}'.format(*name))
            for ax in g.axes.flatten():
                c = Circle((0, 0), 1, ec=(0, 0, 0, 1.), fc=(0, 0, 0, 0))
                ax.add_artist(c)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
#
plotEig = eigDF.loc[eigDF['nDimIsMax'], :]
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalues'))
with PdfPages(pdfPath) as pdf:
    height, width = 2, 2
    aspect = width / height
    for name, eigGroup in plotEig.groupby(['lhsMaskIdx', 'fullFormula']):
        lhsMaskIdx, fullFormula = name
        # if lhsMaskIdx not in [29]:
        #     continue
        g = sns.relplot(
            col='freqBandName',
            x='real', y='imag', hue='eigValType',
            height=height, aspect=aspect,
            palette=eigValPalette.to_dict(),
            facet_kws={'margin_titles': True},
            rasterized=True, edgecolor=None,
            kind='scatter', data=eigGroup)
        for ax in g.axes.flatten():
            c = Circle(
                (0, 0), 1,
                ec=(0, 0, 0, 0.25),
                fc=(0, 0, 0, 0))
            ax.add_artist(c)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, 0, 1])
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1, 0, 1])
        asp.reformatFacetGridLegend(
            g, titleOverrides={
                'eigValType': 'Eigenvalue type'},
            contentOverrides={},
            styleOpts=styleOpts)
        g.set_axis_labels('Real', 'Imaginary')
        g.resize_legend(adjust_subtitles=True)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        #
        fig, ax = plt.subplots(2, 1, figsize=(2, 1.5))
        commonOpts = dict(element='step', stat="probability")
        sns.histplot(
            x='tau', data=eigGroup.query('tau > 0'),
            ax=ax[0], color=eigValPalette['oscillatory decay'], **commonOpts)
        sns.histplot(
            x='chi', data=eigGroup, ax=ax[1],
            color=eigValPalette['pure decay'], **commonOpts)
        ax[0].set_xlabel('Oscillation period (sec)')
        ax[0].set_xlim(eigGroup.query('tau > 0')['tau'].quantile([0, .85]).to_list())
        ax[1].set_xlabel('Decay time constant (sec)')
        ax[1].set_xlim(eigGroup['chi'].quantile([0, .99]).to_list())
        sns.despine(fig)
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        thisB = BDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        if not thisB.empty:
            fig, ax = plt.subplots(figsize=(2, 2))
            colsToEval = [cN for cN in thisB.columns if '[NA]' not in cN]
            BCosineSim = pd.DataFrame(np.nan, index=colsToEval, columns=colsToEval)
            mask = np.zeros_like(BCosineSim)
            mask[np.triu_indices_from(mask)] = True
            for cNr in colsToEval:
                for cNc in colsToEval:
                    BCosineSim.loc[cNr, cNc] = distance.cosine(thisB[cNr], thisB[cNc])
            sns.heatmap(BCosineSim, mask=mask, vmin=0, vmax=1, square=True, ax=ax)
            ax.set_title('Cosine distance between input directions')
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            print('Saving to {}'.format(pdfPath))
        '''thisA = ADF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        
        thisC = CDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        thisD = DDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        sys = ctrl.StateSpace(thisA, thisB, thisC, thisD, dt=binInterval)
        f = np.linspace(1e-3, 100, 1000)
        mag, phase, omega = ctrl.freqresp(sys, 2 * np.pi * f)
        fig, ax = plt.subplots(sys.ninputs, sys.noutputs, sharey=True)
        for inpIdx in range(sys.ninputs):
            for outIdx in range(sys.noutputs):
                ax[inpIdx, outIdx].semilogy(f, mag[outIdx, inpIdx, :])'''
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'OKID_ERA'))
with PdfPages(pdfPath) as pdf:
    for name, thisH in HDF.groupby(['lhsMaskIdx', 'design', 'rhsMaskIdx']):
        lhsMaskIdx, designFormula, rhsMaskIdx = name
        # lhsMasksInfo.loc[lhsMaskIdx, :]
        nLags = int(lhsMasksInfo.loc[lhsMaskIdx, 'historyLen'] / binInterval)
        plotH = thisH.xs(lastFoldIdx, level='fold').dropna(axis='columns')
        u, s, vh = np.linalg.svd(plotH, full_matrices=False)
        optThresh = tdr.optimalSVDThreshold(plotH) * np.median(s[:int(nLags)])
        optNDim = (s > optThresh).sum()
        stateSpaceNDim = min(optNDim, u.shape[0])
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(s)
        ax.set_title('singular values of Hankel matrix (ERA)\n{}'.format(lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr']))
        ax.set_ylabel('s')
        ax.axvline(stateSpaceNDim, c='b', label='state space model order: {}'.format(stateSpaceNDim))
        ax.axvline(nLags, c='g', label='AR(p) order: {}'.format(nLags))
        ax.set_xlabel('Count')
        ax.set_xlim([-1, stateSpaceNDim * 10])
        ax.legend()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        print('Saving to {}'.format(pdfPath))

#
'''pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'A_eigenvalues'))
with PdfPages(pdfPath) as pdf:
    height, width = 3, 3
    aspect = width / height
    plotMask = plotEig['designFormula'] == lOfDesignFormulas[1]
    g = sns.relplot(
        row='designFormulaLabel', col='freqBandName',
        x='real', y='imag', hue='eigValType',
        height=height, aspect=aspect,
        kind='scatter',
        facet_kws={'margin_titles': True}, edgecolor=None, rasterized=True,
        data=plotEig.loc[plotMask, :])
    g.suptitle('design: {} freqBand: {}'.format(*name))
    for ax in g.axes.flatten():
        c = Circle((0,0), 1, ec=(0, 0, 0, 1.), fc=(0, 0, 0, 0))
        ax.add_artist(c)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
'''