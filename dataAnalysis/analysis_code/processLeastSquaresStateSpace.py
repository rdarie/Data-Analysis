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
    --eraMethod=eraMethod                    append a name to the resulting blocks? [default: ERA]
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
from itertools import product, chain
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions

import control as ctrl
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 3.,
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
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc=snsRCParams)
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
transferFuncPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '_{}_tf.h5'.format(arguments['eraMethod'])
    )
loadingMetaPath = estimatorMeta['loadingMetaPath']
with open(loadingMetaPath, 'rb') as _f:
    loadingMeta = pickle.load(_f)
    iteratorOpts = loadingMeta['iteratorOpts']
    binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
#
for hIdx, histOpts in enumerate(addEndogHistoryTerms):
    locals().update({'enhto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
for hIdx, histOpts in enumerate(addExogHistoryTerms):
    locals().update({'exhto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
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
if processSlurmTaskCount is not None:
    ################ collect estimators and scores
    AList = []
    BList = []
    CList = []
    DList = []
    HList = []
    KList = []
    eigList = []
    inputDrivenList = []
    untruncatedHEigenValsList = []
    ctrbList = []
    for workerIdx in range(processSlurmTaskCount):
        thisTFPath = transferFuncPath.replace('_tf.h5', '_{}_tf.h5'.format(workerIdx))
        try:
            AList.append(pd.read_hdf(thisTFPath, 'A'))
            BList.append(pd.read_hdf(thisTFPath, 'B'))
            CList.append(pd.read_hdf(thisTFPath, 'C'))
            DList.append(pd.read_hdf(thisTFPath, 'D'))
            HList.append(pd.read_hdf(thisTFPath, 'H'))
            KList.append(pd.read_hdf(thisTFPath, 'K'))
            eigList.append(pd.read_hdf(thisTFPath, 'eigenvalues'))
            inputDrivenList.append(pd.read_hdf(thisTFPath, 'inputDriven'))
            untruncatedHEigenValsList.append(pd.read_hdf(thisTFPath, 'untruncatedHEigenVals'))
            ctrbList.append(pd.read_hdf(thisTFPath, 'ctrbObsvRanks'))
            print('Loaded state transition matrices from {}'.format(thisTFPath))
        except Exception:
            traceback.print_exc()
    ADF = pd.concat(AList)
    BDF = pd.concat(BList)
    CDF = pd.concat(CList)
    DDF = pd.concat(DList)
    HDF = pd.concat(HList)
    KDF = pd.concat(KList)
    eigDF = pd.concat(eigList)
    inputDrivenDF = pd.concat(inputDrivenList)
    untruncatedHEigDF = pd.concat(untruncatedHEigenValsList)
    ctrbObsvRanksDF = pd.concat(ctrbList)
else:
    with pd.HDFStore(transferFuncPath) as store:
        ADF = pd.read_hdf(store, 'A')
        BDF = pd.read_hdf(store, 'B')
        CDF = pd.read_hdf(store, 'C')
        DDF = pd.read_hdf(store, 'D')
        eigDF = pd.read_hdf(store, 'eigenvalues')
        inputDrivenDF = pd.read_hdf(store, 'inputDriven')
        untruncatedHEigDF = pd.read_hdf(store, 'untruncatedHEigenVals')
        eigValPalette = pd.read_hdf(store, 'eigValPalette')
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
eigDF.loc[:, 'complexS'] = eigDF['complex'].apply(np.log) / binInterval
eigDF.loc[:, 'realS'] = eigDF['complexS'].apply(np.real)
eigDF.loc[:, 'imagS'] = eigDF['complexS'].apply(np.imag) / (2 * np.pi)
#
plotEig = eigDF.loc[eigDF['nDimIsMax'], :]
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}_{}.pdf'.format(expDateTimePathStr, fullEstimatorName, 'OKID_{}'.format(arguments['eraMethod'])))
with PdfPages(pdfPath) as pdf:
    thisCtrb = ctrbObsvRanksDF.set_index(['matrix', 'inputSubset'], append=True).melt(value_name='rank', ignore_index=False).reset_index()
    thisCtrb = thisCtrb.loc[thisCtrb['inputSubset'] != 'NA', :]
    def makeNameColumns(x):
        if x['variable'] == 'nDim':
            return 'system'
        else:
            return '{} ({})'.format(x['matrix'], x['inputSubset'])
    thisCtrb.loc[:, 'rankOf'] = thisCtrb.apply(makeNameColumns, axis='columns')
    thisCtrb.loc[:, 'fullFormulaDescr'] = thisCtrb['lhsMaskIdx'].map(lhsMasksInfo['fullFormulaDescr'])
    height, width = 2, 2
    aspect = width / height
    g = sns.catplot(
        data=thisCtrb,
        x='fullFormulaDescr', y='rank',
        hue='rankOf',
        kind='bar',
        height=height, aspect=aspect
        )
    g.set_xticklabels(rotation=-10, ha='left', va='top')
    g.tight_layout(pad=styleOpts['tight_layout.pad'])
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    if arguments['showFigures']:
        plt.show()
    else:
        plt.close()
    height, width = 2, 2
    aspect = width / height
    for name, eigGroup in plotEig.groupby(['lhsMaskIdx', 'fullFormula']):
        lhsMaskIdx, fullFormula = name
        # if lhsMaskIdx not in [29]:
        # pdb.set_trace()
        eigGroupForPlot = eigGroup.xs(lastFoldIdx, level='fold')
        stateSpaceNDim = np.unique(eigGroupForPlot.index.get_level_values('stateNDim'))[0]
        # maskForScatter = (~np.isinf(eigGroup['tau'])) & (~np.isinf(eigGroup['chi']))
        maskForScatter = (~np.isnan(eigGroupForPlot['complex']))
        # maskForHist = (~np.isinf(eigGroup['tau'])) & (~np.isinf(eigGroup['chi'])) & (eigGroup['tau'] > binInterval) & (eigGroup['chi'] < 1.)
        if not maskForScatter.any():
            continue
        #     continue
        #############################################################################################################################
        ## Z plane
        g = sns.relplot(
            col='freqBandName',
            x='real', y='imag', hue='eigValType',
            height=height, aspect=aspect,
            palette=eigValPalette.to_dict(),
            facet_kws={'margin_titles': True},
            rasterized=True, edgecolor=None, marker='+', linewidth=1.,
            kind='scatter', data=eigGroupForPlot.loc[maskForScatter, :])
        for ax in g.axes.flatten():
            # captions = ['no decay', '1 DT', '2 DT', '4 DT', '8 DT']
            # for radIdx, radius in enumerate([1, np.exp(-1/2), np.exp(-1/4), np.exp(-1/8), np.exp(-1/16)]):
            captions = ['', ]
            for radIdx, radius in enumerate([1, ]):
                c = Circle(
                    (0, 0), radius,
                    ec=(0, 0, 0, 0.25),
                    fc=(0, 0, 0, 0))
                ax.add_artist(c)
                if  len(captions[radIdx]):
                    ax.text(0, radius, captions[radIdx], transform=ax.transData)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, 0, 1])
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1, 0, 1])
            ax.text(
                .95, .95, "Showing {}/{} eigenvalues".format(maskForScatter.sum(), maskForScatter.shape[0]),
                ha='right', va='top', transform=ax.transAxes)
        asp.reformatFacetGridLegend(
            g, titleOverrides={
                'eigValType': 'Eigenvalue type'},
            contentOverrides={},
            styleOpts=styleOpts)
        g.set_axis_labels('Real', 'Imaginary')
        g.suptitle(fullFormula)
        g.resize_legend(adjust_subtitles=True)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        #############################################################################################################################
        #############################################################################################################################
        ### S plane

        maskForTimeConstant = ~np.isinf(eigGroupForPlot['tau']) & (eigGroupForPlot['tau'] > binInterval).to_numpy()
        maskForPeriod = ~np.isinf(eigGroupForPlot['chi']) & (eigGroupForPlot['chi'] < 1.).to_numpy()
        maskForScatter = maskForTimeConstant & maskForPeriod
        if maskForScatter.any():
            tauLims = eigGroupForPlot.loc[maskForScatter, 'tau'].quantile([0, .99]).to_list()
            tauExt = tauLims[1] - tauLims[0]
            tauLims = [tauLims[0] - tauExt * 5e-2, tauLims[1] + tauExt * 5e-2]
            chiLims = eigGroupForPlot.loc[maskForScatter, 'chi'].quantile([0, .99]).to_list()
            chiExt = chiLims[1] - chiLims[0]
            chiLims = [chiLims[0] - chiExt * 5e-2, chiLims[1] + chiExt * 5e-2]
            #
            g = sns.relplot(
                col='freqBandName',
                x='tau', y='chi', hue='eigValType',
                height=height, aspect=aspect,
                palette=eigValPalette.to_dict(),
                facet_kws={'margin_titles': True},
                rasterized=True, edgecolor=None, marker='+', linewidth=1.,
                kind='scatter', data=eigGroupForPlot.loc[maskForScatter, :])
            asp.reformatFacetGridLegend(
                g, titleOverrides={
                    'eigValType': 'Eigenvalue type'},
                contentOverrides={},
                styleOpts=styleOpts)
            #
            for ax in g.axes.flatten():
                ax.text(
                    .95, .95, "Time constants for {}/{} eigenvalues".format(maskForScatter.sum(), maskForScatter.shape[0]),
                    ha='right', va='top', transform=ax.transAxes)
                ax.set_ylim(*chiLims)
                ax.set_xlim(*tauLims)
            #
            g.set_axis_labels('Time constant (sec)', 'Period (sec)')
            g.suptitle(fullFormula)
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #############################################################################################################################
            #
            fig, ax = plt.subplots(1, 2, figsize=(4, 2))
            commonOpts = dict(element='step', stat="probability", cumulative=True)
            sns.histplot(
                x='tau', data=eigGroupForPlot.loc[maskForScatter, :],
                ax=ax[0], color=eigValPalette['pure decay'], **commonOpts)
            ax[0].set_xlim(tauLims)
            ax[0].set_xlabel('Decay time constant (sec)')
            sns.histplot(
                x='chi', data=eigGroupForPlot.loc[maskForScatter, :], ax=ax[1],
                color=eigValPalette['oscillatory decay'], **commonOpts)
            ax[1].set_xlim(chiLims) #
            ax[1].set_xlabel('Oscillation period (sec)')
            sns.despine(fig)
            fig.suptitle(fullFormula)
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        thisB = BDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        if not thisB.empty:
            thisB = thisB.xs(lastFoldIdx, level='fold')
        thisA = ADF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        if not thisA.empty:
            thisA = thisA.xs(lastFoldIdx, level='fold')
        thisA = thisA.iloc[:thisA.shape[1], :]
        thisC = CDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        if not thisC.empty:
            thisC = thisC.xs(lastFoldIdx, level='fold')
        thisD = DDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        if not thisD.empty:
            thisD = thisD.xs(lastFoldIdx, level='fold')
        thisK = KDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        if not thisK.empty:
            thisK = thisK.xs(lastFoldIdx, level='fold')
        widthList = [df.shape[1] if not df.empty else 1 for df in [thisA, thisB, thisC, thisD, thisK]]
        widthList = list(chain.from_iterable([wid, 1] for wid in widthList))
        heightList = [df.shape[0] if not df.empty else 1 for df in [thisA, thisB, thisC, thisD, thisK]]
        figH = 4
        figW = sum(widthList) * 4 / max(heightList)
        fig, ax = plt.subplots(1, 10, figsize=(figW, figH), gridspec_kw=dict(width_ratios=widthList))
        commonHeatmapOpts = dict(
            center=0,  # square=True, cbar=False,
            xticklabels=False, yticklabels=False, rasterized=True)
        if not thisA.empty:
            sns.heatmap(thisA, ax=ax[0], cbar_ax=ax[1], **commonHeatmapOpts)
        ax[0].set_title('A')
        if not thisB.empty:
            sns.heatmap(thisB, ax=ax[2], cbar_ax=ax[3], **commonHeatmapOpts)
        ax[2].set_title('B')
        if not thisC.empty:
            sns.heatmap(thisC, ax=ax[4], cbar_ax=ax[5], **commonHeatmapOpts)
        ax[4].set_title('C')
        if not thisD.empty:
            sns.heatmap(thisD, ax=ax[6], cbar_ax=ax[7], **commonHeatmapOpts)
        ax[6].set_title('D')
        if not thisK.empty:
            sns.heatmap(thisK, ax=ax[8], cbar_ax=ax[9], **commonHeatmapOpts)
        ax[8].set_title('K')
        fig.suptitle('system matrices from validation set\n{}'.format(fullFormula))
        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        if not thisB.empty:
            fig, ax = plt.subplots(figsize=(2, 2))
            colsToEval = [cN for cN in thisB.columns if '[NA]' not in cN]
            BCosineSim = pd.DataFrame(np.nan, index=colsToEval, columns=colsToEval)
            mask = np.zeros_like(BCosineSim)
            mask[np.triu_indices_from(mask)] = True
            for cNr in colsToEval:
                for cNc in colsToEval:
                    BCosineSim.loc[cNr, cNc] = distance.cosine(thisB[cNr], thisB[cNc])
            sns.heatmap(BCosineSim, mask=mask, vmin=0, vmax=1, square=True, ax=ax, rasterized=True)
            ax.set_title('Cosine distance between input directions')
            fig.tight_layout(pad=styleOpts['tight_layout.pad'])
            pdf.savefig(bbox_inches='tight', pad_inches=0)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            print('Saving to {}'.format(pdfPath))
        #
        thisHEig = untruncatedHEigDF.xs(lhsMaskIdx, level='lhsMaskIdx', drop_level=False)
        nLags = int(np.ceil(lhsMasksInfo.loc[lhsMaskIdx, 'historyLen'] / binInterval))
        nMeas = thisC.shape[0]
        plotThisHEig = thisHEig.to_frame(name='s').reset_index()
        rLim = min(plotThisHEig['state'].max(), max(2 * stateSpaceNDim, nMeas * nLags), ) + 1
        g = sns.relplot(
            x='state', y='s', col='rhsMaskIdx',
            data=plotThisHEig.loc[plotThisHEig['state'] < rLim, :], kind='scatter',
            height=3, aspect=1, )
        g.set(yscale='log')
        ax = g.axes[0, 0]
        g.suptitle(
            'singular values of Hankel matrix (ERA)\n{}'.format(lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr']))
        ax.axvline(stateSpaceNDim, c='b', label='state space model order: {}'.format(stateSpaceNDim), lw=1)
        ax.axvline(nLags * nMeas, c='g', label='AR(p) order ({}) * num. channels ({}) = {}'.format(nLags, nMeas, nLags * nMeas), lw=1)
        ax.axhline(np.spacing(1e4), c='r', label='floating point precision cutoff', ls='--')
        ax.set_xlabel('Count')
        ax.set_xlim([-1, rLim])
        ax.legend(loc='lower right', )
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        '''thisA = ADF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        
        thisC = CDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        thisD = DDF.xs(lhsMaskIdx, level='lhsMaskIdx').dropna(axis='columns')
        sys = ctrl.StateSpace(thisA, thisB, thisC, thisD, dt=binInterval)
        f = np.linspace(1e-3, 100, 1000)
        mag, phase, omega = ctrl.freqresp(sys, 2 * np.pi * f)
        fig, ax = plt.subplots(sys.ninputs, sys.noutputs, sharey=True)
        for inpIdx in range(sys.ninputs):
            for outIdx in range(sys.noutputs):
                ax[inpIdx, outIdx].semilogy(f, mag[outIdx, inpIdx, :])
        '''