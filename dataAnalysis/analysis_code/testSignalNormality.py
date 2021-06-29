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
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             load from raw, or regular? [default: False]
    --showFigures                          load from raw, or regular? [default: False]
    --debugging                            load from raw, or regular? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
from dask.distributed import Client, LocalCluster
import os, traceback
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
import pingouin as pg
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer
import joblib as jb
import dill as pickle
import gc
from docopt import docopt
from copy import deepcopy

import sys

if __name__ == '__main__':
    for arg in sys.argv:
        print(arg)
    idxSl = pd.IndexSlice
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    #
    arguments['verbose'] = int(arguments['verbose'])
    #
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], arguments['alignFolderName'])
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    outputPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_normality_tests.h5'
        )
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
    #
    with open(loadingMetaPath, 'rb') as _f:
        loadingMeta = pickle.load(_f)
        # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
        iteratorsBySegment = loadingMeta['iteratorsBySegment']
        # cv_kwargs = loadingMeta['cv_kwargs']
    for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
        loadingMeta['arguments'].pop(argName, None)
    arguments.update(loadingMeta['arguments'])
    normalizeDataset = loadingMeta['normalizeDataset']
    unNormalizeDataset = loadingMeta['unNormalizeDataset']
    normalizationParams = loadingMeta['normalizationParams']
    cvIterator = iteratorsBySegment[0]
    print('loading {} from {}'.format(selectionName, datasetPath))
    dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
    featureMasks = pd.read_hdf(datasetPath, '/{}/featureMasks'.format(selectionName))
    # remove the 'all' column?#
    removeAllColumn = ('spectral' in arguments['unitQuery'])
    if removeAllColumn:
        featureMasks = featureMasks.loc[~ featureMasks.all(axis='columns'), :]
    # only use zero lag targets
    lagMask = dataDF.columns.get_level_values('lag') == 0
    dataDF = dataDF.loc[:, lagMask]
    featureMasks = featureMasks.loc[:, lagMask]
    trialInfo = dataDF.index.to_frame().reset_index(drop=True)
    workIdx = cvIterator.work
    # workingDataDF = dataDF.iloc[workIdx, :]
    prf.print_memory_usage('just loaded data, fitting')
    originalDataDF = unNormalizeDataset(dataDF, normalizationParams)
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'dimensionality')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
    if True:
        hzResDict0 = {}
        ntResDict0 = {}
        hsResDict0 = {}
        for idx, (maskIdx, featureMask) in enumerate(featureMasks.iterrows()):
            maskParams = {k: v for k, v in zip(featureMask.index.names, maskIdx)}
            prf.print_memory_usage('On featureMask {}'.format(maskIdx))
            dataGroup = dataDF.loc[:, featureMask]
            originalDataGroup = originalDataDF.loc[:, featureMask]
            ntResDict1 = {'normalized': {}, 'original': {}}
            if dataGroup.shape[1] > 1:
                hsResDict1 = {'normalized': {}, 'original': {}}
                hzResDict1 = {'normalized': {}, 'original': {}}
            for foldIdx, (tr, te) in enumerate(cvIterator.split(dataGroup)):
                prf.print_memory_usage('On fold {}'.format(foldIdx))
                if dataGroup.shape[1] > 1:
                    # Test equality of variance
                    hsResDict1['normalized'][foldIdx] = pg.homoscedasticity(
                        dataGroup.iloc[tr, :]).reset_index().rename(columns={'index': 'method'})
                    hsResDict1['original'][foldIdx] = pg.homoscedasticity(
                        originalDataGroup.iloc[tr, :]).reset_index().rename(columns={'index': 'method'})
                    #
                    # Henze-Zirkler multivariate normality test.
                    hzr = pg.multivariate_normality(
                        dataGroup.iloc[tr, :].to_numpy(), alpha=0.05)
                    hzResDict1['normalized'][foldIdx] = pd.Series(hzr, index=['hz', 'pval', 'normal']).to_frame(name=foldIdx)
                    hzrO = pg.multivariate_normality(
                        originalDataGroup.iloc[tr, :].to_numpy(), alpha=0.05)
                    hzResDict1['original'][foldIdx] = pd.Series(hzrO, index=['hz', 'pval', 'normal']).to_frame(name=foldIdx)
                # Univariate normality test.
                ntResDict1['normalized'][foldIdx] = pg.normality(
                    dataGroup.iloc[tr, :], method='normaltest')
                ntResDict1['original'][foldIdx] = pg.normality(
                    originalDataGroup.iloc[tr, :], method='normaltest')
            ntTemp = {
                k: pd.concat(ntResDict1[k], names=['fold'])
                for k in ['original', 'normalized']
                }
            ntResDict0[maskParams['freqBandName']] = pd.concat(ntTemp, names=['dataType'])
            if dataGroup.shape[1] > 1:
                hzTemp = {
                    k: pd.concat(hzResDict1[k], axis='columns', names=['fold']).T
                    for k in ['original', 'normalized']
                    }
                hzResDict0[maskParams['freqBandName']] = pd.concat(hzTemp, names=['dataType'])
                hsTemp = {
                    k: pd.concat(hsResDict1[k], names=['fold'])
                    for k in ['original', 'normalized']
                    }
                hsResDict0[maskParams['freqBandName']] = pd.concat(hsTemp, names=['dataType'])
        hzResDF = pd.concat(hzResDict0, names=['freqBandName'])
        hzResDF.to_hdf(outputPath, 'henzeZirkler')
        hzResDF.loc[:, 'hz'] = hzResDF['hz'].astype(float)
        hsResDF = pd.concat(hsResDict0, names=['freqBandName'])
        hsResDF.to_hdf(outputPath, 'homoscedasticity')
        ntResDF = pd.concat(ntResDict0, names=['freqBandName'])
        ntResDF.index = ntResDF.index.droplevel(0)
        ntResDF.to_hdf(outputPath, 'univariateNormality')
        #
        pdfPath = os.path.join(
            figureOutputFolder, '{}_normality_tests.pdf'.format(selectionName))
        # pdb.set_trace()
        with PdfPages(pdfPath) as pdf:
            plotGroup = hzResDF.reset_index()
            plotGroup.loc[:, 'xDummy'] = 0
            g = sns.catplot(
                y='hz', x='freqBandName',
                kind='box', hue='dataType', sharey=False,
                data=plotGroup)
            g.fig.set_size_inches((12, 8))
            g.fig.suptitle('{}'.format('henzeZirkler'))
            g.fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=.2)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ##
            plotGroup = hsResDF.reset_index()
            plotGroup.loc[:, 'xDummy'] = 0
            g = sns.catplot(
                y='W', x='freqBandName',
                kind='box', hue='dataType', sharey=False,
                data=plotGroup)
            g.fig.set_size_inches((12, 8))
            g.fig.suptitle('{}'.format('homoscedasticity W'))
            g.fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=.2)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            ##
            plotGroup = ntResDF.reset_index()
            plotGroup.loc[:, 'xDummy'] = 0
            g = sns.catplot(
                y='W', x='freqBandName',
                kind='box', hue='dataType', sharey=False,
                data=plotGroup)
            g.fig.set_size_inches((12, 8))
            g.fig.suptitle('{}'.format('normality W'))
            g.fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=.2)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    pdfPath = os.path.join(
        figureOutputFolder,
        '{}_{}_signal_distributions.pdf'.format(
            datasetName, selectionName))
    with PdfPages(pdfPath) as pdf:
        for name, group in dataDF.groupby('feature', axis='columns'):
            plotTemp = {
                'normalized': group.copy(),
                'original': originalDataDF.loc[:, group.columns].copy()}
            plotGroup = pd.concat(plotTemp, names=['dataType'])
            plotGroup.columns = plotGroup.columns.get_level_values('feature')
            plotGroup.reset_index(inplace=True)
            plotGroup.loc[:, 'xDummy'] = 0
            g = sns.catplot(
                x='xDummy', y=name, hue='expName',
                kind='violin', col='dataType', sharey=False,
                data=plotGroup)
            g.fig.set_size_inches((12, 8))
            g.fig.suptitle('{}'.format(name))
            g.fig.tight_layout(pad=1)
            pdf.savefig(bbox_inches='tight', pad_inches=0.2)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
