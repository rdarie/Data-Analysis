"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                                  which experimental day to analyze
    --blockIdx=blockIdx                        which trial to analyze [default: 1]
    --analysisName=analysisName                append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName          append a name to the resulting blocks? [default: motion]
    --processAll                               process entire experimental day? [default: False]
    --verbose=verbose                          print diagnostics?
    --lazy                                     load from raw, or regular? [default: False]
    --window=window                            process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix        which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix        which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                      how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                    what will the plot be aligned to? [default: outboundWithStim]
    --estimatorName=estimatorName              what will the plot be aligned to? [default: enr]
    --refEstimatorName=refEstimatorName        what will the plot be aligned to?
    --maskOutlierBlocks                        delete outlier trials? [default: False]
    --invertOutlierMask                        delete outlier trials? [default: False]
    --showFigures                              show plots interactively? [default: False]
    --iteratorSuffix=iteratorSuffix            filename for resulting estimator (cross-validated n_comps)
    --refIteratorSuffix=refIteratorSuffix      filename for resulting estimator (cross-validated n_comps)
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
import os
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import traceback
import dill as pickle
import pandas as pd
import numpy as np
from copy import deepcopy
from dask.distributed import Client, LocalCluster
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
import dataAnalysis.custom_transformers.tdr as tdr
from sklego.preprocessing import PatsyTransformer
from sklearn.compose import ColumnTransformer
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model._coordinate_descent import _alpha_grid
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})
'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'alignFolderName': 'motion', 'showFigures': False, 'unitQuery': 'mahal', 'alignQuery': 'starting',
        'inputBlockPrefix': 'Block', 'invertOutlierMask': False, 'lazy': False, 'exp': 'exp202101281100',
        'analysisName': 'default', 'processAll': True,
        'inputBlockSuffix': 'lfp_CAR_spectral_fa_mahal', 'verbose': False,
        'blockIdx': '2', 'maskOutlierBlocks': False, 'window': 'XL', 'verbose':0,
        'estimatorName': 'enr_rsos_refit', 'refEstimatorName': 'enr_noRos_refit', 'debugging': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
if __name__ == "__main__":
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    #
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    sns.set(
        context='paper', style='whitegrid',
        palette='dark', font='sans-serif',
        font_scale=.8, color_codes=True, rc={
            'figure.dpi': 200, 'savefig.dpi': 200})
    #
    idxSl = pd.IndexSlice
    estimatorName = arguments['estimatorName']
    if arguments['iteratorSuffix'] is not None:
        iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
    else:
        iteratorSuffix = ''
    if arguments['refIteratorSuffix'] is not None:
        refIteratorSuffix = '_{}'.format(arguments['refIteratorSuffix'])
    else:
        refIteratorSuffix = iteratorSuffix
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
        )
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName']
        )
    calcSubFolder = os.path.join(analysisSubFolder, 'dataframes')
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
    blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
    #
    joblibBackendArgs = dict(
        # backend='dask',
        backend='loky',
        ### n_jobs=1
        )
    if joblibBackendArgs['backend'] == 'dask':
        daskComputeOpts = dict(
            scheduler='processes'
            # scheduler='single-threaded'
            )
        if daskComputeOpts['scheduler'] == 'single-threaded':
            daskClient = Client(LocalCluster(n_workers=1))
        elif daskComputeOpts['scheduler'] == 'processes':
            daskClient = Client(LocalCluster(processes=True))
        elif daskComputeOpts['scheduler'] == 'threads':
            daskClient = Client(LocalCluster(processes=False))
        else:
            print('Scheduler name is not correct!')
            daskClient = Client()
    #
    raucPath = os.path.join(
        calcSubFolder,
        blockBaseName + '{}_{}_rauc.h5'.format(
            inputBlockSuffix, arguments['window']))
    resultPath = os.path.join(
        calcSubFolder,
        '{}{}_{}{}_{}_rauc_regression.h5'.format(
            estimatorName, iteratorSuffix, blockBaseName, inputBlockSuffix, arguments['window']))
    resultPicklePath = os.path.join(
        calcSubFolder,
        '{}{}_{}{}_{}_rauc_regression.pickle'.format(
            estimatorName, iteratorSuffix, blockBaseName, inputBlockSuffix, arguments['window']))
    if arguments['refEstimatorName'] is not None:
        refEstimatorName = arguments['refEstimatorName']
        resultPathRef = os.path.join(
            calcSubFolder,
            '{}_{}{}_{}_rauc_regression.h5'.format(
                refEstimatorName, refIteratorSuffix, blockBaseName, inputBlockSuffix, arguments['window']))
        resultPicklePathRef = os.path.join(
            calcSubFolder,
            '{}{}_{}{}_{}_rauc_regression.pickle'.format(
                refEstimatorName, refIteratorSuffix, blockBaseName, inputBlockSuffix, arguments['window']))

    recCurve = pd.read_hdf(raucPath, 'raw')

    with open(resultPicklePath, 'rb') as _f:
        estimatorMetadata = pickle.load(_f)
    cvIterator = estimatorMetadata['cvIterator']
    lhsScaler = estimatorMetadata['lhsScaler']
    with pd.HDFStore(resultPath) as store:
        scoresDF = pd.read_hdf(store, 'cv')
        workEstimatorsDF = pd.read_hdf(store, 'work')
        predDF = pd.read_hdf(store, 'predictions')
        termPalette = pd.read_hdf(store, 'plotOpts')
        coefDF = pd.read_hdf(store, 'coefficients')

    if arguments['refEstimatorName'] is not None:
        with open(resultPicklePathRef, 'rb') as _f:
            refEstimatorMetadata = pickle.load(_f)
        with pd.HDFStore(resultPathRef) as store:
            refScoresDF = pd.read_hdf(store, 'cv')
            refWorkEstimatorsDF = pd.read_hdf(store, 'work')
            refPredDF = pd.read_hdf(store, 'predictions')
            refCoefDF = pd.read_hdf(store, 'coefficients')
        refEstExists = True
    else:
        refEstExists = False

    trialInfo = recCurve.index.to_frame().reset_index(drop=True)
    trialInfo.index.name = 'trial'
    #
    trialInfo.loc[:, 'foldType'] = np.nan
    trialInfo.loc[cvIterator.work, 'foldType'] = 'work'
    trialInfo.loc[cvIterator.validation, 'foldType'] = 'validation'
    lhsDF = trialInfo.loc[:, stimulusConditionNames]
    rhsDF = recCurve.reset_index(drop=True)
    lhsDF.index.name = 'trial'
    rhsDF.index.name = 'trial'
    scaledLhsDF = lhsScaler.transform(lhsDF)
    if arguments['debugging']:
        rhsDF = rhsDF.loc[:, idxSl[['mahal_all#0', 'mahal_gamma#0'], :, :, :, :, :]]
    #
    electrodeNames = [eN for eN in lhsDF['electrode'].unique() if eN != 'NA']
    electrodeMasks = {}
    for eN in electrodeNames:
        eMask = lhsDF['electrode'] == eN
        vMask = lhsDF.index.isin(cvIterator.validation)
        electrodeMasks[eN] = eMask & vMask
    #
    if refEstExists:
        pdfPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '{}_{}_{}_vs_{}_{}.pdf'.format(
                inputBlockSuffix, arguments['window'], estimatorName, refEstimatorName,
                'RAUC_coefficients'))
        with PdfPages(pdfPath) as pdf:
            allCoeffNames = coefDF.index.get_level_values('component').unique()
            vMax = pd.concat([refCoefDF, coefDF]).abs().quantile(1 - 1e-2)
            vMin = vMax * -1
            coefDiff = refCoefDF - coefDF
            diffMax = coefDiff.abs().quantile(1 - 1e-2)
            diffMin = diffMax * -1
            for designFormula, theseCoefs in coefDF.groupby('design'):
                plotCoefs = (
                    theseCoefs.groupby(['target', 'component']).mean()
                        .unstack('component').reindex(columns=allCoeffNames))
                # indexing allCoeffNames adds nans for missing values
                # so that all the designs have identical rows and cols
                grid_kws = {"width_ratios": (30, 1), 'wspace': 0.01}
                aspect = plotCoefs.shape[1] / plotCoefs.shape[0]
                h = 3
                w = h * aspect
                fig, (ax, cbar_ax) = plt.subplots(
                    1, 2,
                    gridspec_kw=grid_kws,
                    figsize=(w, h))
                ax = sns.heatmap(
                    plotCoefs, ax=ax,
                    cbar_ax=cbar_ax, vmin=vMin, vmax=vMax, cmap='vlag')
                titleText = '{} coefficients for model\n{}'.format(estimatorName, designFormula)
                figTitle = fig.suptitle(titleText)
                pdf.savefig(
                    bbox_inches='tight',
                    bbox_extra_artists=[figTitle]
                )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                ##
                theseRefCoefs = refCoefDF.xs(designFormula, level='design', drop_level=False)
                plotRefCoefs = (
                    theseRefCoefs.groupby(['target', 'component']).mean()
                        .unstack('component').reindex(columns=allCoeffNames))
                fig, (ax, cbar_ax) = plt.subplots(
                    1, 2,
                    gridspec_kw=grid_kws,
                    figsize=(w, h))
                ax = sns.heatmap(
                    plotRefCoefs, ax=ax,
                    cbar_ax=cbar_ax, vmin=vMin, vmax=vMax, cmap='vlag')
                titleText = '{} coefficients for model\n{}'.format(refEstimatorName, designFormula)
                figTitle = fig.suptitle(titleText)
                pdf.savefig(
                    bbox_inches='tight',
                    bbox_extra_artists=[figTitle]
                )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()

                theseDiffCoefs = coefDiff.xs(designFormula, level='design', drop_level=False)
                plotDiffCoefs = (
                    theseDiffCoefs.groupby(['target', 'component']).mean()
                        .unstack('component').reindex(columns=allCoeffNames))
                theseCoefDiffs = refCoefDF.xs(designFormula, level='design', drop_level=False)
                fig, (ax, cbar_ax) = plt.subplots(
                    1, 2,
                    gridspec_kw=grid_kws,
                    figsize=(w, h))
                ax = sns.heatmap(
                    plotDiffCoefs, ax=ax,
                    # cbar_ax=cbar_ax, vmin=diffMin, vmax=diffMax, cmap="coolwarm")
                    cbar_ax=cbar_ax, vmin=vMin, vmax=vMax, cmap="vlag")
                titleText = 'Percent difference in coefficients for model\n{}'.format(designFormula)
                figTitle = fig.suptitle(titleText)
                pdf.savefig(
                    bbox_inches='tight',
                    bbox_extra_artists=[figTitle]
                )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
        ########################################
        pdfPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '{}_{}_{}_vs_{}_{}.pdf'.format(
                inputBlockSuffix, arguments['window'], estimatorName, refEstimatorName,
                'RAUC_regression_scores'))
        with PdfPages(pdfPath) as pdf:
            height, width = 3, 3
            aspect = width / height
            #
            trainScores = scoresDF.loc[:, 'train_score'].to_frame(name='score')
            testScores = scoresDF.loc[:, 'test_score'].to_frame(name='score')
            #
            trainScoresRef = refScoresDF.loc[:, 'train_score'].to_frame(name='score')
            testScoresRef = refScoresDF.loc[:, 'test_score'].to_frame(name='score')
            #
            folds = trainScores.index.get_level_values('fold')
            lastFoldIdx = folds.max()
            lookup = {
                k: 'train' if k < lastFoldIdx else 'work' for k in folds.unique()
                }
            trainScores.loc[:, 'scoreType'] = folds.map(lookup)
            trainScores.set_index('scoreType', append=True, inplace=True)
            trainScoresRef.loc[:, 'scoreType'] = folds.map(lookup)
            trainScoresRef.set_index('scoreType', append=True, inplace=True)
            #
            lookup2 = {
                k: 'test' if k < lastFoldIdx else 'validation' for k in folds.unique()
                }
            testScores.loc[:, 'scoreType'] = folds.map(lookup2)
            testScores.set_index('scoreType', append=True, inplace=True)
            testScoresRef.loc[:, 'scoreType'] = folds.map(lookup2)
            testScoresRef.set_index('scoreType', append=True, inplace=True)
            #
            plotScores = pd.concat([testScores, trainScores])
            plotScoresRef = pd.concat([testScoresRef, trainScoresRef])
            plotScoresDiff = plotScoresRef - plotScores
            plotScores.reset_index(inplace=True)
            plotScoresRef.reset_index(inplace=True)
            plotScoresDiff.reset_index(inplace=True)
            #
            plotScores.loc[:, 'xDummy'] = 0
            plotScoresRef.loc[:, 'xDummy'] = 0
            plotScoresDiff.loc[:, 'xDummy'] = 0
            #
            g = sns.catplot(
                data=plotScores, kind='box',
                y='score', x='target',
                col='design', hue='scoreType',
                height=height, aspect=aspect
            )
            g.set_xticklabels(rotation=30, ha='right')
            g.set_titles(template="{col_name}")
            g.tight_layout(pad=0.1)
            titleText = '{} scores'.format(estimatorName)
            figTitle = g.fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            g = sns.catplot(
                data=plotScoresRef, kind='box',
                y='score', x='target',
                col='design', hue='scoreType',
                height=height, aspect=aspect
            )
            g.set_xticklabels(rotation=30, ha='right')
            g.set_titles(template="{col_name}")
            g.tight_layout(pad=0.1)
            titleText = '{} scores'.format(refEstimatorName)
            figTitle = g.fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            g = sns.catplot(
                data=plotScoresDiff, kind='box',
                y='score', x='target',
                col='design', hue='scoreType',
                height=height, aspect=aspect
            )
            g.set_xticklabels(rotation=30, ha='right')
            g.set_titles(template="{col_name}")
            g.tight_layout(pad=0.1)
            titleText = 'difference in scores ({} - {})'.format(refEstimatorName, estimatorName)
            figTitle = g.fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()

    for designFormula, designScores in scoresDF.groupby('design'):
        pT = PatsyTransformer(designFormula, return_type="matrix")
        designMatrix = pT.fit_transform(scaledLhsDF)
        designInfo = designMatrix.design_info
        designDF = (pd.DataFrame(designMatrix, index=lhsDF.index, columns=designInfo.column_names))
        theseScores = scoresDF.xs(designFormula, level='design')
        for targetName, targetScores in theseScores.groupby('target'):
            estimator = targetScores.iloc[0].loc['estimator']
            thesePred = predDF.xs(targetName, level='target').xs(designFormula, level='design')
            thesePred.index = thesePred.index.get_level_values('trial')
            for elecName, elecMask in electrodeMasks.items():
                trialIndices = lhsDF.loc[elecMask, :].index
                thesePred.loc[trialIndices, 'ground_truth']
                break
            break
        break