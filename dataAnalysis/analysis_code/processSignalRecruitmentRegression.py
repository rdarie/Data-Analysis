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
    --debugging                                show plots interactively? [default: False]
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
from pyglmnet.pyglmnet import _logL as pyglmnet_logL
from pyglmnet.metrics import pseudo_R2 as pyglmnet_pseudo_R2
import pyglmnet
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
        'refIteratorSuffix': 'ros', 'iteratorSuffix': 'noRos',
        'refEstimatorName': 'enr_refit', 'estimatorName': 'enr_refit', 'debugging': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''

def getR2(srs):
    lls = float(srs.xs('llSat', level='llType'))
    lln = float(srs.xs('llNull', level='llType'))
    llf = float(srs.xs('llFull', level='llType'))
    return 1 - (lls - llf) / (lls - lln)


def partialR2(srs, testDesign=None, refDesign=None):
    llSat = llSrs.xs(testDesign, level='design').xs('llSat', level='llType')
    assert testDesign is not None
    # llSat2 = llSrs.xs(refDesign, level='design').xs('llSat', level='llType')
    # sanity check: should be same
    # assert (llSat - llSat2).abs().max() == 0
    if refDesign is not None:
        llRef = llSrs.xs(refDesign, level='design').xs('llFull', level='llType')
    else:
        llRef = llSrs.xs(testDesign, level='design').xs('llNull', level='llType')
    llTest = llSrs.xs(testDesign, level='design').xs('llFull', level='llType')
    return 1 - (llSat - llTest) / (llSat - llRef)

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
    fullEstimatorName = '{}{}'.format(estimatorName, iteratorSuffix)
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
        fullRefEstimatorName = '{}{}'.format(refEstimatorName, refIteratorSuffix)
        resultPathRef = os.path.join(
            calcSubFolder,
            '{}_{}{}_{}_rauc_regression.h5'.format(
                fullRefEstimatorName, blockBaseName, inputBlockSuffix, arguments['window']))
        resultPicklePathRef = os.path.join(
            calcSubFolder,
            '{}_{}{}_{}_rauc_regression.pickle'.format(
                fullRefEstimatorName, blockBaseName, inputBlockSuffix, arguments['window']))

    recCurve = pd.read_hdf(raucPath, 'raw')

    with open(resultPicklePath, 'rb') as _f:
        estimatorMetadata = pickle.load(_f)
    cvIterator = estimatorMetadata['cvIterator']
    lhsScaler = estimatorMetadata['lhsScaler']
    with pd.HDFStore(resultPath) as store:
        scoresDF = pd.read_hdf(store, 'cv')
        _estimatorsDF = pd.read_hdf(store, 'cv_estimators')
        scoresDF.loc[:, 'estimator'] = _estimatorsDF
        workEstimatorsDF = pd.read_hdf(store, 'work')
        predDF = pd.read_hdf(store, 'predictions')
        termPalette = pd.read_hdf(store, 'plotOpts')
        coefDF = pd.read_hdf(store, 'coefficients')
        allScoresForQuantile = np.concatenate([scoresDF['train_score'], scoresDF['test_score']])

    if arguments['refEstimatorName'] is not None:
        with open(resultPicklePathRef, 'rb') as _f:
            refEstimatorMetadata = pickle.load(_f)
        with pd.HDFStore(resultPathRef) as store:
            refScoresDF = pd.read_hdf(store, 'cv')
            allScoresForQuantile = np.concatenate([allScoresForQuantile, refScoresDF['train_score'], refScoresDF['test_score']])
            refWorkEstimatorsDF = pd.read_hdf(store, 'work')
            refPredDF = pd.read_hdf(store, 'predictions')
            refCoefDF = pd.read_hdf(store, 'coefficients')
        refEstExists = True
    else:
        refEstExists = False

    #
    allScoreQuantiles = np.quantile(allScoresForQuantile, [1e-2, 1-1e-2])
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


    '''pdb.set_trace()
    for name, estimatorDF in workEstimatorsDF.groupby(['design', 'target']):
        designFormula, targetName = name
        estimator = estimatorDF['estimator'].iloc[0]
        factorNames = estimator.model_.exog_names
        nRows = int(np.ceil(np.sqrt(len(factorNames))))
        nCols = int(np.ceil(len(factorNames) / nRows))
        fig, thisAx = plt.subplots(nRows, nCols)
        for fIdx, fN in enumerate(factorNames):
            fig = estimator.results_.plot_added_variable(focus_exog=fN, ax=thisAx.flat[fIdx])
        plt.show()
        break'''

    if refEstExists:
        pdfPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '{}_{}_{}_vs_{}_{}.pdf'.format(
                inputBlockSuffix, arguments['window'], fullEstimatorName, fullRefEstimatorName,
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
                titleText = '{} coefficients for model\n{}'.format(fullEstimatorName, designFormula)
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
                titleText = '{} coefficients for model\n{}'.format(fullRefEstimatorName, designFormula)
                figTitle = fig.suptitle(titleText)
                pdf.savefig(
                    bbox_inches='tight',
                    bbox_extra_artists=[figTitle]
                )
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
                #
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
                inputBlockSuffix, arguments['window'], fullEstimatorName, fullRefEstimatorName,
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

            g.axes.flat[0].set_ylim(allScoreQuantiles)
            g.tight_layout(pad=0.3)
            titleText = '{} scores'.format(fullEstimatorName)
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
            g.axes.flat[0].set_ylim(allScoreQuantiles)
            g.tight_layout(pad=0.3)
            titleText = '{} scores'.format(fullRefEstimatorName)
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
            g.tight_layout(pad=0.3)
            titleText = 'difference in scores ({} - {})'.format(fullRefEstimatorName, estimatorName)
            figTitle = g.fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    #
    llDict0 = {}
    for designFormula, designScores in scoresDF.groupby('design'):
        pT = PatsyTransformer(designFormula, return_type="matrix")
        designMatrix = pT.fit_transform(scaledLhsDF)
        designInfo = designMatrix.design_info
        designDF = (pd.DataFrame(designMatrix, index=lhsDF.index, columns=designInfo.column_names))
        llDict1 = {}
        for scoreName, targetScores in designScores.groupby(['target', 'fold']):
            targetName, fold = scoreName
            estimator = targetScores.iloc[0].loc['estimator']
            # estimator.model_.exog_names
            # fig = estimator.results_.plot_added_variable('pedalMovementCat[outbound]')
            thesePred = predDF.xs(targetName, level='target').xs(designFormula, level='design').xs(fold, level='fold')
            '''
            #### sanity checks
            np.abs(thesePred.xs('work', level='trialType')['prediction'].to_numpy() - estimator.results_.fittedvalues).max()
            #
            1 - (estimator.results_.llf / estimator.results_.llnull)
            residuals = (thesePred['ground_truth'] - thesePred['prediction'])
            SSR = (residuals ** 2).sum()
            SST = ((thesePred['ground_truth'] - thesePred['ground_truth'].mean()) ** 2).sum()
            R2 = 1 - (SSR / SST)
            #
            llf = estimator.results_.family.loglike(thesePred['prediction'].to_numpy(), thesePred['ground_truth'].to_numpy())
            llsat = estimator.results_.family.loglike(thesePred['ground_truth'].to_numpy(), thesePred['ground_truth'].to_numpy())
            nullModel = ((thesePred['ground_truth'] ** 0) * thesePred['ground_truth'].mean()).to_numpy()
            llnull = estimator.results_.family.loglike_obs(nullModel, thesePred['ground_truth'].to_numpy()).sum()
            llR2 = 1 - ((llsat - llf) / (llsat - llnull))
            #
            L0 = pyglmnet_logL('gaussian', thesePred['ground_truth'].to_numpy(), nullModel)
            L1 = pyglmnet_logL('gaussian', thesePred['ground_truth'].to_numpy(), thesePred['prediction'].to_numpy())
            1 - (L1 / L0)
            '''
            ###
            llDict2 = {}
            for name, predGroup in thesePred.groupby(['electrode', 'trialType']):
                llDict3 = {}
                llDict3['llSat'] = estimator.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
                nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
                llDict3['llNull'] = estimator.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
                llDict3['llFull'] = estimator.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
                llDict2[name] = pd.Series(llDict3)
            for trialType, predGroup in thesePred.groupby('trialType'):
                llDict3 = {}
                llDict3['llSat'] = estimator.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
                nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
                llDict3['llNull'] = estimator.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
                llDict3['llFull'] = estimator.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
                llDict2[('all', trialType)] = pd.Series(llDict3)
            llDict1[scoreName] = pd.concat(llDict2, names=['electrode', 'trialType', 'llType'])
        llDict0[designFormula] = pd.concat(llDict1, names=['target', 'fold', 'electrode', 'trialType', 'llType'])
    llSrs = pd.concat(llDict0, names=['design', 'target', 'fold', 'electrode', 'trialType', 'llType'])

    R2Per = llSrs.groupby(['design', 'target', 'electrode', 'fold', 'trialType']).apply(getR2)

    # scoresDF.index.get_level_values('design').unique()
    # print('all models: {}'.format(scoresDF.index.get_level_values('design').unique()))
    modelsToTest = [
        {
            'testDesign': 'pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1',
            'refDesign': 'pedalMovementCat + (electrode:(amplitude/RateInHz)) - 1',
            'captionStr': 'partial R2 of allowing pedalMovementCat to modulate electrode coefficients, vs assuming their independence'
        },
        {
            'testDesign': 'pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1',
            'refDesign': 'pedalMovementCat - 1',
            'captionStr': 'partial R2 of including any electrode coefficients'
        },
        {
            'testDesign': 'pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1',
            'refDesign': 'pedalMovementCat/(electrode:amplitude) - 1',
            'captionStr': 'partial R2 of including a term for modulation of electrode coefficients by RateInHz'
        },
        {
            'testDesign': 'pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1',
            'refDesign': 'electrode:(amplitude/RateInHz) + 1',
            'captionStr': 'partial R2 of including a term for pedalMovementCat, as opposed to an intercept'
        },
        {
            'testDesign': 'pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1',
            'refDesign': None,
            'captionStr': 'R2 of design (pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1)'
        },
    ]
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}_{}.pdf'.format(
            inputBlockSuffix, arguments['window'], fullEstimatorName,
            'RAUC_partial_scores'))
    with PdfPages(pdfPath) as pdf:
        height, width = 3, 3
        aspect = width / height
        for modelToTest in modelsToTest:
            testDesign = modelToTest['testDesign']
            refDesign = modelToTest['refDesign']
            if 'captionStr' in modelToTest:
                titleText = modelToTest['captionStr']
            else:
                titleText = 'partial R2 scores for {} compared {}'.format(testDesign, refDesign)
            pR2 = partialR2(llSrs, refDesign=refDesign, testDesign=testDesign)
            #
            plotScores = pR2.to_frame(name='score').reset_index()
            #
            plotScores.loc[:, 'xDummy'] = 0
            #
            g = sns.catplot(
                data=plotScores, kind='box',
                y='score', x='target',
                col='electrode', hue='trialType',
                height=height, aspect=aspect
            )
            g.set_xticklabels(rotation=30, ha='right')
            g.set_titles(template="{col_name}")
            g.axes.flat[0].set_ylim(allScoreQuantiles)
            g.tight_layout(pad=0.3)
            figTitle = g.fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()