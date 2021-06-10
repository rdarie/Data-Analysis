"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose=verbose                      print diagnostics?
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --estimatorName=estimatorName          what will the plot be aligned to? [default: outboundWithStim]
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierMask                    delete outlier trials? [default: False]
    --showFigures                          show plots interactively? [default: False]
    --debugging                            show plots interactively? [default: False]
    --iteratorSuffix=iteratorSuffix            filename for resulting estimator (cross-validated n_comps)
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
import dataAnalysis.custom_transformers.tdr as tdr
import os
import statsmodels.api as sm
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
from sklego.preprocessing import PatsyTransformer
from sklearn.compose import ColumnTransformer
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model._coordinate_descent import _alpha_grid
from pyglmnet import GLM
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
        'analysisName': 'default', 'processAll': True, 'inputBlockSuffix': 'lfp_CAR_spectral_fa_mahal', 'verbose': False,
        'blockIdx': '2', 'maskOutlierBlocks': False, 'window': 'XL', 'verbose':0, 'estimatorName': 'enr_noROs_db', 'debugging': True}
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
    idxSl = pd.IndexSlice
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
    if arguments['iteratorSuffix'] is not None:
        iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
    else:
        iteratorSuffix = ''
    estimatorName = arguments['estimatorName']
    cssTableStyles = [
        {
            'selector': 'th',
            'props': [
                ('border-style', 'solid'),
                ('border-color', 'black')]},
        {
            'selector': 'td',
            'props': [
                ('border-style', 'solid'),
                ('border-color', 'black')]},
        {
            'selector': 'table',
            'props': [
                ('border-collapse', 'collapse')
                ]}
        ]
    cm = sns.dark_palette("green", as_cmap=True)
    #
    raucPath = os.path.join(
        calcSubFolder,
        blockBaseName + '{}_{}_rauc.h5'.format(
            inputBlockSuffix, arguments['window']))
    iteratorPath = os.path.join(
        calcSubFolder,
        '{}{}_{}_rauc_iterator{}.pickle'.format(
            blockBaseName, inputBlockSuffix, arguments['window'], iteratorSuffix))
    resultPath = os.path.join(
        calcSubFolder,
        '{}{}_{}{}_{}_rauc_regression.h5'.format(
            estimatorName, iteratorSuffix, blockBaseName, inputBlockSuffix, arguments['window']))
    resultPicklePath = os.path.join(
        calcSubFolder,
        '{}{}_{}{}_{}_rauc_regression.pickle'.format(
            estimatorName, iteratorSuffix, blockBaseName, inputBlockSuffix, arguments['window']))
    print('loading {}'.format(raucPath))
    outlierTrials = ash.processOutlierTrials(
        scratchFolder, blockBaseName, **arguments)
    ################################################
    '''splitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'])
    iteratorKWArgs = dict(
        n_splits=7,
        splitterClass=tdr.trialAwareStratifiedKFold, splitterKWArgs=splitterKWArgs,
        samplerKWArgs=dict(random_state=None, test_size=None,),
        prelimSplitterClass=tdr.trialAwareStratifiedKFold, prelimSplitterKWArgs=splitterKWArgs,
        resamplerClass=RandomOverSampler, resamplerKWArgs={},
        # resamplerClass=None, resamplerKWArgs={},
        )'''
    ################################################
    joblibBackendArgs = dict(
        # backend='dask',
        backend='loky',
        #### n_jobs=1
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
    crossvalKWArgs = dict(
        return_train_score=True, return_estimator=True)
    '''
    ### sklearn elastic net
    estimatorInstance = ElasticNet(fit_intercept=False)
    # names of things in sklearn
    l1_ratio_name, alpha_name = 'l1_ratio', 'alpha'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid=dict(
            # l1_ratio=[.1, .5, .7, .9, .95, .99, 1]
            l1_ratio=[.1, .5, .95, 1]
            )
        )
        '''
    # statsmodels elasticnet
    modelKWArgs = {
        'sm_class': sm.GLM,
        'family': sm.families.Gaussian(),
        'alpha': None, 'L1_wt': None,
        'refit': True, 'tol': 1e-3,
        'maxiter': 1000, 'disp': False
        }
    estimatorInstance = tdr.SMWrapper(**modelKWArgs)
    ## names of things in statsmodels
    l1_ratio_name, alpha_name = 'L1_wt', 'alpha'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid=dict(
            L1_wt=[.1, .5, .95, 1]
        ))
    '''
    estimatorInstance = GLM(fit_intercept=False, distr='gaussian', solver='L-BFGS-B', score_metric='pseudo_R2')
    # names of things in pyglmnet
    l1_ratio_name, alpha_name = 'alpha', 'reg_lambda'
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid=dict(
            alpha=[.1, .5, .95, 1]
        )
    )
    '''
    ####
    nAlphas = 50
    lOfDesignFormulas = [
        "pedalMovementCat/(electrode:(amplitude/RateInHz)) - 1",
        "pedalMovementCat + (electrode:(amplitude/RateInHz)) - 1",
        "pedalMovementCat/(electrode:amplitude) - 1",
        "electrode:(amplitude/RateInHz) + 1",
        "pedalMovementCat - 1"
    ]
    recCurve = pd.read_hdf(raucPath, 'raw')
    ########
    with open(iteratorPath, 'rb') as _f:
        iteratorMeta = pickle.load(_f)
        iteratorKWArgs = iteratorMeta['iteratorKWArgs']
        cvIterator = iteratorMeta['cvIterator']
    '''cvIterator = tdr.trainTestValidationSplitter(
        dataDF=recCurve, **iteratorKWArgs)
    if arguments['showFigures']:
        # if True:
        fig, ax = cvIterator.plot_schema()
        plt.show()'''
    ################
    crossvalKWArgs['cv'] = cvIterator
    gridSearchKWArgs['cv'] = cvIterator
    groupsBreakdown = (
        recCurve.groupby(stimulusConditionNames)
            .count()
            .iloc[:, 0]
            .to_frame(name='count'))
    dfStyler = (
        groupsBreakdown.style
        .background_gradient(cmap=cm)
        .set_precision(1)
        )
    dfStyler.set_table_styles(cssTableStyles)
    breakDownHtml = dfStyler.render()
    breakDownInspectPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}.html'.format(
            inputBlockSuffix, arguments['window'],
            'RAUC_trial_breakdown'))
    with open(breakDownInspectPath, 'w') as _f:
        _f.write(breakDownHtml)
    #
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
    colsToScale = ['amplitude', 'RateInHz']
    lOfTransformers = [
        (['amplitude'], MinMaxScaler(feature_range=(0, 1)),),
        (['RateInHz'], MinMaxScaler(feature_range=(0, .5)),)
        ]
    for cN in lhsDF.columns:
        if cN not in colsToScale:
            lOfTransformers.append(([cN], None,))
    lhsScaler = DataFrameMapper(
        lOfTransformers, input_df=True, df_out=True
        )
    scaledLhsDF = lhsScaler.fit_transform(lhsDF)
    if arguments['debugging']:
        rhsDF = rhsDF.loc[:, idxSl[['mahal_all#0', 'mahal_gamma#0'], :, :, :, :, :]]
    '''
    pt = PatsyTransformer("", return_type="matrix")
    pipe = Pipeline([('scale', lhsScaler), ('design', pt), ('regression', ElasticNet(fit_intercept=False))])
    '''
    patsyTransfDict = {}
    cvScoresDict0 = {}
    gridSearcherDict0 = {}
    gsScoresDict0 = {}
    for designFormula in lOfDesignFormulas:
        print('Fitting {}'.format(designFormula))
        pt = PatsyTransformer(designFormula, return_type="matrix")
        patsyTransfDict[designFormula] = pt
        designMatrix = pt.fit_transform(scaledLhsDF)
        designInfo = designMatrix.design_info
        designDF = (pd.DataFrame(designMatrix, index=lhsDF.index, columns=designInfo.column_names))
        gsParamsPerTarget = None
        if 'param_grid' in gridSearchKWArgs:
            gsKWA = deepcopy(gridSearchKWArgs)
            if l1_ratio_name in gsKWA['param_grid']:
                paramGrid = gsKWA.pop('param_grid')
                lOfL1Ratios = paramGrid.pop(l1_ratio_name)
                gsParamsPerTarget = {}
                for columnTuple in rhsDF.columns:
                    targetName = columnTuple[0]
                    gsParamsPerTarget[targetName] = []
                    for l1Ratio in lOfL1Ratios:
                        alphas = _alpha_grid(
                            designDF, rhsDF.loc[:, columnTuple],
                            l1_ratio=l1Ratio, n_alphas=nAlphas)
                        gsParamsPerTarget[targetName].append(
                            {
                                l1_ratio_name: [l1Ratio],
                                alpha_name: np.atleast_1d(alphas).tolist()
                            }
                        )
        cvScoresDict1 = {}
        gridSearcherDict1 = {}
        gsScoresDict1 = {}
        for columnTuple in rhsDF.columns:
            targetName = columnTuple[0]
            print('Fitting {}'.format(targetName))
            if gsParamsPerTarget is not None:
                gsKWA['param_grid'] = gsParamsPerTarget[targetName]
            cvScores, gridSearcherDict1[targetName], gsScoresDict1[targetName] = tdr.gridSearchHyperparameters(
                designDF, rhsDF.loc[:, columnTuple],
                estimatorInstance=estimatorInstance,
                verbose=int(arguments['verbose']),
                gridSearchKWArgs=gsKWA,
                crossvalKWArgs=crossvalKWArgs,
                joblibBackendArgs=joblibBackendArgs
            )
            cvScoresDF = pd.DataFrame(cvScores)
            cvScoresDF.index.name = 'fold'
            cvScoresDF.dropna(axis='columns', inplace=True)
            cvScoresDict1[targetName] = cvScoresDF
        cvScoresDict0[designFormula] = pd.concat(cvScoresDict1, names=['target'])
        gridSearcherDict0[designFormula] = gridSearcherDict1[targetName]
        gsScoresDict0[designFormula] = pd.concat(gsScoresDict1, names=['target'])
    scoresDF = pd.concat(cvScoresDict0, names=['design'])
    #
    coefDict0 = {}
    predDict0 = {}
    for designFormula in lOfDesignFormulas:
        designMatrixTransformer = Pipeline([('scale', lhsScaler), ('design', patsyTransfDict[designFormula])])
        designMatrix = designMatrixTransformer.transform(lhsDF)
        designInfo = designMatrix.design_info
        designDF = (pd.DataFrame(designMatrix, index=lhsDF.index, columns=designInfo.column_names))
        theseScores = scoresDF.xs(designFormula, level='design')
        coefDict1 = {}
        predDict1 = {}
        for targetName, targetScores in theseScores.groupby('target'):
            targetDF = rhsDF.xs(targetName, level='feature', axis='columns', drop_level=False)
            coefDict2 = {}
            predDict2 = {}
            for foldIdx, estSrs in targetScores.groupby('fold'):
                regressionEstimator = estSrs['estimator'].iloc[0]
                coefs = pd.Series(regressionEstimator.coef_, index=designInfo.column_names)
                coefDict2[foldIdx] = coefs
                ####
                if True:
                    for eN in ['+ E16 - E5', '+ E16 - E2']:
                        ampTermStr = 'pedalMovementCat[NA]:electrode[{}]:amplitude'.format(eN)
                        if ampTermStr in coefs:
                            print(ampTermStr)
                            print('Coefficient:\n{}'.format(coefs[ampTermStr]))
                            print('unique terms:\n{}'.format(designDF.loc[:, ampTermStr].unique()))
                        ampRateTermStr = 'pedalMovementCat[NA]:electrode[{}]:amplitude:RateInHz'.format(eN)
                        if ampRateTermStr in coefs:
                            print(ampRateTermStr)
                            print('Coefficient:\n{}'.format(coefs[ampRateTermStr]))
                            print('unique terms:\n{}\n\n'.format(designDF.loc[:, ampRateTermStr].unique()))
                if foldIdx < cvIterator.n_splits:
                    predictionPerComponent = pd.concat({
                        'train': designDF.iloc[cvIterator.raw_folds[foldIdx][0], :] * coefs,
                        'test': designDF.iloc[cvIterator.raw_folds[foldIdx][1], :] * coefs
                        }, names=['trialType', 'trial'])
                else:
                    predictionPerComponent = pd.concat({
                        'work': designDF.iloc[cvIterator.work, :] * coefs,
                        'validation': designDF.iloc[cvIterator.validation, :] * coefs
                        }, names=['trialType', 'trial'])
                predictionSrs = predictionPerComponent.sum(axis='columns')
                # sanity
                indicesThisFold = predictionPerComponent.index.get_level_values('trial')
                mismatch = predictionSrs - regressionEstimator.predict(designDF.iloc[indicesThisFold, :])
                print('max mismatch is {}'.format(mismatch.abs().max()))
                assert (mismatch.abs().max() < 1e-3)
                predictionPerSource = pd.DataFrame(
                    np.nan, index=predictionPerComponent.index,
                    columns=designInfo.term_names)
                for termName, termSlice in designInfo.term_name_slices.items():
                    predictionPerSource.loc[:, termName] = predictionPerComponent.iloc[:, termSlice].sum(axis='columns')
                predictionPerSource.loc[:, 'prediction'] = predictionSrs
                predictionPerSource.loc[:, 'ground_truth'] = targetDF.iloc[indicesThisFold, :].to_numpy()
                predDict2[foldIdx] = predictionPerSource
            coefDict1[targetName] = pd.concat(coefDict2, names=['fold', 'component'])
            predDict1[targetName] = pd.concat(predDict2, names=['fold', 'trialType', 'trial'])
        coefDict0[designFormula] = pd.concat(coefDict1, names=['target', 'fold', 'component'])
        predDict0[designFormula] = pd.concat(predDict1, names=['target', 'fold', 'trialType', 'trial'])
    coefDF = pd.concat(coefDict0, names=['design', 'target', 'fold', 'component'])
    predDF = pd.concat(predDict0, names=['design', 'target', 'fold', 'trialType', 'trial'])
    predDF.columns.name = 'term'
    extendedTrialInfo = (
        trialInfo.loc[
            predDF.index.get_level_values('trial'),
            stimulusConditionNames + ['foldType', 'segment', 'originalIndex', 't']]
        .reset_index(drop=True))
    extendedTrialInfo = pd.concat([extendedTrialInfo, predDF.index.to_frame().reset_index(drop=True)], axis='columns')
    predDF.index = pd.MultiIndex.from_frame(extendedTrialInfo)
    #
    nTerms = predDF.columns.size
    termPalette = pd.Series(
        sns.color_palette('Set2', nTerms),
        index=predDF.columns)
    #######
    if os.path.exists(resultPath):
        os.remove(resultPath)
    if os.path.exists(resultPicklePath):
        os.remove(resultPicklePath)
    #
    try:
        scoresDF.loc[idxSl[:, :, cvIterator.n_splits], :].to_hdf(resultPath, 'work')
        scoresDF.loc[:, ['test_score', 'train_score']].to_hdf(resultPath, 'cv')
    except Exception:
        traceback.print_exc()
    #
    coefDF.to_hdf(resultPath, 'coefficients')
    predDF.to_hdf(resultPath, 'predictions')
    termPalette.to_hdf(resultPath, 'plotOpts')
    #
    try:
        scoresDF['estimator'].to_hdf(resultPath, 'cv_estimators')
    except Exception:
        traceback.print_exc()
    estimatorMetadata = {
        'path': os.path.basename(resultPath),
        'lhsScaler': lhsScaler,
        'iteratorKWArgs': iteratorKWArgs,
        'cvIterator': cvIterator
        }
    with open(resultPicklePath, 'wb') as _f:
        pickle.dump(estimatorMetadata, _f)
    #
    height, width = 3, 3
    aspect = width / height
    commonOpts = dict(
        col='RateInHz', row='pedalMovementCat',
        row_order=['NA', 'outbound', 'return'],
        x='amplitude', y='rauc', hue='term',
        height=height, aspect=aspect, palette=termPalette.to_dict()
    )
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}{}_{}.pdf'.format(
            inputBlockSuffix, arguments['window'], estimatorName, iteratorSuffix,
            'RAUC_regression'))
    with PdfPages(pdfPath) as pdf:
        for targetName, predGroup in predDF.groupby('target'):
            for foldTypeName, predSubGroup0 in predGroup.groupby('foldType'):
                if foldTypeName == 'validation':
                    continue
                for name, predSubGroup in predGroup.groupby(['design', 'electrode']):
                    designName, elecName = name
                    plotData = predSubGroup.stack().to_frame(name='rauc')
                    plotData.reset_index(inplace=True)
                    plotData.loc[:, 'predType'] = 'component'
                    plotData.loc[plotData['term'] == 'ground_truth', 'predType'] = 'ground_truth'
                    plotData.loc[plotData['term'] == 'prediction', 'predType'] = 'prediction'
                    nAmps = plotData['amplitude'].unique()
                    if nAmps.size == 1:
                        g = sns.catplot(
                            data=plotData,
                            kind='box',
                            facet_kws=dict(
                                margin_titles=True),
                            **commonOpts,
                            )
                    else:
                        g = sns.relplot(
                            data=plotData,
                            kind='line', errorbar='sd',
                            style='predType', dashes={
                                'ground_truth': (10, 0),
                                'prediction': (2, 1),
                                'component': (7, 3)
                            },
                            facet_kws=dict(
                                margin_titles=True),
                            **commonOpts,
                        )
                    g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
                    titleText = 'model {}\n{}, electrode {} ({})'.format(designName, targetName, elecName, foldTypeName)
                    print('Saving plot of {}...'.format(titleText))
                    figTitle = g.fig.suptitle(titleText)
                    g._tight_layout_rect[-1] -= 0.25 / g.fig.get_size_inches()[1]
                    g.tight_layout(pad=0.1)
                    pdf.savefig(
                        bbox_inches='tight',
                        bbox_extra_artists=[figTitle, g.legend]
                        )
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
    ############################
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}{}_{}.pdf'.format(
            inputBlockSuffix, arguments['window'], estimatorName, iteratorSuffix,
            'RAUC_regression_coefficients'))
    with PdfPages(pdfPath) as pdf:
        allCoeffNames = coefDF.index.get_level_values('component').unique()
        vMax = coefDF.abs().quantile(1-1e-2)
        vMin = vMax * -1
        for designFormula, theseCoefs in coefDF.groupby('design'):
            plotCoefs = theseCoefs.unstack('component').reindex(columns=allCoeffNames)
            # indexing allCoeffNames adds nans for missing values, so that all the designs have identical rows and cols
            plotCoefs.index = plotCoefs.index.droplevel('design')
            grid_kws = {"width_ratios": (30, 1), 'wspace': 0.01}
            aspect = plotCoefs.shape[1] / plotCoefs.shape[0]
            h = 12
            w = h * aspect
            fig, (ax, cbar_ax) = plt.subplots(
                1, 2,
                gridspec_kw=grid_kws,
                figsize=(w, h))
            ax = sns.heatmap(
                plotCoefs, ax=ax,
                cbar_ax=cbar_ax, vmin=vMin, vmax=vMax, cmap='vlag')
            titleText = 'Coefficients for model\n{}'.format(designFormula)
            figTitle = fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    #########################
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}{}_{}.pdf'.format(
            inputBlockSuffix, arguments['window'], estimatorName, iteratorSuffix,
            'RAUC_mean_regression_coefficients'))
    with PdfPages(pdfPath) as pdf:
        allCoeffNames = coefDF.index.get_level_values('component').unique()
        vMax = coefDF.abs().quantile(1-1e-2)
        vMin = vMax * -1
        for designFormula, theseCoefs in coefDF.groupby('design'):
            plotCoefs = (
                theseCoefs.groupby(['target', 'component']).mean()
                    .unstack('component').reindex(columns=allCoeffNames))
            # indexing allCoeffNames adds nans for missing values, so that all the designs have identical rows and cols
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
            titleText = 'Coefficients for model\n{}'.format(designFormula)
            figTitle = fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    ########################
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}{}_{}.pdf'.format(
            inputBlockSuffix, arguments['window'], estimatorName, iteratorSuffix,
            'RAUC_regression_scores'))
    with PdfPages(pdfPath) as pdf:
        height, width = 3, 3
        aspect = width / height
        trainScores = scoresDF.loc[:, 'train_score'].to_frame(name='score')
        testScores = scoresDF.loc[:, 'test_score'].to_frame(name='score')
        folds = trainScores.index.get_level_values('fold')
        lastFoldIdx = folds.max()
        lookup = {
            k: 'train' if k < lastFoldIdx else 'work' for k in folds.unique()
            }
        trainScores.loc[:, 'scoreType'] = folds.map(lookup)
        #
        lookup2 = {
            k: 'test' if k < lastFoldIdx else 'validation' for k in folds.unique()
            }
        testScores.loc[:, 'scoreType'] = folds.map(lookup2)
        plotScores = pd.concat([testScores, trainScores]).reset_index()
        plotScores.loc[:, 'xDummy'] = 0
        g = sns.catplot(
            data=plotScores, kind='box',
            y='score', x='target',
            col='design', hue='scoreType',
            height=height, aspect=aspect
            )
        g.set_xticklabels(rotation=30, ha='right')
        g.set_titles(template="{col_name}")
        g.tight_layout(pad=0.1)
        pdf.savefig(bbox_inches='tight')
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
