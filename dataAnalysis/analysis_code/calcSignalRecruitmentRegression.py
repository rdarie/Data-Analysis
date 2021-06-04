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
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierMask                    delete outlier trials? [default: False]
    --showFigures                          show plots interactively? [default: False]
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
        'blockIdx': '2', 'maskOutlierBlocks': False, 'window': 'XL', 'verbose':2}
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
    blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
    #
    resultPath = os.path.join(
        calcSubFolder,
        blockBaseName + '{}_{}_rauc.h5'.format(
            inputBlockSuffix, arguments['window']))
    print('loading {}'.format(resultPath))
    outlierTrials = ash.processOutlierTrials(
        scratchFolder, blockBaseName, **arguments)

    nSplits = 7
    splitterKWArgs = dict(
        shuffle=True,
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'])
    joblibBackendArgs = dict(
        # backend='dask',
        backend='loky',
        # n_jobs=1
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
    estimatorInstance = ElasticNet(fit_intercept=False)
    gridSearchKWArgs = dict(
        return_train_score=True,
        refit=False,
        param_grid=dict(
            # l1_ratio=[.1, .5, .7, .9, .95, .99, 1]
            l1_ratio=[.1, .5, .95, 1]
            )
        )
    nAlphas = 50
    lOfDesignFormulas = [
        "pedalMovementCat/((electrode:amplitude)/RateInHz) - 1"
    ]
    recCurve = pd.read_hdf(resultPath, 'scaled')
    #
    cvIterator = tdr.trainTestValidationSplitter(
        recCurve, tdr.trialAwareStratifiedKFold,
        n_splits=nSplits, splitterKWArgs=splitterKWArgs
        )

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

    '''
    pt = PatsyTransformer("", return_type="matrix")
    pipe = Pipeline([('scale', lhsScaler), ('design', pt), ('regression', ElasticNet(fit_intercept=False))])
    '''
    patsyTransfDict = {}
    cvScoresDict0 = {}
    gridSearcherDict0 = {}
    gsScoresDict0 = {}
    for designFormula in lOfDesignFormulas:
        pt = PatsyTransformer(designFormula, return_type="matrix")
        patsyTransfDict[designFormula] = pt
        designMatrix = pt.fit_transform(scaledLhsDF)
        designInfo = designMatrix.design_info
        designDF = (pd.DataFrame(designMatrix, index=lhsDF.index, columns=designInfo.column_names))
        gsParamsPerTarget = None
        if 'param_grid' in gridSearchKWArgs:
            gsKWA = deepcopy(gridSearchKWArgs)
            if 'l1_ratio' in gsKWA['param_grid']:
                paramGrid = gsKWA.pop('param_grid')
                lOfL1Ratios = paramGrid.pop('l1_ratio')
                gsParamsPerTarget = {}
                for columnTuple in rhsDF.columns:
                    targetName = columnTuple[0]
                    gsParamsPerTarget[targetName] = []
                    for l1Ratio in lOfL1Ratios:
                        alphas = _alpha_grid(
                            designDF, rhsDF.loc[:, columnTuple],
                            l1_ratio=l1Ratio, n_alphas=nAlphas)
                        gsParamsPerTarget[targetName].append(
                            dict(
                                l1_ratio=[l1Ratio],
                                alpha=np.atleast_1d(alphas).tolist()
                            )
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
    lastFoldIdx = scoresDF.index.get_level_values('fold').max()
    bestEstimators = scoresDF.xs(lastFoldIdx, level='fold')
    #
    coefList = {}
    predList = {}
    for designFormula in lOfDesignFormulas:
        designMatrixTransformer = Pipeline([('scale', lhsScaler), ('design', patsyTransfDict[designFormula])])
        designMatrix = designMatrixTransformer.transform(lhsDF)
        designInfo = designMatrix.design_info
        designDF = (pd.DataFrame(designMatrix, index=lhsDF.index, columns=designInfo.column_names))
        for estName, estSrs in bestEstimators.iterrows():
            _, targetName = estName
            regressionEstimator = bestEstimators.loc[(designFormula, targetName), 'estimator']
            coefs = pd.Series(regressionEstimator.coef_, index=designInfo.column_names)
            coefList[targetName] = coefs
            ####
            if True:
                for eN in ['+ E16 - E5', '+ E16 - E2']:
                    ampTermStr = 'pedalMovementCat[NA]:electrode[{}]:amplitude'.format(eN)
                    print(ampTermStr)
                    print('Coefficient:\n{}'.format(coefs[ampTermStr]))
                    print('unique terms:\n{}'.format(designDF.loc[:, ampTermStr].unique()))
                    ampRateTermStr = 'pedalMovementCat[NA]:electrode[{}]:amplitude:RateInHz'.format(eN)
                    print(ampRateTermStr)
                    print('Coefficient:\n{}'.format(coefs[ampRateTermStr]))
                    print('unique terms:\n{}\n\n'.format(designDF.loc[:, ampRateTermStr].unique()))
            predictionPerComponent = designDF * coefs
            predictionSrs = predictionPerComponent.sum(axis='columns')
            # sanity
            mismatch = predictionSrs - regressionEstimator.predict(designDF)
            print('max mismatch is {}'.format(mismatch.abs().max()))
            assert (mismatch.abs().max() < 1e-3)
            predictionPerSource = pd.DataFrame(np.nan, index=designDF.index, columns=designInfo.term_names)
            for termName, termSlice in designInfo.term_name_slices.items():
                predictionPerSource.loc[:, termName] = predictionPerComponent.iloc[:, termSlice].sum(axis='columns')
            predictionPerSource.loc[:, 'prediction'] = predictionSrs
            predictionPerSource.loc[:, 'ground_truth'] = rhsDF[targetName].to_numpy()
            predList[targetName] = predictionPerSource

        coefDF = pd.concat(coefList, names=['target'])
        predDF = pd.concat(predList, axis='columns', names=['target', 'component'])
        #
        predDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, stimulusConditionNames + ['foldType', 'segment', 'originalIndex', 't']])
    #
    height, width = 3, 3
    aspect = width / height
    commonOpts = dict(
        row='RateInHz', col='pedalMovementCat',
        x='amplitude', hue='component',
        height=height, aspect=aspect
    )
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '{}_{}_{}.pdf'.format(
            inputBlockSuffix, arguments['window'],
            'RAUC_regression'))
    with PdfPages(pdfPath) as pdf:
        for targetName, predGroup in predDF.groupby('target', axis='columns'):
            for foldTypeName, predSubGroup0 in predGroup.groupby('foldType'):
                for elecName, predSubGroup in predGroup.groupby('electrode'):
                    plotData = predSubGroup.stack().reset_index()
                    plotData.loc[:, 'predType'] = 'component'
                    plotData.loc[plotData['component'] == 'ground_truth', 'predType'] = 'ground_truth'
                    plotData.loc[plotData['component'] == 'prediction', 'predType'] = 'prediction'
                    nAmps = plotData['amplitude'].unique()
                    if nAmps.size == 1:
                        g = sns.catplot(
                            data=plotData, y=targetName,
                            kind='box',
                            facet_kws=dict(
                                margin_titles=True),
                            **commonOpts,
                            )
                    else:
                        g = sns.relplot(
                            data=plotData, y=targetName,
                            kind='line', errorbar='se',
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
                    titleText = '{}, electrode {} ({})'.format(targetName, elecName, foldTypeName)
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

    print(designInfo.term_names); print(coefs)
