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
import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklego.preprocessing import PatsyTransformer
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from itertools import product
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
idxSl = pd.IndexSlice
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'analysisName': 'hiRes', 'datasetName': 'Synthetic_XL_df_g', 'estimatorName': 'enr',
        'processAll': True, 'verbose': '1', 'plotting': True, 'exp': 'exp202101281100',
        'showFigures': True, 'alignFolderName': 'motion', 'blockIdx': '2', 'debugging': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    scratchPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings'
    scratchFolder = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202101201100-Rupert'
    figureFolder = '/gpfs/data/dborton/rdarie/Neural Recordings/processed/202101201100-Rupert/figures'
    
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
'''fullEstimatorName = '{}_{}_to_{}{}_{}_{}'.format(
    arguments['estimatorName'],
    arguments['unitQueryLhs'], arguments['unitQueryRhs'],
    iteratorSuffix,
    arguments['window'],
    arguments['alignQuery'])'''
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
#
binInterval = iteratorOpts['forceBinInterval'] if iteratorOpts['forceBinInterval'] is not None else rasterOpts['binInterval']
addHistoryTerms = {
    'nb': 5,
    'dt': binInterval,
    # 'historyLen': iteratorOpts['covariateHistoryLen'],
    'historyLen': 100e-3,
    'b': 0.001, 'useOrtho': False,
    'normalize': True, 'groupBy': 'trialUID',
    'zflag': False}
rcb = tdr.patsyRaisedCosTransformer
thisEnv = patsy.EvalEnvironment.capture()

iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
# cv_kwargs = loadingMeta['cv_kwargs'].copy()
cvIterator = iteratorsBySegment[0]
#
selectionNameLhs = estimatorMeta['arguments']['selectionNameLhs']
selectionNameRhs = estimatorMeta['arguments']['selectionNameRhs']
#

#
if estimatorMeta['pipelinePathRhs'] is not None:
    transformedSelectionNameRhs = '{}_{}'.format(
        selectionNameRhs, estimatorMeta['arguments']['transformerNameRhs'])
    transformedRhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(transformedSelectionNameRhs))
    pipelineScoresRhsDF = pd.read_hdf(estimatorMeta['pipelinePathRhs'], 'work')
    workingPipelinesRhs = pipelineScoresRhsDF['estimator']
else:
    workingPipelinesRhs = None
#
if estimatorMeta['pipelinePathLhs'] is not None:
    pipelineScoresLhsDF = pd.read_hdf(estimatorMeta['pipelinePathLhs'], 'work')
    workingPipelinesLhs = pipelineScoresLhsDF['estimator']
else:
    workingPipelinesLhs = None
#
estimatorsDF = pd.read_hdf(estimatorPath, 'cv_estimators')
scoresDF = pd.read_hdf(estimatorPath, 'cv_scores')
lhsDF = pd.read_hdf(estimatorMeta['lhsDatasetPath'], '/{}/data'.format(selectionNameLhs))
rhsDF = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/data'.format(selectionNameRhs))
lhsMasks = pd.read_hdf(estimatorMeta['lhsDatasetPath'], '/{}/featureMasks'.format(selectionNameLhs))
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
#
trialInfoLhs = lhsDF.index.to_frame().reset_index(drop=True)
trialInfoRhs = rhsDF.index.to_frame().reset_index(drop=True)
checkSameMeta = stimulusConditionNames + ['bin', 'trialUID']
assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
trialInfo = trialInfoLhs
#
lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
#
coefDict0 = {}
predDict0 = {}
for lhsRowIdx, rhsRowIdx in product(list(range(lhsMasks.shape[0])), list(range(rhsMasks.shape[0]))):
    lhsMask = lhsMasks.iloc[lhsRowIdx, :]
    rhsMask = rhsMasks.iloc[rhsRowIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMask.index.names, lhsMask.name)}
    rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
    lhGroup = lhsDF.loc[:, lhsMask]
    #
    lhGroup.columns = lhGroup.columns.get_level_values('feature')
    designFormula = lhsMask.name[lhsMasks.index.names.index('designFormula')]
    #
    pt = PatsyTransformer(designFormula, return_type="matrix")
    designMatrix = pt.fit_transform(lhGroup)
    designInfo = designMatrix.design_info
    designDF = (
        pd.DataFrame(
            designMatrix,
            index=lhGroup.index,
            columns=designInfo.column_names))
    designDF.columns.name = 'feature'
    rhGroup = rhsDF.loc[:, rhsMask].copy()
    # transform to PCs
    if workingPipelinesRhs is not None:
        transformPipelineRhs = workingPipelinesRhs.xs(rhsMaskParams['freqBandName'], level='freqBandName').iloc[0]
        rhsPipelineMinusAverager = Pipeline(transformPipelineRhs.steps[1:])
        rhsPipelineAverager = transformPipelineRhs.named_steps['averager']
        rhTransformedColumns = transformedRhsDF.columns[transformedRhsDF.columns.get_level_values('freqBandName') == rhsMaskParams['freqBandName']]
        rhGroup = pd.DataFrame(
            rhsPipelineMinusAverager.transform(rhGroup),
            index=lhGroup.index, columns=rhTransformedColumns)
    else:
        rhsPipelineAverager = Pipeline[('averager', tdr.DataFramePassThrough(), )]
    rhGroup.columns = rhGroup.columns.get_level_values('feature')
    for targetName in rhGroup.columns:
        for foldIdx in range(cvIterator.n_splits + 1):
            targetDF = rhGroup.loc[:, [targetName]]
            estimatorIdx = (lhsRowIdx, rhsRowIdx, targetName, foldIdx)
            print('estimator: {}'.format(estimatorIdx))
            print('in dataframe: {}'.format(estimatorIdx in estimatorsDF.index))
            if not estimatorIdx in estimatorsDF.index:
                continue
            if foldIdx == cvIterator.n_splits:
                # work and validation folds
                trainIdx, testIdx = cvIterator.workIterator.split(rhGroup)[0]
                trainStr, testStr = 'work', 'validation'
                foldType = 'validation'
            else:
                trainIdx, testIdx = cvIterator.raw_folds[foldIdx]
                trainStr, testStr = 'train', 'test'
                foldType = 'train'
            estimator = estimatorsDF.loc[estimatorIdx]
            coefs = pd.Series(
                estimator.regressor_.named_steps['regressor'].coef_, index=designDF.columns)
            coefDict0[estimatorIdx] = coefs
            estPreprocessorLhs = Pipeline(estimator.regressor_.steps[:-1])
            estPreprocessorRhs = estimator.transformer_
            predictionPerComponent = pd.concat({
                trainStr: estPreprocessorLhs.transform(designDF.iloc[trainIdx, :]) * coefs,
                testStr: estPreprocessorLhs.transform(designDF.iloc[testIdx, :]) * coefs
                }, names=['trialType'])
            predictionSrs = predictionPerComponent.sum(axis='columns')
            # sanity
            indicesThisFold = np.concatenate([trainIdx, testIdx])
            predictionsNormalWay = np.concatenate([
                estimator.predict(designDF.iloc[trainIdx, :]),
                estimator.predict(designDF.iloc[testIdx, :])
                ])
            mismatch = predictionSrs - predictionsNormalWay.reshape(-1)
            print('max mismatch is {}'.format(mismatch.abs().max()))
            assert (mismatch.abs().max() < 1e-3)
            predictionPerSource = pd.DataFrame(
                np.nan, index=predictionPerComponent.index,
                columns=designInfo.term_names)
            for termName, termSlice in designInfo.term_name_slices.items():
                predictionPerSource.loc[:, termName] = predictionPerComponent.iloc[:, termSlice].sum(axis='columns')
            predictionPerSource.loc[:, 'prediction'] = predictionSrs
            predictionPerSource.loc[:, 'ground_truth'] = np.concatenate([
                estPreprocessorRhs.transform(targetDF.iloc[trainIdx, :]),
                estPreprocessorRhs.transform(targetDF.iloc[testIdx, :])
                ])
            predDict0[(lhsRowIdx, rhsRowIdx, targetName, foldIdx, foldType)] = predictionPerSource
predDF = pd.concat(predDict0, names=['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'foldType'])
predDF.columns.name = 'term'
coefDF = pd.concat(coefDict0, names=['lhsMaskIdx', 'rhsMaskIdx', 'target', 'fold', 'foldType'])

nTerms = predDF.columns.size
termPalette = pd.Series(
    sns.color_palette('Set2', nTerms),
    index=predDF.columns)

coefDF.to_hdf(estimatorPath, 'coefficients')
predDF.to_hdf(estimatorPath, 'predictions')
termPalette.to_hdf(estimatorPath, 'plotOpts')

groupPagesBy = ['rhsMaskIdx', 'lhsMaskIdx', 'target']
groupSubPagesBy = ['trialType', 'foldType', 'electrode']
scoresForPlot = pd.concat({
        'test': scoresDF['test_score'],
        'train': scoresDF['train_score']},
    names=['trialType']
    ).to_frame(name='score').reset_index()
#
lastFoldMask = (scoresForPlot['fold'] == cvIterator.n_splits)
trainMask = (scoresForPlot['trialType'] == 'train')
#
scoresForPlot.loc[:, 'foldType'] = ''
scoresForPlot.loc[(trainMask & lastFoldMask), 'foldType'] = 'work'
scoresForPlot.loc[(trainMask & (~lastFoldMask)), 'foldType'] = 'train'
scoresForPlot.loc[((~trainMask) & lastFoldMask), 'foldType'] = 'validation'
scoresForPlot.loc[((~trainMask) & (~lastFoldMask)), 'foldType'] = 'test'
scoresForPlot.loc[:, 'dummyX'] = 0
#
height, width = 3, 3
aspect = width / height
commonOpts = dict(
    )
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'reconstructions'))

with PdfPages(pdfPath) as pdf:
    for name0, predGroup0 in predDF.groupby(groupPagesBy):
        nmLk0 = {key: value for key, value in zip(groupPagesBy, name0)} # name lookup
        #
        scoreMasks = [
            scoresForPlot[cN] == nmLk0[cN]
            for cN in groupPagesBy]
        theseScoresPlot = scoresForPlot.loc[np.logical_and.reduce(scoreMasks), :]
        g = sns.catplot(
            data=theseScoresPlot, hue='foldType',
            x='dummyX', y='score',
            kind='box')
        g.fig.suptitle('R^2 for target {target}'.format(**nmLk0))
        newYLims = theseScoresPlot['score'].quantile([0.25, 1 - 1e-3]).to_list()
        for ax in g.axes.flat:
            ax.set_xlabel('regression target')
            ax.set_ylabel('R2 of ordinary least squares fit')
            ax.set_ylim(newYLims)
        g.fig.tight_layout(pad=1)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        for name1, predGroup1 in predGroup0.groupby(groupSubPagesBy):
            nmLk1 = {key: value for key, value in zip(groupSubPagesBy, name1)} # name lookup
            nmLk0.update(nmLk1)
            if nmLk0['trialType'] != 'train':
                continue
            plotData = predGroup1.stack().to_frame(name='signal').reset_index()
            plotData.loc[:, 'predType'] = 'component'
            plotData.loc[plotData['term'] == 'ground_truth', 'predType'] = 'ground_truth'
            plotData.loc[plotData['term'] == 'prediction', 'predType'] = 'prediction'
            g = sns.relplot(
                data=plotData,
                col='trialAmplitude', row='pedalMovementCat',
                row_order=['NA', 'outbound', 'return'],
                x='bin', y='signal', hue='term',
                height=height, aspect=aspect, palette=termPalette.to_dict(),
                kind='line', errorbar='sd',
                style='predType', dashes={
                    'ground_truth': (10, 0),
                    'prediction': (2, 1),
                    'component': (7, 3)
                }, facet_kws=dict(margin_titles=True),
                )
            g.set_titles(template="{col_var}\n{col_name}\n{row_var}\n{row_name}")
            titleText = 'model {lhsMaskIdx}\n{target}, electrode {electrode} ({foldType})'.format(
                **nmLk0)
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