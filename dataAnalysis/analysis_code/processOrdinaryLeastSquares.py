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
import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklego.preprocessing import PatsyTransformer
from dataAnalysis.analysis_code.regression_parameters import *
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
import patsy
from itertools import product
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
idxSl = pd.IndexSlice
useDPI = 200
dpiFactor = 72 / useDPI
snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': 1,
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
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
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
    'figure.titlesize': 7
    }
styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 3e-1, # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''

consoleDebugging = True
if consoleDebugging:
    arguments = {
        'processAll': True, 'blockIdx': '2', 'analysisName': 'hiRes', 'datasetName': 'Block_XL_df_d', 'verbose': '1',
        'debugging': False, 'plotting': True, 'showFigures': False, 'alignFolderName': 'motion',
        'estimatorName': 'enr_ta_spectral', 'exp': 'exp202101281100'}
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

for hIdx, histOpts in enumerate(addHistoryTerms):
    locals().update({'hto{}'.format(hIdx): getHistoryOpts(histOpts, iteratorOpts, rasterOpts)})
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
checkSameMeta = stimulusConditionNames + ['bin', 'trialUID', 'conditionUID']
assert (trialInfoRhs.loc[:, checkSameMeta] == trialInfoLhs.loc[:, checkSameMeta]).all().all()
trialInfo = trialInfoLhs
#
lhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
rhsDF.index = pd.MultiIndex.from_frame(trialInfo.loc[:, checkSameMeta])
#
with pd.HDFStore(estimatorPath) as store:
    if 'coefficients' in store:
        coefDF = pd.read_hdf(store, 'coefficients')
        loadedCoefs = True
    else:
        loadedPreds = False
    if 'predictions' in store:
        predDF = pd.read_hdf(store, 'predictions')
        loadedPreds = True
    else:
        loadedPreds = False
    if 'plotOpts' in store:
        termPalette = pd.read_hdf(store, 'plotOpts')
        loadedPlotOpts = True
    else:
        loadedPlotOpts = False

if not (loadedCoefs and loadedPreds):
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
        pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
        exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
        pt.fit(exampleLhGroup)
        designMatrix = pt.transform(lhGroup)
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
    #
    coefDF.to_hdf(estimatorPath, 'coefficients')
    predDF.to_hdf(estimatorPath, 'predictions')
    termPalette.to_hdf(estimatorPath, 'plotOpts')

if not loadedPlotOpts:
    nTerms = predDF.columns.size
    termPalette = pd.Series(
        sns.color_palette('Set2', nTerms),
        index=predDF.columns)

groupPagesBy = ['rhsMaskIdx', 'lhsMaskIdx', 'target']
groupSubPagesBy = ['trialType', 'foldType', 'electrode']
scoresStack = pd.concat({
        'test': scoresDF['test_score'],
        'train': scoresDF['train_score']},
    names=['trialType']
    ).to_frame(name='score').reset_index()
#
lastFoldMask = (scoresStack['fold'] == cvIterator.n_splits)
trainMask = (scoresStack['trialType'] == 'train')
#
scoresStack.loc[:, 'foldType'] = ''
scoresStack.loc[(trainMask & lastFoldMask), 'foldType'] = 'work'
scoresStack.loc[(trainMask & (~lastFoldMask)), 'foldType'] = 'train'
scoresStack.loc[((~trainMask) & lastFoldMask), 'foldType'] = 'validation'
scoresStack.loc[((~trainMask) & (~lastFoldMask)), 'foldType'] = 'test'
scoresStack.loc[:, 'dummyX'] = 0
#
llDict1 = {}
for scoreName, targetScores in scoresStack.groupby(['lhsMaskIdx', 'target', 'fold']):
    lhsMaskIdx, targetName, fold = scoreName
    estimator = estimatorsDF.loc[idxSl[lhsMaskIdx, :, targetName, fold]].iloc[0]
    regressor = estimator.regressor_.steps[-1][1]
    thesePred = predDF.xs(targetName, level='target').xs(lhsMaskIdx, level='lhsMaskIdx').xs(fold, level='fold')
    llDict2 = {}
    for name, predGroup in thesePred.groupby(['electrode', 'trialType']):
        llDict3 = {}
        llDict3['llSat'] = regressor.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
        nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
        llDict3['llNull'] = regressor.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
        llDict3['llFull'] = regressor.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
        llDict2[name] = pd.Series(llDict3)
    for trialType, predGroup in thesePred.groupby('trialType'):
        llDict3 = {}
        llDict3['llSat'] = regressor.results_.family.loglike(predGroup['ground_truth'].to_numpy(), predGroup['ground_truth'].to_numpy())
        nullModel = ((predGroup['ground_truth'] ** 0) * predGroup['ground_truth'].mean()).to_numpy()
        llDict3['llNull'] = regressor.results_.family.loglike(nullModel, predGroup['ground_truth'].to_numpy())
        llDict3['llFull'] = regressor.results_.family.loglike(predGroup['prediction'].to_numpy(), predGroup['ground_truth'].to_numpy())
        llDict2[('all', trialType)] = pd.Series(llDict3)
    llDict1[scoreName] = pd.concat(llDict2, names=['electrode', 'trialType', 'llType'])
llDF = pd.concat(llDict1, names=['lhsMaskIdx', 'target', 'fold', 'electrode', 'trialType', 'llType']).to_frame(name='ll')
R2Per = llDF['ll'].groupby(['lhsMaskIdx', 'target', 'electrode', 'fold', 'trialType']).apply(getR2).to_frame(name='score')
#
R2Per.loc[:, 'design'] = R2Per.reset_index()['lhsMaskIdx'].apply(lambda x: lOfDesignFormulas[x]).to_numpy()
R2Per.set_index('design', append=True, inplace=True)
#
llDF.loc[:, 'design'] = llDF.reset_index()['lhsMaskIdx'].apply(lambda x: lOfDesignFormulas[x]).to_numpy()
llDF.set_index('design', append=True, inplace=True)

modelsToTest = [
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0))',
            'captionStr': 'partial R2 of allowing pedal velocity to modulate electrode coefficients, vs assuming their independence'
        },
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'rcb(v,**hto0)',
            'captionStr': 'partial R2 of including any electrode coefficients'
        },
        {
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'refDesign': 'rcb(v,**hto0) + e:rcb(a,**hto0) + e:rcb(v*a,**hto0)',
            'captionStr': 'partial R2 of including a term for modulation of electrode coefficients by RateInHz'
        },
        {
            'refDesign': None,
            'testDesign': 'rcb(v,**hto0) + e:(rcb(a,**hto0)+rcb(a*r,**hto0)) + e:(rcb(v*a,**hto0)+rcb(v*a*r,**hto0))',
            'captionStr': 'R2 of design V + AR + VAR'
        },
    ]
pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'partial_scores'))
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
        pR2 = partialR2(llDF['ll'], refDesign=refDesign, testDesign=testDesign)
        #
        plotScores = pR2.reset_index().rename(columns={'ll': 'score'})
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
        # g.axes.flat[0].set_ylim(allScoreQuantiles)
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

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'r2'))
with PdfPages(pdfPath) as pdf:
    for name, theseScores in scoresStack.groupby('rhsMaskIdx'):
        g = sns.catplot(
            data=theseScores, hue='foldType',
            x='target', y='score',
            kind='box')
        g.fig.suptitle('Frequency band')
        g.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()

height, width = 2, 2
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
            scoresStack[cN] == nmLk0[cN]
            for cN in groupPagesBy]
        theseScoresPlot = scoresStack.loc[np.logical_and.reduce(scoreMasks), :]
        g = sns.catplot(
            data=theseScoresPlot, hue='foldType',
            x='dummyX', y='score',
            kind='box')
        g.fig.suptitle('R^2 for target {target}'.format(**nmLk0))
        # newYLims = theseScoresPlot['score'].quantile([0.25, 1 - 1e-3]).to_list()
        # for ax in g.axes.flat:
        #     ax.set_xlabel('regression target')
        #     ax.set_ylabel('R2 of ordinary least squares fit')
        #     ax.set_ylim(newYLims)
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
            leg = g._legend
            if leg is not None:
                t = leg.get_title()
                tContent = t.get_text()
                # if tContent in titleOverrides:
                #     t.set_text(titleOverrides[tContent])
                # for t in leg.texts:
                #     tContent = t.get_text()
                #     if tContent in titleOverrides:
                #         t.set_text(titleOverrides[tContent])
                #     elif tContent in emgNL.index:
                #         t.set_text('{}'.format(emgNL[tContent]))
                for l in leg.get_lines():
                    # l.set_lw(2 * l.get_lw())
                    l.set_lw(styleOpts['legend.lw'])
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