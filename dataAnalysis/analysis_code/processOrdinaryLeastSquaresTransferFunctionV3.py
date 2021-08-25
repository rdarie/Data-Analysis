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
    --forceReprocess                         print diagnostics? [default: False]
    --estimatorName=estimatorName            filename for resulting estimator (cross-validated n_comps)
    --datasetName=datasetName                filename for resulting estimator (cross-validated n_comps)
    --analysisName=analysisName              append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName        append a name to the resulting blocks? [default: motion]
"""

import logging
logging.captureWarnings(True)
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.plotting.aligned_signal_plots as asp
from dataAnalysis.custom_transformers.tdr import getR2, partialR2
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb, traceback
import numpy as np
import pandas as pd
from scipy.stats import zscore, mode
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from dataAnalysis.analysis_code.regression_parameters import *
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import patsy
from sklego.preprocessing import PatsyTransformer
import dill as pickle
pickle.settings['recurse'] = True
import gc
from itertools import product
from docopt import docopt
import colorsys
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
    'tight_layout.pad': 3e-1,  # units of font size
    'panel_heading.pad': 0.
    }
sns.set(
    context='paper', style='white',
    palette='dark', font='sans-serif',
    font_scale=.8, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    matplotlib.rcParams[rcK] = rcV
#
print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'showFigures': False, 'analysisName': 'hiRes', 'exp': 'exp202101271100', 'datasetName': 'Block_XL_df_ra',
        'debugging': False, 'blockIdx': '2', 'alignFolderName': 'motion', 'estimatorName': 'enr_fa_ta',
        'plotting': True, 'verbose': '1', 'processAll': True}
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
'''if estimatorMeta['pipelinePathRhs'] is not None:
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
    workingPipelinesLhs = None'''
#
lhsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'lhsDF')
lhsMasks = pd.read_hdf(estimatorMeta['designMatrixPath'], 'featureMasks')
allTargetsDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'allTargets')
rhsMasks = pd.read_hdf(estimatorMeta['rhsDatasetPath'], '/{}/featureMasks'.format(selectionNameRhs))
#
lhsMasksInfo = lhsMasks.index.to_frame().reset_index(drop=True)
lhsMasksInfo.loc[:, 'ensembleFormulaDescr'] = lhsMasksInfo['ensembleTemplate'].apply(lambda x: x.format('ensemble'))
lhsMasksInfo.loc[:, 'selfFormulaDescr'] = lhsMasksInfo['selfTemplate'].apply(lambda x: x.format('self'))
lhsMasksInfo.loc[:, 'designFormulaShortHand'] = lhsMasksInfo['designFormula'].apply(lambda x: formulasShortHand[x])
lhsMasksInfo.loc[:, 'fullFormulaDescr'] = lhsMasksInfo.loc[:, ['designFormulaShortHand', 'ensembleFormulaDescr', 'selfFormulaDescr']].apply(lambda x: ' + '.join(x), axis='columns')
#
rhsMasksInfo = rhsMasks.index.to_frame().reset_index(drop=True)

estimatorsDict = {}
scoresDict = {}
if processSlurmTaskCount is not None:
    slurmGroupSize = int(np.ceil(allTargetsDF.shape[0] / processSlurmTaskCount))
    allTargetsDF.loc[:, 'parentProcess'] = allTargetsDF['targetIdx'] // slurmGroupSize
for rowIdx, row in allTargetsDF.iterrows():
    lhsMaskIdx, rhsMaskIdx, targetName = row.name
    if processSlurmTaskCount is not None:
        thisEstimatorPath = estimatorPath.replace('.h5', '_{}.h5'.format(row['parentProcess']))
    else:
        thisEstimatorPath = estimatorPath
    scoresDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
        thisEstimatorPath,
        'cv_scores/lhsMask_{}/rhsMask_{}/{}'.format(
            lhsMaskIdx, rhsMaskIdx, targetName
        ))
    estimatorsDict[(lhsMaskIdx, rhsMaskIdx, targetName)] = pd.read_hdf(
        thisEstimatorPath,
        'cv_estimators/lhsMask_{}/rhsMask_{}/{}'.format(
            lhsMaskIdx, rhsMaskIdx, targetName
            ))
estimatorsDF = pd.concat(estimatorsDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])
scoresDF = pd.concat(scoresDict, names=['lhsMaskIdx', 'rhsMaskIdx', 'target'])

with pd.HDFStore(estimatorPath) as store:
    if 'coefficients' in store:
        coefDF = pd.read_hdf(store, 'coefficients')
        loadedCoefs = True
    else:
        loadedCoefs = False
    iRsExist = (
        ('impulseResponsePerTerm' in store) &
        ('impulseResponsePerFactor' in store)
        )
    if iRsExist:
        iRPerTerm = pd.read_hdf(store, 'impulseResponsePerTerm')
        iRPerFactor = pd.read_hdf(store, 'impulseResponsePerFactor')
        stimConditionLookupIR = pd.read_hdf(store, 'impulseResponseStimConditionLookup')
        kinConditionLookupIR = pd.read_hdf(store, 'impulseResponseKinConditionLookup')
        loadedIR = True
    else:
        loadedIR = False
    if 'termPalette' in store:
        sourcePalette = pd.read_hdf(store, 'sourcePalette')
        termPalette = pd.read_hdf(store, 'termPalette')
        factorPalette = pd.read_hdf(store, 'factorPalette')
        trialTypePalette = pd.read_hdf(store, 'trialTypePalette')
        loadedPlotOpts = True
    else:
        loadedPlotOpts = False
####
# prep rhs dataframes
histDesignInfoDict = {}
histImpulseDict = {}
histSourceTermDict = {}
for rhsMaskIdx in range(rhsMasks.shape[0]):
    prf.print_memory_usage('\n Prepping RHS dataframes (rhsRow: {})\n'.format(rhsMaskIdx))
    rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
    rhsMaskParams = {k: v for k, v in zip(rhsMasks.index.names, rhsMask.name)}
    rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
    for ensTemplate in lOfHistTemplates:
        if ensTemplate != 'NULL':
            histSourceTermDict.update({ensTemplate.format(cN): cN for cN in rhGroup.columns})
            ensFormula = ' + '.join([ensTemplate.format(cN) for cN in rhGroup.columns])
            ensFormula += ' - 1'
            print('Generating endog design info for {}'.format(ensFormula))
            ensPt = PatsyTransformer(ensFormula, eval_env=thisEnv, return_type="matrix")
            exampleRhGroup = rhGroup.loc[rhGroup.index.get_level_values('conditionUID') == 0, :]
            ensPt.fit(exampleRhGroup)
            ensDesignMatrix = ensPt.transform(exampleRhGroup)
            ensDesignInfo = ensDesignMatrix.design_info
            print(ensDesignInfo.term_names)
            print('\n')
            histDesignInfoDict[(rhsMaskIdx, ensTemplate)] = ensDesignInfo
            #
            impulseDF = tdr.makeImpulseLike(exampleRhGroup)
            impulseDM = ensPt.transform(impulseDF)
            ensImpulseDesignDF = (
                pd.DataFrame(
                    impulseDM,
                    index=impulseDF.index,
                    columns=ensDesignInfo.column_names))
            ensImpulseDesignDF.columns.name = 'factor'
            histImpulseDict[(rhsMaskIdx, ensTemplate)] = ensImpulseDesignDF
#
designInfoDict = {}
impulseDict = {}
for lhsMaskIdx in range(lhsMasks.shape[0]):
    lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
    lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
    designFormula = lhsMaskParams['designFormula']
    #
    if designFormula != 'NULL':
        if designFormula not in designInfoDict:
            print('Generating exog design info for {}'.format(designFormula))
            formulaIdx = lOfDesignFormulas.index(designFormula)
            lhGroup = lhsDF.loc[:, lhsMask]
            pt = PatsyTransformer(designFormula, eval_env=thisEnv, return_type="matrix")
            exampleLhGroup = lhGroup.loc[lhGroup.index.get_level_values('conditionUID') == 0, :]
            designMatrix = pt.fit_transform(exampleLhGroup)
            designInfo = designMatrix.design_info
            designInfoDict[designFormula] = designInfo
            #
            impulseDF = tdr.makeImpulseLike(
                exampleLhGroup, categoricalCols=['e'], categoricalIndex=['electrode'])
            impulseDM = pt.transform(impulseDF)
            impulseDesignDF = (
                pd.DataFrame(
                    impulseDM,
                    index=impulseDF.index,
                    columns=designInfo.column_names))
            impulseDesignDF.columns.name = 'factor'
            impulseDict[designFormula] = impulseDesignDF
designInfoDF = pd.Series(designInfoDict).to_frame(name='designInfo')
designInfoDF.index.name = 'design'
histDesignInfoDF = pd.DataFrame(
    [value for key, value in histDesignInfoDict.items()],
    columns=['designInfo'])
histDesignInfoDF.index = pd.MultiIndex.from_tuples(
    [key for key, value in histDesignInfoDict.items()],
    names=['rhsMaskIdx', 'ensTemplate'])
################################################################################################
if (not loadedCoefs) or arguments['forceReprocess']:
    coefDict0 = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        lhGroup = lhsDF.loc[:, lhsMask]
        if designFormula != 'NULL':
            designInfo = designInfoDict[designFormula]
            formulaIdx = lOfDesignFormulas.index(designFormula)
            designDF = pd.read_hdf(estimatorMeta['designMatrixPath'], 'designs/formula_{}'.format(formulaIdx))
            exogList = [designDF]
            designTermNames = designInfo.term_names
        else:
            exogList = []
            designInfo = None
            designDF = None
            designTermNames = []
        #
        # add ensemble to designDF?
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        for rhsMaskIdx in range(rhsMasks.shape[0]):
            print('\n    On rhsRow {}\n'.format(rhsMaskIdx))
            rhsMask = rhsMasks.iloc[rhsMaskIdx, :]
            rhsMaskParams = {k: v for k, v in zip(rhsMask.index.names, rhsMask.name)}
            rhGroup = pd.read_hdf(
                estimatorMeta['designMatrixPath'],
                'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
            if ensTemplate != 'NULL':
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
            if selfTemplate != 'NULL':
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
            ####
            for targetName in rhGroup.columns:
                # add targetDF to designDF?
                if ensTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(ensTemplate)
                    thisEnsDesign = pd.read_hdf(
                        estimatorMeta['designMatrixPath'],
                        'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    ensHistList = [
                        thisEnsDesign.iloc[:, sl]
                        for key, sl in ensDesignInfo.term_name_slices.items()
                        if key != ensTemplate.format(targetName)]
                    del thisEnsDesign
                    ensTermNames = [
                        tN
                        for tN in ensDesignInfo.term_names
                        if tN != ensTemplate.format(targetName)]
                else:
                    ensHistList = []
                    ensTermNames = []
                #
                if selfTemplate != 'NULL':
                    templateIdx = lOfHistTemplates.index(selfTemplate)
                    thisSelfDesign = pd.read_hdf(
                        estimatorMeta['designMatrixPath'],
                        'histDesigns/rhsMask_{}/template_{}'.format(rhsMaskIdx, templateIdx))
                    selfHistList = [
                        thisSelfDesign.iloc[:, sl].copy()
                        for key, sl in selfDesignInfo.term_name_slices.items()
                        if key == selfTemplate.format(targetName)]
                    del thisSelfDesign
                    selfTermNames = [
                        tN
                        for tN in selfDesignInfo.term_names
                        if tN == selfTemplate.format(targetName)]
                    # print('selfHistList:\n{}\nselfTermNames:\n{}'.format(selfHistList, selfTermNames))
                else:
                    selfHistList = []
                    selfTermNames = []
                #
                fullDesignList = exogList + ensHistList + selfHistList
                fullDesignDF = pd.concat(fullDesignList, axis='columns')
                del fullDesignList
                for foldIdx in range(cvIterator.n_splits + 1):
                    targetDF = rhGroup.loc[:, [targetName]]
                    estimatorIdx = (lhsMaskIdx, rhsMaskIdx, targetName, foldIdx)
                    if int(arguments['verbose']) > 0:
                        print('estimator: {}'.format(estimatorIdx))
                        print('in dataframe: {}'.format(estimatorIdx in estimatorsDF.index))
                    if not estimatorIdx in estimatorsDF.index:
                        continue
                    estimator = estimatorsDF.loc[estimatorIdx]
                    coefs = pd.Series(
                        estimator.regressor_.named_steps['regressor'].coef_, index=fullDesignDF.columns)
                    coefDict0[(lhsMaskIdx, designFormula, rhsMaskIdx, targetName, foldIdx)] = coefs
    coefDF = pd.concat(coefDict0, names=['lhsMaskIdx', 'design', 'rhsMaskIdx', 'target', 'fold', 'factor'])
    # not allowed to save coefDF 'might interfere with processOrdinaryLeastSquares
    # coefDF.to_hdf(estimatorPath, 'coefficients')
else:
    print('Coefficients loaded from .h5 file')
###
if (not loadedPlotOpts) or arguments['forceReprocess']:
    termPalette = pd.concat({
        'exog': pd.Series(np.unique(np.concatenate([di.term_names for di in designInfoDF['designInfo']]))),
        'endog': pd.Series(np.unique(np.concatenate([di.term_names for di in histDesignInfoDF['designInfo']]))),
        'other': pd.Series(['prediction', 'ground_truth']),
        }, names=['type', 'index']).to_frame(name='term')
    sourceTermLookup = pd.concat({
        'exog': pd.Series(sourceTermDict).to_frame(name='source'),
        'endog': pd.Series(histSourceTermDict).to_frame(name='source'),
        'other': pd.Series(
            ['prediction', 'ground_truth'],
            index=['prediction', 'ground_truth']).to_frame(name='source'),}, names=['type', 'term'])
    #
    primaryPalette = pd.DataFrame(sns.color_palette('colorblind'), columns=['r', 'g', 'b'])
    pickingColors = False
    if pickingColors:
        sns.palplot(primaryPalette.apply(lambda x: tuple(x), axis='columns'))
        palAx = plt.gca()
        for tIdx, tN in enumerate(primaryPalette.index):
            palAx.text(tIdx, .5, '{}'.format(tN))
    rgb = pd.DataFrame(
        primaryPalette.iloc[[1, 0, 2, 4, 7], :].to_numpy(),
        columns=['r', 'g', 'b'], index=['v', 'a', 'r', 'ens', 'prediction'])
    hls = rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hls(*x), index=['h', 'l', 's']), axis='columns')
    hls.loc['a*r', :] = hls.loc[['a', 'r'], :].mean()
    hls.loc['v*r', :] = hls.loc[['v', 'r'], :].mean()
    hls.loc['v*a', :] = hls.loc[['v', 'v', 'a'], :].mean()
    hls.loc['v*a*r', :] = hls.loc[['v', 'a', 'r'], :].mean()
    for sN in ['a*r', 'v*r', 'v*a', 'v*a*r']:
        hls.loc[sN, 's'] = hls.loc[sN, 's'] * 0.75
        hls.loc[sN, 'l'] = hls.loc[sN, 'l'] * 1.2
    hls.loc['v*a*r', 's'] = hls.loc['v*a*r', 's'] * 0.5
    hls.loc['v*a*r', 'l'] = hls.loc['v*a*r', 'l'] * 1.5
    for rhsMaskIdx in range(rhsMasks.shape[0]):
        rhGroup = pd.read_hdf(estimatorMeta['designMatrixPath'], 'rhGroups/rhsMask_{}/'.format(rhsMaskIdx))
        lumVals = np.linspace(0.3, 0.7, rhGroup.shape[1])
        for cIdx, cN in enumerate(rhGroup.columns):
            hls.loc[cN, :] = hls.loc['ens', :]
            hls.loc[cN, 'l'] = lumVals[cIdx]
    hls.loc['ground_truth', :] = hls.loc['prediction', :]
    hls.loc['ground_truth', 'l'] = hls.loc['prediction', 'l'] * 0.25
    primarySourcePalette = hls.apply(lambda x: pd.Series(colorsys.hls_to_rgb(*x), index=['r', 'g', 'b']), axis='columns')
    sourcePalette = primarySourcePalette.apply(lambda x: tuple(x), axis='columns')
    if pickingColors:
        sns.palplot(sourcePalette, size=sourcePalette.shape[0])
        palAx = plt.gca()
        for tIdx, tN in enumerate(sourcePalette.index):
            palAx.text(tIdx, .5, '{}'.format(tN))
    ########################################################################################
    factorPaletteDict = {}
    endoFactors = []
    for designFormula, row in designInfoDF.iterrows():
        for tN, factorIdx in row['designInfo'].term_name_slices.items():
            thisSrs = pd.Series({fN: tN for fN in row['designInfo'].column_names[factorIdx]})
            thisSrs.name = 'term'
            thisSrs.index.name = 'factor'
            endoFactors.append(thisSrs)
    factorPaletteDict['endo'] = pd.concat(endoFactors).to_frame(name='term').reset_index().drop_duplicates(subset='factor')
    exoFactors = []
    for (rhsMaskIdx, ensTemplate), row in histDesignInfoDF.iterrows():
        for tN, factorIdx in row['designInfo'].term_name_slices.items():
            thisSrs = pd.Series({fN: tN for fN in row['designInfo'].column_names[factorIdx]})
            thisSrs.name = 'term'
            thisSrs.index.name = 'factor'
            exoFactors.append(thisSrs)
    factorPaletteDict['exo'] = pd.concat(exoFactors).to_frame(name='term').reset_index().drop_duplicates(subset='factor')
    factorPalette = pd.concat(factorPaletteDict, names=['type', 'index'])
    ############# workaround inconsistent use of whitespace with patsy
    sourceTermLookup.reset_index(inplace=True)
    sourceTermLookup.loc[:, 'term'] = sourceTermLookup['term'].apply(lambda x: x.replace(' ', ''))
    sourceTermLookup.set_index(['type', 'term'], inplace=True)
    ######
    termPalette.loc[:, 'termNoWS'] = termPalette['term'].apply(lambda x: x.replace(' ', ''))
    termPalette.loc[:, 'source'] = termPalette['termNoWS'].map(sourceTermLookup.reset_index(level='type')['source'])
    termPalette = termPalette.sort_values('source', kind='mergesort').sort_index(kind='mergesort')
    #
    factorPalette.loc[:, 'termNoWS'] = factorPalette['term'].apply(lambda x: x.replace(' ', ''))
    factorPalette.loc[:, 'source'] = factorPalette['termNoWS'].map(sourceTermLookup.reset_index(level='type')['source'])
    factorPalette = factorPalette.sort_values('source', kind='mergesort').sort_index(kind='mergesort')
    ############
    termPalette.loc[:, 'color'] = termPalette['source'].map(sourcePalette)
    factorPalette.loc[:, 'color'] = factorPalette['source'].map(sourcePalette)
    #
    trialTypeOrder = ['train', 'work', 'test', 'validation']
    trialTypePalette = pd.Series(
        sns.color_palette('Paired', 12)[::-1][:len(trialTypeOrder)],
        index=trialTypeOrder)
    #
    sourceTermLookup.to_hdf(estimatorPath, 'sourceTermLookup')
    sourcePalette.to_hdf(estimatorPath, 'sourcePalette')
    termPalette.to_hdf(estimatorPath, 'termPalette')
    factorPalette.to_hdf(estimatorPath, 'factorPalette')
    trialTypePalette.to_hdf(estimatorPath, 'trialTypePalette')
##
if (not loadedIR) or arguments['forceReprocess']:
    iRPerFactorDict0 = {}
    iRPerTermDict0 = {}
    for lhsMaskIdx in range(lhsMasks.shape[0]):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        designFormula = lhsMaskParams['designFormula']
        if designFormula != 'NULL':
            designInfo = designInfoDict[designFormula]
            designDF = impulseDict[designFormula]
            designTermNames = designInfo.term_names
        else:
            designInfo = None
            designDF = None
            designTermNames = []
        theseEstimators = estimatorsDF.xs(lhsMaskIdx, level='lhsMaskIdx')
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        #
        iRPerFactorDict1 = {}
        iRPerTermDict1 = {}
        for (rhsMaskIdx, targetName, fold), estimatorSrs in theseEstimators.groupby(['rhsMaskIdx', 'target', 'fold']):
            estimator = estimatorSrs.iloc[0]
            coefs = coefDF.loc[idxSl[lhsMaskIdx, designFormula, rhsMaskIdx, targetName, fold, :]]
            coefs.index = coefs.index.get_level_values('factor')
            allIRList = []
            allIRPerSourceList = []
            histDesignList = []
            if ensTemplate != 'NULL':
                ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
                ensTermNames = [
                    key
                    for key in ensDesignInfo.term_names
                    if key != ensTemplate.format(targetName)]
                ensFactorNames = np.concatenate([
                    np.atleast_1d(ensDesignInfo.column_names[sl])
                    for key, sl in ensDesignInfo.term_name_slices.items()
                    if key != ensTemplate.format(targetName)])
                thisEnsDesignDF = histImpulseDict[(rhsMaskIdx, ensTemplate)].loc[:, ensFactorNames]
                histDesignList.append(thisEnsDesignDF)
                # columnsInDesign = [cN for cN in coefs.index if cN in ensFactorNames]
                # ensIR = thisEnsDesignDF * coefs.loc[columnsInDesign]
                # for cN in ensFactorNames:
                #     outputIR.loc[:, cN] = np.nan
            else:
                ensTermNames = []
            #
            if selfTemplate != 'NULL':
                selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
                selfTermNames = [
                    key
                    for key in selfDesignInfo.term_names
                    if key == selfTemplate.format(targetName)]
                selfFactorNames = np.concatenate([
                    np.atleast_1d(selfDesignInfo.column_names[sl])
                    for key, sl in selfDesignInfo.term_name_slices.items()
                    if key == selfTemplate.format(targetName)])
                thisSelfDesignDF = histImpulseDict[(rhsMaskIdx, selfTemplate)].loc[:, selfFactorNames]
                histDesignList.append(thisSelfDesignDF)
                # columnsInDesign = [cN for cN in coefs.index if cN in selfFactorNames]
                # selfIR = thisSelfDesignDF * coefs.loc[columnsInDesign]
                # for cN in selfFactorNames:
                #     outputIR.loc[:, cN] = np.nan
            else:
                selfTermNames = []
            if len(histDesignList):
                histDesignDF = pd.concat(histDesignList, axis='columns')
                columnsInDesign = [cN for cN in coefs.index if cN in histDesignDF.columns]
                assert len(columnsInDesign) == histDesignDF.columns.shape[0]
                endogIR = histDesignDF.loc[:, columnsInDesign] * coefs.loc[columnsInDesign]
                iRLookup = pd.Series(
                    endogIR.columns.map(factorPalette.loc[:, ['factor', 'term']].set_index('factor')['term']),
                    index=endogIR.columns)
                histIpsList = []
                for endoTermName in termPalette['term']:
                    nameMask = (iRLookup == endoTermName)
                    if nameMask.any():
                        histIpsList.append(endogIR.loc[:, nameMask].sum(axis='columns').to_frame(name=endoTermName))
                endogIRPerSource = pd.concat(histIpsList, axis='columns')
                allIRList.append(endogIR)
                allIRPerSourceList.append(endogIRPerSource)
            else:
                endogIR = None
            #####
            if designFormula != 'NULL':
                columnsInDesign = [cN for cN in coefs.index if cN in designDF.columns]
                assert len(columnsInDesign) == designDF.columns.shape[0]
                # columnsNotInDesign = [cN for cN in coefs.index if cN not in designDF.columns]
                exogIR = designDF.loc[:, columnsInDesign] * coefs.loc[columnsInDesign]
                #####
                extDesignTermNames = []
                ipsList = []
                iRList = []
                for termIdx, (term, subTermInfoList) in enumerate(designInfo.term_codings.items()):
                    # print(term)
                    termName = designTermNames[termIdx]
                    termSlice = designInfo.term_slices[term]
                    offset = 0
                    for subTermInfo in subTermInfoList:
                        # print('\t{}'.format(subTermInfo))
                        if len(subTermInfo.contrast_matrices):
                            extTermNameSuffix = ':'.join([fac.name() for fac in subTermInfo.factors if fac not in subTermInfo.contrast_matrices])
                            for factor, contrastMat in subTermInfo.contrast_matrices.items():
                                # print('\t\t{}'.format(factor))
                                # print('\t\t{}'.format(contrastMat))
                                # fig, ax = plt.subplots(len(contrastMat.column_suffixes))
                                for categIdx, categName in enumerate(contrastMat.column_suffixes):
                                    idxMask = np.asarray([(elecName in categName) for elecName in exogIR.index.get_level_values('electrode')])
                                    colMask = np.asarray([(categName in factorName) for factorName in exogIR.iloc[:, termSlice].columns])
                                    theseIR = exogIR.iloc[:, termSlice].loc[idxMask, colMask].copy()
                                    # sns.heatmap(theseIR.reset_index(drop=True), ax=ax[categIdx])
                                    iRList.append(theseIR.reset_index(drop=True))
                                    extTermName = '{}{}'.format(factor.name(), categName) + ':' + extTermNameSuffix
                                    extDesignTermNames.append(extTermName)
                                    thisIRPerSource = theseIR.reset_index(drop=True).sum(axis='columns').to_frame(name=extTermName)
                                    ipsList.append(thisIRPerSource)
                                    # update sourceTermLookup
                                    if not (extTermName.replace(' ', '') in sourceTermLookup.index.get_level_values('term')):
                                        stlEntry = sourceTermLookup.xs(termName.replace(' ', ''), level='term', drop_level=False).reset_index()
                                        stlEntry.loc[:, 'term'] = extTermName.replace(' ', '')
                                        stlEntry.loc[:, 'source'] = stlEntry.loc[:, 'source'].apply(lambda x: '{}{}'.format(categName, x))
                                        stlEntry.set_index(['type', 'term'], inplace=True)
                                        sourceTermLookup = sourceTermLookup.append(stlEntry)
                                    # update termPalette
                                    if not (extTermName in termPalette['term'].to_numpy()):
                                        termPaletteEntry = termPalette.loc[termPalette['term'] == termName, :].reset_index()
                                        termPaletteEntry.loc[:, 'index'] = termPalette.xs('exog', level='type').index.get_level_values('index').max() + 1
                                        termPaletteEntry.loc[:, 'term'] = extTermName
                                        termPaletteEntry.loc[:, 'source'] = termPaletteEntry.loc[:, 'source'].apply(lambda x: '{}{}'.format(categName, x))
                                        termPaletteEntry.loc[:, 'termNoWS'] = extTermName.replace(' ', '')
                                        termPaletteEntry.set_index(['type', 'index'], inplace=True)
                                        termPalette = termPalette.append(termPaletteEntry)
                                        # update factorPalette
                                        factorPaletteMask = factorPalette['factor'].isin(theseIR.columns)
                                        factorPalette.loc[factorPaletteMask, 'term'] = extTermName
                                        factorPalette.loc[factorPaletteMask, 'termNoWS'] = extTermName.replace(' ', '')
                        else:
                            # no categoricals
                            idxMask = np.asarray(exogIR.index.get_level_values('trialUID') == 0)
                            theseIR = exogIR.iloc[idxMask, termSlice].copy()
                            iRList.append(theseIR.reset_index(drop=True))
                            extDesignTermNames.append(termName)
                            thisIRPerSource = theseIR.reset_index(drop=True).sum(axis='columns').to_frame(name=termName)
                            ipsList.append(thisIRPerSource)
                ####
                designTermNames = extDesignTermNames
                if endogIR is not None:
                    saveIndex = endogIR.index
                else:
                    saveIndex = exogIR.loc[exogIR.index.get_level_values('trialUID') == 0, :].index
                exogIR = pd.concat(iRList, axis='columns')
                exogIR.index = saveIndex
                exogIRPerSource = pd.concat(ipsList, axis='columns')
                exogIRPerSource.index = saveIndex
                allIRList.append(exogIR)
                allIRPerSourceList.append(exogIRPerSource)
            else:
                exogIR = None
                exogIRPerSource = None
            outputIR = pd.concat(allIRList, axis='columns')
            outputIRPerSource = pd.concat(allIRPerSourceList, axis='columns')
            termNames = designTermNames + ensTermNames + selfTermNames
            ###########################
            sanityCheckIRs = False
            # check that the impulse responses are equivalent to the sum of the weighted basis functions
            if sanityCheckIRs:
                plotIR = outputIR.copy()
                plotIR.index = plotIR.index.droplevel([idxName for idxName in plotIR.index.names if idxName not in ['trialUID', 'bin']])
                fig, ax = plt.subplots()
                sns.heatmap(plotIR, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
                for termName, termSlice in designInfo.term_name_slices.items():
                    histOpts = sourceHistOptsDict[termName.replace(' ', '')]
                    factorNames = designInfo.column_names[termSlice]
                    if 'rcb(' in termName:
                        basisApplier = tdr.raisedCosTransformer(histOpts)
                        fig, ax = basisApplier.plot_basis()
                        if histOpts['useOrtho']:
                            basisDF = basisApplier.orthobasisDF
                        else:
                            basisDF = basisApplier.ihbasisDF
                        # hack to multiply by number of electrodes
                        assert (len(factorNames) % basisDF.shape[1]) == 0
                        nReps = int(len(factorNames) / basisDF.shape[1])
                        for trialUID in range(nReps):
                            basisDF.columns = factorNames[trialUID::nReps]
                            irDF = plotIR.xs(trialUID, level='trialUID')
                            fig, ax = plt.subplots(2, 1, sharex=True)
                            for cN in basisDF.columns:
                                ax[0].plot(basisDF.index, basisDF[cN], label='basis {}'.format(cN))
                                ax[1].plot(basisDF.index, basisDF[cN] * coefs[cN], label='basis {} * coef'.format(cN))
                                ax[1].plot(irDF.index.get_level_values('bin'), irDF[cN], '--', label='IR {}'.format(cN))
                            ax[0].legend()
                            ax[1].legend()
            ###########################
            prf.print_memory_usage('Calculated IR for {}, {}\n'.format((lhsMaskIdx, designFormula), (rhsMaskIdx, targetName, fold)))
            iRPerFactorDict1[(rhsMaskIdx, targetName, fold)] = outputIR
            iRPerTermDict1[(rhsMaskIdx, targetName, fold)] = outputIRPerSource
        iRPerFactorDict0[(lhsMaskIdx, designFormula)] = pd.concat(iRPerFactorDict1, names=['rhsMaskIdx', 'target', 'fold'])
        iRPerTermDict0[(lhsMaskIdx, designFormula)] = pd.concat(iRPerTermDict1, names=['rhsMaskIdx', 'target', 'fold'])
    #
    iRPerFactor = pd.concat(iRPerFactorDict0, names=['lhsMaskIdx', 'design'])
    iRPerFactor.columns.name = 'factor'
    iRPerTerm = pd.concat(iRPerTermDict0, names=['lhsMaskIdx', 'design'])
    iRPerTerm.columns.name = 'term'
    #
    trialInfoIR = iRPerTerm.index.to_frame().reset_index(drop=True)
    stimConditionIR = pd.Series(np.nan, index=trialInfoIR.index)
    stimOrderIR = []
    for name, group in trialInfoIR.groupby(['electrode', 'trialRateInHz']):
        stimConditionIR.loc[group.index] = '{} {}'.format(*name)
        stimOrderIR.append('{} {}'.format(*name))
    trialInfoIR.loc[:, 'stimCondition'] = stimConditionIR
    stimConditionLookupIR = (
        trialInfoIR
            .loc[:, ['electrode', 'trialRateInHz', 'stimCondition']]
            .drop_duplicates()
            .set_index(['electrode', 'trialRateInHz'])['stimCondition'])
    kinConditionIR = pd.Series(np.nan, index=trialInfoIR.index)
    kinOrderIR = []
    for name, group in trialInfoIR.groupby(['pedalMovementCat', 'pedalDirection']):
        kinConditionIR.loc[group.index] = '{} {}'.format(*name)
        kinOrderIR.append('{} {}'.format(*name))
    trialInfoIR.loc[:, 'kinCondition'] = kinConditionIR
    kinConditionLookupIR = (
        trialInfoIR
            .loc[:, ['pedalMovementCat', 'pedalDirection', 'kinCondition']]
            .drop_duplicates()
            .set_index(['pedalMovementCat', 'pedalDirection'])['kinCondition'])
    iRPerTerm.index = pd.MultiIndex.from_frame(trialInfoIR)
    #
    iRPerTerm.to_hdf(estimatorPath, 'impulseResponsePerTerm')
    iRPerFactor.to_hdf(estimatorPath, 'impulseResponsePerFactor')
    stimConditionLookupIR.to_hdf(estimatorPath, 'impulseResponseStimConditionLookup')
    kinConditionLookupIR.to_hdf(estimatorPath, 'impulseResponseKinConditionLookup')
    #
    termPalette.sort_index(inplace=True)
    termPalette.to_hdf(estimatorPath, 'termPalette')
    sourceTermLookup.sort_index(inplace=True)
    sourceTermLookup.to_hdf(estimatorPath, 'sourceTermLookup')
    factorPalette.to_hdf(estimatorPath, 'factorPalette')
#
iRGroupNames = ['lhsMaskIdx', 'design', 'rhsMaskIdx', 'fold']
eps = np.spacing(1)
eigenValueDict = {}
ADict = {}
BDict = {}
KDict = {}
CDict = {}
DDict = {}
if arguments['plotting']:
    pdfPath = os.path.join(
        figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'OKID_ERA'))
    cm = PdfPages(pdfPath)
else:
    import contextlib
    cm = contextlib.nullcontext()
with cm as pdf:
    for name, iRGroup0 in iRPerTerm.groupby(iRGroupNames):
        iRWorkingCopy = iRGroup0.copy()
        iRWorkingCopy.dropna(inplace=True, axis='columns')
        ##
        lhsMaskIdx, designFormula, rhsMaskIdx, fold = name
        if not designIsLinear[designFormula]:
            continue
        if not (lhsMaskIdx in [0, 9, 10, 28, 29, 47, 48]):
            continue
        print('Calculating state space coefficients for {}'.format(lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr']))
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        #
        exogNames0 = iRWorkingCopy.columns.map(termPalette.xs('exog', level='type').loc[:, ['source', 'term']].set_index('term')['source']).to_series()
        exogNames = exogNames0.dropna().to_list()
        if designFormula != 'NULL':
            # designInfo = designInfoDict[designFormula]
            # exogNames = designInfo.term_names
            histLens = [designHistOptsDict[designFormula]['historyLen']]
        else:
            histLens = []
        ensTemplate = lhsMaskParams['ensembleTemplate']
        selfTemplate = lhsMaskParams['selfTemplate']
        if ensTemplate != 'NULL':
            # ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
            histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
        if selfTemplate != 'NULL':
            # selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
            histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
        iRWorkingCopy.rename(columns=termPalette.loc[:, ['source', 'term']].set_index('term')['source'], inplace=True)
        # if either ensemble or self are NULL, fill with zeros:
        endogNames = iRWorkingCopy.index.get_level_values('target').unique().to_list()
        for sN in endogNames:
            if sN not in iRWorkingCopy.columns:
                iRWorkingCopy.loc[:, sN] = 0.
        trialInfo = iRWorkingCopy.index.to_frame().reset_index(drop=True)
        ##
        #######################
        # Apply ERA
        Phi = pd.concat([iRWorkingCopy.reset_index(drop=True), trialInfo.loc[:, ['bin', 'target']]], axis='columns')
        Phi.columns.name = 'feature'
        kernelMask = (Phi['bin'] >= 0) & (Phi['bin'] <= max(histLens))
        Phi = Phi.loc[kernelMask, ['bin', 'target'] + endogNames + exogNames]
        Phi.loc[Phi['bin'] == 0, endogNames] = 0. # enforce that y[t] = F(y[t-p]), p positive
        Phi.sort_values(by=['bin', 'target'], kind='mergesort', inplace=True)
        Phi.set_index(['bin', 'target'], inplace=True)
        Phi = Phi.stack().unstack('target').T
        #
        nLags = int(10 * max(histLens) / binInterval)
        #
        A, B, K, C, D, (fig, ax,) = tdr.ERA(
            Phi, maxNDim=None,
            endogNames=endogNames, exogNames=exogNames, nLags=nLags,
            plotting=arguments['plotting'])
        ADict[name] = A
        BDict[name] = B
        KDict[name] = K
        CDict[name] = C
        DDict[name] = D
        #
        # sS = ctrl.ss(A, B, C, D, dt=binInterval)
        theseEigValsList = []
        for nd in range(A.shape[0], 1, -10):
            w, v = np.linalg.eig(A.iloc[:nd, :nd])
            w = pd.Series(w)
            logW = np.log(w)
            theseEigVals = pd.concat({
                'complex': w,
                'logComplex': logW,
                'magnitude': w.abs(),
                'phase': w.apply(np.angle),
                'real': w.apply(np.real),
                'r': logW.apply(np.real).apply(np.exp),
                'delta': logW.apply(np.imag),
                'imag': w.apply(np.imag)}, axis='columns')
            theseEigVals.loc[:, 'stateNDim'] = nd
            theseEigVals.loc[:, 'nDimIsMax'] = (nd == A.shape[0])
            theseEigVals.index.name = 'eigenvalueIndex'
            theseEigVals.set_index('stateNDim', append=True, inplace=True)
            #
            theseEigVals.loc[:, 'tau'] = 2 * np.pi * binInterval / theseEigVals['delta']
            theseEigVals.loc[:, 'chi'] = - (theseEigVals['r'].apply(np.log) ** (-1)) * binInterval
            theseEigVals.loc[:, 'isOscillatory'] = (theseEigVals['imag'].abs() > 1e-6) | (theseEigVals['real'] < 0)
            theseEigVals.loc[:, 'isStable'] = theseEigVals['magnitude'] <= 1
            #
            theseEigVals.loc[:, 'eigValType'] = ''
            theseEigVals.loc[theseEigVals['isOscillatory'] & theseEigVals['isStable'], 'eigValType'] = 'oscillatory decay'
            theseEigVals.loc[~theseEigVals['isOscillatory'] & theseEigVals['isStable'], 'eigValType'] = 'pure decay'
            theseEigVals.loc[theseEigVals['isOscillatory'] & ~theseEigVals['isStable'], 'eigValType'] = 'oscillatory growth'
            theseEigVals.loc[~theseEigVals['isOscillatory'] & ~theseEigVals['isStable'], 'eigValType'] = 'pure growth'
            theseEigValsList.append(theseEigVals)
        if len(theseEigValsList):
            eigenValueDict[name] = pd.concat(theseEigValsList)
        if arguments['plotting']:
            fig.suptitle('{}'.format(name))
            fig.tight_layout()
            pdf.savefig(
                bbox_inches='tight',
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
#
eigDF = pd.concat(eigenValueDict, names=iRGroupNames)
eigDF.to_hdf(estimatorPath, 'eigenvalues')
allA = pd.concat(ADict, names=iRGroupNames)
allA.to_hdf(estimatorPath, 'A')
allB = pd.concat(BDict, names=iRGroupNames)
allB.to_hdf(estimatorPath, 'B')
allK = pd.concat(KDict, names=iRGroupNames)
allK.to_hdf(estimatorPath, 'K')
allC = pd.concat(CDict, names=iRGroupNames)
allC.to_hdf(estimatorPath, 'C')
allD = pd.concat(DDict, names=iRGroupNames)
allD.to_hdf(estimatorPath, 'D')

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
##############################################################################################################

pdfPath = os.path.join(
    figureOutputFolder, '{}_{}.pdf'.format(fullEstimatorName, 'impulse_responses'))
with PdfPages(pdfPath) as pdf:
    height, width = 2, 4
    aspect = width / height
    for (lhsMaskIdx, designFormula, rhsMaskIdx), thisIRPerTerm in iRPerTerm.groupby(
            ['lhsMaskIdx', 'design', 'rhsMaskIdx']):
        lhsMask = lhsMasks.iloc[lhsMaskIdx, :]
        lhsMaskParams = {k: v for k, v in zip(lhsMasks.index.names, lhsMask.name)}
        thisTitleFormula = lhsMasksInfo.loc[lhsMaskIdx, 'fullFormulaDescr']
        print('Saving impulse response plots for {}'.format(thisTitleFormula))
        if designFormula != 'NULL':
            designInfo = designInfoDict[designFormula]
            termNames = designInfo.term_names
            histLens = [sourceHistOptsDict[tN.replace(' ', '')]['historyLen'] for tN in termNames]
        else:
            histLens = []
        ensTemplate = lhsMaskParams['ensembleTemplate']
        if ensTemplate != 'NULL':
            ensDesignInfo = histDesignInfoDict[(rhsMaskIdx, ensTemplate)]
            histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
        selfTemplate = lhsMaskParams['selfTemplate']
        if selfTemplate != 'NULL':
            selfDesignInfo = histDesignInfoDict[(rhsMaskIdx, selfTemplate)]
            histLens.append(templateHistOptsDict[ensTemplate]['historyLen'])
        tBins = thisIRPerTerm.index.get_level_values('bin')
        kernelMask = (tBins >= 0) & (tBins <= max(histLens))
        plotDF = thisIRPerTerm.loc[kernelMask, :].stack().to_frame(name='signal').reset_index()
        kinOrder = kinConditionLookupIR.loc[kinConditionLookupIR.isin(plotDF['kinCondition'])].to_list()
        stimOrder = stimConditionLookupIR.loc[stimConditionLookupIR.isin(plotDF['stimCondition'])].to_list()
        thisTermPalette = termPalette.loc[termPalette['term'].isin(plotDF['term']), :]
        g = sns.relplot(
            # row='kinCondition', row_order=kinOrder,
            # col='stimCondition', col_order=stimOrder,
            row='target',
            x='bin', y='signal', hue='term',
            hue_order=thisTermPalette['term'].to_list(),
            palette=thisTermPalette.loc[:, ['term', 'color']].set_index('term')['color'].to_dict(),
            kind='line', errorbar='se', data=plotDF,
        )
        g.set_axis_labels("Lag (sec)", 'contribution to target')
        g.suptitle('Impulse responses (per term) for model {}'.format(thisTitleFormula))
        asp.reformatFacetGridLegend(
            g, titleOverrides={},
            contentOverrides=termPalette.loc[:, ['term', 'source']].set_index('term')['source'].to_dict(),
            styleOpts=styleOpts)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
#
##############################################################################################################
##############################################################################################################
if False:
    # sanity check that signal dynamics are independent of regressors
    meanA = allA.groupby(['rhsMaskIdx', 'state']).mean()
    stdA = allA.groupby(['rhsMaskIdx', 'state']).std()
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(meanA.reset_index(drop=True), ax=ax[0])
    sns.heatmap(stdA.reset_index(drop=True), ax=ax[1])
    #

    plotEig = pd.concat({
        'magnitude': eigDF.apply(lambda x: np.absolute(x)),
        'phase': eigDF.apply(lambda x: np.angle(x)),
        'real': eigDF.apply(lambda x: x.real),
        'imag': eigDF.apply(lambda x: x.imag)}, axis='columns')
    plotEig.reset_index(inplace=True)
    g = sns.relplot(
        row='lhsMaskIdx', col='rhsMaskIdx',
        x='real', y='imag', hue='electrode', style='electrode',
        kind='scatter', data=plotEig)