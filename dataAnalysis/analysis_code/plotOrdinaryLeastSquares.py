"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                                which experimental day to analyze
    --blockIdx=blockIdx                      which trial to analyze [default: 1]
    --processAll                             process entire experimental day? [default: False]
    --plotting                               make plots? [default: False]
    --showFigures                            show plots? [default: False]
    --verbose                                print diagnostics? [default: False]
    --fullEstimatorName=fullEstimatorName    filename for resulting estimator (cross-validated n_comps)
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
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=.5, color_codes=True)
idxSl = pd.IndexSlice

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
# pdb.set_trace()
# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
    'analysisName': 'default', 'showFigures': True, 'exp': 'exp202101201100', 'processAll': True,
    'verbose': False, 'plotting': True, 'fullEstimatorName': 'enr_lfp_CAR_spectral_to_limbState_a_L_starting',
    'alignFolderName': 'motion', 'blockIdx': '2'}
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
        figureFolder,
        arguments['analysisName'], arguments['alignFolderName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder)
#
'''fullEstimatorName = '{}_{}_to_{}{}_{}_{}'.format(
    arguments['estimatorName'],
    arguments['unitQueryLhs'], arguments['unitQueryRhs'],
    iteratorSuffix,
    arguments['window'],
    arguments['alignQuery'])'''
fullEstimatorName = arguments['fullEstimatorName']
#
estimatorsSubFolder = os.path.join(
    alignSubFolder, 'estimators')
if not os.path.exists(estimatorsSubFolder):
    os.makedirs(estimatorsSubFolder)
estimatorPath = os.path.join(
    estimatorsSubFolder,
    fullEstimatorName + '.h5'
    )
scoresDF = pd.read_hdf(estimatorPath, 'cv')
predDF = pd.read_hdf(estimatorPath, 'predictions')
with open(estimatorPath.replace('.h5', '_meta.pickle'), 'rb') as _f:
    loadingMeta = pickle.load(_f)
arguments.update(loadingMeta['arguments'])
#
cvIteratorSubfolder = os.path.join(
    alignSubFolder, 'testTrainSplits')
if arguments['iteratorSuffix'] is not None:
    iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
else:
    iteratorSuffix = ''
#
if arguments['processAll']:
    rhsBlockBaseName = arguments['rhsBlockPrefix']
    lhsBlockBaseName = arguments['lhsBlockPrefix']
else:
    rhsBlockBaseName = '{}{:0>3}'.format(
        arguments['rhsBlockPrefix'], arguments['blockIdx'])
    lhsBlockBaseName = '{}{:0>3}'.format(
        arguments['lhsBlockPrefix'], arguments['blockIdx'])
# rhs loading paths
if arguments['rhsBlockSuffix'] is not None:
    rhsBlockSuffix = '_{}'.format(arguments['rhsBlockSuffix'])
else:
    rhsBlockSuffix = ''
if arguments['lhsBlockSuffix'] is not None:
    lhsBlockSuffix = '_{}'.format(arguments['lhsBlockSuffix'])
else:
    lhsBlockSuffix = ''
#
iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
cv_kwargs = loadingMeta['cv_kwargs'].copy()
# lhGroupNames = loadingMeta['lhGroupNames']
lOfRhsDF = []
lOfLhsDF = []
lOfLhsMasks = []
experimentsToAssemble = loadingMeta['experimentsToAssemble'].copy()
currBlockNum = 0
for expName, lOfBlocks in experimentsToAssemble.items():
    thisScratchFolder = os.path.join(scratchPath, expName)
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, thisScratchFolder)
    thisDFFolder = os.path.join(alignSubFolder, 'dataframes')
    for bIdx in lOfBlocks:
        theseArgs = arguments.copy()
        theseArgs['blockIdx'] = '{}'.format(bIdx)
        theseArgs['processAll'] = False
        theseArgs['inputBlockSuffix'] = theseArgs['rhsBlockSuffix']
        theseArgs['inputBlockPrefix'] = theseArgs['rhsBlockPrefix']
        thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
        dFPath = os.path.join(
            thisDFFolder,
            '{}_{}_{}_df{}.h5'.format(
                thisBlockBaseName,
                arguments['window'],
                arguments['alignQuery'],
                iteratorSuffix))
        thisRhsDF = pd.read_hdf(dFPath, arguments['unitQueryRhs'])
        thisRhsDF.index = thisRhsDF.index.set_levels([currBlockNum], level='segment')
        lOfRhsDF.append(thisRhsDF)
        thisLhsDF = pd.read_hdf(dFPath, arguments['unitQueryLhs'])
        thisLhsDF.index = thisLhsDF.index.set_levels([currBlockNum], level='segment')
        lOfLhsDF.append(thisLhsDF)
        thisLhsMask = pd.read_hdf(dFPath, arguments['unitQueryLhs'] + '_featureMasks')
        lOfLhsMasks.append(thisLhsMask)
        currBlockNum += 1
lhsDF = pd.concat(lOfLhsDF)
rhsDF = pd.concat(lOfRhsDF)
del lOfRhsDF, lOfLhsDF, thisRhsDF, thisLhsDF

cvIterator = iteratorsBySegment[0]
workIdx = cvIterator.work

lhsMasks = lOfLhsMasks[0]
lhGroupNames = lhsMasks.index.names
trialInfo = rhsDF.index.to_frame().reset_index(drop=True)
attrNameList = lhsMasks.index.names
for attrIdx, attrName in enumerate(attrNameList):
    attrKey = lhGroupNames[attrIdx]
    trialInfo.loc[:, attrKey] = attrName
rhsDF.index = pd.MultiIndex.from_frame(trialInfo)
workingLhsDF = lhsDF.iloc[workIdx, :]
workingRhsDF = rhsDF.iloc[workIdx, :]
# pdb.set_trace()
plotData = pd.concat({
    'ground_truth': workingRhsDF,
    'prediction': predDF}, names=['data_origin'])
plotData.columns = plotData.columns.get_level_values('feature')
predStack = plotData.stack(plotData.columns.names).to_frame(name='signal').reset_index()
#
figureOutputFolder = os.path.join(
    figureFolder,
    arguments['analysisName'], arguments['alignFolderName'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder)

pdfPath = os.path.join(figureOutputFolder, '{}_fitted_signals.pdf'.format(fullEstimatorName))
plotProcFuns = []
with PdfPages(pdfPath) as pdf:
    for name, group in predStack.groupby('feature'):
        print('making {}'.format(plot))
        g = sns.relplot(
            row='pedalMovementCat', hue='freqBandName', style='data_origin',
            x='bin', y='signal', data=group, kind='line', ci='sem')
        g.fig.set_size_inches((12, 8))
        for (ro, co, hu), dataSubset in g.facet_data():
            emptySubset = (
                    (dataSubset.empty) or
                    (dataSubset['signal'].isna().all()))
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        leg = g._legend
        titleOverrides = {}
        entryOverrides = {}
        if leg is not None:
            for t in leg.texts:
                tContent = t.get_text()
                if tContent in titleOverrides:
                    t.set_text(titleOverrides[tContent])
                # elif tContent.replace('.', '', 1).isdigit():
                # e.g. is numeric
                else:
                    try:
                        tNumeric = float(tContent)
                        t.set_text('{:.2f}'.format(tNumeric))
                        '''if tNumeric in entryOverrides.index:
                            t.set_text('{} {}'.format(
                                entryOverrides.loc[float(tContent), 'whichArray'],
                                entryOverrides.loc[float(tContent), 'electrodeSide']))'''
                    except Exception:
                        pass
        g.set_titles('{row_name}')
        '''g.set_xlabels('')
        g.set_ylabels('')
        g.set_yticklabels('')'''
        g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        # figYLabel = g.fig.supylabel('LFP RAUC (a.u.)', x=0.1)
        # figXLabel = g.fig.supxlabel('Stimulation Amplitude (uA)', y=0.1)
        figTitle = g.fig.suptitle('{}'.format(name))
        pdf.savefig(
            bbox_inches='tight', pad_inches=0.2,
            bbox_extra_artists=[leg]
            )
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
