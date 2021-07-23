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
    --winStart=winStart                    start of window
    --winStop=winStop                      end of window
    --lazy                                 load from raw, or regular? [default: False]
    --preScale                             apply normalization per feacture before anything else? [default: False]
    --plotting                             make plots? [default: False]
    --debugging                            restrict datasets for debugging? [default: False]
    --showFigures                          show plots? [default: False]
    --verbose=verbose                      print diagnostics? [default: 0]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --selectionName=selectionName          how to restrict channels? [default: fr_sqrt]
    --iteratorSuffix=iteratorSuffix        filename for cross_val iterator
    --selector=selector                    filename if using a unit selector
    --loadFromFrames                       load data from pre-saved dataframes?
"""

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
if arguments['plotting']:
    import matplotlib, os
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if 'CCV_HEADLESS' in os.environ:
        matplotlib.use('PS')   # generate postscript output
    else:
        matplotlib.use('QT5Agg')   # generate interactive output
#
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set(
        context='talk', style='dark',
        palette='dark', font='sans-serif',
        font_scale=1.5, color_codes=True)
from dask.distributed import Client
import os, traceback
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
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
idxSl = pd.IndexSlice

# if debugging in a console:
'''
consoleDebugging = True
if consoleDebugging:
    arguments = {
        'iteratorSuffix': 'a', 'alignFolderName': 'motion',
        'processAll': True, 'exp': 'exp202101201100', 'analysisName': 'default',
        'blockIdx': '2', 'rhsBlockPrefix': 'Block', 'verbose': False,
        'lhsBlockSuffix': 'lfp_CAR_spectral', 'unitQueryLhs': 'lfp_CAR_spectral',
        'rhsBlockSuffix': 'rig', 'unitQueryRhs': 'jointAngle',
        'loadFromFrames': True, 'estimatorName': 'ols_lfp_CAR_ja',
        'alignQuery': 'starting', 'winStop': '400', 'window': 'L', 'selector': None, 'winStart': '200',
        'plotting': True, 'lazy': False, 'lhsBlockPrefix': 'Block',
        'showFigures': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


if __name__ == '__main__':
    analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
        arguments, scratchFolder)
    dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
    if not os.path.exists(dataFramesFolder):
        os.makedirs(dataFramesFolder)
    blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)

    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder,
            arguments['analysisName'], arguments['alignFolderName'])
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
    #
    cvIteratorSubfolder = os.path.join(
        alignSubFolder, 'testTrainSplits')
    if arguments['iteratorSuffix'] is not None:
        iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
    else:
        iteratorSuffix = ''
    iteratorPath = os.path.join(
        cvIteratorSubfolder,
        '{}_{}_{}{}_cvIterators.pickle'.format(
            blockBaseName,
            arguments['window'],
            arguments['alignQuery'],
            iteratorSuffix))
    print('Loading cv iterator from {}'.format(iteratorPath))
    with open(iteratorPath, 'rb') as f:
        loadingMeta = pickle.load(f)
    datasetName = '{}_{}_df{}'.format(
        blockBaseName,
        arguments['window'],
        iteratorSuffix)
    # loading paths
    triggeredPath = os.path.join(
        alignSubFolder,
        blockBaseName + '{}_{}.nix'.format(
            inputBlockSuffix, arguments['window']))
    iteratorsBySegment = loadingMeta['iteratorsBySegment'].copy()
    # cv_kw##args = loadingMeta['cv_kw##args'].copy()
    cvIterator = iteratorsBySegment[0]
    workIdx = cvIterator.work
    ######### data loading stuff
    lOfDF = []
    lOfFeatureMasks = []
    if not arguments['loadFromFrames']:
        alignedAsigsKWargs = loadingMeta['alignedAsigsKWargs'].copy()
        alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
            namedQueries, scratchFolder, **arguments)
        if arguments['verbose']:
            prf.print_memory_usage('before load data')
            print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        nSeg = len(dataBlock.segments)
        loadingMeta['alignedAsigsKWargs'] = alignedAsigsKWargs
        for segIdx in range(nSeg):
            if arguments['verbose']:
                prf.print_memory_usage('fitting on segment {}'.format(segIdx))
            if 'listOfROIMasks' in loadingMeta:
                alignedAsigsKWargs.update({'finalIndexMask': loadingMeta['listOfROIMasks'][segIdx]})
                alignedAsigsKWargs.update({'finalIndexMask': loadingMeta['listOfROIMasks'][segIdx]})
            try:
                thisDF = ns5.alignedAsigsToDF(
                    dataBlock,
                    whichSegments=[segIdx],
                    **alignedAsigsKWargs)
            except Exception:
                traceback.print_exc()
                continue
            if arguments['verbose']:
                prf.print_memory_usage('loaded LHS')
            lOfDF.append(thisDF)
        if arguments['lazy']:
            dataReader.file.close()
    else:    # loading frames
        loadingMeta['arguments']['preScale'] = arguments['preScale']
        experimentsToAssemble = loadingMeta['experimentsToAssemble'].copy()
        print(experimentsToAssemble)
        currBlockNum = 0
        for expName, lOfBlocks in experimentsToAssemble.items():
            thisScratchFolder = os.path.join(scratchPath, expName)
            analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
                loadingMeta['arguments'], thisScratchFolder)
            thisDFFolder = os.path.join(analysisSubFolder, 'dataframes')
            for bIdx in lOfBlocks:
                theseArgs = loadingMeta['arguments'].copy()
                theseArgs['blockIdx'] = '{}'.format(bIdx)
                theseArgs['processAll'] = False
                thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
                dFPath = os.path.join(
                    thisDFFolder,
                    '{}_{}_df{}.h5'.format(
                        thisBlockBaseName,
                        theseArgs['window'],
                        iteratorSuffix))
                featureLoadingMetaPath = os.path.join(
                    thisDFFolder,
                    '{}_{}_df{}_{}_meta.pickle'.format(
                        thisBlockBaseName,
                        theseArgs['window'],
                        iteratorSuffix, arguments['selectionName']))
                try:
                    with open(featureLoadingMetaPath, 'rb') as _flf:
                        featureLoadingMeta = pickle.load(_flf)
                    for kN in ['outlierTrials']:
                        featureLoadingMeta['alignedAsigsKWargs'].pop(kN)
                    for kN in ['unitQuery', 'selectionName']:
                        if kN in featureLoadingMeta['arguments']:
                            loadingMeta['arguments'][kN] = featureLoadingMeta['arguments'][kN]
                    loadingMeta['alignedAsigsKWargs'] = featureLoadingMeta['alignedAsigsKWargs']
                    with pd.HDFStore(dFPath,  mode='r') as store:
                        theseDF = {}
                        dataKey = '/{}/data'.format(arguments['selectionName'])
                        if dataKey in store:
                            theseDF['main'] = pd.read_hdf(store, dataKey)
                            print('Loaded {} from {}'.format(dataKey, dFPath))
                        controlKey = '/{}/control'.format(arguments['selectionName'])
                        if controlKey in store:
                            theseDF['control'] = pd.read_hdf(store, controlKey)
                            print('Loaded {} from {}'.format(controlKey, dFPath))
                        assert len(theseDF) > 0
                        thisDF = pd.concat(theseDF, names=['controlFlag'])
                        print(' ')
                except Exception:
                    traceback.print_exc()
                    print('Skipping...')
                    continue
                '''thisDF.loc[:, 'expName'] = expName
                thisDF.set_index('expName', inplace=True, append=True)'''
                #
                thisDF.index = thisDF.index.set_levels([currBlockNum], level='segment')
                lOfDF.append(thisDF)
                thisMask = pd.read_hdf(dFPath, '/{}/featureMasks'.format(arguments['selectionName']))
                lOfFeatureMasks.append(thisMask)
                currBlockNum += 1
    dataDF = pd.concat(lOfDF)
    # fill zeros, e.g. if some trials do not have measured position, positions will be NaN
    dataDF.fillna(0, inplace=True)
    ################################################################################################
    # pdb.set_trace()
    if 'controlProportionMask' in loadingMeta:
        if loadingMeta['controlProportionMask'] is not None:
            dataDF = dataDF.loc[loadingMeta['controlProportionMask'], :]
    if 'minBinMask' in loadingMeta:
        if loadingMeta['minBinMask'] is not None:
            # dataTrialInfo = dataDF.index.to_frame().reset_index(drop=True)
            # maskTrialInfo = loadingMeta['minBinMask'].index.to_frame().reset_index(drop=True)
            dataDF = dataDF.loc[loadingMeta['minBinMask'].to_numpy(), :]
    #
    hf.exportNormalizedDataFrame(
        dataDF=dataDF, loadingMeta=loadingMeta, featureInfoMask=thisMask,
        # arguments=loadingMeta['arguments'], selectionName=arguments['selectionName'],
        dataFramesFolder=dataFramesFolder, datasetName=datasetName,
        )