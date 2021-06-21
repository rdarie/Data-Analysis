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
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --lazy                                 load from raw, or regular? [default: False]
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
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.use('QT5Agg')   # generate interactive output
    # matplotlib.use('PS')   # generate postscript output
    # matplotlib.use('Agg')   # generate postscript output
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


'''def exportNormalizedDataFrame(
        dataDF=None, loadingMeta=None, selectionName=None,
        dataFramesFolder=None, datasetName=None,
        ):
    # save, copied from assemble dataframes
    finalDF = dataDF.copy()
    #  #### end of data loading stuff
    if 'spectral' in selectionName:
        normalizationParams = [[], []]
        for expName, dataGroup in dataDF.groupby('expName'):
            for featName, subGroup in dataGroup.groupby('feature', axis='columns'):
                print('calculating pre-normalization params, exp: {}, feature: {}'.format(expName, featName))
                meanLevel = np.nanmean(subGroup.xs(0, level='lag', axis='columns').to_numpy())
                normalizationParams[0].append({
                    'expName': expName,
                    'feature': featName,
                    'mu': meanLevel,
                })
                # finalDF.loc[subGroup.index, subGroup.columns] = dataDF.loc[subGroup.index, subGroup.columns] - meanLevel
                finalDF.loc[subGroup.index, subGroup.columns] = np.sqrt(dataDF.loc[subGroup.index, subGroup.columns] / meanLevel)
        intermediateDF = finalDF.copy()
        for featName, dataGroup in intermediateDF.groupby('feature', axis='columns'):
            print('calculating final normalization params, feature: {}'.format(featName))
            refData = dataGroup.xs(0, level='lag', axis='columns').to_numpy()
            mu = np.nanmean(refData)
            sigma = np.nanstd(refData)
            normalizationParams[1].append({
                'feature': featName,
                'mu': mu,
                'sigma': sigma
            })
            #
            finalDF.loc[:, dataGroup.columns] = (intermediateDF[dataGroup.columns] - mu) / sigma
        #
        def normalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for preParams in params[0]:
                expMask = inputDF.index.get_level_values('expName') == preParams['expName']
                featMask = inputDF.columns.get_level_values('feature') == preParams['feature']
                if expMask.any() and featMask.any():
                    print('pre-normalizing exp {}: feature {}'.format(preParams['expName'], preParams['feature']))
                    # outputDF.loc[expMask, featMask] = inputDF.loc[expMask, featMask] - preParams['mu']
                    outputDF.loc[expMask, featMask] = np.sqrt(inputDF.loc[expMask, featMask] / preParams['mu'])
            intermediateDF = outputDF.copy()
            for postParams in params[1]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('final normalizing feature {}'.format(postParams['feature']))
                    outputDF.loc[:, featMask] = (intermediateDF.loc[:, featMask] - postParams['mu']) / postParams['sigma']
            return outputDF
        #
        def unNormalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for postParams in params[1]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('pre un-normalizing feature {}'.format(postParams['feature']))
                    outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] * postParams['sigma']) + postParams['mu']
            intermediateDF = outputDF.copy()
            for preParams in params[0]:
                expMask = inputDF.index.get_level_values('expName') == preParams['expName']
                featMask = inputDF.columns.get_level_values('feature') == preParams['feature']
                if expMask.any() and featMask.any():
                    print('final un-normalizing exp {}: feature {}'.format(preParams['expName'], preParams['feature']))
                    # outputDF.loc[expMask, featMask] = intermediateDF.loc[expMask, featMask] + preParams['mu']
                    outputDF.loc[expMask, featMask] = intermediateDF.loc[expMask, featMask] ** 2 * preParams['mu']
            return outputDF
        #
        # finalDF = normalizeDataset(finalDF, normalizationParams)
    else:
        # normal time domain data
        normalizationParams = [[]]
        for featName, dataGroup in dataDF.groupby('feature', axis='columns'):
            refData = dataGroup.xs(0, level='lag', axis='columns').to_numpy()
            print('calculating normalization params for {}'.format(featName))
            mu = np.nanmean(refData)
            sigma = np.nanstd(refData)
            print('mu = {} sigma = {}'.format(mu, sigma))
            normalizationParams[0].append({
                'feature': featName,
                'mu': mu,
                'sigma': sigma
            })
            finalDF.loc[:, dataGroup.columns] = (dataDF[dataGroup.columns] - mu) / sigma
        #
        def normalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for postParams in params[0]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('normalizing feature {}'.format(postParams['feature']))
                    print('mu = {} sigma = {}'.format(postParams['mu'], postParams['sigma']))
                    outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] - postParams['mu']) / postParams['sigma']
            return outputDF
        #
        def unNormalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for postParams in params[0]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('un-normalizing feature {}'.format(postParams['feature']))
                    print('mu = {} sigma = {}'.format(postParams['mu'], postParams['sigma']))
                    outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] * postParams['sigma']) + postParams['mu']
            return outputDF
        #
        # finalDF = normalizeDataset(finalDF, normalizationParams)
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    print('saving {} to {}'.format(selectionName, datasetPath))
    finalDF.to_hdf(datasetPath, '/{}/data'.format(selectionName), mode='a')
    thisMask.to_hdf(datasetPath, '/{}/featureMasks'.format(selectionName), mode='a')
    #
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_' + selectionName + '_meta.pickle'
        )
    if os.path.exists(loadingMetaPath):
        os.remove(loadingMetaPath)
    loadingMeta['arguments'] = arguments.copy()
    loadingMeta['normalizationParams'] = normalizationParams
    loadingMeta['normalizeDataset'] = normalizeDataset
    loadingMeta['unNormalizeDataset'] = unNormalizeDataset
    with open(loadingMetaPath, 'wb') as f:
        pickle.dump(loadingMeta, f)
    return'''

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
    with open(iteratorPath, 'rb') as f:
        loadingMeta = pickle.load(f)
    #
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
    cv_kwargs = loadingMeta['cv_kwargs'].copy()
    cvIterator = iteratorsBySegment[0]
    workIdx = cvIterator.work
    ######### data loading stuff
    lOfDF = []
    lOfFeatureMasks = []
    if not arguments['loadFromFrames']:
        alignedAsigsKWargs = loadingMeta['alignedAsigsKWargs'].copy()
        alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
            namedQueries, scratchFolder, **rhsArgs)
        if arguments['verbose']:
            prf.print_memory_usage('before load data')
            print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        nSeg = len(dataBlock.segments)
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
        experimentsToAssemble = loadingMeta['experimentsToAssemble'].copy()
        currBlockNum = 0
        for expName, lOfBlocks in experimentsToAssemble.items():
            thisScratchFolder = os.path.join(scratchPath, expName)
            analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
                arguments, thisScratchFolder)
            thisDFFolder = os.path.join(analysisSubFolder, 'dataframes')
            for bIdx in lOfBlocks:
                theseArgs = arguments.copy()
                theseArgs['blockIdx'] = '{}'.format(bIdx)
                theseArgs['processAll'] = False
                thisBlockBaseName, _ = hf.processBasicPaths(theseArgs)
                dFPath = os.path.join(
                    thisDFFolder,
                    '{}_{}_df{}.h5'.format(
                        thisBlockBaseName,
                        arguments['window'],
                        iteratorSuffix))
                try:
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
                        # db = [print(df.index.names) for df in theseDF]
                        # pdb.set_trace()
                        thisDF = pd.concat(theseDF, names=['controlFlag'])
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
    ################################################################################################
    if 'controlProportionMask' in loadingMeta:
        if loadingMeta['controlProportionMask'] is not None:
            dataDF = dataDF.loc[loadingMeta['controlProportionMask'], :]
    if 'minBinMask' in loadingMeta:
        if loadingMeta['minBinMask'] is not None:
            dataDF = dataDF.loc[loadingMeta['minBinMask'], :]
    #
    hf.exportNormalizedDataFrame(
        dataDF=dataDF, loadingMeta=loadingMeta, featureInfoMask=thisMask,
        arguments=arguments, selectionName=arguments['selectionName'],
        dataFramesFolder=dataFramesFolder, datasetName=datasetName,
        )
    '''finalDF = dataDF.copy()
    #  #### end of data loading stuff
    if 'spectral' in arguments['selectionName']:
        normalizationParams = [[], []]
        for expName, dataGroup in dataDF.groupby('expName'):
            for featName, subGroup in dataGroup.groupby('feature', axis='columns'):
                print('calculating pre-normalization params, exp: {}, feature: {}'.format(expName, featName))
                meanLevel = np.nanmean(subGroup.xs(0, level='lag', axis='columns').to_numpy())
                normalizationParams[0].append({
                    'expName': expName,
                    'feature': featName,
                    'mu': meanLevel,
                })
                # finalDF.loc[subGroup.index, subGroup.columns] = dataDF.loc[subGroup.index, subGroup.columns] - meanLevel
                finalDF.loc[subGroup.index, subGroup.columns] = np.sqrt(dataDF.loc[subGroup.index, subGroup.columns] / meanLevel)
        intermediateDF = finalDF.copy()
        for featName, dataGroup in intermediateDF.groupby('feature', axis='columns'):
            print('calculating final normalization params, feature: {}'.format(featName))
            refData = dataGroup.xs(0, level='lag', axis='columns').to_numpy()
            mu = np.nanmean(refData)
            sigma = np.nanstd(refData)
            normalizationParams[1].append({
                'feature': featName,
                'mu': mu,
                'sigma': sigma
            })
            #
            finalDF.loc[:, dataGroup.columns] = (intermediateDF[dataGroup.columns] - mu) / sigma
        #
        def normalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for preParams in params[0]:
                expMask = inputDF.index.get_level_values('expName') == preParams['expName']
                featMask = inputDF.columns.get_level_values('feature') == preParams['feature']
                if expMask.any() and featMask.any():
                    print('pre-normalizing exp {}: feature {}'.format(preParams['expName'], preParams['feature']))
                    # outputDF.loc[expMask, featMask] = inputDF.loc[expMask, featMask] - preParams['mu']
                    outputDF.loc[expMask, featMask] = np.sqrt(inputDF.loc[expMask, featMask] / preParams['mu'])
            intermediateDF = outputDF.copy()
            for postParams in params[1]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('final normalizing feature {}'.format(postParams['feature']))
                    outputDF.loc[:, featMask] = (intermediateDF.loc[:, featMask] - postParams['mu']) / postParams['sigma']
            return outputDF
        #
        def unNormalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for postParams in params[1]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('pre un-normalizing feature {}'.format(postParams['feature']))
                    outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] * postParams['sigma']) + postParams['mu']
            intermediateDF = outputDF.copy()
            for preParams in params[0]:
                expMask = inputDF.index.get_level_values('expName') == preParams['expName']
                featMask = inputDF.columns.get_level_values('feature') == preParams['feature']
                if expMask.any() and featMask.any():
                    print('final un-normalizing exp {}: feature {}'.format(preParams['expName'], preParams['feature']))
                    # outputDF.loc[expMask, featMask] = intermediateDF.loc[expMask, featMask] + preParams['mu']
                    outputDF.loc[expMask, featMask] = intermediateDF.loc[expMask, featMask] ** 2 * preParams['mu']
            return outputDF
        #
        # finalDF = normalizeDataset(finalDF, normalizationParams)
    else:
        # normal time domain data
        normalizationParams = [[]]
        for featName, dataGroup in dataDF.groupby('feature', axis='columns'):
            refData = dataGroup.xs(0, level='lag', axis='columns').to_numpy()
            print('calculating normalization params for {}'.format(featName))
            mu = np.nanmean(refData)
            sigma = np.nanstd(refData)
            print('mu = {} sigma = {}'.format(mu, sigma))
            normalizationParams[0].append({
                'feature': featName,
                'mu': mu,
                'sigma': sigma
            })
            finalDF.loc[:, dataGroup.columns] = (dataDF[dataGroup.columns] - mu) / sigma
        #
        def normalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for postParams in params[0]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('normalizing feature {}'.format(postParams['feature']))
                    print('mu = {} sigma = {}'.format(postParams['mu'], postParams['sigma']))
                    outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] - postParams['mu']) / postParams['sigma']
            return outputDF
        #
        def unNormalizeDataset(inputDF, params):
            outputDF = inputDF.copy()
            for postParams in params[0]:
                featMask = inputDF.columns.get_level_values('feature') == postParams['feature']
                if featMask.any():
                    print('un-normalizing feature {}'.format(postParams['feature']))
                    print('mu = {} sigma = {}'.format(postParams['mu'], postParams['sigma']))
                    outputDF.loc[:, featMask] = (inputDF.loc[:, featMask] * postParams['sigma']) + postParams['mu']
            return outputDF
        #
        # finalDF = normalizeDataset(finalDF, normalizationParams)
    sanityCheck = False
    if sanityCheck:
        # sanity check that the normalizations are invertible
        originalDF = pd.concat(lOfDF)
        rng = np.random.default_rng()
        plotTIdx = rng.choice(range(originalDF.groupby(['segment', 'originalIndex', 't']).ngroups))
        for tIdx, (name, origGroup) in enumerate(originalDF.groupby(['segment', 'originalIndex', 't'])):
            if tIdx != plotTIdx:
                continue
            plotColumn = rng.choice(origGroup.columns)
            #
            orig = origGroup.loc[:, [plotColumn]]
            normalized = normalizeDataset(orig, normalizationParams)
            final = finalDF.loc[orig.index, [plotColumn]]
            #
            unNormalized = unNormalizeDataset(normalized, normalizationParams)
            unNormalizedFinal = unNormalizeDataset(final, normalizationParams)
            #
            fig, ax = plt.subplots(1, 2)
            tBins = orig.index.get_level_values('bin')
            ax[0].plot(tBins, orig, 'b', lw=4, label='data')
            ax[0].plot(tBins, unNormalized, 'c', ls='dashed', lw=3, label='un-normalize(normalize(data))')
            ax[0].plot(tBins, unNormalizedFinal, 'g', ls='dashdot', lw=2, label='un-normalize(finalData)')
            #
            ax[1].plot(tBins, final, 'r', lw=3, label='finalData')
            ax[1].plot(tBins, normalized, 'm', ls='dashed', lw=2, label='normalize(data)')
            print('orig == unNormalized: {}'.format(float((orig - unNormalized).abs().mean()) < 1e-12))
            print('orig == unNormalizedFinal: {}'.format(float((orig - unNormalizedFinal).abs().mean()) < 1e-12))
            print('orig == normalized: {}'.format(float((orig - normalized).abs().mean()) < 1e-12))
            print('final == normalized: {}'.format(float((final - normalized).abs().mean()) < 1e-12))
            #
            ax[0].legend()
            ax[1].legend()
            plt.show()
            break
    datasetPath = os.path.join(
        dataFramesFolder,
        datasetName + '.h5'
        )
    print('saving {} to {}'.format(arguments['selectionName'], datasetPath))
    finalDF.to_hdf(datasetPath, '/{}/data'.format(arguments['selectionName']), mode='a')
    thisMask.to_hdf(datasetPath, '/{}/featureMasks'.format(arguments['selectionName']), mode='a')
    #
    loadingMetaPath = os.path.join(
        dataFramesFolder,
        datasetName + '_' + arguments['selectionName'] + '_meta.pickle'
        )
    if os.path.exists(loadingMetaPath):
        os.remove(loadingMetaPath)
    loadingMeta['arguments'] = arguments.copy()
    loadingMeta['normalizationParams'] = normalizationParams
    loadingMeta['normalizeDataset'] = normalizeDataset
    loadingMeta['unNormalizeDataset'] = unNormalizeDataset
    with open(loadingMetaPath, 'wb') as f:
        pickle.dump(loadingMeta, f)'''
