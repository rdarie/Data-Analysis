"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --winStart=winStart                    start of absolute window (when loading)
    --winStop=winStop                      end of absolute window (when loading)
    --loadFromFrames                       delete outlier trials? [default: False]
    --useCachedResult                      delete outlier trials? [default: False]
    --plotting                             delete outlier trials? [default: False]
    --showFigures                          delete outlier trials? [default: False]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
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
# from tqdm import tqdm
import pdb
import os
import dill as pickle
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.custom_transformers.tdr as tdr
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler, StandardScaler
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from sklearn.preprocessing import scale, robust_scale
from dask.distributed import Client, LocalCluster
from sklearn.model_selection import LeaveOneOut, PredefinedSplit
from sklearn.utils import shuffle
from math import factorial

if arguments['plotting']:
    sns.set(
        context='paper', style='whitegrid',
        palette='dark', font='sans-serif',
        font_scale=1., color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


def noiseCeilV2(
        group, iterMethod=None, dataColNames=None, continuousFactors=None,
        indexFactors=None,
        plotting=False, corrMethod='pearson', maxIter=1e6):
    indexColMask = ~group.columns.isin(dataColNames)
    groupMeta = group.loc[:, indexColMask].copy()
    infoPerTrial = (
        groupMeta
        .drop_duplicates(subset=continuousFactors)
        .copy().set_index(continuousFactors))
    infoPerTrial.loc[:, 'continuousGroup'] = np.arange(infoPerTrial.shape[0])
    groupMeta.loc[:, 'continuousGroup'] = (
        groupMeta.set_index(continuousFactors).index.map(infoPerTrial['continuousGroup']))
    nSamp = infoPerTrial.shape[0]
    dataColMask = group.columns.isin(dataColNames)
    groupData = group.loc[:, group.columns.isin(dataColNames + indexFactors)]
    if iterMethod == 'half':
        nChoose = int(np.ceil(nSamp / 2))
        if maxIter is None:
            maxIter = int(factorial(nSamp) / (factorial(nChoose) ** 2))
        else:
            maxIter = int(maxIter)
        allCorr = pd.DataFrame(np.nan, index=range(maxIter), columns=dataColNames)
        nBins = group.groupby(indexFactors).ngroups
        # allCov = pd.DataFrame(np.nan, index=range(maxIter), columns=dataColNames)
        # allMSE = pd.DataFrame(np.nan, index=range(maxIter), columns=dataColNames)
        if nBins > 1:
            for idx in range(maxIter):
                testGroups = shuffle(infoPerTrial, n_samples=nChoose)['continuousGroup'].to_numpy()
                testMask = groupMeta['continuousGroup'].isin(testGroups).to_numpy()
                if not testMask.any():
                    testMask[:1] = True
                testXBar = groupData.loc[testMask, :].groupby(indexFactors).mean()
                refMask = ~testMask
                if not refMask.any():
                    refMask[-2:] = True
                refXBar = groupData.loc[refMask, :].groupby(indexFactors).mean()
                for cN in dataColNames:
                    r, pval = stats.pearsonr(testXBar[cN], refXBar[cN])
                    allCorr.loc[idx, cN] = r
            resultData = allCorr.mean()
        else:
            resultData = pd.Series(0, index=dataColNames)
    resultDF = group.iloc[[0], :].copy()
    resultDF.loc[resultDF.index[0], dataColNames] = resultData.loc[dataColNames]
    return resultDF


if __name__ == "__main__":
    blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
    )
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName']
    )
    calcSubFolder = os.path.join(analysisSubFolder, 'dataframes')
    if not os.path.exists(calcSubFolder):
        os.makedirs(calcSubFolder, exist_ok=True)
    funKWargs = dict(
        # baseline='mean',
        tStart=-100e-3, tStop=100e-3)
    useCachedResult = arguments['useCachedResult']
    #  End Overrides
    if not arguments['loadFromFrames']:
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_rauc.h5'.format(
                inputBlockSuffix, arguments['window']))
        alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
        alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
            namedQueries, scratchFolder, **arguments)
        alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
            scratchFolder, blockBaseName, **arguments)
        #
        alignedAsigsKWargs.update(dict(
            transposeToColumns='feature', concatOn='bin'))
        #
        '''alignedAsigsKWargs['procFun'] = ash.genDetrender(
            timeWindow=(-200e-3, -100e-3))'''
        if 'windowSize' not in alignedAsigsKWargs:
            alignedAsigsKWargs['windowSize'] = [ws for ws in rasterOpts['windowSizes'][arguments['window']]]
        if 'winStart' in arguments:
            if arguments['winStart'] is not None:
                alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
        if 'winStop' in arguments:
            if arguments['winStop'] is not None:
                alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)
        triggeredPath = os.path.join(
            alignSubFolder,
            blockBaseName + '{}_{}.nix'.format(
                inputBlockSuffix, arguments['window']))
        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = ns5.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
        cvIterator = tdr.trainTestValidationSplitter(
            dataDF=dataDF, **theseIteratorOpts['cvKWArgs'])
    else:
        # loading from dataframe
        datasetName = arguments['datasetName']
        selectionName = arguments['selectionName']
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc.h5'.format(
                selectionName, arguments['window']))
        dataFramesFolder = os.path.join(analysisSubFolder, 'dataframes')
        datasetPath = os.path.join(
            dataFramesFolder,
            datasetName + '.h5'
        )
        loadingMetaPath = os.path.join(
            dataFramesFolder,
            datasetName + '_{}'.format(selectionName) + '_meta.pickle'
        )
        with open(loadingMetaPath, 'rb') as _f:
            loadingMeta = pickle.load(_f)
            iteratorsBySegment = loadingMeta['iteratorsBySegment']
            cv_kwargs = loadingMeta['cv_kwargs']
        for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
            loadingMeta['arguments'].pop(argName, None)
        arguments.update(loadingMeta['arguments'])
        cvIterator = iteratorsBySegment[0]
        dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
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
    funKWArgs = dict(
        plotting=False, iterMethod='half',
        continuousFactors=['segment', 't', 'originalIndex'],
        indexFactors=['bin'],
        corrMethod='pearson', maxIter=500)
    colFeatureInfo = [nm for nm in dataDF.columns.names if nm != 'feature']
    resDF = ash.splitApplyCombine(
        dataDF, fun=noiseCeilV2, resultPath=resultPath,
        funArgs=[], funKWArgs=funKWArgs,
        rowKeys=stimulusConditionNames, colKeys=colFeatureInfo,
        daskProgBar=True,
        daskPersist=True, useDask=True, retainInputIndex=True,
        daskComputeOpts=daskComputeOpts, columnFeatureInfoHack=True)
    trialInfo = resDF.index.to_frame().reset_index(drop=True)
    resDF.index = pd.MultiIndex.from_frame(
        trialInfo.loc[:, stimulusConditionNames])
    resDF.to_hdf(resultPath, 'noiseCeil')
    
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'])
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
        pdfName = os.path.join(figureOutputFolder, 'noise_ceil_halves.pdf')
        with PdfPages(pdfName) as pdf:
            plotIndex = pd.MultiIndex.from_frame(
                trialInfo.loc[:, stimulusConditionNames])
            plotDF = resDF.copy()
            plotDF.index = plotIndex
            grid_kws = {"width_ratios": (30, 1), 'wspace': 0.01}
            aspect = plotDF.shape[1] / plotDF.shape[0]
            h = 12
            w = h * aspect
            fig, (ax, cbar_ax) = plt.subplots(
                1, 2,
                gridspec_kw=grid_kws,
                figsize=(w, h))
            ax = sns.heatmap(
                plotDF, ax=ax,
                cbar_ax=cbar_ax, vmin=-1, vmax=1, cmap='vlag')
            titleText = 'Noise Ceiling'
            figTitle = fig.suptitle(titleText)
            pdf.savefig(
                bbox_inches='tight',
                bbox_extra_artists=[figTitle]
            )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
