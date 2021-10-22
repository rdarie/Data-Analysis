"""  13: Plot Firing Rates and Rasters aligned to Stim
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
    --plotting                             delete outlier trials? [default: False]
    --datasetName=datasetName              filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName          filename for resulting estimator (cross-validated n_comps)
    --iteratorSuffix=iteratorSuffix        filename for resulting estimator (cross-validated n_comps)
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
#
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import pingouin as pg
import traceback
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.preproc.ns5 as ns5
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import os
from dask.distributed import Client, LocalCluster
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from scipy import stats, signal
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import sys
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})
'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'verbose': False, 'winStart': None, 'window': 'XL', 'exp': 'exp202101271100', 'plotting': True, 'blockIdx': '2',
        'loadFromFrames': True, 'datasetName': 'Block_XL_df_ma', 'alignQuery': 'outboundWithStim', 'iteratorSuffix': 'ma',
        'unitQuery': 'pca', 'alignFolderName': 'motion', 'processAll': True, 'selectionName': 'lfp_CAR_spectral_mahal_ledoit',
        'inputBlockSuffix': 'pca', 'lazy': False, 'selector': None, 'winStop': None, 'maskOutlierBlocks': False,
        'analysisName': 'hiRes', 'inputBlockPrefix': 'Block'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
DEBUGGING = False

def autocorrOneSignal(
        x, mode='full', method='auto', dt=1):
    out = signal.correlate(x, x, mode=mode, method=method)
    out = pd.Series(out[x.size:], index=np.arange(1, x.size) * dt)
    out.index.name = 'autocorr_lag'
    return out * dt

def calcAutocorr(
        partition,
        lags=None, mode='full', method='auto',
        indexFactors=['bin'], dt=1,
        dataColNames=None):
    #
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    partitionMeta = partition.loc[:, ~dataColMask]
    # x = partitionData.iloc[:, 1].to_numpy()
    # bla = signal.correlate(x, x, mode=mode, method=method)
    # autocorrOneSignal(partitionData.iloc[:, 1], mode=mode, method=method, dt=dt)
    # partitionData.T.apply(autocorrOneSignal, axis='columns', raw=True, dt=dt, mode=mode, method=method)
    result = partitionData.apply(autocorrOneSignal, axis='index', dt=dt, mode=mode, method=method)
    if lags is not None:
        keepMask = result.index.isin(lags)
        result = result.loc[keepMask, :]
    indexToMatch = [0 for i in result.index]
    resultDF = pd.concat([partitionMeta.drop(indexFactors, axis='columns').iloc[indexToMatch, :].reset_index(drop=True), result.reset_index()], axis='columns')
    resultDF.name = 'autocorr'
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
    #
    if arguments['iteratorSuffix'] is not None:
        iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
    else:
        iteratorSuffix = ''
    #  Overrides
    limitPages = None
    #  End Overrides
    if not arguments['loadFromFrames']:
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_autocorr.h5'.format(
                inputBlockSuffix, arguments['window']))
        iteratorPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_autocorr_iterator{}.pickle'.format(
                inputBlockSuffix, arguments['window'], iteratorSuffix))
        alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
        alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
            namedQueries, scratchFolder, **arguments)
        alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
            scratchFolder, blockBaseName, **arguments)
        #
        alignedAsigsKWargs.update(dict(
            transposeToColumns='feature', concatOn='bin'))
        #
        '''alignedAsigsKWargs['procFun'] = ash.genDetrender(timeWindow=(-200e-3, -100e-3))'''

        if 'windowSize' not in alignedAsigsKWargs:
            alignedAsigsKWargs['windowSize'] = [ws for ws in rasterOpts['windowSizes'][arguments['window']]]
        if 'winStart' in arguments:
            if arguments['winStart'] is not None:
                alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
        if 'winStop' in arguments:
            if arguments['winStop'] is not None:
                alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)
        binInterval = rasterOpts['binOpts'][arguments['analysisName']]['binInterval']
        triggeredPath = os.path.join(
            alignSubFolder,
            blockBaseName + '{}_{}.nix'.format(
                inputBlockSuffix, arguments['window']))

        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = ns5.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = ns5.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
    else:
        # loading from dataframe
        datasetName = arguments['datasetName']
        selectionName = arguments['selectionName']
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_autocorr.h5'.format(
                selectionName, arguments['window']))
        iteratorPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_autocorr_iterator{}.pickle'.format(
                selectionName, arguments['window'], iteratorSuffix))
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
            # iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
            iteratorsBySegment = loadingMeta['iteratorsBySegment']
            # cv_kwargs = loadingMeta['iteratorOpts']
        binInterval = loadingMeta['iteratorOpts']['forceBinInterval'] if (loadingMeta['iteratorOpts']['forceBinInterval'] is not None) else rasterOpts['binOpts'][loadingMeta['arguments']['analysisName']]['binInterval']
        for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
            loadingMeta['arguments'].pop(argName, None)
        arguments.update(loadingMeta['arguments'])
        cvIterator = iteratorsBySegment[0]
        dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
        # ## hack control time base to look like main time base
        # #trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        # #deltaB = trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'].min() - trialInfo.loc[trialInfo['controlFlag'] == 'main', 'bin'].min()
        # #trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'] -= deltaB
        # #dataDF.index = pd.MultiIndex.from_frame(trialInfo)
        # #### detrend as a group?
        # #tMask = ((trialInfo['bin'] >= funKWArgs['baselineTStart']) & (trialInfo['bin'] < funKWArgs['baselineTStop']))
        # #baseline = dataDF.loc[tMask.to_numpy(), :].median()
        # #dataDF = dataDF.subtract(baseline, axis='columns')
    if DEBUGGING:
        for cIdx, cN in enumerate(dataDF.columns):
            dataDF.loc[:, cN] = float(cIdx)
        binVals = dataDF.index.get_level_values('bin')
        binValMask = (binVals < 0.) | (binVals > 0.1)
        dataDF.loc[binValMask, :] = 0.

    funKWArgs = dict(
        lags=np.arange(1, 500, 5) * binInterval,
        # lags=None,
        dt=binInterval)
    groupNames = ['originalIndex', 'segment', 't']
    daskComputeOpts = dict(
        # scheduler='processes'
        scheduler='single-threaded'
        )
    colFeatureInfo = dataDF.columns.copy()
    dataDF.columns = dataDF.columns.get_level_values('feature')
    autocorrDF = ash.splitApplyCombine(
        dataDF, fun=calcAutocorr, funKWArgs=funKWArgs, resultPath=resultPath,
        newMetadataNames=['autocorr_lag'],
        sortIndexBy=['segment', 'originalIndex', 't', 'autocorr_lag'],
        rowKeys=groupNames,
        daskProgBar=False, daskPersist=True, useDask=False,
        daskComputeOpts=daskComputeOpts)
    # sns.heatmap(autocorrDF.to_numpy()); plt.show()
    ####################################################################################################################################################################################
    # #for cN in autocorrDF.columns:
    # #    autocorrDF.loc[:, cN] = autocorrDF[cN].astype(float)
    autocorrDF.columns = colFeatureInfo
    trialInfo = autocorrDF.index.to_frame().reset_index(drop=True)
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz', ],
        'kinematicCondition': ['pedalDirection', 'pedalMovementCat']
        }
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        trialInfo.loc[:, canName] = compoundAnn
    autocorrDF.index = pd.MultiIndex.from_frame(trialInfo)
    #####
    autocorrDF.to_hdf(resultPath, 'autocorr')
    #
    '''
    amplitudeStatsDict = {}
    relativeStatsDict = {}
    recCurve = relativeRaucDF.reset_index()
    for name, group in recCurve.groupby(['electrode', 'kinematicCondition']):
        elecName, kinName = name
        if elecName != 'NA':
            refMask = (recCurve['electrode'] == 'NA') & (recCurve['kinematicCondition'] == kinName)
            if refMask.any():
                refGroup = recCurve.loc[refMask, :]
            else:
                refMask = (recCurve['electrode'] == 'NA')
                refGroup = recCurve.loc[refMask, :]
            for colName in relativeRaucDF.columns:
                if isinstance(colName, tuple):
                    thisEntry = tuple([elecName, kinName] + [a for a in colName])
                else:
                    thisEntry = tuple([elecName, kinName, colName])
                lm = pg.linear_regression(group['trialAmplitude'], group[colName])
                amplitudeStatsDict[thisEntry] = lm
                maxAmpMask = (group['trialAmplitude'] == group['trialAmplitude'].max())
                tt = pg.ttest(group.loc[maxAmpMask, colName], refGroup[colName])
                relativeStatsDict[thisEntry] = tt
    ampStatsDF = pd.concat(amplitudeStatsDict, names=['electrode', 'kinematicCondition'] + relativeRaucDF.columns.names)
    ampStatsDF.set_index('names', append=True, inplace=True)
    ampStatsDF.to_hdf(resultPath, 'amplitudeStats')
    relativeStatsDF = pd.concat(relativeStatsDict, names=['electrode', 'kinematicCondition'] + relativeRaucDF.columns.names)
    relativeStatsDF.to_hdf(resultPath, 'relativeStatsDF')
    #####
    amplitudeStatsPerFBDict = {}
    relativeStatsPerFBDict = {}
    for name, group in recCurve.groupby(['electrode', 'kinematicCondition']):
        elecName, kinName = name
        if elecName != 'NA':
            refMask = (recCurve['electrode'] == 'NA') & (recCurve['kinematicCondition'] == kinName)
            if refMask.any():
                refGroup = recCurve.loc[refMask, :]
            else:
                refMask = (recCurve['electrode'] == 'NA')
                refGroup = recCurve.loc[refMask, :]
            for freqBandName, freqGroup in relativeRaucDF.groupby('freqBandName', axis='columns'):
                thisEntry = tuple([elecName, kinName, freqBandName])
                freqRefGroup = refGroup.loc[:, refGroup.columns.isin(freqGroup.columns)]
                freqGroup.columns = freqGroup.columns.get_level_values('feature')
                freqRefGroup.columns = freqRefGroup.columns.get_level_values('feature')
                freqGroupStack = freqGroup.stack().to_frame(name='rauc').reset_index()
                freqRefGroupStack = freqRefGroup.stack().to_frame(name='rauc').reset_index()
                lm = pg.linear_regression(freqGroupStack['trialAmplitude'], freqGroupStack['rauc'])
                amplitudeStatsPerFBDict[thisEntry] = lm
                maxAmpMask = (freqGroupStack['trialAmplitude'] == freqGroupStack['trialAmplitude'].max())
                tt = pg.ttest(freqGroupStack.loc[maxAmpMask, 'rauc'], freqRefGroupStack['rauc'])
                relativeStatsPerFBDict[thisEntry] = tt
    ampStatsPerFBDF = pd.concat(amplitudeStatsPerFBDict, names=['electrode', 'kinematicCondition', 'freqBandName'])
    ampStatsPerFBDF.set_index('names', append=True, inplace=True)
    ampStatsPerFBDF.to_hdf(resultPath, 'amplitudeStatsPerFreqBand')
    relativeStatsPerFBDF = pd.concat(relativeStatsPerFBDict, names=['electrode', 'kinematicCondition', 'freqBandName'])
    relativeStatsPerFBDF.to_hdf(resultPath, 'relativeStatsDFPerFreqBand')
    #
        '''
    defaultSamplerKWArgs = dict(random_state=42, test_size=0.5)
    defaultPrelimSamplerKWArgs = dict(random_state=42, test_size=0.1)
    # args for tdr.
    defaultSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=defaultSamplerKWArgs)
    defaultPrelimSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=defaultPrelimSamplerKWArgs)
    iteratorKWArgs = dict(
        n_splits=7,
        splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
        prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
        resamplerClass=None, resamplerKWArgs={},
        )
    cvIterator = tdr.trainTestValidationSplitter(
        dataDF=autocorrDF, **iteratorKWArgs)
    iteratorInfo = {
        'iteratorKWArgs': iteratorKWArgs,
        'cvIterator': cvIterator
        }
    with open(iteratorPath, 'wb') as _f:
        pickle.dump(iteratorInfo, _f)
    # if arguments['plotting']:
    if arguments['lazy']:
        dataReader.file.close()
