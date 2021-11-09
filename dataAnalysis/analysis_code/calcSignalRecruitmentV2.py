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
    --showFigures                          show figures? [default: False]
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
from matplotlib.backends.backend_pdf import PdfPages
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
from scipy import stats
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

from pandas import IndexSlice as idxSl
from datetime import datetime as dt
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#


def calcRauc(
        partition,
        tStart=None, tStop=None,
        maskBaseline=False,
        baselineTStart=None, baselineTStop=None,
        dataColNames=None):
    #
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    partitionMeta = partition.loc[:, ~dataColMask]
    #
    tMask = pd.Series(True, index=partition.index)
    if tStart is not None:
        tMask = tMask & (partition['bin'] >= tStart)
    if tStop is not None:
        tMask = tMask & (partition['bin'] < tStop)
    if not tMask.any():
        tMask = pd.Series(True, index=partition.index)
    #  #else:
    #  #    print('calcRauc; tMask.sum() = {}'.format(tMask.sum()))
    if maskBaseline:
        baselineTMask = pd.Series(True, index=partition.index)
        if baselineTStart is not None:
            baselineTMask = baselineTMask & (partition['bin'] >= baselineTStart)
        if baselineTStop is not None:
            baselineTMask = baselineTMask & (partition['bin'] < baselineTStop)
        if not baselineTMask.any():
            baselineTMask = pd.Series(True, index=partition.index)
        if 'foo' in partition.iloc[0, :].to_numpy():
            baselineTMask.loc[:] = True
        baseline = partitionData.loc[baselineTMask, :].median()
    else:
        baseline = 0
    if 'foo' in partition.iloc[0, :].to_numpy():
        tMask.loc[:] = True
    detrended = partitionData.loc[tMask, :].subtract(baseline, axis='columns')
    # plt.plot(partitionData.loc[tMask, :].abs().to_numpy()); plt.show()
    # plt.plot(detrended.abs().to_numpy(), c='b', alpha=0.3); plt.show()
    #
    result = detrended.abs().mean()
    resultDF = pd.concat([partitionMeta.iloc[0, :], result]).to_frame(name=partition.index[0]).T
    resultDF.name = 'rauc'
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
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'lfp_recruitment')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
    #  Overrides
    limitPages = None
    funKWArgs = dict(
        tStart=-100e-3, tStop=200e-3,
        maskBaseline=False,
        baselineTStart=-300e-3, baselineTStop=-150e-3)
    #  End Overrides
    if not arguments['loadFromFrames']:
        pdfPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '_{}_{}_{}_rauc_normalization.pdf'.format(
                expDateTimePathStr, inputBlockSuffix, arguments['window']))
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_rauc.h5'.format(
                inputBlockSuffix, arguments['window']))
        iteratorPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_rauc_iterator{}.pickle'.format(
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
        pdfPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '_{}_{}_{}_rauc_normalization.pdf'.format(
                expDateTimePathStr, selectionName, arguments['window']))
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc.h5'.format(
                selectionName, arguments['window']))
        iteratorPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc_iterator{}.pickle'.format(
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
            # cv_kwargs = loadingMeta['cv_kwargs']
        for argName in ['plotting', 'showFigures', 'debugging', 'verbose']:
            loadingMeta['arguments'].pop(argName, None)
        arguments.update(loadingMeta['arguments'])
        cvIterator = iteratorsBySegment[0]
        dataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selectionName))
        # hack control time base to look like main time base
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        deltaB = trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'].min() - trialInfo.loc[trialInfo['controlFlag'] == 'main', 'bin'].min()
        trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'] -= deltaB
        dataDF.index = pd.MultiIndex.from_frame(trialInfo)
        ### detrend as a group?
        tMask = ((trialInfo['bin'] >= funKWArgs['baselineTStart']) & (trialInfo['bin'] < funKWArgs['baselineTStop']))
        baseline = dataDF.loc[tMask.to_numpy(), :].median()
        # pdb.set_trace()
        '''
        if True:
            fig, ax = plt.subplots(1, 2)
            sns.histplot(dataDF.loc[tMask.to_numpy(), :], ax=ax[0])
            sns.histplot(dataDF.subtract(baseline, axis='columns').loc[tMask.to_numpy(), :], ax=ax[1])
            plt.show()
            '''
        dataDF = dataDF.subtract(baseline, axis='columns')
    groupNames = ['originalIndex', 'segment', 't']
    daskComputeOpts = dict(
        # scheduler='processes'
        scheduler='single-threaded'
        )
    colFeatureInfo = dataDF.columns.copy()
    dataDF.columns = dataDF.columns.get_level_values('feature')
    rawRaucDF = ash.splitApplyCombine(
        dataDF, fun=calcRauc, funKWArgs=funKWArgs, resultPath=resultPath,
        rowKeys=groupNames,
        daskProgBar=True, daskPersist=True, useDask=True,
        daskComputeOpts=daskComputeOpts)
    for cN in rawRaucDF.columns:
        rawRaucDF.loc[:, cN] = rawRaucDF[cN].astype(float)
    rawRaucDF.columns = colFeatureInfo
    rawRaucDF.index = rawRaucDF.index.droplevel('bin')
    trialInfo = rawRaucDF.index.to_frame().reset_index(drop=True)
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz', ],
        'kinematicCondition': ['pedalDirection', 'pedalMovementCat']
        }
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        trialInfo.loc[:, canName] = compoundAnn
    rawRaucDF.index = pd.MultiIndex.from_frame(trialInfo)
    #####
    qScaler = PowerTransformer()
    qScaler.fit(rawRaucDF)
    scaledRaucDF = pd.DataFrame(
        qScaler.transform(rawRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
    ##
    sdThresh = 3.
    clippedScaledRaucDF = scaledRaucDF.clip(upper=sdThresh, lower=-sdThresh)
    clippedRaucDF = pd.DataFrame(
        qScaler.inverse_transform(clippedScaledRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
    infIndices = clippedRaucDF.index[np.isinf(clippedRaucDF).any(axis='columns')]
    # pdb.set_trace()
    clippedRaucDF.drop(infIndices, inplace=True)
    clippedScaledRaucDF.drop(infIndices, inplace=True)
    rawRaucDF.drop(infIndices, inplace=True)
    #
    mmScaler = MinMaxScaler()
    mmScaler.fit(clippedRaucDF)
    normalizedRaucDF = pd.DataFrame(
        mmScaler.transform(clippedRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
    #
    mmScaler2 = MinMaxScaler()
    mmScaler2.fit(clippedScaledRaucDF)
    scaledNormalizedRaucDF = pd.DataFrame(
        mmScaler.transform(clippedScaledRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
    scalers = pd.Series({'scale': qScaler, 'normalize': mmScaler, 'scale_normalize': mmScaler2})
    #
    dfRelativeTo = clippedRaucDF
    referenceRaucDF = (
        dfRelativeTo
            .xs('NA', level='electrode', drop_level=False)
            .xs('NA_NA', level='kinematicCondition', drop_level=False)
            # .median()
            )
    referenceScaler = StandardScaler()
    # referenceScaler = PowerTransformer()
    referenceScaler.fit(referenceRaucDF)
    #
    #relativeRaucDF = clippedRaucDF / referenceRaucDF
    relativeRaucDF = pd.DataFrame(
        referenceScaler.transform(dfRelativeTo.to_numpy()),
        index=dfRelativeTo.index, columns=dfRelativeTo.columns
        )
    print('Saving Rauc to {} ..'.format(resultPath))
    rawRaucDF.to_hdf(resultPath, 'raw')
    clippedRaucDF.to_hdf(resultPath, 'clipped')
    clippedScaledRaucDF.to_hdf(resultPath, 'scaled')
    normalizedRaucDF.to_hdf(resultPath, 'normalized')
    scaledNormalizedRaucDF.to_hdf(resultPath, 'scaled_normalized')
    referenceRaucDF.to_hdf(resultPath, 'reference_raw')
    relativeRaucDF.to_hdf(resultPath, 'relative')
    scalers.to_hdf(resultPath, 'scalers')
    #
    dfForStats = clippedRaucDF
    confidence_alpha = 0.99
    amplitudeStatsDict = {}
    relativeStatsDict = {}
    recCurve = dfForStats.reset_index()
    for name, group in recCurve.groupby(['stimCondition', 'kinematicCondition']):
        elecName, kinName = name
        if elecName != 'NA':
            refMask = (recCurve['electrode'] == 'NA') & (recCurve['kinematicCondition'] == kinName)
            if refMask.any():
                refGroup = recCurve.loc[refMask, :]
            else:
                refMask = (recCurve['electrode'] == 'NA')
                refGroup = recCurve.loc[refMask, :]
            for colName in dfForStats.columns:
                if isinstance(colName, tuple):
                    thisEntry = tuple([elecName, kinName] + [a for a in colName])
                else:
                    thisEntry = tuple([elecName, kinName, colName])
                lm = pg.linear_regression(group['trialAmplitude'], group[colName])
                amplitudeStatsDict[thisEntry] = lm
                highAmps = np.unique(group['trialAmplitude'])[-2:]
                maxAmpMask = group['trialAmplitude'].isin(highAmps)
                # maxAmpMask = group['trialAmplitude'] == group['trialAmplitude'].max())
                tt = pg.ttest(
                    group.loc[maxAmpMask, colName], refGroup[colName],
                    confidence=confidence_alpha)
                tt.loc[:, 'cohen-d'] = np.sign(tt['T']) * tt['cohen-d']
                tt.loc[:, 'critical_T_max'] = tt['dof'].apply(lambda x: stats.t(x).isf((1 - confidence_alpha) / 2))
                tt.loc[:, 'critical_T_min'] = tt['critical_T_max'] * (-1)
                tt.rename(columns={'p-val': 'pval'}, inplace=True)
                '''
                if True:
                    fig, ax = plt.subplots()
                    tx = np.linspace(-2 * tt['T'].abs().iloc[0], 2 * tt['T'].abs().iloc[0], 100)
                    rv = stats.t(tt['dof'].iloc[0])
                    ax.plot(tx, rv.pdf(tx), 'k-', lw=2, label='frozen pdf')
                    ax.axvline(tt['critical_T_max'].iloc[0])
                    ax.axvline(tt['critical_T_min'].iloc[0])
                    plt.show()
                '''
                relativeStatsDict[thisEntry] = tt
        else:
            pass
            # TODO: implement pairwise ttest between no-stim groups
    ampStatsDF = pd.concat(amplitudeStatsDict, names=['stimCondition', 'kinematicCondition'] + dfForStats.columns.names)
    ampStatsDF.set_index('names', append=True, inplace=True)
    # TODO: concatenate stats, correct multicomp, reassign
    # pdb.set_trace()
    reject, pval = pg.multicomp(ampStatsDF['pval'].to_numpy(), alpha=confidence_alpha)
    ampStatsDF.loc[:, 'pval'] = pval
    ampStatsDF.loc[:, 'reject'] = reject
    relativeStatsDF = pd.concat(relativeStatsDict, names=['stimCondition', 'kinematicCondition'] + dfForStats.columns.names)
    reject, pval = pg.multicomp(relativeStatsDF['pval'].to_numpy(), alpha=confidence_alpha)
    relativeStatsDF.loc[:, 'pval'] = pval
    relativeStatsDF.loc[:, 'reject'] = reject
    relativeStatsDF.to_hdf(resultPath, 'relativeStatsDF')
    ampStatsDF.to_hdf(resultPath, 'amplitudeStats')
    #####
    amplitudeStatsPerFBDict = {}
    relativeStatsPerFBDict = {}
    for name, group in recCurve.groupby(['stimCondition', 'kinematicCondition']):
        elecName, kinName = name
        if elecName != 'NA':
            refMask = (recCurve['electrode'] == 'NA') & (recCurve['kinematicCondition'] == kinName)
            if refMask.any():
                refGroup = recCurve.loc[refMask, :]
            else:
                refMask = (recCurve['electrode'] == 'NA')
                refGroup = recCurve.loc[refMask, :]
            for freqBandName, freqGroup in dfForStats.groupby('freqBandName', axis='columns'):
                thisEntry = tuple([elecName, kinName, freqBandName])
                freqRefGroup = refGroup.loc[:, refGroup.columns.isin(freqGroup.columns)]
                freqGroup.columns = freqGroup.columns.get_level_values('feature')
                freqRefGroup.columns = freqRefGroup.columns.get_level_values('feature')
                freqGroupStack = freqGroup.stack().to_frame(name='rauc').reset_index()
                freqRefGroupStack = freqRefGroup.stack().to_frame(name='rauc').reset_index()
                lm = pg.linear_regression(freqGroupStack['trialAmplitude'], freqGroupStack['rauc'])
                amplitudeStatsPerFBDict[thisEntry] = lm
                maxAmpMask = (freqGroupStack['trialAmplitude'] == freqGroupStack['trialAmplitude'].max())
                highAmps = np.unique(freqGroupStack['trialAmplitude'])[-2:]
                maxAmpMask = freqGroupStack['trialAmplitude'].isin(highAmps)
                tt = pg.ttest(
                    freqGroupStack.loc[maxAmpMask, 'rauc'], freqRefGroupStack['rauc'],
                    confidence=confidence_alpha)
                tt.loc[:, 'critical_T_max'] = tt['dof'].apply(lambda x: stats.t(x).isf((1 - confidence_alpha) / 2))
                tt.loc[:, 'critical_T_min'] = tt['critical_T_max'] * (-1)
                tt.loc[:, 'cohen-d'] = np.sign(tt['T']) * tt['cohen-d']
                tt.rename(columns={'p-val': 'pval'}, inplace=True)
                relativeStatsPerFBDict[thisEntry] = tt
        else:
            pass
            # TODO: implement pairwise ttest between no-stim groups
    ampStatsPerFBDF = pd.concat(amplitudeStatsPerFBDict, names=['stimCondition', 'kinematicCondition', 'freqBandName'])
    ampStatsPerFBDF.set_index('names', append=True, inplace=True)
    reject, pval = pg.multicomp(ampStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha)
    ampStatsPerFBDF.loc[:, 'pval'] = pval
    ampStatsPerFBDF.loc[:, 'reject'] = reject
    ampStatsPerFBDF.to_hdf(resultPath, 'amplitudeStatsPerFreqBand')
    relativeStatsPerFBDF = pd.concat(
        relativeStatsPerFBDict,
        names=['stimCondition', 'kinematicCondition', 'freqBandName'])
    reject, pval = pg.multicomp(relativeStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha)
    relativeStatsPerFBDF.loc[:, 'pval'] = pval
    relativeStatsPerFBDF.loc[:, 'reject'] = reject
    relativeStatsPerFBDF.to_hdf(resultPath, 'relativeStatsDFPerFreqBand')
    #
    defaultSamplerKWArgs = dict(random_state=42, test_size=0.5)
    defaultPrelimSamplerKWArgs = dict(random_state=42, test_size=0.1)
    # args for tdr.
    defaultSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None, samplerKWArgs=defaultSamplerKWArgs)
    defaultPrelimSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None, samplerKWArgs=defaultPrelimSamplerKWArgs)
    iteratorKWArgs = dict(
        n_splits=7,
        splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
        prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
        resamplerClass=None, resamplerKWArgs={},
        )
    cvIterator = tdr.trainTestValidationSplitter(
        dataDF=dfForStats, **iteratorKWArgs)
    iteratorInfo = {
        'iteratorKWArgs': iteratorKWArgs,
        'cvIterator': cvIterator
        }
    with open(iteratorPath, 'wb') as _f:
        pickle.dump(iteratorInfo, _f)
    if arguments['plotting']:
        with PdfPages(pdfPath) as pdf:
            plotDFsDict = {
                'raw': rawRaucDF.reset_index(drop=True),
                'scaled': scaledRaucDF.reset_index(drop=True),
                'clippedScaled': clippedScaledRaucDF.reset_index(drop=True),
                'clipped': dfForStats.reset_index(drop=True)
                }
            plotDF = pd.concat(plotDFsDict, names=['dataType'])
            plotDF.columns = plotDF.columns.get_level_values('feature')
            plotDF = plotDF.stack().reset_index()
            plotDF.columns = ['dataType', 'trial', 'feature', 'rauc']
            g = sns.displot(
                data=plotDF, col='dataType',
                x='rauc', hue='feature', kind='hist', element='step'
                )
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        if arguments['lazy']:
            dataReader.file.close()
    #############
    print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
