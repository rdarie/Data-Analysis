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
from sklego.preprocessing import PatsyTransformer
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

def calcStd(
        partition,
        tStart=None, tStop=None,
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
    if 'foo' in partition.iloc[0, :].to_numpy():
        tMask.loc[:] = True
    dispersion = partitionData.loc[tMask, :].std()
    # plt.plot(partitionData.loc[tMask, :].abs().to_numpy()); plt.show()
    # plt.plot(detrended.abs().to_numpy(), c='b', alpha=0.3); plt.show()
    #
    resultDF = pd.concat([partitionMeta.iloc[0, :], dispersion]).to_frame(name=partition.index[0]).T
    resultDF.name = 'dispersion'
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
        htmlOutputPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '_{}_{}_{}_rauc_sorted.html'.format(
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
        htmlOutputPath = os.path.join(
            figureOutputFolder,
            blockBaseName + '_{}_{}_{}_rauc_sorted.html'.format(
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
        if ('mahal' in arguments['selectionName']) and ('spectral' in arguments['selectionName']):
            oldColumns = dataDF.columns.to_frame().reset_index(drop=True)
            oldColumns.loc[:, 'freqBandName'] = oldColumns['freqBandName'].apply(lambda x: 'cross_frequency' if x in ['all', 'NA'] else x)
            dataDF.columns = pd.MultiIndex.from_frame(oldColumns)
        elif ('spectral' not in arguments['selectionName']):
            oldColumns = dataDF.columns.to_frame().reset_index(drop=True)
            oldColumns.loc[:, 'freqBandName'] = oldColumns['freqBandName'].apply(lambda x: 'broadband' if x in ['all', 'NA'] else x)
            dataDF.columns = pd.MultiIndex.from_frame(oldColumns)
        # hack control time base to look like main time base
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        deltaB = trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'].min() - trialInfo.loc[trialInfo['controlFlag'] == 'main', 'bin'].min()
        trialInfo.loc[trialInfo['controlFlag'] == 'control', 'bin'] -= deltaB
        dataDF.index = pd.MultiIndex.from_frame(trialInfo)
        ### detrend as a group?
        detrendAsGroup = True
        if detrendAsGroup:
            tMask = ((trialInfo['bin'] >= funKWArgs['baselineTStart']) & (trialInfo['bin'] < funKWArgs['baselineTStop']))
            baseline = dataDF.loc[tMask.to_numpy(), :].median()
            dataDF = dataDF.subtract(baseline, axis='columns')
    if os.path.exists(resultPath):
        print('Previous results found at {}. deleting...'.format(resultPath))
        os.remove(resultPath)
    if os.path.exists(pdfPath):
        print('Previous printout found at {}. deleting...'.format(pdfPath))
        os.remove(pdfPath)
    groupNames = ['originalIndex', 'segment', 't']
    daskComputeOpts = dict(
        # scheduler='processes'
        scheduler='single-threaded'
        )
    ####################################################################################
    # Remove amplitudes that were not tested across all movement types
    removeExcessAmps = True
    if removeExcessAmps:
        trimWhichAmps = 'larger'
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        maskForAmps = pd.Series(False, index=trialInfo.index)
        for elecName, tig in trialInfo.groupby('electrode'):
            if elecName != 'NA':
                minAmp = tig.groupby(['pedalDirection', 'pedalMovementCat']).min()['trialAmplitude'].max()
                maxAmp = tig.groupby(['pedalDirection', 'pedalMovementCat']).max()['trialAmplitude'].min()
                if trimWhichAmps == 'larger':
                    thisMaskForAmps = (tig['trialAmplitude'] > maxAmp)
                elif trimWhichAmps == 'smaller':
                    thisMaskForAmps = (tig['trialAmplitude'] < minAmp)
                else: # both
                    thisMaskForAmps = (tig['trialAmplitude'] > maxAmp) | (tig['trialAmplitude'] < minAmp)
                maskForAmps.loc[tig.index] = maskForAmps.loc[tig.index] | thisMaskForAmps
        if maskForAmps.any():
            # dataDF.loc[maskForAmps.to_numpy(), :].index.to_frame().reset_index(drop=True)
            dataDF = dataDF.loc[~maskForAmps.to_numpy(), :]
    ####################################################################################
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
    for cN in rawRaucDF.columns:
        rawRaucDF.loc[:, cN] = rawRaucDF[cN].astype(float)

    dispersionDF = ash.splitApplyCombine(
        dataDF, fun=calcStd, funKWArgs={
            k: funKWArgs[k] for k in funKWArgs.keys() if k in ['tStart', 'tStop']
        }, resultPath=resultPath,
        rowKeys=groupNames,
        daskProgBar=True, daskPersist=True, useDask=True,
        daskComputeOpts=daskComputeOpts)
    for cN in dispersionDF.columns:
        dispersionDF.loc[:, cN] = dispersionDF[cN].astype(float)
    dispersionDF.columns = colFeatureInfo
    dispersionDF.index = dispersionDF.index.droplevel('bin')
    for cN in dispersionDF.columns:
        dispersionDF.loc[:, cN] = dispersionDF[cN].astype(float)
    trialInfo = rawRaucDF.index.to_frame().reset_index(drop=True)
    padStimNoMoveControl = True
    if padStimNoMoveControl:
        paddingListRauc = [rawRaucDF]
        paddingListDisp = [dispersionDF]
        paddingMetaList = [trialInfo]
        for elecName in trialInfo['electrode'].unique():
            if elecName != 'NA':
                padMask = (trialInfo['controlFlag'] == 'control').to_numpy()
                thisPaddingRauc = rawRaucDF.loc[padMask, :].copy()
                paddingListRauc.append(thisPaddingRauc)
                thisPaddingDisp = dispersionDF.loc[padMask, :].copy()
                paddingListDisp.append(thisPaddingDisp)
                thisPaddingMeta = trialInfo.loc[padMask, :].copy()
                thisPaddingMeta.loc[:, 'electrode'] = elecName
                paddingMetaList.append(thisPaddingMeta)
                for name, group in trialInfo.groupby(['pedalDirection', 'pedalMovementCat']):
                    padMask = trialInfo.index.isin(group.index[group['trialRateInHz'] == 0.])
                    thisPaddingRauc = rawRaucDF.loc[padMask, :].copy()
                    paddingListRauc.append(thisPaddingRauc)
                    thisPaddingDisp = dispersionDF.loc[padMask, :].copy()
                    paddingListDisp.append(thisPaddingDisp)
                    thisPaddingMeta = trialInfo.loc[padMask, :].copy()
                    thisPaddingMeta.loc[:, 'electrode'] = elecName
                    paddingMetaList.append(thisPaddingMeta)
        rawRaucDF = pd.concat(paddingListRauc)
        dispersionDF = pd.concat(paddingListDisp)
        trialInfo = pd.concat(paddingMetaList, ignore_index=True)
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz', ],
        'kinematicCondition': ['pedalDirection', 'pedalMovementCat'],
        'kinAndRateCondition': ['pedalDirection', 'pedalMovementCat', 'trialRateInHz'],
        'kinAndElecCondition': ['electrode', 'pedalDirection', 'pedalMovementCat'],
        }
    compoundAnnLookup = []
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=trialInfo.index)
        for name, group in trialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        trialInfo.loc[:, canName] = compoundAnn
        thisCAL = trialInfo.loc[:, can + [canName]].drop_duplicates().set_index(canName)
        thisCAL.index.name = 'compoundAnnotation'
        compoundAnnLookup.append(thisCAL)
    compoundAnnLookupDF = pd.concat(compoundAnnLookup)
    rawRaucDF.index = pd.MultiIndex.from_frame(trialInfo)
    dispersionDF.index = pd.MultiIndex.from_frame(trialInfo)
    #
    # rawRaucDF.index.to_frame().reset_index(drop=True).groupby(['kinematicCondition', 'electrode', 'trialRateInHz']).groups.keys()
    raucForExport = rawRaucDF.xs('main', level='controlFlag').mean(axis=1).sort_values(ascending=False).to_frame(name='average_rauc')
    raucForExportTrialInfo = raucForExport.index.to_frame().reset_index(drop=True)
    raucForExportTrialInfo = raucForExportTrialInfo.loc[:, [
        'expName', 'segment', 'originalIndex', 't', 'stimCondition', 'kinematicCondition', 'kinAndRateCondition', 'kinAndElecCondition']]
    raucForExport.index = pd.MultiIndex.from_frame(raucForExportTrialInfo)
    raucForExport.to_html(htmlOutputPath)
    ############
    dfRelativeTo = rawRaucDF
    # referenceRaucDF = dfRelativeTo
    referenceRaucDF = (
        dfRelativeTo
            .xs('NA', level='electrode', drop_level=False)
            .xs('NA_NA', level='kinematicCondition', drop_level=False)
            # .median()
            )
    qScaler = PowerTransformer()  # method='yeo-johnson'
    qScaler.fit(referenceRaucDF)
    scaledRaucDF = pd.DataFrame(
        qScaler.transform(dfRelativeTo),
        index=dfRelativeTo.index,
        columns=dfRelativeTo.columns)
    ##
    sdThresh = 5.
    clippedScaledRaucDF = scaledRaucDF.clip(upper=sdThresh)
    # pdb.set_trace()
    clippedRaucDF = pd.DataFrame(
        qScaler.inverse_transform(clippedScaledRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
    infIndices = clippedRaucDF.index[np.isinf(clippedRaucDF).any(axis='columns')]
    #
    # clipThresh = 1 - 1e-4
    # clippedRaucDF = rawRaucDF.apply(lambda x: x.clip(upper=x.quantile(clipThresh)))
    # infIndices = clippedRaucDF.index[clippedRaucDF.apply(lambda x: x == x.max()).any(axis='columns')]
    #
    #
    # rawRaucDF.apply(lambda x: print(x.name))
    # (rawRaucDF - rawRaucDF.apply(lambda x: x.clip(upper=x.quantile(0.99)))).to_numpy()
    # (rawRaucDF - rawRaucDF.clip(rawRaucDF.quantile(0.99), axis=1)).to_numpy()
    # (rawRaucDF.apply(lambda x: x.clip(upper=x.quantile(0.99))) - ).to_numpy()
    if infIndices.size > 0:
        scaledRaucDF.drop(infIndices, inplace=True)
        clippedRaucDF.drop(infIndices, inplace=True)
        clippedScaledRaucDF.drop(infIndices, inplace=True)
        rawRaucDF.drop(infIndices, inplace=True)
        dispersionDF.drop(infIndices, inplace=True)
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
    scalers = pd.Series({'boxcox': qScaler, 'minmax': mmScaler, 'boxcox_minmax': mmScaler2})
    #
    #
    dispersionDF.to_hdf(resultPath, 'dispersion')
    rawRaucDF.to_hdf(resultPath, 'raw')
    scaledRaucDF.to_hdf(resultPath, 'boxcox')
    clippedRaucDF.to_hdf(resultPath, 'raw_clipped')
    clippedScaledRaucDF.to_hdf(resultPath, 'boxcox_clipped')
    normalizedRaucDF.to_hdf(resultPath, 'normalized')
    scaledNormalizedRaucDF.to_hdf(resultPath, 'boxcox_normalized')
    referenceRaucDF.to_hdf(resultPath, 'reference_raw')
    scalers.to_hdf(resultPath, 'scalers')
    compoundAnnLookupDF.to_hdf(resultPath, 'compoundAnnLookup')
    #
    dfForStats = scaledRaucDF
    recCurve = dfForStats.reset_index()
    xPt = PatsyTransformer("trialAmplitude / trialRateInHz", return_type="dataframe")
    X = xPt.fit_transform(recCurve)
    xStd = X.std()
    xStd.loc[xStd == 0] = 1.
    yStd = dfForStats.std()
    # recCurve.loc[:, 'trialAmplitude'] /= recCurve['trialAmplitude'].max()
    # recCurve.loc[:, 'trialRateInHz'] /= recCurve['trialRateInHz'].max()
    amplitudeStatsDict = {}
    for name, group in recCurve.groupby(['electrode', 'kinematicCondition']):
        elecName, kinName = name
        if elecName != 'NA':
            kinAndElecName = '{}_{}'.format(elecName, kinName)
            groupMask = X.index.isin(group.index)
            for colName in dfForStats.columns:
                if isinstance(colName, tuple):
                    thisEntry = tuple([elecName, kinName, kinAndElecName] + [a for a in colName])
                else:
                    thisEntry = tuple([elecName, kinName, kinAndElecName, colName])
                thisX = X.loc[groupMask, :].copy()
                if (group['trialRateInHz'].unique() > 0).sum() < 2:
                    affectedColNames = [cN for cN in thisX.columns if 'trialRateInHz' in cN]
                    thisX.loc[:, affectedColNames] = 0.
                lmRes = pg.linear_regression(thisX, group[colName], relimp=True)
                ##
                lmRes.loc[:, 'coefStd'] = lmRes['coef'] * lmRes['names'].map(xStd) / yStd[colName]
                ###
                # lmResStd = pg.linear_regression(X / xStd, group[colName] / group[colName].std())
                # print('{}\n{}'.format(lmRes.loc[:, ['names', 'coefStd']],lmResStd.loc[:, ['names', 'coef']]))
                amplitudeStatsDict[thisEntry] = lmRes
    ampStatsDF = pd.concat(amplitudeStatsDict, names=['electrode', 'kinematicCondition', 'kinAndElecCondition'] + dfForStats.columns.names + ['factorIndex'])
    ampStatsDF.index = ampStatsDF.index.droplevel('factorIndex')
    ampStatsDF.set_index('names', append=True, inplace=True)
    # TODO: concatenate stats, correct multicomp, reassign
    confidence_alpha = 0.05
    correctMultiCompHere = False
    if correctMultiCompHere:
        reject, pval = pg.multicomp(ampStatsDF['pval'].to_numpy(), alpha=confidence_alpha)
        ampStatsDF.loc[:, 'pval'] = pval
        ampStatsDF.loc[:, 'reject'] = reject
    ####
    relativeStatsDict = {}
    for name, group in recCurve.groupby(['stimCondition', 'kinematicCondition']):
        stimConditionName, kinName = name
        elecName = group['electrode'].unique()[0]
        trialRateInHz = group['trialRateInHz'].unique()[0]
        kinAndElecName = '{}_{}'.format(elecName, kinName)
        if not ((elecName == 'NA') or (trialRateInHz == 0)) :
            refMask = (recCurve['electrode'] == 'NA') & (recCurve['kinematicCondition'] == kinName)
            if refMask.any():
                refGroup = recCurve.loc[refMask, :]
            else:
                refMask = (recCurve['electrode'] == 'NA')
                refGroup = recCurve.loc[refMask, :]
            for colName in dfForStats.columns:
                if isinstance(colName, tuple):
                    thisEntry = tuple([stimConditionName, kinName, kinAndElecName] + [a for a in colName])
                else:
                    thisEntry = tuple([stimConditionName, kinName, kinAndElecName, colName])
                ####
                # highAmps = np.unique(group['trialAmplitude'])[-2:]
                # maxAmpMask = group['trialAmplitude'].isin(highAmps)
                ####
                maxAmpMask = (group['trialAmplitude'] == group['trialAmplitude'].max())
                ####
                tt = pg.ttest(
                    group.loc[maxAmpMask, colName], refGroup[colName],
                    #  group.loc[maxAmpMask, colName], group.loc[refMask, colName],
                    confidence=confidence_alpha)
                u1 = np.mean(group.loc[maxAmpMask, colName])
                # u2 = np.mean(group.loc[refMask, colName])
                # s2 = np.std(group.loc[refMask, colName])
                u2 = np.mean(refGroup[colName])
                s2 = np.std(refGroup[colName])
                tt.loc[:, 'glass'] = (u1 - u2) / s2
                tt.loc[:, 'cohen-d'] = np.sign(tt['T']) * tt['cohen-d']
                tt.loc[:, 'critical_T_max'] = tt['dof'].apply(lambda x: stats.t(x).isf((1 - confidence_alpha) / 2))
                tt.loc[:, 'critical_T_min'] = tt['critical_T_max'] * (-1)
                tt.rename(columns={'p-val': 'pval'}, inplace=True)
                tt.loc[:, 'hedges'] = pg.convert_effsize(
                    tt['cohen-d'], 'cohen', 'hedges',
                    nx=group.loc[maxAmpMask, colName].shape[0],
                    ny=group.loc[refMask, colName].shape[0])
                tt.loc[:, 'A'] = refGroup['trialAmplitude'].min()
                tt.loc[:, 'B'] = group['trialAmplitude'].max()
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
    relativeStatsDF = pd.concat(relativeStatsDict, names=['stimCondition', 'kinematicCondition', 'kinAndElecCondition'] + dfForStats.columns.names + ['testType'])
    relativeStatsDF.index = relativeStatsDF.index.droplevel('testType')
    #
    if correctMultiCompHere:
        reject, pval = pg.multicomp(relativeStatsDF['pval'].to_numpy(), alpha=confidence_alpha)
        relativeStatsDF.loc[:, 'pval'] = pval
        relativeStatsDF.loc[:, 'reject'] = reject
    for cN in ['electrode', 'trialRateInHz']:
        relativeStatsDF.loc[:, cN] = relativeStatsDF.index.get_level_values('stimCondition').map(compoundAnnLookupDF[cN])
        relativeStatsDF.set_index(cN, append=True, inplace=True)
    ###
    relativeStatsDF.to_hdf(resultPath, 'relativeStatsDF')
    ampStatsDF.to_hdf(resultPath, 'amplitudeStats')
    #####
    dataNoStim = recCurve.loc[(recCurve['electrode'] == 'NA'), :].copy()
    dataNoStim.columns = dataNoStim.columns.get_level_values('feature')
    noStimAnovaDict = {}
    noStimTTestDict = {}
    for colName in dfForStats.columns:
        featName = colName[0]
        anovaRes = pg.welch_anova(data=dataNoStim, dv=featName, between='kinematicCondition')
        anovaRes.rename(columns={'p-unc': 'pval'}, inplace=True)
        tt = pg.pairwise_gameshowell(data=dataNoStim, dv=featName, between='kinematicCondition')
        tt.loc[:, 'critical_T_max'] = tt['df'].apply(lambda x: stats.t(x).isf((1 - confidence_alpha) / 2))
        tt.loc[:, 'critical_T_min'] = tt['critical_T_max'] * (-1)
        ###
        tt.rename(columns={'pval': 'pval-corr'}, inplace=True)
        groupSizes = dataNoStim.groupby('kinematicCondition', observed=True)[featName].count()
        pUncFun = lambda x: stats.t.sf(np.abs(x['T']), groupSizes[x['A']] + groupSizes[x['B']] - 2) * 2
        tt.loc[:, 'pval'] = tt.apply(pUncFun, axis='columns')
        cohenDFun = lambda x: pg.compute_effsize_from_t(x['T'], nx=groupSizes[x['A']], ny=groupSizes[x['B']], eftype='cohen')
        tt.loc[:, 'cohen-d'] = tt.apply(cohenDFun, axis='columns')
        #
        def glass(x):
            u1 = dataNoStim.loc[dataNoStim['kinematicCondition'] == x['A'], featName].mean()
            u2 = dataNoStim.loc[dataNoStim['kinematicCondition'] == x['B'], featName].mean()
            s2 = dataNoStim.loc[dataNoStim['kinematicCondition'] == x['B'], featName].std()
            return (u1 - u2) / s2
        tt.loc[:, 'glass'] = tt.apply(glass, axis='columns')
        ###
        noStimTTestDict[colName] = tt
        noStimAnovaDict[colName] = anovaRes
    #####
    noStimAnovaDF = pd.concat(noStimAnovaDict, names=dfForStats.columns.names)
    if correctMultiCompHere:
        reject, pval = pg.multicomp(noStimAnovaDF['pval'].to_numpy(), alpha=confidence_alpha)
        noStimAnovaDF.loc[:, 'pval'] = pval
        noStimAnovaDF.loc[:, 'reject'] = reject
    noStimAnovaDF.to_hdf(resultPath, 'noStimAnova')
    ##
    noStimTTestDF = pd.concat(noStimTTestDict, names=dfForStats.columns.names)
    if correctMultiCompHere:
        reject, pval = pg.multicomp(noStimTTestDF['pval'].to_numpy(), alpha=confidence_alpha)
        noStimTTestDF.loc[:, 'pval'] = pval
        noStimTTestDF.loc[:, 'reject'] = reject
    #
    noStimTTestDF.loc[:, 'electrode'] = 'NA'
    noStimTTestDF.loc[:, 'trialAmplitude'] = 0
    noStimTTestDF.loc[:, 'trialRateInHz'] = 0
    noStimTTestDF.loc[:, 'stimCondition'] = 'NA_0.0'
    noStimTTestDF.set_index(['electrode', 'trialAmplitude', 'trialRateInHz', 'stimCondition'], append=True, inplace=True)
    noStimTTestDF.to_hdf(resultPath, 'noStimTTest')
    ###
    amplitudeStatsPerFBDict = {}
    for name, group in recCurve.groupby(['electrode', 'kinematicCondition']):
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
                freqGroup.columns = freqGroup.columns.get_level_values('feature')
                freqGroupStack = freqGroup.stack().to_frame(name='rauc').reset_index()
                # normalization
                freqGroupStack.loc[:, 'trialAmplitude'] /= freqGroupStack['trialAmplitude'].max()
                freqGroupStack.loc[:, 'trialRateInHz'] /= freqGroupStack['trialRateInHz'].max()
                # normalization
                xPt = PatsyTransformer("trialAmplitude * trialRateInHz", return_type="dataframe")
                X = xPt.fit_transform(freqGroupStack)
                lm = pg.linear_regression(X, freqGroupStack['rauc'], relimp=True)
                amplitudeStatsPerFBDict[thisEntry] = lm
    ampStatsPerFBDF = pd.concat(amplitudeStatsPerFBDict, names=['electrode', 'kinematicCondition', 'freqBandName'])
    ampStatsPerFBDF.set_index('names', append=True, inplace=True)
    if correctMultiCompHere:
        reject, pval = pg.multicomp(ampStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha)
        ampStatsPerFBDF.loc[:, 'pval'] = pval
        ampStatsPerFBDF.loc[:, 'reject'] = reject
    ampStatsPerFBDF.to_hdf(resultPath, 'amplitudeStatsPerFreqBand')
    ######
    relativeStatsPerFBDict = {}
    for name, group in recCurve.groupby(['stimCondition', 'kinematicCondition']):
        stimConditionName, kinName = name
        elecName = group['electrode'].unique()[0]
        if elecName != 'NA':
            refMask = (recCurve['electrode'] == 'NA') & (recCurve['kinematicCondition'] == kinName)
            if refMask.any():
                refGroup = recCurve.loc[refMask, :]
            else:
                refMask = (recCurve['electrode'] == 'NA')
                refGroup = recCurve.loc[refMask, :]
            for freqBandName, freqGroup in dfForStats.groupby('freqBandName', axis='columns'):
                thisEntry = tuple([stimConditionName, kinName, freqBandName])
                freqRefGroup = refGroup.loc[:, refGroup.columns.isin(freqGroup.columns)]
                freqGroup.columns = freqGroup.columns.get_level_values('feature')
                freqRefGroup.columns = freqRefGroup.columns.get_level_values('feature')
                freqGroupStack = freqGroup.stack().to_frame(name='rauc').reset_index()
                freqRefGroupStack = freqRefGroup.stack().to_frame(name='rauc').reset_index()
                #####
                # maxAmpMask = (freqGroupStack['trialAmplitude'] == freqGroupStack['trialAmplitude'].max())
                highAmps = np.unique(freqGroupStack['trialAmplitude'])[-2:]
                maxAmpMask = freqGroupStack['trialAmplitude'].isin(highAmps)
                #####
                tt = pg.ttest(
                    freqGroupStack.loc[maxAmpMask, 'rauc'], freqRefGroupStack['rauc'],
                    confidence=confidence_alpha)
                u1 = np.mean(freqGroupStack.loc[maxAmpMask, 'rauc'])
                u2 = np.mean(freqRefGroupStack['rauc'])
                s2 = np.std(freqRefGroupStack['rauc'])
                tt.loc[:, 'glass'] = (u1 - u2) / s2
                tt.loc[:, 'critical_T_max'] = tt['dof'].apply(lambda x: stats.t(x).isf((1 - confidence_alpha) / 2))
                tt.loc[:, 'critical_T_min'] = tt['critical_T_max'] * (-1)
                tt.loc[:, 'cohen-d'] = np.sign(tt['T']) * tt['cohen-d']
                tt.loc[:, 'hedges'] = pg.convert_effsize(
                    tt['cohen-d'], 'cohen', 'hedges',
                    nx=freqGroupStack.loc[maxAmpMask, 'rauc'].shape[0],
                    ny=freqRefGroupStack['rauc'].shape[0])
                tt.rename(columns={'p-val': 'pval'}, inplace=True)
                relativeStatsPerFBDict[thisEntry] = tt
    relativeStatsPerFBDF = pd.concat(
        relativeStatsPerFBDict,
        names=['stimCondition', 'kinematicCondition', 'freqBandName'])
    if correctMultiCompHere:
        reject, pval = pg.multicomp(relativeStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha)
        relativeStatsPerFBDF.loc[:, 'pval'] = pval
        relativeStatsPerFBDF.loc[:, 'reject'] = reject
    relativeStatsPerFBDF.to_hdf(resultPath, 'relativeStatsDFPerFreqBand')
    #
    noStimAnovaPerFBDict = {}
    noStimTTestPerFBDict = {}
    for freqBandName, freqGroup in dfForStats.groupby('freqBandName', axis='columns'):
        refMask = (freqGroup.index.get_level_values('electrode') == 'NA')
        freqRefGroup = freqGroup.loc[refMask, :]
        freqRefGroup.columns = freqRefGroup.columns.get_level_values('feature')
        freqRefGroupStack = freqRefGroup.stack().to_frame(name='rauc').reset_index()
        #
        anovaRes = pg.welch_anova(data=freqRefGroupStack, dv='rauc', between='kinematicCondition')
        anovaRes.rename(columns={'p-unc': 'pval'}, inplace=True)
        tt = pg.pairwise_gameshowell(data=freqRefGroupStack, dv='rauc', between='kinematicCondition')
        tt.loc[:, 'critical_T_max'] = tt['df'].apply(lambda x: stats.t(x).isf((1 - confidence_alpha) / 2))
        tt.loc[:, 'critical_T_min'] = tt['critical_T_max'] * (-1)
        noStimTTestPerFBDict[freqBandName] = tt
        noStimAnovaPerFBDict[freqBandName] = anovaRes
    #####
    noStimAnovaDFPerFB = pd.concat(noStimAnovaPerFBDict, names=['freqBandName'])
    if correctMultiCompHere:
        reject, pval = pg.multicomp(noStimAnovaDFPerFB['pval'].to_numpy(), alpha=confidence_alpha)
        noStimAnovaDFPerFB.loc[:, 'pval'] = pval
        noStimAnovaDFPerFB.loc[:, 'reject'] = reject
    noStimAnovaDFPerFB.to_hdf(resultPath, 'noStimAnovaPerFB')
    ##
    noStimTTestDFPerFB = pd.concat(noStimTTestPerFBDict, names=['freqBandName'])
    if correctMultiCompHere:
        reject, pval = pg.multicomp(noStimTTestDFPerFB['pval'].to_numpy(), alpha=confidence_alpha)
        noStimTTestDFPerFB.loc[:, 'pval'] = pval
        noStimTTestDFPerFB.loc[:, 'reject'] = reject
    noStimTTestDFPerFB.to_hdf(resultPath, 'noStimTTestPerFB')
    #######
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
                'boxcox': scaledRaucDF.reset_index(drop=True),
                'boxcox_clipped': clippedScaledRaucDF.reset_index(drop=True),
                'raw_clipped': clippedRaucDF.reset_index(drop=True)
                }
            plotDF = pd.concat(plotDFsDict, names=['dataType'])
            plotDF.columns = plotDF.columns.get_level_values('feature')
            plotDF = plotDF.stack().reset_index()
            plotDF.columns = ['dataType', 'trial', 'feature', 'rauc']
            g = sns.displot(
                data=plotDF, col='dataType',
                x='rauc', hue='feature', kind='hist', element='step',
                facet_kws=dict(sharex=False),
                legend=False
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
