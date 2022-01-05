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
    result = detrended.abs().max()
    resultDF = pd.concat([partitionMeta.iloc[0, :], result]).to_frame(name=partition.index[0]).T
    resultDF.name = 'rauc'
    return resultDF


if __name__ == "__main__":
    rejectOutliersAnyway = True
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
        tStart=-100e-3, tStop=600e-3,
        maskBaseline=False,
        baselineTStart=-300e-3, baselineTStop=-150e-3)
    #  End Overrides
    datasetName = arguments['datasetName']
    selectionName = arguments['selectionName']
    pdfPath = os.path.join(
        figureOutputFolder,
        blockBaseName + '_{}_{}_{}_rauc_outlier_deletion.pdf'.format(
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
    if rejectOutliersAnyway:
        if 'isOutlierTrial' in dataDF.index.names:
            dataDF = dataDF.xs(False, level='isOutlierTrial')
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
    detrendAsGroup = False
    if detrendAsGroup:
        tMask = ((trialInfo['bin'] >= funKWArgs['baselineTStart']) & (trialInfo['bin'] < funKWArgs['baselineTStop']))
        baseline = dataDF.loc[tMask.to_numpy(), :].median()
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
    del dataDF
    rawRaucDF.columns = colFeatureInfo
    rawRaucDF.index = rawRaucDF.index.droplevel('bin')
    raucTrialInfo = rawRaucDF.index.to_frame().reset_index(drop=True)
    compoundAnnDescr = {
        'stimCondition': ['electrode', 'trialRateInHz', ],
        'kinematicCondition': ['pedalDirection', 'pedalMovementCat']
        }
    compoundAnnLookup = []
    for canName, can in compoundAnnDescr.items():
        compoundAnn = pd.Series(np.nan, index=raucTrialInfo.index)
        for name, group in raucTrialInfo.groupby(can):
            compoundAnn.loc[group.index] = '_'.join(['{}'.format(nm) for nm in name])
        raucTrialInfo.loc[:, canName] = compoundAnn
        thisCAL = raucTrialInfo.loc[:, can + [canName]].drop_duplicates().set_index(canName)
        thisCAL.index.name = 'compoundAnnotation'
        compoundAnnLookup.append(thisCAL)
    compoundAnnLookupDF = pd.concat(compoundAnnLookup)
    rawRaucDF.index = pd.MultiIndex.from_frame(raucTrialInfo)
    #
    raucForExport = rawRaucDF.xs('main', level='controlFlag').mean(axis=1).sort_values(ascending=False).to_frame(name='average_rauc')
    raucForExportTrialInfo = raucForExport.index.to_frame().reset_index(drop=True)
    raucForExportTrialInfo = raucForExportTrialInfo.loc[:, ['expName', 'segment', 'originalIndex', 't', 'stimCondition', 'kinematicCondition']]
    raucForExport.index = pd.MultiIndex.from_frame(raucForExportTrialInfo)
    raucForExport.to_html(htmlOutputPath)
    #
    if arguments['plotting']:
        with PdfPages(pdfPath) as pdf:
            plotDF = rawRaucDF.reset_index(drop=True)
            plotDF.columns = plotDF.columns.get_level_values('feature')
            plotDF = plotDF.stack().reset_index()
            plotDF.columns = ['trial', 'feature', 'rauc']
            g = sns.displot(
                data=plotDF,
                x='rauc', hue='feature', kind='hist', element='step',
                legend=False
                )
            pdf.savefig(
                bbox_inches='tight',
                )
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
    #
    markForDeletion = pd.Series(False, index=rawRaucDF.index)
    for name, group in rawRaucDF.groupby('freqBandName', axis='columns'):
        flatGroup = group.to_numpy().flatten()
        groupRange = flatGroup.max() - flatGroup.min()
        print('flatGroup.max() = {}, range = {}'.format(flatGroup.max(), groupRange))
        trimMask = rawRaucDF.loc[:, group.columns] > (flatGroup.max() - groupRange * 1e-2)
        markForDeletion.loc[trimMask.index[trimMask.any(axis='columns')]] = True
    print('Mark for deletion: {}/{} trials'.format(markForDeletion.sum(), markForDeletion.shape[0]))
    dataTI = pd.MultiIndex.from_frame(trialInfo.loc[:, ['segment', 'originalIndex', 't']])
    deletionMap = (~markForDeletion).to_frame(name='markForDelete').reset_index().loc[:, ['segment', 'originalIndex', 't', 'markForDelete']].set_index(['segment', 'originalIndex', 't'])['markForDelete']
    deletionMask = dataTI.map(deletionMap)
    applyToSelections = [
        'laplace_baseline', 'laplace_scaled', 'laplace_scaled_mahal_ledoit',
        'laplace_spectral_scaled_mahal_ledoit', 'laplace_spectral_baseline', 'laplace_spectral_scaled']
    for selName in applyToSelections:
        thisDataDF = pd.read_hdf(datasetPath, '/{}/data'.format(selName))
        if rejectOutliersAnyway:
            if 'isOutlierTrial' in thisDataDF.index.names:
                thisDataDF = thisDataDF.xs(False, level='isOutlierTrial')
        thisDataDF = thisDataDF.loc[deletionMask.to_numpy(), :]
        thisDataDF.to_hdf(datasetPath, '/{}/data'.format(selName))
    #############
    print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
