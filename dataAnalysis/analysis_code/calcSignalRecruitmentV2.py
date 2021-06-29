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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output by default
matplotlib.use('QT5Agg')   # generate interactive output
import seaborn as sns
import pdb
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
    'processAll': True, 'blockIdx': '2', 'selectionName': 'lfp_CAR_spectral_fa_mahal',
    'lazy': False, 'verbose': False, 'window': 'XL', 'alignQuery': 'starting', 'datasetName': 'Block_XL_df_f',
    'analysisName': 'default', 'exp': 'exp202101281100', 'selector': None, 'winStart': '200', 'alignFolderName': 'motion',
    'winStop': '400', 'loadFromFrames': True, 'maskOutlierBlocks': False, 'unitQuery': 'fr_sqrt', 'inputBlockPrefix': 'Block',
    'inputBlockSuffix': 'lfp_CAR_spectral_mahal', 'iteratorSuffix': 'f', 'plotting': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#


def calcRauc(
        partition, dataColNames=None):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    partitionMeta = partition.loc[:, ~dataColMask]
    result = (partitionData - partitionData.mean()).abs().mean()
    resultDF = pd.concat(
        [result, partitionMeta.iloc[0, :]])
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
    #  Overrides
    limitPages = None
    funKWargs = dict(
        # baseline='mean',
        tStart=-100e-3, tStop=200e-3)
    #  End Overrides
    if not arguments['loadFromFrames']:
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
    else:
        # loading from dataframe
        datasetName = arguments['datasetName']
        selectionName = arguments['selectionName']
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
    groupNames = ['originalIndex', 'segment', 't']
    daskComputeOpts = dict(
        scheduler='processes'
        # scheduler='single-threaded'
        )
    '''if daskComputeOpts['scheduler'] == 'single-threaded':
        daskClient = Client(LocalCluster(n_workers=1))
    elif daskComputeOpts['scheduler'] == 'processes':
        daskClient = Client(LocalCluster(processes=True))
        # daskClient = Client(LocalCluster())
    elif daskComputeOpts['scheduler'] == 'threads':
        daskClient = Client(LocalCluster(processes=False))
    else:
        print('Scheduler name is not correct!')
        daskClient = Client()'''
    colFeatureInfo = [nm for nm in dataDF.columns.names if nm != 'feature']
    rawRaucDF = ash.splitApplyCombine(
        dataDF, fun=calcRauc, resultPath=resultPath,
        rowKeys=groupNames, colKeys=colFeatureInfo,
        daskProgBar=False,
        daskPersist=True, useDask=True, daskComputeOpts=daskComputeOpts)
    rawRaucDF.index = rawRaucDF.index.droplevel('bin')
    #
    qScaler = PowerTransformer()
    qScaler.fit(rawRaucDF)
    scaledRaucDF = pd.DataFrame(
        qScaler.transform(rawRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
    ##
    sdThresh = 2.5
    clippedScaledRaucDF = scaledRaucDF.clip(upper=sdThresh, lower=-sdThresh)
    clippedRaucDF = pd.DataFrame(
        qScaler.inverse_transform(clippedScaledRaucDF),
        index=rawRaucDF.index,
        columns=rawRaucDF.columns)
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

    clippedRaucDF.to_hdf(resultPath, 'raw')
    clippedScaledRaucDF.to_hdf(resultPath, 'scaled')
    normalizedRaucDF.to_hdf(resultPath, 'normalized')
    scaledNormalizedRaucDF.to_hdf(resultPath, 'scaled_normalized')
    scalers.to_hdf(resultPath, 'scalers')
    #
    splitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'])
    iteratorKWArgs = dict(
        n_splits=7,
        splitterClass=tdr.trialAwareStratifiedKFold, splitterKWArgs=splitterKWArgs,
        samplerKWArgs=dict(random_state=None, test_size=None,),
        prelimSplitterClass=tdr.trialAwareStratifiedKFold, prelimSplitterKWArgs=splitterKWArgs,
        resamplerClass=RandomOverSampler, resamplerKWArgs={},
        # resamplerClass=None, resamplerKWArgs={},
        )
    cvIterator = tdr.trainTestValidationSplitter(
        dataDF=clippedRaucDF, **iteratorKWArgs)
    iteratorInfo = {
        'iteratorKWArgs': iteratorKWArgs,
        'cvIterator': cvIterator
        }
    with open(iteratorPath, 'wb') as _f:
        pickle.dump(iteratorInfo, _f)
    if arguments['plotting']:
        plotDFsDict = {
            'raw': rawRaucDF.reset_index(drop=True),
            'scaled': scaledRaucDF.reset_index(drop=True),
            'clippedScaled': clippedScaledRaucDF.reset_index(drop=True),
            'clipped': clippedRaucDF.reset_index(drop=True)
            }
        plotDF = pd.concat(plotDFsDict, names=['dataType'])
        plotDF.columns = plotDF.columns.get_level_values('feature')
        plotDF = plotDF.stack().reset_index()
        plotDF.columns = ['dataType', 'trial', 'feature', 'rauc']
        g = sns.displot(
            data=plotDF, col='dataType',
            x='rauc', hue='feature', kind='hist', element='step'
            )

    if arguments['lazy']:
        dataReader.file.close()
