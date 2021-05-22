"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --profile                              print time and mem diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: short]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --substituteOneChannel                 correct for rank defficiency by using one unsubtracted chan [default: False]
"""
#
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from namedQueries import namedQueries
import os
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle
import math as m
import quantities as pq
from dask.distributed import Client, LocalCluster

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))

alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['verbose'] = arguments['verbose']
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    # transposeToColumns='feature', concatOn='columns',
    transposeToColumns='bin', concatOn='index',
    getMetaData=essentialMetadataFields + ['xCoords', 'yCoords'],
    decimate=1))

daskComputeOpts = dict(
    # scheduler='threads'
    # scheduler='processes'
    scheduler='single-threaded'
    )
daskOpts = dict(
    useDask=True,
    daskPersist=True, daskProgBar=True,
    daskResultMeta=None, daskComputeOpts=daskComputeOpts,
    reindexFromInput=True)

outputPath = os.path.join(
    alignSubFolder,
    blockBaseName + inputBlockSuffix + '_CAR_{}'.format(arguments['window']))
#
groupBy = ['segment', 'originalIndex', 't']
funKWArgs = {
    'substituteOneChannel': arguments['substituteOneChannel'],
    'substituteChannelLabel': 'utah8#0', 'subtractWhat': 'median'}
#
def rereference(
        group, dataColNames=None, subtractWhat='mean',
        substituteOneChannel=False, substituteChannelLabel=None,
        verbose=False):
    dataColMask = group.columns.isin(dataColNames)
    groupData = group.loc[:, dataColMask]
    indexColMask = ~group.columns.isin(dataColNames)
    indexCols = group.columns[indexColMask]
    if substituteOneChannel:
        if group.loc[:, 'feature'].unique()[0] == 'foo':
            subChanLoc = group.index[0]
        elif substituteChannelLabel is None:
            subChanLoc = group.index[0]
        else:
            firstSrsMask = group.loc[:, 'feature'] == substituteChannelLabel
            subChanLoc = group.index[firstSrsMask]
            assert subChanLoc.size == 1
        saveFirstSrs = groupData.loc[subChanLoc, :].copy()
    if subtractWhat == 'mean':
        referenceSignal = groupData.mean(axis='index')
    elif subtractWhat == 'median':
        referenceSignal = groupData.median(axis='index')
    resData = groupData.sub(referenceSignal, axis='columns')
    if substituteOneChannel:
        # implemented based on Milekovic, ..., Brochier 2015
        # check that it works!
        resData.loc[subChanLoc, :] = referenceSignal.to_numpy()
    resDF = pd.concat(
        [group.loc[:, indexCols], resData],
        axis='columns')
    return resDF


if __name__ == "__main__":
    if daskOpts['daskComputeOpts']['scheduler'] == 'single-threaded':
        daskClient = Client(LocalCluster(n_workers=1))
    elif daskOpts['daskComputeOpts']['scheduler'] == 'processes':
        daskClient = Client(LocalCluster(processes=True))
    elif daskOpts['daskComputeOpts']['scheduler'] == 'threads':
        daskClient = Client(LocalCluster(processes=False))
    else:
        daskClient = None
        print('Scheduler name is not correct!')
    #
    if arguments['verbose']:
        print('calcRereferencedTriggered() loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    dataDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    #
    # dataDF.columns = dataDF.columns.droplevel('lag')
    dummySt = dataBlock.filter(
        objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
    fs = float(dummySt.sampling_rate)
    #
    rerefDF = ash.splitApplyCombine(
        dataDF,
        fun=rereference, resultPath=outputPath,
        funArgs=[], funKWArgs=funKWArgs,
        rowKeys=groupBy, colKeys=None, **daskOpts)
    #
    del dataDF
    masterBlock = ns5.alignedAsigDFtoSpikeTrain(
        rerefDF, dataBlock=dataBlock, matchSamplingRate=True)
    masterBlock = ns5.purgeNixAnn(masterBlock)
    if os.path.exists(outputPath + '.nix'):
        os.remove(outputPath + '.nix')
    print('Writing {}.nix...'.format(outputPath))
    writer = ns5.NixIO(
        filename=outputPath + '.nix', mode='ow')
    writer.write_block(masterBlock, use_obj_names=True)
    writer.close()
    if arguments['lazy']:
        dataReader.file.close()
