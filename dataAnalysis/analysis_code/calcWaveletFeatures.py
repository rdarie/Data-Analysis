"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --profile                              print time and mem diagnostics? [default: False]
    --plotting                             display plots [default: False]
    --showNow                              show plots at runtime? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: short]
    --winStart=winStart                    start of window
    --winStop=winStop                      end of window
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
"""
#
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from dataAnalysis.analysis_code.namedQueries import namedQueries
import os
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle
import math as m
import quantities as pq
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from tqdm import tqdm
import pywt
import dataAnalysis.helperFunctions.pywt_helpers as pywthf
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import matplotlib.pyplot as plt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'window': 'L', 'analysisName': 'hiRes', 'lazy': True, 'inputBlockSuffix': 'lfp_CAR',
        'blockIdx': '3', 'verbose': True, 'processAll': False, 'unitQuery': 'lfp',
        'winStop': '1300', 'profile': False, 'alignQuery': 'starting', 'exp': 'exp202101201100',
        'alignFolderName': 'motion', 'inputBlockPrefix': 'Block', 'winStart': '300'}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
'''
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

def calcCWT(
        partition, dataColNames=None,
        verbose=False, waveletList=None, pycwtKWs={}):
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    outputList = []
    for wvlDict in waveletList:
        coefs, freq = pywt.cwt(
            partitionData, wvlDict['scales'],
            wvlDict['wavelet'], **pycwtKWs)
        scalesIdx = np.asarray([
            [wvlDict['scales'][j] for i in range(coefs.shape[1])]
            for j in range(coefs.shape[0])])
        coefDF = pd.DataFrame(
            np.abs(coefs).reshape(-1, partitionData.shape[1]),
            columns=partitionData.columns)
        coefDF.insert(0, 'scale', scalesIdx.reshape(-1))
        coefDF.insert(1, 'freqBandName', wvlDict['name'])
        coefDF.insert(2, 'center', wvlDict['center'])  # TODO reconcile with freq
        coefDF.insert(3, 'bandwidth', wvlDict['bandwidth'])
        coefDF.insert(4, 'waveletName', wvlDict['wavelet'].name)
        outputList.append(coefDF)
    # pdb.set_trace()
    allCoefDF = pd.concat(outputList).reset_index(drop=True)
    nNewFeatures = int(allCoefDF.shape[0] / partition.shape[0])
    newIndexEntries = pd.concat(
        [partition.loc[:, ~dataColMask] for i in range(nNewFeatures)]).reset_index(drop=True)
    result = pd.concat(
        [
            newIndexEntries,
            allCoefDF],
        axis=1)
    result.name = 'cwt'
    return result

outputPath = os.path.join(
    alignSubFolder,
    blockBaseName + inputBlockSuffix + '_spectral_{}'.format(arguments['window']))
#
if __name__ == "__main__":
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    #
    dataDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    dummySt = dataBlock.filter(
        objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
    fs = float(dummySt.sampling_rate)
    dt = fs ** -1
    scale = 50
    waveletList = []
    for fBIdx, fBName in enumerate(freqBandsDict['name']):
        bandwidth = (freqBandsDict['hBound'][fBIdx] - freqBandsDict['lBound'][fBIdx]) / 6  # Hz
        center = (freqBandsDict['hBound'][fBIdx] + freqBandsDict['lBound'][fBIdx]) / 2 # Hz
        B = pywthf.bandwidthToMorletB(bandwidth, fs=fs, scale=scale)
        C = pywthf.centerToMorletC(center, fs=fs, scale=scale)
        minWidth = np.ceil((12 * scale * np.sqrt(B / 2) - 1) / scale)
        waveletName = 'cmor{:.3f}-{:.3f}'.format(B, C)
        wavelet = pywt.ContinuousWavelet(waveletName)
        frequencies = pywt.scale2frequency(wavelet, [scale]) / dt
        if arguments['plotting']:
            bws, fcs, bw_ratios, figsDict = pywthf.plotKernels(
                pywt.ContinuousWavelet(waveletName),
                np.asarray([scale / 2, scale]),
                dt=dt, precision=16, verbose=True,
                width=minWidth)
        if arguments['showNow']:
            plt.show()
        else:
            plt.close()
        waveletList.append({
            'name': fBName, 'center': center,
            'bandwidth': bandwidth, 'wavelet': wavelet,
            'scales': [scale]
        })
    # dataDF.columns = dataDF.columns.droplevel('lag')
    # trialGroupByNames = dataDF.index.droplevel('bin').names
    daskComputeOpts = dict(
            # scheduler='processes'
            scheduler='single-threaded'
            )
    cwtOpts = dict(
        waveletList=waveletList,
        pycwtKWs=dict(sampling_period=dt, axis=-1))
    # daskClient = Client()
    # print(daskClient.scheduler_info()['services'])
    spectralDF = ash.splitApplyCombine(
        dataDF, fun=calcCWT,
        funKWArgs=cwtOpts,
        rowKeys=['feature'], colKeys=None,
        daskProgBar=True,
        daskPersist=True,
        useDask=True, reindexFromInput=False,
        daskComputeOpts=daskComputeOpts
        )
    newIndexCols = ['scale', 'freqBandName', 'center', 'bandwidth', 'waveletName']
    spectralDF.set_index(newIndexCols, append=True, inplace=True)
    spectralDF.columns = spectralDF.columns.astype(np.float)
    tBins = np.unique(spectralDF.columns.get_level_values('bin'))
    featNames = spectralDF.index.get_level_values('feature')
    featNamesClean = featNames.str.replace('#0', '')
    fbNames = spectralDF.index.get_level_values('freqBandName')
    newFeatNames = ['{}_{}'.format(featNamesClean[i], fbNames[i]) for i in range(featNames.shape[0])]
    spectralDF.index = spectralDF.index.droplevel('feature')
    # pdb.set_trace()
    spectralDF.loc[:, 'parentFeature'] = featNames
    spectralDF.loc[:, 'feature'] = newFeatNames
    spectralDF.set_index(['parentFeature', 'feature'], inplace=True)
    trialTimes = np.unique(spectralDF.index.get_level_values('t'))
    spikeTrainMeta = {
        'units': pq.s,
        'wvfUnits': pq.dimensionless,
        'left_sweep': (-1) * tBins[0] * pq.s,
        't_start': min(0, trialTimes[0]) * pq.s,
        't_stop': trialTimes[-1] * pq.s,
        'sampling_rate': ((tBins[1] - tBins[0]) ** (-1)) * pq.Hz
    }
    masterBlock = ns5.alignedAsigDFtoSpikeTrain(
        spectralDF, spikeTrainMeta=spikeTrainMeta, matchSamplingRate=False)
    if arguments['lazy']:
        dataReader.file.close()
    masterBlock = ns5.purgeNixAnn(masterBlock)
    if os.path.exists(outputPath + '.nix'):
        os.remove(outputPath + '.nix')
    print('Writing {}.nix...'.format(outputPath))
    writer = ns5.NixIO(filename=outputPath + '.nix', mode='ow')
    writer.write_block(masterBlock, use_obj_names=True)
    writer.close()
