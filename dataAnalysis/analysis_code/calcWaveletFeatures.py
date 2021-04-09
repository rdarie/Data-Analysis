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
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt

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
        fs=None, dt=None, freqBandsDict=None,
        scale=20,
        verbose=False):
    if dt is None:
        dt = fs ** (-1)
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    # frequencies = pywt.scale2frequency('cmor1.5-1.0', [1, 2, 3, 4]) / dt
    pdb.set_trace()
    for fBIdx, fBName in enumerate(freqBandsDict['name']):
        bandwidth = (freqBandsDict['hBound'][fBIdx] - freqBandsDict['lBound'][fBIdx]) / 2  # Hz
        center = (freqBandsDict['hBound'][fBIdx] + freqBandsDict['lBound'][fBIdx]) / 2 # Hz
        B = (fs / (2 * np.pi * scale * bandwidth)) ** 2
        C = (center * scale) / fs
    pywt.cwt
    result = None
    '''result = pd.DataFrame(
        xxx,
        index=partition.index, columns=xxx)
    result = pd.concat(
        [result, partition.loc[:, ~dataColMask]],
        axis=1)
    result.name = 'cwt' '''
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
    freqBands = pd.DataFrame(freqBandsDict)
    # dataDF.columns = dataDF.columns.droplevel('lag')
    # trialGroupByNames = dataDF.index.droplevel('bin').names
    dummySt = dataBlock.filter(
        objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
    fs = float(dummySt.sampling_rate)
    daskComputeOpts = dict(
            # scheduler='processes'
            scheduler='single-threaded'
            )
    cwtOpts = dict(
        freqBandsDict=freqBandsDict, fs=fs)
    daskClient = Client()
    # print(daskClient.scheduler_info()['services'])
    spectralDF = ash.splitApplyCombine(
        dataDF, fun=calcCWT,
        funKWArgs=cwtOpts,
        rowKeys=['feature'], colKeys=None,
        daskProgBar=False,
        daskPersist=True, useDask=True, reindexFromInput=False,
        daskComputeOpts=daskComputeOpts
        )

'''theseSpectralFeatList = []
for featName in dataDF.columns:
    # dataSrs = dataDF.loc[:, featName]
    if arguments['verbose']:
        print('on feature {}'.format(featName))
    thisSpectrogram = getSpectrogram(
        dataDF.loc[:, featName],
        trialGroupByNames=trialGroupByNames,
        fs=fs, **spectralFeatureOpts)
    fBands = thisSpectrogram.columns
    for rIdx, row in freqBands.iterrows():
        thisMask = (fBands >= row['lBound']) & (fBands < row['hBound'])
        if thisMask.any():
            thisFeat = thisSpectrogram.loc[:, thisMask].mean(axis='columns')
            thisFeat.name = '{}_{}'.format(featName[:-2], row['name'])
            theseSpectralFeatList.append(thisFeat)

spectralDF = pd.concat(theseSpectralFeatList, axis='columns')
spectralDF.columns.name = 'feature'

tBins = np.unique(spectralDF.index.get_level_values('bin'))
# pdb.set_trace()
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
'''
