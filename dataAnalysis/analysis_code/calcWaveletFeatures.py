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
    --showFigures                          show plots at runtime? [default: False]
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
import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('PS')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
#
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import dataAnalysis.helperFunctions.pywt_helpers as pywthf
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
from tqdm import tqdm
import pywt
from docopt import docopt
from numpy.random import default_rng

import seaborn as sns
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

def calcCWT(
        partition, dataColNames=None, originalColumns=None, plotting=False,
        verbose=False, waveletList=None, pycwtKWs={}):
    # print('Process {} starting wavelet chunk'.format(os.getpid()))
    dataColMask = partition.columns.isin(dataColNames)
    partitionData = partition.loc[:, dataColMask]
    # print('Analyzing {}'.format(partition.loc[:, ~dataColMask]))
    if DEBUGGING:
        plotting = True
        #  Replace actual data with reference signal (noisy sine waves),
        #  to check that the bandpassing effect of the wavelets is
        #  correct
        dbf = 30  # Hz
        dbp = dbf ** -1  # sec
        t = partitionData.columns.to_numpy(dtype=float)
        refSignal = (
                np.sin(2 * np.pi * dbf * t) *
                (1 + t))
        refMask = (t < 0) | (t > 3 * dbp)
        refSignal[refMask] = 0
        for rIdx in range(partitionData.shape[0]):
            partitionData.iloc[rIdx, :] = (
                    refSignal +
                    0.3 * rng.standard_normal(size=t.size))
            # fig, ax = plt.subplots(); ax.plot(t, np.sin(2 * np.pi * dbf * t)); plt.show()
    #
    '''if True:
        pycwtKWs = {'sampling_period': 1e-3, 'axis': -1, 'precision': 14}'''
    outputList = []
    if plotting:
        fig, ax = plt.subplots()
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
        #
        coefDF.insert(0, 'scale', scalesIdx.reshape(-1))
        coefDF.insert(1, 'freqBandName', wvlDict['name'])
        coefDF.insert(2, 'center', wvlDict['center'])   # TODO reconcile with freq
        coefDF.insert(3, 'bandwidth', wvlDict['bandwidth'])
        coefDF.insert(4, 'waveletName', wvlDict['wavelet'].name)
        outputList.append(coefDF)
        if plotting:
            print(wvlDict['name'])
            plotData = pd.concat({
                'original': partitionData.reset_index(drop=True).stack().reset_index(),
                wvlDict['name']: coefDF.iloc[:, 5:].reset_index(drop=True).stack().reset_index()})
            plotData.columns = ['rowIdx', 'bin', 'signal']
            plotData.reset_index(inplace=True)
            plotData.columns = ['type', 'level_1', 'rowIdx', 'bin', 'signal']
            sns.lineplot(
                x='bin', y='signal', hue='type',
                data=plotData, ax=ax, errorbar='se', palette={
                    'original': 'b', 'low': 'r', 'beta': 'g', 'hi': 'c',
                    'spb': 'm'
                })
            ax.set_title(('{}_f_c={} Hz'.format(wvlDict['name'], wvlDict['center'])))
    if plotting:
        plt.show()
    allCoefDF = pd.concat(outputList).reset_index(drop=True)
    nNewFeatures = int(allCoefDF.shape[0] / partition.shape[0])
    newIndexEntries = pd.concat(
        [partition.loc[:, ~dataColMask] for i in range(nNewFeatures)]).reset_index(drop=True)
    result = pd.concat(
        [newIndexEntries, allCoefDF], axis=1)
    result.name = 'cwt'
    result.columns.name = partition.columns.name
    # print('Process {} finished wavelet chunk'.format(os.getpid()))
    return result

#
if __name__ == "__main__":
    rng = default_rng()
    arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("notebook")
    sns.set_style("darkgrid")
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
        transposeToColumns='bin', concatOn='index',
        getMetaData=essentialMetadataFields + ['xCoords', 'yCoords'],
        decimate=1, metaDataToCategories=False))

    DEBUGGING = False
    outputPath = os.path.join(
        alignSubFolder,
        blockBaseName + inputBlockSuffix + '_spectral_{}'.format(arguments['window']))
    #
    daskComputeOpts = dict(
        scheduler='processes',
        # scheduler='single-threaded',
        optimize_graph=False,
        )
    '''if daskComputeOpts['scheduler'] == 'single-threaded':
        daskClient = Client(LocalCluster(n_workers=1))
    elif daskComputeOpts['scheduler'] == 'processes':
        daskClient = Client(LocalCluster(processes=True))
    elif daskComputeOpts['scheduler'] == 'threads':
        daskClient = Client(LocalCluster(processes=False))
    else:
        daskClient = Client()
        print('Scheduler name is not correct!')'''
    ##
    #
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    # if True:
    dataDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    # dataDF.to_hdf(os.path.join(scratchFolder, 'temp.h5'), 'temp')
    # else:
    #     dataDF = pd.read_hdf(os.path.join(scratchFolder, 'temp.h5'), 'temp')
    dummySt = dataBlock.filter(
        objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
    fs = float(dummySt.sampling_rate)
    dt = fs ** -1
    scale = 100
    waveletList = []
    ###
    cwtOpts = dict(
        waveletList=waveletList,
        plotting=False,
        pycwtKWs=dict(
            sampling_period=dt, axis=-1,
            precision=14
            # method='conv',
        ))
    ###
    if arguments['plotting']:
        pdfPath = os.path.join(figureFolder, blockBaseName + 'wavelets.pdf')
        waveletReportPDF = PdfPages(pdfPath)
    for fBIdx, fBName in enumerate(freqBandsDict['name']):
        bandwidth = (freqBandsDict['hBound'][fBIdx] - freqBandsDict['lBound'][fBIdx]) / 6  # Hz
        center = (freqBandsDict['hBound'][fBIdx] + freqBandsDict['lBound'][fBIdx]) / 2  # Hz
        B = pywthf.bandwidthToMorletB(bandwidth, fs=fs, scale=scale)
        C = pywthf.centerToMorletC(center, fs=fs, scale=scale)
        minWidth = np.ceil((8 * scale * np.sqrt(B / 2) - 1) / scale)
        waveletName = 'cmor{:.6f}-{:.6f}'.format(B, C)
        wavelet = pywt.ContinuousWavelet(waveletName)
        wavelet.lower_bound = (-1) * minWidth / 2
        wavelet.upper_bound = minWidth / 2
        frequencies = pywt.scale2frequency(wavelet, [scale]) / dt
        if arguments['plotting']:
            # generate new wavelet to not interfere with the objects
            # being used for the calculations
            bws, fcs, bw_ratios, figsDict = pywthf.plotKernels(
                pywt.ContinuousWavelet(waveletName),
                np.asarray([scale / 2, scale]),
                dt=dt, precision=cwtOpts['pycwtKWs']['precision'], verbose=True,
                width=minWidth)
            scalesAx = figsDict['scales'][1][0, 0]
            scalesAx.set_title('{} ({} frequency: {} Hz to {} Hz)'.format(
                scalesAx.get_title(), fBName, freqBandsDict['lBound'][fBIdx], freqBandsDict['hBound'][fBIdx]))
            for axIdx in range(figsDict['scales'][1].shape[0]):
                ax = figsDict['scales'][1][axIdx, -1]
                ax.set_xlim([0, freqBandsDict['hBound'][fBIdx] * 3])
            waveletReportPDF.savefig(figsDict['scales'][0])
        waveletList.append({
            'name': fBName, 'center': center,
            'bandwidth': bandwidth, 'wavelet': wavelet,
            'scales': [scale], 'kernelDuration': (scale * minWidth + 1) * dt
            })
    waveletDF = pd.DataFrame(waveletList)
    if arguments['plotting']:
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
        waveletReportPDF.close()
    ##
    #
    groupBy = ['feature', 'pedalMovementCat', 'stimCat']
    spectralDF = ash.splitApplyCombine(
        dataDF, fun=calcCWT,
        newMetadataNames=[
            'scale', 'freqBandName', 'center',
            'bandwidth', 'waveletName'],
        sortIndexBy=['feature', 'lag', 'freqBandName', 'segment', 'originalIndex', 't'],
        funKWArgs=cwtOpts,
        rowKeys=groupBy, colKeys=None,
        daskProgBar=True, daskPersist=False,
        useDask=True, daskComputeOpts=daskComputeOpts
        )
    # newIndexCols = ['scale', 'freqBandName', 'center', 'bandwidth', 'waveletName']
    # spectralDF.set_index(newIndexCols, append=True, inplace=True)
    # trim edges based on largest kernel size (to 100s of msec)
    spectralDF.columns = spectralDF.columns.astype(float)
    trimEdgesBy = np.ceil(waveletDF['kernelDuration'].max() / 2 * 1e1) / 1e1
    trimMask = (
        (spectralDF.columns >= (spectralDF.columns[0] + trimEdgesBy)) &
        (spectralDF.columns < (spectralDF.columns[-1] - trimEdgesBy)))
    spectralDF = spectralDF.loc[:, trimMask]
    #
    tBins = np.unique(spectralDF.columns.get_level_values('bin'))
    featNames = spectralDF.index.get_level_values('feature')
    featNamesClean = featNames.str.replace('#0', '')
    fbNames = spectralDF.index.get_level_values('freqBandName')
    newFeatNames = ['{}_{}'.format(featNamesClean[i], fbNames[i]) for i in range(featNames.shape[0])]
    spectralDF.index = spectralDF.index.droplevel('feature')
    spectralDF.loc[:, 'parentFeature'] = featNames
    spectralDF.loc[:, 'feature'] = newFeatNames
    spectralDF.set_index(['parentFeature', 'feature'], inplace=True, append=True)
    trialTimes = np.unique(spectralDF.index.get_level_values('t'))
    #
    spikeTrainMeta = {
        'units': pq.s,
        'wvfUnits': pq.dimensionless,
        'left_sweep': (-1) * tBins[0] * pq.s,
        # 't_start': min(0, trialTimes[0]) * pq.s,
        't_start': min(trialTimes) * pq.s,
        't_stop': trialTimes[-1] * pq.s,
        'sampling_rate': dummySt.sampling_rate
        }
    # pdb.set_trace()
    masterBlock = ns5.alignedAsigDFtoSpikeTrain(
        spectralDF, spikeTrainMeta=spikeTrainMeta,
        matchSamplingRate=False, verbose=arguments['verbose'])
    if arguments['lazy']:
        dataReader.file.close()
    #
    masterBlock = ns5.purgeNixAnn(masterBlock)
    if os.path.exists(outputPath + '.nix'):
        os.remove(outputPath + '.nix')
    print('Writing {}.nix...'.format(outputPath))
    writer = ns5.NixIO(filename=outputPath + '.nix', mode='ow')
    writer.write_block(masterBlock, use_obj_names=True)
    writer.close()
