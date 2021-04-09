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
from tqdm import tqdm
import pywt
try:
    import libtfr
    HASLIBTFR = True
except Exception:
    import scipy.signal
    HASLIBTFR = False

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
    transposeToColumns='feature', concatOn='columns',
    getMetaData=essentialMetadataFields,
    decimate=1))

if HASLIBTFR:
    def pSpec(
            data, tStart, winLen, winLen_samp,
            stepLen, stepLen_samp,
            fs, fr, fr_samp, fr_start_idx, fr_stop_idx, NFFT,
            nw, nTapers, indexTName, annCols=None,
            annValues=None):
        nSamples = data.shape[0]
        nWindows = m.floor((nSamples - winLen_samp + 1) / stepLen_samp)
        t = tStart + winLen + np.arange(nWindows) * stepLen
        # generate a transform object with size equal to signal length and ntapers tapers
        D = libtfr.mfft_dpss(NFFT, nw, nTapers, winLen_samp)
        P_libtfr = D.mtspec(data, stepLen_samp).transpose()
        P_libtfr = P_libtfr[:nWindows, fr_start_idx:fr_stop_idx]
        #
        # TODO: specIndex = pd.Multiindex, 'bin' ,etc.
        if annCols is not None:
            indexFrame = pd.DataFrame(np.nan, index=range(t.size), columns=[indexTName] + annCols)
            for annIdx, annNm in enumerate(annCols):
                indexFrame.loc[:, annNm] = annValues[annIdx]
            indexFrame.loc[:, indexTName] = t
            thisIndex = pd.MultiIndex.from_frame(indexFrame)
        else:
            thisIndex = pd.Index(t, name=indexTName)
        spectrum = pd.DataFrame(P_libtfr, index=thisIndex, columns=fr)
        spectrum.columns.name = 'feature'
        spectrum.origin = 'libtfr'
        return spectrum
else:
    def pSpec(
            data, tStart, winLen, winLen_samp,
            stepLen, stepLen_samp,
            fs, fr, fr_samp, fr_start_idx, fr_stop_idx, NFFT,
            nw, nTapers, indexTName, annCols=None,
            annValues=None):
        # nSamples = data.shape[0]
        # nWindows = m.floor((nSamples - NFFT + 1) / stepLen_samp)
        # spectrum = np.zeros((nWindows, fr_samp))
        overlap_samp = NFFT - stepLen_samp
        _, t, P_scipy = scipy.signal.spectrogram(
            data, mode='magnitude',
            window='boxcar', nperseg=NFFT, noverlap=overlap_samp, fs=fs)
        t += (tStart + winLen)
        P_scipy = P_scipy.transpose()[np.newaxis, :, fr_start_idx:fr_stop_idx]
        if annCols is not None:
            indexFrame = pd.DataFrame(np.nan, index=range(t.size), columns=[indexTName] + annCols)
            for annIdx, annNm in enumerate(annCols):
                indexFrame.loc[:, annNm] = annValues[annIdx]
            indexFrame.loc[:, indexTName] = t
            thisIndex = pd.MultiIndex.from_frame(indexFrame)
        else:
            thisIndex = pd.Index(t, name=indexTName)
        spectrum = pd.DataFrame(P_scipy, index=thisIndex, columns=fr)
        spectrum.columns.name = 'feature'
        spectrum.origin = 'scipy'
        return spectrum


def getSpectrogram(
        dataSrs,
        trialGroupByNames=None,
        winLen=None, stepLen=0.02, R=20,
        fs=None, fStart=None, fStop=None, progBar=True):
    # t = np.asarray(dataSrs.index)
    # delta = 1 / fs
    winLen_samp = int(winLen * fs)
    stepLen_samp = int(stepLen * fs)
    NFFT = hf.nextpowof2(winLen_samp)
    nw = winLen * R  # time bandwidth product based on 0.1 sec windows and 200 Hz bandwidth
    nTapers = m.ceil(nw / 2)  # L < nw - 1
    fr_samp = int(NFFT / 2) + 1
    fr = (1 + np.arange(fr_samp)) * fs / (2 * fr_samp)
    #
    if fStart is not None:
        fr_start_idx = np.where(fr > fStart)[0][0]
    else:
        fr_start_idx = 0
    #
    if fStop is not None:
        fr_stop_idx = np.where(fr < fStop)[0][-1]
    else:
        fr_stop_idx = -1
    #
    fr = fr[fr_start_idx: fr_stop_idx]
    fr_samp = len(fr)
    spectra = []
    indexTName = 'bin'
    if progBar:
        pBar = tqdm(total=dataSrs.groupby(trialGroupByNames).ngroups)
    for name, group in dataSrs.groupby(trialGroupByNames):
        tStart = group.index.get_level_values(indexTName)[0]
        # pdb.set_trace()
        ###
        # wav = pywt.ContinuousWavelet('cmor1.5-1.0')
        # dt = fs ** (-1)
        # scales = [1, 2, 3, 4, 10, 15]
        # coef, freqs = pywt.cwt(group.to_numpy(), scales, wav, sampling_period=dt)
        ###
        spectra.append(pSpec(
            group.to_numpy(), tStart, winLen, winLen_samp,
            stepLen, stepLen_samp,
            fs, fr, fr_samp, fr_start_idx, fr_stop_idx,
            NFFT, nw, nTapers,
            indexTName, annCols=trialGroupByNames, annValues=name))
        if progBar:
            pBar.update(1)
    return pd.concat(spectra)

outputPath = os.path.join(
    alignSubFolder,
    blockBaseName + inputBlockSuffix + '_spectral_{}'.format(arguments['window']))
#
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
#
alignedAsigsDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
freqBands = pd.DataFrame(freqBandsDict)
alignedAsigsDF.columns = alignedAsigsDF.columns.droplevel('lag')
trialGroupByNames = alignedAsigsDF.index.droplevel('bin').names
dummySt = dataBlock.filter(
    objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
fs = float(dummySt.sampling_rate)
theseSpectralFeatList = []
for featName in alignedAsigsDF.columns:
    # dataSrs = alignedAsigsDF.loc[:, featName]
    if arguments['verbose']:
        print('on feature {}'.format(featName))
    thisSpectrogram = getSpectrogram(
        alignedAsigsDF.loc[:, featName],
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
