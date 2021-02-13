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

try:
    import libtfr
    HASLIBTFR = True
except Exception:
    import scipy.signal
    HASLIBTFR = False

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
    transposeToColumns='feature', concatOn='columns',
    getMetaData=True, decimate=1))

if HASLIBTFR:
    def pSpec(
            data, stepLen, stepLen_samp, fr_start_idx, fr_stop_idx, NFFT,
            fs, fr, fr_samp, nw, nTapers, indexTName, annCols=None,
            annValues=None):
        nSamples = data.shape[0]
        nWindows = m.floor((nSamples - NFFT + 1) / stepLen_samp)
        t = np.arange(nWindows + 1) * stepLen + NFFT / fs * 0.5
        spectrum = np.zeros((nWindows + 1, fr_samp))
        # generate a transform object with size equal to signal length and ntapers tapers
        D = libtfr.mfft_dpss(NFFT, nw, nTapers, NFFT)
        P_libtfr = D.mtspec(data, stepLen_samp).transpose()
        P_libtfr = P_libtfr[:, fr_start_idx:fr_stop_idx]
        #
        # TODO: specIndex = pd.Multiindex, 'bin' ,etc.
        spectrum = pd.DataFrame(P_libtfr, index=t, columns=fr)
        spectrum.columns.name = 'feature'
        spectrum.origin = 'libtfr'
        return spectrum
else:
    def pSpec(
            data, stepLen, stepLen_samp, fr_start_idx, fr_stop_idx, NFFT,
            fs, fr, fr_samp, nw, nTapers, indexTName, annCols=None,
            annValues=None):
        nSamples = data.shape[0]
        nWindows = m.floor((nSamples - NFFT + 1) / stepLen_samp)
        spectrum = np.zeros((nWindows, fr_samp))
        overlap_samp = NFFT - stepLen_samp
        _, t, P_scipy = scipy.signal.spectrogram(
            data, mode='magnitude',
            window='boxcar', nperseg=NFFT, noverlap=overlap_samp, fs=fs)
        P_scipy = P_scipy.transpose()[np.newaxis, :, fr_start_idx:fr_stop_idx]
        spectrum = pd.DataFrame(P_scipy, index=t, columns=fr)
        spectrum.columns.name = 'feature'
        spectrum.origin = 'scipy'
        return spectrum


def getSpectrogram(
        dataSrs,
        winLen=None, stepLen=0.02, R=20,
        fs=None, fStart=None, fStop=None):
    # t = np.asarray(dataSrs.index)
    # delta = 1 / fs
    winLen_samp = int(winLen * fs)
    stepLen_samp = int(stepLen * fs)
    NFFT = hf.nextpowof2(winLen_samp)
    nw = winLen * R  # time bandwidth product based on 0.1 sec windows and 200 Hz bandwidth
    nTapers = m.ceil(nw / 2)  # L < nw - 1
    fr_samp = int(NFFT / 2) + 1
    fr = np.arange(fr_samp) * fs / (2 * fr_samp)
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
    trialGroupByNames = ['segment', 'originalIndex', 't']
    indexTName = 'bin'
    for name, group in dataSrs.groupby(trialGroupByNames):
        spectra.append(pSpec(
            group.to_numpy(), stepLen, stepLen_samp, fr_start_idx, fr_stop_idx,
            NFFT, fs, fr, fr_samp, nw, nTapers,
            indexTName, annCols=trialGroupByNames, annValues=name))
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
dummySt = dataBlock.filter(
    objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
fs = float(dummySt.sampling_rate)
bla = getSpectrogram(
    alignedAsigsDF.iloc[:, 1],
    winLen=.1, stepLen=0.02, R=20,
    fs=fs, fStart=None, fStop=None)
masterBlock = ns5.alignedAsigDFtoSpikeTrain(spectralDF, dataBlock)
if arguments['lazy']:
    dataReader.file.close()
masterBlock = ns5.purgeNixAnn(masterBlock)
print('Writing {}.nix...'.format(outputPath))
writer = ns5.NixIO(filename=outputPath + '.nix')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
