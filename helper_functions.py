import pdb
from brpylib             import NsxFile, brpylib_ver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
import sys
import libtfr

def getNSxData(filePath, elecIds, startTime_s, dataLength_s, downsample = 1):
    # Version control
    brpylib_ver_req = "1.3.1"
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

    # Open file and extract headers
    nsx_file = NsxFile(filePath)

    # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
    channelData = nsx_file.getdata(elecIds, startTime_s, dataLength_s, downsample)

    # Close the nsx file now that all data is out
    nsx_file.close()

    return channelData, nsx_file.basic_header, nsx_file.extended_headers

def getBadDataMask(channelData, ExtendedHeaders, plotting = False, smoothing_ms = 1, badThresh = .5, kernLen = 2):

    # Look for unchanging signal across channels
    channelDataDiff = np.diff(channelData['data'])
    channelDataDiff = np.concatenate([np.ones((channelDataDiff.shape[0],1)), channelDataDiff], axis = 1)

    cumDiff = np.sum(np.abs(channelDataDiff), axis = 0)

    # convolve with step function to find consecutive
    # points where the derivative is identically zero across electrodes

    kern = np.ones((kernLen))
    cumDiff = np.convolve(cumDiff, kern, 'same')

    cumDiffDf = pd.Series(cumDiff)
    cumDiffBar = cumDiffDf.mean()
    cumDiffStd = cumDiffDf.std()

    badMask = cumDiffDf < badThresh

    nStd = 5

    if plotting:
        bins = np.linspace(cumDiffBar - nStd * cumDiffStd, cumDiffBar + nStd * cumDiffStd, 1000)
        bins = bins[bins > 0]
        leftBin = [] if cumDiffBar - nStd * cumDiffStd < cumDiffDf.min() else [cumDiffDf.min()]
        rightBin = [] if cumDiffBar + nStd * cumDiffStd > cumDiffDf.max() else [cumDiffDf.max()]

        bins = np.concatenate([leftBin, bins, rightBin], axis = 0)
        counts, _ = np.histogram(cumDiffDf, bins = bins)
        bins = bins[:-1]
        plt.plot(bins, counts,'bd-')
        plt.plot(bins[counts > 0], counts[counts > 0], 'rd')
        plt.show()

    maxAcceptable = cumDiffBar + nStd * cumDiffStd
    badMask = np.logical_or(badMask, cumDiffDf > maxAcceptable)

    smoothKernLen = smoothing_ms * 1e-3 * channelData['samp_per_s']
    smoothKern = np.ones((smoothKernLen))
    badMask = np.convolve(badMask, smoothKern, 'same') > 0

    if plotting:
        plot_chan(channelData, ExtendedHeaders,1, mask = badMask, show = True)
    return badMask

def get_spectrogram(cont_data, winLen_s, stepLen_fr = 0.5, R = 50, plotChan = 1):
    Fs = cont_data['samp_per_s']
    nSamples = len(cont_data['data'][0])
    t = np.arange(nSamples)
    nChan = len(cont_data['data'])
    delta = 1 / Fs

    winLen_samp = winLen_s * Fs
    stepLen_s = winLen_s * stepLen_fr
    stepLen_samp = stepLen_fr * winLen_samp

    NFFT = nextpowof2(winLen_samp)
    nw = winLen_s * R # time bandwidth product based on 0.1 sec windows and 200 Hz bandwidth
    ntapers = round(nw / 2) # L < nw - 1
    nWindows = m.floor((nSamples - winLen_samp + 1) / stepLen_samp)
    spectrum = np.zeros((nChan, int(winLen_samp / 2) + 1, nWindows))
    # generate a transform object with size equal to signal length and ntapers tapers
    D = libtfr.mfft_dpss(winLen_samp, nw, ntapers)

    for idx,signal in enumerate(cont_data['data']):
        P = D.mtspec(signal, stepLen_samp)
        P = P[np.newaxis,:,:]
        spectrum[idx,:,:] = P

    f = np.arange(P.shape[1])
    t = np.arange(P.shape[2]) * stepLen_s
    t,f = np.meshgrid(t,f)

    plotSpectrum = spectrum[plotChan,:,:]
    zMin, zMax = plotSpectrum.min(), plotSpectrum.max()
    plt.pcolor(t,f,plotSpectrum, vmin = zMin, vmax = zMax / 500)
    plt.axis([t.min(), t.max(), f.min(), f.max() / 10])
    plt.colorbar()
    #plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Time (s)')
    plt.ylabel("Frequency (Hz)")
    #plt.title(nsx_file.extended_headers[hdr_idx]['ElectrodeLabel'])
    plt.show()
    return spectrum

def plot_chan(channelData, ExtendedHeaders, whichChan, mask = None, show = False):
    # Plot the data channel
    ch_idx  = channelData['elec_ids'].index(whichChan)
    hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]
    t       = channelData['start_time_s'] + np.arange(channelData['data'].shape[1]) / channelData['samp_per_s']

    plt.plot(t, channelData['data'][ch_idx])
    if np.any(mask):
        mask = np.array(mask, dtype = bool)
        plt.plot(t[mask], channelData['data'][ch_idx][mask], 'ro')
    plt.axis([t[0], t[-1], min(channelData['data'][ch_idx]), max(channelData['data'][ch_idx])])
    plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Time (s)')
    plt.ylabel("Output (" + ExtendedHeaders[hdr_idx]['Units'] + ")")
    plt.title(ExtendedHeaders[hdr_idx]['ElectrodeLabel'])
    if show:
        plt.show()

def nextpowof2(x):
    return 2**(m.ceil(m.log(x, 2)))
