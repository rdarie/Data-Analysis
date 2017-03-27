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

    channelData['data'] = pd.DataFrame(channelData['data'].transpose())

    # Close the nsx file now that all data is out
    nsx_file.close()

    return channelData, nsx_file.basic_header, nsx_file.extended_headers

def getBadDataMask(channelData, ExtendedHeaders, plotting = False, smoothing_ms = 1, badThresh = .1, kernLen = 2):
    #Look for abnormally high values in the first difference of each channel
    # how many standard deviations should we keep?
    nStd = 5
    # Look for unchanging signal across channels
    channelDataDiff = channelData['data'].diff()

    #cumDiff = np.sum(np.abs(channelDataDiff), axis = 0)
    cumDiff = channelDataDiff.abs().sum(axis = 1)

    # convolve with step function to find consecutive
    # points where the derivative is identically zero across electrodes
    kern = np.ones((kernLen))
    cumDiff = pd.Series(np.convolve(cumDiff.values, kern, 'same'))
    badMask = cumDiff < badThresh

    for idx, row in channelDataDiff.iteritems():
        rowBar = row.mean()
        rowStd = row.std()
        maxAcceptable = rowBar + nStd * rowStd
        badMask = np.logical_or(badMask, row > maxAcceptable)

        if plotting and idx == 0:
            plt.figure()
            row.plot.hist(bins = 100, kind = 'line')
            plt.show()

    #smooth out the bad data mask
    smoothKernLen = smoothing_ms * 1e-3 * channelData['samp_per_s']
    smoothKern = np.ones((smoothKernLen))
    badMask = np.convolve(badMask, smoothKern, 'same') > 0
    badMask = np.array(badMask, dtype = bool)

    if plotting:
        plot_chan(channelData, ExtendedHeaders, 1, mask = badMask, show = True)

    return badMask

def get_spectrogram(cont_data, winLen_s, stepLen_fr = 0.5, R = 50, plotChan = 1):
    Fs = cont_data['samp_per_s']
    nChan = cont_data['data'].shape[1]
    nSamples = cont_data['data'].shape[0]
    t = np.arange(nSamples)
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
    #pdb.set_trace()

    for idx,signal in cont_data['data'].iteritems():
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

def plot_chan(channelData, ExtendedHeaders, whichChan, mask = None, show = False, prevFig = None):
    # Plot the data channel

    ch_idx  = channelData['elec_ids'].index(whichChan)
    hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]
    t       = channelData['start_time_s'] + np.arange(channelData['data'].shape[0]) / channelData['samp_per_s']

    if not prevFig:
        f, ax = plt.subplots()
    else:
        f = prevFig
        ax = prevFig.axes[0]

    plt.plot(t, channelData['data'][ch_idx])

    if np.any(mask):
        plt.plot(t[mask], channelData['data'][ch_idx][mask], 'ro')

    plt.axis([t[0], t[-1], min(channelData['data'][ch_idx]), max(channelData['data'][ch_idx])])
    plt.locator_params(axis = 'y', nbins = 20)
    plt.xlabel('Time (s)')
    plt.ylabel("Output (" + ExtendedHeaders[hdr_idx]['Units'] + ")")
    plt.title(ExtendedHeaders[hdr_idx]['ElectrodeLabel'])

    if show:
        plt.show()

    return f, ax

def nextpowof2(x):
    return 2**(m.ceil(m.log(x, 2)))


def replaceBad(dfSeries, mask, typeOpt = 'nans'):
    dfSeries[mask] = float('nan')
    if typeOpt == 'nans':
        return dfSeries
    elif typeOpt == 'interp':
        dfSeries.interpolate(method = 'linear', inplace = True)
        return dfSeries

import matplotlib.backends.backend_pdf

def pdfReport(channelData, ExtendedHeaders, mask = None, pdfFilePath = 'pdfReport.pdf'):

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdfFilePath)
    for idx, row in channelData['data'].iteritems():
        f,_ = plot_chan(channelData, ExtendedHeaders, idx + 1, mask = mask, show = False)
        pdf.savefig(f)
        plt.close(f)

    #pdb.set_trace()
    for idx, row in channelData['data'].iteritems():
        if idx == 0:
            f,_ = plot_chan(channelData, ExtendedHeaders, idx + 1, mask = None, show = False)
        elif idx == channelData['data'].shape[1] - 1:
            f,_ = plot_chan(channelData, ExtendedHeaders, idx + 1, mask = mask, show = True, prevFig = f)
            pdf.savefig(f)
            plt.close(f)
        else:
            f,_ = plot_chan(channelData, ExtendedHeaders, idx + 1, mask = None, show = False, prevFig = f)

    pdf.close()
