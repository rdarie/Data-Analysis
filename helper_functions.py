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
    channelData['t'] = channelData['start_time_s'] + np.arange(channelData['data'].shape[0]) / channelData['samp_per_s']
    channelData['spectrum'] = []
    # Close the nsx file now that all data is out
    nsx_file.close()

    return channelData, nsx_file.basic_header, nsx_file.extended_headers

def getBadDataMask(channelData, ExtendedHeaders, plotting = False, smoothing_ms = 1, badThresh = 1e-3, consecLen = 4):
    #Allocate bad data mask as dict
    badMask = {'general' : [], 'perChannel' : []}

    #Look for abnormally high values in the first difference of each channel
    # how many standard deviations should we keep?
    nStdDiff = 20
    nStdAmp = 10
    # Look for unchanging signal across channels
    channelDataDiff = channelData['data'].diff()
    channelDataDiff.fillna(0, inplace = True)

    #cumDiff = np.sum(np.abs(channelDataDiff), axis = 0)
    cumDiff = channelDataDiff.abs().sum(axis = 1)

    # convolve with step function to find consecutive
    # points where the derivative is identically zero across electrodes
    kern = np.ones((int(consecLen)))
    cumDiff = pd.Series(np.convolve(cumDiff.values, kern, 'same'))
    badMask['general'] = cumDiff < badThresh

    #smooth out the bad data mask
    smoothKernLen = smoothing_ms * 1e-3 * channelData['samp_per_s']
    smoothKern = np.ones((int(smoothKernLen)))
    badMask['general'] = np.convolve(badMask['general'], smoothKern, 'same') > 0
    #per channel, only smooth a couple of samples
    shortSmoothKern = np.ones((5))
    # per channel look for abberantly large jumps
    for idx, dRow in channelDataDiff.iteritems():

        # on the data itself
        row = channelData['data'][idx]
        rowVals = row.abs()
        rowBar  = rowVals.mean()
        rowStd  = rowVals.std()
        maxAcceptable = rowBar + nStdAmp * rowStd
        outliers = rowVals > maxAcceptable

        # on the derivative of the data
        dRowVals = dRow.abs()
        dRowBar  = dRowVals.mean()
        dRowStd  = dRowVals.std()
        dMaxAcceptable = dRowBar + nStdDiff * dRowStd

        # append to previous list of outliers
        outliers = np.logical_or(outliers, dRowVals > dMaxAcceptable)

        outliers = np.convolve(outliers.values, shortSmoothKern, 'same') > 0

        badMask['perChannel'].append(np.array(outliers, dtype = bool))

        if plotting and channelData['elec_ids'][idx] == plotting:

            plt.figure()
            dRowVals.plot.hist(bins = 100)
            plt.tight_layout()
            plt.show(block = False)

            plt.figure()
            plot_mask = np.logical_or(badMask['general'], badMask['perChannel'][idx])
            plt.plot(channelData['t'], row)
            plt.plot(channelData['t'][plot_mask], row[plot_mask],'ro')
            plt.plot(channelData['t'], dRowVals)
            print("Current derivative rejection threshold: %f" % dMaxAcceptable)
            plt.tight_layout()
            plt.show(block = False)

    badMask['general'] = np.array(badMask['general'], dtype = bool)

    return badMask

def get_spectrogram(channelData, ExtendedHeaders, winLen_s, stepLen_fr = 0.5, R = 50, whichChan = 1, plotting = False):

    Fs = channelData['samp_per_s']
    nChan = channelData['data'].shape[1]
    nSamples = channelData['data'].shape[0]
    t = np.arange(nSamples)
    delta = 1 / Fs

    winLen_samp = winLen_s * Fs
    stepLen_s = winLen_s * stepLen_fr
    stepLen_samp = stepLen_fr * winLen_samp

    NFFT = nextpowof2(winLen_samp)
    nw = winLen_s * R # time bandwidth product based on 0.1 sec windows and 200 Hz bandwidth
    ntapers = round(nw / 2) # L < nw - 1
    nWindows = m.floor((nSamples - NFFT + 1) / stepLen_samp)
    spectrum = np.zeros((nChan, int(NFFT / 2) + 1, nWindows))
    # generate a transform object with size equal to signal length and ntapers tapers
    D = libtfr.mfft_dpss(NFFT, nw, ntapers)
    #pdb.set_trace()

    for idx,signal in channelData['data'].iteritems():

        sys.stdout.write("Running getSpectrogram: %d%%\r" % int(idx * 100 / nChan + 1))
        sys.stdout.flush()

        P = D.mtspec(signal, stepLen_samp)
        P = P[np.newaxis,:,:]
        spectrum[idx,:,:] = P

    if plotting:
        plt.figure()
        f = np.arange(P.shape[1]) * channelData['samp_per_s'] / (2 * P.shape[1])
        t = np.arange(P.shape[2]) * stepLen_s
        t,f = np.meshgrid(t,f)

        ch_idx  = channelData['elec_ids'].index(whichChan)
        hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]

        plotSpectrum = spectrum[ch_idx,:,:]
        zMin, zMax = plotSpectrum.min(), plotSpectrum.max()
        plt.pcolormesh(t,f,plotSpectrum, vmin = zMin, vmax = zMax / 500)
        plt.axis([t.min(), t.max(), f.min(), 500])
        plt.colorbar()
        #plt.locator_params(axis='y', nbins=20)
        plt.xlabel('Time (s)')
        plt.ylabel("Frequency (Hz)")
        plt.title(ExtendedHeaders[hdr_idx]['ElectrodeLabel'])
        plt.tight_layout()
        plt.show(block = False)
    return spectrum

def plot_chan(channelData, ExtendedHeaders, whichChan, mask = None, show = False, prevFig = None):
    # Plot the data channel
    ch_idx  = channelData['elec_ids'].index(whichChan)
    hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]

    if not prevFig:
        #plt.figure()
        f, ax = plt.subplots()
    else:
        f = prevFig
        ax = prevFig.axes[0]


    ax.plot(channelData['t'], channelData['data'][ch_idx])

    if np.any(mask):
        ax.plot(channelData['t'][mask], channelData['data'][ch_idx][mask], 'ro')

    #pdb.set_trace()
    #channelData['data'][ch_idx].fillna(0, inplace = True)

    ax.axis([channelData['t'][0], channelData['t'][-1], min(channelData['data'][ch_idx]), max(channelData['data'][ch_idx])])
    ax.locator_params(axis = 'y', nbins = 20)
    plt.xlabel('Time (s)')
    plt.ylabel("Output (" + ExtendedHeaders[hdr_idx]['Units'] + ")")
    plt.title(ExtendedHeaders[hdr_idx]['ElectrodeLabel'])
    plt.tight_layout()
    if show:
        plt.show(block = False)


    return f, ax

def nextpowof2(x):
    return 2**(m.ceil(m.log(x, 2)))

def replaceBad(dfSeries, mask, typeOpt = 'nans'):
    dfSeries[mask] = float('nan')
    if typeOpt == 'nans':
        pass
    elif typeOpt == 'interp':
        dfSeries.interpolate(method = 'linear', inplace = True)
        if dfSeries.isnull().any(): # For instance, if begins with bad data, there is nothing there to linearly interpolate
            dfSeries.fillna(method = 'backfill', inplace = True)
            dfSeries.fillna(method = 'ffill', inplace = True)

    return dfSeries

import matplotlib.backends.backend_pdf

def pdfReport(origData, cleanData, ExtendedHeaders, badData = None, pdfFilePath = 'pdfReport.pdf', spectrum = False):

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdfFilePath)
    nChan = cleanData['data'].shape[1]

    if spectrum:
        P = cleanData['spectrum'][0,:,:]
        #pdb.set_trace()
        fr = np.arange(P.shape[0]) / P.shape[0] * cleanData['samp_per_s'] / 2
        t = cleanData['start_time_s'] + np.arange(P.shape[1]) * (cleanData['t'][-1]-cleanData['start_time_s']) / P.shape[1]
        t,fr = np.meshgrid(t,fr)

    for idx, row in origData['data'].iteritems():

        sys.stdout.write("Running pdfReport: %d%%\r" % int(idx * 100 / nChan + 1))
        sys.stdout.flush()

        ch_idx  = origData['elec_ids'].index(idx + 1)
        plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
        f,_ = plot_chan(origData, ExtendedHeaders, idx + 1, mask = None, show = False)
        plot_chan(cleanData, ExtendedHeaders, idx + 1, mask = plot_mask, show = False, prevFig = f)
        plt.tight_layout()
        pdf.savefig(f)
        plt.close(f)

        if spectrum:
            P = cleanData['spectrum'][idx,:,:]
            zMin, zMax = P.min(), P.max()

            f = plt.figure()
            plt.pcolormesh(t,fr,P, vmin = zMin, vmax = zMax / 500)
            plt.axis([t.min(), t.max(), fr.min(), 500])
            plt.colorbar()
            #plt.locator_params(axis='y', nbins=20)
            plt.xlabel('Time (s)')
            plt.ylabel("Frequency (Hz)")
            plt.tight_layout()
            pdf.savefig(f)
            plt.close(f)

    #pdb.set_trace()
    for idx, row in origData['data'].iteritems():
        if idx == 0:
            f,_ = plot_chan(origData, ExtendedHeaders, idx + 1, mask = None, show = False)
        elif idx == origData['data'].shape[1] - 1:
            f,_ = plot_chan(origData, ExtendedHeaders, idx + 1, mask = badData['general'], show = True, prevFig = f)
            plt.tight_layout()
            pdf.savefig(f)
            plt.close(f)
        else:
            f,_ = plot_chan(origData, ExtendedHeaders, idx + 1, mask = None, show = False, prevFig = f)

    pdf.close()
