import pdb
from brpylib             import NsxFile, NevFile, brpylib_ver
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import math as m
import sys, itertools, os, pickle, gc, random
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

try:
    import libtfr
    HASLIBTFR = True
except:
    HASLIBTFR = False

import peakutils
from scipy import interpolate
from copy import *
import matplotlib.backends.backend_pdf
import math
from fractions import gcd

def getNEVData(filePath, elecIds):
    # Version control
    brpylib_ver_req = "1.3.1"
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

    # Open file and extract headers
    nev_file = NevFile(filePath)
    # Extract data and separate out spike data
    spikes = nev_file.getdata(elecIds)['spike_events']

    spikes['basic_headers'] = nev_file.basic_header
    spikes['extended_headers'] = nev_file.extended_headers
    # Close the nev file now that all data is out
    nev_file.close()
    return spikes

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
    channelData['badData'] = dict()
    channelData['spectrum'] = pd.DataFrame({'PSD': [], 't': [], 'fr': [], 'Labels': []})
    channelData['basic_headers'] = nsx_file.basic_header
    channelData['extended_headers'] =  nsx_file.extended_headers
    # Close the nsx file now that all data is out
    nsx_file.close()

    return channelData

def getBadSpikesMask(spikes, nStd = 5, whichChan = 0, plotting = False, deleteBad = False):
    spikesBar = [np.mean(sp, axis = 0) for sp in spikes['Waveforms']]
    spikesStd = [np.std (sp, axis = 0) for sp in spikes['Waveforms']]

    t = np.arange(spikesBar[0].shape[0])
    if plotting:
        fi = plt.figure()
        plt.plot(t, spikesBar[whichChan])
        plt.fill_between(t, spikesBar[whichChan]+spikesStd[whichChan],
                         spikesBar[whichChan]-spikesStd[whichChan],facecolor='blue',
                         alpha = 0.3, label = 'mean(spike)')

    badMask = []
    for idx, sp in enumerate(spikes['Waveforms']):
        maxAcceptable = np.abs(spikesBar[idx]) + nStd*spikesStd[idx]
        outliers = [(np.abs(row) > maxAcceptable).any() for row in sp]
        badMask.append(np.array(outliers, dtype = bool))
    #
    if deleteBad:
        for idx, sp in enumerate(spikes['Waveforms']):
            spikes['Waveforms'][idx] = sp[np.logical_not(badMask[idx])]
            spikes['Classification'][idx] = np.array(spikes['Classification'][idx])[np.logical_not(badMask[idx])]
            spikes['TimeStamps'][idx] = np.array(spikes['TimeStamps'][idx])[np.logical_not(badMask[idx])]
    return badMask

def getBadContinuousMask(channelData, plotting = False, smoothing_ms = 1, badThresh = 1e-3, consecLen = 4):
    #Allocate bad data mask as dict
    badMask = {'general' : [], 'perChannel' : []}

    #Look for abnormally high values in the first difference of each channel
    # how many standard deviations should we keep?
    nStdDiff = 10
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
        rowVals = row.values
        rowBar  = rowVals.mean()
        rowStd  = rowVals.std()
        maxAcceptable = rowBar + nStdAmp * rowStd
        minAcceptable = rowBar - nStdAmp * rowStd

        outliers = np.logical_or(row > maxAcceptable,row < minAcceptable)

        # on the derivative of the data
        dRowVals = dRow.values
        dRowBar  = dRowVals.mean()
        dRowStd  = dRowVals.std()
        dMaxAcceptable = dRowBar + nStdDiff * dRowStd
        dMinAcceptable = dRowBar - nStdDiff * dRowStd

        # append to previous list of outliers
        newOutliers = np.logical_or(dRowVals > dMaxAcceptable, dRowVals < dMinAcceptable)
        outliers = np.logical_or(outliers, newOutliers)

        outliers = np.convolve(outliers.values, shortSmoothKern, 'same') > 0

        badMask['perChannel'].append(np.array(outliers, dtype = bool))

        if plotting and channelData['elec_ids'][idx] == plotting:

            plt.figure()
            dRow.plot.hist(bins = 100)
            plt.tight_layout()
            plt.show(block = False)

            plt.figure()
            plot_mask = np.logical_or(badMask['general'], badMask['perChannel'][idx])
            plt.plot(channelData['t'], row)
            plt.plot(channelData['t'][plot_mask], row[plot_mask],'ro')
            plt.plot(channelData['t'], dRowVals)
            print("Current derivative rejection threshold: %f" % dMaxAcceptable)
            print("Current amplitude rejection threshold: %f" % maxAcceptable)
            plt.tight_layout()
            plt.show(block = False)

    badMask['general'] = np.array(badMask['general'], dtype = bool)

    return badMask

def getCameraTriggers(simiData, plotting = False):
    # sample rate
    fs = simiData['samp_per_s']
    # get camera triggers
    triggers = simiData['data']
     # expected inter trigger interval
    iti = .01
    # minimum distance between triggers (units of samples)
    width = fs * iti / 2
    # first difference of triggers
    triggersPrime = triggers.diff()
    triggersPrime.fillna(0, inplace = True)
    # moments when camera capture occured
    peakIdx = peakutils.indexes(triggersPrime.values.squeeze(), thres=0.7, min_dist=width)

    if plotting:
        f = plt.figure()
        plt.plot(simiData['t'], triggers.values)
        plt.plot(simiData['t'][peakIdx], triggers.values[peakIdx], 'r*')
        ax = plt.gca()
        ax.set_xlim([36.5, 37])
        plt.show(block = False)

    # get time of first simi frame in NSP time:
    trigTimes = simiData['t'][peakIdx]
    # timeOffset = trigTimes[0]

    return peakIdx, trigTimes

def getGaitEvents(trigTimes, simiTable, whichColumns =  ['ToeUp_Left Y', 'ToeDown_Left Y'], plotting = False, fudge = 2, CameraFs = 100):
    # NSP time of first camera trigger
    timeOffset = trigTimes[0]

    # max time recorded on NSP

    # Note: Clocks drift slightly between Simi Computer and NSP. Allow timeMax
    # to go a little bit beyond so that we don't drop the last simi frame, i.e.
    # add fudge times the time increment TODO: FIX THIS.

    timeMax = trigTimes[-1] + fudge/CameraFs

    simiDf = pd.DataFrame(simiTable[whichColumns])
    simiDf = simiDf.notnull()
    simiDf['simiTime'] = simiTable['Time'] + timeOffset
    simiDf.drop(simiDf[simiDf['simiTime'] >= timeMax].index, inplace = True)

    debugging = False
    if debugging:
        nNSPTrigs = len(trigTimes)
        simiHist, simiBins = np.histogram(np.diff(simiDf['simiTime'].values))
        totalSimiTime = np.diff(simiDf['simiTime'].values).sum()
        nSimiTrigs = len(simiDf['simiTime'].values)
        trigHist, trigBins = np.histogram(np.diff(trigTimes))
        totalNSPTime = np.diff(trigTimes).sum()
        np.savetxt('getGaitEvents-trigTimes.txt', trigTimes, fmt = '%4.4f', delimiter = ' \n')
        pdb.set_trace()
    #
    simiDf['NSPTime'] = pd.Series(trigTimes, index = simiDf.index)

    simiDfPadded = deepcopy(simiDf)
    for idx, row in simiDfPadded.iterrows():

        if row[whichColumns[0]]:
            newSimiTime = row['simiTime'] + 1/CameraFs
            newNSPTime = row['NSPTime'] + 1/CameraFs
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: False, whichColumns[1]: False, 'simiTime': row['simiTime'] - 1e-6, 'NSPTime': row['NSPTime'] - 1e-6}), ignore_index = True)
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: True, whichColumns[1]: False, 'simiTime': newSimiTime -1e-6, 'NSPTime': newNSPTime -1e-6}), ignore_index = True)
            #simiDfPadded = simiDfPadded.append(pd.Series({'ToeUp_Left Y': False, 'ToeDown_Left Y': False, 'simiTime': newSimiTime + 1e-6, 'NSPTime': newNSPTime + 1e-6}), ignore_index = True)

        if row[whichColumns[1]]:
            newSimiTime = row['simiTime'] + 1/CameraFs
            newNSPTime = row['NSPTime'] + 1/CameraFs
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: False, whichColumns[1]: False, 'simiTime': row['simiTime'] - 1e-6, 'NSPTime': row['NSPTime'] - 1e-6}), ignore_index = True)
            simiDfPadded = simiDfPadded.append(pd.Series({whichColumns[0]: False, whichColumns[1]: True, 'simiTime': newSimiTime -1e-6, 'NSPTime': newNSPTime -1e-6}), ignore_index = True)
            #simiDfPadded = simiDfPadded.append(pd.Series({'ToeUp_Left Y': False, 'ToeDown_Left Y': False, 'simiTime': newSimiTime + 1e-6, 'NSPTime': newNSPTime + 1e-6}), ignore_index = True)

    simiDfPadded.sort_values('simiTime', inplace = True)
    down = (simiDfPadded[whichColumns[1]].values * 1)
    up = (simiDfPadded[whichColumns[0]].values * 1)
    gait = up.cumsum() - down.cumsum()

    if plotting:
        f = plt.figure()
        plt.plot(simiDfPadded['simiTime'], gait)
        plt.plot(simiDfPadded['simiTime'], down, 'g*')
        plt.plot(simiDfPadded['simiTime'], up, 'r*')
        ax = plt.gca()
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([6.8, 7.8])
        plt.show(block = False)

    gaitLabelFun = interpolate.interp1d(simiDfPadded['simiTime'], gait, bounds_error = False, fill_value = 'extrapolate')
    downLabelFun = interpolate.interp1d(simiDfPadded['simiTime'], down, bounds_error = False, fill_value = 0)
    upLabelFun   = interpolate.interp1d(simiDfPadded['simiTime'], up  , bounds_error = False, fill_value = 0)
    # TODO: check that the padded Df oesn't inerfere with thiese labelrs
    simiDf['Labels'] = pd.Series(['Swing' if x > 0 else 'Stance' for x in gaitLabelFun(simiDf['simiTime'])], index = simiDf.index)
    return simiDf, gaitLabelFun, downLabelFun, upLabelFun

def assignLabels(timeVector, lbl, fnc, CameraFs = 100, oversizeWindow = None):
    dt = np.mean(np.diff(timeVector))

    if dt < 1/CameraFs:
        # sampling faster than the original data! Interpolate!
        labels = pd.Series([lbl if x > 0 else 'Neither' for x in fnc(timeVector)])
        #
    else:
        #sampling slower than the original data! Histogram!
        pseudoTime = np.arange(timeVector[0], timeVector[-1] + dt, 0.1/CameraFs)
        oversampledFnc = fnc(pseudoTime)
        #pdb.set_trace()
        if oversizeWindow is None:
            timeBins = np.append(timeVector, timeVector[-1] + dt)
            #
            histo,_ = np.histogram(pseudoTime[oversampledFnc > 0], timeBins)
            labels = pd.Series([lbl if x > 0 else 'Neither' for x in histo])
        else:
            binInterval = dt
            binWidth = oversizeWindow
            timeStart = timeVector[0] - binWidth / 2

            timeEnd = timeVector[-1] + binWidth / 2
            #pdb.set_trace()
            mat, binCenters, binLeftEdges = binnedEvents(
                [pseudoTime[oversampledFnc > 0]], [0], [0],
                binInterval, binWidth, timeStart, timeEnd)
            mat = np.squeeze(mat)
            """
            It's possible mat will have an extra entry, because we don't know
            what the original timeEnd was.
            timeVector[-1] <= originalTimeEnd - binWidth/2, because we're not
            guaranteed that the time window is divisible by the binInterval.
            Therefore, delete the last entry if need be:
            """
            if len(mat) != len(timeVector):
                mat = mat[:-1]
            labels = pd.Series([lbl if x > 0 else 'Neither' for x in mat])
            #pdb.set_trace()
    return labels

if HASLIBTFR:
    def getSpectrogram(channelData, winLen_s, stepLen_s = 0.02, R = 50, fr_cutoff = None, whichChan = 1, plotting = False):

        Fs = channelData['samp_per_s']
        nChan = channelData['data'].shape[1]
        nSamples = channelData['data'].shape[0]

        delta = 1 / Fs

        winLen_samp = int(winLen_s * Fs)
        stepLen_samp = int(stepLen_s * Fs)

        NFFT = nextpowof2(winLen_samp)
        nw = winLen_s * R # time bandwidth product based on 0.1 sec windows and 200 Hz bandwidth
        ntapers = round(nw / 2) # L < nw - 1
        nWindows = m.floor((nSamples - NFFT + 1) / stepLen_samp)

        fr_samp = int(NFFT / 2) + 1
        fr = np.arange(fr_samp) * channelData['samp_per_s'] / (2 * fr_samp)
        if fr_cutoff is not None:
            fr = fr[fr < fr_cutoff]
            fr_samp = len(fr)

        t = channelData['start_time_s'] + np.arange(nWindows) * stepLen_s
        spectrum = np.zeros((nChan, nWindows, fr_samp))
        # generate a transform object with size equal to signal length and ntapers tapers
        D = libtfr.mfft_dpss(NFFT, nw, ntapers)
        #pdb.set_trace()

        for idx,signal in channelData['data'].iteritems():

            sys.stdout.write("Running getSpectrogram: %d%%\r" % int(idx * 100 / nChan + 1))
            sys.stdout.flush()

            P = D.mtspec(signal, stepLen_samp).transpose()
            #pdb.set_trace()
            P = P[np.newaxis,:,:fr_samp]
            spectrum[idx,:,:] = P

        if plotting:

            ch_idx  = channelData['elec_ids'].index(whichChan)

            #TODO: implement passing elecID to plotSpectrum
            #hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]

            P = spectrum[ch_idx,:,:]
            plotSpectrum(P, Fs, channelData['start_time_s'], channelData['t'][-1], fr =fr, t = t, show = True)

        #pdb.set_trace()

        return {'PSD' : pd.Panel(spectrum, items = channelData['elec_ids'], major_axis = t, minor_axis = fr),
                'fr' : fr,
                't' : t
                }

def nextpowof2(x):
    return 2**(m.ceil(m.log(x, 2)))

def cleanNEVSpikes(spikes, badData):
    pass

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

def plotChan(channelData, whichChan, mask = None, show = False, prevFig = None):
    # Plot the data channel
    ch_idx  = channelData['elec_ids'].index(whichChan)
    hdr_idx = channelData['ExtendedHeaderIndices'][ch_idx]

    if not prevFig:
        #plt.figure()
        f, ax = plt.subplots()
    else:
        f = prevFig
        ax = prevFig.axes[0]

    channelDataForPlotting = channelData['data'].drop(['Labels', 'LabelsNumeric'], axis = 1) if 'Labels' in channelData['data'].columns.values else channelData['data']
    ax.plot(channelData['t'], channelDataForPlotting[ch_idx])

    if np.any(mask):
        ax.plot(channelData['t'][mask], channelDataForPlotting[ch_idx][mask], 'ro')

    #pdb.set_trace()
    #channelData['data'][ch_idx].fillna(0, inplace = True)

    ax.axis([channelData['t'][0], channelData['t'][-1], min(channelDataForPlotting[ch_idx]), max(channelDataForPlotting[ch_idx])])
    ax.locator_params(axis = 'y', nbins = 20)
    plt.xlabel('Time (s)')
    plt.ylabel("Output (" + channelData['extended_headers'][hdr_idx]['Units'] + ")")
    plt.title(channelData['extended_headers'][hdr_idx]['ElectrodeLabel'])
    plt.tight_layout()
    if show:
        plt.show(block = False)


    return f, ax

def pdfReport(origData, cleanData, badData = None, pdfFilePath = 'pdfReport.pdf', spectrum = False):
    with matplotlib.backends.backend_pdf.PdfPages(pdfFilePath) as pdf:
        nChan = cleanData['data'].shape[1]

        if spectrum:
            fr = cleanData['spectrum']['fr']
            t = cleanData['spectrum']['t']
            Fs = cleanData['samp_per_s']

        for idx, row in origData['data'].iteritems():

            sys.stdout.write("Running pdfReport: %d%%\r" % int(idx * 100 / nChan + 1))
            sys.stdout.flush()

            ch_idx  = origData['elec_ids'].index(idx + 1)
            plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
            f,_ = plot_chan(origData, idx + 1, mask = None, show = False)
            plot_chan(cleanData, idx + 1, mask = plot_mask, show = False, prevFig = f)
            plt.tight_layout()
            pdf.savefig(f)
            plt.close(f)

            if spectrum:
                P = cleanData['spectrum']['PSD'][idx,:,:]
                f = plotSpectrum(P, Fs, cleanData['start_time_s'], cleanData['t'][-1], fr = fr, t = t, show = False)
                pdf.savefig(f)
                plt.close(f)

        #pdb.set_trace()
        generateLastPage = False
        if generateLastPage:
            for idx, row in origData['data'].iteritems():
                if idx == 0:
                    f,_ = plot_chan(origData, idx + 1, mask = None, show = False)
                elif idx == origData['data'].shape[1] - 1:
                    f,_ = plot_chan(origData, idx + 1, mask = badData['general'], show = True, prevFig = f)
                    plt.tight_layout()
                    pdf.savefig(f)
                    plt.close(f)
                else:
                    f,_ = plot_chan(origData, idx + 1, mask = None, show = False, prevFig = f)

def plotSpectrum(P, fs, start_time_s, end_time_s, fr = None, t = None, show = False):
    #pdb.set_trace()
    if fr is None:
        fr = np.arange(P.shape[1]) / P.shape[1] * fs / 2
    if t is None:
        t = start_time_s + np.arange(P.shape[0]) * (end_time_s-start_time_s) / P.shape[0]

    if type(P) is pd.DataFrame:
        P = P.values
    zMin, zMax = P.min(), P.max()

    f = plt.figure()
    plt.pcolormesh(t,fr,P.transpose(), norm=colors.LogNorm(vmin=zMin, vmax=zMax))

    plt.axis([t.min(), t.max(), fr.min(), min(300, fr.max())])
    #plt.colorbar()
    #plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Time (s)')
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    if show:
        plt.show(block = False)
    return f

def binnedEvents(timeStamps, chans, ChannelID, binInterval, binWidth, timeStart, timeEnd):

    binCenters = np.arange(timeStart + binWidth / 2, timeEnd - binWidth/2 + binInterval, binInterval)

    #timeInterval = timeEnd - timeStart - binWidth
    binRes = gcd(math.floor(binWidth *1e3 / 2), math.floor(binInterval*1e3))*1e-3 # greatest common denominator
    fineBins = np.arange(timeStart, timeEnd + binRes, binRes)

    fineBinsPerWindow = int(binWidth / binRes)
    fineBinsPerInterval = int(binInterval / binRes)
    fineBinsTotal = len(fineBins) - 1
    centerIdx = np.arange(0, fineBinsTotal - fineBinsPerWindow + fineBinsPerInterval, fineBinsPerInterval)
    binLeftEdges = centerIdx * binRes

    nChans = len(timeStamps)
    spikeMat = np.zeros([nChans, len(binCenters)])
    for idx, chan in enumerate(chans):
        ch_idx = ChannelID.index(chan)
        histo, _ = np.histogram(timeStamps[ch_idx], fineBins)
        #pdb.set_trace()
        spikeMat[idx, :] = np.array(
            [histo[x:x+fineBinsPerWindow].sum() / binWidth for x in centerIdx]
        )

    return spikeMat, binCenters, binLeftEdges

def binnedSpikes(spikes, chans, binInterval, binWidth, timeStart, timeDur):
    timeEnd = timeStart + timeDur
    #binCenters = timeStart + BinWidth / 2 : binInterval : timeEnd - binWidth/2
    timeStamps = [x / spikes['basic_headers']['TimeStampResolution'] for x in spikes['TimeStamps']]
    ChannelID = spikes['ChannelID']
    spikeMat, binCenters, binLeftEdges = binnedEvents(timeStamps, chans, ChannelID, binInterval, binWidth, timeStart, timeEnd)

    return pd.DataFrame(spikeMat), binCenters, binLeftEdges

def plotBinnedSpikes(spikeMat, binCenters, chans, show = True, normalizationType = 'linear'):
    #pdb.set_trace()
    zMin, zMax = spikeMat.min().min(), spikeMat.max().max()
    if normalizationType == 'linear':
        nor = colors.Normalize(vmin=zMin, vmax=zMax)
    elif normalizationType == 'logarithmic':
        nor = colors.SymLogNorm(linthresh= 1, linscale=1,
                                              vmin=zMin, vmax=zMax)
    fi = plt.figure()
    chanIdx = np.arange(len(chans))
    plt.pcolormesh(binCenters, chanIdx, spikeMat.values.transpose(), norm = nor)
    plt.axis([binCenters.min(), binCenters.max(), chanIdx.min(), chanIdx.max()])
    #plt.colorbar()
    #plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Time (s)')
    plt.ylabel("Channel (#)")
    plt.tight_layout()
    if show:
        plt.show()
    return fi

def plot_spikes(spikes, chans):
    # Initialize plots
    colors      = 'kbgrm'
    line_styles = ['-', '--', ':', '-.']
    f, axarr    = plt.subplots(len(chans))
    samp_per_ms = spikes['basic_headers']['SampleTimeResolution'] / 1000.0

    for i in range(len(chans)):

        # Extract the channel index, then use that index to get unit ids, extended header index, and label index
        ch_idx      = spikes['ChannelID'].index(chans[i])
        units       = sorted(list(set(spikes['Classification'][ch_idx])))
        ext_hdr_idx = spikes['NEUEVWAV_HeaderIndices'][ch_idx]
        lbl_idx     = next(idx for (idx, d) in enumerate(spikes['extended_headers'])
                           if d['PacketID'] == 'NEUEVLBL' and d['ElectrodeID'] == chans[i])

        # loop through all spikes and plot based on unit classification
        # note: no classifications in sampleData, i.e., only unit='none' exists in the sample data
        ymin = 0; ymax = 0
        t = np.arange(spikes['extended_headers'][ext_hdr_idx]['SpikeWidthSamples']) / samp_per_ms

        for j in range(len(units)):
            unit_idxs   = [idx for idx, unit in enumerate(spikes['Classification'][ch_idx]) if unit == units[j]]
            unit_spikes = np.array(spikes['Waveforms'][ch_idx][unit_idxs]) / 1000

            if units[j] == 'none':
                color_idx = 0; ln_sty_idx = 0
            else:
                color_idx = (units[j] % len(colors)) + 1
                ln_sty_idx = units[j] // len(colors)

            for k in range(unit_spikes.shape[0]):
                axarr[i].plot(t, unit_spikes[k], (colors[color_idx] + line_styles[ln_sty_idx]))
                if min(unit_spikes[k]) < ymin: ymin = min(unit_spikes[k])
                if max(unit_spikes[k]) > ymax: ymax = max(unit_spikes[k])

        if lbl_idx: axarr[i].set_ylabel(spikes['extended_headers'][lbl_idx]['Label'] + ' ($\mu$V)')
        else:       axarr[i].set_ylabel('Channel ' + str(chans[i]) + ' ($\mu$V)')
        axarr[i].set_ylim((ymin * 1.05, ymax * 1.05))
        axarr[i].locator_params(axis='y', nbins=10)

    axarr[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show(block = False)

def plot_raster(spikes, chans):
    # Initialize plots
    colors      = 'kbgrm'
    line_styles = ['-', '--', ':', '-.']
    f, ax    = plt.subplots()
    samp_per_s = spikes['basic_headers']['SampleTimeResolution']

    timeMax = max([max(x) for x in spikes['TimeStamps']])

    for idx, chan in enumerate(chans):
        # Extract the channel index, then use that index to get unit ids, extended header index, and label index
        #pdb.set_trace()
        ch_idx      = spikes['ChannelID'].index(chan)
        units       = sorted(list(set(spikes['Classification'][ch_idx])))
        ext_hdr_idx = spikes['NEUEVWAV_HeaderIndices'][ch_idx]
        lbl_idx     = next(idx for (idx, d) in enumerate(spikes['extended_headers'])
                           if d['PacketID'] == 'NEUEVLBL' and d['ElectrodeID'] == chan)

        # loop through all spikes and plot based on unit classification
        # note: no classifications in sampleData, i.e., only unit='none' exists in the sample data

        t = np.arange(timeMax) / samp_per_s

        for j in range(len(units)):
            unit_idxs   = [idx for idx, unit in enumerate(spikes['Classification'][ch_idx]) if unit == units[j]]
            unit_spike_times = np.array(spikes['TimeStamps'][ch_idx])[unit_idxs] / samp_per_s

            if units[j] == 'none':
                color_idx = 0; ln_sty_idx = 0
            else:
                color_idx = (units[j] % len(colors)) + 1
                ln_sty_idx = units[j] // len(colors)

            for spt in unit_spike_times:
                ax.vlines(spt, idx, idx + 1, colors = colors[color_idx],  linestyles = line_styles[ln_sty_idx])

        ax.set_ylim((0, len(chans)))
        ax.locator_params(axis='y', nbins=10)

    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show(block = False)

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Modified from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    fi = plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #np.set_printoptions(precision=2)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:4.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.colorbar()
    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fi

def freqDownSampler(nChan, factor):
    def downSample(X):
        support = np.arange(X.shape[1] / nChan) # range of original frequencies
        downSampledSupport = np.linspace(0, support[-1], m.floor(len(support) / factor)) # range of new frequencies

        XReshaped = np.reshape(X.values, (X.shape[0], nChan, -1)) # electrodes by frequency by time bin
        XDownSampled = np.zeros((XReshaped.shape[0], XReshaped.shape[1], m.floor(XReshaped.shape[2] / factor)))# electrodes by new frequency by time bin

        for idx in range(nChan):
            oldX = XReshaped[:, idx, :]
            interpFun = interpolate.interp1d(support, oldX, kind = 'cubic', axis = -1)
            XDownSampled[:, idx, :] = interpFun(downSampledSupport)

        XDownSampledFlat = pd.DataFrame(np.reshape(XDownSampled, (X.shape[0], -1)))
        return XDownSampledFlat
    return downSample

def freqDownSample(X, **kwargs):
    keepChans = kwargs['keepChans']
    whichChans = kwargs['whichChans']
    nChan = len(whichChans)

    newNChan = len(keepChans)

    freqFactor = kwargs['freqFactor']
    support = np.arange(X.shape[1] / nChan) # range of original frequencies
    newFreqRes = m.floor(len(support) / freqFactor)

    nTimePoints = X.shape[0]

    downSampledSupport = np.linspace(support[0], support[-1], newFreqRes) # range of new frequencies

    XReshaped = np.reshape(X, (nTimePoints, nChan, -1)) # electrodes by frequency by time bin
    del(X)
    XDownSampled = np.zeros((XReshaped.shape[0], newNChan, m.floor(XReshaped.shape[2] / freqFactor)))# electrodes by new frequency by time bin

    for newIdx, origIdx in enumerate(keepChans):
        oldX = XReshaped[:, origIdx, :]
        interpFun = interpolate.interp1d(support, oldX, kind = 'linear', axis = -1)
        XDownSampled[:, newIdx, :] = interpFun(downSampledSupport)
        del(interpFun)
        #gc.collect()

    del(XReshaped)
    XDownSampledFlat = pd.DataFrame(np.reshape(XDownSampled, (nTimePoints, -1)))
    return XDownSampledFlat

def trainSpectralMethod(dataName, whichChans, maxFreq, estimator, skf, parameters, outputFileName):

    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

    dataFile = localDir + dataName
    pickleData = pd.read_pickle(dataFile)

    whichFreqs = pickleData['channel']['spectrum']['fr'] < maxFreq

    spectrum = pickleData['channel']['spectrum']['PSD']
    #t = ns5Data['channel']['spectrum']['t']
    y = pickleData['channel']['spectrum']['LabelsNumeric'] # labels
    del(pickleData) # done with pickle data

    reducedSpectrum = spectrum[whichChans, :, whichFreqs]
    del(spectrum) # done with spectrum
    X = reducedSpectrum.transpose(1, 0, 2).to_frame().transpose()
    del(reducedSpectrum)

    grid=GridSearchCV(estimator, parameters, cv = skf, verbose = 4, scoring = 'f1_macro', pre_dispatch='n_jobs', n_jobs = -1)

    #if __name__ == '__main__':
    grid.fit(X,y)

    bestEstimator={'estimator' : grid.best_estimator_, 'info' : grid.cv_results_, 'whichChans' : whichChans, 'maxFreq' : maxFreq}

    with open(localDir + outputFileName, 'wb') as f:
        pickle.dump(bestEstimator, f)

def trainSpikeMethod(dataName, whichChans, estimator, skf, parameters, outputFileName):
    localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
    spikeFile = localDir + dataName
    spikeData = pd.read_pickle(spikeFile)

    spikes = spikeData['spikes']
    binCenters = spikeData['binCenters']
    spikeMat = spikeData['spikeMat']
    binWidth = spikeData['binWidth']

    del(spikeData)
    # get all columns of spikemat that aren't the labels
    nonLabelChans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

    X = spikeMat[nonLabelChans][whichChans]
    y = spikeMat['LabelsNumeric']

    grid=GridSearchCV(estimator, parameters, scoring = 'f1_macro', cv = skf, n_jobs = -1, verbose = 4)

    grid.fit(X,y)
    bestEstimator={'estimator' : grid.best_estimator_, 'info' : grid.cv_results_}

    with open(localDir + outputFileName, 'wb') as f:
        pickle.dump(bestEstimator, f)

def plotValidationCurve(estimator, estimatorInfo):
    keys = sorted(list(estimatorInfo['params'][0].keys()))
    print(keys)

    # TODO: kludge, fix it
    if keys[0][:11] == 'downSampler':
        try:
            keys = sorted(list(estimatorInfo['params'][0]['downSampler__kw_args'].keys()))
            if 'whichChans' in keys: keys.remove('whichChans')
        except:
            pass
        #pdb.set_trace()
    if len(keys) == 1:
        fi, ax = plt.subplots()
        plt.title("Validation Curve with " + estimator.__str__()[:3])
        plt.xlabel("Parameter")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        # TODO: handle dict paramater
        if estimator.steps[0].__str__()[2:13] == 'downSampler':
            param_range = [len(x['downSampler__kw_args']['keepChans']) for x in estimatorInfo['params']]
        else:
            param_range = [x[keys[0]] for x in estimatorInfo['params']]

        ax.semilogx(param_range, estimatorInfo['mean_train_score'], label="Training score",
                     color="darkorange", lw=lw)
        ax.fill_between(param_range, estimatorInfo['mean_train_score'] - estimatorInfo['std_train_score'],
                         estimatorInfo['mean_train_score'] + estimatorInfo['std_train_score'], alpha=0.2,
                         color="darkorange", lw=lw)
        ax.semilogx(param_range, estimatorInfo['mean_test_score'], label="Cross-validation score",
                     color="navy", lw=lw)
        ax.fill_between(param_range, estimatorInfo['mean_test_score'] - estimatorInfo['std_test_score'],
                         estimatorInfo['mean_test_score'] + estimatorInfo['std_test_score'], alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")

    elif len(keys) == 2:
        fi, ax = plt.subplots(nrows = 2, ncols = 1)
        ax[0].set_title("Validation Curve with " + estimator.__str__()[:3])

        if not estimator.steps[0].__str__()[2:13] == 'downSampler':
            nParam1 = len(np.unique(estimatorInfo['param_' + keys[0]].data))
        else:
            #TODO: this is an edge case kludge to deal with downsampler dictionary argument. change
            param1 = [len(x['keepChans']) for x in estimatorInfo['param_' + 'downSampler__kw_args'].data]
            nParam1 = len(np.unique(param1))

        if not estimator.steps[0].__str__()[2:13] == 'downSampler':
            nParam2 = len(np.unique(estimatorInfo['param_' + keys[1]].data))
        else:
            #TODO: this is an edge case kludge to deal with downsampler dictionary argument. change
            param2 = [x['freqFactor'] for x in estimatorInfo['param_' + 'downSampler__kw_args'].data]
            nParam2 = len(np.unique(param2))

        nParams = [nParam1, nParam2]

        plotTest = np.reshape(estimatorInfo['mean_test_score'],(nParams[0], nParams[1]))
        plotTrain = np.reshape(estimatorInfo['mean_train_score'],(nParams[0], nParams[1]))

        try:
            param1 = np.reshape(estimatorInfo['param_' + keys[0]].data,(nParams[1], nParams[0])).astype(np.float32)
        except:
            param1 = np.reshape(param1,(nParams[1], nParams[0])).astype(np.float32)

        try:
            param2 = np.reshape(estimatorInfo['param_' + keys[1]].data,(nParams[1], nParams[0])).astype(np.float32)
        except:
            param2 = np.reshape(param2,(nParams[1], nParams[0])).astype(np.float32)

        zMin = min(estimatorInfo['mean_test_score'])
        zMax = max(estimatorInfo['mean_test_score'])

        im = ax[0].pcolormesh(param1, param2, plotTest.transpose(), norm = colors.Normalize(vmin=zMin, vmax=zMax))

        ax[0].set_xlabel(keys[0])
        ax[0].set_ylabel('Test ' + keys[1])
        im = ax[1].pcolormesh(param1, param2, plotTrain.transpose(), norm = colors.Normalize(vmin=zMin, vmax=zMax))

        ax[1].set_xlabel(keys[0])
        ax[1].set_ylabel('Train ' + keys[1])
        fi.subplots_adjust(right = 0.8)
        cbar_ax = fi.add_axes([0.85, 0.15, 0.05, 0.7])
        fi.colorbar(im, cax = cbar_ax)
    return fi
