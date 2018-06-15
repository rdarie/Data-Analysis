import os, sys, pdb
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from fractions import gcd
import seaborn as sns
from importlib import reload
from dataAnalysis.helperFunctions.helper_functions import *
from dataAnalysis.helperFunctions.motor_encoder import *

def loadParamsPy(filePath):

    """
    # Old implementation, treats params.py as a package and cannot be overriden when processing a new folder.
    pdb.set_trace()

    sys.path.insert(0, filePath)

    try:
        reload(params)
    except:
         import params

    sys.path.remove(filePath)
    """

    with open(filePath + '/params.py') as f:
        paramsText = f.read()

    exec(paramsText)
    del paramsText

    return locals()

def loadKSDir(filePath, excludeNoise = True, loadPCs = False):
    mMapMode = 'r'
    params = loadParamsPy(filePath)

    spikeTimesSamples = np.load(filePath + '/spike_times.npy', mmap_mode = mMapMode).squeeze()
    spikeTimes = spikeTimesSamples.astype(np.float)/params['sample_rate']

    spikeTemplates = np.load(filePath + '/spike_templates.npy', mmap_mode = mMapMode).squeeze()
    try:
        spikeCluster = np.load(filePath + '/spike_clusters.npy', mmap_mode = mMapMode)
    except:
        spikeCluster = spikeTemplates

    tempScalingAmps = np.load(filePath + '/amplitudes.npy', mmap_mode = mMapMode).squeeze()

    if loadPCs:
        pcFeat = np.load(filePath + '/pc_features.npy', mmap_mode = mMapMode) # nSpikes x nFeatures x nLocalChannels
        pcFeatInd = np.load(filePath + '/pc_feature_ind.npy', mmap_mode = mMapMode) # nTemplates x nLocalChannels
    else:
        pcFeat = None
        pcFeatInd = None

    if os.path.isfile(filePath + '/cluster_groups.csv'):
        cgsFile = filePath + '/cluster_groups.csv'
    else:
        cgsFile = None

    if cgsFile:
        try:
            clusterInfo = pd.read_csv(cgsFile, sep='\t')
            hasClusterInfo = True
        except:
            hasClusterInfo = False

        if excludeNoise and hasClusterInfo:
            #pdb.set_trace()
            noiseClusters = clusterInfo[clusterInfo.loc[:,'group'] == 'noise'].loc[:, 'cluster_id'].values
            # identify which spikes are in noise clusters
            spikeMask = np.array([not x in noiseClusters for x in spikeCluster], dtype = np.bool)

            # discard these spike

            #print('originally had %d spikes' % len(spikeTimes))
            spikeTimesSamples = spikeTimesSamples[spikeMask]
            spikeTimes = spikeTimes[spikeMask]
            spikeTemplates =spikeTemplates[spikeMask]
            tempScalingAmps = tempScalingAmps[spikeMask]

            spikeCluster = spikeCluster[spikeMask]

            #print('now have %d spikes' % len(spikeTimes))

            if loadPCs:
                pcFeat = pcFeat[spikeMask, :, :]

            clusterInfo = clusterInfo[clusterInfo.loc[:,'group'] != 'noise']

        if not hasClusterInfo:
            spikeCluster = spikeTemplates
            clusterIDs = np.unique(spikeTemplates)
            clusterInfo = pd.DataFrame(np.unique(spikeTemplates), columns = ['cluster_id'])
            clusterInfo = clusterInfo.assign(group = 'good')

    coords = np.load(filePath + '/channel_positions.npy', mmap_mode = mMapMode)
    xcoords = [coord[0] for coord in coords]
    ycoords = [coord[1] for coord in coords]


    temps = np.load(filePath + '/templates.npy', mmap_mode = mMapMode)
    winv = np.load(filePath + '/whitening_mat_inv.npy', mmap_mode = mMapMode)

    spikeStruct = {
        'dat_path'          : params['dat_path'],
        'dtype'             : params['dtype'],
        'hp_filtered'       : params['hp_filtered'],
        'n_channels_dat'    : params['n_channels_dat'],
        'offset'            : params['offset'],
        'sample_rate'       : params['sample_rate'],
        'spikeTimes'        : spikeTimes,
        'spikeTimesSamples' : spikeTimesSamples,
        'spikeTemplates'    : spikeTemplates,
        'spikeCluster'      : spikeCluster,
        'tempScalingAmps'   : tempScalingAmps,
        'clusterInfo'       : clusterInfo,
        'xcoords'           : xcoords,
        'ycoords'           : ycoords,
        'temps'             : temps,
        'winv'              : winv,
        'pcFeat'            : pcFeat,
        'pcFeatInd'         : pcFeatInd
        }
    return spikeStruct

def getWaveForms(filePath, spikeStruct, nevIDs = None, dataType = np.int16, wfWin = (-40, 81), nWf = None, plotting = False):
    chMap = np.load(filePath + '/channel_map.npy', mmap_mode = 'r').squeeze()
    nCh = len(chMap)
    rawData = np.memmap(filename = filePath + '/' + spikeStruct['dat_path'], dtype = dataType, mode = 'r').reshape((-1, nCh))

    assert wfWin[0] < 0
    assert wfWin[1] > 0
    wfNSamples = -wfWin[0] + wfWin[1]

    #Read spike time-centered waveforms
    unitIDs = np.array(np.unique(spikeStruct['spikeCluster']))
    numUnits = len(unitIDs)
    waveForms = [[] for i in range(numUnits)]
    meanWaveForms = [None for i in range(numUnits)]
    stdWaveForms = [None for i in range(numUnits)]
    unitChannel = [None for i in range(numUnits)]

    spikes = {
        'ChannelID' : [i for i in range(nCh)],
        'Classification' : [[] for i in range(nCh)],
        'NEUEVWAV_HeaderIndices' : [None for i in range(nCh)],
        'TimeStamps' : [[] for i in range(nCh)],
        'Units' : 'nV',
        'Waveforms' : [None for i in range(nCh)],
#        'meanWaveforms' : [None for i in range(nCh)],
#        'stdWaveforms' : [None for i in range(nCh)],
        'basic_headers' : [],
        'extended_headers' : []
        }

    #pdb.set_trace()
    if nevIDs:
        rootFolder = filePath.split('KiloSort')[0]
        rootName = spikeStruct['dat_path'].split('.dat')[0]
        if '_' in rootName:
            rootName = rootName.split('_')[0]
        nevFileName = rootFolder + rootName + '.nev'
        #pdb.set_trace()
        nevFile = NevFile(nevFileName)

        spikes['basic_headers'] = nevFile.basic_header
        spikes['extended_headers'] = [x for x in nevFile.extended_headers[:-2] if x['ElectrodeID'] in nevIDs]

        spikes['ChannelID'] = nevIDs

    for idx, curUnitIdx in enumerate(unitIDs):
        curSpikeTimes = spikeStruct['spikeTimesSamples'][spikeStruct['spikeCluster'] == curUnitIdx]
        curUnitNSpikes = len(curSpikeTimes)
        waveForms[idx] = np.zeros((curUnitNSpikes, wfNSamples, nCh))

        for spikeIdx, curSpikeTime in enumerate(curSpikeTimes):
            #pdb.set_trace()
            tmpWf = rawData[int(curSpikeTime + wfWin[0]) : int(curSpikeTime + wfWin [1]), :]
            waveForms[idx][spikeIdx, :tmpWf.shape[0], :] = tmpWf

        meanWaveForms[idx] = np.mean(waveForms[idx], axis = 0)
        stdWaveForms[idx] = np.std(waveForms[idx], axis = 0)

        #squaredSum = np.sum(meanWaveForms[idx] ** 2, axis = 0)
        unitChannel[idx] = np.argmin(np.min(meanWaveForms[idx], axis = 0))

        if plotting:
            thisSpike = meanWaveForms[idx][:, unitChannel[idx]]
            thisError = stdWaveForms[idx][:, unitChannel[idx]]
            x = np.arange(len(thisSpike))
            plt.fill_between(x, thisSpike-thisError, thisSpike+thisError, alpha=0.4, label='chan %s' % unitChannel[idx])
            plt.plot(x, thisSpike, 'k-')
            plt.legend()
            plt.show()

    # redistribute data on a channel by channel basis to conform to the convention of the blackrock NEV file.
    for idx, chan in enumerate(chMap): # for each channel
        setMask = unitChannel == chan # which indices correspond to units on this channel
        if setMask.any():
            unitsOnThisChan = unitIDs[setMask] # which units are on this channel
            chanMask = np.array([x in unitsOnThisChan for x in spikeStruct['spikeCluster']])# which spikes are on this channel
            spikes['Classification'][idx] = spikeStruct['spikeCluster'][chanMask]
            spikes['TimeStamps'][idx] = spikeStruct['spikeTimes'][chanMask]
            nSpikesOnChan = len(spikes['TimeStamps'][idx])
            waveFormIndices = {i:0 for i in unitsOnThisChan}
            spikes['Waveforms'][idx] = np.zeros((nSpikesOnChan,wfNSamples,nCh))
            for spikeIdx, spikeClass in enumerate(spikes['Classification'][idx]):
                #pdb.set_trace()
                classIdx = np.where(unitIDs==spikeClass)[0][0]
                spikes['Waveforms'][idx][spikeIdx,:,:] = waveForms[classIdx][waveFormIndices[spikeClass],:,:]
                waveFormIndices[spikeClass] = waveFormIndices[spikeClass] + 1
    #pdb.set_trace()
    return spikes

def coordsToIndices(xcoords, ycoords):

    xSpacing = np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), xcoords)
    ySpacing = np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), ycoords)
    xIdx = np.array(np.divide(xcoords, xSpacing), dtype = np.int)
    xIdx = xIdx - min(xIdx)
    yIdx = np.array(np.divide(ycoords, ySpacing), dtype = np.int)
    yIdx = yIdx - min(yIdx)

    return xIdx, yIdx

def plotSpike(spikes, channel, showNow = False, ax = None, acrossArray = False, xcoords = None, ycoords = None):

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][ChanIdx])

    if acrossArray:
        sns.set_style("dark", {"axes.facecolor": ".9"})
        matplotlib.rc('xtick', labelsize=5)
        matplotlib.rc('ytick', labelsize=5)
        matplotlib.rc('legend', fontsize=5)
        matplotlib.rc('axes', xmargin=.01)
        matplotlib.rc('axes', ymargin=.01)
        # Check that we didn't ask to plot the spikes across channels into a single axis
        assert ax is None
        # Check that we have waveform data everywhere
        assert len(spikes['Waveforms'][ChanIdx].shape) == 3

    if ax is None and not acrossArray:
        fig, ax = plt.subplots()
    if ax is not None and not acrossArray:
        fig = ax.figure

    if unitsOnThisChan is not None:

        if acrossArray:
            xIdx, yIdx = coordsToIndices(xcoords, ycoords)
            fig, ax = plt.subplots(nrows = max(np.unique(xIdx)) + 1, ncols = max(np.unique(yIdx)) + 1)

        for unitName in unitsOnThisChan:

            unitMask = spikes['Classification'][ChanIdx] == unitName

            if acrossArray:
                for idx, channel in enumerate(spikes['ChannelID']):
                    curAx = ax[xIdx[idx], yIdx[idx]]
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, idx]
                    thisSpike = np.mean(waveForms, axis = 0)
                    thisError = np.std(waveForms, axis = 0)
                    timeRange = np.arange(len(thisSpike))
                    curAx.fill_between(timeRange, thisSpike-thisError, thisSpike+thisError, alpha=0.4, label='chan %s, unit %s' % (channel, unitName))
                    curAx.plot(timeRange, thisSpike, 'k-')

                sns.despine()
                for curAx in ax.flatten():
                    curAx.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
                plt.tight_layout()

            else:
                waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, ChanIdx]
                thisSpike = np.mean(waveForms, axis = 0)
                thisError = np.std(waveForms, axis = 0)
                timeRange = np.arange(len(thisSpike))
                ax.fill_between(timeRange, thisSpike - thisError, thisSpike + thisError, alpha=0.4, label='chan %s, unit %s' % (channel, unitName))
                ax.plot(timeRange, thisSpike, 'k-')

        if showNow:
            plt.show()

def plotISIHistogram(spikes, channel, showNow = False, ax = None, bins = None, kde_kws = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    idx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][idx])

    if unitsOnThisChan is not None:
        for unitName in unitsOnThisChan:
            unitMask = spikes['Classification'][idx] == unitName
            theseTimes = spikes['TimeStamps'][idx][unitMask]
            theseISI = np.diff(theseTimes)
            sns.distplot(theseISI, bins = bins, ax = ax, kde_kws = kde_kws)
            if bins is not None:
                ax.set_xlim(min(bins), max(bins))
        if showNow:
            plt.show()

def plotSpikePanel(xcoords, ycoords, spikes):
    sns.set_style("dark", {"axes.facecolor": ".9"})
    matplotlib.rc('xtick', labelsize=5)
    matplotlib.rc('ytick', labelsize=5)
    matplotlib.rc('legend', fontsize=5)
    matplotlib.rc('axes', xmargin=.01)
    matplotlib.rc('axes', ymargin=.01)
    xIdx, yIdx = coordsToIndices(xcoords, ycoords)
    fig, ax = plt.subplots(nrows = max(np.unique(xIdx)) + 1, ncols = max(np.unique(yIdx)) + 1)

    for idx, channel in enumerate(spikes['ChannelID']):
        curAx = ax[xIdx[idx], yIdx[idx]]
        plotSpike(spikes, channel, ax = curAx)

    sns.despine()
    for curAx in ax.flatten():
        curAx.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.tight_layout()

def plotRaster(spikes, trialStats, alignTo, channel, windowSize = (-0.1, 0.5), showNow = False, ax = None):

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][ChanIdx])

    if ax is None:
        fig, ax = plt.subplots()

    if unitsOnThisChan is not None:
        for unitName in unitsOnThisChan:
            unitMask = spikes['Classification'][ChanIdx] == unitName
            allSpikeTimes = np.array(spikes['TimeStamps'][ChanIdx][unitMask] * 3e4, dtype = np.int64)
            for idx, startTime in enumerate(trialStats[alignTo]):
                trialTimeMask = np.logical_and(allSpikeTimes > startTime + windowSize[0] * 3e4, allSpikeTimes < startTime + windowSize[1] * 3e4)
                trialSpikeTimes = allSpikeTimes[trialTimeMask]
                ax.vlines(trialSpikeTimes - startTime, idx, idx + 1)

    ax.set_xlabel('Time (samples) aligned to ' + alignTo)
    ax.set_ylabel('Trial')

    if showNow:
        plt.show()

    return ax

def spikePDFReport(filePath, spikes, spikeStruct, plotRastersAlignedTo = None, trialStats = None):
    pdfName = filePath + '/' + spikeStruct['dat_path'].split('.')[0] + '.pdf'
    with PdfPages(pdfName) as pdf:
        plotSpikePanel(spikeStruct['xcoords'], spikeStruct['ycoords'], spikes)
        pdf.savefig()
        plt.close()

        for idx, channel in enumerate(spikes['ChannelID']):
            unitsOnThisChan = np.unique(spikes['Classification'][idx])
            if unitsOnThisChan is not None:
                if len(unitsOnThisChan) > 0:
                    fig, ax = plt.subplots(nrows = 1, ncols = 2)
                    plotSpike(spikes, channel = channel, ax = ax[0])
                    isiBins = np.linspace(0, 50e-3, 100)
                    kde_kws = {'clip' : (isiBins[0] * 0.8, isiBins[-1] * 1.2), 'bw' : 'silverman'}
                    plotISIHistogram(spikes, channel = channel, bins = isiBins, kde_kws = kde_kws, ax = ax[1])
                    pdf.savefig()
                    plt.close()

                    if len(spikes['Waveforms'][idx].shape) == 3:
                        plotSpike(spikes, channel = channel, acrossArray = True, xcoords = spikeStruct['xcoords'], ycoords = spikeStruct['ycoords'])
                        pdf.savefig()
                        plt.close()

                    if plotRastersAlignedTo is not None and trialStats is not None:
                        plotRaster(spikes, trialStats, alignTo = plotRastersAlignedTo, channel = channel)
                        pdf.savefig()
                        plt.close()

if __name__ == "__main__":
    #spikeStructNForm = loadKSDir('D:/KiloSort/Trial001_NForm', loadPCs = True)
    #nevIDs = list(range(65,97))
    #spikesNForm = getWaveForms('D:/KiloSort/Trial001_NForm', spikeStructNForm, nevIDs = None, wfWin = (-30, 80), plotting = False)
    #isiBins = np.linspace(0, 50e-3, 100)
    #plotISIHistogram(spikesNForm, channel = 25, bins = isiBins,kde_kws = {'clip' : (isiBins[0] * 0.8, isiBins[-1] * 1.2), 'bw' : 'silverman'} )
    #plt.show()

    #spikePDFReport('D:/KiloSort/Trial001_NForm', spikesNForm, spikeStructNForm)

    spikeStructUtah = loadKSDir('D:/KiloSort/Trial001_Utah', loadPCs = True)
    nevIDs = list(range(65,97))
    spikes = getWaveForms('D:/KiloSort/Trial001_Utah', spikeStructUtah, nevIDs = None, wfWin = (-30, 80), plotting = False)
    #spikePDFReport('D:/KiloSort/Trial001_Utah', spikesUtah, spikeStructUtah)

    ns5FilePath = 'D:/KiloSort/Trial001.ns5'
    inputIDs = {
        'A+' : 139,
        'A-' : 140,
        'B+' : 141,
        'B-' : 142,
        'Z+' : 143,
        'Z-' : 144,
        'leftLED' : 132,
        'leftBut' : 130,
        'rightLED' : 131,
        'rightBut' : 129,
        'simiTrigs' : 136,
        }

    motorData = getMotorData(ns5FilePath, inputIDs, 0 , 'all')
    trialStats, trialEvents = getTrials(motorData)

    #plotRaster(spikes, trialStats, alignTo = 'FirstOnset', channel = 28)
    spikePDFReport('D:/KiloSort/Trial001_Utah', spikes, spikeStructUtah, plotRastersAlignedTo = 'FirstOnset', trialStats = trialStats)