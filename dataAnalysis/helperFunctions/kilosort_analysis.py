import os, sys, pdb
from tempfile import mkdtemp
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import scipy.io
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from fractions import gcd
import seaborn as sns
from importlib import reload
from dataAnalysis.helperFunctions.helper_functions import *
import line_profiler
import pickle
import h5py

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


def coordsToIndices(xcoords, ycoords):

    xSpacing = np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), xcoords)
    ySpacing = np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), ycoords)
    xIdx = np.array(np.divide(xcoords, xSpacing), dtype = np.int)
    xIdx = xIdx - min(xIdx)
    yIdx = np.array(np.divide(ycoords, ySpacing), dtype = np.int)
    yIdx = yIdx - min(yIdx)

    return xIdx, yIdx

#@profile
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

#@profile
def getWaveForms(filePath, spikeStruct, nevIDs = None, dataType = np.int16, wfWin = (-40, 81), nWf = None, plotting = False, tempFolder = None):
    if tempFolder is None:
        tempFolder = mkdtemp()
    chMap = np.load(filePath + '/channel_map.npy', mmap_mode = 'r').squeeze()
    nCh = len(chMap)
    rawData = np.memmap(filename = filePath + '/' + spikeStruct['dat_path'], dtype = dataType, mode = 'r').reshape((-1, nCh))

    assert wfWin[0] < 0
    assert wfWin[1] > 0
    wfNSamples = -wfWin[0] + wfWin[1]

    #Read spike time-centered waveforms
    unitIDs = np.array(np.unique(spikeStruct['spikeCluster']))
    numUnits = len(unitIDs)
    waveFormTempFileNames = [os.path.join(tempFolder, 'tempfile_%s.dat' % idx) for idx in unitIDs]
    waveForms = [None for idx in range(numUnits)]
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

        waveForms[idx] = np.memmap(waveFormTempFileNames[idx], dtype='int16', mode='w+', shape=(curUnitNSpikes, wfNSamples, nCh))

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


    waveFormTempFileNames = [os.path.join(tempFolder, 'tempfileSpike_%s.dat' % idx) for idx, chan in enumerate(chMap)]
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
            spikes['Waveforms'][idx] = np.memmap(waveFormTempFileNames[idx], dtype='int16', mode='w+', shape=(nSpikesOnChan,wfNSamples,nCh))
            #spikes['Waveforms'][idx] = np.zeros((nSpikesOnChan,wfNSamples,nCh))
            for spikeIdx, spikeClass in enumerate(spikes['Classification'][idx]):
                #pdb.set_trace()
                classIdx = np.where(unitIDs==spikeClass)[0][0]
                spikes['Waveforms'][idx][spikeIdx,:,:] = waveForms[classIdx][waveFormIndices[spikeClass],:,:]
                waveFormIndices[spikeClass] = waveFormIndices[spikeClass] + 1
            spikes['Waveforms'][idx].flush()
    #pdb.set_trace()
    del waveForms
    #spikes['Waveforms'].flush()
    return spikes

#@profile
def numFromWaveClusSpikeFile(spikeFileName):
    return int(spikeFileName.split('times_NSX')[-1].split('.mat')[0])

def getWaveClusSpikes(filePath, nevIDs = None, plotting = False, getMUA = False, tempFolder = None):

    if nevIDs is None:
        spikeFileList = [f for f in os.listdir(filePath + '/wave_clus') if '.mat' in f and 'times_' in f]
        nevIDs = [numFromWaveClusSpikeFile(f) for f in spikeFileList]
    else:
        spikeFileList = [f for f in os.listdir(filePath + '/wave_clus') if '.mat' in f and 'times_' in f and numFromWaveClusSpikeFile(f) in nevIDs]

    nCh = len(nevIDs)

    if tempFolder is None:
        tempFolder = mkdtemp()

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

    unitIDs = []
    lastMaxUnitID = 0
    for idx, spikeFile in enumerate(spikeFileList):
        waveClusData = scipy.io.loadmat(filePath + '/wave_clus/' + spikeFile)
        spikes['ChannelID'][idx] = nevIDs[idx]
        unitsInFile = np.unique(waveClusData['cluster_class'][:,0]) + 1 + lastMaxUnitID

        if getMUA:
            spikes['Classification'][idx] = waveClusData['cluster_class'][:,0] + 1 + lastMaxUnitID
            spikes['TimeStamps'][idx] = waveClusData['cluster_class'][:,1] / 1e3
            spikes['Waveforms'][idx] = waveClusData['spikes']
        else:
            notMUAMask = waveClusData['cluster_class'][:,0] != 0
            spikes['Classification'][idx] = waveClusData['cluster_class'][notMUAMask,0] + 1 + lastMaxUnitID
            spikes['TimeStamps'][idx] = waveClusData['cluster_class'][notMUAMask,1] / 1e3
            spikes['Waveforms'][idx] = waveClusData['spikes'][notMUAMask, :]

        unitIDs+=unitsInFile.tolist()
        lastMaxUnitID = max(unitIDs)

    #pdb.set_trace()
    return spikes

#@profile
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

        colorPalette = sns.color_palette()
        for unitIdx, unitName in enumerate(unitsOnThisChan):

            unitMask = spikes['Classification'][ChanIdx] == unitName

            if acrossArray:
                for idx, channel in enumerate(spikes['ChannelID']):
                    curAx = ax[xIdx[idx], yIdx[idx]]
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, idx]
                    thisSpike = np.mean(waveForms, axis = 0)
                    thisError = np.std(waveForms, axis = 0)
                    timeRange = np.arange(len(thisSpike))
                    curAx.fill_between(timeRange, thisSpike-thisError, thisSpike+thisError, alpha=0.4, facecolor=colorPalette[unitIdx], label='chan %s, unit %s' % (channel, unitName))
                    curAx.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])

                sns.despine()
                for curAx in ax.flatten():
                    curAx.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
                plt.tight_layout()

            else:
                if len(spikes['Waveforms'][ChanIdx].shape) == 3:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, ChanIdx]
                else:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :]
                thisSpike = np.mean(waveForms, axis = 0)
                thisError = np.std(waveForms, axis = 0)
                timeRange = np.arange(len(thisSpike))
                colorPalette = sns.color_palette()
                ax.fill_between(timeRange, thisSpike - thisError, thisSpike + thisError, alpha=0.4, facecolor=colorPalette[unitIdx], label='chan %s, unit %s' % (channel, unitName))
                ax.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])

        if showNow:
            plt.show()


#@profile
def plotISIHistogram(spikes, channel, showNow = False, ax = None, bins = None, kde_kws = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    idx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][idx])

    if unitsOnThisChan is not None:
        colorPalette = sns.color_palette()
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            unitMask = spikes['Classification'][idx] == unitName
            theseTimes = spikes['TimeStamps'][idx][unitMask]
            theseISI = np.diff(theseTimes)
            sns.distplot(theseISI, bins = bins, ax = ax, color = colorPalette[unitIdx],  kde_kws = kde_kws)
            if bins is not None:
                ax.set_xlim(min(bins), max(bins))
        if showNow:
            plt.show()

#@profile
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

#@profile
def plotRaster(spikes, trialStats, alignTo, channel, separateBy = None, windowSize = (-0.25, 1), showNow = False, ax = None, maxTrial = None):

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][ChanIdx])

    if separateBy is not None:
        uniqueCategories = pd.Series(trialStats.loc[:,separateBy].unique())
        uniqueCategories.dropna(inplace = True)
        curLine = {category : 0 for category in uniqueCategories}

        if ax is None:
            fig, ax = plt.subplots(len(uniqueCategories),1)

        else:
            assert len(ax) == len(uniqueCategories)

    else: # only one plot
        if ax is None:
            fig, ax = plt.subplots()


    # timeWindow in milliseconds

    timeWindow = list(range(int(windowSize[0] * 1e3), int(windowSize[1] * 1e3) + 1))
    if unitsOnThisChan is not None:
        colorPalette = sns.color_palette()
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            unitMask = spikes['Classification'][ChanIdx] == unitName
            # time stamps in milliseconds
            allSpikeTimes = np.array(spikes['TimeStamps'][ChanIdx][unitMask] * 1e3, dtype = np.int64)
            for idx, startTime in enumerate(trialStats[alignTo]):
                try:
                    #print('Plotting trial %s' % idx)
                    #convert start time from index to milliseconds
                    trialTimeMask = np.logical_and(allSpikeTimes > startTime / 3e1 + timeWindow[0], allSpikeTimes < startTime / 3e1 + timeWindow[-1])
                    trialSpikeTimes = allSpikeTimes[trialTimeMask]
                    #print(trialSpikeTimes - startTime / 3e1)
                    if separateBy is not None:
                        #pdb.set_trace()
                        curCategory = trialStats.loc[idx, separateBy]
                        whichAx = pd.Index(uniqueCategories).get_loc(curCategory)
                        axToPlotOn = ax[whichAx]
                        lineToPlot = curLine[curCategory]
                        curLine[curCategory] += 1
                    else:
                        axToPlotOn = ax
                        lineToPlot = idx
                    axToPlotOn.vlines(trialSpikeTimes - startTime / 3e1, lineToPlot, lineToPlot + 1, colors = [colorPalette[unitIdx]], linewidths = [0.5])
                    if maxTrial is not None:
                        if idx >= maxTrial -1:
                            break
                except:
                    #pdb.set_trace()
                    pass
            #reset line counts for next pass through for the next unit on this chan
            if separateBy is not None:
                curLine = {category : 0 for category in uniqueCategories}

    if separateBy is not None:
        for idx, thisAx in enumerate(ax):
            thisAx.set_xlabel('Time (milliseconds) aligned to ' + alignTo)
            thisAx.set_ylabel('Trial')
            thisAx.set_title(uniqueCategories[idx])
    else:
        ax.set_xlabel('Time (milliseconds) aligned to ' + alignTo)
        ax.set_ylabel('Trial')

    if showNow:
        plt.show()

    return ax

#@profile
def plotFR(spikes, trialStats, alignTo, channel, separateBy = None, windowSize = (-0.25, 1), showNow = False, ax = None, twin = False, maxTrial = None):
    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][ChanIdx])

    if separateBy is not None:
        uniqueCategories = pd.Series(trialStats.loc[:,separateBy].unique())
        uniqueCategories.dropna(inplace = True)

        if ax is not None and twin:
            for idx, thisAx in enumerate(ax):
                ax[idx] = thisAx.twinx()

        if ax is None:
            fig, ax = plt.subplots(len(uniqueCategories),1)

        else:
            assert len(ax) == len(uniqueCategories)

    else: # only one plot

        if ax is not None and twin:
            ax = ax.twinx()

        if ax is None:
            fig, ax = plt.subplots()

    if maxTrial is not None:
        maxTrial = min(len(trialStats.index), maxTrial)
        trialStats = trialStats.iloc[:maxTrial, :]
    # window size specified in seconds
    # time window in milliseconds
    timeWindow = list(range(int(windowSize[0] * 1e3), int(windowSize[1] * 1e3) + 1))
    if unitsOnThisChan is not None:

        FR = [pd.DataFrame(index = trialStats.index, columns = timeWindow[:-1] + ['discard']) for i in unitsOnThisChan]

        for x in FR:
            x['discard'] = False

        for unitIdx, unitName in enumerate(unitsOnThisChan):
            unitMask = spikes['Classification'][ChanIdx] == unitName
            # spike times in milliseconds
            allSpikeTimes = np.array(spikes['TimeStamps'][ChanIdx][unitMask] * 1e3, dtype = np.int64)
            for idx, startTime in enumerate(trialStats[alignTo]):
                try:
                    #print('Calculating raster for trial %s' % idx)
                    #pdb.set_trace()
                    trialTimeMask = np.logical_and(allSpikeTimes > startTime / 3e1 + timeWindow[0], allSpikeTimes < startTime / 3e1 + timeWindow[-1])
                    if trialTimeMask.sum() != 0:
                        trialSpikeTimes = allSpikeTimes[trialTimeMask] - startTime / 3e1
                        FR[unitIdx].iloc[idx, :-1] = np.histogram(trialSpikeTimes, timeWindow)[0]
                    else:
                        FR[unitIdx].iloc[idx, -1] = True

                    if maxTrial is not None:
                        if idx >= maxTrial -1:
                            break
                except:
                    #pdb.set_trace()
                    FR[unitIdx].iloc[idx, -1] = True
                #pdb.set_trace()

    for idx, x in enumerate(FR):
        FR[idx].drop(x.index[x['discard'] == True], axis = 0, inplace = True)
        FR[idx].drop('discard', axis = 1, inplace = True)

    kernelWidth = 50e-3 # seconds
    if separateBy is not None:
        meanFR = {category : [pd.Series(index = timeWindow[:-1]) for i in unitsOnThisChan] for category in uniqueCategories}
        for category in uniqueCategories:
            for idx, unit in enumerate(unitsOnThisChan):
                meanFR[category][idx] = FR[idx].loc[trialStats[separateBy] == category].mean(axis = 0)
                meanFR[category][idx] = gaussian_filter1d(meanFR[category][idx], kernelWidth * 1e3)
    else:
        meanFR = {'all' : [gaussian_filter1d(x.mean(axis = 0), kernelWidth * 1e3) for x in FR]}
    #pdb.set_trace()
    colorPalette = sns.color_palette()

    for category, meanFRThisCategory in meanFR.items():
        if separateBy is not None:
            categoryIndex = pd.Index(uniqueCategories).get_loc(category)
            thisAx = ax[categoryIndex]
        else:
            thisAx = ax

        for unitIdx, x in enumerate(meanFRThisCategory):
            thisAx.plot(timeWindow[:-1], x * 1e3, linewidth = 1, color = colorPalette[unitIdx])
            thisAx.set_ylabel('Average Firing rate (spk/sec)')
        thisAx.set_title('spikes for ' + category + ' trials')
    if showNow:
        plt.show()
    return ax, FR

#@profile
def spikePDFReport(filePath, spikes, spikeStruct, plotRastersAlignedTo = None, plotRastersSeparatedBy = None, trialStats = None, enableFR = False):
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
                        plotAx = plotRaster(spikes, trialStats, alignTo = plotRastersAlignedTo, windowSize = (-0.5, 2), channel = channel, separateBy = plotRastersSeparatedBy)
                        if enableFR:
                            plotFR(spikes, trialStats, alignTo = plotRastersAlignedTo, windowSize = (-0.5, 2), channel = channel, separateBy = plotRastersSeparatedBy, ax = plotAx, twin = True)
                        pdf.savefig()
                        plt.close()

if __name__ == "__main__":
    from dataAnalysis.helperFunctions.motor_encoder import *
    #spikeStructNForm = loadKSDir('D:/KiloSort/Trial001_NForm', loadPCs = True)
    #nevIDs = list(range(65,97))
    #spikesNForm = getWaveForms('D:/KiloSort/Trial001_NForm', spikeStructNForm, nevIDs = None, wfWin = (-30, 80), plotting = False)
    #isiBins = np.linspace(0, 50e-3, 100)
    #plotISIHistogram(spikesNForm, channel = 25, bins = isiBins,kde_kws = {'clip' : (isiBins[0] * 0.8, isiBins[-1] * 1.2), 'bw' : 'silverman'} )
    #plt.show()

    #spikePDFReport('D:/KiloSort/Trial001_NForm', spikesNForm, spikeStructNForm)

    #pdb.set_trace()
    #spikesUtah = getWaveClusSpikes('D:/Staging', nevIDs = [71,76], plotting = False, getMUA = True, tempFolder = 'D:/Staging/wave_clus/temp')


    try:
        spikeStructUtah = pickle.load(open('D:/Kilosort/Trial001_Utah/Trial001_spikeStructUtah.pickle', 'rb'))
        spikesUtah      = pickle.load(open('D:/Kilosort/Trial001_Utah/Trial001_spikesUtah.pickle', 'rb'))
    except:
        spikeStructUtah = loadKSDir('D:/Kilosort/Trial001_Utah', loadPCs = True)
        nevIDs = list(range(1,65))
        spikesUtah = getWaveForms('D:/Kilosort/Trial001_Utah', spikeStructUtah, nevIDs = None, wfWin = (-30, 80), plotting = False, tempFolder = 'E:/temp')

        pickle.dump(spikeStructUtah, open('D:/Kilosort/Trial001_Utah/Trial001_spikeStructUtah.pickle', 'wb'))
        pickle.dump(spikesUtah, open('D:/Kilosort/Trial001_Utah/Trial001_spikesUtah.pickle', 'wb'))
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

    #pdb.set_trace()
    try:
        trialStats  = pd.read_pickle('D:/Kilosort/Trial001_trialStats.pickle')
        trialEvents = pd.read_pickle('D:/Kilosort/Trial001_trialEvents.pickle')
    except:
        motorData = getMotorData(ns5FilePath, inputIDs, 0 , 'all')
        trialStats, trialEvents = getTrials(motorData)
        trialStats.to_pickle('D:/Kilosort/Trial001_trialStats.pickle')
        trialEvents.to_pickle('D:/Kilosort/Trial001_trialEvents.pickle')

    plotChan = 20
    #plotSpike(spikesUtah, channel = plotChan)
    #isiBins = np.linspace(0, 50e-3, 100)
    #plotISIHistogram(spikesUtah, channel = plotChan, bins = isiBins,kde_kws = {'clip' : (isiBins[0] * 0.8, isiBins[-1] * 1.2), 'bw' : 'silverman'} )
    plotAx = plotRaster(spikesUtah, trialStats, alignTo = 'FirstOnset', windowSize = (-0.5, 2), channel = plotChan, separateBy = 'Direction', maxTrial = 15)
    plotFR(spikesUtah, trialStats, alignTo = 'FirstOnset', windowSize = (-0.5, 2), channel = plotChan, separateBy = 'Direction', ax = plotAx, twin = True, maxTrial = 15)
    plt.show()
#spikePDFReport('D:/KiloSort/Trial001_Utah', spikesUtah, spikeStructUtah, plotRastersAlignedTo = 'FirstOnset', trialStats = trialStats)
