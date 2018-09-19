import os, sys, pdb
from tempfile import mkdtemp
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats
import pandas as pd
import scipy.io
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from fractions import gcd
import seaborn as sns
from importlib import reload
import dataAnalysis.helperFunctions.motor_encoder as mea
import dataAnalysis.helperFunctions.helper_functions as hf
import line_profiler
import pickle
import h5py
import traceback

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

def getSpikeMetaData(filePath, ns5FileName, KSDir):

    coords = np.load(filePath + '/'+ KSDir + '/channel_positions.npy', mmap_mode = mMapMode)
    xcoords = [coord[0] for coord in coords]
    ycoords = [coord[1] for coord in coords]
    spikeStruct = {
    'dat_path'          : filePath + '/' + ns5FileName,
    'dtype'             : None,
    'xcoords'           : xcoords,
    'ycoords'           : ycoords
    }
    return spikeStruct

#@profile
def loadKSDir(filePath, excludeNoise = True, loadPCs = False):
    mMapMode = 'r'
    params = loadParamsPy(filePath)

    spikeTimesSamples = np.load(filePath + '/spike_times.npy', mmap_mode = mMapMode).squeeze()
    spikeTimes = spikeTimesSamples.astype(np.float)/params['sample_rate']

    spikeTemplates = np.load(filePath + '/spike_templates.npy', mmap_mode = mMapMode).squeeze()
    try:
        spikeCluster = np.load(filePath + '/spike_clusters.npy', mmap_mode = mMapMode)
    except Exception:
        traceback.print_exc()
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
        except Exception:
            traceback.print_exc()
            clusterInfo = None
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
        #'meanWaveforms' : [None for i in range(nCh)],
        #'stdWaveforms' : [None for i in range(nCh)],
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

import re
# From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def getWaveClusSpikes(filePath, nevIDs = None, plotting = False, excludeClus = [0], tempFolder = None):
    # TODO: not memory mapped yet
    if nevIDs is None:
        spikeFileList = [f for f in os.listdir(filePath) if '.mat' in f and 'times_' in f]
        nevIDs = [numFromWaveClusSpikeFile(f) for f in spikeFileList]
    else:
        spikeFileList = [f for f in os.listdir(filePath) if '.mat' in f and 'times_' in f and numFromWaveClusSpikeFile(f) in nevIDs]

    spikeFileList.sort(key=natural_keys)
    nCh = len(nevIDs)

    if tempFolder is None:
        tempFolder = mkdtemp()

    spikes = {
        'ChannelID' : [i for i in range(nCh)],
        'Classification' : [[] for i in range(nCh)],
        'NEUEVWAV_HeaderIndices' : [None for i in range(nCh)],
        'TimeStamps' : [[] for i in range(nCh)],
        'Units' : 'uV',
        'Waveforms' : [None for i in range(nCh)],
        #'meanWaveforms' : [None for i in range(nCh)],
        #'stdWaveforms' : [None for i in range(nCh)],
        'basic_headers' : {'TimeStampResolution': 3e4},
        'extended_headers' : []
        }

    unitIDs = []
    lastMaxUnitID = 0
    for idx, spikeFile in enumerate(spikeFileList):
        waveClusData = scipy.io.loadmat(filePath + spikeFile)
        spikes['ChannelID'][idx] = nevIDs[idx]
        unitsInFile = np.unique(waveClusData['cluster_class'][:,0]) + 1 + lastMaxUnitID

        #pdb.set_trace()
        if excludeClus:
            notMUAMask = np.logical_not(np.isin(waveClusData['cluster_class'][:,0], excludeClus))
        else:
            notMUAMask = np.full(len(waveClusData['cluster_class'][:,0]), True, dtype = np.bool)
        #pdb.set_trace()
        spikes['Classification'][idx] = waveClusData['cluster_class'][notMUAMask,0] + 1 + lastMaxUnitID
        spikes['TimeStamps'][idx] = waveClusData['cluster_class'][notMUAMask,1] / 1e3
        spikes['Waveforms'][idx] = waveClusData['spikes'][notMUAMask, :]

        # note that the units of TimeStamps is in SECONDS
        unitIDs+=unitsInFile.tolist()
        lastMaxUnitID = max(unitIDs)

    return spikes

#@profile
def plotSpike(spikes, channel, showNow = False, ax = None, acrossArray = False, xcoords = None, ycoords = None, axesLabel = False):

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
                    timeRange = np.arange(len(thisSpike)) / 3e4 * 1e3
                    curAx.fill_between(timeRange, thisSpike - 2*thisError, thisSpike + 2*thisError, alpha=0.4, facecolor=colorPalette[unitIdx], label='chan %s, unit %s' % (channel, unitName))
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
                timeRange = np.arange(len(thisSpike)) / 3e4 * 1e3
                colorPalette = sns.color_palette()
                ax.fill_between(timeRange, thisSpike - 2*thisError, thisSpike + 2*thisError, alpha=0.4, facecolor=colorPalette[unitIdx], label='chan %s, unit %s' % (channel, unitName))
                ax.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])
                if axesLabel:
                    ax.set_ylabel(spikes['Units'])
                    ax.set_xlabel('Time (msec)')
                    ax.set_title('Units on channel %d' % channel)

        if showNow:
            plt.show()

#@profile
def plotISIHistogram(spikes, channel, showNow = False, ax = None,
    bins = None, kde = False, kde_kws = None):
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
            theseISI = np.diff(theseTimes) * 1e3 # units of msec
            #pdb.set_trace()
            sns.distplot(theseISI, bins = bins, ax = ax,
                color = colorPalette[unitIdx], kde = kde, kde_kws = kde_kws)
            """
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(np.histogram(theseISI, bins = bins))
            rv = stats.gamma(fit_alpha, loc = fit_loc, scale = fit_beta)
            ax.plot(bins, rv.pdf(bins), 'k-', lw=2)
            #pdb.set_trace()

            plt.hist(theseISI, bins = bins, color = colorPalette[unitIdx], density = False)
            """
            plt.xlabel('ISI (msec)')
            yAxLabel = 'Count (normalized)' if kde else 'Count'
            plt.ylabel(yAxLabel)
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
    axHighLims = np.empty(ax.shape)
    axHighLims[:] = np.nan
    axLowLims = np.empty(ax.shape)
    axLowLims[:] = np.nan

    for idx, channel in enumerate(spikes['ChannelID']):
        curAx = ax[xIdx[idx], yIdx[idx]]
        plotSpike(spikes, channel, ax = curAx)
        curAxLim = curAx.get_ylim()
        axHighLims[xIdx[idx], yIdx[idx]] = curAxLim[1]
        axLowLims[xIdx[idx], yIdx[idx]] = curAxLim[0]
        xLim = curAx.get_xlim()
    sns.despine()

    newAxMin = np.nanmean(axLowLims) - 2 * np.nanstd(axLowLims)
    newAxMax = np.nanmean(axHighLims) + 2 * np.nanstd(axHighLims)

    for idx, channel in enumerate(spikes['ChannelID']):
        curAx = ax[xIdx[idx], yIdx[idx]]
        curAx.set_ylim(newAxMin, newAxMax)

    for idx, curAx in enumerate(ax.flatten()):
        if idx != 0:
            curAx.tick_params(left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        else:
            curAx.set_ylim(newAxMin, newAxMax)
            curAx.set_xlim(*xLim)
            curAx.set_xlabel('Time (msec)', fontsize = 5, labelpad = 0)
            curAx.set_ylabel('Voltage (uV)', fontsize = 5, labelpad = 0)
        plt.tight_layout()
    return newAxMin, newAxMax

#def plotEventRaster
#@profile
def plotRaster(spikes, trialStats, alignTo, channel, separateBy = None, windowSize = (-0.25, 1), timeRange = None, showNow = False, ax = None, maxTrial = None):

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

    if timeRange is not None:
        timeMask = np.logical_and(trialStats['FirstOnset'] > timeRange[0] * 3e4, trialStats['ChoiceOnset'] < timeRange[1] * 3e4)
        trialStats = trialStats.loc[timeMask, :]

    if maxTrial is not None:
        maxTrial = min(len(trialStats.index), maxTrial)
        trialStats = trialStats.iloc[:maxTrial, :]

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
                except Exception:
                    print('Error plotting raster for trial %s' % idx)
                    traceback.print_exc()
                    #pdb.set_trace()
                    continue
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
def plotFR(spikes, trialStats, alignTo, channel, separateBy = None, windowSize = (-0.25, 1), timeRange = None, showNow = False, ax = None, twin = False, maxTrial = None, discardEmpty = False):
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

    if timeRange is not None:
        timeMask = np.logical_and(trialStats['FirstOnset'] > timeRange[0] * 3e4, trialStats['ChoiceOnset'] < timeRange[1] * 3e4)
        trialStats = trialStats.loc[timeMask, :]

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
                        if discardEmpty:
                            FR[unitIdx].iloc[idx, -1] = True

                    if maxTrial is not None:
                        if idx >= maxTrial -1:
                            break
                except Exception:
                    print('In plotFR: Error getting firing rate for trial %s' % idx)
                    traceback.print_exc()
                    #pdb.set_trace()
                    FR[unitIdx].iloc[idx, -1] = True
                #pdb.set_trace()

    for idx, x in enumerate(FR):
        FR[idx].drop(x.index[x['discard'] == True], axis = 0, inplace = True)
        FR[idx].drop('discard', axis = 1, inplace = True)

    kernelWidth = 25e-3 # seconds
    if separateBy is not None:
        meanFR = {category : [pd.Series(index = timeWindow[:-1]) for i in unitsOnThisChan] for category in uniqueCategories}
        stdFR = {category : [pd.Series(index = timeWindow[:-1]) for i in unitsOnThisChan] for category in uniqueCategories}
        for category in uniqueCategories:
            for idx, unit in enumerate(unitsOnThisChan):
                meanFR[category][idx] = FR[idx].loc[trialStats[separateBy] == category].mean(axis = 0)
                meanFR[category][idx] = gaussian_filter1d(meanFR[category][idx], kernelWidth * 1e3)

                stdFR[category][idx] = FR[idx].loc[trialStats[separateBy] == category].std(axis = 0)
                stdFR[category][idx] = gaussian_filter1d(stdFR[category][idx], kernelWidth * 1e3)
    else:
        meanFR = {'all' : [gaussian_filter1d(x.mean(axis = 0), kernelWidth * 1e3) for x in FR]}
        stdFR = {'all' : [gaussian_filter1d(x.std(axis = 0), kernelWidth * 1e3) for x in FR]}
    #pdb.set_trace()
    colorPalette = sns.color_palette()

    for category, meanFRThisCategory in meanFR.items():
        stdFRThisCategory = stdFR[category]
        if separateBy is not None:
            categoryIndex = pd.Index(uniqueCategories).get_loc(category)
            thisAx = ax[categoryIndex]
        else:
            thisAx = ax

        for unitIdx, x in enumerate(meanFRThisCategory):
            thisError = stdFRThisCategory[unitIdx]
            thisAx.fill_between(timeWindow[:-1], (x - 2 * thisError) * 1e3, (x + 2 * thisError) * 1e3, alpha=0.4, facecolor=colorPalette[unitIdx])
            thisAx.plot(timeWindow[:-1], x * 1e3, linewidth = 1, color = colorPalette[unitIdx])
            thisAx.set_ylabel('Average Firing rate (spk/sec)')
    if showNow:
        plt.show()
    return ax, FR

#TODO: replace motorData wording with analogData wording
def plotSingleTrial(trialStats, trialEvents, motorData, kinematics, spikes,\
    nevIDs, spikesExclude, whichTrial, orderSpikesBy = None, zAxis = None, startEvent = 'FirstOnset', endEvent = 'ChoiceOnset',\
    analogSubset = ['position'], analogLabel = '', kinematicsSubset = ['Hip_Right_Angle X'],\
    kinematicLabel = '', eventSubset = ['Right LED Onset', 'Right Button Onset', 'Left LED Onset', 'Left Button Onset', 'Movement Onset', 'Movement Offset'],\
    binInterval = 20e-3, binWidth = 50e-3):
    nArrays = len(spikes)
    #pdb.set_trace()
    thisTrial = trialStats.loc[whichTrial, :]
    timeStart = thisTrial[startEvent] - 0.1 * 3e4
    timeEnd = thisTrial[endEvent] + 0.1 * 3e4
    fig, motorPlotAxes = mea.plotMotor(motorData, plotRange = (timeStart, timeEnd), subset = analogSubset, subsampleFactor = 30, addAxes = 1 + nArrays, collapse = True)

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    mea.plotTrialEvents(trialEvents, plotRange = (timeStart, timeEnd), ax = motorPlotAxes[0], colorOffset = len(analogSubset), subset = eventSubset)

    motorPlotAxes[0].set_ylabel(analogLabel)
    #try:
    motorPlotAxes[0].legend(loc = 1)
    #except:
    #    pass

    mea.plotMotor(kinematics, plotRange = (timeStart, timeEnd), subset = kinematicsSubset, subsampleFactor = 1, ax = motorPlotAxes[1], collapse = True)
    motorPlotAxes[1].set_ylabel(kinematicLabel)
    try:
        motorPlotAxes[1].legend(loc = 1)
    except:
        pass

    for idx, spikeDict in enumerate(spikes):
        #pdb.set_trace()
        spikeMatOriginal, binCenters, binLeftEdges = hf.binnedSpikes(spikeDict, nevIDs[idx], binInterval, binWidth, timeStart /3e4, (timeEnd - timeStart)/3e4, timeStampUnits = 'seconds')
        spikeMat = spikeMatOriginal.drop(spikesExclude[idx], axis = 'columns')
        if orderSpikesBy == 'idxmax':
            spikeOrder = spikeMat.idxmax().sort_values().index
            spikeMat = spikeMat.loc[:,spikeOrder]
        fig, im = hf.plotBinnedSpikes(spikeMat, binCenters, spikeMat.columns, show = False, normalizationType = 'linear', ax = motorPlotAxes[idx + 2], zAxis = zAxis[idx])
        # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
        cbAx = fig.add_axes([0.83, 0.1 + 0.21 * (len(spikes) - idx - 1), 0.02, 0.18])
        cbar = fig.colorbar(im, cax=cbAx)
        cbar.set_label('Firing Rate (Hz)')
        #pdb.set_trace()

    fig.suptitle('Trial %d: %s' % (whichTrial, thisTrial['Outcome']))
    return fig, motorPlotAxes

#@profile
def spikePDFReport(filePath, spikes, spikeStruct, plotRastersAlignedTo = None, plotRastersSeparatedBy = None, trialStats = None, enableFR = False, newName = None):
    if newName is None:
        pdfName = filePath + '/' + 'spikePDFReport' + '.pdf'
    else:
        pdfName = filePath + '/' + newName + '.pdf'

    with PdfPages(pdfName) as pdf:
        plotSpikePanel(spikeStruct['xcoords'], spikeStruct['ycoords'], spikes)
        pdf.savefig()
        plt.close()

        for idx, channel in enumerate(spikes['ChannelID']):
            sys.stdout.write("Running spikePDFReport: %d%%\r" % int((idx + 1) * 100 / len(spikes['ChannelID'])))
            sys.stdout.flush()
            unitsOnThisChan = np.unique(spikes['Classification'][idx])
            if unitsOnThisChan is not None:
                if len(unitsOnThisChan) > 0:
                    fig, ax = plt.subplots(nrows = 1, ncols = 2)
                    plotSpike(spikes, channel = channel, ax = ax[0], axesLabel = True)
                    isiBins = np.linspace(0, 80, 40)
                    kde_kws = {'clip' : (isiBins[0] * 0.8, isiBins[-1] * 1.2), 'bw' : 'silverman', 'gridsize' : 500}
                    plotISIHistogram(spikes, channel = channel, bins = isiBins, ax = ax[1], kde = False)
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

def trialPDFReport(filePath, trialStats, trialEvents, motorData, kinematics, spikes,\
    nevIDs, spikesExclude, startEvent = 'FirstOnset', endEvent = 'ChoiceOnset',\
    analogSubset = ['position'], analogLabel = '', kinematicsSubset =  ['Hip Right X', 'Knee Right X', 'Ankle Right X'],\
    kinematicLabel = '', eventSubset = ['Right LED Onset', 'Right Button Onset', 'Left LED Onset', 'Left Button Onset', 'Movement Onset', 'Movement Offset'],\
    binInterval = 20e-3, binWidth = 50e-3, maxTrial = None):

    with PdfPages(filePath) as pdf:
        for idx, curTrial in trialStats.iterrows():
            if maxTrial is not None:
                if idx > maxTrial:
                    break
            try:
                fig, motorPlotAxes = plotSingleTrial(trialStats, trialEvents, motorData, \
                    kinematics, spikes, nevIDs, spikesExclude, idx, startEvent = startEvent, \
                    endEvent = endEvent, analogSubset = analogSubset, analogLabel = analogLabel, \
                    kinematicsSubset = kinematicsSubset, kinematicLabel = kinematicLabel, \
                    eventSubset = eventSubset, binInterval = binInterval, binWidth = binWidth)
                pdf.savefig()
                plt.close()
            except:
                pdf.savefig()
                plt.close()

def capitalizeFirstLetter(stringInput):
    return stringInput[0].capitalize() + stringInput[1:]

def loadEventInfo(folderPath, eventInfo):
    try:
        trialStats  = pd.read_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialStats.pickle')
        trialEvents = pd.read_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialEvents.pickle')
        print('Loaded trial data from pickle.')
    except:
        print('Trial data not pickled. Recalculating...')
        motorData = mea.getMotorData(os.path.join(folderPath, eventInfo['ns5FileName']) + '.ns5', eventInfo['inputIDs'], 0 , 'all')
        trialStats, trialEvents = mea.getTrials(motorData)
        trialStats.to_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialStats.pickle')
        trialEvents.to_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialEvents.pickle')
        print('Recalculated trial data and saved to pickle.')
    return trialStats, trialEvents

def loadSpikeInfo(folderPath, arrayName, arrayInfo):
    try:
        spikeStruct = pickle.load(
            open(os.path.join(folderPath, 'coords') + capitalizeFirstLetter(arrayName) + '.pickle', 'rb'))
        spikes      = pickle.load(
            open(os.path.join(folderPath, arrayInfo['ns5FileName']) + '_spikes' + capitalizeFirstLetter(arrayName) + '.pickle', 'rb'))
        print('Loaded spike data from pickle.')
    except:
        print('Spike data not pickled. Recalculating...')

        spikeStruct = loadKSDir(
            os.path.join(folderPath, 'Kilosort/'+ arrayInfo['ns5FileName'] + '_' + capitalizeFirstLetter(arrayName)),
            loadPCs = False)
        spikes = getWaveClusSpikes(
            os.path.join(folderPath, 'wave_clus', arrayInfo['ns5FileName']),
            nevIDs = arrayInfo['nevIDs'], plotting = False, excludeClus = arrayInfo['excludeClus'])

        pickle.dump(spikeStruct,
            open(os.path.join(folderPath, arrayInfo['ns5FileName']) + '_spikeStruct' + capitalizeFirstLetter(arrayName) + '.pickle', 'wb'))
        pickle.dump(spikes,
            open(os.path.join(folderPath, arrayInfo['ns5FileName']) + '_spikes' + capitalizeFirstLetter(arrayName) + '.pickle', 'wb'))
        print('Recalculated spike data and saved to pickle.')
    return spikeStruct, spikes

def generateSpikeReport(folderPath, eventInfo, trialFiles):
    """
    Read in Trial events
    """
    trialStats, trialEvents = loadEventInfo(folderPath, eventInfo)

    for key, value in trialFiles.items():
        """
        Read in array spikes
        """
        spikeStruct, spikes = loadSpikeInfo(folderPath, key, value)

        newName = value['ns5FileName'] + '_' + capitalizeFirstLetter(key) + '_exclude_' + '_'.join([str(i) for i in value['excludeClus']])
        spikePDFReport(folderPath,
            spikes, spikeStruct, plotRastersAlignedTo = 'FirstOnset',
            plotRastersSeparatedBy = 'Direction', trialStats = trialStats,
            enableFR = True,newName = newName)

        del spikes, spikeStruct

def plotSpikeTriggeredRaster(spikesFrom, spikesTo, spikesFromIdx, spikesToIdx,
    windowSize = (-0.25, 1), timeRange = None, showNow = False,
    ax = None, maxSpikes = None):
    # get spike firing times to align to
    ChanIdx = spikesTo['ChannelID'].index(spikesToIdx['chan'])
    unitsOnThisChan = np.unique(spikesTo['Classification'][ChanIdx])

    # get spike firing times to plot
    ChanIdx = spikesFrom['ChannelID'].index(channel)
    unitsOnThisChan = np.unique(spikes['Classification'][ChanIdx])

    if ax is None:
        fig, ax = plt.subplots()
    pdb.set_trace()

if __name__ == "__main__":
    pass
