import os, sys, pdb
from tempfile import mkdtemp
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats
import scipy.io
import pandas as pd
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
import collections

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
                numSDs = 2
                ax.fill_between(timeRange, thisSpike - numSDs*thisError, thisSpike + numSDs*thisError, alpha=0.4, facecolor=colorPalette[unitIdx], label='chan %s, unit %s (%d SDs)' % (channel, unitName, numSDs))
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

def getTrialAxes(trialStats, alignTo, channel, separateBy = None, ax = None):
    if separateBy is not None:
        uniqueCategories = pd.Series(trialStats.loc[:,separateBy].unique())
        uniqueCategories.dropna(inplace = True)
        curLine = {category : 0 for category in uniqueCategories}

        if ax is not None:
            fig = ax[0].figure
        elif ax is None:
            fig, ax = plt.subplots(len(uniqueCategories),1)
        else:
            assert len(ax) == len(uniqueCategories)

        for idx, thisAx in enumerate(ax):
            thisAx.set_xlabel('Time (milliseconds) aligned to ' + alignTo)
            thisAx.set_ylabel('Trial')
            thisAx.set_title(uniqueCategories[idx])

        return fig, ax, uniqueCategories, curLine

    else: # only one plot
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        ax.set_xlabel('Time (milliseconds) aligned to ' + alignTo)
        ax.set_ylabel('Trial')

        return fig, ax, None, None

#@profile
def plotRaster(spikeMats, fig, ax,
    categories = None, uniqueCategories = None, curLine = None,
    showNow = False, plotOpts = {'type' : 'ticks'}):
    colorPalette = sns.color_palette()
    for unitIdx, thisSpikeMat in enumerate(spikeMats):
        for idx, row in thisSpikeMat.iterrows():

            try:
                if isinstance(ax, collections.Iterable):
                    curCategory = categories.loc[idx]
                    whichAx = pd.Index(uniqueCategories).get_loc(curCategory)
                    axToPlotOn = ax[whichAx]
                    lineToPlot = curLine[curCategory]
                    curLine[curCategory] += 1
                else:
                    axToPlotOn = ax
                    lineToPlot = thisSpikeMat.index.get_loc(idx)
                # # TODO: add option to plot a heatmap style rendering
                trialSpikeTimes = row.index[row > 0] * 1e3
                axToPlotOn.vlines(trialSpikeTimes,
                    lineToPlot, lineToPlot + 1, colors = [colorPalette[unitIdx]],
                    linewidths = [0.5])
            except Exception:
                print('Error plotting raster for line %s' % idx)
                traceback.print_exc()
                continue
        #reset line counts for next pass through for the next unit on this chan
        if isinstance(ax, collections.Iterable):
            curLine = {category : 0 for category in uniqueCategories}

    if showNow:
        plt.show()

    return ax

def plotTrialRaster(spikes = None, trialStats = None, channel = None,
    spikeMats = None, categories = None,
    fig = None, ax = None, uniqueCategories = None, curLine = None,
    alignTo = None, separateBy = None,
    windowSize = (-0.25, 1), timeRange = None, maxTrial = None,
    showNow = False, plotOpts = {'type' : 'ticks'}):

    if plotOpts['type'] == 'ticks':
        binInterval = 1e-3
        binWidth = 1e-3
    elif plotOpts['type'] == 'binned':
        binInterval = plotOpts['binInterval']
        binWidth = plotOpts['binWidth']

    if spikeMats is None:
        assert spikes is not None and trialStats is not None and channel is not None
        spikeMats = hf.binnedSpikesAlignedToTrial(spikes, binInterval, binWidth,
        trialStats, alignTo, channel, windowSize = windowSize, timeRange = timeRange,
            maxTrial = maxTrial)

    if trialStats is not None:
        fig, ax, uniqueCategories, curLine = getTrialAxes(trialStats, alignTo,
            channel, separateBy = separateBy, ax = ax)
    else:
        assert fig is not None
        assert ax is not None
        if separateBy is not None:
            assert uniqueCategories is not None
            assert curLine  is not None

    if categories is None and separateBy is not None:
        categories = trialStats[separateBy]

    plotRaster(spikeMats, fig, ax, categories, uniqueCategories, curLine,
        showNow = showNow, plotOpts = plotOpts)

    return spikeMats, categories, fig, ax, uniqueCategories, curLine

def plotSpikeTriggeredRaster(spikesFrom = None, spikesTo = None,
    spikesFromIdx = None, spikesToIdx = None,
    spikeMats = None,
    fig = None, ax = None, titleOverride = None,
    windowSize = (-0.25, 1), timeRange = None, maxSpikesTo = None,
    showNow = False, plotOpts = {'type' : 'ticks'}):

    if plotOpts['type'] == 'ticks':
        binInterval = 1e-3
        binWidth = 1e-3
    elif plotOpts['type'] == 'binned':
        binInterval = plotOpts['binInterval']
        binWidth = plotOpts['binWidth']

    if spikeMats is None:
        assert spikesFrom is not None and spikesTo is not None
        assert spikesFromIdx is not None and spikesToIdx is not None
        #pdb.set_trace()
        spikeMats = hf.binnedSpikesAlignedToSpikes(spikesFrom, spikesTo,
            spikesFromIdx, spikesToIdx,
            binInterval, binWidth, windowSize = windowSize,
            timeRange = timeRange, maxSpikesTo = maxSpikesTo)
    elif maxSpikesTo is not None and len(spikeMats[0].index) > maxSpikesTo:
        for idx, thisSpikeMat in enumerate(spikeMats):
            spikeMats[idx] = thisSpikeMat.sample(n = maxSpikesTo)

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (milliseconds)')
        ax.set_ylabel('Spike')
        if spikesFromIdx is not None and spikesToIdx is not None:
            ax.set_title('Channel %d triggered by channel %d unit %d' % (
                spikesFromIdx['chan'], spikesToIdx['chan'], spikesToIdx['unit']))
    if titleOverride is not None:
        ax.set_title(titleOverride)

    plotRaster(spikeMats, fig, ax, showNow = showNow, plotOpts = plotOpts)

    return spikeMats, fig, ax

#@profile
def plotFR(spikeMats, fig, ax,
    categories = None, uniqueCategories = None,
    showNow = False,
    plotOpts = {'type' : 'ticks', 'kernelWidth' : 25e-3}):

    if plotOpts['type'] == 'ticks':
        kernelWidth = plotOpts['kernelWidth']
        smoothingStep = True
    elif plotOpts['type'] == 'binned':
        smoothingStep = False

    if isinstance(ax, collections.Iterable):
        meanSpikeMat = {category : [None for i in spikeMats] for category in uniqueCategories}
        stdSpikeMat = {category : [None for i in spikeMats] for category in uniqueCategories}

        for category in uniqueCategories:
            for idx, thisSpikeMat in enumerate(spikeMats):
                tempDF = pd.DataFrame(thisSpikeMat.loc[categories == category], copy = True)
                if smoothingStep:
                    tempDF = tempDF.apply(gaussian_filter1d, axis = 1, raw = True,
                        result_type = 'expand', args = (int(kernelWidth * 1e3 / 2),))
                #pdb.set_trace()
                meanSpikeMat[category][idx] = tempDF.mean(axis = 0)
                stdSpikeMat[category][idx]  = tempDF.std(axis = 0)
    else:
        meanSpikeMat = {'all' : [None for i in spikeMats]}
        stdSpikeMat = {'all' : [None for i in spikeMats]}
        for idx, thisSpikeMat in enumerate(spikeMats):
            tempDF = pd.DataFrame(thisSpikeMat, copy = True)
            if smoothingStep:
                tempDF = tempDF.apply(gaussian_filter1d, axis = 1, raw = True,
                    result_type = 'expand', args = (int(kernelWidth * 1e3 / 2),))

            meanSpikeMat['all'][idx] = tempDF.mean(axis = 0)
            stdSpikeMat['all'][idx]  = tempDF.std(axis = 0)

    colorPalette = sns.color_palette()
    yAxBot, yAxTop = 1e6,-1e6
    for category, meanSpikeMatThisCategory in meanSpikeMat.items():
        stdSpikeMatThisCategory = stdSpikeMat[category]
        if isinstance(ax, collections.Iterable):
            categoryIndex = pd.Index(uniqueCategories).get_loc(category)
            curAx = ax[categoryIndex]
        else:
            curAx = ax

        for unitIdx, x in enumerate(meanSpikeMatThisCategory):
            thisError = stdSpikeMatThisCategory[unitIdx]
            curAx.fill_between(x.index * 1e3, (x - thisError), (x + thisError), alpha=0.4, facecolor=colorPalette[unitIdx])
            curAx.plot(x.index * 1e3, x, linewidth = 1, color = colorPalette[unitIdx])

        curYAxBot, curYAxTop = curAx.get_ylim()
        yAxBot = min(yAxBot, curYAxBot)
        yAxTop = max(yAxTop, curYAxTop)

    if isinstance(ax, collections.Iterable):
        for axIdx, curAx in enumerate(ax):
            curAx.set_ylim(yAxBot, yAxTop)
            curAx.set_ylabel('Average Firing rate (spk/sec)')

    if showNow:
        plt.show()
    return ax

def plotTrialFR(spikes = None, trialStats = None, channel = None,
    spikeMats = None, categories = None,
    fig = None, ax = None, uniqueCategories = None, curLine = None, twin = False,
    alignTo = None, separateBy = None,
    windowSize = (-0.25, 1), timeRange = None, maxTrial = None,
    showNow = False, plotOpts = {'type' : 'ticks', 'kernelWidth' : 25e-3}):

    if plotOpts['type'] == 'ticks':
        binInterval = 1e-3
        binWidth = 1e-3
    elif plotOpts['type'] == 'binned':
        binInterval = plotOpts['binInterval']
        binWidth = plotOpts['binWidth']

    if spikeMats is None:
        assert spikes is not None and trialStats is not None and channel is not None
        spikeMats = hf.binnedSpikesAlignedToTrial(spikes, binInterval, binWidth,
        trialStats, alignTo, channel, windowSize = windowSize, timeRange = timeRange,
            maxTrial = maxTrial)

    if trialStats is not None:
        fig, ax, uniqueCategories, curLine = getTrialAxes(trialStats, alignTo,
            channel, separateBy = separateBy, ax = ax)
    else:
        assert fig is not None
        assert ax is not None
        if separateBy is not None:
            assert uniqueCategories is not None

    if twin:
        if isinstance(ax, collections.Iterable):
            for idx, thisAx in enumerate(ax):
                ax[idx] = thisAx.twinx()
                thisAx.set_xlabel('Time (milliseconds) aligned to ' + alignTo)
                thisAx.set_title(uniqueCategories[idx])
        else:
            ax = ax.twinx()
            ax.set_xlabel('Time (milliseconds) aligned to ' + alignTo)

    if categories is None and separateBy is not None:
        categories = trialStats[separateBy]

    plotFR(spikeMats, fig, ax, categories, uniqueCategories,
        showNow = showNow, plotOpts = plotOpts)

    return spikeMats, categories, fig, ax, uniqueCategories, curLine

def plotSpikeTriggeredFR(spikesFrom = None, spikesTo = None,
    spikesFromIdx = None, spikesToIdx = None,
    spikeMats = None,
    fig = None, ax = None, titleOverride = None, twin = False,
    windowSize = (-0.25, 1), timeRange = None,  maxSpikesTo = None,
    showNow = False, plotOpts = {'type' : 'ticks', 'kernelWidth' : 25e-3}):

    if plotOpts['type'] == 'ticks':
        binInterval = 1e-3
        binWidth = 1e-3
    elif plotOpts['type'] == 'binned':
        binInterval = plotOpts['binInterval']
        binWidth = plotOpts['binWidth']

    if spikeMats is None:
        assert spikesFrom is not None and spikesTo is not None
        assert spikesFromIdx is not None and spikesToIdx is not None
        #pdb.set_trace()
        spikeMats = hf.binnedSpikesAlignedToSpikes(spikesFrom, spikesTo,
            spikesFromIdx, spikesToIdx,
            binInterval, binWidth, windowSize = windowSize,
            timeRange = timeRange, maxSpikesTo = maxSpikesTo)
    elif maxSpikesTo is not None and len(spikeMats[0].index) > maxSpikesTo:
        for idx, thisSpikeMat in enumerate(spikeMats):
            spikeMats[idx] = thisSpikeMat.sample(n = maxSpikesTo)

    if ax is not None and twin:
        ax = ax.twinx()
        ax.set_ylabel('Average Firing rate (spk/sec)')
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (milliseconds)')
        ax.set_ylabel('Average Firing rate (spk/sec)')
        if spikesFromIdx is not None and spikesToIdx is not None:
            ax.set_title('Channel %d triggered by channel %d unit %d' % (
                spikesFromIdx['chan'], spikesToIdx['chan'], spikesToIdx['unit']))
    if titleOverride is not None:
        ax.set_title(titleOverride)
    plotFR(spikeMats, fig, ax, showNow = showNow, plotOpts = plotOpts)

    return spikeMats, fig, ax


def plotSingleTrial(trialStats, trialEvents, motorData, kinematics, spikes,\
    nevIDs, spikesExclude, whichTrial, orderSpikesBy = None, zAxis = None, startEvent = 'FirstOnset', endEvent = 'ChoiceOnset',\
    analogSubset = ['position'], analogLabel = '', kinematicsSubset = ['Hip_Right_Angle X'],\
    kinematicLabel = '', eventSubset = ['Right LED Onset', 'Right Button Onset', 'Left LED Onset', 'Left Button Onset', 'Movement Onset', 'Movement Offset'],\
    binInterval = 20e-3, binWidth = 50e-3, arrayNames = None):
    if arrayNames is None:
        arrayNames = ['' for i in spikes]
    nArrays = len(spikes)
    #pdb.set_trace()
    thisTrial = trialStats.loc[whichTrial, :]
    # time units of samples
    timeStart = thisTrial[startEvent] - 0.1 * 3e4
    timeEnd = thisTrial[endEvent] + 0.1 * 3e4
    fig, motorPlotAxes = mea.plotMotor(motorData,
        plotRange = (timeStart, timeEnd), subset = analogSubset,
        subsampleFactor = 30, addAxes = 1 + nArrays, collapse = True)

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02,
        hspace=0.02)
    mea.plotTrialEvents(trialEvents, plotRange = (timeStart, timeEnd),
        ax = motorPlotAxes[0], colorOffset = len(analogSubset),
        subset = eventSubset)

    motorPlotAxes[0].set_ylabel(analogLabel)
    #try:
    motorPlotAxes[0].legend(loc = 1)
    #except:
    #    pass

    mea.plotMotor(kinematics, plotRange = (timeStart, timeEnd),
        subset = kinematicsSubset, subsampleFactor = 1, ax = motorPlotAxes[1],
        collapse = True)
    motorPlotAxes[1].set_ylabel(kinematicLabel)
    try:
        motorPlotAxes[1].legend(loc = 1)
    except:
        pass

    for idx, spikeDict in enumerate(spikes):
        spikeMatOriginal, binCenters, binLeftEdges = hf.binnedSpikes(spikeDict,
            nevIDs[idx], binInterval, binWidth, timeStart ,
            (timeEnd - timeStart) , timeStampUnits = 'seconds')
        spikeMat = spikeMatOriginal.drop(spikesExclude[idx], axis = 'columns')
        if orderSpikesBy == 'idxmax':
            spikeOrder = spikeMat.idxmax().sort_values().index
            spikeMat = spikeMat.loc[:,spikeOrder]
        fig, im = hf.plotBinnedSpikes(spikeMat, show = False,
            normalizationType = 'linear', ax = motorPlotAxes[idx + 2],
            zAxis = zAxis[idx])
        # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
        cbAx = fig.add_axes([0.83, 0.1 + 0.21 * (len(spikes) - idx - 1), 0.02, 0.18])
        cbar = fig.colorbar(im, cax=cbAx)
        cbar.set_label('Spk/s ' + arrayNames[idx])
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

def loadEventInfo(folderPath, eventInfo, forceRecalc = False):
    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            trialStats  = pd.read_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialStats.pickle')
            trialEvents = pd.read_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialEvents.pickle')
            print('Loaded trial data from pickle.')
        except:
            print('Trial data not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        motorData = mea.getMotorData(os.path.join(folderPath, eventInfo['ns5FileName']) + '.ns5', eventInfo['inputIDs'], 0 , 'all')
        trialStats, trialEvents = mea.getTrials(motorData)
        trialStats.to_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialStats.pickle')
        trialEvents.to_pickle(os.path.join(folderPath, eventInfo['ns5FileName']) + '_trialEvents.pickle')
        print('Recalculated trial data and saved to pickle.')
    return trialStats, trialEvents

def loadSpikeInfo(folderPath, arrayName, arrayInfo, forceRecalc = False):
    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            spikes      = pickle.load(
                open(os.path.join(folderPath, arrayInfo['ns5FileName']) + '_spikes' + capitalizeFirstLetter(arrayName) + '_exclude_' + '_'.join([str(i) for i in arrayInfo['excludeClus']]) + '.pickle', 'rb'))
            print('Loaded spike data from pickle.')
        except:
            # if loading failed, recalculate anyway
            print('Spike data not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        spikes = getWaveClusSpikes(
            os.path.join(folderPath, 'wave_clus', arrayInfo['ns5FileName']) + '/',
            nevIDs = arrayInfo['nevIDs'], plotting = False, excludeClus = arrayInfo['excludeClus'])
        pickle.dump(spikes,
            open(os.path.join(folderPath, arrayInfo['ns5FileName']) + '_spikes' + capitalizeFirstLetter(arrayName) + '_exclude_' + '_'.join([str(i) for i in arrayInfo['excludeClus']]) + '.pickle', 'wb'))
        print('Recalculated spike data and saved to pickle.')

    try:
        spikeStruct = pickle.load(open(os.path.join(folderPath, 'coords') + capitalizeFirstLetter(arrayName) + '.pickle', 'rb'))
    except:
        print('Spike metadata not pickled. Recalculating...')
        spikeStruct = loadKSDir(os.path.join(folderPath, 'Kilosort/'+ arrayInfo['ns5FileName'] + '_' + capitalizeFirstLetter(arrayName)), loadPCs = False)
        pickle.dump(spikeStruct, open(os.path.join(folderPath, arrayInfo['ns5FileName']) + '_spikeStruct' + capitalizeFirstLetter(arrayName) + '.pickle', 'wb'))
        print('Recalculated spike metadata and saved to pickle.')

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

if __name__ == "__main__":
    pass
