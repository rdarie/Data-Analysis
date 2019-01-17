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
import itertools
import re
import math as m
import copy
LABELFONTSIZE = 10

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

def getNevMatSpikes(filePath, nevIDs = None, plotting = False, excludeClus = [0]):
    # TODO: not memory mapped yet
    nCh = len(nevIDs)

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
    markForDeletion = {i: False for i in nevIDs}
    lastMaxUnitID = 0
    with h5py.File(filePath, 'r') as f:
        for idx, chanID in enumerate(nevIDs):
            #pdb.set_trace()
            spikes['ChannelID'][idx] = chanID
            chanMask = np.array(f['NEV']['Data']['Spikes']['Electrode']) == chanID

            markForDeletion[chanID] = not chanMask.any()
            if not markForDeletion[chanID]:
                unitsInFile = np.unique(f['NEV']['Data']['Spikes']['Unit'][chanMask]) +  1 + lastMaxUnitID
                if excludeClus:
                    notMUAMask = np.logical_not(np.isin(f['NEV']['Data']['Spikes']['Unit'][chanMask], excludeClus))
                else:
                    notMUAMask = np.full(len(f['NEV']['Data']['Spikes']['Unit'][chanMask]), True, dtype = np.bool)
                #pdb.set_trace()
                spikes['Classification'][idx] = f['NEV']['Data']['Spikes']['Unit'][chanMask] + 1 + lastMaxUnitID
                spikes['Classification'][idx] = spikes['Classification'][idx][notMUAMask]

                spikes['TimeStamps'][idx] =  f['NEV']['Data']['Spikes']['TimeStamp'][chanMask] / 3e4
                spikes['TimeStamps'][idx] = spikes['TimeStamps'][idx][notMUAMask]

                spikes['Waveforms'][idx] = f['NEV']['Data']['Spikes']['Waveform'][chanMask[:,0],:]
                spikes['Waveforms'][idx] = spikes['Waveforms'][idx][notMUAMask, :]

                # note that the units of TimeStamps is in SECONDS
                unitIDs+=unitsInFile.tolist()
                lastMaxUnitID = max(unitIDs)

    #pdb.set_trace()
    spikes['ChannelID'] = list(filter(lambda it: not markForDeletion[it], spikes['ChannelID']))
    spikes['Classification'] = list(filter(lambda it: not it == [], spikes['Classification']))
    spikes['TimeStamps'] = list(filter(lambda it:not it == [], spikes['TimeStamps']))
    spikes['Waveforms'] = list(filter(lambda it: it is not None, spikes['Waveforms']))
    spikes['NEUEVWAV_HeaderIndices'] = list(filter(lambda it: it is not None, spikes['NEUEVWAV_HeaderIndices']))
    return spikes

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
def getSpikeStats(spikes, channel, whichStats = ['mean', 'std'], bounds = None, windowSize = None):

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = pd.unique(spikes['Classification'][ChanIdx])

    statsDict = {}

    if windowSize is None:
        windowSize = [0, (spikes['Waveforms'][ChanIdx].shape[1] - 1) / spikes['basic_headers']['TimeStampResolution']]

    if unitsOnThisChan is not None:

        statsDict = {unitName: {} for  unitName in unitsOnThisChan}

        for unitIdx, unitName in enumerate(unitsOnThisChan):
            unitMask = spikes['Classification'][ChanIdx] == unitName

            #pdb.set_trace()
            if bounds is not None:
                startIdx = int((bounds[0] - windowSize[0]) * spikes['basic_headers']['TimeStampResolution'])
                endIdx = int((bounds[1] - windowSize[0]) * spikes['basic_headers']['TimeStampResolution'])
                boundsSlice = slice(startIdx, endIdx, 1)
            else:
                boundsSlice = slice(None)
            if len(spikes['Waveforms'][ChanIdx].shape) == 3:
                waveForms = spikes['Waveforms'][ChanIdx][unitMask, boundsSlice, ChanIdx]
            else:
                waveForms = spikes['Waveforms'][ChanIdx][unitMask, boundsSlice]

            if 'mean' in whichStats:
                statsDict[unitName].update({'mean':np.nanmean(waveForms, axis = 0)})
            if 'std' in whichStats:
                statsDict[unitName].update({'std':np.nanstd(waveForms, axis = 0)})
            if 'rms' in whichStats:
                waveFormsDF = pd.DataFrame(waveForms).fillna(method = 'ffill', axis = 1, limit = 30).fillna(method = 'bfill', axis = 1, limit =  30)
                statsDict[unitName].update({'rms':np.linalg.norm(waveFormsDF, ord=None, axis=1)})
            if 'abs max' in whichStats:
                #pdb.set_trace()
                statsDict[unitName].update({'abs max':np.abs(waveForms).max().max()})
            if 'max' in whichStats:
                statsDict[unitName].update({'max':waveForms.max().max()})
            if 'min' in whichStats:
                statsDict[unitName].update({'min':waveForms.min().min()})
    return statsDict

def plotSpike(spikes, channel, showNow = False, ax = None,
    acrossArray = False, xcoords = None, ycoords = None,
    axesLabel = False, errorMultiplier = 2, ignoreUnits = [],
    channelPlottingName = None, chanNameInLegend = True):

    if channelPlottingName is None:
        channelPlottingName = str(channel)

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = pd.unique(spikes['Classification'][ChanIdx])
    if 'ClassificationLabel' in spikes.keys():
        unitsLabelsOnThisChan = pd.unique(spikes['ClassificationLabel'][ChanIdx])
    else:
        unitsLabelsOnThisChan = None

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

        colorPalette = sns.color_palette(n_colors = 40)
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            #print('ignoreUnits are {}'.format([-1] + ignoreUnits))
            if unitName in [-1] + ignoreUnits:
                continue

            unitMask = spikes['Classification'][ChanIdx] == unitName

            if 'ClassificationLabel' in spikes.keys():
                unitPlottingName = unitsLabelsOnThisChan[unitIdx]
            else:
                unitPlottingName = unitName

            if chanNameInLegend:
                labelName = 'chan %s, unit %s (%d SDs)' % (channelPlottingName, unitPlottingName, errorMultiplier)
            else:
                labelName = 'unit %s (%d SDs)' % (unitPlottingName, errorMultiplier)

            if acrossArray:
                for idx, channel in enumerate(spikes['ChannelID']):
                    curAx = ax[xIdx[idx], yIdx[idx]]
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, idx]
                    thisSpike = np.nanmean(waveForms, axis = 0)
                    thisError = np.nanstd(waveForms, axis = 0)
                    timeRange = np.arange(len(thisSpike)) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                    curAx.fill_between(timeRange, thisSpike - errorMultiplier*thisError,
                        thisSpike + errorMultiplier*thisError, alpha=0.4,
                        facecolor=colorPalette[unitIdx],
                        label=labelName)
                    curAx.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])

                sns.despine()
                for curAx in ax.flatten():
                    curAx.tick_params(left='off', top='off', right='off',
                    bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')
                plt.tight_layout(pad = 0.01)

            else:
                if len(spikes['Waveforms'][ChanIdx].shape) == 3:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, ChanIdx]
                else:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :]
                thisSpike = np.nanmean(waveForms, axis = 0)
                thisError = np.nanstd(waveForms, axis = 0)
                timeRange = np.arange(len(thisSpike)) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                colorPalette = sns.color_palette(n_colors = 40)

                ax.fill_between(timeRange, thisSpike - errorMultiplier*thisError,
                    thisSpike + errorMultiplier*thisError, alpha=0.4,
                    facecolor=colorPalette[unitIdx],
                    label=labelName)
                ax.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])
                if axesLabel:
                    ax.set_ylabel(spikes['Units'])
                    ax.set_xlabel('Time (msec)')
                    ax.set_title('Units on channel {}'.format(channelPlottingName))
                    ax.legend()
        if showNow:
            plt.show()

    return fig,ax

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
        colorPalette = sns.color_palette(n_colors = 40)
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
def plotSpikePanel(spikeStruct, spikes):
    sns.set_style("dark", {"axes.facecolor": ".9"})
    matplotlib.rc('xtick', labelsize=5)
    matplotlib.rc('ytick', labelsize=5)
    matplotlib.rc('legend', fontsize=5)
    matplotlib.rc('axes', xmargin=.01)
    matplotlib.rc('axes', ymargin=.01)

    """
    xIdx, yIdx = coordsToIndices(spikeStruct['xcoords'], spikeStruct['ycoords'])
    fig, ax = plt.subplots(nrows = max(np.unique(xIdx)) + 1,
        ncols = max(np.unique(yIdx)) + 1)
    """

    #spikeStruct.dropna(inplace = True)
    xIdx = np.array(spikeStruct['xcoords'].values - spikeStruct['xcoords'].min(), dtype = np.int)
    yIdx = np.array(spikeStruct['ycoords'].values - spikeStruct['ycoords'].min(), dtype = np.int)
    #pdb.set_trace()

    fig, ax = plt.subplots(nrows = int(max(np.unique(xIdx)) + 1),
        ncols = int(max(np.unique(yIdx)) + 1))
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
    # get xLim from last axis that has spikes, in order to make the label
    xLim = ax[xIdx[idx], yIdx[idx]].get_xlim()
    sns.despine()

    newAxMin = np.nanmean(axLowLims) - .5 * np.nanstd(axLowLims)
    newAxMax = np.nanmean(axHighLims) + .5 * np.nanstd(axHighLims)

    for idx, channel in enumerate(spikes['ChannelID']):
        curAx = ax[xIdx[idx], yIdx[idx]]
        curAx.set_ylim(newAxMin, newAxMax)

    for idx, curAx in enumerate(ax.flatten()):
        if idx != 0:
            curAx.tick_params(left=False, top=False, right=False, bottom=False,
                labelleft=False, labeltop=False, labelright=False,
                labelbottom=False)
        else:
            curAx.tick_params(left=True, top=False, right=False, bottom=True,
                labelleft=True, labeltop=False, labelright=False,
                labelbottom=True, direction = 'in')

            curAx.set_ylim(newAxMin, newAxMax)
            curAx.set_xlim(*xLim)
            labelFontSize = LABELFONTSIZE
            curAx.set_xlabel('msec', fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)
            curAx.set_ylabel('uV', fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)

    plt.tight_layout(0.005)
    return newAxMin, newAxMax

def getTrialAxes(trialStats, alignTo, channel, separateBy = None, ax = None):
    ## TODO: leverage a seaborn.FacetGrid to make this more streamlined
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

        for idx, curAx in enumerate(ax):
            labelFontSize = LABELFONTSIZE
            curAx.set_xlabel('Time (milliseconds) aligned to ' + alignTo,
                fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)
            curAx.set_ylabel('Trial', fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)
            curAx.set_title(uniqueCategories[idx])

        return fig, ax, uniqueCategories, curLine
    else: # only one plot
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
        labelFontSize = LABELFONTSIZE
        ax.set_xlabel('Time (milliseconds) aligned to ' + alignTo,
            fontsize = labelFontSize,
            labelpad = - 3 * labelFontSize)
        ax.set_ylabel('Trial', fontsize = labelFontSize,
            labelpad = - 3 * labelFontSize)

        return fig, ax, None, None

#@profile
def plotRaster(spikeMats, fig, ax,
    categories = None, uniqueCategories = None, curLine = None,
    showNow = False,
    plotOpts = {'type' : 'ticks'},
    rasterOpts = {
        'binInterval' : 5* 1e-3, 'binWidth' : 10* 1e-3
        }
    ):

    colorPalette = sns.color_palette(n_colors = 40)

    # spikeMat columns are in seconds, convert to milliseconds
    indexConversionFactor = 1e3
    for unitIdx, thisSpikeMat in enumerate(spikeMats):
        for idx, row in thisSpikeMat.iterrows():
            try:
                if isinstance(ax, collections.Iterable):
                    curCategory = categories.loc[idx]
                    whichAx = pd.Index(uniqueCategories).get_loc(curCategory)
                    curAx = ax[whichAx]
                    lineToPlot = curLine[curCategory]
                    curLine[curCategory] += 1
                else:
                    curAx = ax
                    lineToPlot = thisSpikeMat.index.get_loc(idx)
                # # TODO: add option to plot a heatmap style rendering
                trialSpikeTimes = row.index[row > 0] * indexConversionFactor
                curAx.vlines(trialSpikeTimes,
                    lineToPlot, lineToPlot + 1, colors = [colorPalette[unitIdx]],
                    linewidths = [1])
            except Exception:
                print('Error plotting raster for line %s' % idx)
                traceback.print_exc()
                continue
        #reset line counts for next pass through for the next unit on this chan
        if isinstance(ax, collections.Iterable):
            curLine = {category : 0 for category in uniqueCategories}

    if isinstance(ax, collections.Iterable):
        for curAx in ax:
            curAx.grid(b=True)
    else:
        ax.grid(b=True)

    if showNow:
        plt.show()

    return ax

def plotTrialRaster(spikes = None, trialStats = None, channel = None,
    correctAlignmentSpikes = 0,
    spikeMats = None, categories = None,
    fig = None, ax = None, uniqueCategories = None, curLine = None,
    alignTo = None, separateBy = None,
    showNow = False,
    rasterOpts = {
        'kernelWidth' : 50e-3,
        'binInterval' : 2.5* 1e-3, 'binWidth' : 5* 1e-3,
        'windowSize' : (-0.25, 1),
        'alignTo' : 'FirstOnset',
        'separateBy' : 'Direction',
        'discardEmpty': None, 'maxTrial' : None, 'timeRange' : None},
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'}
    ):

    if correctAlignmentSpikes: #correctAlignmentSpikes units in samples
        spikes = hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

    if spikeMats is None:
        assert spikes is not None and trialStats is not None and channel is not None
        spikeMats, categories, selectedIndices = hf.binnedSpikesAlignedToTrial(spikes, rasterOpts['binInterval'], rasterOpts['binWidth'],
        trialStats, rasterOpts['alignTo'], channel, separateBy = rasterOpts['separateBy'], windowSize = rasterOpts['windowSize'], timeRange = rasterOpts['timeRange'],
            maxTrial = rasterOpts['maxTrial'])

    if trialStats is not None:
        fig, ax, uniqueCategories, curLine = getTrialAxes(trialStats, rasterOpts['alignTo'],
            channel, separateBy = rasterOpts['separateBy'], ax = ax)
    else:
        assert fig is not None
        assert ax is not None
        if rasterOpts['separateBy'] is not None:
            assert uniqueCategories is not None
            assert curLine  is not None

    plotRaster(spikeMats, fig, ax, categories, uniqueCategories, curLine,
        showNow = showNow, plotOpts = plotOpts)
    plt.tight_layout(pad = 0.01)
    return spikeMats, categories, fig, ax, uniqueCategories, curLine

def plotSpikeTriggeredRaster(spikesFrom = None, spikesTo = None,
    spikesFromIdx = None, spikesToIdx = None,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0,
    spikeMats = None,
    fig = None, ax = None, titleOverride = None,
    categories = None,
    showNow = False, rasterOpts = {
        'kernelWidth' : 1e-3,
        'binInterval' : (3**-1)* 1e-3, 'binWidth' : (3**-1)* 1e-3,
        'windowSize' : (-0.01, .11),
        'discardEmpty': None, 'maxSpikesTo' : None, 'timeRange' : None,
        'separateByFunArgs': None,'separateByFunKWArgs': {'type' : 'Classification'}},
    plotOpts = {'type' : 'ticks'}
    ):

    if correctAlignmentSpikesFrom: #correctAlignmentSpikesFrom units in samples
        spikesFrom = hf.correctSpikeAlignment(spikesFrom, correctAlignmentSpikesFrom)
    if correctAlignmentSpikesTo: #correctAlignmentSpikesFrom units in samples
        spikesTo = hf.correctSpikeAlignment(spikesTo, correctAlignmentSpikesTo)

    selectedIndices = None
    if rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is not None:
        separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'], **rasterOpts['separateByFunKWArgs'])
    elif rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is None:
        separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'])
    elif rasterOpts['separateByFunArgs'] is None and rasterOpts['separateByFunKWArgs'] is not None:
        separateByFun = hf.catSpikesGenerator(**rasterOpts['separateByFunKWArgs'])
    else:
        separateByFun = None

    if spikeMats is None:
        assert spikesFrom is not None and spikesTo is not None
        assert spikesFromIdx is not None and spikesToIdx is not None

        spikeMats, categories, selectedIndices = hf.binnedSpikesAlignedToSpikes(spikesFrom, spikesTo,
            spikesFromIdx, spikesToIdx,
            rasterOpts['binInterval'], rasterOpts['binWidth'], windowSize = rasterOpts['windowSize'],
            separateByFun = separateByFun,
            timeRange = rasterOpts['timeRange'], maxSpikesTo = rasterOpts['maxSpikesTo'])

    elif rasterOpts['maxSpikesTo'] is not None and len(spikeMats[0].index) > rasterOpts['maxSpikesTo']:
        spikeMats[0] = spikeMats[0].sample(n = rasterOpts['maxSpikesTo'])
        selectedIndices = spikeMats[0].index
        for idx, thisSpikeMat in enumerate(spikeMats):
            if idx > 0:
                spikeMats[idx] = thisSpikeMat.loc[selectedIndices, :]
        if categories is not None:
            categories = categories.loc[selectedIndices]

    if categories is not None:
        uniqueCategories = pd.Series(np.unique(categories))
        uniqueCategories.dropna(inplace = True)
        curLine = {category : 0 for category in uniqueCategories}
        categoriesButOnlyOne = len(uniqueCategories) == 1
    else:
        categoriesButOnlyOne = False

    if (separateByFun is None and categories is None) or categoriesButOnlyOne:# only one subplot

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
            labelFontSize = LABELFONTSIZE
            ax.set_xlabel('Time (milliseconds)',
                fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)
            ax.set_ylabel('Spike', fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)
            if spikesFromIdx is not None and spikesToIdx is not None:
                ax.set_title('Channel %d triggered by channel %d' % (
                    spikesFromIdx['chan'], spikesToIdx['chan']))
            if titleOverride is not None:
                ax.set_title(titleOverride)
        plotRaster(spikeMats, fig, ax, showNow = showNow, plotOpts = plotOpts, rasterOpts = rasterOpts)
    else: # multiple subplots
        if ax is None:
            fig, ax = plt.subplots(len(uniqueCategories),1)
            for idx, curAx in enumerate(ax):
                labelFontSize = LABELFONTSIZE
                curAx.set_ylabel('Spike', fontsize = labelFontSize,
                    labelpad = - 3 * labelFontSize)
                if idx == len(ax) - 1:
                    curAx.set_xlabel('Time (milliseconds)',
                        fontsize = labelFontSize,
                        labelpad = - 3 * labelFontSize)
            if spikesFromIdx is not None and spikesToIdx is not None:
                fig.suptitle('Channel %d triggered by channel %d' % (
                    spikesFromIdx['chan'], spikesToIdx['chan']))
            if titleOverride is not None:
                fig.suptitle(titleOverride)
        else:
            fig = ax[0].figure
        plotRaster(spikeMats, fig, ax, categories, uniqueCategories, curLine,
            showNow = showNow, plotOpts = plotOpts, rasterOpts = rasterOpts)
    plt.tight_layout(pad = 0.01)
    return spikeMats, fig, ax, selectedIndices

#@profile
def plotFR(spikeMats, fig, ax,
    categories = None, uniqueCategories = None,
    showNow = False,
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'},
    rasterOpts = {
        'kernelWidth' : 50e-3,
        'binInterval' : 5* 1e-3, 'binWidth' : 10* 1e-3
        }
    ):

    if plotOpts['type'] == 'ticks':
        smoothingStep = True
    elif plotOpts['type'] == 'binned':
        smoothingStep = False

    if smoothingStep:
        # kernel width in seconds
        # bin width in seconds
        gaussianRadius = int(rasterOpts['kernelWidth'] / (rasterOpts['binWidth'] * 2)) # samples

    if isinstance(ax, collections.Iterable):
        meanSpikeMat = {category : [None for i in spikeMats] for category in uniqueCategories}
        stdSpikeMat = {category : [None for i in spikeMats] for category in uniqueCategories}

        for category in uniqueCategories:
            for idx, thisSpikeMat in enumerate(spikeMats):
                tempDF = pd.DataFrame(thisSpikeMat.loc[categories == category], copy = True)
                if smoothingStep:
                    tempDF = tempDF.apply(gaussian_filter1d, axis = 1, raw = True,
                        result_type = 'expand', args = (gaussianRadius,))
                meanSpikeMat[category][idx] = tempDF.mean(axis = 0)
                stdSpikeMat[category][idx]  = tempDF.std(axis = 0)

                if 'errorBar' in plotOpts.keys():
                    if plotOpts['errorBar'] == 'sem':
                        stdSpikeMat[category][idx] = 1.96 * stdSpikeMat[category][idx] / len(stdSpikeMat[category][idx].index)
    else:
        meanSpikeMat = {'all' : [None for i in spikeMats]}
        stdSpikeMat = {'all' : [None for i in spikeMats]}

        for idx, thisSpikeMat in enumerate(spikeMats):
            tempDF = pd.DataFrame(thisSpikeMat, copy = True)
            if smoothingStep:
                tempDF = tempDF.apply(gaussian_filter1d, axis = 1, raw = True,
                    result_type = 'expand', args = (gaussianRadius,))

            meanSpikeMat['all'][idx] = tempDF.mean(axis = 0)
            stdSpikeMat['all'][idx]  = tempDF.std(axis = 0)

            if 'errorBar' in plotOpts.keys():
                if plotOpts['errorBar'] == 'sem':
                    stdSpikeMat['all'][idx] = 1.96 * stdSpikeMat['all'][idx] / len(stdSpikeMat['all'][idx].index)

    colorPalette = sns.color_palette(n_colors = 40)
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
            ## TODO: add option for this to be the SEM instead of STD
            curAx.fill_between(x.index * 1e3, (x - thisError), (x + thisError),
                alpha=0.4, facecolor=colorPalette[unitIdx])
            curAx.plot(x.index * 1e3, x, linewidth = 1,
                color = colorPalette[unitIdx])

        curYAxBot, curYAxTop = curAx.get_ylim()
        yAxBot = min(yAxBot, curYAxBot)
        yAxTop = max(yAxTop, curYAxTop)

    #negative firing rates don't make sense
    yAxBot = max(yAxBot, 0)
    if isinstance(ax, collections.Iterable):
        for axIdx, curAx in enumerate(ax):
            curAx.set_ylim(yAxBot, yAxTop)
            curAx.set_ylabel('(spk/sec)')
            curAx.grid(b=True)
    else:
        ax.set_ylabel('(spk/sec)')
        ax.set_ylim(yAxBot, yAxTop)
        ax.grid(b=True)

    if showNow:
        plt.show()

    return ax

def plotTrialFR(spikes = None, trialStats = None, channel = None,
    correctAlignmentSpikes = 0,
    spikeMats = None, categories = None,
    fig = None, ax = None, uniqueCategories = None, curLine = None, twin = False,
    showNow = False,
    rasterOpts = {
        'kernelWidth' : 20e-3,
        'binInterval' : 5* 1e-3, 'binWidth' : 10* 1e-3,
        'windowSize' : (-0.25, 1),
        'alignTo' : 'FirstOnset',
        'separateBy' : 'Direction',
        'discardEmpty': None, 'maxTrial' : None, 'timeRange' : None},
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'}
    ):

    if correctAlignmentSpikes: #correctAlignmentSpikes units in samples
        spikes = hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

    if spikeMats is None:
        assert spikes is not None and trialStats is not None and channel is not None
        spikeMats, categories, selectedIndices = hf.binnedSpikesAlignedToTrial(spikes, rasterOpts['binInterval'], rasterOpts['binWidth'],
        trialStats, rasterOpts['alignTo'], channel, separateBy = rasterOpts['separateBy'], windowSize = rasterOpts['windowSize'], timeRange = rasterOpts['timeRange'],
            maxTrial = rasterOpts['maxTrial'])

    if trialStats is not None:
        fig, ax, uniqueCategories, curLine = getTrialAxes(trialStats, rasterOpts['alignTo'],
            channel, separateBy = rasterOpts['separateBy'], ax = ax)
    else:
        assert fig is not None
        assert ax is not None
        if rasterOpts['separateBy'] is not None:
            assert uniqueCategories is not None

    if twin:
        if isinstance(ax, collections.Iterable):
            for idx, curAx in enumerate(ax):
                ax[idx] = curAx.twinx()
                curAx.set_xlabel('Time (milliseconds) aligned to ' + rasterOpts['alignTo'])
                curAx.set_title(uniqueCategories[idx])
        else:
            ax = ax.twinx()
            ax.set_xlabel('Time (milliseconds) aligned to ' + rasterOpts['alignTo'])

    if categories is None and rasterOpts['separateBy'] is not None:
        categories = trialStats[separateBy]

    plotFR(spikeMats, fig, ax, categories, uniqueCategories,
        showNow = showNow, plotOpts = plotOpts, rasterOpts = rasterOpts)
    plt.tight_layout(pad = 0.1)
    return spikeMats, categories, fig, ax, uniqueCategories, curLine

def plotSpikeTriggeredFR(spikesFrom = None, spikesTo = None,
    spikesFromIdx = None, spikesToIdx = None,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0,
    spikeMats = None,
    fig = None, ax = None, titleOverride = None, twin = False,
    categories = None,
    showNow = False,  rasterOpts = {
        'kernelWidth' : 1e-3,
        'binInterval' : (3**-1)* 1e-3, 'binWidth' : (3**-1)* 1e-3,
        'windowSize' : (-0.01, .11),
        'discardEmpty': None, 'maxSpikesTo' : None, 'timeRange' : None,
        'separateByFunArgs': None,'separateByFunKWArgs': {'type' : 'Classification'}},
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'}
    ):

    if correctAlignmentSpikesFrom: #correctAlignmentSpikesFrom units in samples
        spikesFrom = hf.correctSpikeAlignment(spikesFrom, correctAlignmentSpikesFrom)
    if correctAlignmentSpikesTo: #correctAlignmentSpikesFrom units in samples
        spikesTo = hf.correctSpikeAlignment(spikesTo, correctAlignmentSpikesTo)

    selectedIndices = None
    if rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is not None:
        separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'], **rasterOpts['separateByFunKWArgs'])
    elif rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is None:
        separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'])
    elif rasterOpts['separateByFunArgs'] is None and rasterOpts['separateByFunKWArgs'] is not None:
        separateByFun = hf.catSpikesGenerator(**rasterOpts['separateByFunKWArgs'])
    else:
        separateByFun = None

    if spikeMats is None:
        assert spikesFrom is not None and spikesTo is not None
        assert spikesFromIdx is not None and spikesToIdx is not None

        spikeMats, categories = hf.binnedSpikesAlignedToSpikes(spikesFrom, spikesTo,
            spikesFromIdx, spikesToIdx,
            rasterOpts['binInterval'], rasterOpts['binWidth'], windowSize = rasterOpts['windowSize'],
            separateByFun = separateByFun,
            timeRange = rasterOpts['timeRange'], maxSpikesTo = rasterOpts['maxSpikesTo'])

    elif rasterOpts['maxSpikesTo'] is not None and len(spikeMats[0].index) > rasterOpts['maxSpikesTo']:
        spikeMats[0] = spikeMats[0].sample(n = rasterOpts['maxSpikesTo'])
        selectedIndices = spikeMats[0].index
        for idx, thisSpikeMat in enumerate(spikeMats):
            if idx > 0:
                spikeMats[idx] = thisSpikeMat.loc[selectedIndices, :]
        if categories is not None:
            #pdb.set_trace()
            categories = categories.loc[selectedIndices]

    if categories is not None:
        uniqueCategories = pd.Series(np.unique(categories))
        uniqueCategories.dropna(inplace = True)
        curLine = {category : 0 for category in uniqueCategories}
        categoriesButOnlyOne = len(uniqueCategories) == 1
    else:
        categoriesButOnlyOne = False

    if (separateByFun is None and categories is None) or categoriesButOnlyOne:# only one subplot
        if ax is not None and twin:
            ax = ax.twinx()
            ax.set_ylabel('(spk/sec)')
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots()
            labelFontSize = LABELFONTSIZE
            ax.set_xlabel('(milliseconds)', fontsize = labelFontSize,
                labelpad = - 3 * labelFontSize)
            if spikesFromIdx is not None and spikesToIdx is not None:
                ax.set_title('Channel %d triggered by channel %d' % (
                    spikesFromIdx['chan'], spikesToIdx['chan']), fontsize = labelFontSize)
        if titleOverride is not None:
            ax.set_title(titleOverride)
        plotFR(spikeMats, fig, ax, showNow = showNow, plotOpts = plotOpts, rasterOpts = rasterOpts)
    else: # subplots
        if ax is None:
            fig, ax = plt.subplots(len(uniqueCategories),1)
        else: # ax pre-exists and has multiple subplots
            fig = ax[0].figure
            for idx, curAx in enumerate(ax):
                ax[idx] = curAx.twinx()
                labelFontSize = LABELFONTSIZE
                curAx.set_title('Unit %d' % uniqueCategories[idx], fontsize = labelFontSize)
                if idx == len(ax) - 1:
                    curAx.set_xlabel('(milliseconds)', fontsize = labelFontSize,
                        labelpad = - 3 * labelFontSize)
                    curAx.set_ylabel('(spk/sec)', fontsize = labelFontSize,
                        labelpad = - 3 * labelFontSize)
            if spikesFromIdx is not None and spikesToIdx is not None:
                fig.suptitle('Channel %d triggered by channel %d' % (
                    spikesFromIdx['chan'], spikesToIdx['chan']))
            if titleOverride is not None:
                fig.suptitle(titleOverride)
        plotFR(spikeMats, fig, ax,
            categories, uniqueCategories,
            showNow = showNow, plotOpts = plotOpts, rasterOpts = rasterOpts)
    plt.tight_layout(pad = 0.01)
    return spikeMats, fig, ax

def modOnset(spikeMat):
    preWindow = 0
    dt  = spikeMat.columns[1] - spikeMat.columns[0]

    #pdb.set_trace()
    preStimMask = spikeMat.columns < preWindow
    postStimMask = spikeMat.columns > preWindow
    thresh = spikeMat.loc[:,preStimMask].mean(axis = 1) + 2.576 * spikeMat.loc[:,preStimMask].std(axis = 1)
    sustainCriterionMSec = 25e-3
    lastValid = (spikeMat.columns[-1] - sustainCriterionMSec) * 1e3
    sustainCriterion = int(sustainCriterionMSec / dt)

    tempDF = pd.DataFrame(spikeMat.loc[:,postStimMask])
    pValHolder = pd.DataFrame(np.nan, index = spikeMat.loc[:,postStimMask].index,columns = spikeMat.loc[:,postStimMask].columns)
    magDiffHolder = pd.DataFrame(np.nan, index = spikeMat.loc[:,postStimMask].index,columns = spikeMat.loc[:,postStimMask].columns)
    tempDF.columns.name = 'bin'
    tempDF.index.name = 'unit'
    allDF = {}
    for i in range(sustainCriterion+1):
        allDF.update({i:tempDF.T.shift(-i, axis = 0).fillna(method = 'ffill',axis =0 )})
    matsWithHist = pd.concat(allDF, names = ['lag'], axis = 1).stack(level = 'unit')
    for (whichBin, whichUnit), row in matsWithHist.iterrows():
        magDiffHolder.loc[whichUnit, whichBin], pValHolder.loc[whichUnit, whichBin] = scipy.stats.ttest_ind(row, spikeMat.loc[whichUnit,preStimMask], nan_policy = 'omit')

    nCheck = postStimMask.sum() - sustainCriterion
    bla1 = (pValHolder * nCheck) < 1e-6
    bla2 = magDiffHolder.abs() > 50
    #pdb.set_trace()
    #thisModOnset = ((pValHolder * nCheck) < 0.01).idxmax(axis = 1) * 1e3
    thisModOnset = (bla1 & bla2).idxmax(axis = 1) * 1e3
    #pdb.set_trace()
    thisModOnset[thisModOnset < 3] = np.nan
    thisModOnset[thisModOnset > lastValid] = np.nan
    return thisModOnset, pValHolder, magDiffHolder
"""
tempDF = pd.DataFrame(False, index = spikeMat.loc[:,postStimMask].index, columns = spikeMat.loc[:,postStimMask].columns)
for rowIdx, row in spikeMat.loc[:,postStimMask].iterrows():
    tempDF.loc[rowIdx, :] = row > thresh[rowIdx]

# add future samples to each sample to check if it continues to be modulated
tempDF = tempDF.astype(float)
for i in range(1,sustainCriterion+1):
    tempDF = tempDF + tempDF.shift(-i, axis = 1).fillna(method = 'ffill',axis = 1)

tempDF = (tempDF > sustainCriterion).astype(float)
thisModOnset = tempDF.idxmax(axis = 1)

thisModOnset = (thisModOnset - preWindow) * 1e3
thisModOnset[thisModOnset < 1] = np.nan
"""

def saveModOnset(arrayName, spikeMatDict):
    for key, value in spikeMatDict.items():
        thisModOnset, pValHolder, magDiffHolder = modOnset(value)
        print('modOnset = {}'.format(thisModOnset))
        print('Mean modOnset = {}'.format(thisModOnset))
        thisModOnset.to_hdf('./modOnset.h5', 'modOnset_{}_{}'.format(arrayName, key))
        pValHolder.to_hdf('./modOnset.h5', 'pValHolder_{}_{}'.format(arrayName, key))
        magDiffHolder.to_hdf('./modOnset.h5', 'magDiffHolder_{}_{}'.format(arrayName, key))

def sortBinnedArray(spikeMat, orderSpikesBy):

    if orderSpikesBy == 'idxmax':
        spikeOrder = spikeMat.idxmax(axis = 1).sort_values().index

    elif orderSpikesBy == 'meanFR':
        spikeOrder = spikeMat.mean(axis = 1).sort_values().index

    elif orderSpikesBy == 'modOnset':
        #pdb.set_trace()
        thisModOnset, pValHolder, magDiffHolder = modOnset(spikeMat)
        spikeOrder = thisModOnset.sort_values(ascending=False).index

    spikeMat = spikeMat.loc[spikeOrder,:]

    return spikeMat, spikeOrder

def plotSingleTrial(
    whichTrial,
    spikesExclude,
    folderPath = None,
    trialStats = None, trialEvents = None, motorData = None,
    eventInfo = None,
    spikesList = None, arrayNames = None,
    trialFiles = None,
    kinematics = None,
    kinematicsFileName = None,
    kinPosOpts = {
        'ns5FileName' : '',
        'selectHeaders' : None,
        'reIndex' : None,
        'flip' : None,
        'lowCutoff': None
        },
    orderSpikesBy = None, zAxis = None,\
    rasterOpts = {
    'kernelWidth' : 50e-3,
    'binInterval' : 50 * 1e-3, 'binWidth' : 100 * 1e-3,
    'windowSize' : (-1, .1),
    'alignTo' : 'FirstOnset',
    'separateBy' : 'Direction',
    'endOn' : 'ChoiceOnset',
    'discardEmpty': None, 'maxTrial' : None, 'timeRange' : None},
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'},
    analogSubset = ['position'], analogLabel = '',\
    kinematicsSubset = ['Hip Right X'],\
    kinematicLabel = '',\
    eventSubset = ['Right LED Onset', 'Right Button Onset', 'Left LED Onset',\
    'Left Button Onset', 'Movement Onset', 'Movement Offset'],\
    ):

    if kinematics is None:
        kinematics = hf.loadAngles(folderPath, kinematicsFileName)

    if any((trialStats is None, trialEvents is None, motorData is None)):
        trialStats, trialEvents, motorData = loadEventInfo(folderPath, eventInfo)

    if spikesList is None:
        spikesList = []
        arrayNames = []
        for key, value in trialFiles.items():
            spikeStruct, newSpikes = loadSpikeInfo(folderPath, key, value)
            spikesList.append(newSpikes)
            arrayNames.append(key)
    elif arrayNames is None or trialFiles is None:
        arrayNames = ['spike' for i in spikesList]

    nArrays = len(spikesList)
    spikeMatList = [None for i in spikesList]
    idx = 0
    for key, value in trialFiles.items():
        spikeMatList[idx] = loadTrialBinnedArray(folderPath,
            key, value,
            eventInfo, rasterOpts,
            whichTrial = [whichTrial],
            spikes = spikesList[idx],
            trialStats = trialStats, chans = None,
            correctAlignmentSpikes = 0,
            forceRecalc = False)[0]
        idx += 1
    #plt.show()
    if zAxis is None:
        zAxis = [None for i in spikesList]
    if spikesExclude is None:
        spikesExclude = [[] for i in spikesList]

    # time units of samples
    thisTrial = trialStats.loc[whichTrial, :]
    idxStart = trialStats.loc[whichTrial, rasterOpts['alignTo']] + rasterOpts['windowSize'][0] * 3e4
    idxEnd = trialStats.loc[whichTrial, rasterOpts['endOn']] + rasterOpts['windowSize'][-1] * 3e4
    # time units of seconds
    timeStart = idxStart / 3e4
    timeEnd = idxEnd / 3e4

    #create a list of axes and add one for the kinematics and one for each array
    fig, motorPlotAxes = mea.plotMotor(motorData,
        plotRange = (idxStart, idxEnd), idxUnits = 'samples', subset = analogSubset,
        subsampleFactor = 30, addAxes = 1 + nArrays, collapse = True)
    # every 30th sample give 1 msec resolution
    mea.plotTrialEvents(trialEvents, plotRange = (idxStart, idxEnd),
        ax = motorPlotAxes[0], colorOffset = len(analogSubset),
        subset = eventSubset)

    motorPlotAxes[0].set_ylabel(analogLabel)
    try:
        motorPlotAxes[0].legend(loc = 1)
    except Exception:
        traceback.print_exc()
        pass

    mea.plotMotor(kinematics, plotRange = (timeStart, timeEnd), idxUnits = 'seconds',
        subset = kinematicsSubset, subsampleFactor = 1, ax = motorPlotAxes[1],
        collapse = True)
    # the kinematics, on the other hand, have data every 10 msec
    motorPlotAxes[1].set_ylabel(kinematicLabel)
    try:
        motorPlotAxes[1].legend(loc = 1)
    except Exception:
        traceback.print_exc()
        pass

    for idx, spikeDict in enumerate(spikes):
        if orderSpikesBy is not None:
            spikeMatList[idx], spikeOrder = sortBinnedArray(spikeMatList[idx], orderSpikesBy)

        hf.plotBinnedSpikes(spikeMatList[idx], show = False,
            ax = motorPlotAxes[idx + 2],
            zAxis = zAxis[idx], labelTxt = 'Spk/s ' + arrayNames[idx])
        motorPlotAxes[idx + 2].tick_params(labelleft=True)

    motorPlotAxes[-1].tick_params(bottom=True, labelbottom = True)
    fig.suptitle('Trial %d: %s' % (whichTrial, thisTrial['Outcome']))

    #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02,
    #    hspace=0.02)

    return fig, motorPlotAxes

def resetTrialStatsStimID(folderPath, eventInfo, nBins = 3):
    trialStats, trialEvents, motorData = loadEventInfo(folderPath, eventInfo, requested = ['trialStats'])

    stimIDs, stimIDsAbs, firstStimID, secondStimID = mea.getStimID(trialStats, nBins = nBins)
    trialStats["Stimulus ID Pair"]= stimIDs
    trialStats["Stimulus ID Pair (Abs)"]= stimIDsAbs

    trialStats["firstStimID"]= firstStimID.astype(float)
    trialStats["firstStimID (Abs)"]= firstStimID.astype(float).abs()
    trialStats["secondStimID"]= secondStimID.astype(float)
    trialStats["secondStimID (Abs)"]= secondStimID.astype(float).abs()
    setName = eventInfo['ns5FileName'] + '_eventInfo'
    setPath = os.path.join(folderPath, setName + '.h5')
    trialStats.to_hdf(setPath, 'trialStats')

def addLags(X, nBins = 1, binStride = 1, binOffset = 0, setPath = None, recordName = None):
    # todo make this accept arbitrarily shuffled data
    # print('Loading {} from {}'.format(recordName, setPath))
    spikeMats = pd.read_hdf(setPath, 'spikeMats')
    stackedByBin = spikeMats.unstack(level = ['alignLabel', 'trial'])
    origX = pd.read_hdf(setPath, recordName + '/X')
    shuffledIdx = origX.index
    allDFs = {}
    for i in range(binOffset, binOffset+nBins+1, binStride):
        thisStackedBin = stackedByBin.shift(i).fillna(method = 'ffill', axis = 0).fillna(method = 'bfill', axis = 0)
        thisUnstackedBin = thisStackedBin.stack(level = ['alignLabel', 'trial']).reorder_levels(['alignLabel', 'trial', 'bin']).loc[shuffledIdx, :]
        allDFs.update({i:thisUnstackedBin})
    outputDF = pd.concat(allDFs, axis = 1, names = ['lag'])
    #print('addLags outputing DF with shape = {}'.format(outputDF.shape))
    return outputDF.values

def addLagsCentered(X, nBins = 1, binStride = 1, binOffset = 0, setPath = None, recordName = None):
    print("!!!!!!!!!!!!!!!!!!!!! Setpath is {}".format(setPath))
    print("LOoking for {}".format(recordName))
    nBins = int(nBins / 2 + 1)
    # todo make this accept arbitrarily shuffled data
    # print('Loading {} from {}'.format(recordName, setPath))

    #pdb.set_trace()
    spikeMats = pd.read_hdf(setPath, 'spikeMats')
    stackedByBin = spikeMats.unstack(level = ['alignLabel', 'trial'])
    origX = pd.read_hdf(setPath, recordName + '/X')
    shuffledIdx = origX.index
    allDFs = {}
    for i in range(binOffset-nBins, binOffset+nBins+1, binStride):
        #pdb.set_trace()
        #thisStackedBin = stackedByBin.shift(i).dropna(how = 'all')
        thisStackedBin = stackedByBin.shift(i).fillna(method = 'ffill', axis = 0).fillna(method = 'bfill', axis = 0)
        thisUnstackedBin = thisStackedBin.stack(level = ['alignLabel', 'trial']).reorder_levels(['alignLabel', 'trial', 'bin']).loc[shuffledIdx, :]
        allDFs.update({i:thisUnstackedBin})
    #pdb.set_trace()
    outputDF = pd.concat(allDFs, axis = 1, names = ['lag'])
    #print('outputDF.isna().sum() = {}'.format(outputDF.isna().sum().sum()))
    #print('X.shape = {}'.format(X.shape))
    #print('outputDF.shape = {}'.format(outputDF.shape))
    #print('addLags outputing DF with shape = {}'.format(outputDF.shape))
    return outputDF.values

def addHistory(X, nHistBins = 0, binStride = 1, setPath = None, recordName = None):
    # todo make this accept arbitrarily shuffled data
    # print('Loading {} from {}'.format(recordName, setPath))
    spikeMats = pd.read_hdf(setPath, 'spikeMats')
    stackedByBin = spikeMats.unstack(level = ['alignLabel', 'trial'])
    origX = pd.read_hdf(setPath, recordName + '/X')
    shuffledIdx = origX.index
    allDFs = {}
    for i in range(0, nHistBins+1, binStride):
        thisStackedBin = stackedByBin.shift(binStride * i).fillna(method = 'ffill', axis = 0).fillna(method = 'bfill', axis = 0)
        thisUnstackedBin = thisStackedBin.stack(level = ['alignLabel', 'trial']).reorder_levels(['alignLabel', 'trial', 'bin']).loc[shuffledIdx, :]
        allDFs.update({i:thisUnstackedBin})
    outputDF = pd.concat(allDFs, axis = 1, names = 'lag')
    outputDF = outputDF.fillna(method = 'pad', axis = 1)
    #print('addHistory outputing DF with shape = {}'.format(outputDF.shape))
    return outputDF.values

"""
def addHistory(X, nHistBins = 0, binStride = 1, setPath = None, recordName = None):
    # todo make this accept arbitrarily shuffled data
    spikeMats = pd.read_hdf(setPath, 'spikeMats')
    origX = pd.read_hdf(setPath, recordName + '/X')
    pdb.set_trace()
    shuffledIdx = []
    origIndex = spikeMats.index
    for i in range(X.shape[0]):
        for rowIdx, row in origX.iterrows():
            xRow = X[i,:]
            if (xRow == row.values).all():
                shuffledIdx.append(rowIdx)
    print('Bing!')
    #tempDF = pd.DataFrame(X, index = shuffledIndex, columns = spikeMats.columns)
    allDFs = {}
    for i in range(0, nHistBins, binStride):
        allDFs.update({i:spikeMats.shift(binStride * i).loc[shuffledIndex, :]})
    outputDF = pd.concat(allDFs, axis = 1, names = 'lag')
    outputDF = outputDF.fillna(method = 'pad', axis = 1)
    print('addHistory outputing DF with shape = {}'.format(outputDF.shape))
    return outputDF.values
"""

def featureUnstackerGenerator(spikeMats, whichLevel = None, nCol = None):
    if nCol is None:
        nCol = len(spikeMats.columns)
    # make a custom function that only works on spikeMats to unstack it
    def unstacker(X):
        tempDF = pd.DataFrame(X, index = spikeMats.index, columns = range(nCol))
        tempDF = tempDF.unstack(level = whichLevel)
        #fig, ax = plt.subplots(); plt.spy(tempDF.isnull()); plt.axis('equal'); plt.show()
        return tempDF #maybe add .values here
    return unstacker

def getSpikeMatsForIdx(spikeMats, whichTrials, startBin = 0):

    exampleSpikeMat = next(iter(spikeMats.values()))
    spikeMatIdx = exampleSpikeMat.index
    spikeMatCols = exampleSpikeMat.columns - exampleSpikeMat.columns[0] + startBin
    theseSpikeMats =  np.full((exampleSpikeMat.shape[0],
        exampleSpikeMat.shape[1], len(whichTrials)), np.nan)
    for idx, trialIdx in enumerate(whichTrials):
        try:
            theseSpikeMats[:,:,idx] = spikeMats[trialIdx]
        except Exception:
            traceback.print_exc()
            pdb.set_trace()
    return theseSpikeMats, spikeMatIdx, spikeMatCols

def getSpikeMatsForCategories(categories, uniqueCategories, validTrials, spikeMats, startBin = 0):

    spikeMatsCategorized = {category:None for category in uniqueCategories.values}

    for category in uniqueCategories:
        if category == 'all':
            catMask = np.full((categories.shape[0],), True)
        else:
            catMask = categories == category

        catMask = np.logical_and(catMask, validTrials)
        whichTrials = np.flatnonzero(catMask).tolist()
        #prune nans
        whichTrials = [i for i in whichTrials if spikeMats[i] is not None]
        spikeMatsCategorized[category], spikeMatIdx, spikeMatCols = getSpikeMatsForIdx(spikeMats, whichTrials, startBin = startBin)

    return spikeMatsCategorized, spikeMatIdx, spikeMatCols

def getAverageSpikeMatsForIdx(spikeMats, whichTrials):
    theseSpikeMats = getSpikeMatsForIdx(spikeMats, whichTrials)

    exampleSpikeMat = next(iter(spikeMats.values()))
    spikeMatIdx = exampleSpikeMat.index
    spikeMatCols = exampleSpikeMat.columns

    averageSpikeMats = pd.DataFrame(np.nanmean(theseSpikeMats, axis = 2),
        columns = spikeMatCols, index = spikeMatIdx)
    stdSpikeMats = pd.DataFrame(np.nanstd(theseSpikeMats, axis = 2),
        columns = spikeMatCols, index = spikeMatIdx)

    return averageSpikeMats, stdSpikeMats

def getAverageSpikeMatsForCategories(categories, uniqueCategories, validTrials, spikeMats, startBin = 0):

    spikeMatsCategorized, spikeMatIdx, spikeMatCols = getSpikeMatsForCategories(categories,
        uniqueCategories, validTrials, spikeMats, startBin = startBin)

    averageSpikeMats = {category:None for category in uniqueCategories.values}
    stdSpikeMats = {category:None for category in uniqueCategories.values}

    for category, theseSpikeMats in spikeMatsCategorized.items():
        #pdb.set_trace()
        averageSpikeMats[category] = pd.DataFrame(np.nanmean(theseSpikeMats, axis = 2),
            columns = spikeMatCols, index = spikeMatIdx)
        stdSpikeMats[category] = pd.DataFrame(np.nanstd(theseSpikeMats, axis = 2),
            columns = spikeMatCols, index = spikeMatIdx)

    return averageSpikeMats, stdSpikeMats

def getAverageSpikeMatsFromList(categories, uniqueCategories, validTrials, spikeMatList, arrayNames, startBin = 0):
    averageSpikeMats = {arrayName: None for arrayName in arrayNames}
    stdSpikeMats = {arrayName: None for arrayName in arrayNames}
    for arrayName in arrayNames:
        averageSpikeMats[arrayName], stdSpikeMats[arrayName] =\
            getAverageSpikeMatsForCategories(categories, uniqueCategories, validTrials, spikeMatList[arrayName], startBin = startBin)
        saveModOnset(arrayName, averageSpikeMats[arrayName])
    return averageSpikeMats, stdSpikeMats

def getPairedSpikeMatTTest(spikeMatCubeCatA, spikeMatCubeCatB):
    #pdb.set_trace()
    tTestStatistic, tTestPVal = scipy.stats.ttest_ind(spikeMatCubeCatA, spikeMatCubeCatB, axis=2, nan_policy = 'omit')
    return tTestPVal * spikeMatCubeCatA.shape[0] * spikeMatCubeCatA.shape[1]

def getAllPairedSpikeMatTTestForCategories(categories, uniqueCategories, validTrials, spikeMats):

    spikeMatsCategorized, spikeMatIdx, spikeMatCols = getSpikeMatsForCategories(categories,
        uniqueCategories, validTrials, spikeMats)

    tTestSpikeMats = {}

    for categoryA, categoryB in itertools.combinations(uniqueCategories,2):
        spikeMatCubeCatA = spikeMatsCategorized[categoryA]
        spikeMatCubeCatB = spikeMatsCategorized[categoryB]

        resultingMat = getPairedSpikeMatTTest(spikeMatCubeCatA,spikeMatCubeCatB)
        resultingMat = pd.DataFrame(resultingMat, index = spikeMatIdx, columns = spikeMatCols)
        tTestSpikeMats.update({(categoryA, categoryB) : resultingMat})

    return tTestSpikeMats

def getTrialCategoriesAsFeatures(folderPath,
    trialStats = None, eventInfo = None,
    targetCategory = None,
    alignToList = None, overrideLabel = None):

    targetsDict = {}

    if trialStats is None:
        trialStats, _, _ = loadEventInfo(folderPath, eventInfo, requested = ['trialStats'])

    if type(targetCategory) == list:
        renameType = True if 'Type' in targetCategory else False
    else:
        renameType = True if targetCategory == 'Type' else False
    if renameType:
        trialStats.loc[trialStats['First'].abs() > trialStats['Second'].abs(), 'Type'] = 'A>B'
        trialStats.loc[trialStats['First'].abs() < trialStats['Second'].abs(), 'Type'] = 'A<B'

    categories, uniqueCategories, catIndices, validTrials =\
         getTrialCategories(trialStats, targetCategory)

    for alignTo in alignToList:
        #theseValidTrials = np.logical_and(validTrials,trialStats[alignTo].notnull())
        if type(targetCategory) == list:
            theseTargets = pd.DataFrame(trialStats.loc[validTrials, tuple(targetCategory)], index=trialStats.index[validTrials])
        else:
            theseTargets = pd.Series(trialStats.loc[validTrials, targetCategory], index=trialStats.index[validTrials])
        theseTargets.index.name = 'trial'
        targetsDict.update({alignTo:theseTargets})

    targets = pd.concat(targetsDict, names = ['alignLabel'])
    #pdb.set_trace()
    if type(targetCategory) == list:
        targets = pd.Series(targets.apply(tuple, axis=1),index = targets.index)
    if overrideLabel is not None:
        targets.loc[:] = overrideLabel

    targets = targets.astype('category')

    return targets

def getTrialTriggeredTimeSeriesAsFeatures(folderPath, binCenters,
    trialStats = None, eventInfo = None,
    tsInfo = None,
    trialFiles = None,
    targetCategory = None,
    alignToList = None, targetVariables = ['Knee Right X']):

    setName = regressionRunNameGenerator(trialFiles, alignToList, tsInfo)
    setPath = os.path.join(folderPath, setName + '.h5')
    if trialStats is None:
            trialStats, _, _ = loadEventInfo(folderPath, eventInfo, requested = ['trialStats'])
    trialsIdx = trialStats.index

    timeSeriesFeatures = {}
    tsInfo.update({'whichColumns':targetVariables})
    for targetVariable in targetVariables:
        waveformsDict = {}
        for idx, alignTo in enumerate(alignToList):
            tsInfo.update({'alignTo':alignTo})
            timeSeries = loadTrialTriggeredTimeSeries(folderPath, tsInfo,
                dataDF = None, trialStats = trialStats, eventInfo = eventInfo)

            chIdx = timeSeries['ChannelLabel'].index(targetVariable)
            waveforms = timeSeries['Waveforms'][chIdx][trialsIdx, :]

            dT = timeSeries['basic_headers']['timeStampResolution'] ** -1
            t = np.linspace(timeSeries['basic_headers']['windowSize'][0],
                timeSeries['basic_headers']['windowSize'][1],waveforms.shape[1])
            tFull = np.unique(np.concatenate([t, binCenters.values]))
            theseWaveformsDF = pd.DataFrame(np.nan, index = trialsIdx, columns = tFull)
            theseWaveformsDF.loc[:,t] = waveforms
            theseWaveformsDF.index.name = 'trial'
            waveformsDict.update({alignTo:theseWaveformsDF})

        waveformsDF = pd.concat(waveformsDict, names = ['alignLabel'])
        waveformsDF.interpolate(axis = 1, inplace = True)
        targets = waveformsDF.loc[:, binCenters].copy()
        targets.columns.name = 'bin'
        targets = targets.stack()
        timeSeriesFeatures.update({targetVariable:targets})
    #pdb.set_trace()
    return pd.concat(timeSeriesFeatures, names = ['joint'], axis = 1)

def getSpikeMatsAsFeatures(folderPath,
    trialStats = None, eventInfo = None,
    trialFiles = None,
    binCenters = None,
    rasterOpts = None,
    targetCategory = None,
    alignToList = None
    ):
    arrayNames = []

    if trialStats is None:
        trialStats, _, _ = loadEventInfo(folderPath, eventInfo, requested = ['trialStats'])

    if binCenters is None:
        doWeHaveTheBinsYet = False
    else:
        theBins = binCenters
        doWeHaveTheBinsYet = True

    rasterOpts.update({'separateBy':targetCategory})

    categories, uniqueCategories, catIndices, validTrials =\
         getTrialCategories(trialStats, targetCategory)

    spikeMatsDict = {}

    for key, value in trialFiles.items():
        # key are ARRAY NAMES!
        spikeStruct, newSpikes = loadSpikeInfo(folderPath, key, value)
        arrayNames.append(key)
        spikeMatsThisArrayDict = {}

        for alignTo in alignToList:
            rasterOpts.update({'alignTo':alignTo})

            newSpikeMats = loadTrialBinnedArray(folderPath,
                key, value,
                eventInfo, rasterOpts,
                whichTrial = None,
                spikes = newSpikes,
                trialStats = trialStats, chans = None,
                correctAlignmentSpikes = 0)

            #theseValidTrials = np.logical_and(validTrials,trialStats[alignTo].notnull())
            whichTrials = trialStats.index[validTrials]
            spikeMatCubes, spikeMatIdx, spikeMatCols = getSpikeMatsForIdx(newSpikeMats, whichTrials, startBin = rasterOpts['windowSize'][0])

            if not doWeHaveTheBinsYet:
                theBins = spikeMatCols
                doWeHaveTheBinsYet = True
            theseOservationColumns = pd.MultiIndex.from_product((whichTrials,theBins), names=['trial','bin'])
            spikeMat2D = pd.DataFrame(np.nan,index = spikeMatIdx, columns = theseOservationColumns)

            spikeMat2D.index.name = 'unit'
            for idx1, binIDx in enumerate(theBins):
                for idx2, trialIDx in enumerate(whichTrials):
                    spikeMat2D.loc[:, (trialIDx, binIDx)] = spikeMatCubes[:, idx1, idx2]

            #fig, ax = plt.subplots(); plt.spy(spikeMat2D); plt.axis('equal'); plt.show()
            #fig, ax = plt.subplots(); plt.spy(spikeMatCubes[0, :, :]); plt.axis('equal'); plt.show()
            #spikeMat2D.index = spikeMat2D.index.remove_unused_levels()
            spikeMat2D.columns = spikeMat2D.columns.remove_unused_levels()
            spikeMatsThisArrayDict.update({alignTo:spikeMat2D})

        spikeMatsThisArray = pd.concat(spikeMatsThisArrayDict, axis = 1, names = ['alignLabel'])
        #spikeMatsThisArray.index = spikeMatsThisArray.index.remove_unused_levels()
        spikeMatsThisArray.columns = spikeMatsThisArray.columns.remove_unused_levels()
        spikeMatsDict.update({key:spikeMatsThisArray})

    spikeMats = pd.concat(spikeMatsDict, axis = 0, names = ['arrayName'])
    spikeMats.index = spikeMats.index.remove_unused_levels()
    spikeMats.columns = spikeMats.columns.remove_unused_levels()
    spikeMats = spikeMats.transpose()
    spikeMats.fillna(method = 'pad', inplace = True)

    return spikeMats, theBins

def loadCategorizationData(
    folderPath = None,
    trialFiles = None, eventInfo = None,
    rasterOpts = None,
    targetCategory = None, alignToList = None,
    addOverrideLabels = None):

    setName = categorizationRunNameGenerator(trialFiles, alignToList, targetCategory)
    setPath = os.path.join(folderPath, setName + '.h5')
    try:
        spikeMats = pd.read_hdf(setPath, 'spikeMatsFeatures')
        targets = pd.read_hdf(setPath, 'targets')
    except Exception:
        traceback.print_exc()
        ################
        spikeMats, binCenters = getSpikeMatsAsFeatures(folderPath,
            trialStats = None, eventInfo = eventInfo,
            trialFiles = trialFiles,
            rasterOpts = rasterOpts,
            targetCategory = targetCategory,
            alignToList = alignToList
            )
        ############################# functionalize
        targets = getTrialCategoriesAsFeatures(folderPath,
            trialStats = None, eventInfo = eventInfo,
            targetCategory = targetCategory,
            alignToList = alignToList)

        ###############################################################
        if addOverrideLabels is not None:
            for overrideLabel, overrideAlignToList in addOverrideLabels.items():
                ################
                spikeMatsOverride, _ = getSpikeMatsAsFeatures(folderPath,
                    trialStats = None, eventInfo = eventInfo,
                    trialFiles = trialFiles,
                    binCenters = binCenters,
                    rasterOpts = rasterOpts,
                    targetCategory = targetCategory,
                    alignToList = overrideAlignToList
                    )
                ############################# functionalize
                targetsOverride = getTrialCategoriesAsFeatures(folderPath,
                    trialStats = None, eventInfo = eventInfo,
                    targetCategory = targetCategory,
                    alignToList = overrideAlignToList, overrideLabel = overrideLabel)

                nOverridesToKeep = targets.value_counts().min()
                selectOverrides = targetsOverride[targetsOverride == overrideLabel].sample(n = targetsOverride.value_counts()[overrideLabel] - nOverridesToKeep).index
                spikeMatsOverride.drop(selectOverrides, axis = 0, inplace = True)
                targetsOverride.drop(selectOverrides, axis = 0, inplace = True)

                spikeMats = pd.concat((spikeMats, spikeMatsOverride), join = 'inner')
                targets = pd.concat((targets, targetsOverride), join = 'inner')

        try:
            spikeMats.to_hdf(setPath, 'spikeMatsFeatures')
            targets.to_hdf(setPath, 'targets',format="table")
        except Exception:
            traceback.print_exc()
            pass

    targets = targets.astype('category')
    return spikeMats, targets

def plotAverageTrialPDFReport(
    folderPath,
    spikesExclude = None,
    trialStats = None, trialEvents = None, motorData = None,
    eventInfo = None,
    spikeMatList = None, arrayNames = None,
    trialFiles = None,
    eventTsInfo = {
        'seriesName' : None,
        'recordName' : 'motorData',
        'whichColumns' : ['position'],
        'units' : 'a.u.',
        'subSample' : 30,
        'timeStampResolution' : int(3e4),
        'idxUnits' : 'samples',
        'alignTo' : 'FirstOnset',
        'separateBy' : 'Direction',
        'endOn' : 'ChoiceOnset',
        'windowSize' : [-0.1, .1],
        'removeBaseline' : False,
        'timeRange' : None
        },
    plotOpts = {
    'orderSpikesBy' : 'meanFR',
    'normalizationType' : 'logarithmic',
    'zAxis' : None,
    'forceRecalc' : True},
    kinTsInfo = {
    'seriesName' : None,
        'recordName' : 'angles',
        'whichColumns' : None,
        'units' : 'Degrees',
        'subSample' : 1,
        'timeStampResolution' : int(1e2),
        'idxUnits' : 'seconds',
        'alignTo' : 'FirstOnset',
        'separateBy' : 'Direction',
        'endOn' : 'ChoiceOnset',
        'windowSize' : [-0.1, .1],
        'removeBaseline' : False,
        'timeRange' : None
        },
    rasterOpts = {
    'kernelWidth' : 50e-3,
    'binInterval' : 50 * 1e-3, 'binWidth' : 100 * 1e-3,
    'windowSize' : (-1, .1),
    'alignTo' : 'FirstOnset',
    'separateBy' : 'Direction',
    'endOn' : 'ChoiceOnset',
    'discardEmpty': None, 'maxTrial' : None, 'timeRange' : None}
    ):

    newName = trialTriggeredTimeSeriesReportNameGenerator(eventTsInfo)
    pdfName = os.path.join(folderPath, newName + '.pdf')

    if trialStats is None:
        trialStats, _, _ = loadEventInfo(folderPath, eventInfo, requested = ['trialStats'])

    trialTriggeredAngles = loadTrialTriggeredTimeSeries(folderPath, kinTsInfo,
        dataDF = None, trialStats = trialStats, eventInfo = eventInfo, forceRecalc = plotOpts['forceRecalc'])

    trialTriggeredPosition = loadTrialTriggeredTimeSeries(folderPath, eventTsInfo,
        dataDF = None, trialStats = trialStats, eventInfo = eventInfo, forceRecalc = plotOpts['forceRecalc'])

    if spikeMatList is None:
        spikeMatList = {}
        arrayNames = []
        for key, value in trialFiles.items():
            spikeStruct, newSpikes = loadSpikeInfo(folderPath, key, value)
            newSpikeMats = loadTrialBinnedArray(folderPath,
                key, value,
                eventInfo, rasterOpts,
                whichTrial = None,
                spikes = newSpikes,
                trialStats = trialStats, chans = None,
                correctAlignmentSpikes = 0, forceRecalc = plotOpts['forceRecalc'])
            spikeMatList.update({key:newSpikeMats})
            arrayNames.append(key)

    nArrays = len(spikeMatList)
    #pdb.set_trace()
    categories, uniqueCategories, catIndices, validTrials = getTrialCategories(trialStats,
        eventTsInfo['separateBy'])

    zAxisAverage = {arrayName: {category:None for category in uniqueCategories.values} for arrayName in arrayNames}
    zAxisStd = {arrayName: {category:None for category in uniqueCategories.values} for arrayName in arrayNames}
    tTestResults = {arrayName : None for arrayName in arrayNames}
    averageSpikeMats, stdSpikeMats = getAverageSpikeMatsFromList(categories, uniqueCategories, validTrials, spikeMatList, arrayNames, startBin = rasterOpts['windowSize'][0])

    allValidTrials = [True if spikeMatList[arrayNames[0]][i] is not None else False\
        for i in trialStats.index]
    averageSpikeMatsAll,\
        stdSpikeMatsAll = getAverageSpikeMatsFromList(np.full((trialStats.shape[0],), 'all'),
        pd.Series(('all',)), allValidTrials, spikeMatList, arrayNames, startBin = rasterOpts['windowSize'][0])

    zAxisAverageAll = {arrayName: {'all' : None} for arrayName in arrayNames}
    zAxisStdAll = {arrayName: {'all' : None} for arrayName in arrayNames}

    for arrayName in arrayNames:

        tTestResults[arrayName] = getAllPairedSpikeMatTTestForCategories(categories, uniqueCategories, validTrials, spikeMatList[arrayName])

        zAxisAverageAll[arrayName]['all'] = (averageSpikeMatsAll[arrayName]['all'].min().min(),
        averageSpikeMatsAll[arrayName]['all'].max().max())
        zAxisStdAll[arrayName]['all'] = (stdSpikeMatsAll[arrayName]['all'].min().min(),
        stdSpikeMatsAll[arrayName]['all'].max().max())

        for category in uniqueCategories:
            zAxisAverage[arrayName][category] = (averageSpikeMats[arrayName][category].min().min(),
            averageSpikeMats[arrayName][category].max().max())
            zAxisStd[arrayName][category] = (stdSpikeMats[arrayName][category].min().min(),
            stdSpikeMats[arrayName][category].max().max())

        if plotOpts['orderSpikesBy'] is not None:
            #averageSpikeMatsAll[arrayName]['all'], spikeOrder = sortBinnedArray(averageSpikeMatsAll[arrayName]['all'], 'meanFR')
            averageSpikeMatsAll[arrayName]['all'], spikeOrder = sortBinnedArray(averageSpikeMatsAll[arrayName]['all'], plotOpts['orderSpikesBy'])
            stdSpikeMatsAll[arrayName]['all'] = stdSpikeMatsAll[arrayName]['all'].loc[spikeOrder, :]

            for categoryA, categoryB in itertools.combinations(uniqueCategories,2):
                tTestResults[arrayName][(categoryA, categoryB)] = tTestResults[arrayName][(categoryA, categoryB)].loc[spikeOrder, :]

            for category in uniqueCategories:
                stdSpikeMats[arrayName][category]     = stdSpikeMats[arrayName][category].loc[spikeOrder, :]
                averageSpikeMats[arrayName][category] = averageSpikeMats[arrayName][category].loc[spikeOrder, :]

    zAxisToUseAverage = [np.inf, -np.inf]
    zAxisToUseStd = [np.inf, -np.inf]

    for arrayName in arrayNames:
        zAxisToUseAverage[0] = min(zAxisToUseAverage[0], zAxisAverageAll[arrayName]['all'][0])
        zAxisToUseAverage[1] = max(zAxisToUseAverage[1], zAxisAverageAll[arrayName]['all'][1])
        zAxisToUseStd[0] = min(zAxisToUseStd[0], zAxisStdAll[arrayName]['all'][0])
        zAxisToUseStd[1] = max(zAxisToUseStd[1], zAxisStdAll[arrayName]['all'][1])
        for category in uniqueCategories:
            zAxisToUseAverage[0] = min(zAxisToUseAverage[0], zAxisAverage[arrayName][category][0])
            zAxisToUseAverage[1] = max(zAxisToUseAverage[1], zAxisAverage[arrayName][category][1])
            zAxisToUseStd[0] = min(zAxisToUseStd[0], zAxisStd[arrayName][category][0])
            zAxisToUseStd[1] = max(zAxisToUseStd[1], zAxisStd[arrayName][category][1])

    print('zAxisToUseAverage = ({})'.format(zAxisToUseAverage))
    print('zAxisToUseStd = ({})'.format(zAxisToUseStd))
    labelFontSize = LABELFONTSIZE
    with PdfPages(pdfName) as pdf:
        matplotlib.rc('figure', figsize=(12,36))
        fig, ax = plt.subplots(2 + 2 * nArrays,1)
        for chanIdx in trialTriggeredPosition['ChannelID']:
            plotSpike(trialTriggeredPosition, ax = ax[0], channel = chanIdx,
                axesLabel = True, showNow = False, errorMultiplier = 1)
        ax[0].set_title(eventTsInfo['whichColumns'], fontsize = labelFontSize)
        for chanIdx in trialTriggeredAngles['ChannelID']:
            plotSpike(trialTriggeredAngles, ax = ax[1], channel = chanIdx,
                axesLabel = True, showNow = False, errorMultiplier = 1)
        ax[1].set_title(kinTsInfo['recordName'], fontsize = labelFontSize)
        for idx, arrayName in enumerate(arrayNames):
            fig, im = hf.plotBinnedSpikes(averageSpikeMatsAll[arrayName]['all'], show = False,
                ax = ax[2 + 2 *idx],zAxis = zAxisToUseAverage,
                normalizationType = plotOpts['normalizationType'])
            ax[2 + 2 *idx].set_title(arrayName + ' Average', fontsize = labelFontSize)
            fig, im = hf.plotBinnedSpikes(stdSpikeMatsAll[arrayName]['all'], show = False,
                ax = ax[2 + 2 *idx + 1],zAxis = zAxisToUseStd,
                normalizationType = plotOpts['normalizationType'])
            ax[2 + 2 *idx+1].set_title(arrayName + ' Std', fontsize = labelFontSize)
        pdf.savefig()
        plt.close()

        for idx, category in enumerate(uniqueCategories):
            allOtherCategories = np.delete(catIndices.values, [uniqueCategories[uniqueCategories == category].index[0]])
            print('Plotting {}'.format(category))
            fig, ax = plt.subplots(2 + 2 * nArrays,1)
            for chanIdx in trialTriggeredPosition['ChannelID']:
                plotSpike(trialTriggeredPosition, ax = ax[0], channel = chanIdx,
                    axesLabel = True, showNow = False, errorMultiplier = 1, ignoreUnits = allOtherCategories.tolist())
            ax[0].set_title(eventTsInfo['whichColumns'], fontsize = labelFontSize)
            for chanIdx in trialTriggeredAngles['ChannelID']:
                plotSpike(trialTriggeredAngles, ax = ax[1], channel = chanIdx,
                    axesLabel = True, showNow = False, errorMultiplier = 1, ignoreUnits = allOtherCategories.tolist())
            ax[1].set_title(kinTsInfo['recordName'], fontsize = labelFontSize)
            for idx, arrayName in enumerate(arrayNames):
                fig, im = hf.plotBinnedSpikes(averageSpikeMats[arrayName][category], show = False,
                    ax = ax[2 + 2 *idx], zAxis = zAxisToUseAverage,
                    normalizationType = plotOpts['normalizationType'])
                ax[2 + 2 *idx].set_title('{} {} Average'.format(arrayName, category), fontsize = labelFontSize)
                fig, im = hf.plotBinnedSpikes(stdSpikeMats[arrayName][category], show = False,
                    ax = ax[2 + 2 *idx + 1],zAxis = zAxisToUseStd,
                    normalizationType = plotOpts['normalizationType'])
                ax[2 + 2 *idx+1].set_title('{} {} Std'.format(arrayName, category), fontsize = labelFontSize)
            pdf.savefig()
            plt.close()

        for categoryA, categoryB in itertools.combinations(uniqueCategories,2):
            allOtherCategories = np.delete(catIndices.values,
                [uniqueCategories[uniqueCategories == categoryA].index[0],
                uniqueCategories[uniqueCategories == categoryB].index[0]])
            print('Plotting {}'.format((categoryA, categoryB)))
            fig, ax = plt.subplots(2 + 3*nArrays,1)
            for chanIdx in trialTriggeredPosition['ChannelID']:
                plotSpike(trialTriggeredPosition, ax = ax[0], channel = chanIdx,
                    axesLabel = True, showNow = False, errorMultiplier = 1, ignoreUnits = allOtherCategories.tolist())
            ax[0].set_title(eventTsInfo['whichColumns'], fontsize = labelFontSize)
            for chanIdx in trialTriggeredAngles['ChannelID']:
                plotSpike(trialTriggeredAngles, ax = ax[1], channel = chanIdx,
                    axesLabel = True, showNow = False, errorMultiplier = 1, ignoreUnits = allOtherCategories.tolist())
            ax[1].set_title(kinTsInfo['recordName'], fontsize = labelFontSize)

            if plotOpts['normalizationType'] == 'LogNorm':
                thisNorm = 'SymLogNorm'
            else:
                thisNorm =  plotOpts['normalizationType']

            nComparisons = 1
            pThres = 1e-2
            # correct for multiple comparisons across arrays:
            for idx, arrayName in enumerate(arrayNames):
                theseTTestResults = tTestResults[arrayName][(categoryA,categoryB)]
                nComparisons = nComparisons * theseTTestResults.shape[0] * theseTTestResults.shape[1]

            for idx, arrayName in enumerate(arrayNames):
                theseTTestResults = tTestResults[arrayName][(categoryA,categoryB)]
                # correct for multiple comparisons across arrays:
                theseTTestResults = theseTTestResults * nComparisons / (theseTTestResults.shape[0] * theseTTestResults.shape[1])
                # Difference in means
                fig, im = hf.plotBinnedSpikes(averageSpikeMats[arrayName][categoryA] - averageSpikeMats[arrayName][categoryB], show = False,
                    ax = ax[2 + 3*idx],
                    normalizationType = thisNorm )
                ax[2 + 3*idx].set_title('{}: Difference Between {} and {}'.format(arrayName, categoryA, categoryB), fontsize = labelFontSize)
                # Significance of difference
                fig, im = hf.plotBinnedSpikes(theseTTestResults, show = False,
                    ax = ax[2 + 3*idx+1],
                    normalizationType = 'LogNorm')
                ax[2 + 3*idx+1].set_title('{}: T Test P Value for above comparison'.format(arrayName), fontsize = labelFontSize)

                significantBins = theseTTestResults < pThres
                significantBins.columns.name = 'Time (sec)'
                significantBins.index.name = 'Trial'

                nameStr = 'p < {}'.format(pThres)
                significantBins = significantBins.sum()
                #pdb.set_trace()
                ax[2 + 3*idx+2].bar(significantBins.index, significantBins, width = rasterOpts['binInterval'])
                ax[2 + 3*idx+2].set_ylabel('Count', fontsize = labelFontSize)
                ax[2 + 3*idx+2].set_title('{}: count of {}'.format(arrayName, nameStr), fontsize = labelFontSize)
            pdf.savefig()
            plt.close()
#@profile
def spikePDFReport(folderPath, spikes, spikeStruct,
    arrayName = None, arrayInfo = None,
    correctAlignmentSpikes = 0,
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'},
    rasterOpts = {
    'kernelWidth' : 50e-3,
    'binInterval' : 2.5* 1e-3, 'binWidth' : 5* 1e-3,
    'windowSize' : (-0.25, 1),
    'alignTo' : 'FirstOnset',
    'separateBy' : 'Direction',
    'discardEmpty': None, 'maxTrial' : None, 'timeRange' : None},
    trialStats = None, enableFR = False, newName = None):

    if correctAlignmentSpikes: #correctAlignmentSpikes units in samples
        spikes = hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

    if newName is None:
        pdfName = os.path.join(folderPath, 'spikePDFReport.pdf')
    else:
        pdfName = os.path.join(folderPath , newName + '.pdf')

    if any((arrayName is None, arrayInfo is None)):
        arrayName, arrayInfo, partialRasterOpts = trialBinnedSpikesNameRetrieve(newName)
        arrayInfo['nevIDs'] = spikes['ChannelID']

    with PdfPages(pdfName) as pdf:
        plotSpikePanel(spikeStruct, spikes)
        pdf.savefig()
        plt.close()

        for idx, channel in enumerate(spikes['ChannelID']):
            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
                print("Running spikePDFReport: %d%%" % int((idx + 1) * 100 / len(spikes['ChannelID'])), end = endChar)
            else:
                print("Running spikePDFReport: %d%%" % int((idx + 1) * 100 / len(spikes['ChannelID'])))

            unitsOnThisChan = np.unique(spikes['Classification'][idx])
            if unitsOnThisChan is not None:
                if len(unitsOnThisChan) > 0:
                    fig, ax = plt.subplots(nrows = 1, ncols = 2)
                    plotSpike(spikes, channel = channel, ax = ax[0],
                        axesLabel = True)
                    isiBins = np.linspace(0, 80, 40)
                    kde_kws = {'clip' : (isiBins[0] * 0.8, isiBins[-1] * 1.2),
                        'bw' : 'silverman', 'gridsize' : 500}
                    plotISIHistogram(spikes, channel = channel, bins = isiBins,
                        ax = ax[1], kde_kws = kde_kws)
                    pdf.savefig()
                    plt.close()

                    if len(spikes['Waveforms'][idx].shape) == 3:
                        plotSpike(spikes, channel = channel, acrossArray = True,
                         xcoords = spikeStruct['xcoords'], ycoords = spikeStruct['ycoords'])
                        pdf.savefig()
                        plt.close()

                    if rasterOpts['alignTo'] is not None and trialStats is not None:
                        spikeMats, categories, selectedIndices = loadTrialBinnedSpike(folderPath,
                            arrayName, arrayInfo,
                            channel,
                            rasterOpts,
                            trialStats = trialStats, spikes = spikes,
                            correctAlignmentSpikes = 0,
                            forceRecalc = False)

                        spikeMats, categories, plotFig, plotAx, uniqueCategories, curLine = plotTrialRaster(
                            trialStats = trialStats, channel = channel,
                            spikeMats = spikeMats, categories = categories,
                            plotOpts = plotOpts, rasterOpts = rasterOpts)

                        #plotAx = plotRaster(spikes, trialStats, alignTo = plotRastersAlignedTo, windowSize = (-0.5, 2), channel = channel, separateBy = plotRastersSeparatedBy)
                        if enableFR:
                            plotTrialFR(spikeMats = spikeMats, categories = categories,
                                fig = plotFig, ax = plotAx, uniqueCategories = uniqueCategories, twin = True,
                                plotOpts = plotOpts, rasterOpts = rasterOpts)
                            #plotFR(spikes, trialStats, alignTo = plotRastersAlignedTo, windowSize = (-0.5, 2), channel = channel, separateBy = plotRastersSeparatedBy, ax = plotAx, twin = True)
                        pdf.savefig()
                        plt.close()

def trialPDFReport(folderPath,
    eventInfo,
    trialFiles,
    kinematicsFileName,
    trialStats = None,
    kinPosOpts = {
    'ns5FileName' : 'Trial001',
    'selectHeaders' : None,
    'reIndex' : None,
    'flip' : None,
    'lowCutoff': None
    },
    correctAlignmentSpikes = 0, rasterOpts = {
    'kernelWidth' : 50e-3,
    'binInterval' : 50 * 1e-3, 'binWidth' : 100 * 1e-3,
    'windowSize' : (-1, .1),
    'alignTo' : 'FirstOnset',
    'separateBy' : 'Direction',
    'endOn' : 'ChoiceOnset',
    'discardEmpty': None, 'maxTrial' : None, 'timeRange' : None},
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'},
    analogSubset = ['position'], analogLabel = '', kinematicsSubset =  ['Hip Right X', 'Knee Right X', 'Ankle Right X'],\
    kinematicLabel = '', eventSubset = ['Right LED Onset', 'Right Button Onset', 'Left LED Onset', 'Left Button Onset', 'Movement Onset', 'Movement Offset'],\
    maxTrial = None):

    filePath = os.path.join(folderPath, eventInfo['ns5FileName'] + '_report.pdf')
    if trialStats is None:
        trialStats, trialEvents, motorData = loadEventInfo(folderPath, eventInfo)
    with PdfPages(filePath) as pdf:
        if maxTrial is None:
            numTrials = len(trialStats.index)
        else:
            numTrials = maxTrial
        for idx, curTrial in trialStats.iterrows():

            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
                print("Running trialPDFReport: %d%%" % int((idx + 1) * 100 /numTrials), end = endChar)
            else:
                print("Running trialPDFReport: %d%%" % int((idx + 1) * 100 / numTrials))

            if maxTrial is not None:
                if idx > maxTrial:
                    break
            try:
                fig, motorPlotAxes = plotSingleTrial(
                    idx,
                    spikesExclude = None,
                    folderPath = folderPath,
                    eventInfo = eventInfo,
                    trialStats = trialStats,
                    trialFiles = trialFiles,
                    kinematicsFileName = kinematicsFileName,
                    zAxis = None,
                    kinPosOpts = kinPosOpts,
                    rasterOpts = rasterOpts,
                    plotOpts = plotOpts,
                    analogSubset = ['position'], analogLabel = 'Pedal Position (a.u.)',
                    kinematicsSubset = ['Hip Right X', 'Knee Right X', 'Ankle Right X'],
                    kinematicLabel = 'Joint Angle (deg)', orderSpikesBy = None)

                pdf.savefig()
                plt.close()
            except Exception:
                traceback.print_exc()
                pdf.savefig()
                plt.close()

def spikeTriggeredAveragePDFReport(folderPath,
    spikesFrom, spikesTo,
    spikesFromList, spikesToList,
    arrayNameFrom = None, arrayInfoFrom = None, arrayNameTo = None, arrayInfoTo = None,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0,
    newName = None, enableFR = True,
    plotOpts = {'type' : 'ticks', 'errorBar' : 'sem'},
    rasterOpts = {'kernelWidth' : 5e-3, 'binInterval' : (3**-1)* 1e-3,
        'binWidth' : (3**-1)* 1e-3, 'windowSize' : (-0.01, .11),'maxSpikesTo':None,
        'discardEmpty':None, 'timeRange':None,
        'separateByFunArgs': None,'separateByFunKWArgs': {'type' : 'Classification'}}
    ):

    if newName is None:
        pdfName = os.path.join(folderPath,'STA_Report.pdf')
    else:
        pdfName = os.path.join(folderPath, newName + '.pdf')

    if any((arrayNameFrom is None, arrayInfoFrom is None, arrayNameTo is None, arrayInfoTo is None)):
        arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo = spikeBinnedSpikesNameRetrieve(newName)
        arrayInfoFrom['nevIDs'] = spikesFrom['ChannelID']
        arrayInfoTo['nevIDs'] = spikesTo['ChannelID']

    if correctAlignmentSpikesFrom: #correctAlignmentSpikesFrom units in samples
        spikesFrom = hf.correctSpikeAlignment(spikesFrom, correctAlignmentSpikesFrom)
    if correctAlignmentSpikesTo: #correctAlignmentSpikesFrom units in samples
        spikesTo = hf.correctSpikeAlignment(spikesTo, correctAlignmentSpikesTo)

    with PdfPages(pdfName) as pdf:
        pairsNum = len(spikesFromList) * len(spikesToList)
        pairCount = 0
        for spikesFromIdx, spikesToIdx in itertools.product(spikesFromList, spikesToList):
            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
                print("Running staPDFReport: %d%%" % int((pairCount + 1) * 100 / pairsNum), end = endChar)
            else:
                print("Running staPDFReport: %d%%" % int((pairCount + 1) * 100 / pairsNum))

            pairCount += 1

            spikeMats, categories, selectedIndices = loadSpikeBinnedSpike(folderPath,
                arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo,
                spikesFromIdx, spikesToIdx,
                rasterOpts,
                spikesFrom = spikesFrom, spikesTo = spikesTo,
                correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0)

            """
            spikeMats, categories, selectedIndices = hf.binnedSpikesAlignedToSpikes(spikesFrom, spikesTo,
                spikesFromIdx, spikesToIdx,
                rasterOpts['binInterval'], rasterOpts['binWidth'], windowSize = rasterOpts['windowSize'])
            """
            titleOverride = 'Channel %d triggered by channel %d' % (
                            spikesFromIdx['chan'], spikesToIdx['chan'])

            _, fig, ax, selectedIndices = plotSpikeTriggeredRaster(\
                spikesFrom = None, spikesTo = None,
                spikesFromIdx = None, spikesToIdx = None,
                spikeMats = spikeMats,
                fig = None, ax = None,
                categories = categories,
                rasterOpts = rasterOpts,
                showNow = False, plotOpts = plotOpts)

            plotSpikeTriggeredFR(spikesFrom = None, spikesTo = None,
                spikesFromIdx = None, spikesToIdx = None,
                spikeMats = spikeMats, titleOverride = titleOverride,
                fig = fig, ax = ax, twin = True,
                categories = categories,
                rasterOpts = rasterOpts,
                showNow = False, plotOpts = plotOpts)
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(nrows = 1, ncols = 2)
            plotSpike(spikesFrom, channel = spikesFromIdx['chan'], ax = ax[0],
                axesLabel = True)
            plotSpike(spikesTo, channel = spikesToIdx['chan'], ax = ax[1],
                axesLabel = True)
            pdf.savefig()
            plt.close()

def generateSpikeTriggeredAverageReport(folderPath, trialFileFrom, trialFileTo,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0,
    plotOpts = None,
    rasterOpts = None):

    key, value = next(iter(trialFileFrom.items()))
    arrayInfoFrom = value
    arrayNameFrom = key
    spikeStructFrom, spikesFrom = loadSpikeInfo(folderPath, key, value)
    key, value = next(iter(trialFileTo.items()))
    arrayInfoTo = value
    arrayNameTo = key
    spikeStructTo, spikesTo = loadSpikeInfo(folderPath, key, value)
    newName = spikeBinnedSpikesNameGenerator(arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo)
    spikesFromList = []

    for idx, channel in enumerate(spikesFrom['ChannelID']):
        unitsOnThisChan = np.unique(spikesFrom['Classification'][idx])
        if unitsOnThisChan.any():
            spikesFromList.append({'chan':channel,'units':list(range(len(unitsOnThisChan)))})

    spikesToList = []
    for idx, channel in enumerate(spikesTo['ChannelID']):
        unitsOnThisChan = np.unique(spikesTo['Classification'][idx])
        if unitsOnThisChan.any():
            spikesToList.append({'chan':channel})
    #pdb.set_trace()
    spikeTriggeredAveragePDFReport(folderPath, spikesFrom, spikesTo, spikesFromList,
        spikesToList,
        arrayNameFrom = arrayNameFrom, arrayInfoFrom = arrayInfoFrom,
        arrayNameTo = arrayNameTo, arrayInfoTo = arrayInfoTo,
        correctAlignmentSpikesFrom = correctAlignmentSpikesFrom, correctAlignmentSpikesTo = correctAlignmentSpikesTo,
        newName = newName, enableFR = True,
        plotOpts = plotOpts, rasterOpts = rasterOpts)

def generateStimTriggeredAverageReport(folderPath, trialFileFrom, trialFileStim,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesStim = 0,
    plotOpts = None,
    rasterOpts = None):

    key, value = next(iter(trialFileFrom.items()))
    arrayInfoFrom = value
    arrayNameFrom = key
    spikeStructFrom, spikesFrom = loadSpikeInfo(folderPath, key, value)

    spikesFromList = []
    for idx, channel in enumerate(spikesFrom['ChannelID']):
        unitsOnThisChan = np.unique(spikesFrom['Classification'][idx])
        if unitsOnThisChan.any():
            spikesFromList.append({'chan':channel,'units':list(range(len(unitsOnThisChan)))})

    key, value = next(iter(trialFileStim.items()))

    arrayInfoTo = value
    arrayNameTo = key
    spikeStructStim, spikesStim = loadSpikeInfo(folderPath, key, value) # stim only from mat file
    newName = spikeBinnedSpikesNameGenerator(arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo)
    spikesStimList = []

    impedances = pd.read_csv(os.path.join(folderPath, 'Murdoc Impedances.txt'), skiprows = [0, 1, 2, 3, 4, 5, 6, 7, 9], delim_whitespace = True)
    impedances['MaxAmp(uA)'] = (8.5* 1e3) / impedances['Mag(kOhms)']
    impedances.index = impedances.index + 5121

    spikesStim['Units'] = 'uA'
    catSpikeFun = hf.catSpikesGenerator(5, type = 'minPeak', subSet = slice(10, 30))
    for idx, channel in enumerate(spikesStim['ChannelID']):
        unitsOnThisChan = np.unique(spikesStim['Classification'][idx])
        if unitsOnThisChan.any():
            spikesStimList.append({'chan':channel})
        spikesStim['Waveforms'][idx] = (np.array(spikesStim['Waveforms'][idx],
            dtype = np.float32) * 8.5 *1e3) / (2 **15 * impedances.loc[channel, 'Mag(kOhms)'])

        spikesStim['Classification'][idx] = catSpikeFun(spikesStim, idx)
    #pdb.set_trace()
    spikeTriggeredAveragePDFReport(folderPath, spikesFrom, spikesStim, spikesFromList,
        spikesStimList,
        arrayNameFrom = arrayNameFrom, arrayInfoFrom = arrayInfoFrom,
        arrayNameTo = arrayNameTo, arrayInfoTo = arrayInfoTo,
        correctAlignmentSpikesFrom = correctAlignmentSpikesFrom,
        correctAlignmentSpikesTo = correctAlignmentSpikesStim,
        newName = newName, enableFR = True,
        plotOpts = plotOpts, rasterOpts = rasterOpts)

def loadEventInfo(folderPath, eventInfo,
    requested = ['trialStats', 'trialEvents', 'motorData'],
    forceRecalc = False):

    setName = eventInfo['ns5FileName'] + '_eventInfo'
    setPath = os.path.join(folderPath, setName + '.h5')
    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            if 'trialStats' in requested:
                trialStats  = pd.read_hdf(setPath, 'trialStats')
            else:
                trialStats = None

            if 'trialEvents' in requested:
                trialEvents = pd.read_hdf(setPath, 'trialEvents')
            else:
                trialEvents = None
            if 'motorData' in requested:
                motorData = pd.read_hdf(setPath, 'motorData')
            else:
                motorData = None
            #print('Loaded trial data from pickle.')
        except Exception:
            traceback.print_exc()
            print('Trial data not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        motorData = mea.getMotorData(os.path.join(folderPath,
            eventInfo['ns5FileName']) + '.ns5', eventInfo['inputIDs'], 0 , 'all')
        trialStats, trialEvents = mea.getTrials(motorData)

        trialStats.to_hdf(setPath, 'trialStats')
        if 'trialStats' not in requested:
            trialStats = None

        trialEvents.to_hdf(setPath, 'trialEvents')
        if 'trialEvents' not in requested:
            trialEvents = None

        motorData.to_hdf(setPath, 'motorData')
        if 'motorData' not in requested:
            motorData = None

        print('Recalculated trial data and saved to pickle.')
    return trialStats, trialEvents, motorData

def capitalizeFirstLetter(stringInput):
    return stringInput[0].capitalize() + stringInput[1:]

def unCapitalizeFirstLetter(stringInput):
    return stringInput[0].lower() + stringInput[1:]

def spikesNameGenerator(arrayName, arrayInfo):
    if arrayInfo['excludeClus'] is None:
        excludeStr = ''
    else:
        excludeStr = '_'.join([str(i) for i in arrayInfo['excludeClus']])
    return arrayInfo['ns5FileName'] + '_spikes' + capitalizeFirstLetter(arrayName) + '_exclude_' + excludeStr + '_' + arrayInfo['origin']

def timeTriggeredTimeSeriesNameGenerator():
    pass
    return
def spikesNameGenerator(arrayName, arrayInfo):
    if arrayInfo['excludeClus'] is None:
        excludeStr = ''
    else:
        excludeStr = '_'.join([str(i) for i in arrayInfo['excludeClus']])
    return arrayInfo['ns5FileName'] + '_spikes' + capitalizeFirstLetter(arrayName) + '_exclude_' + excludeStr + '_' + arrayInfo['origin']

def trialBinnedSpikesNameGenerator(arrayName, arrayInfo, rasterOpts):
    return spikesNameGenerator(arrayName, arrayInfo) + '_ALIGNEDTO_' + rasterOpts['alignTo'] + '_SEPARATEDBY_' + rasterOpts['separateBy']

def spikeBinnedSpikesNameGenerator(arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo):
    return spikesNameGenerator(arrayNameFrom, arrayInfoFrom) + '_ALIGNEDTO_' + spikesNameGenerator(arrayNameTo, arrayInfoTo)

def spikeBinnedArrayNameGenerator(arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo):
    return spikesNameGenerator(arrayNameFrom, arrayInfoFrom) + '_TRIGGEREDBY_' + spikesNameGenerator(arrayNameTo, arrayInfoTo)

def trialBinnedArrayNameGenerator(arrayName, arrayInfo, rasterOpts):
    return spikesNameGenerator(arrayName, arrayInfo) + '_BETWEEN_' + rasterOpts['alignTo'] +"_AND_"+ '{}'.format(rasterOpts['endOn'])

def categorizationRunNameGenerator(trialFiles, alignToList, targetCategory):

    arrayNames = []
    for arrayName, arrayInfo in trialFiles.items():
        arrayNames.append(arrayName)

    outputName = arrayInfo['ns5FileName'] + '_'

    for arrayName in sorted(arrayNames):
        outputName += arrayName + '_'

    outputName += 'ALIGNEDTO_{}'.format(alignToList)
    outputName += '_CATEGORIZATION_TARGET_{}'.format(targetCategory)

    return outputName

def regressionRunNameGenerator(trialFiles, alignToList, tsInfo):
    arrayNames = []
    for arrayName, arrayInfo in trialFiles.items():
        arrayNames.append(arrayName)
    outputName = arrayInfo['ns5FileName'] + '_'
    for arrayName in sorted(arrayNames):
        outputName += arrayName + '_'

    outputName += 'ALIGNEDTO_{}'.format(alignToList)
    outputName += '_REGRESSION_TARGET_' + tsInfo['seriesName'] + '_{}'.format(tsInfo['recordName'])

    return outputName

def spikesNameRetrieve(spikesName):

    arrayInfo = {'ns5FileName' : None, 'nevIDs' : None, 'excludeClus' : None, 'origin' : None}

    arrayInfo['ns5FileName'] = spikesName.split('_spikes')[0]
    arrayInfo['origin'] = spikesName.split('_')[-1]
    excludeRe = r'_exclude(_\d*)*_'
    match = re.search(excludeRe, spikesName)
    arrayInfo['excludeClus'] = []
    for i in match.groups():
        if i[1:]:
            arrayInfo['excludeClus'].append(int(i[1:]))

    arrayName = spikesName.split('_spikes')[1].split('_exclude_')[0]
    arrayName = unCapitalizeFirstLetter(arrayName)

    return arrayName, arrayInfo

def spikeBinnedSpikesNameRetrieve(spikesName):
    subNames = spikesName.split('_ALIGNEDTO_')
    fromName = subNames[0]
    arrayNameFrom, arrayInfoFrom = spikesNameRetrieve(fromName)
    toName   = subNames[1]
    arrayNameTo, arrayInfoTo = spikesNameRetrieve(toName)
    return arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo

def trialBinnedSpikesNameRetrieve(spikesName):
    subNames = spikesName.split('_ALIGNEDTO_')
    unitsName = subNames[0]
    arrayName, arrayInfo = spikesNameRetrieve(unitsName )

    rasterOpts = {'alignTo' : None, 'separateBy' : None}
    rasterOpts['alignTo'] = subNames[-1].split('_SEPARATEDBY_')[0]
    rasterOpts['separateBy'] = subNames[-1].split('_SEPARATEDBY_')[-1]

    return arrayName, arrayInfo, rasterOpts

def trialBinnedArrayNameRetrieve(spikeNames):
    subNames = spikesName.split('_ALIGNEDTO_')
    unitsName = subNames[0]
    arrayName, arrayInfo = spikesNameRetrieve(unitsName )

    rasterOpts = {'alignTo' : None, 'separateBy' : None}
    rasterOpts['alignTo'] = subNames[-1].split('_BETWEEN_')[0]
    rasterOpts['endOn'] = spikeNames.split('_AND_')[-1]

    return arrayName, arrayInfo, rasterOpts

def loadSpikeInfo(folderPath, arrayName, arrayInfo, forceRecalc = False):

    setName = spikesNameGenerator(arrayName, arrayInfo)
    setPath = os.path.join(folderPath, setName + '.pickle')

    try:
        #raise Exception('DEBUGG')
        spikeStruct = pd.read_hdf(os.path.join(folderPath, arrayInfo['ns5FileName'] + '_spikeStruct' + capitalizeFirstLetter(arrayName) + '.h5'), 'spikeStruct')
    except Exception:
        traceback.print_exc()
        print('Spike metadata not pickled. Recalculating...')
        arrayMap = pd.read_table(os.path.join(folderPath, 'array_map.cmp'), skiprows = 13)
        spikeStruct = pd.DataFrame(np.nan, index = range(1,129), columns = ['xcoords', 'ycoords', 'nevID', 'bank', 'bankID'])
        bankLookup = {'A' :0, 'B':1, 'C':2, 'D': 3}
        for rowIdx, row in arrayMap.iterrows():
            if arrayName in row['label']:
                elec = int(row['elec']) + bankLookup[row['bank']] * 32
                elecLabel = int(row['label'].split(arrayName)[-1])
                spikeStruct.loc[elecLabel, 'xcoords'] = row['row']
                spikeStruct.loc[elecLabel, 'ycoords'] = row['//col']
                spikeStruct.loc[elecLabel, 'nevID'] = elec
                spikeStruct.loc[elecLabel, 'bank'] = row['bank']
                spikeStruct.loc[elecLabel, 'bankID'] = int(row['elec'])
        #pdb.set_trace()
        spikeStruct.dropna(inplace = True)
        #spikeStruct = loadKSDir(os.path.join(folderPath, 'Kilosort/'+ arrayInfo['ns5FileName'] + '_' + capitalizeFirstLetter(arrayName)), loadPCs = False)
        spikeStruct.to_hdf(os.path.join(folderPath, arrayInfo['ns5FileName'] + '_spikeStruct' + capitalizeFirstLetter(arrayName) + '.h5'), 'spikeStruct')
        print('Recalculated spike metadata and saved to pickle.')


    #pdb.set_trace()
    nevIDs = spikeStruct.loc[arrayInfo['nevIDs'], 'nevID'].astype(int).tolist()
    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            spikes = pickle.load(open(setPath, 'rb'))

            # make sure the fifle contains all requested channels
            for chanIdx in spikes['ChannelID']:
                assert chanIdx in arrayInfo['nevIDs']
            #print('Loaded spike data from pickle.')
        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway=
            print('Spike data not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        if arrayInfo['origin'] == 'wave_clus':
            spikes = getWaveClusSpikes(
                os.path.join(folderPath, 'wave_clus', arrayInfo['ns5FileName']) + '/',
                nevIDs = nevIDs, plotting = False, excludeClus = arrayInfo['excludeClus'])

        elif arrayInfo['origin'] == 'nev':
            filePath = os.path.join(folderPath, arrayInfo['ns5FileName'] + '.nev')
            spikes = hf.getNEVData(filePath, nevIDs)

        elif arrayInfo['origin'] == 'mat':
            spikes = getNevMatSpikes(
                os.path.join(folderPath, arrayInfo['ns5FileName']+'.mat'),
                nevIDs = nevIDs, plotting = False, excludeClus = arrayInfo['excludeClus'])
        #pdb.set_trace()
        spikes['ChannelID'] = arrayInfo['nevIDs']
        pickle.dump(spikes,
            open(os.path.join(folderPath, setPath), 'wb')
            )
        print('Recalculated spike data from wave_clus folder and saved to pickle.')

    return spikeStruct, spikes

def loadSpikeBinnedSpike(folderPath,
    arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo,
    spikesFromIdx, spikesToIdx,
    rasterOpts,
    spikesFrom = None, spikesTo = None,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0,
    forceRecalc = False):

    setName = spikeBinnedSpikesNameGenerator(arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo)
    setPath = os.path.join(folderPath, setName + '.h5')

    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            with h5py.File(setPath, "r") as f:
                recordAttributes = f['/'+"rasterOpts"].attrs
                for key, value in rasterOpts.items():
                    if type(value) is not dict:
                        thisAttr = recordAttributes[key]

                        if isinstance(value, collections.Iterable):
                            for idx, valueComponent in enumerate(value):
                                assert (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx]))
                        else:
                            assert (value == thisAttr) or (value is None and np.isnan(thisAttr))
                    else:
                        for subKey, subValue in value.items():
                            thisAttr = recordAttributes[key + '_' + subKey]

                            if isinstance(subValue, collections.Iterable):
                                for idx, valueComponent in enumerate(subValue):
                                    assert (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx]))
                            else:
                                assert (subValue == thisAttr) or (subValue is None and np.isnan(thisAttr))

                #spikeMats, categories, selectedIndices = None, None, None
                requestedSpikeMat = '/' + str(spikesFromIdx['chan']) + '_to_' + str(spikesToIdx['chan'])

                spikeMatShape = f[requestedSpikeMat + '/spikeMats'].shape
                spikeMats = [f[requestedSpikeMat + '/spikeMats'][:,:,i] for i in range(spikeMatShape[2])]
                for idx, spikeMat in enumerate(spikeMats):
                    spikeMats[idx] = pd.DataFrame(spikeMat, index = f[requestedSpikeMat + '/index'], columns = f[requestedSpikeMat + '/columns'])

                categories = np.array(f[requestedSpikeMat + '/categories'])
                categories = pd.Series(categories, index = f[requestedSpikeMat + '/index'])

                selectedIndices = np.array(f[requestedSpikeMat + '/selectedIndices'])

                if selectedIndices.any():
                    selectedIndices = pd.Series(selectedIndices, index = f[requestedSpikeMat + '/index'])
                else:
                    selectedIndices = None
            # TODO: figure out how to load it...
            #print('Loaded spikeMats from h5.')
        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('SpikeMats not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        if spikesFrom is None:
            spikeStructFrom, spikesFrom = loadSpikeInfo(folderPath, arrayNameFrom, arrayInfoFrom)

        if correctAlignmentSpikesFrom: #correctAlignmentSpikesFrom units in samples
            spikesFrom = hf.correctSpikeAlignment(spikesFrom, correctAlignmentSpikesFrom)

        spikesFromList = []
        for idx, channel in enumerate(spikesFrom['ChannelID']):
            unitsOnThisChan = np.unique(spikesFrom['Classification'][idx])
            if unitsOnThisChan.any():
                spikesFromList.append({'chan':channel,'units':list(range(len(unitsOnThisChan)))})

        if spikesTo is None:
            spikeStructTo, spikesTo = loadSpikeInfo(folderPath, arrayNameTo, arrayInfoTo)

        if correctAlignmentSpikesTo: #correctAlignmentSpikesFrom units in samples
            spikesTo = hf.correctSpikeAlignment(spikesTo, correctAlignmentSpikesTo)

        spikesToList = []
        for idx, channel in enumerate(spikesTo['ChannelID']):
            unitsOnThisChan = np.unique(spikesTo['Classification'][idx])
            if unitsOnThisChan.any():
                spikesToList.append({'chan':channel,'units':list(range(len(unitsOnThisChan)))})

        with h5py.File(setPath, "w") as f:
            grp = f.create_group("rasterOpts")
            for key, value in rasterOpts.items():
                if type(value) is not dict:
                    if value is not None:
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = np.nan
                else:
                    for subKey, subValue in value.items():
                        if value is not None:
                            grp.attrs[key + '_' + subKey] = subValue
                        else:
                            grp.attrs[key + '_' + subKey] = np.nan

            if rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is not None:
                separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'], **rasterOpts['separateByFunKWArgs'])
            elif rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is None:
                separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'])
            elif rasterOpts['separateByFunArgs'] is None and rasterOpts['separateByFunKWArgs'] is not None:
                separateByFun = hf.catSpikesGenerator(**rasterOpts['separateByFunKWArgs'])

            pairsNum = len(spikesFromList) * len(spikesToList)
            pairCount = 0

            saveSpikesFromIdx, saveSpikesToIdx = spikesFromIdx, spikesToIdx
            saveSpikeMats, saveCategories, saveSelectedIndices = None, None, None

            for spikesToIdx in spikesToList:

                alignTimes, categories, selectedIndices = hf.spikeAlignmentTimes(spikesTo,spikesToIdx,
                    separateByFun = separateByFun,
                    timeRange = rasterOpts['timeRange'],
                    maxSpikesTo =rasterOpts['maxSpikesTo'], discardEmpty = rasterOpts['discardEmpty'])

                for spikesFromIdx in spikesFromList:
                    if os.fstat(0) == os.fstat(1):
                        endChar = '\r'
                        print("Running loadSpikeBinnedSpike: %d%%" % int((pairCount + 1) * 100 / pairsNum), end = endChar)
                    else:
                        print("Running loadSpikeBinnedSpike: %d%%" % int((pairCount + 1) * 100 / pairsNum))

                    try:
                        spikeMats = hf.binnedSpikesAligned(spikesFrom, alignTimes, rasterOpts['binInterval'],
                            rasterOpts['binWidth'], spikesFromIdx['chan'], windowSize = rasterOpts['windowSize'],
                            discardEmpty = rasterOpts['discardEmpty'])
                    except Exception:
                        traceback.print_exc()
                        pdb.set_trace()

                    spikeMatSetName = str(spikesFromIdx['chan']) + '_to_' + str(spikesToIdx['chan'])
                    grp = f.create_group(spikeMatSetName)
                    spikeMatSet = grp.create_dataset("spikeMats", (spikeMats[0].shape[0], spikeMats[0].shape[1], len(spikeMats)), dtype = 'f')
                    binCentersSet =  grp.create_dataset("columns", (spikeMats[0].shape[1],), data = spikeMats[0].columns, dtype = 'f')
                    allRowIdxSet = grp.create_dataset("index", (spikeMats[0].shape[0],), data = spikeMats[0].index, dtype = 'f')
                    categSet = grp.create_dataset("categories", (spikeMats[0].shape[0],), data = categories, dtype = 'f')
                    idxSet = grp.create_dataset("selectedIndices", (spikeMats[0].shape[0],), data = selectedIndices, dtype = 'f')

                    for idx, spikeMat in enumerate(spikeMats):
                        spikeMatSet[:,:,idx] = spikeMat

                    pairCount += 1

                    if saveSpikesFromIdx['chan'] == spikesFromIdx['chan'] and saveSpikesToIdx['chan'] == spikesToIdx['chan']:
                        saveSpikeMats, saveCategories, saveSelectedIndices = spikeMats, categories, selectedIndices
        # after looping through everything and saving, return the requested channel
        spikeMats, categories, selectedIndices = saveSpikeMats, saveCategories, saveSelectedIndices
    return spikeMats, categories, selectedIndices

def triggeredTimeSeries(alignTimes, dataDF, categories,
    whichColumns = None, removeBaseline = False,
    windowSize= [-0.25, 1], timeStampResolution = 3e4,
    units = 'uV', idxUnits = 'seconds', subSample = 1):

    if subSample:
        dataDF = dataDF.iloc[::subSample, :]
    # dataDF.index units must be consistent with windowSize!!
    if idxUnits == 'samples':
        dataDF.index = dataDF.index / 3e4

    if whichColumns is None:
        whichColumns = dataDF.columns

    # round WindowSize to the nearest samples
    windowSize = [m.ceil(i * timeStampResolution) / timeStampResolution for i in windowSize]
    nCh = len(whichColumns)
    windowIdx = [int(i * timeStampResolution) for i in windowSize]
    nSampsInWindow = len(range(windowIdx[0], windowIdx[1]))

    spikesTriggered = {
        'ChannelID' : [idx for idx, columnName in enumerate(whichColumns)],
        'ChannelLabel' : whichColumns,
        'Classification' : [categories.astype('category').cat.codes.values for i in whichColumns],
        'ClassificationLabel' : [categories.values for i in whichColumns],
        'NEUEVWAV_HeaderIndices' : [None for i in whichColumns],
        'TimeStamps' : [alignTimes.values for i in whichColumns],
        'Units' : units,
        'Waveforms' : [np.full((len(alignTimes.index), nSampsInWindow), np.nan
            ) for i in whichColumns],
        'basic_headers' : {'TimeStampResolution': timeStampResolution, 'windowSize' : windowSize},
        'extended_headers' : [],
        'name' : '',
        'info' : ''
        }

    startTimeIdxTriggeredChan = -windowIdx[0]
    for rowIdx, startTime in alignTimes.items():
        if pd.isnull(categories[rowIdx]):
            continue
        try:
            startTimeIdxChan = np.flatnonzero(dataDF.index > startTime)[0]
        except Exception:
            traceback.print_exc()
            continue
        maskPre = dataDF.index > (startTime + windowSize[0])
        maskPost = dataDF.index < (startTime + windowSize[1])
        mask = np.logical_and(maskPre, maskPost)
        rowIdxWave = alignTimes.index.get_loc(rowIdx)
        #print(rowIdxWave) #Debugging
        for idx, columnName in enumerate(whichColumns):
            try:
                chanSlice = dataDF.loc[mask, columnName]

                if removeBaseline:
                    chanSlice = chanSlice - chanSlice.mean()
                    #chanSlice = chanSlice - peakutils.baseline(chanSlice)

                idxIntoStart = np.flatnonzero(maskPre)[0] - startTimeIdxChan + startTimeIdxTriggeredChan

                idxMismatch = max(-idxIntoStart, 0)
                if idxMismatch > 0:
                    idxIntoStart += idxMismatch
                idxIntoEnd = np.flatnonzero(maskPost)[-1] - startTimeIdxChan + startTimeIdxTriggeredChan + 1 + idxMismatch

                if idxIntoEnd > nSampsInWindow:
                    idxIntoEnd = nSampsInWindow

                try:
                    spikesTriggered['Waveforms'][idx][rowIdxWave,idxIntoStart:idxIntoEnd] = chanSlice.values
                except Exception:
                    #traceback.print_exc()
                    spikesTriggered['Waveforms'][idx][rowIdxWave,idxIntoStart:idxIntoEnd] = chanSlice.values[idxIntoStart:idxIntoEnd]

            except Exception:
                pdb.set_trace()
                traceback.print_exc()

    return spikesTriggered

def triggeredNSxChanData(alignTimes, channelData,
    nevIDs, categories, removeBaseline = False,
    windowSize = [-0.25, 1]):

    dataDF  = pd.DataFrame(channelData['data'].values, index = channelData['t'],
        columns = channelData['data'].columns)

    spikesTriggered = triggeredTimeSeries(alignTimes, dataDF, categories,
        whichColumns = nevIDs, removeBaseline = removeBaseline,
        windowSize= windowSize, timeStampResolution = channelData['samp_per_s'])

    for idx, chIdx in enumerate(nevIDs):
        spikesTriggered['ChannelID'][idx] = chIdx
    return spikesTriggered

def spikeTriggeredTimeSeries(spikes, dataDF,
    whichColumns, spikesToIdx,
    timeStampResolution = 3e4, units = 'uV', idxUnits = 'seconds', subSample = 1,
    windowSize = [-0.25, 1], removeBaseline = False,
    separateByFun = hf.catSpikesGenerator(type = 'Classification'),
    timeRange = None, maxSpikesTo = None, discardEmpty = False):

    if timeRange is None:
        timeRange = (dataDF.index[0], dataDF.index[-1])
    # get spike firing times to align to
    alignTimes, categories, selectedIndices = hf.spikeAlignmentTimes(spikes,
        spikesToIdx,
        separateByFun = separateByFun,
        timeRange = timeRange, maxSpikesTo = maxSpikesTo, discardEmpty =  discardEmpty)

    spikesTriggered = triggeredTimeSeries(alignTimes, dataDF,
        categories, whichColumns, removeBaseline = removeBaseline,
        windowSize= windowSize, timeStampResolution = timeStampResolution, units = units, idxUnits = idxUnits, subSample = subSample)
    return spikesTriggered, selectedIndices

def getTrialCategories(trialStats, separateBy, validTrials = None):
    if validTrials is None:
        validTrials = np.full((trialStats.shape[0],), True)

    if separateBy is not None:
        if not isinstance(separateBy, list):
            categories = trialStats[separateBy].copy()
            validTrials = np.logical_and(validTrials, trialStats[separateBy].notnull())
        else:
            categories = pd.Series([[] for i in trialStats.index], index = trialStats.index)
            for idx, thisCriterion in enumerate(separateBy):
                validTrials = np.logical_and(validTrials, trialStats[thisCriterion].notnull())
                for rowIdx in trialStats.index:
                    categories[rowIdx].append(trialStats.loc[rowIdx,thisCriterion])

        categories[np.logical_not(validTrials)] = None
        # lists are mutable but unhashable, so convert to tuple once done
        if isinstance(separateBy, list):
            for rowIdx in trialStats.index:
                if categories[rowIdx] is not None:
                    categories[rowIdx] = tuple(categories[rowIdx])

        categories.loc[np.logical_not(validTrials)] = np.nan

        uniqueCategories = pd.Series(pd.unique(categories))

        uniqueCategories.dropna(inplace = True)
        #sort unique categories
        #pdb.set_trace()
        if not isinstance(uniqueCategories[0], tuple):
            uniqueCategories.sort_values(inplace = True)
        elif isinstance(uniqueCategories[0], tuple):
            for dim in range(len(uniqueCategories[0])):
                sortBy = pd.Series([value[dim] for value in uniqueCategories], index = uniqueCategories.index)
                sortIdx = sortBy.sort_values().index
                uniqueCategories = uniqueCategories[sortIdx]
        uniqueCategories.reset_index(drop=True, inplace=True)
    else:
        categories = pd.Series('all', index = trialStats.index)
        uniqueCategories = pd.Series(('all',))


    catIndices = uniqueCategories.astype('category').cat.codes
    return categories, uniqueCategories, catIndices, validTrials

def trialTriggeredTimeSeries(
    dataDF, trialStats, tsInfo):

    alignTimes = trialStats[tsInfo['alignTo']] / 3e4
    validMask = trialStats[tsInfo['alignTo']].notnull()

    if tsInfo['endOn'] is not None:
        #expand end of window to always catch the endOn event
        endTimes = trialStats[tsInfo['endOn']] - trialStats[tsInfo['alignTo']]
        tsInfo['windowSize'][1] = tsInfo['windowSize'][1] + endTimes.max() / 3e4 # samples to seconds conversion
        validMask = np.logical_and(validMask, trialStats[tsInfo['endOn']].notnull())

    categories, uniqueCategories, catIndices, validMask =  getTrialCategories(trialStats,
        tsInfo['separateBy'], validTrials = validMask)

    spikesTriggered = triggeredTimeSeries(alignTimes, dataDF, categories,
        whichColumns = tsInfo['whichColumns'], removeBaseline = tsInfo['removeBaseline'],
        windowSize = tsInfo['windowSize'],
        timeStampResolution = tsInfo['timeStampResolution'],
        units = tsInfo['units'], idxUnits = tsInfo['idxUnits'], subSample = tsInfo['subSample'])

    spikesTriggered['basic_headers'].update(tsInfo)
    return spikesTriggered

def spikeTriggeredNSxChanData(spikes, channelData,
    nevIDs, spikesToIdx,
    windowSize = [-0.25, 1], removeBaseline = False,
    separateByFun = hf.catSpikesGenerator(type = 'Classification'),
    timeRange = None, maxSpikesTo = None, discardEmpty = False):

    dataDF = pd.DataFrame(channelData['data'].values, index = channelData['t'],
        columns = channelData['data'].columns)

    spikesTriggered, selectedIndices = spikeTriggeredTimeSeries(spikes, dataDF,
        nevIDs, spikesToIdx,
        timeStampResolution = channelData['samp_per_s'],
        windowSize = windowSize, removeBaseline = removeBaseline,
        separateByFun = separateByFun,
        timeRange = timeRange, maxSpikesTo = maxSpikesTo, discardEmpty = discardEmpty)

    for idx, chIdx in enumerate(nevIDs):
        spikesTriggered['ChannelID'][idx] = chIdx
    return spikesTriggered, selectedIndices

def loadSpikeTriggeredTimeSeries():
    pass

def trialTriggeredTimeSeriesNameGenerator(tsInfo):
    seriesName = tsInfo['seriesName'] + '_' + tsInfo['recordName'] + '_ALIGNEDTO_' + tsInfo['alignTo']
    if tsInfo['separateBy'] is not None:
        seriesName = seriesName + '_SEPARATEDBY_' + '{}'.format(tsInfo['separateBy'])
    if tsInfo['endOn'] is not None:
        seriesName = seriesName + '_ENDON_' + tsInfo['endOn']
    return seriesName

def trialTriggeredTimeSeriesReportNameGenerator(tsInfo):
    seriesName = tsInfo['seriesName'] + '_ALIGNEDTO_' + tsInfo['alignTo']
    if tsInfo['separateBy'] is not None:
        seriesName = seriesName + '_SEPARATEDBY_' + '{}'.format(tsInfo['separateBy'])
    if tsInfo['endOn'] is not None:
        seriesName = seriesName + '_ENDON_' + tsInfo['endOn']
    return seriesName

def loadTrialTriggeredTimeSeries(folderPath, tsInfo,
    dataDF = None, trialStats = None, eventInfo = None,
    forceRecalc = False):

    setName = trialTriggeredTimeSeriesNameGenerator(tsInfo)
    setPath = os.path.join(folderPath, setName + '.pickle')
    if not forceRecalc:
        try:
            spikesTriggered = pickle.load(open(setPath, 'rb'))
            # make sure the metadata matches
            for key, value in tsInfo.items():
                if key in spikesTriggered['basic_headers']:
                    if value != spikesTriggered['basic_headers'][key]:
                        raise Exception('Parameter {} was requested to be {} but was {}'.format(key, value, spikesTriggered['basic_headers'][key]))
        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('Triggered Time Series data not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        if dataDF is None:
            dataDFPath = os.path.join(folderPath, tsInfo['seriesName'] + '.h5')
            dataDF = pd.read_hdf(dataDFPath, tsInfo['recordName'])
        if trialStats is None:
            trialStats, trialEvents, motorData = loadEventInfo(folderPath, eventInfo)

        spikesTriggered = trialTriggeredTimeSeries(
            dataDF, trialStats, tsInfo)

        pickle.dump(spikesTriggered, open(setPath, 'wb'))
    return spikesTriggered

def loadTrialBinnedSpike(folderPath,
    arrayName, arrayInfo,
    channel,
    rasterOpts,
    trialStats = None, spikes = None,
    correctAlignmentSpikes = 0,
    forceRecalc = False):

    setName = trialBinnedSpikesNameGenerator(arrayName, arrayInfo, rasterOpts)
    setPath = os.path.join(folderPath, setName + '.h5')
    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            with h5py.File(setPath, "r") as f:
                recordAttributes = f['/'+"rasterOpts"].attrs
                for key, value in rasterOpts.items():
                    if type(value) is not dict:
                        thisAttr = recordAttributes[key]

                        if isinstance(value, collections.Iterable):
                            for idx, valueComponent in enumerate(value):
                                assert (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx]))
                        else:
                            assert (value == thisAttr) or (value is None and np.isnan(thisAttr))
                    else:
                        for subKey, subValue in value.items():
                            thisAttr = recordAttributes[key + '_' + subKey]

                            if isinstance(subValue, collections.Iterable):
                                for idx, valueComponent in enumerate(subValue):
                                    assert (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx]))
                            else:
                                assert (subValue == thisAttr) or (subValue is None and np.isnan(thisAttr))

                requestedSpikeMat = '/' + str(channel)

                spikeMatShape = f[requestedSpikeMat + '/spikeMats'].shape
                spikeMats = [f[requestedSpikeMat + '/spikeMats'][:,:,i] for i in range(spikeMatShape[2])]
                for idx, spikeMat in enumerate(spikeMats):
                    spikeMats[idx] = pd.DataFrame(spikeMat, index = f[requestedSpikeMat + '/index'], columns = f[requestedSpikeMat + '/columns'])

                categories = np.array(f[requestedSpikeMat + '/categories'])
                categories = pd.Series(categories, index = f[requestedSpikeMat + '/index'])

                selectedIndices = np.array(f[requestedSpikeMat + '/selectedIndices'])

                if selectedIndices.any():
                    selectedIndices = pd.Series(selectedIndices, index = f[requestedSpikeMat + '/index'])
                else:
                    selectedIndices = None

            #print('Loaded spikeMats from h5.')

        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('SpikeMats not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        if spikes is None:
            spikeStruct, spikes = loadSpikeInfo(folderPath, arrayName, arrayInfo)

        if correctAlignmentSpikes: #correctAlignmentSpikes units in samples
            spikes= hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

        with h5py.File(setPath, "w") as f:
            grp = f.create_group("rasterOpts")
            for key, value in rasterOpts.items():
                if type(value) is not dict:
                    if value is not None:
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = np.nan
                else:
                    for subKey, subValue in value.items():
                        if value is not None:
                            grp.attrs[key + '_' + subKey] = subValue
                        else:
                            grp.attrs[key + '_' + subKey] = np.nan

            saveChanIdx = channel
            saveSpikeMats, saveCategories, saveSelectedIndices = None, None, None

            nCh = len(spikes['ChannelID'])
            for idx, chanIdx in enumerate(spikes['ChannelID']):
                unitsOnThisChan = np.unique(spikes['Classification'][idx])

                if unitsOnThisChan.any():
                    spikeMats, categories, selectedIndices = hf.binnedSpikesAlignedToTrial(spikes,
                        rasterOpts['binInterval'], rasterOpts['binWidth'],
                        trialStats, rasterOpts['alignTo'], chanIdx,
                        separateBy = rasterOpts['separateBy'],
                        windowSize = rasterOpts['windowSize'], timeRange = rasterOpts['timeRange'],
                        maxTrial = rasterOpts['maxTrial'])

                    if os.fstat(0) == os.fstat(1):
                        endChar = '\r'
                        print("Running loadTrialBinnedSpike: %d%%" % int((idx + 1) * 100 / nCh), end = endChar)
                    else:
                        print("Running loadTrialBinnedSpike: %d%%" % int((idx + 1) * 100 / nCh))

                    spikeMatSetName = str(chanIdx)
                    grp = f.create_group(spikeMatSetName)

                    spikeMatSet = grp.create_dataset("spikeMats", (spikeMats[0].shape[0], spikeMats[0].shape[1], len(spikeMats)), dtype = 'f')
                    binCentersSet =  grp.create_dataset("columns", (spikeMats[0].shape[1],), data = spikeMats[0].columns, dtype = 'f')
                    allRowIdxSet = grp.create_dataset("index", (spikeMats[0].shape[0],), data = spikeMats[0].index, dtype = 'f')
                    dt = h5py.special_dtype(vlen=str)
                    categSet = grp.create_dataset("categories", (spikeMats[0].shape[0],), data = categories, dtype=dt)
                    idxSet = grp.create_dataset("selectedIndices", (spikeMats[0].shape[0],), data = selectedIndices, dtype = 'f')

                    for idx, spikeMat in enumerate(spikeMats):
                        spikeMatSet[:,:,idx] = spikeMat

                    if saveChanIdx == chanIdx:
                        saveSpikeMats, saveCategories, saveSelectedIndices = spikeMats, categories, selectedIndices
        # after looping through everything and saving, return the requested channel
        spikeMats, categories, selectedIndices = saveSpikeMats, saveCategories, saveSelectedIndices
    return spikeMats, categories, selectedIndices

def loadSpikeBinnedArray(folderPath,
    arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo,
    spikesToIdx,
    rasterOpts,
    spikesFrom = None, spikesTo = None,
    correctAlignmentSpikesFrom = 0, correctAlignmentSpikesTo = 0,
    forceRecalc = False):

    setName = spikeBinnedArrayNameGenerator(arrayNameFrom, arrayInfoFrom, arrayNameTo, arrayInfoTo)
    setPath = os.path.join(folderPath, setName + '.h5')

    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            with h5py.File(setPath, "r") as f:
                recordAttributes = f['/'+"rasterOpts"].attrs
                for key, value in rasterOpts.items():
                    if type(value) is not dict:
                        thisAttr = recordAttributes[key]

                        if isinstance(value, collections.Iterable):
                            for idx, valueComponent in enumerate(value):
                                assert (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx]))
                        else:
                            assert (value == thisAttr) or (value is None and np.isnan(thisAttr))
                    else:
                        for subKey, subValue in value.items():
                            thisAttr = recordAttributes[key + '_' + subKey]

                            if isinstance(subValue, collections.Iterable):
                                for idx, valueComponent in enumerate(subValue):
                                    assert (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx]))
                            else:
                                assert (subValue == thisAttr) or (subValue is None and np.isnan(thisAttr))

                #spikeMats, categories, selectedIndices = None, None, None
                requestedRecord = str(spikesToIdx['chan'])

                categories = np.array(f[requestedRecord + '/categories'])
                categories = pd.Series(categories, index = f[requestedRecord + '/index'])

                spikeMats = {i:None for i in categories.index}

                for trialNum in categories.index:
                    try:
                        requestedSpikeMat = '/' + str(trialNum)
                        if len(f[requestedRecord + requestedSpikeMat + '/spikeMat'].shape) > 1:
                            spikeMats[trialNum] = pd.DataFrame(f[requestedRecord + requestedSpikeMat + '/spikeMat'][:,:],
                                index = f[requestedRecord + requestedSpikeMat + '/index'],
                                columns = f[requestedRecord + requestedSpikeMat + '/columns'])
                    except Exception:
                        traceback.print_exc()
                        pdb.set_trace()

                selectedIndices = np.array(f[requestedRecord + '/selectedIndices'])

                if selectedIndices.any():
                    selectedIndices = pd.Series(selectedIndices, index = f[requestedSpikeMat + '/index'])
                else:
                    selectedIndices = None
            # TODO: figure out how to load it...
            #print('Loaded spikeMats from h5.')
        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('SpikeMats not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:

        if spikesTo is None:
            spikeStructTo, spikesTo = loadSpikeInfo(folderPath, arrayNameTo, arrayInfoTo)

        if correctAlignmentSpikesTo: #correctAlignmentSpikesFrom units in samples
            spikesTo = hf.correctSpikeAlignment(spikesTo, correctAlignmentSpikesTo)

        spikesToList = []
        for idx, channel in enumerate(spikesTo['ChannelID']):
            unitsOnThisChan = np.unique(spikesTo['Classification'][idx])
            if unitsOnThisChan.any():
                spikesToList.append({'chan':channel,'units':list(range(len(unitsOnThisChan)))})

        with h5py.File(setPath, "w") as f:
            grp = f.create_group("rasterOpts")
            for key, value in rasterOpts.items():
                if type(value) is not dict:
                    if value is not None:
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = np.nan
                else:
                    for subKey, subValue in value.items():
                        if value is not None:
                            grp.attrs[key + '_' + subKey] = subValue
                        else:
                            grp.attrs[key + '_' + subKey] = np.nan

            if rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is not None:
                separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'], **rasterOpts['separateByFunKWArgs'])
            elif rasterOpts['separateByFunArgs'] is not None and rasterOpts['separateByFunKWArgs'] is None:
                separateByFun = hf.catSpikesGenerator(*rasterOpts['separateByFunArgs'])
            elif rasterOpts['separateByFunArgs'] is None and rasterOpts['separateByFunKWArgs'] is not None:
                separateByFun = hf.catSpikesGenerator(**rasterOpts['separateByFunKWArgs'])

            saveSpikesToIdx = spikesToIdx
            saveSpikeMats, saveCategories, saveSelectedIndices = None, None, None

            nCh = len(spikesToList)
            iterCount = 0
            for spikesToIdx in spikesToList:
                if os.fstat(0) == os.fstat(1):
                    endChar = '\r'
                    print("Running loadSpikeBinnedArray: %d%%" % int((iterCount + 1) * 100 / nCh), end = endChar)
                else:
                    print("Running loadSpikeBinnedArray: %d%%" % int((iterCount + 1) * 100 / nCh))

                spikesToSetName = str(spikesToIdx['chan'])
                grp = f.create_group(spikesToSetName)

                alignTimes, categories, selectedIndices = hf.spikeAlignmentTimes(spikesTo, spikesToIdx,
                    separateByFun = separateByFun,
                    timeRange = rasterOpts['timeRange'],
                    maxSpikesTo =rasterOpts['maxSpikesTo'], discardEmpty = rasterOpts['discardEmpty'])

                try:
                    spikeMats = {i:None for i in categories.index}
                    spikeMats.update(hf.binnedArray(spikesFrom, rasterOpts, alignTimes))
                except Exception:
                    traceback.print_exc()
                    pdb.set_trace()

                for idx, spikeMat in spikeMats.items():
                    spikeMatSetName = str(idx)
                    spikeMatGrp = grp.create_group(spikeMatSetName)
                    if spikeMat is not None:
                        spikeMatSet = spikeMatGrp.create_dataset("spikeMat", data = spikeMat.values)
                        binCentersSet =  spikeMatGrp.create_dataset("columns", data = spikeMat.columns)
                        allRowIdxSet = spikeMatGrp.create_dataset("index", data  = spikeMat.index)
                    else:
                        spikeMatSet = spikeMatGrp.create_dataset("spikeMat", data = [np.nan])
                        binCentersSet =  spikeMatGrp.create_dataset("columns", data = [np.nan])
                        allRowIdxSet = spikeMatGrp.create_dataset("index", data = [np.nan])

                categSet = grp.create_dataset("categories", data = categories, dtype = 'f')
                rowIdxSet = grp.create_dataset("index", data = categories.index, dtype = 'f')
                idxSet = grp.create_dataset("selectedIndices", data = selectedIndices, dtype = 'f')

                iterCount += 1

                if saveSpikesToIdx['chan'] == spikesToIdx['chan']:
                    saveSpikeMats, saveCategories, saveSelectedIndices = spikeMats, categories, selectedIndices
        # after looping through everything and saving, return the requested channel
        spikeMats, categories, selectedIndices = saveSpikeMats, saveCategories, saveSelectedIndices
    return spikeMats, categories, selectedIndices

def loadTrialBinnedArray(folderPath,
    arrayName, arrayInfo,
    eventInfo, rasterOpts,
    whichTrial  = None,
    spikes = None,
    trialStats = None, chans = None,
    correctAlignmentSpikes = 0,
    forceRecalc = False):

    setName = trialBinnedArrayNameGenerator(arrayName, arrayInfo, rasterOpts)
    setPath = os.path.join(folderPath, setName + '.h5')
    if not forceRecalc:
    # if not requiring a recalculation, load from pickle
        try:
            with h5py.File(setPath, "r") as f:
                recordAttributes = f['/'+"rasterOpts"].attrs
                for key, value in rasterOpts.items():
                    #value can be dict, iterable or single value
                    if type(value) is dict:
                        for subKey, subValue in value.items():
                            thisAttr = recordAttributes[key + '_' + subKey]
                            if isinstance(subValue, collections.Iterable):
                                for idx, valueComponent in enumerate(subValue):
                                    if not ( (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx])) ):
                                        raise Exception('Parameter {} was requested to be {} but was {}'.format(subKey, subValue, thisAttr))
                            else:
                                if not ( (subValue == thisAttr) or (subValue is None and np.isnan(thisAttr).all()) ):
                                    raise Exception('Parameter {} was requested to be {} but was {}'.format(subKey, subValue, thisAttr))
                    elif isinstance(value, collections.Iterable):
                        thisAttr = recordAttributes[key]
                        if  key not in ['separateBy']:
                            for idx, valueComponent in enumerate(value):
                                if not ( (valueComponent == thisAttr[idx]) or (valueComponent is None and np.isnan(thisAttr[idx])) ):
                                    raise Exception('Parameter {} was requested to be {} but was {}'.format(key, value, thisAttr))
                    else:
                        thisAttr = recordAttributes[key]
                        if not ( (value == thisAttr) or (value is None and np.isnan(thisAttr))):
                            raise Exception('Parameter {} was requested to be {} but was {}'.format(key, value, thisAttr))

                if whichTrial is None:
                    whichTrial = [int(i) for i in f if i != 'rasterOpts']

                spikeMats = {i:None for i in whichTrial}

                for trialNum in whichTrial:
                    try:
                        requestedSpikeMat = '/' + str(trialNum)
                        if len(f[requestedSpikeMat + '/spikeMat'].shape) > 1:
                            spikeMats[trialNum] = pd.DataFrame(f[requestedSpikeMat + '/spikeMat'][:,:],
                                index = f[requestedSpikeMat + '/index'],
                                columns = f[requestedSpikeMat + '/columns'])
                    except Exception:
                        traceback.print_exc()
                        pdb.set_trace()

        except Exception:
            traceback.print_exc()
            # if loading failed, recalculate anyway
            print('SpikeMats not pickled. Recalculating...')
            forceRecalc = True

    if forceRecalc:
        if spikes is None:
            spikeStruct, spikes = ksa.loadSpikeInfo(folderPath, 'utah', trialFiles['utah'])
        if trialStats is None:
            trialStats, trialEvents, motorData = loadEventInfo(folderPath, eventInfo)
        if correctAlignmentSpikes: #correctAlignmentSpikes units in samples
            spikes= hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)
        if whichTrial is None:
            whichTrial = trialStats.index

        with h5py.File(setPath, "w") as f:
            grp = f.create_group("rasterOpts")
            for key, value in rasterOpts.items():
                #value can be dict, iterable or single value
                if type(value) is dict:
                    for subKey, subValue in value.items():
                        if subValue is not None:
                            grp.attrs[key + '_' + subKey] = subValue
                        else:
                            grp.attrs[key + '_' + subKey] = np.nan
                else:
                    if value is not None and key not in ['separateBy']:
                        try: grp.attrs[key] = value
                        except Exception: traceback.print_exc(); pdb.set_trace()
                    else:
                        grp.attrs[key] = np.nan

            spikeMats = {i:None for i in trialStats.index}
            spikeMats.update(hf.trialBinnedArray(spikes, rasterOpts, trialStats, chans = None))
            #pdb.set_trace()
            saveSpikeMats = {i:None for i in whichTrial}
            for idx, spikeMat in spikeMats.items():
                spikeMatSetName = str(idx)
                grp = f.create_group(spikeMatSetName)
                if spikeMat is not None:
                    spikeMatSet = grp.create_dataset("spikeMat", data = spikeMat.values)
                    binCentersSet =  grp.create_dataset("columns", data = spikeMat.columns)
                    allRowIdxSet = grp.create_dataset("index", data  = spikeMat.index)
                else:
                    spikeMatSet = grp.create_dataset("spikeMat", data = [np.nan])
                    binCentersSet =  grp.create_dataset("columns", data = [np.nan])
                    allRowIdxSet = grp.create_dataset("index", data = [np.nan])

                if idx in whichTrial:
                    saveSpikeMats[idx]= spikeMat
            #after looping through everything and saving, return the requested channel
            spikeMats = saveSpikeMats
    return spikeMats

def generateSpikeReport(folderPath, eventInfo, trialFiles,
    rasterOpts = None, plotOpts = None,
    correctAlignmentSpikes = 0):

    """
    Read in Trial events
    """
    trialStats, trialEvents, motorData = loadEventInfo(folderPath, eventInfo)

    for key, value in trialFiles.items():

        """
        Read in array spikes
        """
        spikeStruct, spikes = loadSpikeInfo(folderPath, key, value)

        newName = spikesNameGenerator(key, value)
        spikePDFReport(folderPath,
            spikes, spikeStruct,
            arrayName = key, arrayInfo = value,
            correctAlignmentSpikes = correctAlignmentSpikes,
            rasterOpts = rasterOpts, plotOpts = plotOpts,
            trialStats = trialStats,
            enableFR = True, newName = newName)

        del spikes, spikeStruct

if __name__ == "__main__":
    pass
