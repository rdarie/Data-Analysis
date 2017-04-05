from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brpylib             import NevFile, brpylib_ver
import sys
import pickle
# Plotting options
font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 20
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

ns5Dir = 'E:/Google Drive/Github/tempdata/Data-Analysis/'

ns5Name = 'saveRight.p'
ns5File = ns5Dir + ns5Name
data = pd.read_pickle(ns5File)

simiName = 'saveSimi.p'
simiFile = ns5Dir + simiName
simiData = pd.read_pickle(simiFile)
simiDf = simiData['simiGait']
gaitLabelFun = simiData['gaitLabelFun']
upLabelFun = simiData['upLabelFun']
downLabelFun = simiData['downLabelFun']

spikeName = 'saveSpikeRight.p'
spikeFile = ns5Dir + spikeName
spikeData = pd.read_pickle(spikeFile)
spikes = spikeData['spikes']
binCenters = spikeData['binCenters']
spikeMat = spikeData['spikeMat']
binWidth = spikeData['binWidth']

labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}

tempUpLabels = assignLabels(binCenters, 'Toe Up', upLabelFun)
tempDownLabels = assignLabels(binCenters, 'Toe Down', downLabelFun)
spikeMat['Labels'] = pd.Series([x if x == 'Toe Up' else y for x,y in zip(tempUpLabels, tempDownLabels)])
spikeMat['LabelsNumeric'] = pd.Series([labelsNumeric[x] for x in spikeMat['Labels']])

tempUpLabels = assignLabels(data['channel']['t'], 'Toe Up', upLabelFun)
tempDownLabels = assignLabels(data['channel']['t'], 'Toe Down', downLabelFun)
binnedLabels = pd.Series([x if x == 'Toe Up' else y for x,y in zip(tempUpLabels, tempDownLabels)])
data['channel']['data']['Labels'] = binnedLabels
data['channel']['data']['LabelsNumeric'] = pd.Series([labelsNumeric[x] for x in data['channel']['data']['Labels']])

tempUpLabels = assignLabels(data['channel']['spectrum']['t'], 'Toe Up', upLabelFun)
tempDownLabels = assignLabels(data['channel']['spectrum']['t'], 'Toe Down', downLabelFun)
binnedLabels = pd.Series([x if x == 'Toe Up' else y for x,y in zip(tempUpLabels, tempDownLabels)])
data['channel']['spectrum']['Labels'] = binnedLabels
data['channel']['spectrum']['LabelsNumeric'] = pd.Series([labelsNumeric[x] for x in data['channel']['spectrum']['Labels']])
with open(ns5Dir + "saveRightLabeled.p", "wb" ) as f:
    pickle.dump(data, f, protocol=4 )
with open(ns5Dir + "saveSpikeRightLabeled.p", "wb" ) as f:
    pickle.dump(spikeData, f, protocol=4 )

plotting = True
if plotting:
    #Plot the spikes
    chans = spikes['ChannelID']
    fi = plotBinnedSpikes(spikeMat.drop(['Labels', 'LabelsNumeric'], axis = 1), binCenters, chans, show = False)

    upMaskSpikes = (spikeMat['Labels'] == 'Toe Up').values
    downMaskSpikes = (spikeMat['Labels'] == 'Toe Down').values
    dummyVar = np.ones(binCenters.shape[0]) * 1
    ax = fi.axes[0]
    ax.plot(binCenters[upMaskSpikes], dummyVar[upMaskSpikes], 'ro')
    ax.plot(binCenters[downMaskSpikes], dummyVar[downMaskSpikes] + 1, 'go')
    plt.show(block = False)

    #plot the spectrum
    upMaskSpectrum = (data['channel']['spectrum']['Labels'] == 'Toe Up').values
    downMaskSpectrum = (data['channel']['spectrum']['Labels'] == 'Toe Down').values
    dummyVar = np.ones(data['channel']['spectrum']['t'].shape[0]) * 1

    fi = plotSpectrum(data['channel']['spectrum']['PSD'][1],
        data['channel']['samp_per_s'],
        data['channel']['start_time_s'],
        data['channel']['t'][-1],
        fr = data['channel']['spectrum']['fr'],
        t = data['channel']['spectrum']['t'],
        show = False)
    ax = fi.axes[0]
    ax.plot(data['channel']['spectrum']['t'][upMaskSpectrum], dummyVar[upMaskSpectrum], 'ro')
    ax.plot(data['channel']['spectrum']['t'][downMaskSpectrum], dummyVar[downMaskSpectrum] + 1, 'go')
    #ax.plot(data['channel']['spectrum']['t'][stanceMask], dummyVar[stanceMask], 'go')
    plt.show(block = False)

    # plot one channel
    f,ax = plotChan(data['channel'], 25, mask = None, show = False)
    dummyVar = np.ones(data['channel']['t'].shape[0]) * 100
    upMaskChan = (data['channel']['data']['Labels'] == 'Toe Up').values
    downMaskChan = (data['channel']['data']['Labels'] == 'Toe Down').values
    ax.plot(data['channel']['t'][downMaskChan], dummyVar[downMaskChan], 'go')
    ax.plot(data['channel']['t'][upMaskChan], dummyVar[upMaskChan], 'ro')
    plt.show()
