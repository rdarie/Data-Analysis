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

ns5Name = 'save.p'
ns5File = ns5Dir + ns5Name
data = pd.read_pickle(ns5File)

simiName = 'saveSimi.p'
simiFile = ns5Dir + simiName
simiData = pd.read_pickle(simiFile)
simiDf = simiData['simiGait']
gaitLabelFun = simiData['gaitLabelFun']
upLabelFun = simiData['upLabelFun']
downLabelFun = simiData['downLabelFun']

spikeName = 'saveSpike.p'
spikeFile = ns5Dir + spikeName
spikeData = pd.read_pickle(spikeFile)
spikes = spikeData['spikes']
binCenters = spikeData['binCenters']
spikeMat = spikeData['spikeMat']
binWidth = spikeData['binWidth']
#TODO: binned centers are only labeled as food event if the event binInterval spaces to its right. Should be BinWindow around.

chans = spikes['ChannelID']
fi = plotBinnedSpikes(spikeMat, binCenters, chans, show = False)
binnedLabels = assignLabels(binCenters, 'Toe Up', upLabelFun)
swingMask = (binnedLabels == 'Toe Up').values
dummyVar = np.ones(binCenters.shape[0]) * 1
ax = fi.axes[0]
ax.plot(binCenters[swingMask], dummyVar[swingMask], 'ro')
plt.show(block = False)

data['channel']['data']['Labels'] = assignLabels(data['channel']['t'], 'Toe Up', upLabelFun)
data['channel']['spectrum']['Labels'] = assignLabels(data['channel']['spectrum']['t'], 'Toe Up', upLabelFun)

swingMask = (data['channel']['spectrum']['Labels'] == 'Toe Up').values
stanceMask = np.logical_not(swingMask)
dummyVar = np.ones(data['channel']['spectrum']['t'].shape[0]) * 1

fi = plotSpectrum(data['channel']['spectrum']['PSD'][0,:,:],
    data['channel']['samp_per_s'],
    data['channel']['start_time_s'],
    data['channel']['t'][-1],
    fr = data['channel']['spectrum']['fr'],
    t = data['channel']['spectrum']['t'],
    show = False)
ax = fi.axes[0]
ax.plot(data['channel']['spectrum']['t'][swingMask], dummyVar[swingMask], 'ro')
#ax.plot(data['channel']['spectrum']['t'][stanceMask], dummyVar[stanceMask], 'go')
plt.show(block = False)

f,ax = plot_chan(data['channel'], 25, mask = None, show = False)
dummyVar = np.ones(data['channel']['t'].shape[0]) * 100
swingMask = (data['channel']['data']['Labels'] == 'Toe Up').values
ax.plot(data['channel']['t'][swingMask], dummyVar[swingMask], 'ro')
plt.show()
