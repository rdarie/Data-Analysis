# -*- coding: utf-8 -*-
"""
Example of how to extract and plot neural spike data saved in Blackrock nev data files
current version: 1.1.2 --- 08/04/2016

@author: Mitch Frankel - Blackrock Microsystems
"""

"""
Version History:
v1.0.0 - 07/05/2016 - initial release - requires brpylib v1.0.0 or higher
v1.1.0 - 07/12/2016 - addition of version checking for brpylib starting with v1.2.0
                      simplification of entire example for readability
                      plot loop now done based on unit extraction first
v1.1.1 - 07/22/2016 - minor modifications to use close() functionality of NevFile class
v1.1.2 - 08/04/2016 - minor modifications to allow use of Python 2.6+
"""

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

# Init
chans    = [25, 90]
#fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/'
fileDir = 'E:/Google Drive/Github/tempdata/Data-Analysis/'
fileName = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev';
datafile = fileDir + fileName
spikes = getNEVData(datafile, chans)

ns5Dir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/'
ns5Name = 'Python/save.p'
ns5File = ns5Dir + ns5Name
data = pd.read_pickle(ns5File)

simiName = 'Python/saveSimi.p'
simiFile = ns5Dir + simiName
simiData = pd.read_pickle(simiFile)

badMask = getBadSpikesMask(spikes, nStd = 5, whichChan = 0, plotting = False, deleteBad = True)
#plt.show()

#plot_spikes(spikes, chans)

#plot_raster(spikes, chans)

binInterval = 20e-3
binWidth = 150e-3
timeStart = 3
timeEnd = 33
mat, binCenters = binnedSpikes(spikes, chans, binInterval, binWidth, timeStart, timeEnd)

fi = plotBinnedSpikes(mat, binCenters, chans, show = False)
binnedLabels = assignLabels(binCenters, 'Toe Up', upLabelFun)
swingMask = (binnedLabels == 'Toe Up').values
dummyVar = np.ones(binCenters.shape[0]) * 2
ax = fi.axes[0]
ax.plot(data['channel']['spectrum']['t'][swingMask], dummyVar[swingMask], 'ro')
plt.show(block = False)

# Other plots
simiDf = simiData['simiGait']
gaitLabelFun = simiData['gaitLabelFun']
upLabelFun = simiData['upLabelFun']
downLabelFun = simiData['downLabelFun']

data['channel']['data']['Labels'] = assignLabels(data['channel']['t'], 'Toe Up', upLabelFun)
data['channel']['spectrum']['Labels'] = assignLabels(data['channel']['spectrum']['t'], 'Toe Up', upLabelFun)

swingMask = (data['channel']['spectrum']['Labels'] == 'Toe Up').values
stanceMask = np.logical_not(swingMask)
dummyVar = np.ones(data['channel']['spectrum']['t'].shape[0]) * 400

fi = plot_spectrum(data['channel']['spectrum']['PSD'][0,:,:], data['channel']['samp_per_s'], data['channel']['start_time_s'], data['channel']['t'][-1], show = False)
ax = fi.axes[0]
ax.plot(data['channel']['spectrum']['t'][swingMask], dummyVar[swingMask], 'ro')
#ax.plot(data['channel']['spectrum']['t'][stanceMask], dummyVar[stanceMask], 'go')
plt.show(block = False)

f,ax = plot_chan(data['channel'], 1, mask = None, show = False)
dummyVar = np.ones(data['channel']['t'].shape[0]) * 100
swingMask = (data['channel']['data']['Labels'] == 'Toe Up').values
ax.plot(data['channel']['t'][swingMask], dummyVar[swingMask], 'ro')
plt.show()


x = input("Press any key")
