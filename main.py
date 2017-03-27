import matplotlib, math
from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import libtfr
import sys
import pickle
import peakutils
from scipy.signal import *

font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 20
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rcParams.keys()

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

# Inits
fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/';
fileName = 'Python/save.p'
simiName = 'Trial01_Step_Timing.txt'

dataFile = fileDir + fileName
simiFile = fileDir + simiName

# Read in simi text file
simiTable = pd.read_table(simiFile)
simiTable.drop(simiTable.index[[0,1,2]], inplace = True) # first rows contain giberrish
data = pd.read_pickle(dataFile)

# sample rate
fs = data['simiTrigger']['samp_per_s']
# get camera triggers
simiTriggers = data['simiTrigger']['data']
iti = .01 # inter trigger interval
width = fs * iti / 2 # minimum distance between triggers
simiTriggersPrime = simiTriggers.diff()
simiTriggersPrime.fillna(0, inplace = True)
# moments when camera capture occured
peakIdx = peakutils.indexes((-1) * simiTriggersPrime.values.squeeze(), thres=0.7, min_dist=width)

plt.plot(data['simiTrigger']['t'], simiTriggers.values)
plt.plot(data['simiTrigger']['t'][peakIdx], simiTriggers.values[peakIdx], 'r*')
ax = plt.gca()
ax.set_xlim([5.2, 5.6])
plt.show()

# get time of first simi frame in NSP time:
trigTimes = data['simiTrigger']['t'][peakIdx]
timeOffset = trigTimes[0]
timeMax = data['simiTrigger']['t'].max()

simiDf = pd.DataFrame(simiTable[['ToeUp_Left Y', 'ToeDown_Left Y']])
simiDf = simiDf.notnull()
simiDf['simiTime'] = simiTable['Time'] + timeOffset
simiDf.drop(simiDf[simiDf['simiTime'] >= timeMax].index, inplace = True)

simiDf['NSPTime'] = pd.Series(trigTimes, index = simiDf.index)

down = (simiDf['ToeDown_Left Y'].values * 1)
up = (simiDf['ToeUp_Left Y'].values * 1)
gait = up.cumsum() - down.cumsum()

plt.plot(simiDf['simiTime'], gait)
plt.plot(simiDf['simiTime'], down, 'g*')
plt.plot(simiDf['simiTime'], up, 'r*')
ax = plt.gca()
ax.set_ylim([-1.1, 1.1])
ax.set_xlim([6.8, 7.8])
plt.show()

# TODO: interpolate labels onto neural data
simiDf['Labels'] = pd.Series(['Swing' if x == 1 else 'Stance' for x in gait], index = simiDf.index)
data['channel']['data']['NSPTime'] = data['channel']['t']
data['channel']['data']['Labels'] = np.nan

data['channel']['data'].loc[peakIdx,'Labels'] = simiDf['Labels'] #unsure if this is producing what I want it to..
data['channel']['data']['Labels'].fillna(method = 'ffill', inplace = True)
data['channel']['data']['Labels'].fillna(method = 'bfill', inplace = True)
data['channel']['data'][data['channel']['data']['NSPTime']> 6.9]
