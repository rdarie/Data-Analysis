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
simiTable.drop(simiTable.iloc[1], inplace = True) # first row contains giberrish
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

plt.plot(data['simiTrigger']['t'], simiTriggersPrime.values)
plt.plot(data['simiTrigger']['t'][peakIdx], simiTriggersPrime.values[peakIdx], 'r*')
ax = plt.gca()
ax.set_xlim([5.2, 5.6])
plt.show()

# get time of first simi frame in NSP time:
timeOffset = data['simiTrigger']['t'][peakIdx[0]]
simiDf = pd.DataFrame(simiTable[['ToeUp_Left Y', 'ToeDown_Left Y']], index = simiTable['Time'] + timeOffset)
simiDf = simiDf.notnull()
