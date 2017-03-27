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
        'size'   : 30
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rcParams.keys()

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

# Inits
fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/';
fileName = 'Right_Array/Python/save.p'
simiName = 'Trial01_Step_Timing.txt'

dataFile = fileDir + fileName
simiFile = fileDir + simiName

simiDf = pd.read_table(simiFile)
data = pd.read_pickle(dataFile)

# sample rate
fs = data['simiTrigger']['samp_per_s']
# expected interval between triggers
rates = np.arange(50, 150, 10) * 1e-3
#find peaks
widths = fs/rates
width = fs / 200
simiPrime = data['simiTrigger']['data'].diff()
simiPrime.fillna(0, inplace = True)
simiPrime.values.squeeze().shape

#peakIdx = argrelmax(simiPrime.values.squeeze(), order = int(widths[0]))
peakIdx = peakutils.indexes(simiPrime.values.squeeze(), thres=0.7, min_dist=width)
#peakIdx = find_peaks_cwt(simiPrime.values.squeeze(), widths)
peakIdx.shape
simiDf['Time'].max()
t = data['channel']['start_time_s'] + np.arange(data['channel']['data'].shape[0]) / data['channel']['samp_per_s']
plt.plot(t, simiPrime.values)
plt.plot(t[peakIdx], simiPrime.values[peakIdx], 'r*')

ax = plt.gca()
ax.set_xlim([100, 100.3])
plt.show()
