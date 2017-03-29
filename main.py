import matplotlib, math
from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import libtfr
import sys
import pickle
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
# Read in NSP data from preproc_ns5
data = pd.read_pickle(dataFile)

peakIdx, trigTimes = get_camera_triggers(data['simiTrigger'], plotting = True)

simiDf, gaitLabelFun, downLabelFun, upLabelFun = get_gait_events(trigTimes, simiTable, plotting = True)

data['channel']['data']['Labels'] = pd.Series(['Swing' if x == 1 else 'Stance' for x in upLabelFun(data['channel']['t'])], index = data['channel']['data'].index)
data['channel']['spectrum']['Labels'] = pd.Series(['Swing' if x == 1 else 'Stance' for x in upLabelFun(data['channel']['spectrum']['t'])])

swingMask = (data['channel']['spectrum']['Labels'] == 'Swing').values
stanceMask = np.logical_not(swingMask)
dummyVar = np.ones(data['channel']['spectrum']['t'].shape[0]) * 400

fi = plot_spectrum(data['channel']['spectrum']['PSD'][0,:,:], data['channel']['samp_per_s'], data['channel']['start_time_s'], data['channel']['t'][-1], show = False)
ax = fi.axes[0]
ax.plot(data['channel']['spectrum']['t'][swingMask], dummyVar[swingMask], 'ro')
#ax.plot(data['channel']['spectrum']['t'][stanceMask], dummyVar[stanceMask], 'go')
plt.show(block = False)

f,ax = plot_chan(data['channel'], 1, mask = None, show = False)
dummyVar = np.ones(data['channel']['t'].shape[0]) * 100
swingMask = (data['channel']['data']['Labels'] == 'Swing').values
ax.plot(data['channel']['t'][swingMask], dummyVar[swingMask], 'ro')
plt.show()
