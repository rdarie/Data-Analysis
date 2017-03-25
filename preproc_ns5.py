# -*- coding: utf-8 -*-
"""
Example of how to extract and plot continuous data saved in Blackrock nsX data files
current version: 1.1.1 --- 07/22/2016

@author: Mitch Frankel - Blackrock Microsystems
"""

"""
Version History:
v1.0.0 - 07/05/2016 - initial release - requires brpylib v1.0.0 or higher
v1.1.0 - 07/12/2016 - addition of version checking for brpylib starting with v1.2.0
                      minor code cleanup for readability
v1.1.1 - 07/22/2016 - now uses 'samp_per_sec' as returned by NsxFile.getdata()
                      minor modifications to use close() functionality of NsxFile class
"""
import matplotlib, math
from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import libtfr
import sys

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
fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/Right_Array/';
fileName = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5';

datafile = fileDir + fileName

elec_ids     = range(1,97)  # 'all' is default for all (1-indexed)
start_time_s = 40                      # 0 is default for all
data_time_s  = 3                        # 'all' is default for all
whichChan    = 2                       # 1-indexed

cont_data, _, extended_headers = getNSxData(datafile, elec_ids, start_time_s, data_time_s)
badData = getBadDataMask(cont_data, extended_headers, plotting = False, smoothing_ms = 5)

plot_chan(cont_data, extended_headers, whichChan, mask = badData, show = False)

# mask bad data with nans
# to do: make the data natively be a pandas dataframe
for arr in cont_data['data']:
    arr[badData] = float('nan')
    arrDf = pd.Series(arr)
    arrDf.interpolate(method = 'linear', inplace = True)
    arr[:] = arrDf

plot_chan(cont_data, extended_headers, whichChan, mask = badData, show = True)
winLen = 1000
#for idx in range(len(signal) - winLen, winLen):
    # load signal of dimension (npoints,)

Fs = cont_data['samp_per_s']
nSamples = len(cont_data['data'][0])
t = np.arange(nSamples)
nChan = len(cont_data['data'])
delta = 1 / Fs

# function parameters
winLen_s = 0.1
stepLen_fr = 0.25 # window step as a fraction of window length
R = 50 # target bandwidth for spectrogram

spectrum = get_spectrogram(cont_data, winLen_s, stepLen_fr, R, whichChan)
