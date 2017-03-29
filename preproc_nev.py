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
chans    = range(1,10)
#fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/'
fileDir = 'E:/Google Drive/Github/tempdata/Data-Analysis/'
fileName = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev';

datafile = fileDir + fileName
spikes = getNEVData(datafile, chans)

badMask = getBadSpikesMask(spikes, nStd = 5, whichChan = 0, plotting = False, deleteBad = True)

plt.show()
"""
plot_spikes(spikes, chans)

plot_raster(spikes, chans)

# Inits

ns5Dir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/'
ns5Name = 'Python/save.p'
ns5File = ns5Dir + ns5Name
data = pd.read_pickle(ns5File)

ns5Path = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/Right_Array/201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5'

nsx_file = NsxFile(ns5Path)

"""
x = input("Press any key")
