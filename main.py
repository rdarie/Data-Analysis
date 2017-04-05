from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

localDir = 'E:/Google Drive/Github/tempdata/Data-Analysis/'
spikeName = 'saveSpikeRightLabeled.p'
spikeFile = localDir + spikeName
spikeData = pd.read_pickle(spikeFile)
spikes = spikeData['spikes']
binCenters = spikeData['binCenters']
spikeMat = spikeData['spikeMat']
binWidth = spikeData['binWidth']
