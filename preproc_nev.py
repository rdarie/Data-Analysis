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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev')
parser.add_argument('--binInterval', default = '20e-3')
parser.add_argument('--binWidth', default = '100e-3')
args = parser.parse_args()
argFile = args.file
argBinInterval = args.binInterval
argBinWidth = args.binWidth

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
chans    = range(1,97)
localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
fileName = '/' + argFile
datafile = localDir + fileName
spikes = getNEVData(datafile, chans)

badMask = getBadSpikesMask(spikes, nStd = 5, whichChan = 0, plotting = False, deleteBad = True)

binInterval = float(argBinInterval)
binWidth = float(argBinWidth)
timeStart = 0
#pdb.set_trace()
timeDur = max([max(sp) for sp in spikes['TimeStamps']]) / spikes['basic_headers']['TimeStampResolution'] - timeStart

mat, binCenters, binLeftEdges = binnedSpikes(spikes, chans, binInterval, binWidth, timeStart, timeDur)

spikeData = {'spikes':spikes, 'spikeMat':mat, 'binCenters':binCenters, 'binLeftEdges': binLeftEdges, 'binWidth':binWidth}
with open( localDir + '/' + argFile.split('.')[0] + '_saveSpike.p', "wb" ) as f:
    pickle.dump(spikeData, f, protocol=4 )

#x = input("Press any key")
