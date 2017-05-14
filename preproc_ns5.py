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
import sys
import pickle
from copy import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stepLen', default = '0.1')
parser.add_argument('--winLen', default = '0.1')
parser.add_argument('--file', default = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5')
args = parser.parse_args()
argWinLen = float(args.winLen)
argStepLen = float(args.stepLen)
argFile = args.file


print("Preprocessing spectral data with a window length of {:4.4f} seconds and a step length of {:4.4f} seconds".format(argWinLen, argStepLen))
# Reformat figures
font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 20
        }
fig_opts = {
    'figsize' : (10,5),
    }
matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

# Inits
localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
fileName = '/' + argFile

datafile = localDir + fileName

elec_ids = range(1,97) # 'all' is default for all (1-indexed)
start_time_s = 0 # 0 is default for all
data_time_s = 90 # 'all' is default for all
whichChan = 25 # 1-indexed

simi_triggers = getNSxData(datafile, 136, start_time_s, data_time_s)

ChannelData = getNSxData(datafile, elec_ids, start_time_s, data_time_s)

ch_idx  = ChannelData['elec_ids'].index(whichChan)

badData = getBadContinuousMask(ChannelData, plotting = whichChan, smoothing_ms = 0.5)

f,_ = plotChan(ChannelData, whichChan, mask = None, show = False)

clean_data = deepcopy(ChannelData)

# interpolate bad data
for idx, row in clean_data['data'].iteritems():
    mask = np.logical_or(badData['general'], badData['perChannel'][idx])
    row = replaceBad(row, mask, typeOpt = 'interp')

clean_data['badData'] = badData
# check interpolation results
plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
plotChan(clean_data, whichChan, mask = plot_mask, show = True, prevFig = f)

# spectrum function parameters
winLen_s = argWinLen
stepLen_s = argStepLen # window step as a fraction of window length
R = 30 # target bandwidth for spectrogram

# get the spectrum
clean_data['spectrum'] = getSpectrogram(
    clean_data, winLen_s, stepLen_s, R, 600, whichChan, plotting = True)

data = {'channel':clean_data, 'simiTrigger': simi_triggers,
    'origin' : clean_data['spectrum']['origin'],
    'winLen' : argWinLen, 'stepLen' : argStepLen}

with open(localDir + '/' + argFile.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
    pickle.dump(data, f, protocol=4 )

#print('Starting to write PDF Report.')

#pdfFile = localDir + 'pdfReport.pdf'
#pdfReport(ChannelData, clean_data, badData = badData, pdfFilePath = pdfFile, spectrum = True)

#x = input("Press any key")
