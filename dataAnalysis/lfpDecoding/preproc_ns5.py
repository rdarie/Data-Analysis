# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""

import matplotlib, math
from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle
from copy import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stepLen', default = '0.05')
parser.add_argument('--winLen', default = '0.1')
parser.add_argument('--file', default = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5')
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc')

args = parser.parse_args()
argWinLen = float(args.winLen)
argStepLen = float(args.stepLen)
argFile = args.file
fileDir = args.folder

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
fileName = '/' + argFile

datafile = fileDir + fileName

elec_ids = range(1,97) # 'all' is default for all (1-indexed)
start_time_s = 120 # 0 is default for all
data_time_s = 180 # 'all' is default for all
whichChan = 25 # 1-indexed

simi_triggers = getNSxData(datafile, 104, start_time_s, data_time_s)

ChannelData = getNSxData(datafile, elec_ids, start_time_s, data_time_s)

ch_idx  = ChannelData['elec_ids'].index(whichChan)

badData = getBadContinuousMask(ChannelData, plotting = whichChan, smoothing_ms = 0.5)

f,_ = plotChan(ChannelData, whichChan, label = 'Raw data', mask = None, show = False)

clean_data = deepcopy(ChannelData)

# interpolate bad data
for idx, row in clean_data['data'].iteritems():
    mask = np.logical_or(badData['general'], badData['perChannel'][idx])
    row = replaceBad(row, mask, typeOpt = 'interp')

clean_data['badData'] = badData
# check interpolation results
plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
plotChan(clean_data, whichChan, label = 'Clean data', mask = plot_mask,
    maskLabel = "Dropout", show = True, prevFig = f)

plt.legend()
plt.savefig(fileDir + '/' + argFile.split('.')[0] + '_ns5Clean.png')
with open(fileDir + '/' + argFile.split('.')[0] + '_ns5Clean.pickle', 'wb') as File:
    pickle.dump(f, File)
# spectrum function parameters
winLen_s = argWinLen
stepLen_s = argStepLen # window step as a fraction of window length
R = 30 # target bandwidth for spectrogram

# get the spectrum
clean_data['spectrum'] = getSpectrogram(
    clean_data, winLen_s, stepLen_s, R, 600, whichChan, plotting = True)

plt.savefig(fileDir + '/' + argFile.split('.')[0] + '_SpectrumClean.png')
with open(fileDir + '/' + argFile.split('.')[0] + '_SpectrumClean.pickle', 'wb') as File:
    pickle.dump(plt.gcf(), File)

compareBad = True
if compareBad:
    # get the spectrum
    ChannelData['spectrum'] = getSpectrogram(
        ChannelData, winLen_s, stepLen_s, R, 600, whichChan, plotting = True)

    plt.savefig(fileDir + '/' + argFile.split('.')[0] + '_SpectrumRaw.png')
    with open(fileDir + '/' + argFile.split('.')[0] + '_SpectrumRaw.pickle', 'wb') as File:
        pickle.dump(plt.gcf(), File)


data = {'channel':clean_data, 'simiTrigger': simi_triggers,
    'origin' : clean_data['spectrum']['origin'],
    'winLen' : argWinLen, 'stepLen' : argStepLen}

with open(fileDir + '/' + argFile.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
    pickle.dump(data, f, protocol=4 )

#print('Starting to write PDF Report.')

#pdfFile = localDir + 'pdfReport.pdf'
#pdfReport(ChannelData, clean_data, badData = badData, pdfFilePath = pdfFile, spectrum = True)

#x = input("Press any key")
