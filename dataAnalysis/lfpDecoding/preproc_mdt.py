import pandas as pd
import matplotlib, math
from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
from copy import *
import argparse, linecache

parser = argparse.ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--stepLen', default = 0.05, type = float)
parser.add_argument('--winLen', default = 0.1, type = float)

args = parser.parse_args()

argFile = args.file
#argFile = 'W:/ENG_Neuromotion_Shared/group/BSI/Shepherd/Recordings/201711201344-Shepherd-Treadmill/201711201344-Shepherd_Treadmill-Data.txt'
fileDir = '/'.join(argFile.split('/')[:-1])
fileName = argFile.split('/')[-1]

stepLen_s = args.stepLen
#stepLen_s = 0.05
winLen_s = args.winLen
#winLen_s = 0.1

#elecID = [4, 5, 6, 7]

#elecLabel = ['Mux4', 'Mux5', 'Mux6', 'Mux7']

#elecUnits = 'uV'

whichChan = 4

data = {'raw' : pd.read_table(argFile, skiprows = 2, header = 0)}
data.update({'elec_ids' : elecID})
data.update(
    {
        'ExtendedHeaderIndices' : [0, 1, 2, 3],
        'extended_headers' : [{'Units' : elecUnits, 'ElectrodeLabel' : elecLabel[i]} for i in [0, 1, 2, 3]],
    }
    )

data.update({'data' : data['raw'].loc[:, [' SenseChannel1 ',
    ' SenseChannel2 ', ' SenseChannel3 ', ' SenseChannel4 ']]})
data['data'].columns = [0,1,2,3]
data.update({'t' : data['raw'].loc[:,  ' Timestamp '].values - 636467827693977000})

sR = linecache.getline(argFile, 2)
sR = [int(s) for s in sR.split() if s.isdigit()][0]
data.update({'samp_per_s' : sR})
data.update({'start_time_s' : 0})
f,_ = plotChan(data, whichChan, label = 'Raw data', mask = None, show = False)

plt.legend()

plotName = fileName.split('.')[0] + '_' +\
    data['extended_headers'][0]['ElectrodeLabel'] +\
    '_plot'
plt.savefig(fileDir + '/' + plotName + '.png')

with open(fileDir + '/' + plotName + '.pickle', 'wb') as File:
    pickle.dump(f, File)

### Get the Spectrogram
R = 30 # target bandwidth for spectrogram

data['spectrum'] = getSpectrogram(
    data, winLen_s, stepLen_s, R, 100, whichChan, plotting = True)

plotName = fileName.split('.')[0] + '_' +\
    data['extended_headers'][0]['ElectrodeLabel'] +\
    '_spectrum_plot'

plt.savefig(fileDir + '/' + plotName + '.png')
with open(fileDir + '/' + plotName + '.pickle', 'wb') as File:
    pickle.dump(plt.gcf(), File)

data.update({'winLen' : winLen_s, 'stepLen' : stepLen_s})

with open(fileDir + '/' + fileName.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
    pickle.dump(data, f, protocol=4 )
