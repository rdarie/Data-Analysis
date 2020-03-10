""".ns5 to MountainSort file converter.

Usage:
  prepareForKiloSort.py [--filePath=<str>] [--nSec=<float>] [--plotting] [--whichChan]
  prepareForKiloSort.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

import sys
sys.path.append("/users/rdarie/Github/mountainsort/packages/pyms/mlpy")
import mdaio
from dataAnalysis.helperFunctions.helper_functions import *
import numpy as np
import pandas as pd
from scipy import signal
import itertools
from docopt import docopt
import os

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

plotting = False
if arguments['plotting']:
    plotting = True
if arguments['whichChan']:
    whichChan = float(arguments['whichChan'])
else:
    whichChan = 1

if arguments['filePath']:
    filePath = arguments['filePath']
    #filePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings\\201804271016-Proprio\\Block002.ns5'
    dataDir = os.path.dirname(os.path.abspath(filePath))

fileName = (os.path.basename(filePath).split('.'))[0]

# Open file and extract headers
nsxFile = NsxFile(filePath)
nsxFile.basic_header.keys()
nsxFile.basic_header['BytesInHeader']
nsxFile.extended_headers[0]
# Peek at first electrode to find out how many samples there are
# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
dummyData = nsxFile.getdata(1)
#all_elec_ids = [range(1,65), range(65,97)] # 'all' is default for all (1-indexed)
elec_ids = range(1,97)
elecGroupNames = ['Utah', 'NForm']
start_time_s = 0 # 0 is default for all
    #import pdb; #)

if arguments['nSec']:
    if arguments['nSec'] != 'all':
        data_time_s = float(arguments['nSec']) # 'all' is default for all
        maxTime_s = min( data_time_s, dummyData['data_time_s'])
        #data_time_s = 2
    else:
        data_time_s = arguments['nSec']
        maxTime_s = dummyData['data_time_s']

# Create filters
nyq = 0.5 * dummyData['samp_per_s']
low = 250 / nyq
high = 5e3 / nyq
b, a = signal.butter(4, [low, high], btype='band')

del dummyData, nsxFile
ChannelData = getNSxData(filePath, elec_ids, start_time_s, data_time_s)

ch_idx  = ChannelData['elec_ids'].index(whichChan)

badData = getBadContinuousMask(ChannelData, plotting = whichChan, smoothing_ms = 0.5)

f,_ = plotChan(ChannelData, whichChan, label = 'Raw data', mask = None, show = False)

# interpolate bad data
for idx, row in ChannelData['data'].iteritems():
    mask = np.logical_or(badData['general'], badData['perChannel'][idx])
    row = replaceBad(row, mask, typeOpt = 'interp')

exportData = ChannelData['data'].transpose()
# check interpolation results
plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
plotChan(ChannelData, whichChan, label = 'Clean data', mask = plot_mask,
    maskLabel = "Dropout", show = False, prevFig = f)

plt.legend()

subDir = dataDir + '/' + fileName + '_MountainSort/'
if not os.path.exists(subDir):
    os.makedirs(subDir)
plt.savefig(subDir + fileName + '_ns5Clean.png')

## Write geom.csv
x = np.arange(0, 400 * 10, 400)
y = np.arange(0, 400 * 10, 400)
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

geom = []
for x, y in zip(xv, yv):
    for i,j in zip(x,y):
        geom.append((i,j))
# remove corners
geom.remove((0,0))
geom.remove((0,3600))
geom.remove((3600,0))
geom.remove((3600, 3600))

geomSubset = [geom[i-1] for i in elec_ids]
groupIn = 4
for idxNum, idx in enumerate(range(0,len(exportData),groupIn)):
    #print(list(range(idx, idx + groupIn)))
    #print(exportData.loc[idx: idx + groupIn - 1, :])
    mdaio.writemda32(exportData.loc[idx: idx + groupIn - 1, :].values, subDir + fileName + '_' + ('%d' % idxNum) + '.mda')
    theseGeoms = pd.DataFrame(geomSubset[idx: idx + groupIn])
    theseGeoms.to_csv(subDir + 'geom_' + ('%d' % idxNum) + '.csv', header = False, index = False)
