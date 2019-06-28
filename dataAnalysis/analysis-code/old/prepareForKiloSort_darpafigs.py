""".ns5 to KiloSort int16 .dat file converter.

Usage:
  prepareForKiloSort.py [--filePath=<str>] [--nSec=<float>] [--plotting]
  prepareForKiloSort.py (-h | --help)

Options:
  -h --help     Show this screen.
"""

from dataAnalysis.helperFunctions.helper_functions import *
import numpy as np
import pandas as pd
from docopt import docopt
import itertools, os, sys, math
from scipy import signal

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

plotting = True
if arguments['plotting']:
    plotting = True

if arguments['filePath']:
    filePath = arguments['filePath']
    #filePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings\\201804250947-Proprio\\Trial003.ns5'
    dataDir = os.path.dirname(os.path.abspath(filePath))

fileName = (os.path.basename(filePath).split('.'))[0] + '_darpa'
subDir = dataDir + '/KiloSort/' + fileName + '/'
if not os.path.exists(subDir):
    os.makedirs(subDir)

elec_ids = range(1,97) # 'all' is default for all (1-indexed)
start_time_s = 216 # 0 is default for all
    #import pdb; pdb.set_trace()

# Open file and extract headers
nsxFile = NsxFile(filePath)
nsxFile.basic_header.keys()
nsxFile.basic_header['BytesInHeader']
nsxFile.extended_headers[0]
# Peek at first electrode to find out how many samples there are
# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
dummyData = nsxFile.getdata(1)

if arguments['nSec']:
    if arguments['nSec'] != 'all':
        data_time_s = float(arguments['nSec']) # 'all' is default for all
        #data_time_s = 180
        maxTime_s = min( data_time_s, dummyData['data_time_s'])
    else:
        data_time_s = arguments['nSec']
        maxTime_s = dummyData['data_time_s']

# Create filters
nyq = 0.5 * dummyData['samp_per_s']
low = 250 / nyq
high = 5e3 / nyq
b, a = signal.butter(4, [low, high], btype='band')

del dummyData
timeGroupBy_s = 250 #extract groups of 100 seconds
timeRanges = range(0, timeGroupBy_s * math.ceil(maxTime_s / timeGroupBy_s), timeGroupBy_s)
#idx, time = next(enumerate(timeRanges))
for idx, startTime in enumerate(timeRanges):
    currData = nsxFile.getdata(elec_ids, startTime, timeGroupBy_s)

    ChannelData = {
        'data' : pd.DataFrame(currData['data']).transpose(),
        'samp_per_s' : currData['samp_per_s'],
        'elec_ids' : currData['elec_ids'],
        'ExtendedHeaderIndices' : currData['ExtendedHeaderIndices'],
        't' : currData['start_time_s'] + np.arange(currData['data'].shape[1]) / currData['samp_per_s'],
        'basic_headers' : nsxFile.basic_header,
        'extended_headers' :  nsxFile.extended_headers
        }

    del currData
    badData = getBadContinuousMask(ChannelData, smoothing_ms = 0.5)

    if plotting and idx == 0:
        whichChan = 34 # 1-indexed
        ch_idx  = ChannelData['elec_ids'].index(whichChan)
        f,_ = plotChan(ChannelData, whichChan, label = 'Raw data', mask = None, show = False)

    # interpolate bad data
    #idx, row = next(ChannelData['data'].iteritems())
    for rowIdx, row in ChannelData['data'].iteritems():
        mask = np.logical_or(badData['general'], badData['perChannel'][rowIdx])
        cleanRow = replaceBad(row, mask, typeOpt = 'interp')
        filteredRow = signal.filtfilt(b, a, cleanRow.values)
        ChannelData['data'].iloc[:, rowIdx] = filteredRow
        print('On row: %d' % rowIdx)

    if plotting and idx == 0:
        # check interpolation results
        plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
        plotChan(ChannelData, whichChan, label = 'Clean data', mask = plot_mask,
            maskLabel = "Dropout", show = False, prevFig = f)

        plt.legend()
        plt.savefig(subDir + fileName + '_ns5Clean.png')
    #plt.show()

    #re-digitize to uint16 for kiloSort
    if idx == 0:
        allStd = ChannelData['data'].values.flatten().std()
        leftBin = ChannelData['data'].mean().mean() - 15 * allStd
        rightBin = ChannelData['data'].mean().mean() + 15 * allStd
        #digitizationBins = np.linspace(ChannelData['data'].min().min(),ChannelData['data'].max().max(), 2 ** 16)
        digitizationBins = np.linspace(leftBin,rightBin, 2 ** 16)

    exportDataInt = np.array(np.digitize(ChannelData['data'],digitizationBins) - 2 ** 15, np.int16)

    #exportDataInt.shape
    #plt.plot(exportDataInt[10,:])
    #plt.show()
    if idx == 0:
        with open(subDir + fileName + '.dat', 'wb') as fid:
            exportDataInt.astype('int16').tofile(fid)
    else:
        with open(subDir + fileName + '.dat', 'wb+') as fid:
            exportDataInt.astype('int16').tofile(fid)

# Close the nsx file now that all data is out
nsxFile.close()
print('Complete.')
