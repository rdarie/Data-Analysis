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

arguments = docopt(__doc__)

plotting = False
if arguments['--plotting']:
    plotting = True

if arguments['--filePath']:
    filePath = arguments['--filePath']
    #filePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings\\201804271016-Proprio\\Trial002.ns5'
    dataDir = os.path.dirname(os.path.abspath(filePath))

fileName = (os.path.basename(filePath).split('.'))[0]
subDir = dataDir + '/KiloSort/' + fileName + '/'
if not os.path.exists(subDir):
    os.makedirs(subDir)

elec_ids = range(1,97) # 'all' is default for all (1-indexed)
start_time_s = 0 # 0 is default for all
    #import pdb; pdb.set_trace()

# Open file and extract headers
nsxFile = NsxFile(filePath)
nsxFile.basic_header.keys()
nsxFile.basic_header['BytesInHeader']
nsxFile.extended_headers[0]
# Peek at first electrode to find out how many samples there are
# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
dummyData = nsxFile.getdata(1)

if arguments['--nSec']:
    if arguments['--nSec'] != 'all':
        data_time_s = float(arguments['--nSec']) # 'all' is default for all
        maxTime_s = min( data_time_s, dummyData['data_time_s'])
        #data_time_s = 2
    else:
        data_time_s = arguments['--nSec']
        maxTime_s = dummyData['data_time_s']

# Create filters
nyq = 0.5 * dummyData['samp_per_s']
low = 250 / nyq
high = 5e3 / nyq
b, a = signal.butter(4, [low, high], btype='band')

del dummyData
timeGroupBy_s = 250 #extract groups of 100 seconds
timeRanges = range(timeGroupBy_s, timeGroupBy_s * math.ceil(maxTime_s / timeGroupBy_s), timeGroupBy_s)

for idx, time in enumerate(timeRanges):
    print('data batch: %d, time points: %d to %d' % (idx, time - timeGroupBy_s, time))
    currData = nsxFile.getdata(elec_ids, time - timeGroupBy_s, timeGroupBy_s)

    channelData = {
        'data' : pd.DataFrame(currData['data']),
        'samp_per_s' : currData['samp_per_s'],
        'elec_ids' : currData['elec_ids'],
        'ExtendedHeaderIndices' : currData['ExtendedHeaderIndices'],
        't' : currData['start_time_s'] + np.arange(currData['data'].shape[1]) / currData['samp_per_s'],
        'basic_headers' : nsxFile.basic_header,
        'extended_headers' :  nsxFile.extended_headers
        }

    del currData
    #channelData = fillInOverflow(channelData, plotting = plotting)
    channelData['data'] = channelData['data'].transpose()
    badData = getBadContinuousMask(channelData, smoothing_ms = 0.5)

    if plotting and idx == 0:
        whichChan = 34 # 1-indexed
        ch_idx  = channelData['elec_ids'].index(whichChan)
        f,_ = plotChan(channelData, whichChan, label = 'Raw data', mask = None, show = False)

    # interpolate bad data
    #idx, row = next(channelData['data'].iteritems())
    for rowIdx, row in channelData['data'].iteritems():
        mask = np.logical_or(badData['general'], badData['perChannel'][rowIdx])
        cleanRow = replaceBad(row, mask, typeOpt = 'interp')
        filteredRow = signal.filtfilt(b, a, cleanRow.values)
        channelData['data'].iloc[:, rowIdx] = filteredRow
        print('On row: %d' % rowIdx)

    if plotting and idx == 0:
        # check interpolation results
        plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
        plotChan(channelData, whichChan, label = 'Clean data', mask = plot_mask,
            maskLabel = "Dropout", show = False, prevFig = f)

        plt.legend()
        plt.savefig(subDir + fileName + '_ns5Clean.png')
    #plt.show()

    #re-digitize to uint16 for kiloSort
    if idx == 0:
        allStd = channelData['data'].values.flatten().std()
        leftBin = channelData['data'].mean().mean() - 15 * allStd
        rightBin = channelData['data'].mean().mean() + 15 * allStd
        #digitizationBins = np.linspace(ChannelData['data'].min().min(),ChannelData['data'].max().max(), 2 ** 16)
        digitizationBins = np.linspace(leftBin,rightBin, 2 ** 16)

    exportDataInt = np.array(np.digitize(channelData['data'],digitizationBins) - 2 ** 15, np.int16)

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
