# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:07:44 2019

@author: Radu
"""

from importlib import reload
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import matplotlib, os, pickle
matplotlib.use('TkAgg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
#  matplotlib.use('PS')   # generate interactive output by default
experimentName = '201901271000-Proprio'
folderPath = './' + experimentName

jsonBaseFolder = '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/ORCA Logs'
deviceName = 'DeviceNPC700373H'

jsonSessionNames = {
    1: ['Session1548605159361',
        'Session1548605586940'],
    5: ['Session1548611405556',
        'Session1548612434879',
        'Session1548612688167']
    }

trialIdx = 5

trialFilesFrom = {
    'utah': {
        'origin': 'mat',
        'experimentName': experimentName,
        'folderPath': folderPath,
        'ns5FileName': 'Trial00{}'.format(trialIdx),
        'elecIDs': list(range(1, 97)) + [135],
        'excludeClus': []}
    }

#  DEBUGGING:
#  trialFilesFrom['utah']['elecIDs'] = [70, 71, 135]

trialFilesStim = {
    'ins': {
        'origin': 'ins',
        'experimentName': experimentName,
        'folderPath': jsonBaseFolder,
        'ns5FileName': 'Trial00{}'.format(trialIdx),
        'jsonSessionNames': jsonSessionNames[trialIdx],
        'elecIDs': range(17),
        'excludeClus': []}
    }

stimOptsName = ksa.spikesNameGenerator(
    'ins', trialFilesStim['ins']) + '_opts'
optsPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    stimOptsName + '.pickle'
)

with open(optsPath, 'rb') as f:
    extendedTrialStim = pickle.load(f)
trialFilesStim.update(extendedTrialStim)
# raster
rasterOpts = {
    'kernelWidth': 20e-3,
    'binInterval': (5**(-1)) * 1e-3, 'binWidth': (5**(-1)) * 1e-3,
    'windowSize': (-0.4, 0.4),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.99)}

#  generate report
ksa.generateSpikeTriggeredAverageReport(
    folderPath, trialFilesFrom,
    trialFilesStim, plotOpts=plotOpts, rasterOpts=rasterOpts)
