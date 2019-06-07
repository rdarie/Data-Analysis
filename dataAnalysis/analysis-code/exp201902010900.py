#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

miniRCTrial = False
#  plottingFigures = True
plottingFigures = False
plotBlocking = True
trialIdx = 1
experimentName = '201901211000-Proprio'
deviceName = 'DeviceNPC700373H'
#  remote paths
remoteBasePath = '..'
scratchPath = '/gpfs/scratch/rdarie/rdarie/Murdoc Neural Recordings'
#  remoteBasePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings'
insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
nspFolder = os.path.join(remoteBasePath, 'raw', experimentName)

ns5FileName = 'Trial00{}'.format(trialIdx)

jsonSessionNames = {
    1: ['Session1549035660603'],
    2: ['Session1549037174687'],
    3: ['Session1549038622083'],
    4: ['Session1549039928719'],
    5: ['Session1549041618954']
    }

#  options for automatic tap detection on ins data
tapDetectOpts = {}
tapDetectOpts[1] = {
    0: {
        'timeRanges': [(245, 255)],
        'accChan': 'ins_accinertia',
        'accThres': 2,
        'iti': 0.2,
        'keepIndex': slice(None)
        }
    }
tapDetectOpts[2] = {
    0: {
        'timeRanges': [(10, 12)],
        'accChan': 'ins_accz',
        'accThres': 2,
        'iti': 0.25,
        'keepIndex': slice(None)
        }
    }
tapDetectOpts[3] = {
    0: {
        'timeRanges': [(12.9, 14)],
        'tdChan': 'ins_td0',
        'tdThres': 2,
        'iti': 0.2,
        'keepIndex': [0, 1, 3]
        }
    }
tapDetectOpts[4] = {
    0: {
        'timeRanges': [(32.6, 34.2)],
        'accChan': 'ins_accinertia',
        'accThres': 2,
        'iti': 0.2,
        'keepIndex': slice(None)
        }
    }
tapDetectOpts[5] = {
    0: {
        'timeRanges': [(17.3, 22)],
        'tdChan': 'ins_td2',
        'tdThres': 3.5,
        'iti': 0.4,
        'keepIndex': slice(None, 3)
        }
    }

sessionTapRangesNSP = {
    1: {0: {'timeRanges': [1600, 1602], 'keepIndex': slice(None)}},
    2: {0: {'timeRanges': [130, 140], 'keepIndex': slice(None)}},
    3: {0: {'timeRanges': [150, 160], 'keepIndex': slice(None)}},
    4: {0: {'timeRanges': [150, 153], 'keepIndex': slice(None)}},
    5: {0: {'timeRanges': [180, 190], 'keepIndex': slice(None)}},
    }

#  make placeholders for interpolation functions
interpFunINStoNSP = {
    key: [None for i in value.keys()]
    for key, value in sessionTapRangesNSP.items()
    }
interpFunHUTtoINS = {
    key: [None for i in value.keys()]
    for key, value in sessionTapRangesNSP.items()
    }

eventInfo = {
    'inputIDs': {
        'A+': 1,
        'B+': 2,
        'Z+': 3,
        'A-': 5,
        'B-': 4,
        'Z-': 6,
        'rightBut': 11,
        'leftBut': 12,
        'rightLED': 9,
        'leftLED': 10,
        'simiTrigs': 8,
        }
}
#  process int names into strings
for key, value in eventInfo['inputIDs'].items():
    eventInfo['inputIDs'][key] = 'ainp{}'.format(value)

trialFilesFrom = {
    'utah': {
        'origin': 'mat',
        'experimentName': experimentName,
        'folderPath': nspFolder,
        'ns5FileName': ns5FileName,
        'elecIDs': list(range(1, 97)) + [135],
        'excludeClus': []
        }
    }
trialFilesFrom['utah'].update(dict(eventInfo=eventInfo))

#  get stim onset times (one set of params per program, assuming only group 0)
stimDetectOpts = {0: {
    0: {'detectChannels': ['ins_td2'], 'thres': 2, 'keep_max': False},
    1: {'detectChannels': ['ins_td2'], 'thres': 2, 'keep_max': False},
    2: {'detectChannels': ['ins_td2'], 'thres': 2, 'keep_max': False},
    3: {'detectChannels': ['ins_td2'], 'thres': 2, 'keep_max': False}
    }}

trialFilesStim = {
    'ins': {
        'origin': 'ins',
        'experimentName': experimentName,
        'folderPath': insFolder,
        'ns5FileName': ns5FileName,
        'jsonSessionNames': jsonSessionNames[trialIdx],
        'elecIDs': range(17),
        'excludeClus': [],
        'forceRecalc': True,
        'detectStim': True,
        'getINSkwargs': {
            'stimDetectOpts': stimDetectOpts,
            'fixedDelay': -60e-3,
            'delayByFreqMult': 1,
            'gaussWid': 100e-3,
            'minDist': 0.2, 'minDur': 0.2,
            'cyclePeriodCorrection': 20e-3,
            'plotAnomalies': True,
            'recalculateExpectedOffsets': True,
            'maxSpikesPerGroup': 1, 'plotting': []  # range(1, 1000, 5)
            }
        }
    }

miniRCDetectionOpts = {
        'minDist': 1.2,
        'gaussWid': 200e-3,
        'maxSpikesPerGroup': 0,
    }
if miniRCTrial:
    trialFilesStim['ins']['getINSkwargs'].update(miniRCDetectionOpts)
    trialFilesFrom['utah'].update({
        'eventInfo': None
    })

nspPrbPath = os.path.join('.', 'nsp_map.prb')
triFolder = os.path.join(
    nspFolder, 'tdc_Trial00{}'.format(1))
'''
triFolder = os.path.join(
    scratchPath, 'tdc',
    experimentName, 'tdc_Trial00{}'.format(1)
    )
if not os.path.exists(triFolder):
    os.makedirs(triFolder, exist_ok=True)
'''
triFolderSource = triFolder
triDestinations = [
    'Trial00{}'.format(trialIdx)
    for trialIdx in []]

remakePrb = False
nspCmpPath = os.path.join('.', 'nsp_map.cmp')
cmpDF = tdch.cmpToDF(nspCmpPath)
# make .prb file for spike sorting
#  groupIn={'xcoords': 2, 'ycoords': 2}
if remakePrb:
    nspCsvPath = os.path.join('.', 'nsp_map.csv')
    cmpDF.to_csv(nspCsvPath)
    tdch.cmpDFToPrb(
        cmpDF, filePath=nspPrbPath,
        names=['elec', 'ainp'],
        groupIn={'xcoords': 1, 'ycoords': 1})

#  should rename these to something more intuitive
#  paths relevant to individual trials
processedFolder = os.path.join(
    remoteBasePath, 'processed', experimentName)
if not os.path.exists(processedFolder):
    os.makedirs(processedFolder, exist_ok=True)
analysisDataPath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    ns5FileName + '_analyze.nix')
trialBasePath = os.path.join(
    nspFolder,
    ns5FileName + '.nix')
spikePath = os.path.join(
    nspFolder,
    'tdc_' + trialFilesFrom['utah']['ns5FileName'],
    'tdc_' + trialFilesFrom['utah']['ns5FileName'] + '.nix')
insDataPath = os.path.join(
    remoteBasePath, 'raw', experimentName,
    ns5FileName + '_ins.nix')
binnedSpikePath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    ns5FileName + '_binarized.nix')
trialTriggeredPath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    ns5FileName + '_triggered.nix')

#  paths relevant to the entire experimental day
estimatorPath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    experimentName + '_estimator.joblib')
experimentDataPath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    experimentName + '_analyze.nix')
experimentTriggeredPath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    experimentName + '_triggered.nix')
experimentBinnedSpikePath = os.path.join(
    remoteBasePath, 'processed', experimentName,
    experimentName + '_binarized.nix')

#  figure paths
figureFolder = os.path.join(
    remoteBasePath, 'figures', experimentName
    )
alignedRastersFolder = os.path.join(figureFolder, 'alignedRasters')
if not os.path.exists(alignedRastersFolder):
    os.makedirs(alignedRastersFolder, exist_ok=True)
alignedFeaturesFolder = os.path.join(figureFolder, 'alignedFeatures')
if not os.path.exists(alignedFeaturesFolder):
    os.makedirs(alignedFeaturesFolder, exist_ok=True)
spikeSortingFiguresFolder = os.path.join(figureFolder, 'spikeSorting')
if not os.path.exists(spikeSortingFiguresFolder):
    os.makedirs(spikeSortingFiguresFolder, exist_ok=True)

rasterOpts = {
    'binInterval': 1e-3, 'binWidth': 30e-3,
    'windowSize': (-.5, .5),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'alignTo': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}
