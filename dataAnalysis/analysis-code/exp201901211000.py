#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

trialIdx = 3
ns5FileName = 'Trial00{}'.format(trialIdx)
#
miniRCTrialLookup = {
    1: False,
    2: False,
    3: False
    }
miniRCTrial = miniRCTrialLookup[trialIdx]
#  plottingFigures = True
plottingFigures = False
plotBlocking = True

experimentName = '201901211000-Proprio'
deviceName = 'DeviceNPC700373H'
#  remote paths
remoteBasePath = '..'
scratchPath = '/gpfs/scratch/rdarie/rdarie/Murdoc Neural Recordings'
insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
nspFolder = os.path.join(remoteBasePath, 'raw', experimentName)

scratchFolder = os.path.join(scratchPath, experimentName)
if not os.path.exists(scratchFolder):
    os.makedirs(scratchFolder, exist_ok=True)

jsonSessionNames = {
    #  per trial
    1: ['Session1548087177984', 'Session1548087855083'],
    2: ['Session1548088399924'],
    3: ['Session1548089184035', 'Session1548090076537', 'Session1548090536025']
    }

spikeWindow = [-32, 64]
#  options for automatic tap detection on ins data
tapDetectOpts = {
    #  per trial
    1: {
        #  per trialSegment
        0: {
            'timeRanges': [(115, 119)],
            'accChan': 'ins_accinertia',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': [(725, 728)],
            'accChan': 'ins_accinertia',
            'accThres': 1,
            'iti': 0.2,
            'keepIndex': slice(None, 2)
            }
        },
    2: {
        #  per trialSegment
        0: {
            'timeRanges': [(77.5, 79.5)],
            'accChan': 'ins_accinertia',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        },
    3: {
        #  per trialSegment
        0: {
            'timeRanges': [(63, 71)],
            'accChan': 'ins_accz',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': [0, 1, 2, 4, 5]
            },
        1: {
            'timeRanges': [(910, 914)],
            'accChan': 'ins_accx',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None, 2)
            },
        2: {
            'timeRanges': [(1360, 1364)],
            'accChan': 'ins_accz',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    }

sessionTapRangesNSP = {
    #  per trial
    1: {
        #  per trialSegment
        0: {'timeRanges': [60, 64], 'keepIndex': slice(None)},
        1: {'timeRanges': [669, 672], 'keepIndex': slice(None)}
        },
    2: {
        0: {'timeRanges': [101, 103], 'keepIndex': slice(None)}
        },
    3: {
        0: {'timeRanges': [48, 56],
            'keepIndex': slice(None)},
        1: {'timeRanges': [896, 900],
            'keepIndex': slice(None, 2)},
        2: {'timeRanges': [1344, 1348],
            'keepIndex': slice(None)}
        }
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

overrideSegmentsForTapSync = {
    #  each key is a trial
    1: {},
    2: {},
    3: {}
    }

trialFilesFrom = {
    'utah': {
        'origin': 'mat',
        'experimentName': experimentName,
        'folderPath': nspFolder,
        'ns5FileName': ns5FileName,
        'elecIDs': list(range(1, 97)) + [135],
        'excludeClus': [],
        'calcRigEvents': True
        }
    }
trialFilesFrom['utah']['calcRigEvents'] = miniRCTrial
#  get stim onset times (one set of params per program, assuming only group 0)
stimDetectOpts = {grpIdx: {
    0: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
    1: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
    2: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
    3: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1}
    } for grpIdx in range(4)}

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
            'spikeWindow': spikeWindow,
            'plotAnomalies': False,
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
miniRCRigInputs = {
    'tapSync': 'ainp7',
    'simiTrigs': 'ainp8'
    }
fullRigInputs = {
    'A+': 'ainp1',
    'B+': 'ainp2',
    'Z+': 'ainp3',
    'A-': 'ainp5',
    'B-': 'ainp4',
    'Z-': 'ainp6',
    'rightBut': 'ainp11',
    'leftBut': 'ainp12',
    'rightLED': 'ainp9',
    'leftLED': 'ainp10',
    'simiTrigs': 'ainp8',
    'tapSync': 'ainp7',
    }

if miniRCTrial:
    #  override with settings for detecting cycling stim trains
    trialFilesStim['ins']['getINSkwargs'].update(miniRCDetectionOpts)
    #  only parse sync lines
    eventInfo = {
        'inputIDs': miniRCRigInputs}
else:
    #  should rename eventInfo to something more intuitive
    eventInfo = {'inputIDs': fullRigInputs}
    
trialFilesFrom['utah'].update(dict(eventInfo=eventInfo))

nspPrbPath = os.path.join('.', 'nsp_map.prb')
triFolder = os.path.join(
    scratchFolder, 'tdc_Trial00{}'.format(trialIdx))

triFolderSource = triFolder
triDestinations = [
    'Trial00{}'.format(trialIdx)
    for trialIdx in [2, 3]]

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
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    ns5FileName + '_analyze.nix')
trialBasePath = os.path.join(
    scratchFolder,
    ns5FileName + '.nix')
    
insDataPath = os.path.join(
    scratchFolder,
    ns5FileName + '_ins.nix')
binnedSpikePath = os.path.join(
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    ns5FileName + '_binarized.nix')
trialTriggeredPath = os.path.join(
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    ns5FileName + '_triggered.nix')

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
    
#  paths relevant to the entire experimental day
estimatorPath = os.path.join(
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    experimentName + '_estimator.joblib')
experimentDataPath = os.path.join(
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    experimentName + '_analyze.nix')
experimentTriggeredPath = os.path.join(
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    experimentName + '_triggered.nix')
experimentBinnedSpikePath = os.path.join(
    #  remoteBasePath, 'processed', experimentName,
    scratchFolder,
    experimentName + '_binarized.nix')

#  Options relevant to the assembl
experimentsToAssemble = {
    '201901201200-Proprio': [2],
    '201901211000-Proprio': [1, 2, 3],
    }

trialsToAssemble = []
for key in sorted(experimentsToAssemble.keys()):
    val = experimentsToAssemble[key]
    for trialIdx in val:
        trialsToAssemble.append(
            os.path.join(
                scratchPath, key, 'Trial00{}.nix'.format(trialIdx)
            )
        )

movementSizeBins = [-0.9, -0.45, -0.2, 0.2, 0.55, 0.9]
alignTimeBounds = [
    #  each key is a trial
    [
        [200, 667],
        [1370, 1595],
        [2175, 2315],
        [2475, 2495]
        ],
    #  per trial
    [
        #  per trialSegment
        [92, 527],
        [775, 887]
        ],
    [
        [210, 295]
        ],
    [
        [80, 430],
        [917, 1163],
        [1367, 2024]
        ]
    ]