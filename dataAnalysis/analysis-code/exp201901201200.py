#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

trialIdx = 1
ns5FileName = 'Trial00{}'.format(trialIdx)

miniRCTrialLookup = {
    1: True,
    2: False
    }
miniRCTrial = miniRCTrialLookup[trialIdx]
#  plottingFigures = True
plottingFigures = False
plotBlocking = True

experimentName = '201901201200-Proprio'
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
    #  each key is a trial
    1: ['Session1548006829607'],
    2: [
        'Session1548008243454',
        'Session1548009494100',
        'Session1548010328823',
        'Session1548010624175']
    }

spikeWindow = [-32, 64]
tapDetectOpts = {
    #  each key is a trial
    1: {
        #  each key is a trialSegment
        0: {
            'timeRanges': [(79.5, 80.75)],
            'tdChan': 'ins_td3',
            'tdThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None, 2)
            }
        },
    2: {
        #  each key is a trialSegment
        0: {
            'timeRanges': [(113, 115)],
            'accChan': 'ins_accz',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': [(1260, 1262)],
            'accChan': 'ins_accinertia',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        2: {
            'timeRanges': [(2115, 2118)],
            'accChan': 'ins_accinertia',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        3: {
            'timeRanges': [(2407, 2409)],
            'accChan': 'ins_accz',
            'accThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        }
    }

sessionTapRangesNSP = {
    #  each key is a trial
    1: {
        #  each key is a trialSegment
        0: {'timeRanges': [74, 75.25], 'keepIndex': slice(None)}
        },
    2: {
        0: {'timeRanges': [142, 144], 'keepIndex': slice(None)},
        1: {'timeRanges': [1289, 1291], 'keepIndex': slice(None)},  # missing
        2: {'timeRanges': [2144, 2147], 'keepIndex': [0, 1, 3, 4]},
        3: {'timeRanges': [2436, 2438], 'keepIndex': [0, 1]}
        }
    }

#  if not possible to use taps, override with good taps from another segment
#  not ideal, because segments are only synchronized to the nearest **second**
overrideSegmentsForTapSync = {
    #  each key is a trial
    1: {},
    2: {1: 0}  # override segment 1 based on segment 0
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

trialFilesFrom['utah']['calcRigEvents'] = not miniRCTrial
if miniRCTrial:
    #  override with settings for detecting cycling stim trains
    trialFilesStim['ins']['getINSkwargs'].update(miniRCDetectionOpts)
    #  only parse sync lines
    eventInfo = {'inputIDs': miniRCRigInputs}
else:
    #  should rename eventInfo to something more intuitive
    eventInfo = {'inputIDs': fullRigInputs}
    
trialFilesFrom['utah'].update(dict(eventInfo=eventInfo))

nspPrbPath = os.path.join('.', 'nsp_map.prb')
triFolder = os.path.join(
    scratchFolder, 'tdc_Trial00{}'.format(trialIdx))

triFolderSource = os.path.join(
    scratchPath, '201901211000-Proprio', 'tdc_Trial001')
triDestinations = [
    'Trial00{}'.format(trialIdx)
    for trialIdx in [1, 2]]

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
    scratchFolder,
    ns5FileName + '_binarized.nix')
trialTriggeredPath = os.path.join(
    scratchFolder,
    ns5FileName + '_triggered.nix')
    
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
