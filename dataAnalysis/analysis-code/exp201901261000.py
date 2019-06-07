#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

miniRCTrial = False
#  plottingFigures = True
plottingFigures = False
plotBlocking = True
trialIdx = 3
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
    4: ['Session1548524126669'],
    }

tapDetectOpts = {}
tapDetectOpts[4] = {
    #  per trialSegment
    0: {
        'timeRanges': [(15, 17)],
        'tdChan': 'ins_td3',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        }
    }

sessionTapRangesNSP = {
    #  per trialSegment
    4: {
        0: {'timeRanges': [938, 940], 'keepIndex': slice(None)}
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
#  should rename to something more intuitive
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
stimDetectOpts = {grpIdx: {
    0: {'detectChannels': ['ins_td2'], 'thres': 1},
    1: {'detectChannels': ['ins_td2'], 'thres': 1},
    2: {'detectChannels': ['ins_td2'], 'thres': 1},
    3: {'detectChannels': ['ins_td2'], 'thres': 1}
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
            'fixedDelay': 0e-3,
            'delayByFreqMult': 1,
            'gaussWid': 100e-3,
            'minDist': 0.2, 'minDur': 0.2,
            'cyclePeriodCorrection': 17.5e-3,
            'plotAnomalies': False,
            'recalculateExpectedOffsets': False,
            'maxSpikesPerGroup': 1, 'plotting': []  # range(1, 1000, 5)
            }
        }
    }

if miniRCTrial:
    trialFilesStim['ins']['getINSkwargs'].update({
        'minDist': 1.2,
        'gaussWid': 200e-3,
        'maxSpikesPerGroup': 0,
    })
    trialFilesFrom['utah'].update({
        'eventInfo': None
    })

nspPrbPath = os.path.join('.', 'nsp_map.prb')
triFolder = os.path.join(
    nspFolder, 'tdc_' + ns5FileName)

# make .prb file for spike sorting
#  {'xcoords': 2, 'ycoords': 2}
nspCmpPath = os.path.join('.', 'nsp_map.cmp')
cmpDF = tdch.cmpToDF(nspCmpPath)
tdch.cmpDFToPrb(
    cmpDF, filePath=nspPrbPath,
    names=['elec'],
    groupIn={'xcoords': 1, 'ycoords': 1}, appendDummy=16)

#  should rename these to something more intuitive
#  paths relevant to individual trials
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
figureFolder = os.path.join(
    remoteBasePath, 'figures', experimentName
    )
rasterOpts = {
    'binInterval': 1e-3, 'binWidth': 30e-3,
    'windowSize': (-5, 5),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'alignTo': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}

alignedRastersFolder = os.path.join(figureFolder, 'alignedRasters')
if not os.path.exists(alignedRastersFolder):
    os.makedirs(alignedRastersFolder, exist_ok=True)
alignedFeaturesFolder = os.path.join(figureFolder, 'alignedFeatures')
if not os.path.exists(alignedFeaturesFolder):
    os.makedirs(alignedFeaturesFolder, exist_ok=True)
spikeSortingFiguresFolder = os.path.join(figureFolder, 'spikeSorting')
if not os.path.exists(spikeSortingFiguresFolder):
    os.makedirs(spikeSortingFiguresFolder, exist_ok=True)