#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

miniRCTrial = True
#  plottingFigures = True
plottingFigures = False
plotBlocking = True
trialIdx = 1
experimentName = '201901201200-Proprio'
deviceName = 'DeviceNPC700373H'
#  remote paths
remoteBasePath = '..'
#  remoteBasePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings'
insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
nspFolder = os.path.join(remoteBasePath, 'raw', experimentName)
ns5FileName = 'Trial00{}'.format(trialIdx)

jsonSessionNames = {
    #  per trial
    1: ['Session1548006829607'],
    2: [
        'Session1548008243454',
        'Session1548009494100',
        'Session1548010328823',
        'Session1548010624175']
    }

tapDetectOpts = {}
tapDetectOpts[1] = {
    #  per trialSegment
    0: {
        'timeRanges': [(79, 81)],
        'tdChan': 'ins_td3',
        'tdThres': 2,
        'iti': 0.2,
        'keepIndex': slice(None, 2)
        }
    }
tapDetectOpts[2] = {
    #  per trialSegment
    0: {
        'timeRanges': [(18, 20)],
        'tdChan': 'ins_td2',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        },
    1: {
        'timeRanges': [(18, 20)],
        'tdChan': 'ins_td2',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        },
    2: {
        'timeRanges': [(18, 20)],
        'tdChan': 'ins_td2',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        },
    3: {
        'timeRanges': [(18, 20)],
        'tdChan': 'ins_td2',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        },
    }

sessionTapRangesNSP = {
    #  per trialSegment
    1: {
        0: {'timeRanges': [212, 214], 'keepIndex': slice(None, 2)}
        },
    2: {
        0: {'timeRanges': [203, 205], 'keepIndex': slice(1, None)},
        1: {'timeRanges': [203, 205], 'keepIndex': slice(1, None)},
        2: {'timeRanges': [203, 205], 'keepIndex': slice(1, None)},
        3: {'timeRanges': [203, 205], 'keepIndex': slice(1, None)}
        }
    }

interpFunINStoNSP = {
    #  per trial
    1: [None],
    2: [None, None, None, None]
    }
interpFunHUTtoINS = {
    #  per trial
    1: [None],
    2: [None, None, None, None]
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
    0: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
    1: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
    2: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
    3: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1}
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
triFolderSource = os.path.join(
    '..', 'raw', '201901211000-Proprio', 'tdc_Trial001')
triDestinations = [
    'Trial00{}'.format(trialIdx)
    for trialIdx in [1, 2]]

remakePrb = True
# make .prb file for spike sorting
#  groupIn={'xcoords': 2, 'ycoords': 2}
if remakePrb:
    nspCmpPath = os.path.join('.', 'nsp_map.cmp')
    cmpDF = tdch.cmpToDF(nspCmpPath)
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
    remoteBasePath, 'processed', experimentName,
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
    'windowSize': (-.5, .5),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'alignTo': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}
