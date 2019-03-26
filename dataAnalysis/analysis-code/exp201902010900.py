#  options
import os
trialIdx = 1
miniRCTrial = False
#plottingFigures = False
plottingFigures = True
plotBlocking = True

experimentName = '201902010900-Proprio'
deviceName = 'DeviceNPC700373H'
jsonBaseFolder = './ORCA Logs'
folderPath = './' + experimentName

jsonSessionNames = {
    1: ['Session1549035660603'],
    2: ['Session1549037174687'],
    3: ['Session1549038622083'],
    4: ['Session1549039928719'],
    5: ['Session1549041618954']
    }

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

interpFunINStoNSP = {
    1: [None],
    2: [None],
    3: [None],
    4: [None],
    5: [None]
    }
interpFunHUTtoINS = {
    1: [None],
    2: [None],
    3: [None],
    4: [None],
    5: [None]
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

for key, value in eventInfo['inputIDs'].items():
    eventInfo['inputIDs'][key] = 'ainp{}'.format(value)
trialFilesFrom = {
    'utah': {
        'origin': 'mat',
        'experimentName': experimentName,
        'folderPath': folderPath,
        'ns5FileName': 'Trial00{}'.format(trialIdx),
        'elecIDs': list(range(1, 97)) + [135],
        'excludeClus': []
        }
    }
trialFilesFrom['utah'].update(dict(eventInfo=eventInfo))
#  DEBUGGING:
trialFilesFrom['utah']['elecIDs'] = [70, 71, 135]

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
        'folderPath': jsonBaseFolder,
        'ns5FileName': 'Trial00{}'.format(trialIdx),
        'jsonSessionNames': jsonSessionNames[trialIdx],
        'elecIDs': range(17),
        'excludeClus': [],
        'forceRecalc': True,
        'detectStim': True,
        'getINSkwargs': {
            'stimDetectOpts': stimDetectOpts,
            'stimIti': 0, 'fixedDelay': 10e-3,
            'minDist': 0.2, 'minDur': 0.2, 'thres': 3,
            'gaussWid': 200e-3,
            'gaussKerWid': 75e-3,
            'maxSpikesPerGroup': 1, 'plotting': []  # range(1, 1000, 5)
            }
        }
    }
if miniRCTrial:
    trialFilesStim['ins']['getINSkwargs'].update({
        'minDist': 1.2,
        'gaussWid': 400e-3,
        'gaussKerWid': 250e-3,
        'maxSpikesPerGroup': 0,
    })
    trialFilesFrom['utah'].update({
        'eventInfo': None
    })
rasterOpts = {
    'binInterval': (5) * 1e-3, 'binWidth': (20) * 1e-3,
    'windowSize': (-.05, .15),
    'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
    'separateByFunArgs': None,
    'separateByFunKWArgs': {'type': 'Classification'}
    }
plotOpts = {
    'type': 'ticks', 'errorBar': 'sem',
    'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}
    
analysisDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['ns5FileName'] + '_analyze.nix')
trialBasePath = os.path.join(
    trialFilesFrom['utah']['folderPath'],
    trialFilesFrom['utah']['ns5FileName'] + '.nix')
