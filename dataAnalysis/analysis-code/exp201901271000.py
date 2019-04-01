#  options
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

miniRCTrial = False
#  plottingFigures = False
plottingFigures = False
plotBlocking = True
trialIdx = 3
experimentName = '201901271000-Proprio'
deviceName = 'DeviceNPC700373H'
#  remote paths
#  remoteBasePath = '..'
remoteBasePath = 'Z:\\data\\rdarie\\Murdoc Neural Recordings'
insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
nspFolder = os.path.join(remoteBasePath, experimentName)
ns5FileName = 'Trial00{}'.format(trialIdx)

jsonSessionNames = {
    #  per trial
    1: ['Session1548605159361', 'Session1548605586940'],
    2: ['Session1548606783930'],
    3: ['Session1548608122565'],
    4: ['Session1548609521574']
    }

tapDetectOpts = {}
tapDetectOpts[1] = {
    #  per trialSegment
    0: {
        'timeRanges': [(14, 16)],
        'tdChan': 'ins_td0',
        'tdThres': 2,
        'iti': 0.2,
        'keepIndex': slice(None, 2)
        },
    1: {
        'timeRanges': [(448, 450)],
        'tdChan': 'ins_td0',
        'tdThres': 2,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        }
    }
tapDetectOpts[2] = {
    #  per trialSegment
    0: {
        'timeRanges': [(18, 20)],
        'tdChan': 'ins_td0',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(1, None)
        }
    }
tapDetectOpts[3] = {
    #  per trialSegment
    0: {
        'timeRanges': [(20, 22)],
        'tdChan': 'ins_td0',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(None)
        }
    }
tapDetectOpts[4] = {
    #  per trialSegment
    0: {
        'timeRanges': [(21, 24)],
        'tdChan': 'ins_td0',
        'tdThres': 2.5,
        'iti': 0.2,
        'keepIndex': slice(None)
        }
    }

sessionTapRangesNSP = {
    #  per trialSegment
    1: {
        0: {'timeRanges': [212, 214], 'keepIndex': slice(None, 2)},
        1: {'timeRanges': [647, 649], 'keepIndex': slice(1, None)}
        },
    2: {
        0: {'timeRanges': [203, 205], 'keepIndex': slice(1, None)}
        },
    3: {
        0: {'timeRanges': [105, 107], 'keepIndex': slice(None)}
        },
    4: {
        0: {'timeRanges': [140, 144], 'keepIndex': slice(None)}
        }
    }

interpFunINStoNSP = {
    #  per trial
    1: [None, None],
    2: [None],
    3: [None],
    4: [None]
    }
interpFunHUTtoINS = {
    #  per trial
    1: [None, None],
    2: [None],
    3: [None],
    4: [None]
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
        'ns5FileName': 'Trial00{}'.format(trialIdx),
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
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['ns5FileName'] + '_analyze.nix')
trialBasePath = os.path.join(
    trialFilesFrom['utah']['folderPath'],
    trialFilesFrom['utah']['ns5FileName'] + '.nix')
spikePath = os.path.join(
        trialFilesFrom['utah']['folderPath'],
        'tdc_' + trialFilesFrom['utah']['ns5FileName'],
        'tdc_' + trialFilesFrom['utah']['ns5FileName'] + '.nix')
insDataPath = os.path.join(
        trialFilesStim['ins']['folderPath'],
        trialFilesStim['ins']['experimentName'],
        trialFilesStim['ins']['ns5FileName'] + '_ins.nix'
    )
binnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['ns5FileName'] + '_binarized.nix')

#  paths relevant to the entire experimental day
masterFeaturePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_MasterFeature.hdf')
estimatorPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_estimator.joblib')
experimentDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_analyze.nix')
experimentBinnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_binarized.nix')
featurePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_features.hdf')
figureFolder = os.path.join(
    '..', 'figures'
    )