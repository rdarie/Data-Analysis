import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio-miniRC',
        }
    fullRigInputs = {
        'B+': 'ainp1',
        'A+': 'ainp2',
        'Z+': 'ainp3',
        'B-': 'ainp5',
        'A-': 'ainp4',
        'Z-': 'ainp6',
        'rightBut': 'ainp11',
        'leftBut': 'ainp12',
        'rightLED': 'ainp9',
        'leftLED': 'ainp10',
        'simiTrigs': 'ainp8',
        'tapSync': 'ainp7',
        }
    miniRCRigInputs = {
        'tapSync': 'ainp7',
        'simiTrigs': 'ainp8'
        }
    RCRigInputs = {
        'kinectSync': 'ainp16',
        }

    experimentName = '201901301000-Murdoc'
    deviceName = 'DeviceNPC700373H'
    subjectName = 'Murdoc'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548863850778'],
        2: ['Session1548865200417'],
        3: ['Session1548866698989'],
        4: ['Session1548868212519', 'Session1548869346906'],
        }

    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': ['ins_td0', 'ins_td2'],
                # 'synchChanName': [
                #     'ins_accx', 'ins_accy',
                #     'ins_accz', 'ins_accinertia'],
                'synchStimUnitName': None,
                'synchByXCorrTapDetectSignal': False,
                'xCorrSamplingRate': None,
                'xCorrGaussWid': 50e-3,
                'minStimAmp': 0,
                'unixTimeAdjust': None,
                'zeroOutsideTargetLags': False,
                'thres': 1,
                'iti': 200e-3,
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    extraSynchOffset = -1.5
    synchInfo['ins'][1][0].update({
        'timeRanges': [(58.1 + extraSynchOffset, 62.1 + extraSynchOffset,)],
        'unixTimeAdjust': extraSynchOffset * (-1)})
    synchInfo['ins'][2][0].update({
        'timeRanges': [(48.9 + extraSynchOffset, 52.9 + extraSynchOffset,)],
        'unixTimeAdjust': extraSynchOffset * (-1)})
    synchInfo['ins'][3][0].update({
        'timeRanges': [(49.2 + extraSynchOffset, 53.2 + extraSynchOffset,)],
        'unixTimeAdjust': extraSynchOffset * (-1)})
    #
    synchInfo['ins'][4][0].update({
        'timeRanges': [(141.2 + extraSynchOffset, 145.2 + extraSynchOffset)],
        'unixTimeAdjust': extraSynchOffset * (-1)})
    synchInfo['ins'][4][1].update({
        'timeRanges': [(65.7 + extraSynchOffset, 69.7 + extraSynchOffset)],
        'unixTimeAdjust': extraSynchOffset * (-1)})
    ############################################################
    synchInfo['nsp'] = {
        # per block
        i: {
            #  per trialSegment
            j: {
                'timeRanges': None, 'keepIndex': slice(None),
                'usedTENSPulses': False,
                'synchChanName': [fullRigInputs['tapSync']], 'iti': 200e-3,
                'synchByXCorrTapDetectSignal': False,
                'zScoreTapDetection': True,
                'trigFinder': 'getThresholdCrossings',
                'unixTimeAdjust': None,
                'minAnalogValue': None, 'thres': 7}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    ############################################################
    # manually add special instructions, e.g
    synchInfo['nsp'][1][0].update({'timeRanges': [(118.6, 122.6)]})
    synchInfo['nsp'][2][0].update({'timeRanges': [(77.1, 81.1)]})
    synchInfo['nsp'][3][0].update({'timeRanges': [(140.0, 144.0)]})
    #
    synchInfo['nsp'][4][0].update({'timeRanges': [(84.9, 88.9)]})
    synchInfo['nsp'][4][1].update({'timeRanges': [(1143.7, 1147.7)]})
    ############################################################
    #  overrideSegmentsForTapSync
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a Block
        #  1: {0: 'coarse'},  # e.g. for ins session 0, use the coarse alignment based on system unix time
        #  2: {2: 1},  # e.g. for ins session 2, use the alignment of ins session 1
        #
        }
    #
    # options for stim artifact detection
    stimDetectOverrideStartTimes = {
        #  each key is a Block
        1: None,
        # 2: [96.43, 191.689],
    }
    ####################
    detectStim = True
    fractionForRollover = 0.2
    stimDetectThresDefault = 500
    stimDetectChansDefault = ['ins_td0', 'ins_td2']
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
        
    triFolderSourceBase = 1
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [2, 3, 4]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901301000-Murdoc': [1],
        }

    # Options relevant to the classifcation of proprio trials
    # movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    # movementSizeBinLabels = ['XS', 'S', 'M', 'L', 'XL']
    movementSizeBins = [0, 0.25, 1.25, 1.5]
    movementSizeBinLabels = ['S', 'M', 'L']
    #
    alignTimeBoundsLookup = {
        # e.g.
        #  per trial
        #2: [
        #    #  per trialSegment
        #    [238, 1198],
        #    ],
        }
    spikeSortingOpts = {
        'utah': {
            'asigNameList': [
                [
                    'utah{:d}'.format(i)
                    for i in range(1, 97)
                    if i not in [14, 17, 28, 40, 50, 60, 82]
                    ]
                ],
            'ainpNameList': [
                'ainp{:d}'.format(i)
                for i in range(1, 17)
            ],
            'electrodeMapPath': './Utah_Murdoc.cmp',
            'rawBlockName': 'utah',
            ############################################################
            'excludeChans': [],  # CHECK
            'prbOpts': dict(
                contactSpacing=400,
                groupIn={
                    'xcoords': np.arange(-.1, 10.1, 1),
                    'ycoords': np.arange(-.1, 10.1, 1)}),
            'previewDuration': 480,
            'previewOffset': 0,
            'interpolateOutliers': True,
            'outlierThreshold': 1 - 1e-6,
            'shape_distance_threshold': None,
            'shape_boundary_threshold': None,
            'energy_reduction_threshold': 0.25,
            'make_classifier': True,
            'refit_projector': True,
            'n_max_peeler_passes': 2,
            'confidence_threshold': .5,
            'refractory_period': 2e-3,
            'triFolderSource': {
                'exp': experimentName, 'block': 3,
                'nameSuffix': 'spike_preview'},
            'triFolderDest': [
                {
                    'exp': experimentName, 'block': i,
                    'nameSuffix': 'mean_subtracted'}
                for i in blockExperimentTypeLookup.keys()]
        }
    }
    #
    expIteratorOpts = {
        'ca': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'cb': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'cc': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'ccm': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'ccs': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'cd': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'ra': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'rb': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'rc': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'rd': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        're': {
            'experimentsToAssemble': {
                experimentName: [],
                }
            },
        'pa': {
            'experimentsToAssemble': {
                '201901301000-Murdoc': [1],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                '201901301000-Murdoc': [1],
                }
            },
        'na': {
            'experimentsToAssemble': {
                '201901301000-Murdoc': [1],
                }
            }
        }
    return locals()