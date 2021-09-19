import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio-miniRC'
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
    
    experimentName = '201901261000-Murdoc'
    deviceName = 'DeviceNPC700373H'
    subjectName = 'Murdoc'
    
    jsonSessionNames = {
        #  per trial
        1: [
            'Session1548517963713', 'Session1548518261171',
            'Session1548518496294', 'Session1548518727243',
            'Session1548518982240'
            ],
        2: ['Session1548520562275'],
        3: ['Session1548521956580'],
        4: ['Session1548524126669', ],
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': [
                    'ins_accx', 'ins_accy',
                    'ins_accz', 'ins_accinertia'],
                'synchStimUnitName': None,
                'synchByXCorrTapDetectSignal': False,
                'xCorrSamplingRate': None,
                'xCorrGaussWid': 10e-3,
                'minStimAmp': 0,
                'unixTimeAdjust': None,
                'zeroOutsideTargetLags': False,
                'thres': 5,
                'iti': 300e-3,  # not tens taps!
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    synchInfo['ins'][1][0].update({
        'timeRanges': [(67.7, 71.7)],
        'unixTimeAdjust': None,})
    synchInfo['ins'][1][1].update({
        'timeRanges': [(47.7, 51.7)],
        'unixTimeAdjust': None})
    synchInfo['ins'][1][2].update({
        'timeRanges': [(44.1, 48.1)],
        'unixTimeAdjust': None})
    synchInfo['ins'][1][3].update({
        'timeRanges': [(45.3, 49.3)],
        'unixTimeAdjust': None})
    synchInfo['ins'][1][4].update({
        'timeRanges': [(43.3, 47.3)],
        'unixTimeAdjust': None})
    synchInfo['ins'][2][0].update({
        'timeRanges': [(42.2, 46.2)],
        'unixTimeAdjust': None})
    synchInfo['ins'][3][0].update({
        'timeRanges': [(37.4, 41.4)],
        'unixTimeAdjust': None})
    synchInfo['ins'][4][0].update({
        'timeRanges': [(67.0, 71.0)],
        'unixTimeAdjust': None})
    #
    #
    ############################################################
    ############################################################
    synchInfo['nsp'] = {
        # per block
        i: {
            #  per trialSegment
            j: {
                'timeRanges': None, 'keepIndex': slice(None),
                'usedTENSPulses': False,
                'synchChanName': [fullRigInputs['tapSync']], 'iti': 300e-3,
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
    ############################################################
    # manually add special instructions, e.g
    synchInfo['nsp'][1][0].update({'timeRanges': [(566.8, 570.8)]})
    synchInfo['nsp'][1][1].update({'timeRanges': [(844.1, 848.1)]})
    synchInfo['nsp'][1][2].update({'timeRanges': [(1075.7, 1079.7)]})
    synchInfo['nsp'][1][3].update({'timeRanges': [(1307.8, 1311.8)]})
    synchInfo['nsp'][1][4].update({'timeRanges': [(1560.8, 1564.8)]})
    synchInfo['nsp'][2][0].update({'timeRanges': [(132.6, 136.6)]})
    synchInfo['nsp'][3][0].update({'timeRanges': [(109.9, 113.9)]})
    synchInfo['nsp'][4][0].update({'timeRanges': [(937.6, 941.6)]})
    #
    #
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
    #########################################
    # options for stim artifact detection
    detectStim = True
    stimDetectThresDefault = 1000
    stimDetectChansDefault = ['ins_td2', 'ins_td3']
    stimDetectOverrideStartTimes = {
        1: None,
        2: None,
        3: None,
        4: None,
        # 4: [
        #     402.074, 404.084, 406.084, 408.094, 410.114, 412.124,
        #     414.144, 416.164, 418.194, 420.204, 422.224, 424.244,
        #     426.264, 428.284, 430.304],
    }
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': ['ins_td2'], 'thres': 1000, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td3'], 'thres': 1000, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 2000, 'useForSlotDetection': True},
            3: {'detectChannels': stimDetectChansDefault, 'thres': 1000, 'useForSlotDetection': True}
        },
        1: {
            # program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 500, 'useForSlotDetection': True},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td3'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901240900-Murdoc': [2],
        '201901251000-Murdoc': [1, 2],
        '201901261000-Murdoc': [1, 2, 3, 4],
        }

    # movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    # movementSizeBinLabels = ['XS', 'S', 'M', 'L', 'XL']
    movementSizeBins = [0, 0.25, 1.25, 1.5]
    movementSizeBinLabels = ['S', 'M', 'L']
    alignTimeBoundsLookup = {
        }
    motorEncoderBoundsLookup = {
        1: [
            [860, 882], [1092, 1149.8],
            [1408, 1476], [1589, 2561]
        ],
        2: [
            [159, 989]
        ],
        3: [
            [131, 1085.8]
        ]
    }
    dropMotionRounds = {
        1: [0, 1, 2]
        }
    ##############
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
                '201901240900-Murdoc': [2],
                '201901251000-Murdoc': [1, 2],
                '201901261000-Murdoc': [1, 2, 3, 4],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                '201901240900-Murdoc': [2],
                '201901251000-Murdoc': [1, 2],
                '201901261000-Murdoc': [1, 2, 3, 4],
                }
            },
        'na': {
            'experimentsToAssemble': {
                '201901240900-Murdoc': [2],
                '201901251000-Murdoc': [1, 2],
                '201901261000-Murdoc': [1, 2, 3],
                }
            }
        }
    
    return locals()
