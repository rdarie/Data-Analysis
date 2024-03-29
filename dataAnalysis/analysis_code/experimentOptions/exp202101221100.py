import numpy as np


def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-miniRC',
        2: 'proprio',
        3: 'proprio-motionOnly',
        }
    fullRigInputs = {
        'A+': 'ainp12',
        'B+': 'ainp11',
        'Z+': 'ainp3',
        'A-': 'ainp9',
        'B-': 'ainp10',
        'Z-': 'ainp4',
        'rightBut': 'ainp5',
        'leftBut': 'ainp6',
        'rightLED': 'ainp7',
        'leftLED': 'ainp8',
        'simiTrigs': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        'tapSync': 'ainp1',
        }
    miniRCRigInputs = {
        'tapSync': 'ainp1',
        'simiTrigs': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        }
    RCRigInputs = {
        'tapSync': 'ainp1',
        'kinectSync': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        }
    experimentName = '202101221100-Rupert'
    deviceName = 'DeviceNPC700246H'
    subjectName = 'Rupert'
    #
    jsonSessionNames = {
        #  per block
        1: [
            'Session1611332995649',
            'Session1611333273122',
            'Session1611333553519',
            'Session1611334040372',
            'Session1611334349859'  # ???
            ],
        2: [
            'Session1611334604163', 'Session1611335022326',
            'Session1611335197637', 'Session1611335825108'
        ],
        3: ['Session1611336429421']
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': [],
                'synchStimUnitName': ['g0p0#0'],
                'zeroOutsideTargetLags': True,
                'synchByXCorrTapDetectSignal': False,
                'xCorrSamplingRate': None,
                'xCorrGaussWid': 10e-3,
                'minStimAmp': 0,
                'unixTimeAdjust': None,
                'thres': 5,
                'iti': 10e-3,
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    # synchInfo['ins'][3][0].update({'minStimAmp': 0})
    #
    synchInfo['ins'][1][0].update({'unixTimeAdjust': -3})
    synchInfo['ins'][1][1].update({'unixTimeAdjust': -3})
    # #synchInfo['ins'][1][1] = {
    # #    'timeRanges': None,
    # #    'chan': ['ins_td2'],
    # #    'thres': 5,
    # #    'iti': 50e-3,
    # #    'keepIndex': slice(None)
    # #    }
    ############################################################
    ############################################################
    synchInfo['nsp'] = {
        # per block
        i: {
            #  per trialSegment
            j: {
                'timeRanges': None, 'keepIndex': slice(None),
                'usedTENSPulses': True,
                'zScoreTapDetection': False, 'trigFinder': 'getThresholdCrossings',
                'synchChanName': ['utah_artifact_0'], 'iti': 10e-3,
                'synchByXCorrTapDetectSignal': False,
                'unixTimeAdjust': None,
                'minAnalogValue': None, 'thres': 10}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    ############################################################
    ############################################################
    # manually add special instructions, e.g
    # synchInfo['nsp'][2][0].update({'timeRanges': [(40, 9999)]})
    #
    #
    #
    #
    #
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
    detectStim = True
    fractionForRollover = 0.3
    stimDetectThresDefault = 1e6
    stimDetectChansDefault = ['ins_td0', 'ins_td2']
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202101221100-Rupert': [1, 2],
        }
    # Options relevant to the classifcation of proprio trials
    # movementSizeBins = [0, 0.6, 1]
    # movementSizeBinLabels = ['S', 'L']
    movementSizeBins = [0, 1]
    movementSizeBinLabels = ['M']

    ############################################################
    ############################################################
    alignTimeBoundsLookup = None
    # alignTimeBoundsLookup = {
    #     3: [
    #         [108, 99999]
    #         ],
    #     }
    #
    motorEncoderBoundsLookup = None
    #
    # e.g.
    # motorEncoderBoundsLookup = {
    #     2: [
    #         [180, 400], [490, 605], [650, 1215], [1280, 1750]
    #     ]
    # }
    pedalPositionZeroEpochs = {
        2: [450, 620, 1250]
    }
    dropMotionRounds = {
        2: [42, 43, 44]
    }
    ############################################################
    manualOutlierOverrideDict = {
        'motion': {
            1: [211],
            2: [99]}
        }
    ############################################################
    ############################################################
    spikeSortingOpts = {
        'utah': {
            'asigNameList': [
                [
                    'utah{:d}'.format(i)
                    for i in range(1, 97) if i not in [25, 39, 69, 89]]
                ],
            'ainpNameList': [
                'ainp{:d}'.format(i)
                for i in range(1, 17)
            ],
            'electrodeMapPath': './Utah_SN6251_002374_Rupert.cmp',
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
                for i in [1, 2, 3]]
        }
    }
    expIteratorOpts = {
        'ca': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [2, 3],
                }
            },
        'cb': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [2, 3],
                }
            },
        'cc': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2],
                }
            },
        'ccm': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [2],
                }
            },
        'ccs': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1],
                }
            },
        'cd': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2, 3],
                }
            },
        'ra': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2, 3],
                }
            },
        'rb': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2, 3],
                }
            },
        'rc': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2, 3],
                }
            },
        'pa': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2, 3],
                }
            },
        'na': {
            'experimentsToAssemble': {
                '202101221100-Rupert': [1, 2, 3],
                }
            }
        }
    return locals()
