import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio',
        5: 'proprio-miniRC',
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
    miniRCRigInputs = {
        'tapSync': 'ainp7',
        'simiTrigs': 'ainp8'
        }
    RCRigInputs = {
        'kinectSync': 'ainp16',
        }

    experimentName = '201902031100-Murdoc'
    deviceName = 'DeviceNPC700373H'
    subjectName = 'Murdoc'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1549212037478'],
        2: ['Session1549213456453', 'Session1549214214738'],
        3: ['Session1549215485078'],
        4: ['Session1549217011561'],
        5: ['Session1549219242904', 'Session1549220517478', 'Session1549221267205']
        }

    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': ['ins_td0', 'ins_td2'],
                'synchStimUnitName': None,
                'synchByXCorrTapDetectSignal': False,
                'xCorrSamplingRate': None,
                'xCorrGaussWid': 30e-3,
                'minStimAmp': 0,
                'unixTimeAdjust': None,
                'zeroOutsideTargetLags': False,
                'thres': 5,
                'iti': 200e-3,
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    synchInfo['ins'][1][0].update({
        'timeRanges': [(49.1, 53.1)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][2][0].update({
        'timeRanges': [(48.7, 52.7)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })
    synchInfo['ins'][2][1].update({
        'timeRanges': [(36.0, 40.0)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][3][0].update({
        'timeRanges': [(44.5, 48.5)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][4][0].update({
        'timeRanges': [(46.1, 50.1)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][5][0].update({
        'timeRanges': [(51.8, 55.8)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })
    synchInfo['ins'][5][1].update({
        'timeRanges': [(54.5, 58.5)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })
    synchInfo['ins'][5][2].update({
        'timeRanges': [(83.4, 87.4)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

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
    synchInfo['nsp'][1][0].update({'timeRanges': [(123.4, 127.4)]})
    synchInfo['nsp'][2][0].update({'timeRanges': [(158.3, 162.3)]})
    synchInfo['nsp'][2][1].update({'timeRanges': [(903.9, 907.9)]})
    synchInfo['nsp'][3][0].update({'timeRanges': [(101.7, 105.7)]})
    synchInfo['nsp'][4][0].update({'timeRanges': [(111.4, 115.4)]})
    synchInfo['nsp'][5][0].update({'timeRanges': [(679.9, 683.9)]})
    synchInfo['nsp'][5][1].update({'timeRanges': [(1957.1, 1961.1)]})
    synchInfo['nsp'][5][2].update({'timeRanges': [(2735.8, 2739.8)]})
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
        experimentName: [1, 2, 3, 4, 5],
        }

    # Options relevant to the classifcation of proprio trials
    # movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    # movementSizeBinLabels = ['XS', 'S', 'M', 'L', 'XL']
    movementSizeBins = [0, 1.5]
    movementSizeBinLabels = ['M']
    #
    alignTimeBoundsLookup = {
        #  per block
        # 1: [
        #     #  per ins session
        #     [257, 552],
        #     [670, 1343],
        #     ],
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
                experimentName: [1, 2, 3, 4],
                '201902041100-Murdoc': [1, 2, 3, 4],
                '201902051100-Murdoc': [1, 2, 3, 4],
                }
            },
        'cb': {
            'experimentsToAssemble': {
                experimentName: [1, 2, 3, 4],
                '201902041100-Murdoc': [1, 2, 3, 4],
                '201902051100-Murdoc': [1, 2, 3, 4],
                }
            },
        'cc': {
            'experimentsToAssemble': {
                experimentName: [5],
                '201902041100-Murdoc': [5],
                }
            },
        'ccm': {
            'experimentsToAssemble': {
                experimentName: [1, 2, 3, 4],
                '201902041100-Murdoc': [1, 2, 3, 4],
                '201902051100-Murdoc': [1, 2, 3, 4],
                }
            },
        'ccs': {
            'experimentsToAssemble': {
                experimentName: [5],
                '201902041100-Murdoc': [5],
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
                '201902041100-Murdoc': [1, 2, 3, 4, 5],
                '201902051100-Murdoc': [1, 2, 3, 4],
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
                experimentName: [1, 2, 3, 4, 5],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                experimentName: [1, 2, 3, 4, 5],
                '201902041100-Murdoc': [1, 2, 3, 4, 5],
                '201902051100-Murdoc': [1, 2, 3, 4],
                }
            },
        'na': {
            'experimentsToAssemble': {
                experimentName: [1, 2, 3, 4, 5],
                }
            }
        }
    return locals()