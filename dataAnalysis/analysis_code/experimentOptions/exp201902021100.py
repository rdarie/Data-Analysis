import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio', # bad! discard;
        2: 'proprio', # bad! discard;
        3: 'proprio',
        4: 'proprio',
        5: 'proprio',
        6: 'proprio-miniRC',
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

    experimentName = '201902021100-Murdoc'
    deviceName = 'DeviceNPC700373H'
    subjectName = 'Murdoc'
    
    jsonSessionNames = {
        #  per trial
        1: [],
        2: [],
        3: ['Session1549126277709', 'Session1549127148513'],
        4: ['Session1549128140992'],
        5: ['Session1549129817710', 'Session1549131928937', 'Session1549132573173'],
        6: ['Session1549133243162'],
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
                'xCorrGaussWid': 25e-3,
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
    synchInfo['ins'][3][0].update({
        'timeRanges': [(47.4, 53.4)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })
    synchInfo['ins'][3][1].update({
        'timeRanges': [(43.7, 49.7)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][4][0].update({
        'timeRanges': [(78.6, 82.6)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][5][0].update({
        'timeRanges': [(48.2, 52.2)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })
    synchInfo['ins'][5][1].update({
        'timeRanges': [(37.4, 41.4)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })
    synchInfo['ins'][5][2].update({
        'timeRanges': [(41.9, 45.9)],
        'unixTimeAdjust': None,
        'synchChanName': ['ins_accx', 'ins_accy', 'ins_accz'],
        })

    synchInfo['ins'][6][0].update({
        'timeRanges': [(71.0, 75.0)],
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
    synchInfo['nsp'][3][0].update({'timeRanges': [(121.5, 127.5)]})
    synchInfo['nsp'][3][1].update({'timeRanges': [(988.6, 994.6)]})
    #
    synchInfo['nsp'][4][0].update({'timeRanges': [(187.7, 191.7)]})
    #
    synchInfo['nsp'][5][0].update({'timeRanges': [(109.2, 113.2)]})
    synchInfo['nsp'][5][1].update({'timeRanges': [(2209.6, 2213.6)]})
    synchInfo['nsp'][5][2].update({'timeRanges': [(2858.3, 2862.3)]})
    #
    synchInfo['nsp'][6][0].update({'timeRanges': [(214.0, 218.0)]})
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
    fractionForRollover = 0.3
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
        experimentName: [3, 4, 5, 6],
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
    pedalPositionZeroEpochs = None
    dropMotionRounds = {
        5: [216]
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
                experimentName: [3, 4, 5, 6],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                experimentName: [3, 4, 5, 6],
                }
            },
        'na': {
            'experimentsToAssemble': {
                experimentName: [3, 4, 5, 6],
                }
            }
        }
    return locals()