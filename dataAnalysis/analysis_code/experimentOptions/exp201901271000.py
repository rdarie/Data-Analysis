import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio',
        5: 'proprio-miniRC'
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

    experimentName = '201901271000-Murdoc'
    deviceName = 'DeviceNPC700373H'
    subjectName = 'Murdoc'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548605159361', 'Session1548605586940'],
        2: ['Session1548606783930'],
        3: ['Session1548608122565'],
        4: ['Session1548609521574'],
        5: [
            'Session1548611405556', # programs 2 and 3 match 2019-01-27 config
            # 'Session1548612434879',
            'Session1548612688167'
            ]
        }

    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': ['ins_td0'],
                'synchStimUnitName': None,
                'synchByXCorrTapDetectSignal': False,
                'xCorrSamplingRate': None,
                'xCorrGaussWid': 10e-3,
                'minStimAmp': 0,
                'unixTimeAdjust': None,
                'zeroOutsideTargetLags': False,
                'thres': 5,
                'iti': 100e-3,
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    synchInfo['ins'][1][0].update({
        'timeRanges': [(37.9, 43.9)],
        'unixTimeAdjust': None})
    synchInfo['ins'][1][1].update({
        'timeRanges': [(44.8, 50.8)],
        'unixTimeAdjust': None})
    synchInfo['ins'][2][0].update({
        'timeRanges': [(51.4, 55.4)],
        'unixTimeAdjust': 2.})
    synchInfo['ins'][3][0].update({
        'timeRanges': [(51.0, 55.0)],
        'unixTimeAdjust': 2.})
    synchInfo['ins'][4][0].update({
        'timeRanges': [(54.4, 57.4)],
        'unixTimeAdjust': 2.})
    #
    synchInfo['ins'][5][0].update({
        'timeRanges': [(53.5, 59.5)],
        'unixTimeAdjust': 2.})
    synchInfo['ins'][5][1].update({
        'timeRanges': [(59.4, 65.4)],
        'unixTimeAdjust': 2.})
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
                'synchChanName': [fullRigInputs['tapSync']], 'iti': 100e-3,
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
    synchInfo['nsp'][1][0].update({'timeRanges': [(208.9, 214.9)]})
    synchInfo['nsp'][1][1].update({'timeRanges': [(643.4, 649.4)]})
    #
    synchInfo['nsp'][2][0].update({'timeRanges': [(202.2, 206.2)]})
    synchInfo['nsp'][3][0].update({'timeRanges': [(104.1, 108.1)]})
    synchInfo['nsp'][4][0].update({'timeRanges': [(140., 144.)]})
    #
    synchInfo['nsp'][5][0].update({'timeRanges': [(133.1, 139.1)]})
    synchInfo['nsp'][5][1].update({'timeRanges': [(1421.6, 1427.6)]})
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
    ##########################

    '''synchInfo['nsp'] = {
        #  per trialSegment
        1: {
            0: {'timeRanges': [212, 214], 'keepIndex': slice(None)},
            1: {'timeRanges': [647, 649], 'keepIndex': slice(None)}
            },
        2: {
            0: {'timeRanges': [203, 205], 'keepIndex': slice(None)}
            },
        3: {
            0: {'timeRanges': [105, 107], 'keepIndex': slice(None)}
            },
        4: {
            0: {'timeRanges': [141, 143], 'keepIndex': slice(None)}
            },
        5: {
            0: {'timeRanges': [145, 147], 'keepIndex': slice(None)},
            1: {'timeRanges': [1169, 1171], 'keepIndex': slice(None)},
            2: {'timeRanges': [1423, 1426], 'keepIndex': slice(None)}
            }
        }'''
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
            1: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': 125, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
        
    triFolderSourceBase = 1
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [2, 3, 4]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901261000-Murdoc': [4],
        '201901271000-Murdoc': [1, 2, 3, 4],
        }

    # Options relevant to the classifcation of proprio trials
    # movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    # movementSizeBinLabels = ['XS', 'S', 'M', 'L', 'XL']
    movementSizeBins = [0, 1.5]
    movementSizeBinLabels = ['M']
    #
    alignTimeBoundsLookup = {
        1: [
            [257, 552],
            [670, 1343],
            ],
        #  per trial
        2: [
            #  per trialSegment
            [238, 1198],
            ],
        3: [
            [171, 1050]
            ],
        4: [
            [185, 1501],
            ],
        5: [
            [100, 2010]
        ]
        }
    #  outlierDetectOptions = dict(
    #      targetEpochSize=10e-3,
    #      windowSize=(-1, 1),
    #      # conditionNames=[
    #      #     'electrode', 'amplitude', 'RateInHz',
    #      #     'pedalMovementCat', 'pedalSizeCat', 'pedalDirection'],
    #      conditionNames=[
    #          'electrode', 'amplitude', 'RateInHz'],
    #      twoTailed=True,
    #      )
    #
    # minNConditionRepetitions = {
    #     'n': 1,
    #     'categories': ['amplitude', 'electrode', 'RateInHz']
    #     }
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
                '201901251000-Murdoc': [2],
                '201901261000-Murdoc': [1, 2, 3],
                experimentName: [1, 2, 3, 4],
                }
            },
        'cb': {
            'experimentsToAssemble': {
                '201901251000-Murdoc': [2],
                '201901261000-Murdoc': [1, 2, 3],
                experimentName: [1, 2, 3, 4],
                }
            },
        'cc': {
            'experimentsToAssemble': {
                }
            },
        'ccm': {
            'experimentsToAssemble': {
                '201901251000-Murdoc': [2],
                '201901261000-Murdoc': [1, 2, 3],
                experimentName: [1, 2, 3, 4],
                }
            },
        'ccs': {
            'experimentsToAssemble': {
                '201901251000-Murdoc': [1],
                '201901261000-Murdoc': [4],
                experimentName: [5],
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
                # '201901251000-Murdoc': [1, 2],
                # '201901261000-Murdoc': [1, 2, 3, 4],
                experimentName: [1, 2, 3, 4, 5],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                '201901251000-Murdoc': [1, 2],
                '201901261000-Murdoc': [1, 2, 3, 4],
                experimentName: [1, 2, 3, 4],
                }
            },
        'na': {
            'experimentsToAssemble': {
                # '201901261000-Murdoc': [4],
                experimentName: [1, 2, 3, 4, 5],
                }
            }
        }
    return locals()