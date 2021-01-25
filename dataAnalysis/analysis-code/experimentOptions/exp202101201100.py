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
    experimentName = '202101201100-Rupert'
    deviceName = 'DeviceNPC700246H'
    subjectName = 'Rupert'
    #
    jsonSessionNames = {
        #  per block
        1: [
            'Session1611161648041'
            ],
        2: [
            'Session1611162632629',
            'Session1611163616272'
            ],
        3: [
            'Session1611164507435'],
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': ['ins_td0', 'ins_td2'],
                'minStimAmp': 0,
                'thres': 5,
                'iti': 10e-3,
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    # synchInfo['ins'][3][0].update({'minStimAmp': 0})
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
                'synchChanName': ['utah_rawAverage_0'], 'iti': 10e-3,
                'minAnalogValue': None, 'thres': 3}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    ############################################################
    ############################################################
    # manually add special instructions, e.g
    # synchInfo['nsp'][2][0].update({'timeRanges': [(40, 6000)]})
    #
    # synchInfo['nsp'][3][0].update({'timeRanges': [(110, 6000)]})
    # synchInfo['nsp'][3][1].update({'timeRanges': [(685, 6000)]})
    # synchInfo['nsp'][3][2].update({'timeRanges': [(1238, 6000)]})
    # synchInfo['nsp'][3][3].update({'timeRanges': [(1695, 6000)]})
    # synchInfo['nsp'][3][4].update({'timeRanges': [(2023, 6000)]})
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
        '202101201100-Rupert': [1],
        }
    # Options relevant to the classifcation of proprio trials
    movementSizeBins = [0, 0.4, 0.8]
    movementSizeBinLabels = ['S', 'L']

    ############################################################
    ############################################################
    alignTimeBoundsLookup = None
    # alignTimeBoundsLookup = {
    #     3: [
    #         [275, 1732]
    #         ],
    #     }
    #
    motorEncoderBoundsLookup = {
        2: [
            [100, 772], [1173, 1896]
        ]
    }
    ############################################################
    ############################################################
    outlierDetectOptions = dict(
        targetEpochSize=100e-3,
        windowSize=(-.2, .8),
        twoTailed=True,
        )
    #
    minNConditionRepetitions = {
        'n': 1,
        'categories': ['amplitude', 'electrode', 'RateInHz']
        }
    spikeSortingOpts = {
        'utah': {
            'asigNameList': [
                [
                    'utah{:d}'.format(i)
                    for i in range(1, 97)]
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
    return locals()
