import numpy as np


def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-miniRC',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio-motionOnly',
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
    
    experimentName = '202101211100-Rupert'
    deviceName = 'DeviceNPC700246H'
    subjectName = 'Rupert'
    #
    jsonSessionNames = {
        #  per block
        1: ['Session1611246044886', 'Session1611246220465'],
        2: ['Session1611247033380', 'Session1611247376867'],
        3: [
            'Session1611248043534', 'Session1611248599896',
            'Session1611248732385', 'Session1611249056228',
            'Session1611249262364', 'Session1611249767605'
            ],
        4: ['Session1611250728338']
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': ['ins_td0', 'ins_td2'],
                'synchStimUnitName': ['g0p0#0'],
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
    synchInfo['ins'][2][0].update({'unixTimeAdjust': 6.4})
    # #synchInfo['ins'][1][1] = {
    # #    'timeRanges': None,
    # #    'chan': ['ins_td2'],
    # #    'thres': 5,
    # #    'iti': 50e-3,
    # #    'keepIndex': slice(None)
    # #    }
    # synchInfo['ins'][1][0].update({
    #     'minStimAmp': 700})
    # synchInfo['ins'][4][0].update({
    #    'synchStimUnitName': None,
    #     'synchByXCorrTapDetectSignal': True})
    ############################################################
    ############################################################
    synchInfo['nsp'] = {
        # per block
        i: {
            #  per trialSegment
            j: {
                'timeRanges': None, 'keepIndex': slice(None),
                'synchChanName': ['utah_artifact_0'], 'iti': 10e-3,
                'synchByXCorrTapDetectSignal': False,
                'unixTimeAdjust': None,
                'minAnalogValue': None, 'thres': 5}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    ############################################################
    ############################################################
    # manually add special instructions, e.g
    synchInfo['nsp'][1][0].update({
        'timeRanges': [(70, 9999)],
        # 'thres': 100
        })
    synchInfo['nsp'][1][1].update({
        'timeRanges': [(221, 9999)]
        })
    #
    synchInfo['nsp'][2][0].update({'timeRanges': [(90, 9999)]})
    synchInfo['nsp'][2][1].update({'timeRanges': [(417, 9999)]})
    #
    synchInfo['nsp'][3][0].update({'timeRanges': [(38, 9999)]})
    synchInfo['nsp'][3][1].update({'timeRanges': [(598, 9999)]})
    synchInfo['nsp'][3][2].update({'timeRanges': [(721, 9999)]})
    synchInfo['nsp'][3][3].update({'timeRanges': [(1045, 9999)]})
    synchInfo['nsp'][3][4].update({'timeRanges': [(1257, 9999)]})
    synchInfo['nsp'][3][5].update({'timeRanges': [(1941, 9999)]})
    #
    synchInfo['nsp'][4][0].update({
        'timeRanges': [(46, 9999)],
        'synchByXCorrTapDetectSignal': True})
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
        '202101211100-Rupert': [2, 3],
        }
    # Options relevant to the classifcation of proprio trials
    movementSizeBins = [0, 0.6, 1]
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
    motorEncoderBoundsLookup = None
    # motorEncoderBoundsLookup = {
    #     3: [
    #         [100, 299], [730, 2290]
    #     ]
    # }
    pedalPositionZeroEpochs = None
    dropMotionRounds = {
        3: [83]
    }

    ############################################################
    ############################################################
    outlierDetectOptions = dict(
        targetEpochSize=100e-3,
        windowSize=(-.2, .8),
        conditionNames=[
            'electrode', 'amplitude', 'RateInHz',
            'pedalSizeCat', 'pedalDirection'],
        # conditionNames=[
        #     'electrode', 'amplitude', 'RateInHz'],
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
                    for i in range(1, 97) if i not in [39, 89]]
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
                'exp': experimentName, 'block': 1,
                'nameSuffix': 'spike_preview'},
            'triFolderDest': [
                {
                    'exp': experimentName, 'block': i,
                    'nameSuffix': 'mean_subtracted'}
                for i in [1,  2]]
        }
    }
    expIteratorOpts = {
        'ca': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [2, 3, 4],
                }
            },
        'cb': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [2, 3, 4],
                }
            },
        'cc': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3],
                }
            },
        'ccm': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [2, 3],
                }
            },
        'ccs': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1],
                }
            },
        'cd': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3, 4],
                }
            },
        'ra': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3, 4],
                }
            },
        'rb': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3, 4],
                }
            },
        'pa': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3, 4],
                }
            },
        'ma': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3, 4],
                }
            },
        'na': {
            'experimentsToAssemble': {
                '202101211100-Rupert': [1, 2, 3, 4],
                }
            }
        }
    return locals()
