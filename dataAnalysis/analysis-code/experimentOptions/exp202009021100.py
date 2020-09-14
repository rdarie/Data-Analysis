import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio',
        5: 'proprio',
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
    
    experimentName = '202009021100-Rupert'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
        }

    synchInfo = {'ins': {}, 'nsp': {}}
    synchInfo['ins'][1] = {
        #  per trialSegment
        0: {
            'timeRanges': [(14, 16)],
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': [(448, 450)],
            'chan': ['ins_td0', 'ins_td2'],
            # 'chan': ['ins_accx', 'ins_accy', 'ins_accz'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][2] = {
        #  per trialSegment
        0: {
            'timeRanges': [(18, 20)],
            # 'chan': ['ins_td0', 'ins_td2'],
            'chan': ['ins_accx', 'ins_accy', 'ins_accz'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][3] = {
        #  per trialSegment
        0: {
            'timeRanges': [(20, 22)],
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][4] = {
        #  per trialSegment
        0: {
            'timeRanges': [(21.5, 23.5)],
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(1, None)
            }
        }
    synchInfo['ins'][5] = {
        #  per trialSegment
        0: {
            'timeRanges': [(22, 24)],
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': [(1047, 1049)],
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        2: {
            'timeRanges': [(1301, 1304)],
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None, 3)
            }
        }

    synchInfo['nsp'] = {
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
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {},
        4: {},
        5: {}
        }
    # options for stim artifact detection
    stimDetectOverrideStartTimes = {
        1: None,
        # 2: [96.43, 191.689],
        2: None,
        3: None,
        4: None,
        5: None,
    }
    detectStim = True
    stimDetectThresDefault = 500
    stimDetectChansDefault = ['ins_td0', 'ins_td2']
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': 250, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
        
    triFolderSourceBase = 1
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [2, 3, 4]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202009021100-Rupert': [1, 2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
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
    outlierDetectOptions = dict(
        targetEpochSize=10e-3,
        windowSize=(-1, 1),
        # conditionNames=[
        #     'electrode', 'amplitude', 'RateInHz',
        #     'pedalMovementCat', 'pedalSizeCat', 'pedalDirection'],
        conditionNames=[
            'electrode', 'amplitude', 'RateInHz'],
        twoTailed=True,
        )
    #
    minNConditionRepetitions = {
        'n': 1,
        'categories': ['amplitude', 'electrode', 'RateInHz']
        }
    spikeSortingOpts = {
        'utah': {
            'electrodeMapPath': './Utah_SN6251_002374_Rupert.cmp',
            'excludeChans': [],
            'prbOpts': dict(
                contactSpacing=400,
                groupIn={
                    'xcoords': np.arange(-.1, 10.1, 2),
                    'ycoords': np.arange(-.1, 10.1, 2)})

        },
        'nform': {
            'electrodeMapPath': './NForm_Rupert_flat_v2.map',
            'excludeChans': ['nform_20', 'nform_27', 'nform_12', 'nform_60'],
            'prbOpts': dict(
                contactSpacing=500,
                groupIn={
                    'xcoords': np.arange(-.1, 18.1, 2),
                    'ycoords': np.arange(-.1, 5.1, 2)})
        }
    }
    return locals()
