import numpy as np

def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-miniRC',
        2: 'proprio-miniRC',
        3: 'proprio-miniRC',
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
    
    experimentName = '202010271200-Rupert'
    deviceName = 'DeviceNPC700246H'
    subjectName = 'Rupert'

    jsonSessionNames = {
        #  per trial
        1: [],
        2: ['Session1603815232182', 'Session1603815636014', 'Session1603816009303'],
        3: ['Session1603816446027', 'Session1603816775688', ]
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
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
            'timeRanges': None,
            'chan': ['ins_td0'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': None,
            'chan': ['ins_td0'],
            # 'chan': ['ins_accx', 'ins_accy', 'ins_accz'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        2: {
            'timeRanges': None,
            'chan': ['ins_td0'],
            # 'chan': ['ins_accx', 'ins_accy', 'ins_accz'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][3] = {
        #  per trialSegment
        0: {
            'timeRanges': None,
            'chan': ['ins_td0', 'ins_td2'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': None,
            'chan': ['ins_td0', 'ins_td2'],
            # 'chan': ['ins_accx', 'ins_accy', 'ins_accz'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['nsp'] = {
        #  per trialSegment
        1: {
            0: {'timeRanges': None, 'keepIndex': slice(None)},
            1: {'timeRanges': None, 'keepIndex': slice(None)}
            },
        2: {
            0: {'timeRanges': None, 'keepIndex': slice(None)},
            1: {'timeRanges': None, 'keepIndex': slice(None)},
            2: {'timeRanges': None, 'keepIndex': slice(None, 3)}
            },
        3: {
            0: {'timeRanges': None, 'keepIndex': slice(None)},
            1: {'timeRanges': None, 'keepIndex': slice(None)}
            }
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {},
        }
    # options for stim artifact detection
    stimDetectOverrideStartTimes = {
        1: None,
        2: None,
        3: None,
        # 2: [96.43, 191.689],
    }
    detectStim = True
    stimDetectThresDefault = 100
    stimDetectChansDefault = ['ins_td0']
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': ['ins_td0'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td0'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td0'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202010271200-Rupert': [2, 3],
        }
    # Options relevant to the classifcation of proprio trials
    movementSizeBins = [0, 0.4, 0.8]
    movementSizeBinLabels = ['S', 'L']
    alignTimeBoundsLookup = {
        # 2: [
        #     [66, 400]
        #     ],
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
            'excludeChans': [],
            'prbOpts': dict(
                contactSpacing=400,
                groupIn={
                    'xcoords': np.arange(-.1, 10.1, 2),
                    'ycoords': np.arange(-.1, 10.1, 1)}),
            'triFolderSource': {'exp': experimentName, 'block': 2},
            'triFolderDest': [
                {'exp': experimentName, 'block': i}
                for i in [3]]
        },
        'nform': {
            'asigNameList': [
                [
                    'nform_{:02d}'.format(i)
                    for i in range(1, 33)
                    if i not in [20, 27]],
                [
                    'nform_{:02d}'.format(i)
                    for i in range(33, 65)
                    if i not in []]
                ],
            'ainpNameList': ['analog 1'],
            'electrodeMapPath': './NForm_Rupert_flat_v2.map',
            'excludeChans': ['nform_20', 'nform_27'],
            'prbOpts': dict(
                contactSpacing=500,
                groupIn={
                    'xcoords': np.arange(-.1, 18.1, 1),
                    'ycoords': np.arange(-.1, 5.1, 2)}),
            'triFolderSource': {'exp': experimentName, 'block': 1},
            'triFolderDest': [
                {'exp': experimentName, 'block': i}
                for i in [2, 3]]
        }
    }
    return locals()
