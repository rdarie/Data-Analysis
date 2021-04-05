import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        }
    fullRigInputs = {
        'A+': 'ainp9',
        'B+': 'ainp11',
        'Z+': 'ainp3',
        'A-': 'ainp10',
        'B-': 'ainp12',
        'Z-': 'ainp4',
        'rightBut': 'ainp5',
        'leftBut': 'ainp6',
        'rightLED': 'ainp7',
        'leftLED': 'ainp8',
        'simiTrigs': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        'tapSync': 'ainp13',
        }
    miniRCRigInputs = {
        'tapSync': 'ainp13',
        'simiTrigs': 'ainp16'
        }
    RCRigInputs = {
        'kinectSync': 'ainp16',
        }
    
    experimentName = '202009101200-Rupert'
    deviceName = 'DeviceNPC700373H'
    jsonSessionNames = {
        #  per trial
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    synchInfo['nsp'][2] = {'timeRanges': [1, 986], 'chooseCrossings': slice(None)}
    synchInfo['nform'][2] = {'timeRanges': [2, 987], 'chooseCrossings': slice(None)}
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
        '202009101200-Rupert': [1, 2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        2: [
            [2, 987],
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
                    'ycoords': np.arange(-.1, 10.1, 2)})

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
                    if i not in [44]]
                ],
            'ainpNameList': ['analog 1'],
            'electrodeMapPath': './NForm_Rupert_flat_v2.map',
            'excludeChans': ['nform_20', 'nform_27', 'nform_44'],
            'prbOpts': dict(
                contactSpacing=500,
                groupIn={
                    'xcoords': np.arange(-.1, 18.1, 2),
                    'ycoords': np.arange(-.1, 5.1, 2)})
        }
    }
    return locals()
