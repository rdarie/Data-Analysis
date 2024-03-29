import numpy as np

def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-miniRC',
        2: 'proprio-miniRC',
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
    
    experimentName = '202012181200-Rupert'
    deviceName = 'DeviceNPC700246H'
    subjectName = 'Rupert'

    # spikeSortingFilterOpts = {
    #     'bandstop': {
    #         'Wn': 60,
    #         'nHarmonics': 1,
    #         'Q': 20,
    #         'N': 1,
    #         'rp': 1,
    #         'btype': 'bandstop',
    #         'ftype': 'butter'
    #     },
    #     'low': {
    #         'Wn': 3000,
    #         'N': 4,
    #         'btype': 'low',
    #         'ftype': 'butter'
    #     },
    #     'high': {
    #         'Wn': 300,
    #         'N': 4,
    #         'btype': 'high',
    #         'ftype': 'butter'
    #     }
    # }
    jsonSessionNames = {
        #  per trial
        1: [],
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'chan': ['ins_td2'],
                'thres': 5,
                'iti': 50e-3,
                'keepIndex': slice(None)
                }
    # manually add special instructions
    # #synchInfo['ins'][1][1] = {
    # #    'timeRanges': None,
    # #    'chan': ['ins_td2'],
    # #    'thres': 5,
    # #    'iti': 50e-3,
    # #    'keepIndex': slice(None)
    # #    }
    synchInfo['nsp'] = {
        #  per trialSegment
        i: {
            j: {'timeRanges': None, 'keepIndex': slice(None, 3)}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    # manually add special instructions
    # synchInfo['nsp'][1][1] = {'timeRanges': None, 'keepIndex': slice(3, None)}
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a Block
        1: {},
        2: {},
        }
    # options for stim artifact detection
    stimDetectOverrideStartTimes = {
        #  each key is a Block
        1: None,
        # 2: [96.43, 191.689],
    }
    detectStim = True
    stimDetectThresDefault = 100
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
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202012181200-Rupert': [1, 2],
        }
    # Options relevant to the classifcation of proprio trials
    movementSizeBins = [0, 0.4, 0.8]
    movementSizeBinLabels = ['S', 'L']
    alignTimeBoundsLookup = None
    # alignTimeBoundsLookup = {
    #     1: [
    #             [42, 1241]
    #         ],
    #     2: [
    #         [25, 430]
    #         ],
    #     }
    outlierDetectOptions = dict(
        targetEpochSize=100e-3,
        windowSize=(-.2, .8),
        # conditionNames=[
        #     'electrode', 'amplitude', 'RateInHz',
        #     'pedalMovementCat', 'pedalSizeCat', 'pedalDirection'],
        conditionNames=[
            'electrode', 'amplitude', 'RateInHz'],
        twoTailed=False,
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
            'excludeChans': [],
            'prbOpts': dict(
                contactSpacing=400,
                groupIn={
                    'xcoords': np.arange(-.1, 10.1, 1),
                    'ycoords': np.arange(-.1, 10.1, 1)}),
            'previewDuration': 600,
            'previewOffset': 0,
            'interpolateOutliers': True,
            'outlierThreshold': 1 - 1e-6,
            'shape_distance_threshold': None,
            'shape_boundary_threshold': None,
            'energy_reduction_threshold': 0.1,
            'confidence_threshold': 0.5,
            'refractory_period': None,
            'triFolderSource': {
                'exp': experimentName, 'block': 1,
                'nameSuffix': 'spike_preview'},
            'triFolderDest': [
                {
                    'exp': experimentName, 'block': i,
                    'nameSuffix': 'mean_subtracted'}
                for i in [1, 2]]
        },
        'nform': {
            'asigNameList': [
                [
                    'nform_{:02d}'.format(i)
                    for i in range(33, 65)
                    if i not in [52, 59]]
                ],
            'ainpNameList': ['analog 1'],
            'electrodeMapPath': './NForm_Rupert_flat_1port_secondHalf.map',
            'rawBlockName': 'nform',
            'excludeChans': ['nform_52', 'nform_59'],
            'prbOpts': dict(
                contactSpacing=500,
                groupIn={
                    'xcoords': np.arange(-.1, 18.1, 1),
                    'ycoords': np.arange(-.1, 5.1, 1)}),
            'previewDuration': 600,
            'previewOffset': 0,
            'interpolateOutliers': True,
            'outlierThreshold': .999,
            'shape_distance_threshold': None,
            'shape_boundary_threshold': None,
            'energy_reduction_threshold': 0.5,
            'confidence_threshold': 0.5,
            'refractory_period': None,
            'triFolderSource': {
                'exp': experimentName, 'block': 2,
                'nameSuffix': 'spike_preview'},
            'triFolderDest': [
                {
                    'exp': experimentName, 'block': i,
                    'nameSuffix': 'mean_subtracted'}
                for i in [2, 3]]
        }
    }
    return locals()
