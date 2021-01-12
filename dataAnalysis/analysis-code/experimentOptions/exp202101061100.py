import numpy as np


def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-miniRC',
        2: 'proprio',
        3: 'proprio',
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
    
    experimentName = '202101061100-Rupert'
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
        1: ['Session1609950588323', 'Session1609951132662'],
        2: ['Session1609951940258', 'Session1609952078942', 'Session1609952463973'],
        3: ['Session1609952950984'],
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}}
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'chan': ['ins_td0'],
                'thres': 5,
                'iti': 50e-3,
                'keepIndex': slice(-5, None)
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
            j: {'timeRanges': None, 'keepIndex': slice(-5, None)}
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
        2: {2: 1},
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
            0: {'detectChannels': ['ins_td2'], 'thres': 250, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td0', 'ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202101061100-Rupert': [1],
        }
    # Options relevant to the classifcation of proprio trials
    movementSizeBins = [0, 0.4, 0.8]
    movementSizeBinLabels = ['S', 'L']
    alignTimeBoundsLookup = None
    # alignTimeBoundsLookup = {
    #     3: [
    #         [275, 1732]
    #         ],
    #     }
    outlierDetectOptions = dict(
        targetEpochSize=100e-3,
        windowSize=(-.2, .8),
        conditionNames=[
            'electrode', 'amplitude', 'RateInHz',
            'pedalMovementCat', 'pedalSizeCat', 'pedalDirection'],
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
                    for i in range(1, 97)]
                ],
            'ainpNameList': [
                'ainp{:d}'.format(i)
                for i in range(1, 17)
            ],
            'electrodeMapPath': './Utah_SN6251_002374_Rupert.cmp',
            'rawBlockName': 'utah',
            'excludeChans': ['utah8'],
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
            'energy_reduction_threshold': .25,
            'make_classifier': True,
            'refit_projector': False,
            'n_max_peeler_passes': 2,
            'confidence_threshold': .5,
            'refractory_period': None,
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
