import numpy as np


def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-miniRC',
        2: 'proprio',
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
    
    experimentName = '202101141100-Rupert'
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
        #  per block
        1: [
            'Session1610641839881', 'Session1610642040064',
            'Session1610642516780', 'Session1610642924085'],
        2: [
            'Session1610643294975', 'Session1610643999548'],
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
                'iti': 52.2e-3,
                'keepIndex': slice(-5, None)
                }
    # manually add special instructions, e.g.
    # #synchInfo['ins'][1][1] = {
    # #    'timeRanges': None,
    # #    'chan': ['ins_td2'],
    # #    'thres': 5,
    # #    'iti': 50e-3,
    # #    'keepIndex': slice(None)
    # #    }
    synchInfo['ins'][2][0].update({
        'iti': 20e-3, 'keepIndex': slice(None, 3),
    })
    synchInfo['ins'][2][1].update({
        'iti': 20e-3, 'keepIndex': slice(None, 3),
    })
    synchInfo['nsp'] = {
        # per block
        i: {
            #  per trialSegment
            j: {
                'timeRanges': None, 'keepIndex': slice(-5, None),
                'synchChanName': miniRCRigInputs['tapSync'], 'iti': 52e-3,
                'minAnalogValue': 100, 'thres': 3}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    # manually add special instructions, e.g.
    #  synchInfo['nsp'][1][1] = {'timeRanges': None, 'keepIndex': slice(3, None)}
    synchInfo['nsp'][2][0].update({
        'timeRanges': (136, 138),
        'synchChanName': 'utah_rawAverage_0',
        'iti': 20e-3, 'keepIndex': slice(None, 3),
        'minAnalogValue': None, 'thres': 150,
        })
    synchInfo['nsp'][2][1].update({
        'timeRanges': (956, 958),
        'synchChanName': 'utah_rawAverage_0',
        'iti': 20e-3, 'keepIndex': slice(None, 3),
        'minAnalogValue': None, 'thres': 150
        })
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
    ###############################################################
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
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202101141100-Rupert': [1],
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
            'excludeChans': ['utah69', 'utah44', 'utah54'],  # CHECK
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
    return locals()
