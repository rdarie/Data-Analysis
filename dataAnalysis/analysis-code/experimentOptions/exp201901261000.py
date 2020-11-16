import os
import numpy as np

def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        4: 'proprio-miniRC'
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
    
    experimentName = '201901261000-Proprio'
    deviceName = 'DeviceNPC700373H'
    subjectName = 'Murdoc'
    
    jsonSessionNames = {
        #  per trial
        1: [
            'Session1548517963713', 'Session1548518261171',
            'Session1548518496294', 'Session1548518727243',
            'Session1548518982240'],
        2: ['Session1548520562275'],
        3: ['Session1548521956580'],
        4: ['Session1548524126669', ],
        }

    synchInfo = {'ins': {}, 'nsp': {}}
    synchInfo['ins'][1] = {
        #  TODO
        #  per trialSegment
        0: {
            'timeRanges': [(14, 16)],
            'tdChan': 'ins_td0',
            'tdThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': [(448, 450)],
            'tdChan': 'ins_td0',
            'tdThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][2] = {
        #  TODO
        #  per trialSegment
        0: {
            'timeRanges': [(18, 20)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][3] = {
        #  TODO
        #  per trialSegment
        0: {
            'timeRanges': [(20, 22)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][4] = {
        #  per trialSegment
        0: {
            # 'timeRanges': [(15.5, 17.5)],
            'timeRanges': None,
            'chan': ['ins_td2', 'ins_td3'],
            'thres': 5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }

    synchInfo['nsp'] = {
        #  per trialSegment
        #  TODO
        1: {
            0: {'timeRanges': [212, 214], 'keepIndex': slice(None)},
            1: {'timeRanges': [647, 649], 'keepIndex': slice(None)}
            },
        #  TODO
        2: {
            0: {'timeRanges': [203, 205], 'keepIndex': slice(None)}
            },
        #  TODO
        3: {
            0: {'timeRanges': [105, 107], 'keepIndex': slice(None)}
            },
        4: {
            # 0: {'timeRanges': [939, 941], 'keepIndex': slice(None)}
            0: {'timeRanges': None, 'keepIndex': slice(None)}
            },
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {},
        4: {}
        }
    # options for stim artifact detection
    detectStim = True
    stimDetectThresDefault = 1000
    stimDetectChansDefault = ['ins_td2', 'ins_td3']
    stimDetectOverrideStartTimes = {
        1: None,
        2: None,
        3: None,
        4: None,
        # 4: [
        #     402.074, 404.084, 406.084, 408.094, 410.114, 412.124,
        #     414.144, 416.164, 418.194, 420.204, 422.224, 424.244,
        #     426.264, 428.284, 430.304],
    }
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': ['ins_td2'], 'thres': 1000, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td3'], 'thres': 1000, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 2000, 'useForSlotDetection': True},
            3: {'detectChannels': stimDetectChansDefault, 'thres': 1000, 'useForSlotDetection': True}
        },
        1: {
            # program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 500, 'useForSlotDetection': True},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': ['ins_td3'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901261000-Proprio': [4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    movementSizeBinLabels = ['XS', 'S', 'M', 'L', 'XL']
    alignTimeBoundsLookup = {
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
        'n': 3,
        'categories': ['amplitude', 'electrode', 'RateInHz']
        }
    #
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
            'electrodeMapPath': './Utah_Murdoc.cmp',
            'rawBlockName': 'Block',
            'excludeChans': [],
            'prbOpts': dict(
                contactSpacing=400,
                groupIn={
                    'xcoords': np.arange(-.1, 10.1, 1),
                    'ycoords': np.arange(-.1, 10.1, 1)}),
            'triFolderSource': {
                'exp': experimentName, 'block': 4,
                'nameSuffix': 'spike_preview'},
            'triFolderDest': [
                {
                    'exp': experimentName, 'block': i,
                    'nameSuffix': 'mean_subtracted'}
                for i in [4]
                ]
        }
    }
    triFolderSourceBase = os.path.join(
        '201901271000-Proprio', 'tdc_Block001')
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [4]]
    
    return locals()
