import os


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
    
    jsonSessionNames = {
        #  per trial
        1: [
            'Session1548517963713', 'Session1548518261171',
            'Session1548518496294', 'Session1548518727243',
            'Session1548518982240'],
        2: ['Session1548520562275'],
        3: ['Session1548521956580'],
        4: ['Session1548524126669'],
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
            'timeRanges': [(15.5, 17.5)],
            'tdChan': 'ins_td3',
            'tdThres': 2.5,
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
            0: {'timeRanges': [939, 941], 'keepIndex': slice(None)}
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
    stimDetectThresDefault = 3
    stimDetectChansDefault = ['ins_td2', 'ins_td3']
    stimDetectOverrideStartTimes = {
        1: None,
        2: None,
        3: None,
        4: None,
    }
    stimDetectOptsByChannelSpecific = {
        #group
        0: {
            # program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault}
        }}
        
    triFolderSourceBase = os.path.join(
        '201901271000-Proprio', 'tdc_Block001')
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [4]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901261000-Proprio': [4],
        '201901271000-Proprio': [1, 2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  TODO
        1: [
            [257, 552],
            [670, 1343],
            ],
        #  per trial
        #  TODO
        2: [
            #  per trialSegment
            [238, 1198],
            ],
        #  TODO
        3: [
            [171, 1050]
            ],
        4: [
            [939, 2252],
            ]
        }
    return locals()