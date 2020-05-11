def getExpOpts():
    #
    miniRCBlockLookup = {
        1: True,
        2: False,
        3: False
        }
    RCBlockLookup = {
        i: False
        for i in miniRCBlockLookup.keys()}

    experimentName = '201901221000-Proprio'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548172348944'],
        2: ['Session1548173292049'],
        3: ['Session1548174549812', 'Session1548176296411', 'Session1548176490066']
        }
    synchInfo = {'ins': {}, 'nsp': {}}
    #  options for automatic tap detection on ins data
    synchInfo['ins'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {
                'timeRanges': [(10.5, 13.5)],
                'tdChan': 'ins_td3',
                'tdThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            },
        2: {
            #  per trialSegment
            0: {
                'timeRanges': [(127, 130)],
                'accChan': 'ins_accinertia',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            },
        3: {
            #  per trialSegment
            0: {
                'timeRanges': [(88, 91)],
                'tdChan': 'ins_td3',
                'tdThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            1: {
                'timeRanges': [(1763.9, 1767.9)],
                'tdChan': 'ins_td3',
                'tdThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            2: {
                'timeRanges': [(1958, 1961)],
                'accChan': 'ins_accinertia',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            }
        }
    #
    synchInfo['nsp'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {
                'timeRanges': [197.1, 200.1],
                'keepIndex': slice(None)}
            },
        2: {
            0: {
                'timeRanges': [165.5, 168.5],
                'keepIndex': slice(None)}
            },
        3: {
            0: {
                'timeRanges': [113, 116],
                'keepIndex': [0, 2]},
            1: {
                'timeRanges': [1790, 1794],
                'keepIndex': slice(None)},
            2: {
                'timeRanges': [1983.6, 1986.6],
                'keepIndex': slice(None)}
            }
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {1: 0}
        }
    # options for stim artifact detection
    detectStim = True
    stimDetectThresDefault = 5
    stimDetectChansDefault = ['ins_td2', 'ins_td3']
    stimDetectOverrideStartTimes = {
        1: None,
        2: None,
        3: None,
    }

    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault}
        }
    }
    triFolderSourceBase = 2
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [1, 3]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901221000-Proprio': [2],
        '201901231000-Proprio': [1, 2, 3],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [230, 850],
            ],
        2: [
            [190, 447]
            ],
        3: [
            [130, 1408],
            [1835, 1856],
            [2005, 2768]
            ]
        }
    return locals()