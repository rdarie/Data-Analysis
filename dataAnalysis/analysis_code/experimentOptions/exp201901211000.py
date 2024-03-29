def getExpOpts():
    #
    miniRCBlockLookup = {
        1: False,
        2: False,
        3: False
        }
    RCBlockLookup = {
        i: False
        for i in miniRCBlockLookup.keys()} 
    #
    experimentName = '201901211000-Proprio'
    deviceName = 'DeviceNPC700373H'
    #
    jsonSessionNames = {
        #  per trial
        1: ['Session1548087177984', 'Session1548087855083'],
        2: ['Session1548088399924'],
        3: ['Session1548089184035', 'Session1548090076537', 'Session1548090536025']
        }
    synchInfo = {'ins': {}, 'nsp': {}}
    #  options for automatic tap detection on ins data
    synchInfo['ins'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {
                'timeRanges': [(115, 119)],
                'accChan': 'ins_accinertia',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            1: {
                'timeRanges': [(725, 728)],
                'accChan': 'ins_accinertia',
                'accThres': 1,
                'iti': 0.2,
                'keepIndex': slice(None, 2)
                }
            },
        2: {
            #  per trialSegment
            0: {
                'timeRanges': [(77.5, 79.5)],
                'accChan': 'ins_accinertia',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            },
        3: {
            #  per trialSegment
            0: {
                'timeRanges': [(63, 71)],
                'accChan': 'ins_accz',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': [0, 1, 2, 4, 5]
                },
            1: {
                'timeRanges': [(910, 914)],
                'accChan': 'ins_accx',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None, 2)
                },
            2: {
                'timeRanges': [(1360, 1364)],
                'accChan': 'ins_accz',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            }
        }
    synchInfo['nsp'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {'timeRanges': [60, 64], 'keepIndex': slice(None)},
            1: {'timeRanges': [669, 672], 'keepIndex': slice(None)}
            },
        2: {
            0: {'timeRanges': [101, 103], 'keepIndex': slice(None)}
            },
        3: {
            0: {'timeRanges': [48, 56],
                'keepIndex': slice(None)},
            1: {'timeRanges': [896, 900],
                'keepIndex': slice(None, 2)},
            2: {'timeRanges': [1344, 1348],
                'keepIndex': slice(None)}
            }
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {}
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
    #
    triFolderSourceBase = 1
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [2, 3]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901201200-Proprio': [2],
        '201901211000-Proprio': [1, 2, 3],
        }
    #
    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [92, 527],
            [775, 887]
            ],
        2: [
            [210, 272]
            ],
        3: [
            [80, 430],
            [917, 1163],
            [1367, 2024]
            ]
        }
    #
    return locals()