def getExpOpts():
    #
    miniRCTrialLookup = {
        1: False,
        2: False,
        3: False
        }
        
    plottingFigures = False
    plotBlocking = True
    remakePrb = False

    experimentName = '201901211000-Proprio'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548087177984', 'Session1548087855083'],
        2: ['Session1548088399924'],
        3: ['Session1548089184035', 'Session1548090076537', 'Session1548090536025']
        }

    #  options for automatic tap detection on ins data
    tapDetectOpts = {
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

    sessionTapRangesNSP = {
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

    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {}
        }

    stimDetectThres = 1
    stimDetectChans = ['ins_td2', 'ins_td3']

    triFolderSourceBase = 1
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [2, 3]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901201200-Proprio': [2],
        '201901211000-Proprio': [1, 2, 3],
        }

    movementSizeBins = [-0.9, -0.45, -0.2, 0.2, 0.55, 0.9]
    alignTimeBounds = [
        #  each key is a trial
        [
            [200, 667],
            [1370, 1595],
            [2175, 2315],
            [2475, 2495]
            ],
        #  per trial
        [
            #  per trialSegment
            [92, 527],
            [775, 887]
            ],
        [
            [210, 295]
            ],
        [
            [80, 430],
            [917, 1163],
            [1367, 2024]
            ]
        ]
    return locals()