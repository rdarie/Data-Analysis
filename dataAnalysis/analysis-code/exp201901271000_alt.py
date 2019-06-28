def getExpOpts():
    #
    miniRCTrialLookup = {
        1: False,
        2: False,
        3: False,
        4: False,
        5: True
        }

    plottingFigures = False
    plotBlocking = True
    remakePrb = False
    
    experimentName = '201901271000-Proprio'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548605159361', 'Session1548605586940'],
        2: ['Session1548606783930'],
        3: ['Session1548608122565'],
        4: ['Session1548609521574'],
        5: ['Session1548611405556']
        }

    tapDetectOpts = {}
    tapDetectOpts[1] = {
        #  per trialSegment
        0: {
            'timeRanges': [(14, 16)],
            'tdChan': 'ins_td0',
            'tdThres': 2,
            'iti': 0.2,
            'keepIndex': slice(None, 2)
            },
        1: {
            'timeRanges': [(448, 450)],
            'tdChan': 'ins_td0',
            'tdThres': 2,
            'iti': 0.2,
            'keepIndex': slice(1, None)
            }
        }
    tapDetectOpts[2] = {
        #  per trialSegment
        0: {
            'timeRanges': [(18, 20)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(1, None)
            }
        }
    tapDetectOpts[3] = {
        #  per trialSegment
        0: {
            'timeRanges': [(20, 22)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    tapDetectOpts[4] = {
        #  per trialSegment
        0: {
            'timeRanges': [(21, 24)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    tapDetectOpts[5] = {
        #  per trialSegment
        0: {
            'timeRanges': [(22, 25)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }

    sessionTapRangesNSP = {
        #  per trialSegment
        1: {
            0: {'timeRanges': [212, 214], 'keepIndex': slice(None, 2)},
            1: {'timeRanges': [647, 649], 'keepIndex': slice(1, None)}
            },
        2: {
            0: {'timeRanges': [203, 205], 'keepIndex': slice(1, None)}
            },
        3: {
            0: {'timeRanges': [105, 107], 'keepIndex': slice(None)}
            },
        4: {
            0: {'timeRanges': [140, 144], 'keepIndex': slice(None)}
            },
        5: {
            0: {'timeRanges': [145, 148], 'keepIndex': slice(None)}
            }
        }

    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {},
        4: {},
        5: {}
        }

    stimDetectThres = 1
    stimDetectChans = ['ins_td0', 'ins_td2']

    triFolderSourceBase = 3
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [1, 2, 4, 5]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901271000-Proprio': [1, 2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  each key is a trial
        1: [
            [257, 552],
            [670, 1343],
            ],
        #  per trial
        2: [
            #  per trialSegment
            [238, 1198],
            ],
        3: [
            [171, 1050]
            ],
        4: [
            [185, 1501],
            ],
        5: [
            []
        ]
        }

    alignTimeBounds = [
        #  each key is a trial
        [
            [257, 552],
            [670, 1343],
            ],
        #  per trial
        [
            #  per trialSegment
            [238, 1198],
            ],
        [
            [171, 1050]
            ],
        [
            [185, 1501],
            ]
        ]
    gpfaRunIdx = 1
    return locals()