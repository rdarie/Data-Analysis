def getExpOpts():
    #
    miniRCTrialLookup = {
        1: True,
        2: False,
        3: False
        }
    RCTrialLookup = {i: False for i in miniRCTrialLookup.keys()}
    plottingFigures = False
    plotBlocking = True
    remakePrb = False

    experimentName = '201901221000-Proprio'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548172348944'],
        2: ['Session1548173292049'],
        3: ['Session1548174549812', 'Session1548176296411', 'Session1548176490066']
        }

    #  options for automatic tap detection on ins data
    tapDetectOpts = {
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
                'tdChan': 'ins_td3',
                'tdThres': 2,
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
                'tdChan': 'ins_td3',
                'tdThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            }
        }

    sessionTapRangesNSP = {
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

    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {}
        }

    stimDetectThres = 1
    stimDetectChans = ['ins_td2', 'ins_td3']

    triFolderSourceBase = 3
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [1, 2]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901221200-Proprio': [2],
        '201901231000-Proprio': [1, 2, 3],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [92, 527]
            ],
        2: [
            [210, 295]
            ],
        3: [
            [80, 430],
            [917, 1163],
            [1367, 2024]
            ]
        }
    return locals()