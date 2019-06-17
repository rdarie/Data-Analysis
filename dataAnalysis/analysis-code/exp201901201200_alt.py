import os


def getExpOpts():
    plottingFigures = False
    plotBlocking = True
    remakePrb = False

    experimentName = '201901201200-Proprio'
    deviceName = 'DeviceNPC700373H'

    jsonSessionNames = {
        #  each key is a trial
        1: ['Session1548006829607'],
        2: [
            'Session1548008243454',
            'Session1548009494100',
            'Session1548010328823',
            'Session1548010624175']
        }

    miniRCTrialLookup = {
        1: True,
        2: False
        }

    tapDetectOpts = {
        #  each key is a trial
        1: {
            #  each key is a trialSegment
            0: {
                'timeRanges': [(79.5, 80.75)],
                'tdChan': 'ins_td3',
                'tdThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None, 2)
                }
            },
        2: {
            #  each key is a trialSegment
            0: {
                'timeRanges': [(113, 115)],
                'accChan': 'ins_accz',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            1: {
                'timeRanges': [(1260, 1262)],
                'accChan': 'ins_accinertia',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            2: {
                'timeRanges': [(2115, 2118)],
                'accChan': 'ins_accinertia',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            3: {
                'timeRanges': [(2407, 2409)],
                'accChan': 'ins_accz',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            }
        }

    sessionTapRangesNSP = {
        #  each key is a trial
        1: {
            #  each key is a trialSegment
            0: {'timeRanges': [74, 75.25], 'keepIndex': slice(None)}
            },
        2: {
            0: {'timeRanges': [142, 144], 'keepIndex': slice(None)},
            1: {'timeRanges': [1289, 1291], 'keepIndex': slice(None)},  # missing
            2: {'timeRanges': [2144, 2147], 'keepIndex': [0, 1, 3, 4]},
            3: {'timeRanges': [2436, 2438], 'keepIndex': [0, 1]}
            }
        }

    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {1: 0}  # override segment 1 based on segment 0
        }

    stimDetectThres = 1
    stimDetectChans = ['ins_td2', 'ins_td3']
    
    triFolderSourceBase = os.path.join(
        '201901211000-Proprio', 'tdc_Trial001')
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [1, 2]]
    
    alignTimeBoundsLookup = {
        #  each key is a trial
        1: [
            [144, 829]
        ],
        2: [
            [247, 667],
            [1370, 1595],
            [2175, 2315],
            [2475, 2495]
            ],
        }
    return locals()