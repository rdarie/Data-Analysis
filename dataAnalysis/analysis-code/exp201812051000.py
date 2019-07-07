import os


def getExpOpts():
    #
    miniRCTrialLookup = {i: True for i in range(1, 10)}
        
    plottingFigures = False
    plotBlocking = True
    remakePrb = False

    experimentName = '201812051000-PadawanRecruitmentCurve'
    deviceName = 'DeviceNPC'
    
    jsonSessionNames = {
        #  per trial
        i: []
        for i in range(1, 10)
        }
    
    openEphysChanNames = {
        'CH1': 'Right Calf Distal Lateral',
        'CH2': 'Right Calf Proximal Medial',
        'CH3': 'Right Shin Central Central',
        'CH4': 'Right Hamstring Distal Lateral',
        'CH5': 'Right Hamstring Proximal Medial',
        'CH6': 'Right Quadriceps Central Central',
        'CH7': 'Right Gluteus Central Central',
        'CH8': 'Left Gluteus Central Central',
        'CH9': 'Left Calf Central Central',
        'CH10': 'Left Shin Central Central',
        'CH11': 'Left Hamstring Central Central',
        'CH12': 'Left Quadriceps Central Central',
        'ADC1': 'KinectSync',
        'ADC2': 'TensSync'}
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
                },
            2: {
                'timeRanges': [(725, 728)],
                'accChan': 'ins_accinertia',
                'accThres': 1,
                'iti': 0.2,
                'keepIndex': slice(None, 2)
                },
            3: {
                'timeRanges': [(725, 728)],
                'accChan': 'ins_accinertia',
                'accThres': 1,
                'iti': 0.2,
                'keepIndex': slice(None, 2)
                }
            }
        }

    sessionTapRangesNSP = {
        #  per trial
        1: {
            #  per trialSegment
            0: {'timeRanges': [60, 64], 'keepIndex': slice(None)},
            1: {'timeRanges': [669, 672], 'keepIndex': slice(None)},
            2: {'timeRanges': [669, 672], 'keepIndex': slice(None)},
            3: {'timeRanges': [669, 672], 'keepIndex': slice(None)}
            }
        }

    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        }

    stimDetectThres = 1
    stimDetectChans = ['ins_td2', 'ins_td3']

    triFolderSourceBase = os.path.join(
        '201901221000-Proprio', 'tdc_Trial003')
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [1]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201812051000-PadawanRecruitmentCurve': [1, 2, 4, 5, 7, 8, 9],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [92, 527],
            [775, 887]
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
    
    alignTimeBounds = [
        #  per trial
        [
            [247, 667],
            [1370, 1595],
            [2175, 2315],
            [2475, 2495]
            ],
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
    gpfaRunIdx = 1
    return locals()