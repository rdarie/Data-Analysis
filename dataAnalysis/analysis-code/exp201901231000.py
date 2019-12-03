import os


def getExpOpts():
    #
    miniRCTrialLookup = {
        1: False,
        }
    RCTrialLookup = {
        i: False
        for i in miniRCTrialLookup.keys()} 

    experimentName = '201901231000-Proprio'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: [
            'Session1548258716429', 'Session1548259084797',
            'Session1548260509595', 'Session1548260943898']
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
    synchInfo['nsp'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {'timeRanges': [60, 64], 'keepIndex': slice(None)},
            1: {'timeRanges': [669, 672], 'keepIndex': slice(None)},
            2: {'timeRanges': [669, 672], 'keepIndex': slice(None)},
            3: {'timeRanges': [669, 672], 'keepIndex': slice(None)}
            }
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        }
    # options for stim artifact detection
    detectStim = True
    stimDetectThresDefault = 15
    stimDetectChansDefault = ['ins_td2', 'ins_td3']
    stimDetectOverrideStartTimes = {
        1: None,
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
    triFolderSourceBase = 1
    triDestinations = []
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901231200-Proprio': [1]
        }
    #
    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [92, 527],
            [775, 887],
            [775, 887],
            [775, 887]
            ]
        }
    #
    alignTimeBounds = [
        #  per trial
        [
            #  per trialSegment
            [92, 527],
            [775, 887],
            [775, 887],
            [775, 887]
            ]
        ]
    return locals()
