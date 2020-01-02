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
                'timeRanges': [(41, 43)],
                'tdChan': 'ins_td2',
                'tdThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            1: {
                'timeRanges': [(378, 380)],
                'accChan': 'ins_accz',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            2: {
                'timeRanges': [(1809, 1811)],
                'accChan': 'ins_accz',
                'accThres': 2,
                'iti': 0.2,
                'keepIndex': slice(None)
                },
            3: {
                'timeRanges': [(2237, 2239)],
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
            0: {'timeRanges': [61, 63], 'keepIndex': slice(None)},
            1: {'timeRanges': [398, 400], 'keepIndex': slice(None)},
            2: {'timeRanges': [1829, 1831], 'keepIndex': slice(None)},
            3: {'timeRanges': [2256, 2258], 'keepIndex': slice(None)}
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
    triFolderSourceBase = os.path.join(
        '201901221000-Proprio', 'tdc_Trial002')
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [1]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901221000-Proprio': [2, 3],
        '201901231000-Proprio': [1]
        }
    #
    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [90, 239.5],
            [412, 1679.5],
            [1897, 1967.8],
            [2284, 3419]
            ]
        }
    #
    return locals()
