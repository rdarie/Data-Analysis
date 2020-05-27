def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        }
    
    experimentName = '201804271016-Proprio'
    deviceName = ''
    
    jsonSessionNames = {
        #  per trial
        1: [],
        2: []
        }

    synchInfo = {'ins': {}, 'nsp': {}}
    synchInfo['ins'][1] = {
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
        #  per trialSegment
        0: {
            'timeRanges': [(18, 20)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }

    synchInfo['nsp'] = {
        #  per trialSegment
        1: {
            0: {'timeRanges': [212, 214], 'keepIndex': slice(None)},
            1: {'timeRanges': [647, 649], 'keepIndex': slice(None)}
            },
        2: {
            0: {'timeRanges': [203, 205], 'keepIndex': slice(None)}
            }
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {}
        }
    # options for stim artifact detection
    detectStim = True
    stimDetectThresDefault = 4
    stimDetectChansDefault = ['ins_td0', 'ins_td2']
    stimDetectOptsByChannelSpecific = {
        #group
        0: {
            #program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault}
        }}
        
    triFolderSourceBase = 1
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [2]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201804271016-Proprio': [1, 2],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        1: [
            [999, 999],
            ],
        #  per trial
        2: [
            #  per trialSegment
            [999, 999],
            ]
        }
    return locals()