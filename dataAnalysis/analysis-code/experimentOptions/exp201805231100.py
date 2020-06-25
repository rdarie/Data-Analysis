def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'proprio',
        2: 'proprio',
        3: 'proprio',
        }
    fullRigInputs = {
        'A+': 'ainp11',
        'B+': 'ainp13',
        'Z+': 'ainp15',
        'A-': 'ainp12',
        'B-': 'ainp14',
        'Z-': 'ainp16',
        'rightBut': 'ainp1',
        'leftBut': 'ainp2',
        'rightLED': 'ainp3',
        'leftLED': 'ainp2',
        'simiTrigs': 'ainp8',
        'tapSync': 'ainp7',
        }
    miniRCRigInputs = {
        'tapSync': 'ainp7',
        'simiTrigs': 'ainp8'
        }
    RCRigInputs = {
        'kinectSync': 'ainp16',
        }
    
    experimentName = '201805231100-Proprio'
    deviceName = ''
    
    jsonSessionNames = {
        #  per trial
        1: [],
        2: [],
        3: []
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
    synchInfo['ins'][3] = {
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
            },
        3: {
            0: {'timeRanges': [203, 205], 'keepIndex': slice(None)}
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
        for blockIdx in [2, 3]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201805231100-Proprio': [1, 2, 3],
        }

    movementSizeBins = [
        i / 180e2
        for i in [
            0, 2500, 5000,
            7500, 10000, 12500,
            15000, 17500, 20000, 22500]
        ]
    movementSizeBinLabels = [
        'XXXS', 'XXS', 'XS',
        'S', 'M', 'L', 'XL',
        'XXL', 'XXXL'
        ]
    alignTimeBoundsLookup = {
        1: [
            [8, 1822],
            ],
        #  per trial
        2: [
            #  per trialSegment
            [5, 460],
            ],
        #  per trial
        3: [
            #  per trialSegment
            [5, 1048],
            ]
        }
    return locals()