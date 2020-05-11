def getExpOpts():
    #
    miniRCBlockLookup = {
        1: False,
        2: False,
        3: False,
        4: False,
        5: True
        }
    RCBlockLookup = {i: False for i in miniRCBlockLookup.keys()} 
    
    experimentName = '202001231400-Duke-Spontaneous'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1548605159361', 'Session1548605586940'],
        2: ['Session1548606783930'],
        3: ['Session1548608122565'],
        4: ['Session1548609521574'],
        5: ['Session1548611405556', 'Session1548612434879', 'Session1548612688167']
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
            'timeRanges': [(20, 22)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][4] = {
        #  per trialSegment
        0: {
            'timeRanges': [(21, 24)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            }
        }
    synchInfo['ins'][5] = {
        #  per trialSegment
        0: {
            'timeRanges': [(22, 24)],
            'tdChan': 'ins_td0',
            'tdThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        1: {
            'timeRanges': [(1047, 1049)],
            'accChan': 'ins_accinertia',
            'accThres': 2.5,
            'iti': 0.2,
            'keepIndex': slice(None)
            },
        2: {
            'timeRanges': [(1301, 1304)],
            'accChan': 'ins_accz',
            'accThres': 2.5,
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
            0: {'timeRanges': [105, 107], 'keepIndex': slice(None)}
            },
        4: {
            0: {'timeRanges': [140, 144], 'keepIndex': slice(None)}
            },
        5: {
            0: {'timeRanges': [145, 147], 'keepIndex': slice(None)},
            1: {'timeRanges': [1169, 1171], 'keepIndex': slice(None)},
            2: {'timeRanges': [1423, 1426], 'keepIndex': slice(None)}
            }
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        3: {},
        4: {},
        5: {}
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
        
    triFolderSourceBase = 3
    triDestinations = [
        'Block00{}'.format(blockIdx)
        for blockIdx in [1, 2, 4, 5]]
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901271000-Proprio': [1, 2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
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
            [100, 2010]
        ]
        }
    return locals()