def getExpOpts():
    #
    miniRCTrialLookup = {
        1: False,
        2: False,
        3: False,
        4: False
        }
    RCTrialLookup = {
        1: True,
        2: True,
        3: True,
        4: True
        }
        
    plottingFigures = False
    plotBlocking = True
    remakePrb = False

    experimentName = '201901070700-ProprioRC'
    deviceName = 'DeviceNPC700373H'
    
    jsonSessionNames = {
        #  per trial
        1: ['Session1546879988028'],
        2: ['Session1546881633940'],
        3: ['Session1546883057983'],
        4: ['Session1546884658617']
        }

    openEphysChanNames = {
        'CH1': 'Right Calf Distal Lateral',
        'CH2': 'Right Calf Proximal Medial',
        'CH3': 'Right Shin Proximal',
        'CH4': 'Right Hamstring Distal Lateral',
        'CH5': 'Right Quadriceps Central',
        'CH6': 'Right Hamstring Proximal Medial',
        'CH7': 'Right Lower Back',
        'CH8': 'Left Lower Back',
        'CH9': 'Left Calf Proximal Medial',
        'CH10': 'Left Shin Proximal',
        'CH11': 'Left Hamstring Proximal Medial',
        'CH12': 'Left Hamstring Distal Lateral',
        'CH13': 'Left Quadriceps Central',
        'CH14': 'Left Calf Lateral',
        'CH15': 'Central Back',
        'ADC1': 'KinectSync',
        'ADC2': 'TensSync'}
    #
    openEphysIgnoreSegments = {
        1: [0],
        2: None,
        3: [0],
        4: [1],
    }
    #
    openEphysBaseNames = {i: 'Trial{:0>3}_EMG'.format(i) for i in RCTrialLookup.keys()}
    RCTrialInfo = {'oe': {}, 'ins': {}, 'nsp': {}}
    #  discard first n seconds
    RCTrialInfo['oe']['discardTime'] = {
        i: None
        for i in RCTrialLookup.keys()}

    synchInfo = {'oe': {}, 'ins': {}, 'nsp': {}}
    #  options for automatic tap detection on ins data
    synchInfo['ins'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {
                'timeRanges': [(261, 273), (1339, 1352)],
                'tdChan': 'ins_td0',
                'tdThres': 2.5,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            },
        2: {
            #  per trialSegment
            0: {
                'timeRanges': [(37.9, 51.8), (1137, 1147)],
                'tdChan': 'ins_td0',
                'tdThres': 2.5,
                'iti': 0.2,
                'keepIndex': slice(None)
                }
            },
        3: {
            #  per trialSegment
            0: {
                'timeRanges': [(15, 24), (898, 904)],
                'tdChan': 'ins_td0',
                'tdThres': 2.5,
                'iti': 0.2,
                'keepIndex': slice(None)
                }            
            },
        4: {
            #  per trialSegment
            0: {
                'timeRanges': [(920, 931)],
                'tdChan': 'ins_td0',
                'tdThres': 2.5,
                'iti': 0.2,
                'keepIndex': slice(None)
                }            
            }
        }

    synchInfo['oe'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {
                'timeRanges': [(123.79, 133.92), (1201.94, 1213.65)],
                'chanName': 'CH4', 'thresh': 40,
                'iti': 0.1, 'keepIndex': slice(None)},
            },
        2: {
            #  per trialSegment
            0: {
                'timeRanges': [(227.796, 241.50), (1327.08, 1336.38)],
                'chanName': 'CH13', 'thresh': 40,
                'iti': 0.1, 'keepIndex': slice(None)},
            },
        3: {
            #  per trialSegment
            0: {
                'timeRanges': [(180.273, 187.877), (1063.614, 1068.417)],
                'chanName': 'CH13', 'thresh': 40,
                'iti': 0.1, 'keepIndex': slice(None)},
            },
        4: {
            #  per trialSegment
            0: {
                'timeRanges': [(890.654, 900.359)],
                'chanName': 'CH13', 'thresh': 40,
                'iti': 0.1, 'keepIndex': slice(None)},
            }
        }
    
    synchInfo['nsp'] = {
        #  per trial
        1: {
            #  per trialSegment
            0: {'timeRanges': [60, 64], 'keepIndex': slice(None)},
            },
        2: {
            0: {'timeRanges': [101, 103], 'keepIndex': slice(None)}
            },
        3: {
            #  per trialSegment
            0: {'timeRanges': [60, 64], 'keepIndex': slice(None)},
            },
        4: {
            0: {'timeRanges': [101, 103], 'keepIndex': slice(None)}
            }
        }
    # options for stim artifact detection
    detectStim = True
    stimDetectThresDefault = 15
    stimDetectChansDefault = ['ins_td0', 'ins_td1', 'ins_td2', 'ins_td3']
    stimDetectOverrideStartTimes = {
        1: [600.740],
        2: [322.320, 860.524],
        3: [123.124],
        4: None,
    }
    stimDetectOptsByChannelSpecific = {
        #  group
        0: {
            #  program
            0: {'detectChannels': ['ins_td3'], 'thres': 25},
            1: {'detectChannels': ['ins_td3'], 'thres': 25},
            2: {'detectChannels': ['ins_td0', 'ins_td1', 'ins_td2'], 'thres': 25},
            3: {'detectChannels': ['ins_td3'], 'thres': 25}
        },
        1: {
            #  program
            0: {'detectChannels': ['ins_td0', 'ins_td2', 'ins_td3'], 'thres': stimDetectThresDefault},
            1: {'detectChannels': ['ins_td1'], 'thres': stimDetectThresDefault},
            2: {'detectChannels': ['ins_td0', 'ins_td2', 'ins_td3'], 'thres': stimDetectThresDefault},
            3: {'detectChannels': ['ins_td0', 'ins_td2', 'ins_td3'], 'thres': stimDetectThresDefault}
        },
        2: {
            #  program
            0: {'detectChannels': ['ins_td0', 'ins_td1'], 'thres': stimDetectThresDefault},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            2: {'detectChannels': ['ins_td2'], 'thres': stimDetectThresDefault},
            3: {'detectChannels': ['ins_td3'], 'thres': stimDetectThresDefault}
        },
        #  1: {  # trial004 has a different group 1 configured
        #      #  program
        #      0: {'detectChannels': ['ins_td0', 'ins_td2', 'ins_td3'], 'thres': stimDetectThresDefault},
        #      1: {'detectChannels': ['ins_td1'], 'thres': stimDetectThresDefault},
        #      2: {'detectChannels': ['ins_td0', 'ins_td2', 'ins_td3'], 'thres': stimDetectThresDefault},
        #      3: {'detectChannels': ['ins_td1', 'ins_td3'], 'thres': stimDetectThresDefault}
        #  }
        }
    # stimDetectChans = None
    triFolderSourceBase = 1
    triDestinations = [
        'Trial00{}'.format(trialIdx)
        for trialIdx in [2, 3]]
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '201901070700-ProprioRC': [1, 2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    alignTimeBoundsLookup = {
        #  per trial
        1: [
            #  per trialSegment
            [177, 1189],
            ],
        2: [
            [242, 1326]
            ],
        3: [
            [188, 1060],
            ],
        4: [
            [0, 890],
            ]
        }
    alignTimeBounds = [
        #  per trial
        [177, 1189],
        [242, 1326],
        [188, 1060],
        [0, 890]
        ]
    gpfaRunIdx = 1
    return locals()