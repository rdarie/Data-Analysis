def getExpOpts():
    #
    miniRCBlockLookup = {
        1: False,
        2: False,
        }
    RCBlockLookup = {
        1: True,
        2: True,
        }
    RippleBlockLookup = {
        1: True,
        2: True,
        }
    
    experimentName = '202003091200-ISI'
    deviceName = None
    rippleMapFile = 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_inverted.map'
    asigNameList = [
        ['caudalZ_e{:02d}_a'.format(i) for i in range(17, 25) if i not in [21]],
        ['rostralY_e{:02d}_a'.format(i) for i in range(9, 16)],
        ['rostralZ_e{:02d}_a'.format(i) for i in range(17, 25) if i not in [24]]
        ]
    jsonSessionNames = {
        #  per trial
        1: [],
        2: [],
        }

    synchInfo = {'ins': {}, 'nsp': {}}
    synchInfo['ins'][1] = dict()
    synchInfo['ins'][2] = dict()
    

    synchInfo['nsp'] = {
        #  per trialSegment
        1: dict(),
        2: dict()
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
        2: {},
        }
    # options for stim artifact detection
    detectStim = False
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
    triDestinations = []
    
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202003091200': [1, 2],
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
            ]
        }
    return locals()