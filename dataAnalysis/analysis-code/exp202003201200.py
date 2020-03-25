def getExpOpts():
    #
    miniRCBlockLookup = {
        1: False,
        2: False,
        3: False,
        4: False,
        }
    RCBlockLookup = {
        1: True,
        2: True,
        3: True,
        4: True,
        }
    RippleBlockLookup = {
        1: True,
        2: True,
        3: True,
        4: True,
        }
    
    experimentName = '202003201200-Peep'
    deviceName = None
    rippleMapFile = 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map'
    # exclude caudal_e08, rostral_e03 and rostral_e016
    asigNameList = [
        ['caudalX_e{:02d}_a'.format(i) for i in range(1, 8)] +
        ['caudalY_e{:02d}_a'.format(i) for i in range(9, 17)] +
        ['caudalZ_e{:02d}_a'.format(i) for i in range(17, 25)],
        ['rostralX_e{:02d}_a'.format(i) for i in range(1, 9) if i not in [3]] +
        ['rostralY_e{:02d}_a'.format(i) for i in range(9, 16)] +
        ['rostralZ_e{:02d}_a'.format(i) for i in range(17, 25)]
        ]
    ainpNameList = ['analog 1']
    jsonSessionNames = {
        #  per block
        1: [],
        2: [],
        3: [],
        4: [],
        }

    synchInfo = {'ins': {}, 'nsp': {}}
    synchInfo['ins'][1] = dict()

    synchInfo['nsp'] = {
        #  per trialSegment
        1: dict(),
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a trial
        1: {},
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
            [1, 2000]
            ],
        }
    return locals()