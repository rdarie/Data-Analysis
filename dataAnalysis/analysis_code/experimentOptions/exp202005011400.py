def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'isi',
        2: 'isi',
        3: 'isi',
        4: 'isi',
        5: 'isi',
        6: 'isi',
        }
    experimentName = '202005011400-Peep'
    deviceName = None
    rippleMapFile = {
        1: 'isi_nano1caudal_xAzByC_ortho_nano2rostral_xAyBzC_ortho.map',
        2: 'isi_nano1caudal_xAzByC_ortho_nano2rostral_xAyBzC_ortho.map',
        3: 'isi_nano1caudal_xAzByC_ortho_nano2rostral_xAyBzC_ortho.map',
        4: 'isi_nano1caudal_xAzByC_ortho_nano2rostral_xAyBzC_ortho.map',
        5: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        6: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        }
    # use "original" file in edge cases where the ns5 file was saved incorrectly
    # with the wrong map. Original is the incorrect, old one, above is the corrected one.
    rippleOriginalMapFile = {
        1: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        2: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        3: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        4: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        5: None,
        6: None,
        }
    rippleFastSettleTriggers = {
        1: {'stim': 'same'},
        2: {'stim': 'same'},
        3: {'stim': 'same'},
        4: {'stim': 'same'},
        5: {'stim': 'same'},
        6: {'stim': 'same'}
    }
    # exclude dummy electrodes 8 and 16
    asigNameList = [
        ['caudalX_e{:02d}_a'.format(i) for i in range(1, 8)] +
        ['caudalY_e{:02d}_a'.format(i) for i in range(9, 16)] +
        ['caudalZ_e{:02d}_a'.format(i) for i in range(17, 25)] +
        ['rostralX_e{:02d}_a'.format(i) for i in range(1, 8)] +
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
        5: [],
        6: [],
        }

    synchInfo = {'delsys': {}, 'nsp': {}, 'ins': {}}
    synchInfo['delsys'][1] = {'timeRanges': [303.6, 1589], 'chooseCrossings': slice(None)}  #########
    synchInfo['delsys'][2] = {'timeRanges': [15, 1373.5], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][3] = {'timeRanges': [30, 1722], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][4] = {'timeRanges': [8, 580.3], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][5] = {'timeRanges': [236, 1676], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][6] = {'timeRanges': [7, 1693], 'chooseCrossings': slice(None)}
    #
    synchInfo['nsp'][1] = {'timeRanges': [176.6, 1525.2], 'chooseCrossings': slice(None, 2570)}
    synchInfo['nsp'][2] = {'timeRanges': [3, 1361.1], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][3] = {'timeRanges': [3, 1695], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][4] = {'timeRanges': [7, 579.9], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][5] = {'timeRanges': [240, 1680], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][6] = {'timeRanges': [6, 1692], 'chooseCrossings': slice(None)}
    
    # For emg analysis - emg missing for some time ranges
    alignTimeBoundsLookup = {
        1: [
            [175, 1461.8]
            ],
        2: [
            [3, 1361.1]
            ],
        3: [
            [3, 1695]
            ],
        4: [
            [7, 579.9]
            ],
        5: [
            [240, 1680]
            ],
        6: [
            [6, 1692]
            ],
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a block
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {},
        }
    # options for stim artifact detection
    detectStim = False
    stimDetectThresDefault = 4
    stimDetectChansDefault = ['ins_td0', 'ins_td2']
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault}
        }}
    triFolderSourceBase = 1
    triDestinations = []
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        # For emg analysis - emg missing from block 2
        '202005011400-Peep': [1, 2, 3, 4, 5, 6],
        }
    assembledSegmentToBlockLookup = {
        i - 1: i for i in [1, 2, 3, 4, 5, 6]
        }
    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    rowColOverrides = {
        'electrode': [
            # Block001
            '-rostralZ_e20',
            '-caudalY_e12',
            # Block002
            '-rostralZ_e24',
            '-caudalY_e15',
            # Block 003
            '-rostralY_e15',
            '-caudalZ_e21',
            '-caudalZ_e12+caudalZ_e17caudalZ_e21',
            # Block 004
            '-rostralZ_e20+rostralY_e09rostralY_e13'
            # Block 005
            '-caudalZ_e20',
            '-caudalZ_e21',
            # Block 006
            '-caudalZ_e23',
            '-caudalZ_e21+caudalZ_e17',
            '-rostralZ_e24+rostralZ_e19',
            ]
    }
    outlierDetectColumns = [
            # 'LThoracolumbarFasciaEmg#0',
            # 'LGracilisEmg#0',
            # 'LTensorFasciaeLataeEmg#0',
            # 'LPeroneusLongusEmg#0',
            # 'LBicepsFemorisEmg#0',
            # 'LGastrocnemiusEmg#0',
            # 'RThoracolumbarFasciaEmg#0',
            # 'RGracilisEmg#0',
            # 'RTensorFasciaeLataeEmg#0',
            # 'RPeroneusLongusEmg#0',
            # 'RBicepsFemorisEmg#0',
            # 'RGastrocnemiusEmg#0'
            'RThoracolumbarFasciaEmgEnv#0',
            'RGracilisEmgEnv#0',
            'RTensorFasciaeLataeEmgEnv#0',
            'RPeroneusLongusEmgEnv#0',
            'RBicepsFemorisEmgEnv#0',
            'RGastrocnemiusEmgEnv#0'
        ]
    RCPlotOpts = {
        'keepFeatures': [
            'LBicepsFemoris', 'LGastrocnemius', 'LGracilis',
            'LPeroneusLongus', 'LTensorFasciaeLatae',
            'LThoracolumbarFascia', 'RBicepsFemoris',
            'RGastrocnemius', 'RGracilis', 'RPeroneusLongus',
            'RTensorFasciaeLatae', 'RThoracolumbarFascia',
            # 'RExtensorDigitorum',
            # 'LSemitendinosus', 'RSemitendinosus',
            ],
        'keepElectrodes': None,
        'significantOnly': False,
        }
    # based on vanderhorst and hostege distanges along lumbar enlargement
    delsysMapDict = ({
        'label': [
            'RExtensorDigitorumEmg',
            'LThoracolumbarFasciaEmg',    'RThoracolumbarFasciaEmg',
            'LGracilisEmg',               'RGracilisEmg',  # ~40
            'LTensorFasciaeLataeEmg',     'RTensorFasciaeLataeEmg',  # ~60
            'LPeroneusLongusEmg',         'RPeroneusLongusEmg',  # ~75
            'LSemitendinosusEmg',         'RSemitendinosusEmg',  # ~80
            'LBicepsFemorisEmg',          'RBicepsFemorisEmg',  # ~90
            'LGastrocnemiusEmg',          'RGastrocnemiusEmg',  # ~90
            'RExtensorDigitorumEmgEnv',
            'LThoracolumbarFasciaEmgEnv', 'RThoracolumbarFasciaEmgEnv',
            'LGracilisEmgEnv',            'RGracilisEmgEnv',
            'LTensorFasciaeLataeEmgEnv',  'RTensorFasciaeLataeEmgEnv',
            'LPeroneusLongusEmgEnv',      'RPeroneusLongusEmgEnv',
            'LSemitendinosusEmgEnv',      'RSemitendinosusEmgEnv',
            'LBicepsFemorisEmgEnv',       'RBicepsFemorisEmgEnv',
            'LGastrocnemiusEmgEnv',       'RGastrocnemiusEmgEnv',
            ],
        'ycoords': [
            0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
        'xcoords': [
            0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 10, 10, 12, 12,
            0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 10, 10, 12, 12]
        })
    return locals()
