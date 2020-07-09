def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'isi',
        2: 'isi',
        3: 'isi',
        }
    experimentName = '202006171300-Peep'
    deviceName = None
    rippleMapFile = {
        1: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        2: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        3: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        }
    # use "original" file in edge cases where the ns5 file was saved incorrectly
    # with the wrong map. Original is the incorrect, old one, above is the corrected one.
    rippleOriginalMapFile = {
        1: None,
        2: None,
        3: None,
        }
    #
    rippleFastSettleTriggers = {
        1: {'stim': 'same'},
        2: {'stim': 'same'},
        3: {'stim': 'same'},
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
        }
    synchInfo = {'delsys': {}, 'nsp': {}, 'ins': {}}
    synchInfo['delsys'][1] = {'timeRanges': [56, 2756], 'chooseCrossings': slice(None)}  #########
    synchInfo['delsys'][2] = {'timeRanges': [4, 3599.7], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][3] = {'timeRanges': [6, 592], 'chooseCrossings': slice(None)}
    #
    synchInfo['nsp'][1] = {'timeRanges': [53, 2752], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][2] = {'timeRanges': [11, 3606.7], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][3] = {'timeRanges': [14, 600], 'chooseCrossings': slice(None)}
    # For emg analysis - emg missing for some time ranges
    alignTimeBoundsLookup = {
        1: [
            [53, 2752]
            ],
        2: [
            [11, 3606.7]
            ],
        3: [
            [14, 600]
            ]
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a block
        1: {},
        2: {},
        3: {}
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
        '202006171300-Peep': [1, 2, 3],
        }
    assembledSegmentToBlockLookup = {
        i - 1: i for i in [1, 2, 3]
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
