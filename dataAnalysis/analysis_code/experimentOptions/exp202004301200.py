def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'isi',
        2: 'isi',
        3: 'isi',
        4: 'isi',
        5: 'isi',
        }
    experimentName = '202004301200-Peep'
    deviceName = None
    rippleMapFile = {
        1: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        2: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        3: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        4: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
        5: 'isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map',
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
    rippleFastSettleTriggers = {
        1: {'stim': 'same'},
        2: {'stim': 'same'},
        3: {'stim': 'same'},
        4: {'stim': 'same'},
        5: {'stim': 'same'},
        6: {'stim': 'same'}
    }
    jsonSessionNames = {
        #  per block
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        }

    synchInfo = {'delsys': {}, 'nsp': {}, 'ins': {}}
    synchInfo['delsys'][4] = {'timeRanges': [3.5, 1325.5], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][5] = {'timeRanges': [2, 3343], 'chooseCrossings': slice(None)}
    #
    synchInfo['nsp'][4] = {'timeRanges': [19.5, 1341.5], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][5] = {'timeRanges': [2, 3363], 'chooseCrossings': slice(None)}
    ### For emg analysis - emg missing for some time ranges
    #  alignTimeBoundsLookup = {
    #      4: [
    #          [19.5, 1341.5]
    #          ],
    #      5: [
    #          [2, 3340]
    #          ],
    #      }
    # For lfp analysis
    alignTimeBoundsLookup = {
        4: [
            [19.5, 2405]
            ],
        5: [
            [2, 3340]
            ],
        }
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a block
        4: {},
        5: {},
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
        '202004301200-Peep': [4, 5],
        }
    assembledSegmentToBlockLookup = {
        i - 1: i for i in [4, 5]
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    # rowColOverrides = {
    #     'electrode': [
    #         '-caudalY_e12+caudalX_e05',
    #         '-caudalY_e13+caudalZ_e20',
    #         '-caudalX_e06+caudalX_e02',
    #         '-caudalZ_e17+caudalZ_e21',
    #         '-rostralY_e12+rostralZ_e24',
    #         '-rostralZ_e19+rostralZ_e24',
    #         '-rostralY_e14+rostralZ_e21',
    #         '-rostralZ_e17+rostralZ_e21',
    #         '-rostralY_e12+rostralX_e05',
    #         ]
    # }
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
