def getExpOpts():
    #
    blockExperimentTypeLookup = {
        1: 'isi',
        2: 'isi',
        3: 'isi',
        4: 'isi',
        5: 'isi',
        6: 'isi',
        7: 'isi',
        8: 'isi',
        }
    experimentName = '202012171300-Goat'
    deviceName = None
    subjectName = 'Goat'
    rippleMapFile = {
        1: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        2: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        3: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        4: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        5: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        6: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        7: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        8: 'isi_port1nano1caudal_xAyBzC_ortho_port2nano1rostral_xAyBzC_ortho.map',
        }
    delsysExampleHeaderPath = './delsys_example_header_20200922.csv'
    # use "original" file in edge cases where the ns5 file was saved incorrectly
    # with the wrong map. Original is the incorrect, old one, above is the corrected one.
    rippleOriginalMapFile = {
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
        }
    #
    rippleFastSettleTriggers = {
        1: {'stim': 'none'},
        2: {'stim': 'none'},
        3: {'stim': 'none'},
        4: {'stim': 'none'},
        5: {'stim': 'none'},
        6: {'stim': 'none'},
        7: {'stim': 'none'},
        8: {'stim': 'none'},
    }
    # exclude dummy electrodes 8 and 16
    asigNameList = [
        ['caudalX_e{:02d}_a'.format(i) for i in range(1, 8)] +
        ['caudalY_e{:02d}_a'.format(i) for i in range(9, 16)] +
        ['caudalZ_e{:02d}_a'.format(i) for i in range(17, 25)],
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
        7: [],
        8: [],
        }
    synchInfo = {'delsysToNsp': {}, 'nspForDelsys': {}, 'ins': {}}
    for blockIdx in blockExperimentTypeLookup.keys():
        synchInfo['nspForDelsys'][blockIdx] = {
            'synchChanName': 'analog 1'
            }
        synchInfo['delsysToNsp'][blockIdx] = {
            'synchChanName': 'AnalogInputAdapterAnalog'
        }
    # For emg analysis - emg missing for some time ranges
    synchInfo['delsysToNsp'][3].update({'timeRanges': [10, 3127], 'chooseCrossings': slice(None)})
    synchInfo['delsysToNsp'][6].update({'timeRanges': [90, 2080], 'chooseCrossings': slice(None, 1000)})
    synchInfo['delsysToNsp'][7].update({'timeRanges': [12, 1203], 'chooseCrossings': slice(None, 1200)})
    synchInfo['delsysToNsp'][7].update({'timeRanges': [19, 674], 'chooseCrossings': slice(None, 1200)})
    #
    synchInfo['nspForDelsys'][3].update({'timeRanges': [10, 3127], 'chooseCrossings': slice(None)})
    synchInfo['nspForDelsys'][6].update({'timeRanges': [50, 2040], 'chooseCrossings': slice(None, 1000)})
    synchInfo['nspForDelsys'][7].update({'timeRanges': [6, 1244], 'chooseCrossings': slice(None, 1200)})
    synchInfo['nspForDelsys'][7].update({'timeRanges': [3, 659], 'chooseCrossings': slice(None, 1200)})
    alignTimeBoundsLookup = {
        # 1: [
        #     [3, 2290.5]
        #     ],
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
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
        7: {},
        8: {},
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
        '202012171300-Goat': [3, 5, 6],
        }
    assembledSegmentToBlockLookup = {
        i - 1: i for i in [3, 5, 6]
        }
    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    '''rowColOverrides = {
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
    }'''
    outlierDetectOptions = dict(
        targetEpochSize=10e-3,
        windowSize=(0, 100e-3),
        # conditionNames=[
        #     'electrode', 'amplitude', 'RateInHz',
        #     'pedalMovementCat', 'pedalSizeCat', 'pedalDirection'],
        conditionNames=[
            'electrode', 'nominalCurrent', 'RateInHz'],
        twoTailed=False,
        )
    outlierDetectColumns = [
            # 'LThoracolumbarFasciaEmg#0',
            'LGracilisEmg#0',
            # 'LTensorFasciaeLataeEmg#0',
            'LPeroneusLongusEmg#0',
            'LBicepsFemorisEmg#0',
            'LGastrocnemiusEmg#0',
            # 'RThoracolumbarFasciaEmg#0',
            'RGracilisEmg#0',
            # 'RTensorFasciaeLataeEmg#0',
            'RPeroneusLongusEmg#0',
            'RBicepsFemorisEmg#0',
            'RGastrocnemiusEmg#0',
            # 'LThoracolumbarFasciaEmgEnv#0',
            #
            # 'LGracilisEmgEnv#0',
            # 'LTensorFasciaeLataeEmgEnv#0',
            # 'LPeroneusLongusEmgEnv#0',
            # 'LBicepsFemorisEmgEnv#0',
            # 'LGastrocnemiusEmgEnv#0',
            # 'RThoracolumbarFasciaEmgEnv#0',
            # 'RGracilisEmgEnv#0',
            # 'RTensorFasciaeLataeEmgEnv#0',
            # 'RPeroneusLongusEmgEnv#0',
            # 'RBicepsFemorisEmgEnv#0',
            # 'RGastrocnemiusEmgEnv#0'
        ]
    RCPlotOpts = {
        'rejectFeatures': ['rostralY_e12'],
        'keepFeatures': [
            'LBicepsFemoris', 'LGastrocnemius', 'LGracilis', 'LPeroneusLongus',
            'RBicepsFemoris', 'RGastrocnemius', 'RGracilis', 'RPeroneusLongus',
            # 'RTensorFasciaeLatae', 'LTensorFasciaeLatae',
            # 'LThoracolumbarFascia', 'RThoracolumbarFascia',
            # 'RExtensorDigitorum', 'LSemitendinosus', 'RSemitendinosus',
            ],
        # 'keepElectrodes': ['caudalY_e11', 'caudalZ_e18', 'caudalZ_e23'],
        # 'keepElectrodes': ['caudalZ_e23', 'caudalZ_e18', 'caudalZ_e22', 'caudalZ_e24'],
        'keepElectrodes': ['caudalY_e11', 'caudalZ_e18', 'caudalZ_e23'],
        'significantOnly': False,
        }
    RCCalcOpts = {
        'keepFeatures': [
            'LBicepsFemoris', 'LGastrocnemius', 'LGracilis', 'LPeroneusLongus',
            'RBicepsFemoris', 'RGastrocnemius', 'RGracilis', 'RPeroneusLongus',
            # 'RTensorFasciaeLatae', 'LTensorFasciaeLatae',
            # 'LThoracolumbarFascia', 'RThoracolumbarFascia',
            # 'RExtensorDigitorum', 'LSemitendinosus', 'RSemitendinosus',
        ],
        'rejectFeatures': ['rostralY_e12'],
        'keepElectrodes': None,
        # 'keepElectrodes': ['-caudalY_e11', '-caudalZ_e18', '-caudalZ_e23'],
        'significantOnly': False,
        }
    EMGStyleMarkers = {
        'LThoracolumbarFasciaEmgEnv#0': 'o',
        'LGracilisEmgEnv#0': 'v',
        'LTensorFasciaeLataeEmgEnv#0': '^',
        'LPeroneusLongusEmgEnv#0': '<',
        'LBicepsFemorisEmgEnv#0': '>',
        'LGastrocnemiusEmgEnv#0': 's',
        'RThoracolumbarFasciaEmgEnv#0': 'p',
        'RGracilisEmgEnv#0': '*',
        'RTensorFasciaeLataeEmgEnv#0': 'H',
        'RPeroneusLongusEmgEnv#0': 'X',
        'RBicepsFemorisEmgEnv#0': 'd',
        'RGastrocnemiusEmgEnv#0': 'D'
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
    minNConditionRepetitions = {
        'n': 5,
        'categories': ['nominalCurrent', 'electrode', 'RateInHz']
        }
    '''rippleFilterOpts = {
        'high': {
            'Wn': .1,
            'N': 4,
            'btype': 'high',
            'ftype': 'butter'
        }
    }'''
    delsysFilterOpts = {
        'ACC': {
            'bandstop': {
                'Wn': 75,
                'Q': 5,
                'nHarmonics': 1,
                'N': 4,
                'btype': 'bandstop',
                'ftype': 'butter'
            }
        },
        'EMG': {
            'bandstop': {
                'Wn': 60,
                'Q': 5,
                'nHarmonics': 1,
                'N': 4,
                'btype': 'bandstop',
                'ftype': 'butter'
            }
        }
    }
    lmfitFunKWArgs = dict(
        tBounds=[1.3e-3, 39e-3],
        scoreBounds=[1.3e-3, 6e-3],
        #
        expOpts=dict(
            exp1_=dict(
                tBounds=[19e-3, 39e-3],
                assessModel=True
            ),
            exp2_=dict(
                tBounds=[2e-3, 19e-3],
                assessModel=True
            ),
            exp3_=dict(
                tBounds=[1.3e-3, 1.9e-3],
                assessModel=True
            )
        ),
        # fit_kws=dict(loss='soft_l1'),
        # method='least_squares',
        method='nelder',
        iterMethod='sampleOneManyTimes',
        plotting=False, verbose=False,
        maxIter=1
        )
    return locals()
