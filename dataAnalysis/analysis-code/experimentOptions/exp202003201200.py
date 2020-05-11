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
        }

    synchInfo = {'delsys': {}, 'nsp': {}, 'ins': {}}
    synchInfo['delsys'][1] = {'timeRanges': [0, 1001], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][2] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None, 1000)}
    synchInfo['delsys'][3] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None)}
    # synchInfo['delsys'][3] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None, 59000)}
    synchInfo['delsys'][4] = {'timeRanges': [0, 822], 'chooseCrossings': slice(None)}
    # synchInfo['delsys'][4] = {'timeRanges': [0, 822], 'chooseCrossings': [i for i in range(1000)] + [i for i in range(-1000, -1)]}
    #
    synchInfo['nsp'][1] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][2] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None, 1000)}
    synchInfo['nsp'][3] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None)}
    # synchInfo['nsp'][3] = {'timeRanges': [0, 3924], 'chooseCrossings': slice(None, 59000)}
    synchInfo['nsp'][4] = {'timeRanges': [0, 802], 'chooseCrossings': slice(None)}
    # synchInfo['nsp'][4] = {'timeRanges': [0, 802], 'chooseCrossings': [i for i in range(1000)] + [i for i in range(-1000, -1)]}

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
        '202003201200-Peep': [3, 4],
        # For lfp analysis - all time ranges are valid
        # '202003201200-Peep': [2, 3, 4],
        }

    movementSizeBins = [0, 0.25, 0.5, 1, 1.25, 1.5]
    # For lfp analysis - all time ranges are valid
    #  alignTimeBoundsLookup = {
    #      1: [
    #          [1, 2000]
    #          ],
    #      2: [
    #          [1, 675]
    #          ],
    #      3: [
    #          [1, 1623]
    #          ],
    #      4: [
    #          [1, 800]
    #          ],
    #      }
    # For emg analysis - emg missing for some time ranges
    alignTimeBoundsLookup = {
        1: [
            [1, 2000]
            ],
        2: [
            [1, 60]
            ],
        3: [
            [1, 1500]
            ],
        4: [
            [1, 795]
            ],
        }
    rowColOverrides = {
        'electrode': [
            '-caudalY_e12+caudalX_e05',
            '-caudalY_e13+caudalZ_e20',
            '-caudalX_e06+caudalX_e02',
            '-caudalZ_e17+caudalZ_e21',
            '-rostralY_e12+rostralZ_e24',
            '-rostralZ_e19+rostralZ_e24',
            '-rostralY_e14+rostralZ_e21',
            '-rostralZ_e17+rostralZ_e21',
            '-rostralY_e12+rostralX_e05',
            ]
    }
    outlierDetectColumns = [
        'LPeroneusLongusEmg#0',
        'LSemitendinosusEmg#0',
        'LVastusLateralisEmg#0',
        # 'RPeroneusLongusEmg#0',
        # 'RSemitendinosusEmg#0',
        # 'RVastusLateralisEmg#0'
        ]
    delsysMapDict = ({
        'label': [
            'LBicepsBrachiiEmgEnv',   'RBicepsBrachiiEmgEnv',
            'LSemitendinosusEmgEnv',  'RSemitendinosusEmgEnv',
            'LVastusLateralisEmgEnv', 'RVastusLateralisEmgEnv',
            'LPeroneusLongusEmgEnv',  'RPeroneusLongusEmgEnv',
            'LBicepsBrachiiEmg',   'RBicepsBrachiiEmg',
            'LSemitendinosusEmg',  'RSemitendinosusEmg',
            'LVastusLateralisEmg', 'RVastusLateralisEmg',
            'LPeroneusLongusEmg',  'RPeroneusLongusEmg'],
        'ycoords': [
            3, 4, 3, 4, 3, 4, 3, 4,
            0, 1, 0, 1, 0, 1, 0, 1],
        'xcoords': [
            0, 0, 2, 2, 3, 3, 5, 5,
            0, 0, 2, 2, 3, 3, 5, 5]
        })
    return locals()
