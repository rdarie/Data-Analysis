def getExpOpts():
    #
    miniRCBlockLookup = {
        7: False,
        8: False,
        }
    RCBlockLookup = {
        7: True,
        8: True,
        }
    RippleBlockLookup = {
        7: True,
        8: True,
        }
    
    experimentName = '202004251400-Benchtop'
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
        7: [],
        8: [],
        }

    synchInfo = {'delsys': {}, 'nsp': {}, 'ins': {}}
    synchInfo['delsys'][7] = {'timeRanges': [0, 400], 'chooseCrossings': slice(None)}
    synchInfo['delsys'][8] = {'timeRanges': [10, 976], 'chooseCrossings': slice(None, 10)}
    #
    synchInfo['nsp'][7] = {'timeRanges': [0, 545], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][8] = {'timeRanges': [2, 964], 'chooseCrossings': slice(None, 10)}
    
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
        '202004251400-Benchtop': [7, 8],
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
        7: [
            [0, 545]
            ],
        8: [
            [0, 222]
            ]
        }
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
        'LPeroneusLongusEmg#0',
        'LSemitendinosusEmg#0',
        'LVastusLateralisEmg#0',
        # 'RPeroneusLongusEmg#0',
        # 'RSemitendinosusEmg#0',
        # 'RVastusLateralisEmg#0'
        ]
    
    delsysMapDict = ({
        'label': [
            # 'LBicepsBrachiiEmgEnv',   'RBicepsBrachiiEmgEnv',
            'LSemitendinosusEmgEnv',  'RSemitendinosusEmgEnv',
            'LVastusLateralisEmgEnv', 'RVastusLateralisEmgEnv',
            'LPeroneusLongusEmgEnv',  'RPeroneusLongusEmgEnv',
            # 'LBicepsBrachiiEmg',   'RBicepsBrachiiEmg',
            'LSemitendinosusEmg',  'RSemitendinosusEmg',
            'LVastusLateralisEmg', 'RVastusLateralisEmg',
            'LPeroneusLongusEmg',  'RPeroneusLongusEmg'],
        'ycoords': [
            # 3, 4,
            3, 4, 3, 4, 3, 4,
            # 0, 1,
            0, 1, 0, 1, 0, 1],
        'xcoords': [
            # 0, 0,
            2, 2, 3, 3, 5, 5,
            # 0, 0,
            2, 2, 3, 3, 5, 5]
        })
    return locals()
