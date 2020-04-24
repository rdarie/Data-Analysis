def getExpOpts():
    #
    miniRCBlockLookup = {
        3: False,
        }
    RCBlockLookup = {
        3: True,
        }
    RippleBlockLookup = {
        3: True,
        }
    
    experimentName = '202003181300-Peep'
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
        3: [],
        }

    synchInfo = {'delsys': {}, 'nsp': {}, 'ins': {}}
    synchInfo['delsys'][3] = {'timeRanges': [4, 2048], 'chooseCrossings': slice(None)}
    synchInfo['nsp'][3] = {'timeRanges': [2, 2060], 'chooseCrossings': slice(None)}
    
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
        '202003181300-Peep': [3],
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
        3: [
            [2, 2060]
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
    return locals()
