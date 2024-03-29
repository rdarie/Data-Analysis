import numpy as np
import pandas as pd

def getExpOpts():
    blockExperimentTypeLookup = {
        1: 'proprio-RC',
        2: 'proprio-RC',
        3: 'proprio-RC',
        }
    fullRigInputs = {
        'A+': 'ainp12',
        'B+': 'ainp11',
        'Z+': 'ainp3',
        'A-': 'ainp9',
        'B-': 'ainp10',
        'Z-': 'ainp4',
        'rightBut': 'ainp5',
        'leftBut': 'ainp6',
        'rightLED': 'ainp7',
        'leftLED': 'ainp8',
        'simiTrigs': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        'tapSync': 'ainp1',
        'delsysSynch': 'ainp2',
        }
    miniRCRigInputs = {
        'tapSync': 'ainp1',
        'simiTrigs': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        'delsysSynch': 'ainp2',
        }
    RCRigInputs = {
        'tapSync': 'ainp1',
        'kinectSync': 'ainp16',
        'forceX': 'ainp14',
        'forceY': 'ainp15',
        'delsysSynch': 'ainp2',
        }
    experimentName = '202102151100-Rupert'
    deviceName = 'DeviceNPC700246H'
    subjectName = 'Rupert'
    #
    jsonSessionNames = {
        #  per blo
        1: ['Session1613402898768', 'Session1613403672383'], # ~148.6
        2: ['Session1613406217371'],
        3: [],
        }
    synchInfo = {'nform': {}, 'nsp': {}, 'ins': {}, 'nspForDelsys': {}, 'delsysToNsp': {}}
    for blockIdx in blockExperimentTypeLookup.keys():
        synchInfo['nspForDelsys'][blockIdx] = {
            'synchChanName': 'ainp2'
            }
        synchInfo['delsysToNsp'][blockIdx] = {
            'synchChanName': 'AnalogInputAdapterAnalog'
        }
    # populate with defaults
    for blockIdx in jsonSessionNames.keys():
        synchInfo['ins'][blockIdx] = {}
        for idx, sessionName in enumerate(jsonSessionNames[blockIdx]):
            synchInfo['ins'][blockIdx][idx] = {
                'timeRanges': None,
                'synchChanName': ['ins_td0', 'ins_td2'],
                'synchStimUnitName': ['g0p0#0'],
                'synchByXCorrTapDetectSignal': False,
                'xCorrSamplingRate': None,
                'xCorrGaussWid': 50e-3,
                'minStimAmp': 0,
                'unixTimeAdjust': None,
                'thres': 5,
                'iti': 50e-3,
                'minAnalogValue': None,
                'keepIndex': slice(None)
                }
    ############################################################
    ############################################################
    # manually add special instructions, e.g.
    # synchInfo['ins'][3][0].update({'minStimAmp': 0})
    #
    #
    ############################################################
    ############################################################
    synchInfo['nsp'] = {
        # per block
        i: {
            #  per trialSegment
            j: {
                'timeRanges': None, 'keepIndex': slice(None),
                'synchChanName': ['utah_artifact_0'], 'iti': 50e-3,
                'synchByXCorrTapDetectSignal': False,
                'unixTimeAdjust': None,
                'minAnalogValue': None, 'thres': 4}
            for j, sessionName in enumerate(jsonSessionNames[i])
            }
        for i in jsonSessionNames.keys()
        }
    ############################################################
    ############################################################
    # manually add special instructions, e.g
    # synchInfo['nsp'][2][0].update({'timeRanges': [(40, 9999)]})
    #
    #
    ############################################################
    #  overrideSegmentsForTapSync
    #  if not possible to use taps, override with good taps from another segment
    #  not ideal, because segments are only synchronized to the nearest **second**
    overrideSegmentsForTapSync = {
        #  each key is a Block
        #  1: {0: 'coarse'},  # e.g. for ins session 0, use the coarse alignment based on system unix time
        #  2: {2: 1},  # e.g. for ins session 2, use the alignment of ins session 1
        #
        }
    #
    # options for stim artifact detection
    stimDetectOverrideStartTimes = {
        #  each key is a Block
        1: {
            #  each key is an ins session
            0: [130.996, 370.780, 469.282, 546.28, 618.28, 713.06]
        },
        2: {
            #  each key is an ins session
            0: [
                132.681, 180.683, 220.183, 260.187, 300.182, 444.183, 487.686,
                ]
        },
        # 2: [96.43, 191.689],
    }
    detectStim = True
    stimDetectThresDefault = 250
    stimDetectChansDefault = ['ins_td0', 'ins_td2']
    stimDetectOptsByChannelSpecific = {
        # group
        0: {
            # program
            0: {'detectChannels': ['ins_td0'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            1: {'detectChannels': ['ins_td0'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            2: {'detectChannels': ['ins_td2'], 'thres': stimDetectThresDefault, 'useForSlotDetection': True},
            3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault, 'useForSlotDetection': True}
        }}
    #  Options relevant to the assembled trial files
    experimentsToAssemble = {
        '202102151100-Rupert': [1],
        }
    # Options relevant to the classifcation of proprio trials
    movementSizeBins = [0, 0.6, 1]
    movementSizeBinLabels = ['S', 'L']

    ############################################################
    ############################################################
    alignTimeBoundsLookup = None
    '''
    alignTimeBoundsLookup = {
        3: [
            [108, 99999]
            ],
        }
    '''
    #
    motorEncoderBoundsLookup = None
    '''
    motorEncoderBoundsLookup = {
        2: [
            [100, 772], [1173, 1896]
        ]
    }
    '''
    ############################################################
    ############################################################
    #   outlierDetectOptions = dict(
    #       targetEpochSize=100e-3,
    #       windowSize=(-.2, .8),
    #       twoTailed=True,
    #       )
    #
    minNConditionRepetitions = {
        'n': 1,
        'categories': ['amplitude', 'electrode', 'RateInHz']
        }
    #
    spikeSortingOpts = {
        'utah': {
            'asigNameList': [
                [
                    'utah{:d}'.format(i)
                    for i in range(1, 97)]
                ],
            'ainpNameList': [
                'ainp{:d}'.format(i)
                for i in range(1, 17)
            ],
            'electrodeMapPath': './Utah_SN6251_002374_Rupert.cmp',
            'rawBlockName': 'utah',
            ############################################################
            'excludeChans': [],  # CHECK
            'prbOpts': dict(
                contactSpacing=400,
                groupIn={
                    'xcoords': np.arange(-.1, 10.1, 1),
                    'ycoords': np.arange(-.1, 10.1, 1)}),
            'previewDuration': 480,
            'previewOffset': 0,
            'interpolateOutliers': True,
            'outlierThreshold': 1 - 1e-6,
            'shape_distance_threshold': None,
            'shape_boundary_threshold': None,
            'energy_reduction_threshold': 0.25,
            'make_classifier': True,
            'refit_projector': True,
            'n_max_peeler_passes': 2,
            'confidence_threshold': .5,
            'refractory_period': 2e-3,
            'triFolderSource': {
                'exp': experimentName, 'block': 3,
                'nameSuffix': 'spike_preview'},
            'triFolderDest': [
                {
                    'exp': experimentName, 'block': i,
                    'nameSuffix': 'mean_subtracted'}
                for i in [1, 2, 3]]
        }
    }

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

    def delsysCustomRenamer(featName):
        lookupSrs = pd.Series({
            'Avanti sensor 1: ': 'RightInnerThigh',
            'Avanti sensor 2: ': 'LeftInnerThigh',
            'Avanti sensor 3: ': 'RightGluteus',
            'Avanti sensor 4: ': 'LeftGluteus',
            'Avanti sensor 5: ': 'LeftQuad',
            'Avanti sensor 6: ': 'LeftHamstring',
            'Avanti sensor 7: ': 'RightQuad',
            'Avanti sensor 8: ': 'RightHamstring',
            'Avanti sensor 10: ': 'LeftShin',
            'Avanti sensor 11: ': 'LeftCalf',
            'Avanti sensor 12: ': 'RightShin',
            'Avanti sensor 13: ': 'RightCalf',
            'Analog Input adapter 16: ': 'Analog Input adapter'
            })
        featPrefix = featName.split(':')[0]
        chanNumStr = featPrefix.split(' ')[-1]
        nameMatchesMask = lookupSrs.index.str.contains(featPrefix + ': ')
        if nameMatchesMask.any():
            assert nameMatchesMask.sum() == 1
            inpStr = lookupSrs.index[nameMatchesMask][0]
            outStr = (
                lookupSrs[nameMatchesMask].iloc[0] +
                ' {}: '.format(chanNumStr))
            updatedFeat = featName.replace(inpStr, outStr)
            return updatedFeat
        else:
            return None
    
    return locals()
