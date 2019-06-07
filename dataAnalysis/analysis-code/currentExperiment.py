import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch

#  from exp201901201200 import *
from exp201901211000 import *
'''
def parseAnalysisOptions(optsDict):
    ns5FileName = 'Trial00{}'.format(trialIdx)
    miniRCTrial = miniRCTrialLookup[trialIdx]

    #  make placeholders for interpolation functions
    interpFunINStoNSP = {
        key: [None for i in value.keys()]
        for key, value in sessionTapRangesNSP.items()
        }
    interpFunHUTtoINS = {
        key: [None for i in value.keys()]
        for key, value in sessionTapRangesNSP.items()
        }

    scratchFolder = os.path.join(scratchPath, experimentName)
    if not os.path.exists(scratchFolder):
        os.makedirs(scratchFolder, exist_ok=True)

    nspOpts = {
        'excludeClus': [],
        'calcRigEvents': None,
        'spikeWindow': [-32, 64]
        }

    insOpts = {
        'elecIDs': range(17),
        'excludeClus': [],
        'forceRecalc': True,
        'detectStim': True,
        'stimDetectionOpts': None
        }

    stimDetectOptsByChannelDefault = {grpIdx: {
        0: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
        1: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
        2: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1},
        3: {'detectChannels': ['ins_td2', 'ins_td3'], 'thres': 1}
        } for grpIdx in range(4)}

    commonStimDetectionOpts = {
        'stimDetectOptsByChannel': stimDetectOptsByChannelDefault,
        'fixedDelay': -60e-3,
        'delayByFreqMult': 1, 'minDur': 0.2,
        'cyclePeriodCorrection': 20e-3,
        'plotAnomalies': False,
        'recalculateExpectedOffsets': True,
        'plotting': [],  # range(1, 1000, 5)
        }
        
    miniRCStimDetectionOpts = {
            'minDist': 1.2,
            'gaussWid': 200e-3,
            'maxSpikesPerGroup': 0,
        }
    fullStimDetectionOpts = {
        'minDist': 0.2,
        'gaussWid': 100e-3,
        'maxSpikesPerGroup': 1,
        }

    miniRCRigInputs = {
        'tapSync': 'ainp7',
        'simiTrigs': 'ainp8'
        }
    fullRigInputs = {
        'A+': 'ainp1',
        'B+': 'ainp2',
        'Z+': 'ainp3',
        'A-': 'ainp5',
        'B-': 'ainp4',
        'Z-': 'ainp6',
        'rightBut': 'ainp11',
        'leftBut': 'ainp12',
        'rightLED': 'ainp9',
        'leftLED': 'ainp10',
        'simiTrigs': 'ainp8',
        'tapSync': 'ainp7',
        }
        
    nspOpts['calcRigEvents'] = not miniRCTrial
    if miniRCTrial:
        #  override with settings for detecting cycling stim trains
        insOpts['stimDetectionOpts'].update(miniRCStimDetectionOpts)
        #  only parse sync lines
        eventInfo = {'inputIDs': miniRCRigInputs}
    else:
        insOpts['stimDetectionOpts'].update(fullStimDetectionOpts)
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': fullRigInputs}
    nspOpts.update(dict(eventInfo=eventInfo))
    insOpts['stimDetectionOpts'].update(commonStimDetectionOpts)
    
    nspCmpPath = os.path.join('.', 'nsp_map.cmp')
    cmpDF = tdch.cmpToDF(nspCmpPath)
    # make .prb file for spike sorting
    #  groupIn={'xcoords': 2, 'ycoords': 2}
    if remakePrb:
        nspCsvPath = os.path.join('.', 'nsp_map.csv')
        cmpDF.to_csv(nspCsvPath)
        tdch.cmpDFToPrb(
            cmpDF, filePath=nspPrbPath,
            names=['elec', 'ainp'],
            groupIn={'xcoords': 1, 'ycoords': 1})

    #  should rename these to something more intuitive
    #  paths relevant to individual trials
    processedFolder = os.path.join(
        remoteBasePath, 'processed', experimentName)
    if not os.path.exists(processedFolder):
        os.makedirs(processedFolder, exist_ok=True)

    analysisDataPath = os.path.join(
        #  remoteBasePath, 'processed', experimentName,
        scratchFolder,
        ns5FileName + '_analyze.nix')
    trialBasePath = os.path.join(
        scratchFolder,
        ns5FileName + '.nix')

    insDataPath = os.path.join(
        scratchFolder,
        ns5FileName + '_ins.nix')
    binnedSpikePath = os.path.join(
        scratchFolder,
        ns5FileName + '_binarized.nix')
    trialTriggeredPath = os.path.join(
        scratchFolder,
        ns5FileName + '_triggered.nix')

    #  paths relevant to the entire experimental day
    estimatorPath = os.path.join(
        #  remoteBasePath, 'processed', experimentName,
        scratchFolder,
        experimentName + '_estimator.joblib')
    experimentDataPath = os.path.join(
        #  remoteBasePath, 'processed', experimentName,
        scratchFolder,
        experimentName + '_analyze.nix')
    experimentTriggeredPath = os.path.join(
        #  remoteBasePath, 'processed', experimentName,
        scratchFolder,
        experimentName + '_triggered.nix')
    experimentBinnedSpikePath = os.path.join(
        #  remoteBasePath, 'processed', experimentName,
        scratchFolder,
        experimentName + '_binarized.nix')

    figureFolder = os.path.join(
        remoteBasePath, 'figures', experimentName
        )
    alignedRastersFolder = os.path.join(figureFolder, 'alignedRasters')
    if not os.path.exists(alignedRastersFolder):
        os.makedirs(alignedRastersFolder, exist_ok=True)
    alignedFeaturesFolder = os.path.join(figureFolder, 'alignedFeatures')
    if not os.path.exists(alignedFeaturesFolder):
        os.makedirs(alignedFeaturesFolder, exist_ok=True)
    spikeSortingFiguresFolder = os.path.join(figureFolder, 'spikeSorting')
    if not os.path.exists(spikeSortingFiguresFolder):
        os.makedirs(spikeSortingFiguresFolder, exist_ok=True)

    rasterOpts = {
        'binInterval': 1e-3, 'binWidth': 30e-3,
        'windowSize': (-.5, .5),
        'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
        'separateByFunArgs': None,
        'alignTo': None,
        'separateByFunKWArgs': {'type': 'Classification'}
        }
    plotOpts = {
        'type': 'ticks', 'errorBar': 'sem',
        'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}

    return bla
'''
