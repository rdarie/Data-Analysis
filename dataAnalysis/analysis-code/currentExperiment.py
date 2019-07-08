import os, pdb
#  import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import importlib


def parseAnalysisOptions(trialIdx=1, experimentShorthand=None):
    optsModule = importlib.import_module(experimentShorthand, package=None)
    expOpts = optsModule.getExpOpts()
    #  globals().update(expOpts)
    
    #  remote paths
    remoteBasePath = '..'
    nspPrbPath = os.path.join('.', 'nsp_map.prb')
    scratchPath = '/gpfs/scratch/rdarie/rdarie/Murdoc Neural Recordings'
    insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
    experimentName = expOpts['experimentName']
    nspFolder = os.path.join(remoteBasePath, 'raw', experimentName)
    oeFolder = os.path.join(remoteBasePath, 'raw', experimentName, 'open_ephys')
    sipFolder = os.path.join(remoteBasePath, 'raw', experimentName, 'ins_sip')
    ns5FileName = 'Trial{:0>3}'.format(trialIdx)
    miniRCTrialLookup = expOpts['miniRCTrialLookup']
    
    try:
        openEphysChanNames = expOpts['openEphysChanNames']
    except Exception:
        openEphysChanNames = []
    openEphysFilterOpts = {
        'matched': {
            'type': 'sin',
            'Wn': 60,
            'nHarmonics': 10,
            'nCycles': 3
        },
        'bandstop': {
            'Wn': 60,
            'nHarmonics': 1,
            'Q': 20,
            'N': 4,
            'rp': 1,
            'btype': 'bandstop',
            'ftype': 'cheby1'
        },
        'low': {
            'Wn': 500,
            'N': 4,
            'btype': 'low',
            'ftype': 'butter'
        }
    }
    miniRCTrial = miniRCTrialLookup[trialIdx]
    
    gpfaOpts = {
        'xDim': 3,
        'segLength': 20,
        'binWidth': 30,
        'kernSD': 50,
        'installFolder': '/gpfs_home/rdarie/Github/NeuralTraj'
    }

    defaultTapDetectOpts = {
        'iti': 0.2,
        'keepIndex': slice(None)
        }
    tapDetectOpts = expOpts['tapDetectOpts']
    for trialKey in tapDetectOpts.keys():
        for trialSegmentKey in tapDetectOpts[trialKey].keys():
            for key in defaultTapDetectOpts.keys():
                if key not in tapDetectOpts[trialKey][trialSegmentKey].keys():
                    tapDetectOpts[trialKey][trialSegmentKey].update(
                        {key: defaultTapDetectOpts[key]}
                        )
    defaultSessionTapRangesNSP = {
        'keepIndex': slice(None)
        }
    sessionTapRangesNSP = expOpts['sessionTapRangesNSP']
    for trialKey in sessionTapRangesNSP.keys():
        for trialSegmentKey in sessionTapRangesNSP[trialKey].keys():
            for key in defaultSessionTapRangesNSP.keys():
                if key not in sessionTapRangesNSP[trialKey][trialSegmentKey].keys():
                    sessionTapRangesNSP[trialKey][trialSegmentKey].update(
                        {key: defaultSessionTapRangesNSP[key]}
                        )
    #  make placeholders for interpolation functions
    interpFunINStoNSP = {
        key: [None for i in value.keys()]
        for key, value in sessionTapRangesNSP.items()
        }
    interpFunHUTtoINS = {
        key: [None for i in value.keys()]
        for key, value in sessionTapRangesNSP.items()
        }

    trialFilesFrom = {
        'utah': {
            'origin': 'mat',
            'experimentName': experimentName,
            'folderPath': nspFolder,
            'ns5FileName': ns5FileName,
            'calcRigEvents': True,
            'spikeWindow': [-32, 64]
            }
        }
    spikeWindow = trialFilesFrom['utah']['spikeWindow']
    jsonSessionNames = expOpts['jsonSessionNames']
    trialFilesStim = {
        'ins': {
            'origin': 'ins',
            'experimentName': experimentName,
            'folderPath': insFolder,
            'ns5FileName': ns5FileName,
            'jsonSessionNames': jsonSessionNames[trialIdx],
            'elecIDs': range(17),
            'excludeClus': [],
            'forceRecalc': True,
            'detectStim': True,
            'getINSkwargs': {}
            }
        }
    stimDetectChans = expOpts['stimDetectChans']
    stimDetectThres = expOpts['stimDetectThres']
    stimDetectOptsByChannelDefault = {grpIdx: {
        0: {'detectChannels': stimDetectChans, 'thres': stimDetectThres},
        1: {'detectChannels': stimDetectChans, 'thres': stimDetectThres},
        2: {'detectChannels': stimDetectChans, 'thres': stimDetectThres},
        3: {'detectChannels': stimDetectChans, 'thres': stimDetectThres}
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
        
    trialFilesStim['ins']['getINSkwargs'].update(commonStimDetectionOpts)
    if miniRCTrial:
        #  override with settings for detecting cycling stim trains
        trialFilesStim['ins']['getINSkwargs'].update(miniRCStimDetectionOpts)
        #  only parse sync lines
        eventInfo = {'inputIDs': miniRCRigInputs}
    else:
        trialFilesStim['ins']['getINSkwargs'].update(fullStimDetectionOpts)
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': fullRigInputs}

    trialFilesFrom['utah']['calcRigEvents'] = not miniRCTrial
    trialFilesFrom['utah'].update({'eventInfo': eventInfo})
    
    nspCmpPath = os.path.join('.', 'nsp_map.cmp')
    cmpDF = prb_meta.cmpToDF(nspCmpPath)
    
    remakePrb = expOpts['remakePrb']
    if remakePrb:
        nspCsvPath = os.path.join('.', 'nsp_map.csv')
        cmpDF.to_csv(nspCsvPath)
        prb_meta.cmpDFToPrb(
            cmpDF, filePath=nspPrbPath,
            names=['elec', 'ainp'],
            groupIn={'xcoords': 1, 'ycoords': 1})

    #  should rename these to something more intuitive
    #  paths relevant to individual trials
    processedFolder = os.path.join(
        remoteBasePath, 'processed', experimentName)
    if not os.path.exists(processedFolder):
        os.makedirs(processedFolder, exist_ok=True)

    scratchFolder = os.path.join(scratchPath, experimentName)
    if not os.path.exists(scratchFolder):
        os.makedirs(scratchFolder, exist_ok=True)

    #  paths relevant to single trial files
    triFolder = os.path.join(
        scratchFolder, 'tdc_Trial{:0>3}'.format(trialIdx))
    
    if isinstance(expOpts['triFolderSourceBase'], int):
        triFolderSource = os.path.join(
            scratchFolder, 'tdc_Trial{:0>3}'.format(expOpts['triFolderSourceBase']))
    else:
        triFolderSource = os.path.join(
            scratchPath, expOpts['triFolderSourceBase'])

    analysisDataPath = os.path.join(
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

    #  paths relevant to the entire experimental day
    estimatorPath = os.path.join(
        scratchFolder,
        experimentName + '_estimator.joblib')
    experimentDataPath = os.path.join(
        scratchFolder,
        experimentName + '_analyze.nix')
    experimentBinnedSpikePath = os.path.join(
        scratchFolder,
        experimentName + '_binarized.nix')

    figureFolder = os.path.join(
        remoteBasePath, 'figures', experimentName
        )
    if not os.path.exists(figureFolder):
        os.makedirs(figureFolder, exist_ok=True)

    #  alignedRastersFolder = os.path.join(figureFolder, 'alignedRasters')
    #  if not os.path.exists(alignedRastersFolder):
    #      os.makedirs(alignedRastersFolder, exist_ok=True)
    #  alignedFeaturesFolder = os.path.join(figureFolder, 'alignedFeatures')
    #  if not os.path.exists(alignedFeaturesFolder):
    #      os.makedirs(alignedFeaturesFolder, exist_ok=True)
    spikeSortingFiguresFolder = os.path.join(figureFolder, 'spikeSorting')
    if not os.path.exists(spikeSortingFiguresFolder):
        os.makedirs(spikeSortingFiguresFolder, exist_ok=True)
    
    alignedAsigsKWargs = {}
    if miniRCTrial:
        alignedAsigsKWargs.update(dict(
            amplitudeColumn='amplitude',
            programColumn='program',
            electrodeColumn='electrode'))
    else:
        alignedAsigsKWargs.update(dict(
            amplitudeColumn='amplitudeFuzzy',
            programColumn='programFuzzy',
            electrodeColumn='electrodeFuzzy'))
    
    alignedAsigsChunkSize = 500
    rasterOpts = {
        'binInterval': 1e-3, 'binWidth': 30e-3, 'smoothKernelWidth': 50e-3,
        'windowSizes': {
            'short': (-0.5, 0.5),
            'long': (-2.25, 2.25),
            'RC': (-0.2, 0.5),
            'miniRC': (-1, 1)
        },
        'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
        'separateByFunArgs': None,
        'alignTo': None,
        'separateByFunKWArgs': {'type': 'Classification'}
        }
    statsTestOpts = dict(
        testStride=20e-3,
        testWidth=100e-3,
        tStart=0,
        tStop=500e-3,
        pThresh=1e-24,
        correctMultiple=True
        )
    relplotKWArgs = dict(
        ci='sem', estimator='mean',
        palette="ch:0.6,-.2,dark=.2,light=0.7,reverse=1",
        height=5, aspect=1.5, kind='line')
    twinRelplotKWArgs = dict(
        palette="ch:0.6,-.2,dark=.2,light=0.7,reverse=1",
        func1_kws={'marker': '|', 'alpha': 0.6},
        func2_kws={'ci': 'sem'},
        facet1_kws={'sharey': False},
        facet2_kws={'sharey': True},
        height=5, aspect=1.5,
        kind1='scatter', kind2='line'
    )
    plotOpts = {
        'type': 'ticks', 'errorBar': 'sem',
        'pageSize': (6, 12), 'removeOutliers': (0.01, 0.975)}
    try:
        experimentsToAssemble = expOpts['experimentsToAssemble']
        trialsToAssemble = []
        for key in sorted(experimentsToAssemble.keys()):
            val = experimentsToAssemble[key]
            for tIdx in val:
                trialsToAssemble.append(
                    os.path.join(
                        scratchPath, key, 'Trial00{}.nix'.format(tIdx)
                    )
                )
    except Exception:
        pass

    return expOpts, locals()
