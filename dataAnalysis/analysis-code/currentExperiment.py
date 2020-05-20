import os, pdb, re, platform
#  import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import importlib


def parseAnalysisOptions(blockIdx=1, experimentShorthand=None):
    plottingFigures = False
    plotBlocking = True
    remakePrb = False
    #
    with open("../../paths.py") as fp:
        d = {}
        exec(fp.read(), d)
        ccvUsername = d['ccvUsername']
        scratchPath = d['scratchPath']
        remoteBasePath = d['remoteBasePath']
    #
    optsModule = importlib.import_module(
        'experimentOptions.' + experimentShorthand, package=None)
    expOpts = optsModule.getExpOpts()
    #  globals().update(expOpts)
    #  remote paths
    # if platform.system() == 'Linux':
    #     remoteBasePath = '..'
    #     scratchPath = '/gpfs/scratch/{}/{}/Neural Recordings'.format(
    #         ccvUsername, ccvUsername)
    # else:
    #     remoteBasePath = os.path.join('E:', 'Neural Recordings')
    #     scratchPath = os.path.join('E:', 'Neural Recordings', 'scratch')

    nspPrbPath = os.path.join('.', 'nsp_map.prb')
    insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
    experimentName = expOpts['experimentName']
    assembledName = ''
    nspFolder = os.path.join(remoteBasePath, 'raw', experimentName)
    simiFolder = os.path.join(remoteBasePath, 'processed', experimentName, 'simi')
    oeFolder = os.path.join(remoteBasePath, 'raw', experimentName, 'open_ephys')
    sipFolder = os.path.join(remoteBasePath, 'raw', experimentName, 'ins_sip')
    ns5FileName = 'Block{:0>3}'.format(blockIdx)
    # miniRCBlockLookup = expOpts['miniRCBlockLookup']
    # RCBlockLookup = expOpts['RCBlockLookup']
    # RippleBlockLookup = expOpts['RippleBlockLookup']
    blockExperimentType = expOpts['blockExperimentTypeLookup'][blockIdx]
    try:
        openEphysChanNames = expOpts['openEphysChanNames']
    except Exception:
        openEphysChanNames = []
    #
    EMGFilterOpts = {
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
            'N': 8,
            'rp': 1,
            'btype': 'bandstop',
            'ftype': 'cheby1'
        },
        'low': {
            'Wn': 1000,
            'N': 8,
            'btype': 'low',
            'ftype': 'butter'
        },
        'high': {
            'Wn': 15,
            'N': 10,
            'btype': 'high',
            'ftype': 'butter'
        }
    }
    #
    LFPFilterOpts = {
        'bandstop': {
            'Wn': 60,
            'nHarmonics': 1,
            'Q': 20,
            'N': 1,
            'rp': 1,
            'btype': 'bandstop',
            'ftype': 'butter'
        },
        'low': {
            'Wn': 1000,
            'N': 2,
            'btype': 'low',
            'ftype': 'butter'
        },
        'high': {
            'Wn': 5,
            'N': 2,
            'btype': 'high',
            'ftype': 'butter'
        }
    }
    
    gpfaOpts = {
        'xDim': 3,
        'segLength': 20,
        'binWidth': 30,
        'kernSD': 50,
        'installFolder': '/gpfs_home/{}/Github/NeuralTraj'
    }

    defaultTapDetectOpts = {
        'iti': 0.2,
        'keepIndex': slice(None)
        }
    tapDetectOpts = expOpts['synchInfo']['ins']
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
    sessionTapRangesNSP = expOpts['synchInfo']['nsp']
    if not blockExperimentType == 'isi':
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
            'spikeWindow': [-25, 50]
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
            'jsonSessionNames': jsonSessionNames[blockIdx],
            'elecIDs': range(17),
            'excludeClus': [],
            'upsampleRate': 4,
            'interpKind': 'linear',
            'forceRecalc': True,
            'detectStim': expOpts['detectStim'],
            'getINSkwargs': {}
            }
        }
    stimDetectChansDefault = expOpts['stimDetectChansDefault']
    stimDetectThresDefault = expOpts['stimDetectThresDefault']
    stimDetectOptsByChannelDefault = {grpIdx: {
        0: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
        1: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
        2: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault},
        3: {'detectChannels': stimDetectChansDefault, 'thres': stimDetectThresDefault}
        } for grpIdx in range(4)}
    stimDetectOptsByChannel = stimDetectOptsByChannelDefault
    stimDetectOptsByChannel.update(expOpts['stimDetectOptsByChannelSpecific'])
    if 'stimDetectOverrideStartTimes' in expOpts:
        overrideStartTimes = expOpts['stimDetectOverrideStartTimes'][blockIdx]
    else:
        overrideStartTimes = None
    commonStimDetectionOpts = {
        'stimDetectOptsByChannel': stimDetectOptsByChannelDefault,
        'spikeWindow': spikeWindow,
        'offsetFromPeak': 4e-3,
        'cyclePeriodCorrection': 20e-3,
        'predictSlots': True, 'snapToGrid': True,
        'plotAnomalies': False,
        'overrideStartTimes': overrideStartTimes,
        'plotting': []  # range(1, 1000, 5) [] range(1000)
        }
    miniRCStimDetectionOpts = {
        'minDist': 1.2,
        'gaussWid': 100e-3,
        'treatAsSinglePulses': False
        }
    RCStimDetectionOpts = {
        'minDist': 0.25,
        'gaussWid': 120e-3,
        'treatAsSinglePulses': True
        }
    fullStimDetectionOpts = {
        'minDist': 0.2,
        'gaussWid': 100e-3,
        'treatAsSinglePulses': False
        }
    miniRCRigInputs = {
        'tapSync': 'ainp7',
        'simiTrigs': 'ainp8'
        }
    RCRigInputs = {
        'kinectSync': 'ainp16',
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
    #
    trialFilesStim['ins']['getINSkwargs'].update(commonStimDetectionOpts)
    if blockExperimentType == 'proprio-miniRC':
        #  override with settings for detecting cycling stim trains
        trialFilesStim['ins']['getINSkwargs'].update(miniRCStimDetectionOpts)
        #  only parse sync lines
        eventInfo = {'inputIDs': miniRCRigInputs}
    elif blockExperimentType == 'proprio-RC':
        trialFilesStim['ins']['getINSkwargs'].update(RCStimDetectionOpts)
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': RCRigInputs}
    elif blockExperimentType == 'isi':
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': dict()}
    else:
        trialFilesStim['ins']['getINSkwargs'].update(fullStimDetectionOpts)
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': fullRigInputs}
    #
    trialFilesFrom['utah']['calcRigEvents'] = ((blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC'))
    trialFilesFrom['utah'].update({'eventInfo': eventInfo})
    
    nspCmpPath = os.path.join('.', 'nsp_map.cmp')
    cmpDF = prb_meta.cmpToDF(nspCmpPath)
    experimentDateStr = re.search(r'(\d*)', experimentName).groups()[0]
    # pdb.set_trace()
    impedances = prb_meta.getLatestImpedance(
        impedanceFilePath=os.path.join(remoteBasePath, 'impedances.h5'),
        recordingDateStr=experimentDateStr, elecType='utah')
    impedances['elec'] = cmpDF['label'].iloc[:impedances.shape[0]].to_numpy()
    impedances.set_index('elec', inplace=True)
    impedancesRipple = prb_meta.getLatestImpedance(
        impedanceFilePath=os.path.join(remoteBasePath, 'impedances.h5'),
        recordingDateStr=experimentDateStr, elecType='isi_paddle')
    #
    if remakePrb:
        nspCsvPath = os.path.join('.', 'nsp_map.csv')
        cmpDF.to_csv(nspCsvPath)
        prb_meta.cmpDFToPrb(
            cmpDF, filePath=nspPrbPath,
            names=['elec', 'ainp'],
            #names=['elec'],
            groupIn={'xcoords': 2, 'ycoords': 2})
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
        scratchFolder, 'tdc_Block{:0>3}'.format(blockIdx))
    
    if isinstance(expOpts['triFolderSourceBase'], int):
        triFolderSource = os.path.join(
            scratchFolder, 'tdc_Block{:0>3}'.format(expOpts['triFolderSourceBase']))
    else:
        triFolderSource = os.path.join(
            scratchPath, expOpts['triFolderSourceBase'])
    analysisDataPath = os.path.join(
        scratchFolder, '{}',
        ns5FileName + '_analyze.nix')
    trialBasePath = os.path.join(
        scratchFolder,
        ns5FileName + '.nix')
    insDataPath = os.path.join(
        scratchFolder,
        ns5FileName + '_ins.nix')
    oeDataPath = os.path.join(
        scratchFolder,
        ns5FileName + '_oe.nix')
    oeRawDataPath = os.path.join(
        scratchFolder,
        ns5FileName + '_raw_oe.nix')
    binnedSpikePath = os.path.join(
        scratchFolder, '{}',
        ns5FileName + '_binarized.nix')
    #  paths relevant to the entire experimental day
    experimentDataPath = os.path.join(
        scratchFolder, '{}',
        assembledName + '_analyze.nix')
    experimentBinnedSpikePath = os.path.join(
        scratchFolder, '{}',
        assembledName + '_binarized.nix')
    #
    figureFolder = os.path.join(
        remoteBasePath, 'figures', experimentName
        )
    if not os.path.exists(figureFolder):
        os.makedirs(figureFolder, exist_ok=True)
    #
    alignedRastersFolder = os.path.join(figureFolder, 'alignedRasters')
    if not os.path.exists(alignedRastersFolder):
        os.makedirs(alignedRastersFolder, exist_ok=True)
    alignedFeaturesFolder = os.path.join(figureFolder, 'alignedFeatures')
    if not os.path.exists(alignedFeaturesFolder):
        os.makedirs(alignedFeaturesFolder, exist_ok=True)
    spikeSortingFiguresFolder = os.path.join(figureFolder, 'spikeSorting')
    if not os.path.exists(spikeSortingFiguresFolder):
        os.makedirs(spikeSortingFiguresFolder, exist_ok=True)
    GLMFiguresFolder = os.path.join(figureFolder, 'GLM')
    if not os.path.exists(GLMFiguresFolder):
        os.makedirs(GLMFiguresFolder, exist_ok=True)
    #
    alignedAsigsKWargs = dict(
        amplitudeColumn='amplitude',
        programColumn='program',
        electrodeColumn='electrode',
        removeFuzzyName=False)
    # alignedAsigsKWargs.update(dict(
    #     windowSize=(-50e-3, 350e-3),
    #     decimate=3))
    # alignedAsigsKWargs.update(dict(
    #     windowSize=(-1e-3, 9e-3),
    #     decimate=1))
    alignedAsigsKWargs.update(dict(
        windowSize=(-250e-6, 2250e-6),
        decimate=1))
    # alignedAsigsKWargs.update(dict(
    #     windowSize=None, decimate=1))
    # if (miniRCBlock or RCBlock):
    #     alignedAsigsKWargs.update(dict(
    #         amplitudeColumn='amplitude',
    #         programColumn='program',
    #         electrodeColumn='electrode',
    #         removeFuzzyName=False))
    # else:
    #     alignedAsigsKWargs.update(dict(
    #         amplitudeColumn='amplitudeFuzzy',
    #         programColumn='programFuzzy',
    #         electrodeColumn='electrodeFuzzy',
    #         removeFuzzyName=True))
    overrideChanNames = None
    # overrideChanNames = [
    #     'elec75#0', 'elec75#1', 'elec83#0', 'elec78#0', 'elec78#1']
    overrideUnitNames = None
    # overrideUnitNames = [
    #     'elec75#0', 'elec75#1', 'elec83#0', 'elec78#0', 'elec78#1']
    alignedAsigsChunkSize = 150000
    rasterOpts = {
        # 'binInterval': 2e-4, 'binWidth': 10e-3, 'smoothKernelWidth': 10e-3,  # 5 kHz, EMG Lo-Res
        # 'binInterval': 1e-3, 'binWidth': 10e-3, 'smoothKernelWidth': 10e-3,
        # 'binInterval': 1e-3, 'binWidth': 30e-3, 'smoothKernelWidth': 50e-3,
        # 'binInterval': 0.2e-3, 'binWidth': 5e-3, 'smoothKernelWidth': 10e-3,
        'binInterval': (3e4) ** (-1), 'binWidth': 5e-3, 'smoothKernelWidth': 10e-3,  # 30 kHz, Full-Res
        # 'binInterval': 1e-4, 'binWidth': 5e-3, 'smoothKernelWidth': 10e-3,  # 10 kHz, EMG Hi-Res
        'windowSizes': {
            'extraShort': (-0.15, 0.4),
            'extraExtraShort': (-0.15, 0.01),
            'short': (-0.5, 0.5),
            'long': (-2.25, 2.25),
            'RC': (-0.33, 0.33),
            'miniRC': (-1, 1)},
        'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
        'separateByFunArgs': None,
        'alignTo': None,
        'separateByFunKWArgs': {'type': 'Classification'}
        }
    statsTestOpts = dict(
        testStride=500e-3,
        testWidth=500e-3,
        tStart=None,
        tStop=None,
        pThresh=1e-2,
        correctMultiple=True
        )
    relplotKWArgs = dict(
        ci='sem',
        # ci=95, n_boot=1000,
        estimator='mean',
        # estimator=None, units='t',
        palette="ch:0.6,-.3,dark=.1,light=0.7,reverse=1",
        height=6, aspect=30, kind='line')
    vLineOpts = {'color': 'm', 'alpha': 0.5}
    asigPlotShadingOpts = {
        'facecolor': vLineOpts['color'],
        'alpha': 0.1, 'zorder': -100}
    asigSigStarOpts = {
        'color': 'm',
        'linestyle': 'None',
        'markersize': 20,
        'marker': '*'
        }
    nrnRelplotKWArgs = dict(
        palette="ch:1.6,-.3,dark=.1,light=0.7,reverse=1",
        func1_kws={
            'marker': 'd', 's': 15, 'edgecolors': 'face',
            'alpha': 0.2, 'rasterized': True},
        func2_kws={'ci': 'sem'},
        facet1_kws={'sharey': False},
        facet2_kws={'sharey': True},
        height=5, aspect=3,
        kind1='scatter', kind2='line')
    nrnVLineOpts = {'color': 'y'}
    nrnBlockShadingOpts = {
        'facecolor': nrnVLineOpts['color'],
        'alpha': 0.1, 'zorder': -100}
    nrnSigStarOpts = {
        'color': 'y',
        # 'edgecolors': 'face',
        'linestyle': 'None',
        'markersize': 20,
        'marker': '*'}
    plotOpts = {
        'type': 'ticks', 'errorBar': 'sem',
        'pageSize': (6, 22),
        'removeOutliers': (0.01, 0.975)}
    try:
        experimentsToAssemble = expOpts['experimentsToAssemble']
        trialsToAssemble = []
        for key in sorted(experimentsToAssemble.keys()):
            val = experimentsToAssemble[key]
            for tIdx in val:
                trialsToAssemble.append(
                    os.path.join(
                        scratchPath, key, '{}',
                        'Block00{}.nix'.format(tIdx)
                    )
                )
    except Exception:
        pass
    glmOptsLookup = {
        'ensembleHistoryLen': .30,
        'covariateHistoryLen': .30,
        'nHistoryBasisTerms': 5,
        'glm_50msec': dict(
            subsampleOpts=dict(rollingWindow=50, decimate=50),
            covariateSpacing=25e-3),
        'glm_30msec': dict(
            subsampleOpts=dict(rollingWindow=30, decimate=30),
            covariateSpacing=15e-3),
        'glm_20msec': dict(
            subsampleOpts=dict(rollingWindow=20, decimate=20),
            covariateSpacing=10e-3),
        'glm_10msec': dict(
            subsampleOpts=dict(rollingWindow=10, decimate=10),
            covariateSpacing=10e-3),
        'glm_5msec': dict(
            subsampleOpts=dict(rollingWindow=5, decimate=5),
            covariateSpacing=10e-3),
        'glm_3msec': dict(
            subsampleOpts=dict(rollingWindow=3, decimate=3),
            covariateSpacing=10e-3),
        'glm_1msec': dict(
            subsampleOpts=dict(rollingWindow=None, decimate=1),
            covariateSpacing=10e-3),
    }
    return expOpts, locals()
