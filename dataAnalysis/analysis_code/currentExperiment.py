import os, pdb, re, platform, traceback
#  import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import importlib


def parseAnalysisOptions(
        blockIdx=1, experimentShorthand=None):
    plottingFigures = False
    plotBlocking = True
    #
    with open("../../paths.py") as fp:
        d = {}
        exec(fp.read(), d)
        ccvUsername = d['ccvUsername']
        scratchPath = d['scratchPath']
        remoteBasePath = d['remoteBasePath']
    #
    optsModule = importlib.import_module(
        'dataAnalysis.analysis_code.experimentOptions.' + experimentShorthand,
        package=None)
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
    try:
        insFolder = os.path.join(
            remoteBasePath, 'ORCA Logs', expOpts['subjectName'])
    except Exception:
        insFolder = os.path.join(remoteBasePath, 'ORCA Logs')
    experimentName = expOpts['experimentName']
    assembledName = 'Block'
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
    if 'spikeSortingFilterOpts' not in expOpts:
        spikeSortingFilterOpts = {
            'low': {
                'Wn': 5000,
                'N': 8,
                'btype': 'low',
                'ftype': 'butter'
            },
            'high': {
                'Wn': 100,
                'N': 2,
                'btype': 'high',
                'ftype': 'butter'
            }
        }
    else:
        spikeSortingFilterOpts = expOpts['spikeSortingFilterOpts']
    if 'stimArtifactFilterOpts' not in expOpts:
        stimArtifactFilterOpts = {
            'high': {
                'Wn': 200,
                'N': 4,
                'btype': 'high',
                'ftype': 'butter'
            }
        }
    else:
        stimArtifactFilterOpts = expOpts['stimArtifactFilterOpts']
    tapDetectFilterOpts = {
        'high': {
            'Wn': 20,
            'N': 2,
            'btype': 'high',
            'ftype': 'butter'
        }
    }
    if 'outlierMaskFilterOpts' not in expOpts:
        outlierMaskFilterOpts = {
            'low': {
                'Wn': 1000,
                'N': 4,
                'btype': 'low',
                'ftype': 'butter'
            }}
    else:
        outlierMaskFilterOpts = expOpts['outlierMaskFilterOpts']
    
    gpfaOpts = {
        'xDim': 3,
        'segLength': 20,
        'binWidth': 30,
        'kernSD': 50,
        'installFolder': '/gpfs_home/{}/Github/NeuralTraj'
    }

    trialFilesFrom = {
        'utah': {
            'origin': 'mat',
            'experimentName': experimentName,
            'folderPath': nspFolder,
            'ns5FileName': ns5FileName,
            'calcRigEvents': (blockExperimentType == 'proprio') or (blockExperimentType == 'proprio-motionOnly'),
            'spikeWindow': [-24, 40]
            }
        }
    spikeWindow = trialFilesFrom['utah']['spikeWindow']
    try:
        jsonSessionNames = expOpts['jsonSessionNames']
    except Exception:
        traceback.print_exc()
        jsonSessionNames = {blockIdx: []}
    if blockIdx not in jsonSessionNames:
        jsonSessionNames[blockIdx] = []
    trialFilesStim = {
        'ins': {
            'origin': 'ins',
            'experimentName': experimentName,
            'folderPath': insFolder,
            'ns5FileName': ns5FileName,
            'jsonSessionNames': jsonSessionNames[blockIdx],
            'elecIDs': range(17),
            'excludeClus': [],
            'upsampleRate': 8,
            'eventsFromFirstInTrain': True,
            'interpKind': 'linear',
            # 'upsampleRate': 10,
            # 'interpKind': 'akima',
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
    commonStimDetectionOpts = {
        'stimDetectOptsByChannel': stimDetectOptsByChannelDefault,
        'spikeWindow': spikeWindow,
        'cyclePeriodCorrection': 20e-3,
        'plotAnomalies': False,
        'plotting':  range(0, 1000),  # [] range(1000)
        'overrideStartTimes': None
        }
    miniRCStimDetectionOpts = {
        'minDist': 1.2,
        'gaussWid': 100e-3,
        'offsetFromPeak': 1e-3,
        # 'artifactKeepWhat': 'first',
        'artifactKeepWhat': 'max',
        'expectRateProportionalStimOnDelay': True,
        'expectRateProportionalStimOffDelay': True,
        'predictSlots': False, 'snapToGrid': False,
        'treatAsSinglePulses': True
        }
    RCStimDetectionOpts = {
        'minDist': 0.2,
        'gaussWid': 50e-3,
        'offsetFromPeak': 1e-3,
        'artifactKeepWhat': 'max',
        'expectRateProportionalStimOnDelay': True,
        'expectRateProportionalStimOffDelay': True,
        'predictSlots': True, 'snapToGrid': True,
        'treatAsSinglePulses': True
        }
    fullStimDetectionOpts = {
        'minDist': 0.2,
        'gaussWid': 50e-3,
        'offsetFromPeak': 1e-3,
        'artifactKeepWhat': 'max',
        'expectRateProportionalStimOnDelay': True,
        'expectRateProportionalStimOffDelay': True,
        'predictSlots': False, 'snapToGrid': False,
        'treatAsSinglePulses': True
        }
    # pdb.set_trace()
    trialFilesStim['ins']['getINSkwargs'].update(commonStimDetectionOpts)
    amplitudeFieldName = 'nominalCurrent' if blockExperimentType == 'isi' else 'amplitude'
    stimConditionNames = [
        'electrode', amplitudeFieldName, 'RateInHz']
    motionConditionNames = [
        'pedalMovementCat', 'pedalSizeCat', 'pedalDirection']
    if blockExperimentType in ['proprio-miniRC', 'proprio-RC', 'isi']:
        #if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC') or (blockExperimentType == 'isi'):
        # has stim but no motion
        stimulusConditionNames = stimConditionNames
    elif blockExperimentType == 'proprio-motionOnly':
        # has motion but no stim
        stimulusConditionNames = motionConditionNames
    else:
        stimulusConditionNames = stimConditionNames + motionConditionNames
    print('Block type {}; using the following stimulus condition breakdown:'.format(blockExperimentType))
    #
    if blockExperimentType == 'proprio-miniRC':
        #  override with settings for detecting cycling stim trains
        trialFilesStim['ins']['getINSkwargs'].update(miniRCStimDetectionOpts)
        #  only parse sync lines
        eventInfo = {'inputIDs': expOpts['miniRCRigInputs']}
    elif blockExperimentType == 'proprio-RC':
        trialFilesStim['ins']['getINSkwargs'].update(RCStimDetectionOpts)
        trialFilesStim['ins']['eventsFromFirstInTrain'] = False
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': expOpts['RCRigInputs']}
    elif blockExperimentType == 'isi':
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': dict()}
    else:
        trialFilesStim['ins']['getINSkwargs'].update(fullStimDetectionOpts)
        #  should rename eventInfo to something more intuitive
        eventInfo = {'inputIDs': expOpts['fullRigInputs']}
    #
    trialFilesFrom['utah'].update({'eventInfo': eventInfo})
    
    nspCmpPath = os.path.join('.', 'murdoc_map.cmp')
    cmpDF = prb_meta.cmpToDF(nspCmpPath)
    experimentDateStr = re.search(r'(\d*)', experimentName).groups()[0]
    # pdb.set_trace()
    try:
        impedances = prb_meta.getLatestImpedance(
            impedanceFilePath=os.path.join(remoteBasePath, 'impedances.h5'),
            recordingDateStr=experimentDateStr, elecType='utah')
        impedances['elec'] = cmpDF['label'].iloc[:impedances.shape[0]].to_numpy()
        impedances.set_index('elec', inplace=True)
        impedancesRipple = prb_meta.getLatestImpedance(
            impedanceFilePath=os.path.join(remoteBasePath, 'impedances.h5'),
            recordingDateStr=experimentDateStr, elecType='isi_paddle')
    except Exception:
        impedances = None
        impedancesRipple = None
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
    #
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
        remoteBasePath, 'processed', experimentName, 'figures'
        )
    if not os.path.exists(figureFolder):
        os.makedirs(figureFolder, exist_ok=True)
    #
    # alignedRastersFolder = os.path.join(figureFolder, 'alignedRasters')
    # if not os.path.exists(alignedRastersFolder):
    #     os.makedirs(alignedRastersFolder, exist_ok=True)
    # alignedFeaturesFolder = os.path.join(figureFolder, 'alignedFeatures')
    # if not os.path.exists(alignedFeaturesFolder):
    #     os.makedirs(alignedFeaturesFolder, exist_ok=True)
    # spikeSortingFiguresFolder = os.path.join(figureFolder, 'spikeSorting')
    # if not os.path.exists(spikeSortingFiguresFolder):
    #     os.makedirs(spikeSortingFiguresFolder, exist_ok=True)
    # GLMFiguresFolder = os.path.join(figureFolder, 'GLM')
    # if not os.path.exists(GLMFiguresFolder):
    #     os.makedirs(GLMFiguresFolder, exist_ok=True)
    #
    if blockExperimentType in ['proprio-miniRC', 'proprio', 'proprio-RC', 'proprio-motionOnly']:
        essentialMetadataFields = [
            'segment', 'originalIndex', 't', 'expName', 'amplitude', 'program',
            'activeGroup', 'RateInHz', 'stimCat', 'electrode',
            'pedalDirection', 'pedalSize', 'pedalSizeCat', 'pedalMovementCat',
            'pedalMetaCat',
            ]
    else:
        essentialMetadataFields = [
            'segment', 'originalIndex', 't', 'expName', 'nominalCurrent', 'program',
            'activeGroup', 'RateInHz', 'stimCat', 'electrode',
            ]
    alignedAsigsKWargs = dict(
        amplitudeColumn='amplitude',
        programColumn='program',
        electrodeColumn='electrode',
        removeFuzzyName=False,
        getMetaData=essentialMetadataFields,
        decimate=1)
    overrideChanNames = None
    # overrideChanNames = [
    #     'elec75#0', 'elec75#1', 'elec83#0', 'elec78#0', 'elec78#1']
    overrideUnitNames = None
    # overrideUnitNames = [
    #     'elec75#0', 'elec75#1', 'elec83#0', 'elec78#0', 'elec78#1']
    alignedAsigsChunkSize = 150000
    rasterOpts = {
        'binOpts': {
            'loRes': {
                'binInterval': 5e-4,
                'binWidth': 5e-3,
                'smoothKernelWidth': 5e-3},  # 2 kHz,
            'default': {
                'subfolder': 'default',
                'binInterval': 1e-3,
                'binWidth': 10e-3,
                'smoothKernelWidth': 10e-3},  # default, 1kHz
            'normalizedByImpedance': {
                'subfolder': 'default',
                'binInterval': 1e-3,
                'binWidth': 10e-3,
                'smoothKernelWidth': 10e-3},  # default
            'parameter_recovery': {
                'binInterval': 1e-3,
                'binWidth': 10e-3,
                'smoothKernelWidth': 10e-3},  # same as default
            'fullRes': {
                'binInterval': (3e4) ** (-1),
                'binWidth': 5e-3,
                'smoothKernelWidth': 5e-3},  # 30 kHz
            'hiRes': {
                'binInterval': 2e-4,
                'binWidth': 5e-3,
                'smoothKernelWidth': 5e-3},  # 5 kHz
            },
        'windowSizes': {
            'XS': (-0.2, 0.4),
            'XSPre': (-0.65, -0.05),
            'XXS': (-0.1, 0.1),
            'XXXS': (-0.005, 0.025),
            'M': (-0.2, 0.8),
            'short': (-0.5, 0.5),
            'L': (-0.5, 1.5),
            'XL': (-1.5, 2.5),
            'RC': (-0.33, 0.33),
            'miniRC': (-1, 1)},
        'discardEmpty': None, 'maxSpikesTo': None, 'timeRange': None,
        'separateByFunArgs': None,
        'alignTo': None,
        'separateByFunKWArgs': {'type': 'Classification'}
        }
    statsTestOpts = dict(
        testStride=25e-3,
        testWidth=25e-3,
        tStart=0,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=False
        )
    relplotKWArgs = dict(
        errorbar='se',
        # ci=95, n_boot=1000,
        estimator='mean',
        # estimator=None, units='t',
        palette="ch:0.6,-.3,dark=.1,light=0.7,reverse=1",
        # facet_kws={'sharey': True},
        height=6, aspect=2, kind='line')
    vLineOpts = {'color': 'm', 'alpha': 0.5}
    asigPlotShadingOpts = {
        'facecolor': vLineOpts['color'],
        'alpha': 0.1, 'zorder': -100}
    asigSigStarOpts = {
        'c': vLineOpts['color'],
        # 'linestyle': 'None',
        's': 50,
        'marker': '*'
        }
    nrnRelplotKWArgs = dict(
        palette="ch:1.6,-.3,dark=.1,light=0.7,reverse=1",
        func1_kws={
            'marker': 'd',
            'edgecolor': None,
            'edgecolors': 'face',
            'alpha': .3, 'rasterized': True},
        func2_kws={'ci': 'sem'},
        facet1_kws={'sharey': False},
        facet2_kws={'sharey': True},
        height=4, aspect=2,
        kind1='scatter', kind2='line')
    nrnVLineOpts = {'color': 'y'}
    nrnBlockShadingOpts = {
        'facecolor': nrnVLineOpts['color'],
        'alpha': 0.3, 'zorder': -100}
    nrnSigStarOpts = {
        'c': nrnVLineOpts['color'],
        # 'edgecolor': None,
        'edgecolors': 'face',
        # 'linestyle': 'None',
        's': 20,
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
                        scratchPath, key, '{}', '{}',
                        '{{}}{:0>3}{{}}.nix'.format(int(tIdx))
                    )
                )
    except Exception:
        pass
    #
    iteratorOpts = {
        # rest period from before movement onset
        'a': {
            'nSplits': 7,
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 10e-3,
            'calcTimeROI': True,
            'controlProportion': None,
            'timeROIOpts': {
                'alignQuery': None,
                'winStart': -700e-3,
                'winStop': -400e-3
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': -700e-3,
                'winStop': -400e-3
            }
        },
        'f': {
            'nSplits': 7,
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': None,
            'calcTimeROI': True,
            'controlProportion': 0.1,
            'timeROIOpts': {
                'alignQuery': None,
                'winStart': -100e-3,
                'winStop': 100e-3
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': -400e-3,
                'winStop': -300e-3
            }
        }
    }
    if 'expIteratorOpts' in expOpts:
        for key in iteratorOpts.keys():
            if key in expOpts['expIteratorOpts']:
                iteratorOpts[key].update(expOpts['expIteratorOpts'][key])
    glmOptsLookup = {
        'ensembleHistoryLen': .30,
        'covariateHistoryLen': .50,
        'nHistoryBasisTerms': 5,
        'nCovariateBasisTerms': 1,
        'regressionBinInterval': 20e-3,
        'glm_50msec': dict(
            subsampleOpts=dict(
                rollingWindow=50, decimate=50,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=25e-3),
        'glm_30msec': dict(
            subsampleOpts=dict(
                rollingWindow=30, decimate=30,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=15e-3),
        'glm_20msec': dict(
            subsampleOpts=dict(
                rollingWindow=20, decimate=20,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=10e-3),
        'glm_10msec': dict(
            subsampleOpts=dict(
                rollingWindow=10, decimate=10,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=10e-3),
        'glm_5msec': dict(
            subsampleOpts=dict(
                rollingWindow=5, decimate=5,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=10e-3),
        'glm_3msec': dict(
            subsampleOpts=dict(
                rollingWindow=3, decimate=3,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=10e-3),
        'glm_1msec': dict(
            subsampleOpts=dict(
                rollingWindow=None, decimate=1,
                windowSize=(-1750e-3, 1750e-3)),
            covariateSpacing=10e-3),
    }
    spectralFeatureOpts = dict(
        winLen=100e-3, stepLen=20e-3, R=20,
        fStart=None, fStop=None)
    freqBandsDict = ({
        'name':   ['low', 'alpha', 'beta', 'gamma', 'higamma', 'spb'],
        'lBound': [1.5,   7,       15,     30,      60,        250],
        'hBound': [7,     14,      29,     55,      120,       1000]
        })
    return expOpts, locals()
