import os, pdb, re, platform, traceback, sys
#  import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
import importlib
import pandas as pd

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
    expDateTime = pd.to_datetime(experimentName.split('-')[0], format='%Y%m%d%H%M')
    expDateTimePathStr = expDateTime.strftime('%Y-%m-%d')
    assembledName = 'Block'
    nspFolder = os.path.join(remoteBasePath, 'raw', experimentName)
    simiFolder = os.path.join(remoteBasePath, 'processed', experimentName, 'simi')
    oeFolder = os.path.join(remoteBasePath, 'raw', experimentName, 'open_ephys')
    sipFolder = os.path.join(remoteBasePath, 'raw', experimentName, 'ins_sip')
    ns5FileName = 'Block{:0>3}'.format(blockIdx)
    # miniRCBlockLookup = expOpts['miniRCBlockLookup']
    # RCBlockLookup = expOpts['RCBlockLookup']
    # RippleBlockLookup = expOpts['RippleBlockLookup']
    if blockIdx in expOpts['blockExperimentTypeLookup']:
        blockExperimentType = expOpts['blockExperimentTypeLookup'][blockIdx]
    else:
        print('{} has no defined block {}. Exiting.'.format(experimentName, blockIdx))
        sys.exit()
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
            'fractionForRollover': expOpts['fractionForRollover'] if 'fractionForRollover' in expOpts else 0.1,
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
    amplitudeFieldName = 'nominalCurrent' if blockExperimentType == 'isi' else 'trialAmplitude'
    stimConditionNames = [
        'electrode', amplitudeFieldName, 'trialRateInHz']
    motionConditionNames = [
        'pedalMovementCat', 'pedalDirection',
        'pedalSizeCat',
        ]
    if blockExperimentType in ['proprio-miniRC', 'proprio-RC', 'isi']:
        #if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC') or (blockExperimentType == 'isi'):
        # has stim but no motion
        stimulusConditionNames = stimConditionNames
    elif blockExperimentType == 'proprio-motionOnly':
        # has motion but no stim
        stimulusConditionNames = motionConditionNames
    else:
        stimulusConditionNames = stimConditionNames + motionConditionNames
    # print('Block type {}; using the following stimulus condition breakdown:'.format(blockExperimentType))
    # print('\n'.join(['    {}'.format(scn) for scn in stimulusConditionNames]))
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
    minNConditionRepetitions = {
        'n': 1,
        'categories': stimulusConditionNames
        }
    nspCmpPath = os.path.join('.', 'murdoc_map.cmp')
    cmpDF = prb_meta.cmpToDF(nspCmpPath)
    experimentDateStr = re.search(r'(\d*)', experimentName).groups()[0]
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
            'segment', 'originalIndex', 't', 'expName', amplitudeFieldName, 'program',
            'activeGroup', 'trialRateInHz', 'stimCat', 'electrode',
            'pedalDirection', 'pedalSize', 'pedalSizeCat', 'pedalMovementCat',
            'pedalMetaCat',
            ]
    else:
        essentialMetadataFields = [
            'segment', 'originalIndex', 't', 'expName', amplitudeFieldName, 'program',
            'activeGroup', 'trialRateInHz', 'stimCat', 'electrode',
            ]
    alignedAsigsKWargs = dict(
        amplitudeColumn=amplitudeFieldName,
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
                'binInterval': (3e3) ** (-1),
                'binWidth': 1e-3,
                'smoothKernelWidth': 1e-3},  # 3 kHz
            'hiResHiFreq': {
                'binInterval': (3e3) ** (-1),
                'binWidth': 1e-3,
                'smoothKernelWidth': 1e-3},  # 3 kHz
            },
        'windowSizes': {
            'XS': (-0.2, 0.4),
            'XSPre': (-0.65, -0.05),
            'XXS': (-0.1, 0.1),
            'XXXS': (-0.005, 0.025),
            'M': (-0.2, 0.8),
            'short': (-0.5, 0.5),
            'L': (-0.5, 1.5),
            'XL': (-1.5, 2.),
            'RC': (-0.33, 0.33),
            'miniRC': (-1, 1)},
        'analysisFilterKWArgs': {
            'hiRes': dict(
                filterSignals=True, medianFilterSignals=False,
                hampelFilterSignals=True, diagnosticPlots=False,
                analysisHampelFilterOpts=dict(
                    average_across_channels=True,
                    window_size=31, thresh=3, imputation=True  # window must be odd
            ))
        },
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
        estimator='mean',
        # estimator=None, units='t',
        palette="ch:0.6,-.3,dark=.1,light=0.7,reverse=1", err_kws=dict(alpha=0.2, edgecolor=None),
        height=3, aspect=2, kind='line', rasterized=True)
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
    # args for the sklearn StratifiedShuffleSplit
    defaultSamplerKWArgs = dict(random_state=42, test_size=0.2)
    defaultPrelimSamplerKWArgs = dict(random_state=42, test_size=0.15)
    #
    covarianceSamplerKWArgs = dict(random_state=42, test_size=0.2)
    covariancePrelimSamplerKWArgs = dict(random_state=42, test_size=0.)
    #
    equalSamplerKWArgs = dict(random_state=42, test_size=0.5)
    equalPrelimSamplerKWArgs = dict(random_state=42, test_size=0.15)
    # args for tdr.
    defaultSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=defaultSamplerKWArgs)
    defaultPrelimSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=defaultPrelimSamplerKWArgs)
    covarianceSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=covarianceSamplerKWArgs)
    covariancePrelimSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=covariancePrelimSamplerKWArgs)
    equalSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=equalSamplerKWArgs)
    equalPrelimSplitterKWArgs = dict(
        stratifyFactors=stimulusConditionNames,
        continuousFactors=['segment', 'originalIndex', 't'],
        samplerClass=None,
        samplerKWArgs=equalPrelimSamplerKWArgs)
    #
    iteratorOpts = {
        # rest period from before movement onset
        'ca': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=25,
                splitterClass=None, splitterKWArgs=covarianceSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=covariancePrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': None,
                'winStart': None,
                'winStop': None
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': -600e-3,
                'winStop': -200e-3
                }
            },
        # perimovement, no stim
        'cb': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=25,
                splitterClass=None, splitterKWArgs=covarianceSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=covariancePrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'startingNoStim',
                'winStart': 0.,
                'winStop': 400e-3
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': None,
                'winStop': None
                }
            },
        # stimOnE#
        'cc': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=25,
                splitterClass=None, splitterKWArgs=covarianceSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=covariancePrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'startingOrStimOn',
                'winStart': 0.,
                'winStop': 400e-3
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': None,
                'winStop': None
                }
            },
        # rest period from before movement onset
        'cd': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=25,
                splitterClass=None, splitterKWArgs=covarianceSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=covariancePrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': None,
                'winStart': None,
                'winStop': None
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': -600e-3,
                'winStop': -200e-3
                }
            },
        # perimovement, any stim, for regression
        'ra': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'procFun': {
                'laplace_scaled': 'ash.genDetrender(timeWindow=[-0.2, 0.6], useMean=True)',
                'laplace_spectral_scaled': 'ash.genDetrender(timeWindow=[-0.2, 0.6], useMean=True)',
                },
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=10,
                splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'startingOrStimOn',
                'winStart': -0.35,  # start 0.2 ( + .15 burn in period) before whatever the query was
                'winStop': .6  # stop .5 sec after startingOrStimOn
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': None,
                'winStop':  None,
                }
            },
        # perimovement, any stim, for regression
        'rb': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 25e-3,
            'forceRollingWindow': 300,
            # 'procFun': {
            #     'laplace_scaled': 'ash.genDetrender(timeWindow=[-0.2, 0.6], useMean=True)',
            #     'laplace_spectral_scaled': 'ash.genDetrender(timeWindow=[-0.2, 0.6], useMean=True)',
            #     },
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'maskEachFreqBand': False,
            'cvKWArgs': dict(
                n_splits=5,
                splitterClass=None, splitterKWArgs=equalSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=equalPrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'stoppingOrStimOff',
                'winStart': -0.35,  # start 0.2 ( + .15 burn in period) before whatever the query was
                'winStop': 0.2  # stop .1 sec after startingOrStimOn
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': None,
                'winStop':  None,
                }
            },
        # perimovement, any stim, for plotting
        'pa': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 1e-3,
            # # #'procFun': {
            # # #    'laplace': 'ash.genDetrender(timeWindow=[-0.6, -0.3], useMean=True)',
            # # #    'laplace_spectral': 'ash.genDetrender(timeWindow=[-0.6, -0.3], useMean=True)',
            # # #    },
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=2,
                splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'startingOrStimOn',
                'winStart': -0.6,  # start 0.6 before whatever the query was
                'winStop': 1.2  # stop .6 sec after startingOrStimOn
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': None,
                'winStop':  None,
                }
            },
        # perimovement, any stim, for normalization of spectral data
        'na': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=2,
                splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'startingOrStimOn',
                'winStart': -0.6,  # start 0.6 before whatever the query was
                'winStop': 1.2  # stop .6 sec after startingOrStimOn
            },
            'timeROIOpts_control': {
                'alignQuery': None,
                'winStart': None,
                'winStop':  None,
                }
            },
        # perimovement onset (or peristim onset if stim only) for RAUC
        'ma': {
            'ensembleHistoryLen': .30,
            'covariateHistoryLen': .50,
            'nHistoryBasisTerms': 1,
            'nCovariateBasisTerms': 1,
            'forceBinInterval': 2e-3,
            'minBinCount': 5,
            'calcTimeROI': True,
            'controlProportion': None,
            'cvKWArgs': dict(
                n_splits=5,
                splitterClass=None, splitterKWArgs=defaultSplitterKWArgs,
                prelimSplitterClass=None, prelimSplitterKWArgs=defaultPrelimSplitterKWArgs,
                resamplerClass=None, resamplerKWArgs={},
                ),
            'timeROIOpts': {
                'alignQuery': 'startingOrStimOn',
                'winStart': -0.3,  # start 0.3 before whatever the query was
                'winStop': 0.4  # stop .6 sec after startingOrStimOn
            },
            'timeROIOpts_control': {
                'alignQuery': 'startingNoStim',
                'winStart': -0.6,
                'winStop': -0.1,
                }
            },
        }
    #
    for optsKey in ['ca', 'cb', 'cc', 'cd']:
        iteratorOpts[optsKey]['cvKWArgs']['splitterKWArgs']['samplerKWArgs']['test_size'] = 0.2
    iteratorOpts['ccs'] = iteratorOpts['cc'].copy()
    iteratorOpts['ccm'] = iteratorOpts['cc'].copy()
    #
    if 'expIteratorOpts' in expOpts:
        for key in iteratorOpts.keys():
            if key in expOpts['expIteratorOpts']:
                iteratorOpts[key].update(expOpts['expIteratorOpts'][key])
    #
    spectralFeatureOpts = dict(
        winLen=100e-3, stepLen=20e-3, R=20,
        fStart=None, fStop=None)
    freqBandOrder = ['alpha', 'beta', 'gamma', 'higamma', 'spb']
    freqBandsDict = ({
        'name':   freqBandOrder,
        'lBound': [7,       15,     30,      60,        250],
        'hBound': [14,      29,     55,      120,       1000]
        })
    freqBandOrderExtended = freqBandOrder + ['NA', 'all', 'broadband', 'cross_frequency']
    outlierDetectOptions = dict(
        windowSize=(-.8, 1.2),
        twoTailed=False,
        qThresh=7.62e-24,
        devQuantile=None,
        fitWindow=(-150e-3, 350e-3)
        )
    csdOpts = {
        'filterOpts': {
            'low': {
                # 'Wn': 400,
                'N': 8,
                'btype': 'low',
                'ftype': 'butter'
            },
            'high': {
                'Wn': .1,
                'N': 8,
                'btype': 'high',
                'ftype': 'butter'
            },
        }
        }
    prettyNameLookup = {
        'T': 'T',
        'hedges': 'g',
        'coef': r"$\beta$",
        'electrode = NA': 'No stim.',
        'stimCondition = NA_0.0': 'No stim.',
        'NA': 'No stim.',
        'kinematicCondition = NA_NA': 'No movement',
        'kinematicCondition = CW_outbound': 'Start of movement\n(extension)',
        'kinematicCondition = CW_return': 'Return to start\n(flexion)',
        'kinematicCondition = CCW_outbound': 'Start of movement\n(flexion)',
        'kinematicCondition = CCW_return': 'Return to start\n(extension)',
        'feature = mahal_ledoit_all': 'Mahal. dist.\n(Broadband)',
        'feature = mahal_ledoit_alpha': 'Mahal. dist.\n(Alpha)',
        'feature = mahal_ledoit_beta': 'Mahal. dist.\n(Beta)',
        'feature = mahal_ledoit_gamma': 'Mahal. dist.\n(Gamma)',
        'feature = mahal_ledoit_higamma': 'Mahal. dist.\n(High gamma)',
        'feature = mahal_ledoit_spb': 'Mahal. dist.\n(Spike band)',
        'freqBandName': 'Frequency band',
        'electrode': 'Stim. electrode (cathode)',
        'NA': 'Broadband',
        'all': 'Broadband',
        'broadband': 'Broadband',
        'cross_frequency': 'Cross Freq.',
        'alpha': 'Alpha',
        'beta': 'Beta',
        'gamma': 'Gamma',
        'higamma': 'High gamma',
        'spb': 'Spike band',
        'trialRateInHzStr': 'Stim. rate (Hz)',
        '0.0': 'No stim.',
        '50.0': '50',
        '100.0': '100',
        '0.0_md': 'No stim. (Mahal. dist.)',
        '50.0_md': '50 (Mahal. dist.)',
        '100.0_md': '100 (Mahal. dist.)',
        'namesAndMD': 'Regressors',
        'trialAmplitude': 'Stim. amplitude',
        'trialAmplitude:trialRateInHz': 'Stim. rate interaction',
        'trialAmplitude_md': 'Stim. amplitude (Mahal. dist.)',
        'trialAmplitude:trialRateInHz_md': 'Stim. rate interaction (Mahal. dist.)',
        }
    for eIdx in range(16):
        prettyNameLookup['electrode = -E{:02d}+E16'.format(eIdx)] = 'Stim. E{:02d}'.format(eIdx)
        prettyNameLookup['-E{:02d}+E16'.format(eIdx)] = 'Stim. E{:02d}'.format(eIdx)
        prettyNameLookup['stimCondition = -E{:02d}+E16_50.0'.format(eIdx)] = 'Stim. E{:02d} (50 Hz)'.format(eIdx)
        prettyNameLookup['-E{:02d}+E16_50.0'.format(eIdx)] = 'E{:02d} (50 Hz'.format(eIdx)
        prettyNameLookup['stimCondition = -E{:02d}+E16_100.0'.format(eIdx)] = 'Stim. E{:02d} (100 Hz)'.format(eIdx)
        prettyNameLookup['-E{:02d}+E16_100.0'.format(eIdx)] = 'E{:02d} (100 Hz)'.format(eIdx)
    for fbn in freqBandOrderExtended + [None]:
        if fbn in prettyNameLookup:
            prettyNameLookup['freqBandName = {}'.format(fbn)] = prettyNameLookup[fbn]
        for eIdx in range(97):
            fbSuffix = '_{}'.format(fbn) if fbn is not None else ''
            fbPrettyName = '\n({})'.format(prettyNameLookup[fbn]) if fbn is not None else ''
            prettyNameLookup['feature = utah_csd_{}{}'.format(eIdx, fbSuffix)] = 'LFP chan. #{}{}'.format(eIdx, fbPrettyName)
            prettyNameLookup['utah_csd_{}{}'.format(eIdx, fbSuffix)] = 'LFP chan. #{}{}'.format(eIdx, fbPrettyName)
    applyPrettyNameLookup = lambda x: prettyNameLookup[x] if x in prettyNameLookup else x
    expNameElectrodeLookup = {
        'Murdoc': {
            'exp201901251000': ['-E14+E16', ],
            'exp201902031100': ['-E00+E16', '-E09+E16', '-E11+E16']
        },
        'Rupert': {
            'exp202101201100': ['-E13+E16'],
            'exp202101211100': ['-E04+E16'],
            'exp202101221100': ['-E12+E16'],
            'exp202101251100': ['-E02+E16'],
            'exp202101271100': ['-E05+E16'],
            'exp202101281100': ['-E09+E16'],
            'exp202102021100': ['-E11+E16']
        }
    }
    spinalElectrodeMaps = {
        'Murdoc': pd.DataFrame({
            'NA': [0, 0, True],
            '-E14+E16': [1, 1, False],
            '-E11+E16': [2, 1, False],
            '-E09+E16': [3, 1, False],
            '-E00+E16': [4, 1, False],
            }, index=['xCoords', 'yCoords', 'isDummy']).T,
        'Rupert': pd.DataFrame({
            'NA': [0, 0, True],
            '-E13+E16': [1, 2, False],
            '-E09+E16': [2, 1, False],
            '-E11+E16': [3, 2, False],
            '-E12+E16': [4, 1, False],
            '-E02+E16': [5, 2, False],
            '-E05+E16': [6, 1, False],
            '-E04+E16': [7, 2, False],
            }, index=['xCoords', 'yCoords', 'isDummy']).T,
        }
    return expOpts, locals()
