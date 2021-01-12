#!/users/rdarie/anaconda/nda/bin/python
"""01: Preprocess spikes, then 04: Run peeler and 05: Assemble the spike nix file

Usage:
    tridesclousCCV.py [options]

Options:
    --exp=exp                                   which experimental day to analyze
    --blockIdx=blockIdx                         which trial to analyze [default: 1]
    --arrayName=arrayName                       which electrode array to analyze [default: utah]
    --sourceFileSuffix=sourceFileSuffix         which source file to analyze
    --remakePrb                                 whether to rewrite the electrode map file [default: False]
    --purgePeeler                               delete previous sort results [default: False]
    --purgePeelerDiagnostics                    delete previous sort results [default: False]
    --batchPrepWaveforms                        extract snippets [default: False]
    --batchRunClustering                        extract features, run clustering [default: False]
    --batchCleanConstructor                     extract features, run clustering [default: False]
    --batchPreprocess                           extract snippets and features, run clustering [default: False]
    --batchPeel                                 run peeler [default: False]
    --removeExistingCatalog                     delete previous sort results [default: False]
    --initCatalogConstructor                    whether to init a catalogue constructor [default: False]
    --fromNS5                                   save peeler results to a neo block [default: False]
    --makeCoarseNeoBlock                        save peeler results to a neo block [default: False]
    --makeStrictNeoBlock                        save peeler results to a neo block [default: False]
    --overrideSpikeSource                       save peeler results to a neo block [default: False]
    --exportSpikesCSV                           save peeler results to a csv file [default: False]
    --chan_start=chan_start                     which chan_grp to start on [default: 0]
    --chan_stop=chan_stop                       which chan_grp to stop on [default: 96]
"""

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

try:
    import tensorflow as tf
except Exception:
    pass
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import os, gc, traceback, re
import pdb
from copy import copy
import json
import matplotlib
from numba.core.errors import NumbaPerformanceWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
#
matplotlib.use('PS')   # generate postscript output
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
#
from currentExperiment import parseAnalysisOptions
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


def tridesclousCCV(
        blockBaseName,
        nameSuffix=None,
        extractOpts=None,
        featureOpts=None,
        clusterOpts=None,
        preprocOpts=None,
        spikeSortingOpts=None,
        scratchFolder=None,
        nspFolder=None,
        partNameSuffix=None,
        arrayName=None,
        ):
    triFolder = os.path.join(
        scratchFolder, 'tdc_{}'.format(
            blockBaseName + nameSuffix))
    prbPath = os.path.join(
        scratchFolder, arrayName + '_map.prb'
        )
    if arguments['overrideSpikeSource']:
        altDataIOInfo = {
            'datasource_type': 'NIX',
            'datasource_kargs': {
                'filenames': [
                    os.path.join(
                        scratchFolder,
                        (
                            blockBaseName +
                            '_spike_preview_unfiltered' +
                            partNameSuffix +
                            '.nix'))
                ]
            }}
        waveformSignalType = 'initial'
    else:
        altDataIOInfo = None
        waveformSignalType = 'processed'
    #
    if arguments['initCatalogConstructor']:
        try:
            if arguments['fromNS5']:
                tdch.initialize_catalogueconstructor(
                    nspFolder,
                    blockBaseName,
                    triFolder,
                    prbPath=prbPath,
                    removeExisting=arguments['removeExistingCatalog'],
                    fileFormat='Blackrock')
            else:
                tdch.initialize_catalogueconstructor(
                    scratchFolder,
                    blockBaseName + nameSuffix,
                    triFolder,
                    spikeSortingOpts=spikeSortingOpts,
                    removeExisting=arguments['removeExistingCatalog'],
                    fileFormat='NIX')
        except Exception:
            traceback.print_exc()
            print('Ignoring Exception')
    #
    chan_start = int(arguments['chan_start'])
    chan_stop = int(arguments['chan_stop'])
    # dataio = tdc.DataIO(dirname=triFolder)
    # print(dataio)
    # chansToAnalyze = [
    #     chNum
    #     for chNum in list(range(chan_start, chan_stop))
    #     if chNum in list(dataio.channel_groups.keys())]
    chansToAnalyze = list(range(chan_start, chan_stop))
    print('Analyzing channels:\n{}'.format(chansToAnalyze))
    ######################################################################
    if arguments['purgePeeler']:
        tdch.purgeNeoBlock(triFolder)
        tdch.purgePeelerResults(
            triFolder, purgeAll=True)
    #
    if arguments['purgePeelerDiagnostics']:
        tdch.purgePeelerResults(
            triFolder, diagnosticsOnly=True,
            purgeAll=True)
    ######################################################################
    if arguments['batchPreprocess']:
        tdch.batchPreprocess(
            triFolder, chansToAnalyze,
            nb_noise_snippet=1000,
            minWaveformRate=None,
            minWaveforms=10,
            alien_value_threshold=50.,
            make_classifier=spikeSortingOpts['make_classifier'],
            classifier_opts=None,
            extractOpts=extractOpts,
            featureOpts=featureOpts,
            clusterOpts=clusterOpts,
            **preprocOpts)
    ######################################################################
    if arguments['batchRunClustering']:
        tdch.batchRunClustering(
            triFolder, chansToAnalyze,
            featureOpts=featureOpts,
            clusterOpts=clusterOpts,
            autoMerge=False, auto_merge_threshold=0.99)
    #
    if arguments['batchPrepWaveforms']:
        tdch.batchPrepWaveforms(
            triFolder, chansToAnalyze,
            featureOpts=featureOpts,
            clusterOpts=clusterOpts,
            extractOpts=extractOpts,
            **preprocOpts
            )
    #
    if arguments['batchCleanConstructor']:
        tdch.batchCleanConstructor(
            triFolder, chansToAnalyze,
            make_classifier=spikeSortingOpts['make_classifier'],
            classifier_opts=None,
            refit_projector=spikeSortingOpts['refit_projector'],
            )
    ######################################################################
    if arguments['batchPeel']:
        tdch.purgePeelerResults(
            triFolder, chan_grps=chansToAnalyze)
        tdch.batchPeel(
            triFolder, chansToAnalyze,
            chunksize=preprocOpts['chunksize'],
            shape_distance_threshold=spikeSortingOpts['shape_distance_threshold'],
            shape_boundary_threshold=spikeSortingOpts['shape_boundary_threshold'],
            confidence_threshold=spikeSortingOpts['confidence_threshold'],
            energy_reduction_threshold=spikeSortingOpts['energy_reduction_threshold'],
            n_max_passes=spikeSortingOpts['n_max_peeler_passes'],  #
            )
    ######################################################################
    #
    if arguments['makeCoarseNeoBlock']:
        tdch.purgeNeoBlock(triFolder)
        tdch.neo_block_after_peeler(
            triFolder, chan_grps=chansToAnalyze,
            shape_distance_threshold=None, refractory_period=None,
            ignoreTags=[]
            )
    #
    if arguments['makeStrictNeoBlock']:
        tdch.purgeNeoBlock(triFolder)
        tdch.neo_block_after_peeler(
            triFolder, chan_grps=chansToAnalyze,
            shape_distance_threshold=spikeSortingOpts['shape_distance_threshold'],
            refractory_period=spikeSortingOpts['refractory_period'],
            shape_boundary_threshold=spikeSortingOpts['shape_boundary_threshold'],
            energy_reduction_threshold=spikeSortingOpts['energy_reduction_threshold'],
            ignoreTags=['so_bad'], altDataIOInfo=altDataIOInfo,
            waveformSignalType=waveformSignalType,
            )
    return
######################


def tdcCCVWrapper():
    #  electrode array name (changes the prefix of the file)
    arrayName = arguments['arrayName']
    if 'rawBlockName' in spikeSortingOpts[arrayName]:
        blockBaseName = ns5FileName.replace(
            'Block', spikeSortingOpts[arrayName]['rawBlockName'])
    else:
        blockBaseName = copy(ns5FileName)
    #  source file suffix (changes the suffix of the file to load)
    if arguments['sourceFileSuffix'] is not None:
        sourceFileSuffix = '_' + arguments['sourceFileSuffix']
    else:
        sourceFileSuffix = ''
    chunkingInfoPath = os.path.join(
        scratchFolder,
        blockBaseName + sourceFileSuffix + '_chunkingInfo.json'
        )
    #
    if os.path.exists(chunkingInfoPath):
        with open(chunkingInfoPath, 'r') as f:
            chunkingMetadata = json.load(f)
    else:
        chunkingMetadata = {
            '0': {
                'filename': os.path.join(
                    scratchFolder, blockBaseName + sourceFileSuffix + '.nix'
                    ),
                'partNameSuffix': '',
                'chunkTStart': 0,
                'chunkTStop': 'NaN'
            }}
    #
    spikeSortingOpts[arrayName]['remakePrb'] = arguments['remakePrb']
    # ########## waveform extraction options
    theseExtractOpts = dict(
        mode='rand',
        n_left=spikeWindow[0] - 2,
        n_right=spikeWindow[1] + 2,
        nb_max=10000,
        align_waveform=False)
    #
    # ########## decomposition options
    #
    #  ### parametric umap (with tensorflow) projection options
    '''
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `loss` is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=5e-3,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
            )
        ]
    #
    theseFeatureOpts = {
        'method': 'global_pumap',
        'n_components': 3,
        'n_neighbors': 50,
        'min_dist': 0,
        'metric': 'euclidean',
        'set_op_mix_ratio': 0.9,
        'parametric_reconstruction': False,
        'autoencoder_loss': False,
        'verbose': False,
        'batch_size': 10000,
        'n_training_epochs': 15,
        'keras_fit_kwargs': {'verbose': 2, 'callbacks': callbacks}
        }
    '''
    #  ### PCA opts
    theseFeatureOpts = {
        'method': 'global_pca',
        'n_components': 5
        }
    #  ########## clustering options
    #
    '''
    theseClusterOpts = {
        'method': 'agglomerative',
        'n_clusters': 2
        }
    '''
    theseClusterOpts = {
        'method': 'onecluster',
        }
    
    thesePreprocOpts = dict(
        relative_threshold=4,
        fill_overflow=False,
        highpass_freq=100.,
        lowpass_freq=5000.,
        common_ref_freq=None,
        common_ref_removal=False,
        notch_freq=None,
        filter_order=2,
        noise_estimate_duration=spikeSortingOpts[arrayName]['previewDuration'],
        sample_snippet_duration=spikeSortingOpts[arrayName]['previewDuration'],
        chunksize=2**19,
        autoMerge=False, auto_merge_threshold=0.99,
        auto_make_catalog=True,
        )
    for chunkIdxStr, chunkMeta in chunkingMetadata.items():
        # chunkIdx = int(chunkIdxStr)
        nameSuffix = sourceFileSuffix + chunkMeta['partNameSuffix']
        tridesclousCCV(
            blockBaseName,
            nameSuffix=nameSuffix,
            extractOpts=theseExtractOpts,
            featureOpts=theseFeatureOpts,
            clusterOpts=theseClusterOpts,
            preprocOpts=thesePreprocOpts,
            spikeSortingOpts=spikeSortingOpts[arrayName],
            scratchFolder=scratchFolder, partNameSuffix=None,
            nspFolder=nspFolder, arrayName=arrayName
            )
    return


if __name__ == "__main__":
    runProfiler = True
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        taskNameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=tdcCCVWrapper,
            modulesToProfile=[tdch],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=taskNameSuffix, outputUnits=1e-3)
    else:
        tdcCCVWrapper()
    print('Done running tridesclousCCV.py')
