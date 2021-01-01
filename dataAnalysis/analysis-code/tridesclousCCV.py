#!/users/rdarie/anaconda/nda/bin/python
"""01: Preprocess spikes, then 04: Run peeler and 05: Assemble the spike nix file

Usage:
    tridesclousCCV.py [options]

Options:
    --exp=exp                                   which experimental day to analyze
    --blockIdx=blockIdx                         which trial to analyze [default: 1]
    --arrayName=arrayName                       which electrode array to analyze [default: utah]
    --sourceFileSuffix=sourceFileSuffix         which source file to analyze
    --remakePrb                                 whether to try to load MPI [default: False]
    --purgePeeler                               delete previous sort results [default: False]
    --purgePeelerDiagnostics                    delete previous sort results [default: False]
    --batchPrepWaveforms                        extract snippets [default: False]
    --batchRunClustering                        extract features, run clustering [default: False]
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

import tensorflow as tf
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import os, gc, traceback, re
import pdb
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

SLURM_ARRAY_TASK_ID = os.environ.get('SLURM_ARRAY_TASK_ID')
if SLURM_ARRAY_TASK_ID is not None:
    RANK = int(SLURM_ARRAY_TASK_ID)
else:
    RANK = 0

arrayName = arguments['arrayName']
if 'rawBlockName' in spikeSortingOpts[arrayName]:
    ns5FileName = ns5FileName.replace(
        'Block', spikeSortingOpts[arrayName]['rawBlockName'])
    triFolder = os.path.join(
        scratchFolder, 'tdc_{}{:0>3}'.format(
            spikeSortingOpts[arrayName]['rawBlockName'], blockIdx))
else:
    triFolder = os.path.join(
        scratchFolder, 'tdc_Block{:0>3}'.format(blockIdx))
if arguments['sourceFileSuffix'] is not None:
    triFolder = triFolder + '_{}'.format(arguments['sourceFileSuffix'])
spikeSortingOpts[arrayName]['remakePrb'] = arguments['remakePrb']

if arguments['initCatalogConstructor'] and RANK == 0:
    try:
        if arguments['fromNS5']:
            tdch.initialize_catalogueconstructor(
                nspFolder,
                ns5FileName,
                triFolder,
                prbPath=nspPrbPath,
                removeExisting=arguments['removeExistingCatalog'],
                fileFormat='Blackrock')
        else:
            if arguments['sourceFileSuffix'] is not None:
                ns5FileName = (
                    ns5FileName +
                    '_{}'.format(arguments['sourceFileSuffix']))
            tdch.initialize_catalogueconstructor(
                scratchFolder,
                ns5FileName,
                triFolder,
                spikeSortingOpts=spikeSortingOpts[arrayName],
                removeExisting=arguments['removeExistingCatalog'],
                fileFormat='NIX')
    except Exception:
        traceback.print_exc()
        print('Ignoring Exception')

chan_start = int(arguments['chan_start'])
chan_stop = int(arguments['chan_stop'])
dataio = tdc.DataIO(dirname=triFolder)
print(dataio)
# chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[chan_start:chan_stop]
chansToAnalyze = list(range(chan_start, chan_stop))
print('Analyzing channels:\n{}'.format(chansToAnalyze))

theseExtractOpts = dict(
    mode='rand',
    n_left=spikeWindow[0] - 2,
    n_right=spikeWindow[1] + 2,
    nb_max=10000,
    align_waveform=False)
#
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
#
theseClusterOpts = {
    'method': 'agglomerative',
    'n_clusters': 2
    }
#
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
    chunksize=2**18
    )

if arguments['purgePeeler']:
    tdch.purgeNeoBlock(triFolder)
    tdch.purgePeelerResults(
        triFolder, purgeAll=True)
if arguments['purgePeelerDiagnostics']:
    tdch.purgePeelerResults(
        triFolder, diagnosticsOnly=True, purgeAll=True)

if arguments['batchPreprocess']:
    tdch.batchPreprocess(
        triFolder, chansToAnalyze,
        nb_noise_snippet=1000,
        minWaveformRate=None,
        minWaveforms=10,
        alien_value_threshold=100.,
        extractOpts=theseExtractOpts,
        featureOpts=theseFeatureOpts,
        clusterOpts=theseClusterOpts,
        autoMerge=False, auto_merge_threshold=0.99,
        **thesePreprocOpts)

if arguments['batchRunClustering']:
    tdch.batchRunClustering(
        triFolder, chansToAnalyze,
        featureOpts=theseFeatureOpts,
        clusterOpts=theseClusterOpts,
        autoMerge=False, auto_merge_threshold=0.99)

if arguments['batchPrepWaveforms']:
    tdch.batchPrepWaveforms(
        triFolder, chansToAnalyze,
        featureOpts=theseFeatureOpts,
        clusterOpts=theseClusterOpts,
        extractOpts=theseExtractOpts,
        **thesePreprocOpts
        )

if arguments['batchPeel']:
    if spikeSortingOpts[arrayName]['shape_distance_threshold'] is not None:
        shape_distance_threshold = spikeSortingOpts[arrayName]['shape_distance_threshold']
    else:
        shape_distance_threshold = 3
    #
    if spikeSortingOpts[arrayName]['shape_boundary_threshold'] is not None:
        shape_boundary_threshold = spikeSortingOpts[arrayName]['shape_boundary_threshold']
    else:
        shape_boundary_threshold = 4
    #
    tdch.batchPeel(
        triFolder, chansToAnalyze,
        chunksize=thesePreprocOpts['chunksize'],
        shape_distance_threshold=shape_distance_threshold,
        shape_boundary_threshold=shape_boundary_threshold,
        confidence_threshold=spikeSortingOpts[arrayName]['confidence_threshold'],
        energy_reduction_threshold=spikeSortingOpts[arrayName]['energy_reduction_threshold'],
        )


if arguments['overrideSpikeSource']:
    altDataIOInfo = {
        'datasource_type': 'NIX',
        'datasource_kargs': {
            'filenames': [
                os.path.join(
                    scratchFolder,
                    ns5FileName + '_spike_preview_unfiltered.nix')
            ]
        }}
    waveformSignalType = 'initial'
else:
    altDataIOInfo = None
    waveformSignalType = 'processed'


if arguments['makeCoarseNeoBlock']:
    tdch.purgeNeoBlock(triFolder)
    tdch.neo_block_after_peeler(
        triFolder, chan_grps=chansToAnalyze,
        shape_distance_threshold=None, refractory_period=None,
        ignoreTags=[])

if arguments['makeStrictNeoBlock']:
    tdch.purgeNeoBlock(triFolder)
    tdch.neo_block_after_peeler(
        triFolder, chan_grps=chansToAnalyze,
        shape_distance_threshold=spikeSortingOpts[arrayName]['shape_distance_threshold'],
        refractory_period=spikeSortingOpts[arrayName]['refractory_period'],
        shape_boundary_threshold=spikeSortingOpts[arrayName]['shape_boundary_threshold'],
        energy_reduction_threshold=spikeSortingOpts[arrayName]['energy_reduction_threshold'],
        ignoreTags=['so_bad'], altDataIOInfo=altDataIOInfo,
        waveformSignalType=waveformSignalType,
        )

# featureOpts={
#     'method': 'global_pca',
#     'n_components': 5
# },
# featureOpts={
#     'method': 'global_umap',
#     'n_components': 4,
#     'n_neighbors': 75,
#     'min_dist': 0,
#     'metric': 'euclidean',
#     'set_op_mix_ratio': 0.9,
#     'init': 'spectral',
#     'n_epochs': 1000,
# },
#  theseClusterOpts = {
#      'method': 'hdbscan',
#      'min_cluster_size': 100,
#      'min_samples': 50,
#      'allow_single_cluster': False}
