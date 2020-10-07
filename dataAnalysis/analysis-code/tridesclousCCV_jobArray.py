#!/users/rdarie/anaconda/nda/bin/python
"""01: Preprocess spikes, then 04: Run peeler and 05: Assemble the spike nix file

Usage:
    tridesclousCCV.py [options]

Options:
    --exp=exp                      which experimental day to analyze
    --blockIdx=blockIdx            which trial to analyze [default: 1]
    --arrayName=arrayName          which electrode array to analyze [default: utah]
    --sourceFile=sourceFile        which source file to analyze [default: raw]
    --attemptMPI                   whether to try to load MPI [default: False]
    --remakePrb                    whether to try to load MPI [default: False]
    --removeExistingCatalog        delete previous sort results [default: False]
    --purgePeeler                  delete previous sort results [default: False]
    --purgePeelerDiagnostics       delete previous sort results [default: False]
    --batchPrepWaveforms           extract snippets [default: False]
    --batchRunClustering           extract features, run clustering [default: False]
    --batchPreprocess              extract snippets and features, run clustering [default: False]
    --batchPeel                    run peeler [default: False]
    --makeCoarseNeoBlock           save peeler results to a neo block [default: False]
    --makeStrictNeoBlock           save peeler results to a neo block [default: False]
    --exportSpikesCSV              save peeler results to a csv file [default: False]
    --chan_start=chan_start        which chan_grp to start on [default: 0]
    --chan_stop=chan_stop          which chan_grp to stop on [default: 47]
"""

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

import tensorflow as tf
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import os, gc, traceback, re
import pdb
from numba.core.errors import NumbaPerformanceWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
#
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
#

RANK = os.getenv('SLURM_ARRAY_TASK_ID')

from currentExperiment import parseAnalysisOptions
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
arrayName = arguments['arrayName']

ns5FileName = ns5FileName.replace('Block', arrayName)
triFolder = os.path.join(
    scratchFolder, 'tdc_{}{:0>3}'.format(arrayName, blockIdx))

chan_start = int(arguments['chan_start'])
chan_stop = int(arguments['chan_stop'])
dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[chan_start:chan_stop]
print('Analyzing channels:\n{}'.format(chansToAnalyze))

theseExtractOpts = dict(
    mode='rand',
    n_left=spikeWindow[0] - 2,
    n_right=spikeWindow[1] + 2,
    nb_max=16000, align_waveform=False)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `loss` is no longer improving
        monitor="loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    )
]
theseFeatureOpts = {
    'method': 'global_pumap',
    'n_components': 6,
    'n_neighbors': 50,
    'min_dist': 0,
    'metric': 'euclidean',
    'set_op_mix_ratio': 0.9,
    'parametric_reconstruction': False,
    'autoencoder_loss': False,
    'verbose': False,
    'batch_size': 10000,
    'n_training_epochs': 5,
    'keras_fit_kwargs': {'verbose': 2, 'callbacks': callbacks}
}

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
theseClusterOpts = {
    'method': 'hdbscan',
    'min_cluster_size': 100,
    'min_samples': 50,
    'allow_single_cluster': False}

if RANK == 0:
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
        relative_threshold=3.5,
        fill_overflow=False,
        highpass_freq=300.,
        lowpass_freq=5000.,
        common_ref_freq=300.,
        common_ref_removal=False,
        notch_freq=None,
        filter_order=3,
        featureOpts=theseFeatureOpts,
        clusterOpts=theseClusterOpts,
        noise_estimate_duration=240,
        sample_snippet_duration=240,
        chunksize=2**18,
        extractOpts=theseExtractOpts,
        autoMerge=False, auto_merge_threshold=0.99)

if arguments['batchRunClustering']:
    tdch.batchRunClustering(
        triFolder, chansToAnalyze,
        featureOpts=theseFeatureOpts,
        clusterOpts={
            'method': 'hdbscan',
            'min_cluster_size': 100,
            'min_samples': 50,
            'allow_single_cluster': True},
        autoMerge=False, auto_merge_threshold=0.99)

if arguments['batchPrepWaveforms']:
    tdch.batchPrepWaveforms(
        triFolder, chansToAnalyze,
        relative_threshold=4,
        fill_overflow=False,
        highpass_freq=300.,
        lowpass_freq=5000.,
        common_ref_freq=300.,
        common_ref_removal=False,
        notch_freq=None,
        filter_order=3,
        noise_estimate_duration=300,
        sample_snippet_duration=300,
        chunksize=2**18,
        extractOpts=dict(
            mode='rand',
            n_left=spikeWindow[0] - 2,
            n_right=spikeWindow[1] + 2,
            nb_max=32000, align_waveform=False)
        )

if arguments['batchPeel']:
    tdch.batchPeel(
        triFolder, chansToAnalyze,
        shape_boundary_threshold=3,
        confidence_threshold=0.65,
        shape_distance_threshold=2)

if arguments['exportSpikesCSV'] and RANK == 0:
    tdch.export_spikes_after_peeler(triFolder)

if arguments['makeCoarseNeoBlock'] and RANK == 0:
    tdch.purgeNeoBlock(triFolder)
    tdch.neo_block_after_peeler(
        triFolder, chan_grps=chansToAnalyze,
        shape_distance_threshold=None, refractory_period=None,
        ignoreTags=[])

if arguments['makeStrictNeoBlock'] and RANK == 0:
    tdch.purgeNeoBlock(triFolder)
    tdch.neo_block_after_peeler(
        triFolder, chan_grps=chansToAnalyze,
        shape_distance_threshold=None, refractory_period=2.5e-3,
        ignoreTags=['so_bad'])
