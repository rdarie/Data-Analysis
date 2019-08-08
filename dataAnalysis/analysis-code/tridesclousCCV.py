#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""01: Preprocess spikes, then 04: Run peeler and 05: Assemble the spike nix file

Usage:
    tridesclousCCV.py [options]

Options:
    --exp=exp                  which experimental day to analyze
    --trialIdx=trialIdx        which trial to analyze [default: 1]
    --attemptMPI               whether to try to load MPI [default: False]
    --purgePeeler              delete previous sort results [default: False]
    --batchPreprocess          extract snippets and features, run clustering [default: False]
    --batchPeel                run peeler [default: False]
    --makeCoarseNeoBlock       save peeler results to a neo block [default: False]
    --makeStrictNeoBlock       save peeler results to a neo block [default: False]
    --exportSpikesCSV          save peeler results to a csv file [default: False]
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import os, gc, traceback

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

try:
    if not arguments['attemptMPI']:
        raise(Exception('MPI aborted by cmd line argument'))
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    HAS_MPI = True
except Exception:
    traceback.print_exc()
    RANK = 0
    SIZE = 1
    HAS_MPI = False

if RANK == 0:
    from currentExperiment import parseAnalysisOptions
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['trialIdx']),
        arguments['exp'])
    print("globals:")
    print(globals().keys())
    print('allOpts:')
    print(allOpts.keys())
    print('expOpts:')
    print(expOpts.keys())
    globals().update(expOpts)
    globals().update(allOpts)
    try:
        tdch.initialize_catalogueconstructor(
            nspFolder,
            ns5FileName,
            triFolder,
            nspPrbPath,
            removeExisting=False, fileFormat='Blackrock')
    except Exception:
        pass
else:
    nspFolder = None
    nspPrbPath = None
    triFolder = None
    spikeWindow = None

if HAS_MPI:
    COMM.Barrier()  # sync MPI threads, waith for 0
    nspFolder = COMM.bcast(nspFolder, root=0)
    nspPrbPath = COMM.bcast(nspPrbPath, root=0)
    triFolder = COMM.bcast(triFolder, root=0)
    spikeWindow = COMM.bcast(spikeWindow, root=0)

if RANK == 0:
    if arguments['purgePeeler']:
        tdch.purgeNeoBlock(triFolder)
        tdch.purgePeelerResults(
            triFolder, purgeAll=True)
    dataio = tdc.DataIO(dirname=triFolder)
    chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:96]
else:
    chansToAnalyze = None

if HAS_MPI:
    COMM.Barrier()  # sync MPI threads, waith for 0 to gather chansToAnalyze
    chansToAnalyze = COMM.bcast(chansToAnalyze, root=0)

'''
chansToAnalyze = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95]
'''
if arguments['batchPreprocess']:
    tdch.batchPreprocess(
        triFolder, chansToAnalyze,
        relative_threshold=4,
        highpass_freq=300.,
        lowpass_freq=3000.,
        filter_order=8,
        featureOpts={
            'method': 'global_umap',
            'n_components': 5,
            'n_neighbors': 30,
            'min_dist': 0,
        },
        clusterOpts={
            'method': 'hdbscan',
            'min_cluster_size': 30},
        noise_estimate_duration='all',
        sample_snippet_duration='all',
        chunksize=2**20,
        extractOpts=dict(
            mode='rand',
            n_left=spikeWindow[0] - 2,
            n_right=spikeWindow[1] + 2,
            nb_max=10000, align_waveform=False),
        autoMerge=False, auto_merge_threshold=0.99,
        attemptMPI=HAS_MPI)

if arguments['batchPeel']:
    tdch.batchPeel(
        triFolder, chansToAnalyze,
        # shape_boundary_threshold=3,
        shape_distance_threshold=2, attemptMPI=HAS_MPI)

if HAS_MPI:
    COMM.Barrier()  # wait until all threads finish sorting

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
        shape_distance_threshold=None, refractory_period=1.5e-3,
        ignoreTags=['so_bad'])
