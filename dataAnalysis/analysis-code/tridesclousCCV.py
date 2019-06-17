#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""01: Preprocess spikes, then 04: Run peeler and 05: Assemble the spike nix file

Usage:
    tridesclousCCV.py [options]

Options:
    --trialIdx=trialIdx        which trial to analyze [default: 1]
    --exp=exp                  which experimental day to analyze
    --attemptMPI               whether to try to load MPI [default: False]
    --purgePeeler              delete previous sort results [default: False]
    --batchPreprocess          extract snippets and features, run clustering [default: False]
    --batchPeel                run peeler [default: False]
    --makeCoarseNeoBlock       save peeler results to a neo block [default: False]
    --makeStrictNeoBlock       save peeler results to a neo block [default: False]
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import os, gc, traceback

arguments = docopt(__doc__)

try:
    if not arguments['--attemptMPI']:
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
    from currentExperiment_alt import parseAnalysisOptions
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['--trialIdx']),
        arguments['--exp'])
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
    if arguments['--purgePeeler']:
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

if arguments['--batchPreprocess']:
    tdch.batchPreprocess(
        triFolder, chansToAnalyze,
        n_components_by_channel=15,
        cluster_method='agglomerative',
        n_clusters=5,
        noise_estimate_duration=900.,
        sample_snippet_duration=900.,
        chunksize=2**13, n_left=spikeWindow[0] - 2,
        n_right=spikeWindow[1] + 2,
        align_waveform=False, subsample_ratio=10,
        autoMerge=True, auto_merge_threshold=0.85,
        relative_threshold=5.5, attemptMPI=HAS_MPI)

if arguments['--batchPeel']:
    tdch.batchPeel(
        triFolder, chansToAnalyze,
        shape_boundary_threshold=3,
        shape_distance_threshold=1.5, attemptMPI=HAS_MPI)

if HAS_MPI:
    COMM.Barrier()  # wait until all threads finish sorting

if arguments['--makeCoarseNeoBlock'] and RANK == 0:
    tdch.purgeNeoBlock(triFolder)
    tdch.neo_block_after_peeler(
        triFolder, chan_grps=chansToAnalyze,
        shape_distance_threshold=None, refractory_period=None,
        ignoreTags=['so_bad'])

if arguments['--makeStrictNeoBlock'] and RANK == 0:
    tdch.purgeNeoBlock(triFolder)
    tdch.neo_block_after_peeler(
        triFolder, chan_grps=chansToAnalyze,
        shape_distance_threshold=None, refractory_period=3.5e-3,
        ignoreTags=['so_bad'])
