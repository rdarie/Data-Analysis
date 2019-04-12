#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    tridesclousCCV.py [options]

Arguments:

Options:
    --trialIdx=trialIdx        which trial to analyze [default: 1]
    --attemptMPI               whether to try to load MPI [default: False]
    --purgePeeler              delete previous sort results [default: False]
    --batchPreprocess          extract snippets and features, run clustering [default: False]
    --visConstructor           include visualization step for catalogue constructor [default: False]
    --batchPeel                run peeler [default: False]
    --visPeeler                include visualization step for catalogue [default: False]
    --makeNeoBlock             save peeler results to a neo block [default: False]
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.helper_functions as hf
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
    from currentExperiment import *
    #  if overriding currentExperiment
    if arguments['--trialIdx']:
        trialIdx = int(arguments['--trialIdx'])
        ns5FileName = 'Trial00{}'.format(trialIdx)
        triFolder = os.path.join(
            nspFolder, 'tdc_' + ns5FileName)
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


if HAS_MPI:
    COMM.Barrier()  # sync MPI threads, waith for 0
    nspFolder = COMM.bcast(nspFolder, root=0)
    nspPrbPath = COMM.bcast(nspPrbPath, root=0)
    triFolder = COMM.bcast(triFolder, root=0)


if RANK == 0:
    if arguments['--purgePeeler']:
        tdch.purgePeelerResults(
            triFolder, purgeAll=True)
    
    dataio = tdc.DataIO(dirname=triFolder)
    chansToAnalyze = sorted(list(dataio.channel_groups.keys()))
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
        relative_threshold=5.5)

chansToAnalyze = chansToAnalyze[:-1]

if arguments['--visConstructor']:
    for idx, chan_grp in enumerate(chansToAnalyze):
        if idx % SIZE == RANK:
            tdch.open_cataloguewindow(triFolder, chan_grp=chan_grp)

if arguments['--batchPeel']:
    tdch.batchPeel(
        triFolder, chansToAnalyze,
        shape_distance_threshold=2e-2)

if arguments['--visPeeler']:
    for idx, chan_grp in enumerate(chansToAnalyze):
        if idx % SIZE == RANK:
            tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)

if HAS_MPI:
    COMM.Barrier()  # wait until all threads finish sorting

if arguments['--makeNeoBlock'] and RANK == 0:
    tdch.neo_block_after_peeler(triFolder, chan_grps=chansToAnalyze)
