#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""01: Preprocess spikes, then 04: Run peeler and 05: Assemble the spike nix file

Usage:
    tridesclousCCV.py [options]

Options:
    --blockIdx=blockIdx        which trial to analyze [default: 1]
    --exp=exp                  which experimental day to analyze
    --attemptMPI               whether to try to load MPI [default: False]
    --purgePeeler              delete previous sort results [default: False]
    --batchPreprocess          extract snippets and features, run clustering [default: False]
    --batchPeel                run peeler [default: False]
    --makeCoarseNeoBlock       save peeler results to a neo block [default: False]
    --makeStrictNeoBlock       save peeler results to a neo block [default: False]
"""

from docopt import docopt
import traceback
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import tridesclous as tdc

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    HAS_MPI = True
except Exception:
    traceback.print_exc()
    RANK = 0
    HAS_MPI = False

print(RANK)
if RANK == 0:
    from currentExperiment import parseAnalysisOptions
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']),
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

COMM.Barrier()  # sync MPI threads, waith for 0
nspFolder = COMM.bcast(nspFolder, root=0)
nspPrbPath = COMM.bcast(nspPrbPath, root=0)
triFolder = COMM.bcast(triFolder, root=0)
spikeWindow = COMM.bcast(spikeWindow, root=0)
print(triFolder)

if RANK == 0:
    tdch.purgeNeoBlock(triFolder)
    tdch.purgePeelerResults(
        triFolder, purgeAll=True)
    dataio = tdc.DataIO(dirname=triFolder)
    chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:96]
else:
    chansToAnalyze = None

print("triFolder: {}".format(triFolder))
COMM.Barrier()  # sync MPI threads, waith for 0 to gather chansToAnalyze
chansToAnalyze = COMM.bcast(chansToAnalyze, root=0)
print("RANK: {}".format(RANK))