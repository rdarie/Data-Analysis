#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    tridesclousCCV.py [options]

Arguments:

Options:
    --trialIdx=trialIdx        which trial to analyze [default: 1]
    --purgePeeler              delete previous sort results [default: False]
    --batchPreprocess          extract snippets and features, run clustering [default: False]
    --visConstructor           include visualization step for catalogue constructor [default: False]
    --batchPeel                run peeler [default: False]
    --visPeeler                include visualization step for catalogue [default: False]
    --makeNeoBlock             save peeler results to a neo block [default: False]
"""

from docopt import docopt
import traceback

print(__doc__)
arguments = docopt(__doc__)
print(arguments)

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
