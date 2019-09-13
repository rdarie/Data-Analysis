"""

Usage:
    printLineProfilerStats.py [options]

Options:
    --scriptName=scriptName    which result to analyze
"""

#  load options
import line_profiler, pdb, sys
import collections
import numpy as np
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
#

#
