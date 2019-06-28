"""

Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
"""

from currentExperiment import parseAnalysisOptions
from docopt import docopt
import pdb

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
pdb.set_trace()