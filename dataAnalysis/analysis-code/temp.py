"""

Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
"""

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
import pdb

arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)
pdb.set_trace()