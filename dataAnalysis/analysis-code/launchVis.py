"""Visualize NIX and Ns5 Files

Usage:
    preprocNS5.py [options]

Options:
    --blockIdx=blockIdx                which trial to analyze
    --exp=exp                          which experimental day to analyze
"""

import dataAnalysis.ephyviewer.scripts as vis_scripts
import pdb, os, sys
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

os.chdir(scratchPath)
sys.argv = [sys.argv[0]]
vis_scripts.launch_standalone_ephyviewer()
