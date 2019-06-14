"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --trialIdx=trialIdx        which trial to analyze
    --exp=exp                  which experimental day to analyze
"""

import dataAnalysis.preproc.mdt as preprocINS
import os
import dataAnalysis.ephyviewer.scripts as vis_scripts
from importlib import reload

#  load options
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']),
    arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

insBlock = preprocINS.preprocINS(
    trialFilesStim['ins'],
    insDataPath, plottingFigures=False)