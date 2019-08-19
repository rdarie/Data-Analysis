"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --trialIdx=trialIdx        which trial to analyze
    --exp=exp                  which experimental day to analyze
    --showPlots                show plots? [default: False]
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate interactive qt output
import dataAnalysis.preproc.mdt as preprocINS
import os
from importlib import reload
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

insBlock = preprocINS.preprocINS(
    trialFilesStim['ins'],
    insDataPath, plottingFigures=arguments['showPlots'])