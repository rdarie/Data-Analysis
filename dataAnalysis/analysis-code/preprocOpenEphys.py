"""06c: Preprocess the open ephys recording File

Usage:
    preprocNS5.py [options]

Options:
    --exp=exp                       which experimental day to analyze
"""
import dataAnalysis.preproc.open_ephys as ppOE
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pdb, traceback
import quantities as pq
import os
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    experimentShorthand=arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

trialList = sorted([
    f
    for f in os.listdir(oeFolder)
    if (not os.path.isfile(os.path.join(oeFolder, f))) and ('Trial008' in f)
    ])

# pdb.set_trace()
for folderPath in trialList:
    try:
        print('Loading {}...'.format(folderPath))
        ppOE.preprocOpenEphysFolder(
            os.path.join(oeFolder, folderPath),
            chanNames=openEphysChanNames, plotting=True,
            filterOpts=openEphysFilterOpts)
    except Exception:
        traceback.print_exc()
