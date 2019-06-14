"""03: Transfer curated templates
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
"""

import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import traceback
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']),
    arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

import os

dataio = tdc.DataIO(dirname=triFolderSource)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:96]

for fileNameDest in triDestinations:
    #  import pdb; pdb.set_trace()
    triFolderDest = os.path.join(
        scratchFolder, 'tdc_{}'.format(fileNameDest))
    try:
        tdch.initialize_catalogueconstructor(
            nspFolder,
            fileNameDest,
            triFolderDest,
            nspPrbPath,
            removeExisting=False, fileFormat='Blackrock')
    except Exception:
        traceback.print_exc()
        #  pass
    #  
    tdch.transferTemplates(
        triFolderSource, triFolderDest,
        chansToAnalyze)
