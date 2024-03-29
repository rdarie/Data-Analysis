"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --blockIdx=blockIdx        which trial to analyze
    --exp=exp                  which experimental day to analyze
"""

import dataAnalysis.preproc.mdt as preprocINS
import os, pdb
import dataAnalysis.ephyviewer.scripts as vis_scripts
from importlib import reload
from datetime import datetime as dt
import tzlocal
import pandas as pd
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

#)
sessionNames = pd.DataFrame([
    f
    for f in os.listdir(insFolder)
    if 'Session' in f
    ], columns=['name'])
sessionNames.loc[:, 'unixTime'] = (
    sessionNames['name']
    .apply(str.lstrip, args=(['Session']))
    .astype('int') / 1000.)
local_timezone = tzlocal.get_localzone()  # get pytz timezone
sessionNames.loc[:, 'timestamp'] = (
    sessionNames['unixTime']
    .apply(dt.fromtimestamp, tz=local_timezone)
    .dt.strftime("%Y-%m-%d-%H:%M"))
sessionNames.sort_values('timestamp', inplace=True)
sessionNames.reset_index(drop=True, inplace=True)
sessionNames.to_csv(os.path.join(insFolder, 'formattedNames.csv'))
