"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --trialIdx=trialIdx   which trial to analyze
"""

import dataAnalysis.preproc.mdt as preprocINS
from docopt import docopt
import os
from currentExperiment import *
import dataAnalysis.ephyviewer.scripts as vis_scripts
from importlib import reload

arguments = docopt(__doc__)
'''
arguments = {
    '--trialIdx': '5',
    '--miniRCTrial': True
    }
'''
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    #  insDataPath = os.path.join(
    #      remoteBasePath, 'raw', experimentName,
    #      ns5FileName + '_ins.nix'
    #  )
    insDataPath = os.path.join(
        scratchFolder,
        ns5FileName + '_ins.nix')
    trialFilesStim['ins']['ns5FileName'] = ns5FileName
    trialFilesStim['ins']['jsonSessionNames'] = jsonSessionNames[trialIdx]
    miniRCTrial = miniRCTrialLookup[trialIdx]

if miniRCTrial:
    trialFilesStim['ins']['getINSkwargs'].update(miniRCDetectionOpts)

insBlock = preprocINS.preprocINS(
    trialFilesStim['ins'],
    insDataPath, plottingFigures=False)