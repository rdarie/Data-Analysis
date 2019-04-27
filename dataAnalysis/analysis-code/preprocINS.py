"""
Usage:
    preprocINSData.py [options]

Options:
    --trialIdx=trialIdx   which trial to analyze 
    --miniRCTrial         whether this is a movement trial [default: False]
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
    insDataPath = os.path.join(
        remoteBasePath, 'processed', experimentName,
        ns5FileName + '_ins.nix'
    )
    trialFilesStim = {
        'ins': {
            'origin': 'ins',
            'experimentName': experimentName,
            'folderPath': insFolder,
            'ns5FileName': ns5FileName,
            'jsonSessionNames': jsonSessionNames[trialIdx],
            'elecIDs': range(17),
            'excludeClus': [],
            'forceRecalc': True,
            'detectStim': True,
            'getINSkwargs': {
                'stimDetectOpts': stimDetectOpts,
                'fixedDelay': 0e-3,
                'delayByFreqMult': 1,
                'gaussWid': 100e-3,
                'minDist': 0.2, 'minDur': 0.2,
                'cyclePeriodCorrection': 17.5e-3,
                'plotAnomalies': True,
                'recalculateExpectedOffsets': False,
                'maxSpikesPerGroup': 1, 'plotting': [] # range(1, 1000, 5)
                }
            }
        }
if arguments['--miniRCTrial']:
    trialFilesStim['ins']['getINSkwargs'].update({
        'minDist': 1.2,
        'maxSpikesPerGroup': 0,
        'gaussWid': 200e-3,
        'recalculateExpectedOffsets': True,
    })

insBlock = preprocINS.preprocINS(
    trialFilesStim['ins'],
    insDataPath, plottingFigures=False)