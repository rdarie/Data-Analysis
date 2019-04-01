"""
Usage:
    preprocINSData.py [--trialIdx=trialIdx]

Arguments:
    trialIdx            which trial to analyze
"""

import dataAnalysis.preproc.mdt as preprocINS
from docopt import docopt
from currentExperiment import *
import dataAnalysis.ephyviewer.scripts as vis_scripts

arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
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
                'stimIti': 0, 'fixedDelay': 10e-3,
                'minDist': 0.2, 'minDur': 0.2, 'thres': 3,
                'gaussWid': 200e-3,
                'gaussKerWid': 75e-3,
                'maxSpikesPerGroup': 1, 'plotting': []  # range(1, 1000, 5)
                }
            }
        }

insBlock = preprocINS.preprocINS(
    trialFilesStim['ins'], plottingFigures=False)