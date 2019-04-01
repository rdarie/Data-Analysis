"""
Usage:
    preprocNS5.py [--trialIdx=trialIdx]

Arguments:
    trialIdx            which trial to analyze
"""

from docopt import docopt
import dataAnalysis.preproc.ns5 as preproc
from currentExperiment import *

arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)

reader = preproc.preproc(
        fileName=ns5FileName,
        folderPath=nspFolder,
        fillOverflow=False, removeJumps=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSource='tdc', writeMode='ow',
        chunkSize=2500
        )
