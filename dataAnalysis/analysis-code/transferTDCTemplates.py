"""
Usage:
    transferTDCTemplates.py [--trialIdx=trialIdx]

Arguments:

Options:
    --trialIdx=trialIdx            which trial to pull templates from
"""
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
from currentExperiment import *
import os

from docopt import docopt
arguments = docopt(__doc__)

#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    triFolder = os.path.join(
        nspFolder, 'tdc_' + ns5FileName)

dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:-1]
triFolderSource = triFolder

for i in [1, 2, 4, 5]:
    fileNameDest = ns5FileName.replace(
        '{}'.format(trialIdx),
        '{}'.format(i))
    triFolderDest = os.path.join(
        nspFolder,
        'tdc_' + fileNameDest
        )
    try:
        tdch.initialize_catalogueconstructor(
            nspFolder,
            fileNameDest,
            triFolderDest,
            nspPrbPath,
            removeExisting=False, fileFormat='Blackrock')
    except Exception:
        import traceback
        traceback.print_exc()
        pass
    
    tdch.transferTemplates(
        triFolderSource, triFolderDest,
        chansToAnalyze)
