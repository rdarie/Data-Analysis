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
#  !!! kludgey solution to problem of transfering template across days

triFolder = os.path.join(
    '..', 'raw', '201901271000-Proprio', 'tdc_Trial003')
dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:-1]
triFolderSource = triFolder

for i in [4]:
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
