import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
from currentExperiment import *
import os

dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:-1]
triFolderSource = triFolder

for i in [2, 3, 4]:
    fileNameDest = trialFilesFrom['utah']['ns5FileName'].replace(
        '{}'.format(trialIdx),
        '{}'.format(i))
    triFolderDest = os.path.join(
        trialFilesFrom['utah']['folderPath'],
        'tdc_' + fileNameDest
        )
    try:
        tdch.initialize_catalogueconstructor(
            trialFilesFrom['utah']['folderPath'],
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
