import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
from currentExperiment import *
import os

dataio = tdc.DataIO(dirname=triFolderSource)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:96]
#  import pdb; pdb.set_trace()

for fileNameDest in triDestinations:
    triFolderDest = os.path.join(nspFolder, 'tdc_' + fileNameDest)
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
