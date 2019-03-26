import dataAnalysis.preproc.ns5 as preproc
from currentExperiment import *
#   eventInfo=trialFilesFrom['utah']['eventInfo']
reader = preproc.preproc(
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
        fillOverflow=False, removeJumps=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSource='tdc', writeMode='ow',
        chunkSize=2500
        )