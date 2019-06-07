"""06a: Preprocess the NS5 File

Usage:
    preprocNS5.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --makeFull                      whether to make a .nix file that has all raw traces [default: False]
    --makeTruncated                 whether to make a .nix file that only has analog inputs [default: False]
"""

from docopt import docopt
import dataAnalysis.preproc.ns5 as preproc
from currentExperiment import *
import pdb

arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    miniRCTrial = miniRCTrialLookup[trialIdx]
    #
    trialFilesFrom['utah']['calcRigEvents'] = not miniRCTrial
    #
    if miniRCTrial:
        trialFilesFrom['utah'].update({
            'eventInfo': {'inputIDs': miniRCRigInputs}
        })
    else:
        trialFilesFrom['utah'].update({
            'eventInfo': {'inputIDs': fullRigInputs}
        })

chunkSize = 2600
chunkList = [0]
equalChunks = False

if arguments['--makeTruncated']:
    analogInputNames = sorted(
        trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
    reader = preproc.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        asigNameList=analogInputNames,
        spikeSourceType='tdc', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList, calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )

if arguments['--makeFull']:
    reader = preproc.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSourceType='tdc', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList, nameSuffix='_full',
        calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )
