"""06a: Preprocess the NS5 File

Usage:
    preprocNS5.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --makeFull                      whether to make a .nix file that has all raw traces [default: False]
    --makeTruncated                 whether to make a .nix file that only has analog inputs [default: False]
"""

import dataAnalysis.preproc.ns5 as ns5
import pdb

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

chunkSize = 2900
chunkList = [0]
equalChunks = False

if arguments['makeTruncated']:
    analogInputNames = sorted(
        trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
    reader =ns5.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        calcAverageLFP=True,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        asigNameList=analogInputNames,
        spikeSourceType='tdc', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList, calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )

if arguments['makeFull']:
    reader = ns5.preproc(
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
