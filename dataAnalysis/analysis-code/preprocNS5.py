"""06a: Preprocess the NS5 File

Usage:
    preprocNS5.py [options]

Options:
    --blockIdx=blockIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --makeFull                      whether to make a .nix file that has all raw traces [default: False]
    --previewMotorEncoder           whether to make a .nix file to preview the motor encoder analysis [default: False]
    --makeTruncated                 whether to make a .nix file that only has analog inputs [default: False]
    --maskMotorEncoder              whether to ignore motor encoder activity outside the alignTimeBounds window [default: False]
    --ISI                           special options for parsing Ripple files from ISI [default: False]
"""

import dataAnalysis.preproc.ns5 as ns5
import pdb

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

chunkSize = 4000
chunkList = [0]
equalChunks = False
if arguments['maskMotorEncoder']:
    motorEncoderMask = alignTimeBoundsLookup[int(arguments['blockIdx'])]
else:
    motorEncoderMask = None
#
if arguments['previewMotorEncoder']:
    analogInputNames = sorted(
        trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
    assert trialFilesFrom['utah']['calcRigEvents']
    reader =ns5.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        motorEncoderMask=None,
        calcAverageLFP=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        asigNameList=analogInputNames,
        spikeSourceType='nev', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList,
        nameSuffix='_motorPreview',
        calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )
#
if arguments['makeTruncated']:
    analogInputNames = sorted(
        trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
    reader =ns5.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        motorEncoderMask=motorEncoderMask,
        calcAverageLFP=True,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        asigNameList=analogInputNames,
        spikeSourceType='tdc', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList, calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )
# 
if arguments['makeFull']:
    reader = ns5.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        motorEncoderMask=motorEncoderMask,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSourceType='tdc', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList, nameSuffix='_full',
        calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )
#
if arguments['ISI']:
    reader = ns5.preproc(
        fileName=ns5FileName,
        rawFolderPath=nspFolder,
        outputFolderPath=scratchFolder,
        fillOverflow=False, removeJumps=False,
        motorEncoderMask=motorEncoderMask,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSourceType='nev', writeMode='ow',
        chunkSize=chunkSize, equalChunks=equalChunks,
        chunkList=chunkList,
        calcRigEvents=trialFilesFrom['utah']['calcRigEvents']
        )
