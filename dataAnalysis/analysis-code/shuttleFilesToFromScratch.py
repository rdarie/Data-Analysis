"""
Usage:
    calcTrialAnalysisNix.py [options]

Options:
    --trialIdx=trialIdx               which trial to analyze
    --exp=exp                         which experimental day to analyze
    --analysisName=analysisName       append a name to the resulting blocks? [default: default]
    --processAll                      process entire experimental day? [default: False]
    --dataToScratch                   process entire experimental day? [default: False]
    --scratchToData                   process entire experimental day? [default: False]
    --removeSource                    process entire experimental day? [default: False]
    --removeDate                      process entire experimental day? [default: False]
"""

#  load options
import pdb
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import shutil, os
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

scratchPath = os.path.join(
    scratchFolder,
    prefix + '.nix')
dataPath = os.path.join(
    processedFolder,
    prefix + '.nix')

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)

if arguments['dataToScratch']:
    originPath = dataPath
    destinPath = scratchPath
    shutil.copyfile(originPath, destinPath)
elif arguments['scratchToData']:
    originPath = scratchPath
    destinPath = dataPath
    shutil.copyfile(originPath, destinPath)

#  Try to delete the file ##
if arguments['removeSource']:
    try:
        os.remove(originPath)
    except OSError as e:
        #  if failed, report it back to the user ##
        print("Error: %s - %s." % (e.filename, e.strerror))

if arguments['removeDate']:
    for fileName in os.listdir(analysisSubFolder):
        filePath = os.path.join(analysisSubFolder, fileName)
        if os.path.isfile(filePath):
            if experimentName in filePath:
                os.rename(
                    filePath,
                    os.path.join(
                        analysisSubFolder,
                        fileName.replace(experimentName, assembledName)))