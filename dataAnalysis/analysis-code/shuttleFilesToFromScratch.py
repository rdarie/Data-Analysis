"""
Usage:
    calcTrialAnalysisNix.py [options]

Options:
    --trialIdx=trialIdx               which trial to analyze
    --exp=exp                         which experimental day to analyze
    --processAll                      process entire experimental day? [default: False]
    --dataToScratch                   process entire experimental day? [default: False]
    --removeSource                    process entire experimental day? [default: False]
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
    prefix = experimentName
else:
    prefix = ns5FileName

scratchPath = os.path.join(
    scratchFolder,
    prefix + '.nix')
dataPath = os.path.join(
    processedFolder,
    prefix + '.nix')

if arguments['dataToScratch']:
    originPath = dataPath
    destinPath = scratchPath
else:
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
