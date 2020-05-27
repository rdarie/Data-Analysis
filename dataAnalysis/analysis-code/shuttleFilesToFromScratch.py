"""
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze
    --exp=exp                              which experimental day to analyze
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --processAll                           process entire experimental day? [default: False]
    --dataToScratch                        process entire experimental day? [default: False]
    --scratchToData                        process entire experimental day? [default: False]
    --removeSource                         process entire experimental day? [default: False]
    --removeDate                           process entire experimental day? [default: False]
    --removePNGs                           process entire experimental day? [default: False]
    --tdcNIXFromProcessedToScratch         process entire experimental day? [default: False]
"""

#  load options
import pdb, traceback
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import shutil, os, glob
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
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

if arguments['removePNGs']:
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob('{}/**/*.png'.format(scratchFolder), recursive=True)
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except Exception:
            traceback.print_exc()
            print("Error while deleting file : ", filePath)

if arguments['tdcNIXFromProcessedToScratch']:
    tdcFolderName = 'tdc_{}'.format(ns5FileName)
    originPath = os.path.join(
        processedFolder,
        tdcFolderName,
        'tdc_' + ns5FileName + '.nix')
    destinFolder = os.path.join(
        scratchFolder,
        tdcFolderName)
    if not os.path.exists(destinFolder):
        os.makedirs(destinationFolder, exist_ok=True)
    destinPath = os.path.join(
        destinFolder, 'tdc_' + ns5FileName + '.nix'
        )
    shutil.copyfile(originPath, destinPath)