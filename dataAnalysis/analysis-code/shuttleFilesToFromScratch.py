"""
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                          which trial to analyze
    --exp=exp                                    which experimental day to analyze
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --processAll                                 process entire experimental day? [default: False]
    --analysisFolderFromScratchToData            process entire experimental day? [default: False]
    --alignFolderFromScratchToData               process entire experimental day? [default: False]
    --dataToScratch                              process entire experimental day? [default: False]
    --scratchToData                              process entire experimental day? [default: False]
    --removeSource                               process entire experimental day? [default: False]
    --removeDate                                 process entire experimental day? [default: False]
    --removePNGs                                 process entire experimental day? [default: False]
    --tdcNIXFromProcessedToScratch               process entire experimental day? [default: False]
    --preprocFolderFilesFromScratchToData        process entire experimental day? [default: False]
    --preprocFolderFilesFromDataToScratch        process entire experimental day? [default: False]
    --purgePreprocFolder                         process entire experimental day? [default: False]
    --purgeAnalysisFolder                        process entire experimental day? [default: False]
    --purgeAlignFolder                           process entire experimental day? [default: False]
    --forcePurge                                 process entire experimental day? [default: False]
"""

#  load options
import pdb, traceback
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import shutil, os, glob
from distutils.dir_util import copy_tree, remove_tree
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
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
if arguments['dataToScratch']:
    originPath = dataPath
    destinPath = scratchPath
    shutil.copyfile(originPath, destinPath)

if arguments['scratchToData']:
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
        os.makedirs(destinFolder, exist_ok=True)
    destinPath = os.path.join(
        destinFolder, 'tdc_' + ns5FileName + '.nix'
        )
    shutil.copyfile(originPath, destinPath)

if arguments['analysisFolderFromScratchToData']:
    if os.path.exists(analysisSubFolder):
        destinationFolder = os.path.join(
            processedFolder, arguments['analysisName']
            )
        copy_tree(analysisSubFolder, destinationFolder)
    else:
        print('{} does not exist!!'.format(analysisSubFolder))

if arguments['alignFolderFromScratchToData']:
    if os.path.exists(alignSubFolder):
        destinationFolder = os.path.join(
            processedFolder, arguments['alignFolderName']
            )
        if not os.path.exists(destinationFolder):
            os.makedirs(destinationFolder, exist_ok=True)
        copy_tree(alignSubFolder, destinationFolder)
    else:
        print('{} does not exist!!'.format(analysisSubFolder))

if arguments['preprocFolderFilesFromScratchToData']:
    for itemName in os.listdir(scratchFolder):
        originPath = os.path.join(scratchFolder, itemName)
        if os.path.isfile(originPath):
            destinPath = os.path.join(processedFolder, itemName)
            shutil.copyfile(originPath, destinPath)
            print('Copying from\n{}\ninto{}'.format(originPath, destinPath))

if arguments['preprocFolderFilesFromDataToScratch']:
    for itemName in os.listdir(processedFolder):
        originPath = os.path.join(processedFolder, itemName)
        if os.path.isfile(originPath):
            destinPath = os.path.join(scratchFolder, itemName)
            shutil.copyfile(originPath, destinPath)
            print('Copying from\n{}\ninto{}'.format(originPath, destinPath))

if arguments['purgePreprocFolder']:
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob('{}/*'.format(scratchFolder), recursive=True)
    # Iterate over the list of filepaths & remove each file.
    # pdb.set_trace()
    safeToPurge = True
    for filePath in fileList:
        if os.path.isfile(filePath):
            itemName = os.path.basename(filePath)
            destinPath = os.path.join(processedFolder, itemName)
            if not os.path.exists(destinPath):
                safeToPurge = False
                print('{} is missing from processed!'.format(destinPath))
    if safeToPurge or arguments['forcePurge']:
        print('purging {}'.format(scratchFolder))
        shutil.rmtree(scratchFolder)