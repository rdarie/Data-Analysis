"""
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                              which trial to analyze [default: 1]
    --exp=exp                                        which experimental day to analyze
    --analysisName=analysisName                      append a name to the resulting blocks? [default: default]
    --processAll                                     process entire experimental day? [default: False]
    --alignFolderName=alignFolderName                append a name to the resulting blocks? [default: motion]
    --searchTerm=searchTerm                          shuttle all files and folders that include this term
    --excludeSearchTerm=excludeSearchTerm            shuttle all files and folders that include this term
    --moveItems                                      move items, as opposed to copy them
    --deleteItems                                    remove items entirely from origin path
    --fromScratchToData                              process entire experimental day? [default: False]
    --fromDataToScratch                              process entire experimental day? [default: False]
    --preprocFolderFiles                             process entire experimental day? [default: False]
    --analysisFolderFromScratchToData                process entire experimental day? [default: False]
    --alignFolderFromScratchToData                   process entire experimental day? [default: False]
    --removeSource                                   process entire experimental day? [default: False]
    --removeDate                                     process entire experimental day? [default: False]
    --removePNGs                                     process entire experimental day? [default: False]
    --tdcNIXFromProcessedToScratch                   process entire experimental day? [default: False]
    --purgePreprocFolder                             process entire experimental day? [default: False]
    --purgeAnalysisFolder                            process entire experimental day? [default: False]
    --purgeAlignFolder                               process entire experimental day? [default: False]
    --forcePurge                                     process entire experimental day? [default: False]
    --preprocFolderSubfolders                        process entire experimental day? [default: False]
    --filesIncluded                                  process entire experimental day? [default: False]
    --foldersIncluded                                process entire experimental day? [default: False]
    --requireKeypress                                process entire experimental day? [default: False]
    
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

#######################
#  Global moves
#######################
if arguments['fromDataToScratch']:
    originFolder, destinFolder = processedFolder, scratchFolder
elif arguments['fromScratchToData']:
    originFolder, destinFolder = scratchFolder, processedFolder
print('Moving items from {} to {}'.format(originFolder, destinFolder))

if arguments['excludeSearchTerm'] is not None:
    itemsToExclude = glob.glob(
        os.path.join(
            originFolder, '**', '{}'.format(arguments['excludeSearchTerm'])),
        recursive=True)
else:
    itemsToExclude = []

if arguments['preprocFolderFiles']:
    itemsToMove = [
        itemName
        for itemName in os.listdir(originFolder)
        if os.path.isfile(os.path.join(originFolder, itemName))
        ]
    print('\nAbout to move:\n')
    print('\n'.join(itemsToMove))
    if arguments['requireKeypress']:
        _ = input('\n********\nPress any key to continue.')
    for itemName in os.listdir(originFolder):
        originPath = os.path.join(originFolder, itemName)
        destinPath = os.path.join(destinFolder, itemName)
        if os.path.isfile(originPath):
            if arguments['moveItems']:
                shutil.move(originPath, destinPath)
            elif arguments['deleteItems']:
                os.remove(originPath)
            else:
                shutil.copyfile(originPath, destinPath)
            print('Copying from\n{}\ninto\n{}'.format(originPath, destinPath))
#
if arguments['preprocFolderSubfolders']:
    itemsToMove = [
        itemName
        for itemName in os.listdir(originFolder)
        if os.path.isdir(os.path.join(originFolder, itemName))
        ]
    print('\nAbout to move:\n')
    print('\n'.join(itemsToMove))
    if arguments['requireKeypress']:
        _ = input('\n********\nPress any key to continue.')
    for itemName in os.listdir(originFolder):
        originPath = os.path.join(originFolder, itemName)
        destinPath = os.path.join(destinFolder, itemName)
        if os.path.isdir(originPath):
            if os.path.exists(destinPath):
                shutil.rmtree(destinPath)
            if arguments['moveItems']:
                shutil.move(originPath, destinPath)
            elif arguments['deleteItems']:
                shutil.rmtree(originPath)
            else:
                shutil.copytree(originPath, destinPath)
            print('Copying from\n{}\ninto\n{}'.format(originPath, destinPath))
#
if arguments['searchTerm'] is not None:
    itemsToMoveFullPath = glob.glob(
        os.path.join(
            originFolder, '**', '{}'.format(arguments['searchTerm'])),
        recursive=True)
    fileNamesToMove = [
        itemName.replace(originFolder, '')[1:]
        for itemName in itemsToMoveFullPath
        if (itemName not in itemsToExclude) and (os.path.isfile(itemName)) and arguments['filesIncluded']
        ]
    folderNamesToMove = [
        itemName.replace(originFolder, '')[1:]
        for itemName in itemsToMoveFullPath
        if (itemName not in itemsToExclude) and (os.path.isdir(itemName)) and arguments['foldersIncluded']
        ]
    if len(folderNamesToMove):
        print('\nAbout to move:\n')
        print('\n'.join(folderNamesToMove))
        if arguments['requireKeypress']:
            _ = input('\n********\nPress any key to continue.')
        for itemName in folderNamesToMove:
            originPath = os.path.join(originFolder, itemName)
            destinPath = os.path.join(destinFolder, itemName)
            if os.path.exists(destinPath):
                shutil.rmtree(destinPath)
            if arguments['moveItems']:
                shutil.move(originPath, destinPath)
            elif arguments['deleteItems']:
                shutil.rmtree(originPath)
            else:
                shutil.copytree(originPath, destinPath)
            print('Copying from\n{}\ninto\n{}'.format(originPath, destinPath))
    if len(fileNamesToMove):
        print('\nAbout to move:\n')
        print('\n'.join(fileNamesToMove))
        if arguments['requireKeypress']:
            _ = input('\n********\nPress any key to continue.')
        for itemName in fileNamesToMove:
            originPath = os.path.join(originFolder, itemName)
            destinPath = os.path.join(destinFolder, itemName)
            if not os.path.exists(os.path.dirname(destinPath)):
                os.makedirs(os.path.dirname(destinPath))
            if arguments['moveItems']:
                shutil.move(originPath, destinPath)
            elif arguments['deleteItems']:
                os.remove(originPath)
            else:
                shutil.copyfile(originPath, destinPath)
            print('Copying from\n{}\ninto\n{}'.format(originPath, destinPath))