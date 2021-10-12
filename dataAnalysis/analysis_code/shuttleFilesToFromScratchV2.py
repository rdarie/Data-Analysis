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
    --printFolderSize                                move items, as opposed to copy them [default: False]
    --moveItems                                      move items, as opposed to copy them
    --fromScratchToData                              process entire experimental day? [default: False]
    --fromDataToScratch                              process entire experimental day? [default: False]
    --filesIncluded                                  process entire experimental day? [default: False]
    --foldersIncluded                                process entire experimental day? [default: False]
    --requireKeypress                                process entire experimental day? [default: False]
    
"""

#  load options
import pdb, traceback
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import shutil, os, glob
import subprocess
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

if arguments['printFolderSize']:
    from pathlib import Path

    def check_out_path(target_path, level=0):
        """"
        This function recursively prints all contents of a pathlib.Path object
        """
        def print_indented(folder, level):
            print('\t' * level + folder)
        print_indented(target_path.name, level)
        ### du -a -d 1 --time -h ./ | sort -h
        commandList = [
            'du -a -d 1 --time -h "{}" | sort -h'.format(target_path),
        ]
        p = subprocess.Popen(commandList, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for l in p.stdout.readlines():
            print_indented(l.decode("utf-8").strip(), level)
        for file in target_path.iterdir():
            if file.is_dir():
                check_out_path(file, level + 1)
        ###
        # hasSubDirs = any([child.is_dir() for child in target_path.iterdir()])
        # if hasSubDirs:
        #     for file in target_path.iterdir():
        #         if file.is_dir():
        #             check_out_path(file, level + 1)
        # else:
        #     commandList = [
        #         'find {} -type f -exec du -a {} + | sort -n -r | less'.format(target_path)
        #         # 'du -a --time -h "{}" | sort -h'.format(target_path),
        #         ]
        #     p = subprocess.Popen(commandList, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #     for l in p.stdout.readlines():
        #         print_indented(l.decode("utf-8").strip(), level)
    my_path = Path(scratchFolder)
    check_out_path(my_path)

####
# find and remove
'''
-o means "or"
find . \( -wholename '*foldername*.h5' -o -name '*.jpg' \) -print
-exec rm -f {} \;
find . \( -wholename '*estimators*' \) -print
find . \( -wholename '*estimators*' \) -exec rm -f {} \;
'''
#######################
#  Global moves
#######################
if arguments['fromDataToScratch']:
    originFolder, destinFolder = processedFolder, scratchFolder
elif arguments['fromScratchToData']:
    originFolder, destinFolder = scratchFolder, processedFolder

if arguments['excludeSearchTerm'] is not None:
    itemsToExclude = glob.glob(
        os.path.join(
            originFolder, '**', '{}'.format(arguments['excludeSearchTerm'])),
        recursive=True)
else:
    itemsToExclude = []
#
if arguments['searchTerm'] is not None:
    print('Moving items from {} to {}'.format(originFolder, destinFolder))
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
            else:
                shutil.copyfile(originPath, destinPath)
            print('Copying from\n{}\ninto\n{}'.format(originPath, destinPath))