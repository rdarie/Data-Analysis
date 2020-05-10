"""06c: Preprocess the open ephys recording File

Usage:
    preprocNS5.py [options]

Options:
    --exp=exp                       which experimental day to analyze
    --blockIdx=blockIdx             which trial to analyze [default: 1]
    --plotting                      plot diagnostic figures? [default: False]
    --loadMat                       choose loader [default: False]
"""
import dataAnalysis.preproc.open_ephys as ppOE
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pdb, traceback
import quantities as pq
import os
import shutil
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
# #)
# trialList = sorted([
#     f
#     for f in os.listdir(oeFolder)
#     if (not os.path.isfile(os.path.join(oeFolder, f))) # and ('Block001' in f)
#     ])
invEmgBlockLookup = {
    v: k
    for k, v in openEphysBaseNames.items()}
folderPath = openEphysBaseNames[blockIdx]
#
# for blockIdx, folderPath in openEphysBaseNames.items():
print('Loading {}...'.format(folderPath))
emgBaseName = os.path.basename(folderPath)
#blockIdx = invEmgBlockLookup[emgBaseName]
ppOE.preprocOpenEphysFolder(
    os.path.join(oeFolder, folderPath),
    chanNames=openEphysChanNames, plotting=arguments['plotting'],
    ignoreSegments=openEphysIgnoreSegments[blockIdx],
    makeFiltered=True, loadMat=arguments['loadMat'],
    filterOpts=EMGFilterOpts)
emgDataPath = os.path.join(
    oeFolder, folderPath, emgBaseName + '_filtered.nix'
)
outputPath = os.path.join(
    scratchFolder, 'Block{:0>3}_oe.nix'.format(blockIdx))
shutil.copyfile(emgDataPath, outputPath)
emgRawDataPath = os.path.join(
    oeFolder, folderPath, emgBaseName + '.nix'
)
outputPath = os.path.join(
    scratchFolder, 'Block{:0>3}_raw_oe.nix'.format(blockIdx))
shutil.copyfile(emgRawDataPath, outputPath)