"""06c: Preprocess the open ephys recording File

Usage:
    preprocNS5.py [options]

Options:
    --exp=exp                       which experimental day to analyze
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
    experimentShorthand=arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
trialList = sorted([
    f
    for f in os.listdir(oeFolder)
    if (not os.path.isfile(os.path.join(oeFolder, f))) and ('Trial001' in f)
    ])
invEmgTrialLookup = {v: k for k, v in openEphysBaseNames.items()}
#
for folderPath in trialList:
    try:
        print('Loading {}...'.format(folderPath))
        emgBaseName = os.path.basename(folderPath)
        trialIdx = invEmgTrialLookup[emgBaseName]
        ppOE.preprocOpenEphysFolder(
            os.path.join(oeFolder, folderPath),
            chanNames=openEphysChanNames, plotting=False,
            ignoreSegments=openEphysIgnoreSegments[trialIdx],
            makeFiltered=True,
            filterOpts=openEphysFilterOpts)
        emgDataPath = os.path.join(
            oeFolder, folderPath, emgBaseName + '_filtered.nix'
        )
        outputPath = os.path.join(
            scratchFolder, 'Trial{:0>3}_oe.nix'.format(trialIdx))
        shutil.copyfile(emgDataPath, outputPath)
        emgRawDataPath = os.path.join(
            oeFolder, folderPath, emgBaseName + '.nix'
        )
        outputPath = os.path.join(
            scratchFolder, 'Trial{:0>3}_raw_oe.nix'.format(trialIdx))
        shutil.copyfile(emgRawDataPath, outputPath)
    except Exception:
        traceback.print_exc()
