"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --window=window                        process with short window? [default: long]
    --inputBlockName=inputBlockName        filename for inputs [default: raster]
    --alignQuery=alignQuery                choose a data subset? [default: midPeak]
    --selector=selector                    filename if using a unit selector
    --unitQuery=unitQuery                  how to restrict channels?
"""
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.neuralTrajInterface.neural_traj_interface as nti
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import os
import pandas as pd
import numpy as np
import scipy.io as sio
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import dill as pickle
import subprocess

from currentExperiment import parseAnalysisOptions
from docopt import docopt
from namedQueries import namedQueries
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['verbose'] = arguments['verbose']

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    removeFuzzyName=False, getMetaData=False,
    procFun=lambda wfdf: wfdf > 0,
    transposeToColumns='bin', concatOn='index'))

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName

intermediatePath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}_for_gpfa_{}.mat'.format(
        arguments['window'], arguments['alignQuery']))
modelName = '{}_{}_{}'.format(
    prefix,
    arguments['window'],
    arguments['alignQuery'])

if not os.path.exists(intermediatePath):
    triggeredPath = os.path.join(
        scratchFolder,
        prefix + '_raster_{}.nix'.format(
            arguments['window']))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath,
        lazy=arguments['lazy'])
    alignedRastersDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    nti.saveRasterForNeuralTraj(alignedRastersDF, intermediatePath)
    if arguments['lazy']:
        dataReader.file.close()
    
# dataPath, xDim, segLength, binWidth, kernSD, runIdx, baseDir
gpfaArg = ', '.join([
    '\'' + intermediatePath + '\'',
    '{}'.format(gpfaOpts['xDim']),
    '{}'.format(gpfaOpts['segLength']),
    '{}'.format(gpfaOpts['binWidth']),
    '{}'.format(gpfaOpts['kernSD']),
    '\'{}\''.format(modelName),
    '\'' + scratchFolder + '\'',
    ])
execStr = 'matlab -r \"calculate_gpfa({}); exit\"'.format(gpfaArg)
print(execStr)
result = subprocess.run([execStr], shell=True)