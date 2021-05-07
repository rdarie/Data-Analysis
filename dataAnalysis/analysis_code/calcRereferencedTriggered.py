"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --profile                              print time and mem diagnostics? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: short]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --unitQuery=unitQuery                  how to restrict channels?
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --substituteOneChannel                 correct for rank defficiency by using one unsubtracted chan [default: False]
"""
#
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from namedQueries import namedQueries
import os
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import pickle
import math as m
import quantities as pq

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))

alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['verbose'] = arguments['verbose']
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    # transposeToColumns='feature', concatOn='columns',
    transposeToColumns='bin', concatOn='index',
    getMetaData=essentialMetadataFields + ['xCoords', 'yCoords'],
    decimate=1))

outputPath = os.path.join(
    alignSubFolder,
    blockBaseName + inputBlockSuffix + '_CAR_{}'.format(arguments['window']))
#
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
dataDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
#
# dataDF.columns = dataDF.columns.droplevel('lag')
dummySt = dataBlock.filter(
    objects=[ns5.SpikeTrain, ns5.SpikeTrainProxy])[0]
fs = float(dummySt.sampling_rate)

pdb.set_trace()
for trialIdx, trialData in dataDF.groupby(['segment', 'originalIndex', 't']):
    if arguments['substituteOneChannel']:
        saveFirstSrs = trialData.iloc[0, :].copy()
    referenceSignal = trialData.mean(axis='index')
    dataDF.loc[trialData.index, :] = trialData.sub(referenceSignal, axis='columns')
    if arguments['substituteOneChannel']:
        # implemented based on Milekovic, ..., Brochier 2015
        # check that it works!
        dataDF.loc[trialData.index[0], :] = saveFirstSrs
#
masterBlock = ns5.alignedAsigDFtoSpikeTrain(
    dataDF, dataBlock=dataBlock, matchSamplingRate=True)

if arguments['lazy']:
    dataReader.file.close()
masterBlock = ns5.purgeNixAnn(masterBlock)
if os.path.exists(outputPath + '.nix'):
    os.remove(outputPath + '.nix')
print('Writing {}.nix...'.format(outputPath))
if os.path.exists(outputPath):
    os.remove(outputPath)
writer = ns5.NixIO(
    filename=outputPath + '.nix',
    mode='ow')
writer.write_block(masterBlock, use_obj_names=True)
writer.close()
