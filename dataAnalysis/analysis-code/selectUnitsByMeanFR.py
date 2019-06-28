"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: (pedalMovementCat==\'midPeak\')]
    --window=window                        process with short window? [default: long]
    --selectorName=selectorName            filename for resulting selector [default: minfr]
    --inputBlockName=inputBlockName        filename for inputs [default: raster]
    --unitQuery=unitQuery                  how to restrict channels? [default: (chanName.str.endswith(\'raster#0\'))]
"""

import os
import dataAnalysis.helperFunctions.profiling as prf
#  import numpy as np
#  import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  import neo
import dill as pickle
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import gc
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    ns5FileName + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
selectorPath = os.path.join(
    scratchFolder,
    ns5FileName + '_' + arguments['selectorName'] + '.pickle')

if arguments['verbose']:
    prf.print_memory_usage('before load data')

dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

alignedAsigsKWargs = dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    removeFuzzyName=False)

if arguments['alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['alignQuery']
    ])

unitNames = None


def procFun(wfDF):
    return wfDF.mean(axis=1).to_frame()


alignedRastersDF = ns5.alignedAsigsToDF(
    dataBlock, unitNames,
    unitQuery=arguments['unitQuery'], dataQuery=dataQuery,
    transposeToColumns='feature', concatOn='columns',
    verbose=True, procFun=procFun,
    **alignedAsigsKWargs)
prf.print_memory_usage('just loaded rasters')
dataReader.file.close()
#  free up memory
del dataBlock
gc.collect()


def selFun(rastersDF, meanFRThresh=5):
    assert rastersDF.columns.name == 'feature'
    unitMask = rastersDF.mean(axis=0) > meanFRThresh
    return unitMask[unitMask].index.to_list()


selectorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(selectorPath),
    'name': arguments['selectorName'],
    'inputBlockName': arguments['inputBlockName'],
    'inputFeatures': alignedRastersDF.columns.to_list(),
    'outputFeatures': selFun(alignedRastersDF),
    'dataQuery': dataQuery,
    'alignedAsigsKWargs': alignedAsigsKWargs,
    'selFun': selFun,
    'procFun': procFun,
    'selFunInputs': {'meanFRThresh': 5}
    }

with open(selectorPath, 'wb') as f:
    pickle.dump(
        selectorMetadata, f)
