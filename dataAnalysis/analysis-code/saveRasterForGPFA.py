"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --exp=exp                              which experimental day to analyze
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                choose a subset of the data? [default: (pedalMovementCat==\'midPeak\')]
    --alignSuffix=alignSuffix              what name to append in order to identify the align query? [default: midPeak]
    --selector=selector                    filename if using a unit selector
    --window=window                        process with short window? [default: long]
    --unitQuery=unitQuery                  how to restrict channels? [default: (chanName.str.endswith(\'raster#0\'))]
"""
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.profiling as prf
import os
#  import seaborn as sns
#  import numpy as np
#  import quantities as pq
import pandas as pd
import numpy as np
import scipy.io as sio
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  from neo.io.proxyobjects import (
#      AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import joblib as jb
import dill as pickle
#  import gc
import subprocess

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

verbose = arguments['verbose']

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}.nix'.format(
        arguments['window']))

intermediatePath = triggeredPath.replace(
    '.nix',
    '_for_gpfa_{}.mat'.format(arguments['alignSuffix']))

if verbose:
    prf.print_memory_usage('before load data')
dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

if arguments['alignQuery'] is None:
    dataQuery = None
elif len(arguments['alignQuery']) == 0:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['alignQuery']
    ])

if verbose:
    prf.print_memory_usage('before load firing rates')

if arguments['selector'] is not None:
    with open(
        os.path.join(
            scratchFolder,
            arguments['selector'] + '.pickle'),
            'rb') as f:
        selectorMetadata = pickle.load(f)
    unitNames = selectorMetadata['outputFeatures']
    alignedAsigsKWargs = selectorMetadata['alignedAsigsKWargs']
else:
    unitNames = None
    alignedAsigsKWargs = dict(
        duplicateControlsByProgram=False,
        makeControlProgram=True,
        removeFuzzyName=False)

if miniRCTrial:
    alignedAsigsKWargs.update(dict(
        amplitudeColumn='amplitude',
        programColumn='program',
        electrodeColumn='electrode'))

alignedRastersDF = ns5.alignedAsigsToDF(
    dataBlock, unitNames,
    unitQuery=arguments['unitQuery'], dataQuery=dataQuery,
    procFun=lambda wfdf: wfdf > 0,
    transposeToColumns='bin', concatOn='index',
    **alignedAsigsKWargs, verbose=True)

#  keepMetaCols = ['segment', 'originalIndex', 'feature']
#  dropMetaCols = np.setdiff1d(alignedRastersDF.index.names, keepMetaCols).tolist()
#  alignedRastersDF.index = alignedRastersDF.index.droplevel(dropMetaCols)

alignedRasterList = [
    g.to_numpy(dtype='uint8')
    for n, g in alignedRastersDF.groupby(['segment', 'originalIndex'])]
trialIDs = [
    np.atleast_2d(i).astype('uint16')
    for i in range(len(alignedRasterList))]
structDType = np.dtype([('trialId', 'O'), ('spikes', 'O')])

dat = np.array(list(zip(trialIDs, alignedRasterList)), dtype=structDType)
sio.savemat(intermediatePath, {'dat': dat})