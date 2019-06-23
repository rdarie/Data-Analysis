"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --exp=exp                              which experimental day to analyze
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --window=window                        process with short window? [default: short]
    --chanQuery=chanQuery                  how to restrict channels?
"""
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.profiling as prf
import os
#  import seaborn as sns
#  import numpy as np
#  import quantities as pq
import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  from neo.io.proxyobjects import (
#      AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import joblib as jb
import pickle
#  import gc

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

verbose = arguments['--verbose']

if arguments['--processAll']:
    triggeredPath = os.path.join(
        scratchFolder,
        experimentName + '_trig_raster_{}.nix'.format(
            arguments['--window']))
    outputPath = os.path.join(
        scratchFolder,
        experimentName + '_trig_raster_{}'.format(
            arguments['--window']))
else:
    triggeredPath = os.path.join(
        scratchFolder,
        ns5FileName + '_trig_raster_{}.nix'.format(
            arguments['--window']))
    outputPath = os.path.join(
        scratchFolder,
        ns5FileName + '_trig_raster_{}'.format(
            arguments['--window']))

if verbose:
    prf.print_memory_usage('before load data')
dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

if arguments['--alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['--alignQuery']
    ])

alignedAsigsKWargs = dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    amplitudeColumn='amplitudeFuzzy',
    programColumn='programFuzzy',
    electrodeColumn='electrodeFuzzy',
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=False)

if verbose:
    prf.print_memory_usage('before load firing rates')
unitNames = None
alignedAsigsDF = ns5.alignedAsigsToDF(
    dataBlock, unitNames, dataQuery,
    **alignedAsigsKWargs, verbose=True)
if verbose:
    prf.print_memory_usage('after load firing rates')

pdb.set_trace()