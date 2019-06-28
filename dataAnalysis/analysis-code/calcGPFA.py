"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --exp=exp                              which experimental day to analyze
    --verbose                              print diagnostics? [default: False]
    --inputDataName=inputDataName          what name to append in order to identify the align query? [default: midPeak]
    --window=window                        process with short window? [default: long]
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

intermediatePath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}_for_gpfa_{}.mat'.format(
        arguments['window'], arguments['inputDataName']))

modelName = '{}_{}'.format(prefix, arguments['inputDataName'])
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