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
    --window=window                        process with short window? [default: long]
"""
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default

import matplotlib.pyplot as plt
import seaborn as sns
import dataAnalysis.helperFunctions.profiling as prf
import os
#  import seaborn as sns
#  import numpy as np
#  import quantities as pq
import pandas as pd
import numpy as np
import scipy.io as sio
import pdb
import h5py
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

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")
verbose = arguments['verbose']

trainSetNames = ['midPeak', 'midPeakNoStim']
modelNames = ['fa', 'pca', 'ppca', 'gpfa']
modelResults = {k: {} for k in trainSetNames}

for trainSetName in trainSetNames:
    modelPath = os.path.join(
        scratchFolder, 'gpfa_results',
        '{}_{}'.format(prefix, trainSetName),
        'predErrorVsDim.mat'
        )
    with h5py.File(modelPath, 'r') as f:
        nEntries = f['D']['name'].shape[1]
        fields = ['method', 'sse', 'r2', 'xDim']
        results = {f: [] for f in fields}

        for fieldName in fields:
            for idx in range(nEntries):
                ref = h5py.h5r.get_name(f['D'][fieldName][0, idx], f.id)
                dataArray = f[ref].value
                if fieldName == 'method':
                    letters = [chr(i) for i in dataArray]
                    results[fieldName].append(''.join(letters))
                elif fieldName in ['sse', 'r2', 'xDim']:
                    results[fieldName].append(dataArray.flatten()[0])
        resultsDF = pd.DataFrame(results)
    modelResults[trainSetName] = resultsDF

modelResultsDF = pd.concat(modelResults, names=['trainSet']).reset_index()
ax = sns.lineplot(x='xDim', y='sse', hue='method', style='trainSet', data=modelResultsDF)
plt.savefig(os.path.join(figureFolder, 'gpfa_reconstruction_error.pdf'))
plt.close()
##)
# stdout=subprocess.PIPE
#print(result.stdout)
# plt.spy(alignedRasterList[2]); plt.show()