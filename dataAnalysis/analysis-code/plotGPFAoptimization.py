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
# matplotlib.use('PS')   # generate postscript output by default
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

from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['--processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName

sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")
verbose = arguments['--verbose']

modelPath = os.path.join(
    scratchFolder, 'gpfa_results',
    '{}_{}'.format(prefix, arguments['--alignSuffix']),
    'predErrorVsDim.mat'
    )
f = h5py.File(modelPath, 'r')
nMethods = f['method']['name'].shape[0]
fields = [i for i in f['method'].keys()]
results = {f: [] for f in fields}

for fieldName in fields:
    for idx in range(nMethods):
        ref = h5py.h5r.get_name(f['method'][fieldName][idx, 0], f.id)
        dataArray = f[ref].value
        if fieldName == 'name':
            letters = [chr(i) for i in dataArray]
            results[fieldName].append(''.join(letters))
        else:
            results[fieldName].append(dataArray.flatten())
resultsDF = pd.DataFrame(results)
resultsByName = {}
for idx in range(nMethods):
    name = results['name'][idx]
    values = {}
    for k, v in results.items():
        if k not in ['name', 'numTrials']:
            values.update({k: v[idx]})
    resultsByName.update({name: pd.DataFrame(values)})

data = pd.concat(resultsByName, names=['method']).reset_index()
ax = sns.lineplot(x='xDim', y='sse', hue='method', data=data)
plt.show()
# stdout=subprocess.PIPE
#print(result.stdout)
# plt.spy(alignedRasterList[2]); plt.show()