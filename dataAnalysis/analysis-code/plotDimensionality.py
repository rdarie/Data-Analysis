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
matplotlib.use('qt5agg')   # generate postscript output by default
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

regularNames = ['pca_midPeakStim_full', 'pca_midPeakNoStim_full']
miniRCNames = []
estimatorNames = (
    [experimentName + '_' + eName for eName in regularNames] +
    ['Trial005_' + eName for eName in miniRCNames]
)
explainedVar = {}
for eName in estimatorNames:
    estimatorPath = os.path.join(
        scratchFolder,
        eName + '.joblib')
    metadataPath = estimatorPath.replace('.joblib', '_meta.pickle')
    with open(metadataPath, 'rb') as f:
        estimatorMetadata = pickle.load(f)
    estimator = jb.load(
        os.path.join(estimatorPath))
    explainedVar.update({eName: estimator.explained_variance_ratio_})

modelResultsDF = pd.DataFrame(explainedVar).cumsum()
modelResultsDF.index.name = 'pc'
modelResultsDF.columns.name = 'model'
modelResultsDF = modelResultsDF.stack().to_frame(name='explained variance').reset_index()
ax = sns.lineplot(x='pc', y='explained variance', hue='model', data=modelResultsDF)
figPath = os.path.join(figureFolder, 'pca_explained_variance.pdf')
plt.savefig(figPath)
plt.show()
plt.close()
'''

selectorNames = ['minfrmaxcorrstim', 'minfrmaxcorrnostim']
corrDFs = {}
meanFRs = {}
for sName in selectorNames:
    selectorPath = os.path.join(
        scratchFolder,
        prefix + '_' + sName + '.pickle')
    with open(selectorPath, 'rb') as f:
        selectorMetadata = pickle.load(f)
    meanFRDF = selectorMetadata['data']['meanFRDF']
    corrDFs.update({sName: selectorMetadata['data']['correlationDF']})
    meanFRs.update({sName: meanFRDF})

unitsTooLow = meanFRs['minfrmaxcorrstim'].index[ meanFRs['minfrmaxcorrstim'] < 5]

plotMatrices = {}
plotMatrices['stim'] = corrDFs['minfrmaxcorrstim'].drop(columns=unitsTooLow, index=unitsTooLow)
plotMatrices['nostim'] = corrDFs['minfrmaxcorrnostim'].drop(columns=unitsTooLow, index=unitsTooLow)
plotMatrices['delta'] = plotMatrices['stim'] - plotMatrices['nostim']
for k, v in plotMatrices.items():
    v[v == 1] = 0
    plotMatrices[k] = v
mask = np.zeros_like(plotMatrices['stim'], dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
from matplotlib.backends.backend_pdf import PdfPages
for k, v in plotMatrices.items():
    if k == 'delta':
        vmax = None
    else:
        vmax = 0.5
    with PdfPages(os.path.join(figureFolder, 'unit_correlation_{}.pdf'.format(k))) as pdf:
        sns.heatmap(
            v.to_numpy(), mask=mask, cmap=cmap, vmax=vmax, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
        pdf.savefig()
        plt.close()

'''