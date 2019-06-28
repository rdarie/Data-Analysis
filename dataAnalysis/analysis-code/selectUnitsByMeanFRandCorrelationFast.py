"""
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: (pedalMovementCat==\'midPeak\')]
    --window=window                        process with short window? [default: long]
    --selectorName=selectorName            filename for resulting selector [default: minfrmaxcorr]
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --unitQuery=unitQuery                  how to restrict channels? [default: (chanName.str.endswith(\'fr#0\'))]
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
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
import gc
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['--processAll']:
    triggeredPath = os.path.join(
        scratchFolder,
        experimentName + '_trig_{}_{}.nix'.format(
            arguments['--inputBlockName'], arguments['--window']))
    selectorPath = os.path.join(
        scratchFolder,
        experimentName + '_' + arguments['--selectorName'] + '.pickle')
else:
    triggeredPath = os.path.join(
        scratchFolder,
        ns5FileName + '_trig_{}_{}.nix'.format(
            arguments['--inputBlockName'], arguments['--window']))
    selectorPath = os.path.join(
        scratchFolder,
        ns5FileName + '_' + arguments['--selectorName'] + '.pickle')
#
if arguments['--verbose']:
    prf.print_memory_usage('before load data')
#
dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
#
if arguments['--alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['--alignQuery']
    ])
#
#import warnings
#warnings.filterwarnings('error')
if arguments['--verbose']:
    prf.print_memory_usage('after load data')

alignedAsigsKWargs = dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    amplitudeColumn='amplitudeFuzzy',
    programColumn='programFuzzy',
    electrodeColumn='electrodeFuzzy',
    removeFuzzyName=False)
specificKWargs = dict(
    unitQuery=arguments['--unitQuery'], dataQuery=dataQuery,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=False, decimate=5,
    verbose=False, procFun=None)
#  turn into general pairwise analysis
from neo import Unit
from itertools import combinations
import pandas as pd
import numpy as np
from copy import copy

unitNames = None

masterSpikeMat = ns5.alignedAsigsToDF(
    dataBlock, unitNames,
    **alignedAsigsKWargs, **specificKWargs)

unitNames = masterSpikeMat.columns.to_list()
correlationDF = masterSpikeMat.corr()
for n in correlationDF.index:
    correlationDF.loc[n, n] = 0
meanFRDF = masterSpikeMat.mean()

if arguments['--verbose']:
    prf.print_memory_usage('just loaded frs')
dataReader.file.close()
#  free up memory
del dataBlock
gc.collect()
#  based on https://seaborn.pydata.org/examples/many_pairwise_correlations.html
import matplotlib.pyplot as plt
import seaborn as sns
mask = np.zeros_like(correlationDF, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(os.path.join(figureFolder, 'unit_correlation_{}.pdf'.format(arguments['--selectorName']))) as pdf:
    sns.heatmap(
        correlationDF.to_numpy(), mask=mask, cmap=cmap, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5})
    pdf.savefig()
    plt.close()


def selFun(
        meanDF, corrDF, meanThresh=5,
        corrThresh=0.85):
    unitMask = ((meanDF > meanThresh) & (corrDF.max() < corrThresh))
    return unitMask[unitMask].index.to_list()


thisCorrThresh = 0.85

selectorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(selectorPath),
    'name': arguments['--selectorName'],
    'inputBlockName': arguments['--inputBlockName'],
    'inputFeatures': correlationDF.columns.to_list(),
    'outputFeatures': selFun(meanFRDF, correlationDF, corrThresh=thisCorrThresh),
    'dataQuery': dataQuery,
    'alignedAsigsKWargs': alignedAsigsKWargs,
    'selFun': selFun,
    'data': {
        'meanFRDF': meanFRDF,
        'correlationDF': correlationDF
    },
    'selFunInputs': {'meanThresh': 5, 'corrThresh': thisCorrThresh}
    }

with open(selectorPath, 'wb') as f:
    pickle.dump(
        selectorMetadata, f)
