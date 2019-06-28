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
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['--inputBlockName'], arguments['--window']))
selectorPath = os.path.join(
    scratchFolder,
    prefix + '_' + arguments['--selectorName'] + '.pickle')
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

alignedAsigsKWargs = dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False)
specificKWargs = dict(
    unitQuery=arguments['--unitQuery'], dataQuery=dataQuery,
    transposeToColumns='bin', concatOn='index',
    getMetaData=False, decimate=5,
    verbose=False, procFun=None)
#  turn into general pariwise analysis
from neo import Unit
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import copy

unitNames = None
if unitNames is None:
        unitNames = ns5.listChanNames(
            dataBlock, arguments['--unitQuery'], objType=Unit)
remainingUnits = copy(unitNames)

correlationDF = pd.DataFrame(
    0, index=unitNames, columns=unitNames, dtype='float32')
meanFRDF = pd.Series(0, index=unitNames, dtype='float32')
for idxOuter, firstUnit in enumerate(unitNames):
    remainingUnits.remove(firstUnit)
    if arguments['--verbose']:
        prf.print_memory_usage(' firstUnit: {}'.format(firstUnit))
        print('{} secondary units to analyze'.format(len(remainingUnits)))
    firstDF = ns5.alignedAsigsToDF(
        dataBlock, [firstUnit],
        **specificKWargs,
        **alignedAsigsKWargs)
    meanFRDF[firstUnit] = firstDF.mean().mean()
    # print('firstDF uses {:.1f} MB'.format(firstDF.memory_usage(deep=True).sum() / 2**10))
    for idxInner, secondUnit in enumerate(remainingUnits):
        if arguments['--verbose']:
            prf.print_memory_usage('secondUnit: {}'.format(secondUnit))
        secondDF = ns5.alignedAsigsToDF(
            dataBlock, [secondUnit],
            **specificKWargs,
            **alignedAsigsKWargs)
        # print('secondDF uses {:.1f} MB'.format(secondDF.memory_usage(deep=True).sum() / 2**10))
        correlationDF.loc[firstUnit, secondUnit], _ = pearsonr(
            firstDF.to_numpy().flatten(), secondDF.to_numpy().flatten())

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
