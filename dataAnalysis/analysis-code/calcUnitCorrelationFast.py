"""
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: (pedalMovementCat==\'midPeak\')]
    --window=window                        process with short window? [default: long]
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --resultName=resultName                filename for result [default: corr]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: (chanName.str.endswith(\'fr#0\'))]
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
    ns5FileName + '_{}_{}.nix'.format(
        arguments['--inputBlockName'], arguments['--window']))
resultPath = os.path.join(
    scratchFolder,
    ns5FileName + '_' + arguments['--resultName'] + '.pickle')
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
if arguments['--verbose']:
    prf.print_memory_usage('after load data')

if arguments['--alignQuery'] is None:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['--alignQuery']
    ])

if miniRCTrial:
    alignedAsigsKWargs = dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False,
        removeFuzzyName=False,
        amplitudeColumn='amplitude',
        programColumn='program',
        electrodeColumn='electrode')
else:
    alignedAsigsKWargs = dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False,
        removeFuzzyName=False,
        amplitudeColumn='amplitudeFuzzy',
        programColumn='programFuzzy',
        electrodeColumn='electrodeFuzzy')

specificKWargs = dict(
    unitQuery=arguments['--unitQuery'], dataQuery=dataQuery,
    transposeToColumns='feature', concatOn='columns',
    getMetaData=False, decimate=5,
    verbose=False, procFun=None)

unitNames = None

if arguments['--verbose']:
    prf.print_memory_usage('about to load frs')
masterSpikeMat = ns5.alignedAsigsToDF(
    dataBlock, unitNames,
    **alignedAsigsKWargs, **specificKWargs)
if arguments['--verbose']:
    prf.print_memory_usage('just loaded frs')

correlationDF = masterSpikeMat.corr()
for n in correlationDF.index:
    correlationDF.loc[n, n] = 0

dataReader.file.close()
#  free up memory
del dataBlock
gc.collect()
#  based on https://seaborn.pydata.org/examples/many_pairwise_correlations.html
#  TODO turn into general pairwise analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

mask = np.zeros_like(correlationDF, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
with PdfPages(os.path.join(figureFolder, 'unit_correlation.pdf')) as pdf:
    sns.heatmap(
        correlationDF.to_numpy(), mask=mask, cmap=cmap, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5})
    pdf.savefig()
    plt.close()
