"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                    which experimental day to analyze
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --processAll                                 process entire experimental day? [default: False]
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --arrayName=arrayName                        name of electrode array? (for map file) [default: utah]
    --inputBlockSuffix=inputBlockSuffix          which block to pull
    --inputBlockPrefix=inputBlockPrefix          which block to pull [default: Block]
    --lazy                                       load from raw, or regular? [default: False]
    --chanQuery=chanQuery                        how to restrict channels? [default: raster]
    --verbose                                    print diagnostics? [default: False]
    --plotting                                   plot results?
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import seaborn as sns
from docopt import docopt

import pdb, traceback
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
# import pingouin as pg
import pandas as pd
import numpy as np
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from copy import deepcopy
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr

sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)

calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
#
dataPath = os.path.join(
    analysisSubFolder,
    blockBaseName + '{}.nix'.format(inputBlockSuffix))

arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
    namedQueries, scratchFolder, **arguments)

if __name__ == "__main__":
    print('loading {}'.format(dataPath))
    #
    dataReader, dataBlock = ns5.blockFromPath(
        dataPath, lazy=arguments['lazy'])
    if arguments['chanNames'] is None:
        arguments['chanNames'] = ns5.listChanNames(
            dataBlock, arguments['chanQuery'], objType=ns5.ChannelIndex, condition='hasAsigs')
    chanNames = arguments['chanNames']
    dummyAsigLike = None
    for asigLike in dataBlock.filter(objects=[ns5.AnalogSignal, ns5.AnalogSignalProxy]):
        if asigLike.channel_index.name in chanNames:
            dummyAsigLike = asigLike
    assert dummyAsigLike is not None
    if not (('xcoords' in dummyAsigLike.annotations) and ('ycoords' in dummyAsigLike.annotations)):
        electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
        mapExt = electrodeMapPath.split('.')[-1]
        if mapExt == 'cmp':
            mapDF = prb_meta.cmpToDF(electrodeMapPath)
        elif mapExt == 'map':
            mapDF = prb_meta.mapToDF(electrodeMapPath)
    pdb.set_trace()