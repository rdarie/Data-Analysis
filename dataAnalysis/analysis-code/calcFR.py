"""  10b: Calculate Firing Rates
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx               which trial to analyze [default: 1]
    --exp=exp                         which experimental day to analyze
    --analysisName=analysisName       append a name to the resulting blocks? [default: default]
    --processAll                      process entire experimental day? [default: False]
"""

import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
from copy import copy
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from joblib import dump, load
from importlib import reload
import quantities as pq

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)

if overrideChanNames is not None:
    chanNames = [i + '_raster' for i in overrideChanNames]
else:
    chanNames = None

# rasterOpts.update({
#     'binWidth': 5e-3,
#     'smoothKernelWidth': None})
rasterOpts.update({
    'binWidth': 50e-3,
    'smoothKernelWidth': None})

experimentBinnedSpikePath = experimentBinnedSpikePath.format(arguments['analysisName'])
experimentDataPath = experimentDataPath.format(arguments['analysisName'])
binnedSpikePath = binnedSpikePath.format(arguments['analysisName'])
analysisDataPath = analysisDataPath.format(arguments['analysisName'])

if arguments['processAll']:
    masterBlock = preproc.calcFR(
        experimentBinnedSpikePath,
        experimentDataPath,
        suffix='fr',
        aggregateFun=None,
        chanNames=chanNames,
        rasterOpts=rasterOpts)
else:
    masterBlock = preproc.calcFR(
        binnedSpikePath,
        analysisDataPath,
        suffix='fr',
        aggregateFun=None,
        chanNames=chanNames,
        rasterOpts=rasterOpts)

allSegs = list(range(len(masterBlock.segments)))
if arguments['processAll']:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeSpikes=False, writeEvents=False,
        fileName=experimentName + '_analyze',
        folderPath=analysisSubFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
else:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeSpikes=False, writeEvents=False,
        fileName=ns5FileName + '_analyze',
        folderPath=analysisSubFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )