"""  10b: Calculate Firing Rates
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
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
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']),
    arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

chanNames = None
#  chanNames = [
#      'elec75#0_raster', 'elec75#1_raster', 'elec83#0_raster',
#      'elec78#0_raster', 'elec78#1_raster']

def aggregateFun(
        DF, fs=None, nSamp=None):
    # if a bin is nonzero, it encodes the
    # firing rate of 1 spike/bin
    tSpan = nSamp / fs  # how wide is the window?
    spikeCount = np.sqrt(
        np.sum(DF / fs))
    return spikeCount / tSpan

if arguments['--processAll']:
    masterBlock = preproc.calcFR(
        experimentBinnedSpikePath,
        experimentDataPath,
        suffix='fr_sqrt',
        aggregateFun=aggregateFun,
        chanNames=chanNames,
        rasterOpts=rasterOpts)
else:
    masterBlock = preproc.calcFR(
        binnedSpikePath,
        analysisDataPath,
        suffix='fr_sqrt',
        aggregateFun=aggregateFun,
        chanNames=chanNames,
        rasterOpts=rasterOpts)

allSegs = list(range(len(masterBlock.segments)))

if arguments['--processAll']:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeSpikes=False, writeEvents=False,
        fileName=experimentName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
else:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeSpikes=False, writeEvents=False,
        fileName=ns5FileName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )