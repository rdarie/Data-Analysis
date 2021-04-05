"""

Usage:
    preprocNS5.py [options]

Options:
    --exp=exp                       which experimental day to analyze
"""

import matplotlib, pdb, pickle, traceback
matplotlib.rcParams['agg.path.chunksize'] = 10000
#matplotlib.use('PS')   # generate interactive output by default
#matplotlib.use('TkAgg')   # generate interactive output by default
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dataAnalysis.preproc.sip as sip
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.motor_encoder_new as mea
import dataAnalysis.helperFunctions.helper_functions_new as hf
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from neo.io import NixIO, nixio_fr
import quantities as pq
import h5py
import os
import math as m
import seaborn as sns
from importlib import reload

import scipy.interpolate as intrp
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    experimentShorthand=arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

trialMetadata = {}

baseNameList = sorted([
    f.split('-Data')[0]
    for f in os.listdir(sipFolder)
    if ('-Data' in f)
    ])

for baseName in baseNameList:
    try:
        sipBasePath = os.path.join(
            sipFolder, baseName)
        outputPath = sipBasePath + '.nix'
        if os.path.exists(outputPath):
            os.remove(outputPath)
        print('Loading {}...'.format(sipBasePath))
        insTimeSeries = sip.loadSip(sipBasePath)
        dataCol = [
            i
            for i in insTimeSeries.columns
            if 'Sense' in i]
        insBlock = ns5.dataFrameToAnalogSignals(
            insTimeSeries, idxT='calculatedTimestamp',
            samplingRate=500*pq.Hz,
            probeName='ins_{}'.format(baseName),
            dataCol=dataCol, useColNames=True)
        writer = NixIO(filename=outputPath)
        writer.write_block(insBlock, use_obj_names=True)
        writer.close()
    except Exception:
        traceback.print_exc()
