"""07: Combine SIMI and NSP Data
Usage:
    synchronizeSIMItoNSP [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
"""
import matplotlib, pdb, traceback
matplotlib.use('Qt5Agg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
#  matplotlib.use('PS')   # noninteract output
from matplotlib import pyplot as plt
import dill as pickle
from scipy import stats
from importlib import reload
from datetime import datetime as dt
import peakutils
import numpy as np
import pandas as pd
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.motor_encoder_new as mea
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.estimateElectrodeImpedances as eti
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.mdt_constants as mdt_constants

import h5py
import os, glob
import math as m
import seaborn as sns
import scipy.interpolate as intrp
import quantities as pq
import json
import rcsanalysis.packetizer as rcsa
import rcsanalysis.packet_func as rcsa_helpers
import datetime

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain)
import neo
import elephant.pandas_bridge as elphpdb

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

#  Load NSP Data
############################################################
startTime_s = None
dataLength_s = None
print('Loading NSP Block...')
ainName = trialFilesFrom['utah']['eventInfo']['inputIDs']['simiTrigs']

try:
    channelData, _ = ns5.getNIXData(
        fileName=ns5FileName,
        folderPath=scratchFolder,
        elecIds=[ainName], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)
except Exception:
    traceback.print_exc()
#
#  load SIMI Data
############################################################
kinematicsPathList = glob.glob(
    os.path.join(simiFolder, '*{}*'.format(ns5FileName)))
assert len(kinematicsPathList) == 1
kinematicsPath = kinematicsPathList[0]
# pdb.set_trace()
calcAngles = [
    ['C_Right', 'GT_Right', 'K_Right'],
    ['GT_Right', 'K_Right', 'M_Right'],
    ['K_Right', 'M_Right', 'MT_Right'],
    ['M_Right', 'MT_Right', 'T_Right'],
    ]

filterOpts = dict(
    lowPass=10, lowOrder=2,
    highPass=None, highOrder=2,
    notch=False, filtFun='butter',
    columns=None)
kin = hf.getKinematics(
    kinematicsPath,
    trigTimes=None,
    trigSeries=channelData['data'][ainName],
    nspTime=channelData['data']['t'],
    thres=None,
    selectHeaders=None, selectTime=None,
    flip=None, reIndex=None, calcAngles=calcAngles, filterOpts=filterOpts)

############################################################
kin.columns = ['_'.join(i) for i in kin.columns]
pdb.set_trace()
addingToNix = True
if addingToNix:
    kinInterp = hf.interpolateDF(
        kin.reset_index(), channelData['t'],
        kind='linear', fill_value=(0, 0),
        x='t', columns=None)
    tdBlock = ns5.dataFrameToAnalogSignals(
        kinInterp,
        idxT='t',
        probeName='kinematics', samplingRate=3e4*pq.Hz,
        dataCol=kin.columns,
        forceColNames=kin.columns)
    ns5.addBlockToNIX(
        tdBlock, neoSegIdx=[0],
        writeAsigs=True, writeSpikes=False, writeEvents=False,
        fileName=ns5FileName,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )