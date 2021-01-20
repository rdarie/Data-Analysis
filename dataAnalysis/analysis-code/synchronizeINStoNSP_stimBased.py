"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --blockIdx=blockIdx                                   which trial to analyze
    --exp=exp                                             which experimental day to analyze
    --inputNSPBlockSuffix=inputNSPBlockSuffix             append a name to the input block?
    --inputINSBlockSuffix=inputINSBlockSuffix             append a name to the input block?
    --lazy                                                whether to fully load data from blocks [default: True]
    --addToBlockSuffix=addToBlockSuffix                   whether to also add stim info to the high pass filtered NSP blocks
    --addToNIX                                            whether to interpolate and add the ins data to the nix file [default: False]
    --curateManually                                      whether to manually confirm synch [default: False]
    --preparationStage                                    get aproximate timings and print to shell, to help identify time ranges? [default: False]
    --plotting                                            whether to display confirmation plots [default: False]
    --usedTENSPulses                                      whether the sync was done using TENS pulses (as opposed to mechanical taps) [default: False]
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
import warnings
import h5py
import os
import math as m
import seaborn as sns
import scipy.interpolate as intrp
import quantities as pq
import json
import rcsanalysis.packetizer as rcsa
import rcsanalysis.packet_func as rcsa_helpers
from datetime import datetime as dt
from datetime import timezone
import pytz
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
import neo
import elephant.pandas_bridge as elphpdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#############################################################
if arguments['inputNSPBlockSuffix'] is None:
    inputNSPBlockSuffix = ''
else:
    inputNSPBlockSuffix = '_{}'.format(arguments['inputNSPBlockSuffix'])
if arguments['inputINSBlockSuffix'] is None:
    inputINSBlockSuffix = ''
else:
    inputINSBlockSuffix = "_{}".format(arguments['inputINSBlockSuffix'])
#  load INS Data
############################################################
jsonSessionNames = jsonSessionNames[blockIdx]
#  Load NSP Data
############################################################
if 'rawBlockName' in spikeSortingOpts['utah']:
    BlackrockFileName = ns5FileName.replace(
        'Block', spikeSortingOpts['utah']['rawBlockName'])
else:
    BlackrockFileName = ns5FileName
nspPath = os.path.join(
    scratchFolder,
    BlackrockFileName + inputNSPBlockSuffix +
    '.nix')
print('Loading NSP Block from: {}'.format(nspPath))
nspReader, nspBlock = ns5.blockFromPath(
    nspPath, lazy=arguments['lazy'],
    reduceChannelIndexes=True)
#
synchFunFolder = os.path.join(
    scratchFolder, 'synchFuns'
    )
if not os.path.exists(synchFunFolder):
    os.makedirs(synchFunFolder, exist_ok=True)
synchFunPath = os.path.join(
    synchFunFolder,
    '{}_{}_synchFun.pickle'.format(experimentName, ns5FileName))
if os.path.exists(synchFunPath):
    print('Synch already performed. Turning off plotting')
    arguments['plotting'] = False

tapDetectOptsNSP = expOpts['synchInfo']['nsp'][blockIdx]
tapDetectOptsINS = expOpts['synchInfo']['ins'][blockIdx]
#### INS Loading
insChanNames = []
for insSessIdx, tdo in tapDetectOptsINS.items():
    if isinstance(tdo['synchChanName'], str):
        scn = tdo['synchChanName']
        if ('seg0_' + scn) not in insChanNames:
            insChanNames.append(('seg0_' + scn))
    else:
        for scn in tdo['synchChanName']:
            if ('seg0_' + scn) not in insChanNames:
                insChanNames.append(('seg0_' + scn))
insDFList = {}
for insSessIdx, jsonSessName in enumerate(jsonSessionNames):
    insPath = os.path.join(
        scratchFolder,
        jsonSessName +
        inputINSBlockSuffix +
        '.nix')
    print('Loading INS Block from: {}'.format(insPath))
    insReader, insBlock = ns5.blockFromPath(
        insPath, lazy=arguments['lazy'],
        reduceChannelIndexes=True)
    if arguments['lazy']:
        asigList = [
            asigP.load()
            for asigP in insBlock.filter(objects=AnalogSignalProxy)
            if asigP.name in insChanNames
            ]
    else:
        asigList = [
            asig
            for asig in insBlock.filter(objects=AnalogSignal)
            if asig.name in insChanNames
            ]
    insDFList[insSessIdx] = ns5.analogSignalsToDataFrame(asigList)
pdb.set_trace()
insDF = (
    pd.concat(insDFList, names=['trialSegment', 'index'])
    .reset_index().drop(columns=['index']))
sessionUnixTimeList = [
    int(sessionName.split('Session')[-1])
    for sessionName in jsonSessionNames
    ]
homeTZ = pytz.timezone("America/New_York")
sessionDatetimeList = [
    pd.Timestamp(sut, unit='ms', tz='America/New_York').astimezone('utc')
    for sut in sessionUnixTimeList
]
#  ##### NSP Loading
nspChanNames = []
for insSessIdx, tdo in tapDetectOptsNSP.items():
    if isinstance(tdo['synchChanName'], str):
        scn = tdo['synchChanName']
        if ('seg0_' + scn) not in nspChanNames:
            nspChanNames.append(('seg0_' + scn))
    else:
        for scn in tdo['synchChanName']:
            if ('seg0_' + scn) not in nspChanNames:
                nspChanNames.append(('seg0_' + scn))
if arguments['lazy']:
    asigList = [
        asigP.load()
        for asigP in nspBlock.filter(objects=AnalogSignalProxy)
        if asigP.name in nspChanNames
        ]
else:
    asigList = [
        asig
        for asig in nspBlock.filter(objects=AnalogSignal)
        if asig.name in nspChanNames
        ]
nspDF = ns5.analogSignalsToDataFrame(asigList)
nspAbsStart = pd.Timestamp(nspBlock.annotations['recDatetimeStr'])
nspDF['unixTime'] = pd.TimedeltaIndex(nspDF['t'], unit='s') + nspAbsStart

for insSessIdx, insGroup in insDF.groupby('trialSegment'):
    insSessIdx = int(insSessIdx)
    sessTapOptsNSP = tapDetectOptsNSP[insSessIdx]
    sessTapOptsINS = tapDetectOptsINS[insSessIdx]
    #
    sessStartUnix = sessionDatetimeList[insSessIdx]
    sessStopUnix = sessStartUnix + pd.Timedelta(insGroup['t'].max(), unit='s')
    nspSessMask = (nspDF['unixTime'] > sessStartUnix) & (nspDF['unixTime'] < sessStopUnix)
    #
    if isinstance(sessTapOptsNSP['synchChanName'], str):
        theseChanNames = ['seg0_' + sessTapOptsNSP['synchChanName']]
    else:
        theseChanNames = [
            'seg0_' + scn
            for scn in sessTapOptsNSP['synchChanName']
        ]
    thisNspDF = nspDF.loc[nspSessMask, ['t', 'unixTime']].copy().reset_index(drop=True)
    if True:
        nspVals = nspDF.loc[nspSessMask, theseChanNames].to_numpy()
        empCov = EmpiricalCovariance().fit(nspVals)
        thisNspDF.loc[:, 'signal'] = empCov.mahalanobis(nspVals)
    else:
        nspVals = nspDF.loc[nspSessMask, theseChanNames].mean(axis=1).to_numpy()
        thisNspDF.loc[:, 'signal'] = nspVals
    ## expand to mahaladist later
    if sessTapOptsNSP['timeRanges'] is not None:
        restrictMask = hf.getTimeMaskFromRanges(
            thisNspDF['t'], sessTapOptsNSP['timeRanges'])
        if restrictMask.any():
            thisNspDF.loc[~restrictMask, 'signal'] = np.nan
            thisNspDF.loc[:, 'signal'] = (
                thisNspDF.loc[:, 'signal']
                .interpolate(method='linear', limit_area='inside')
                .fillna(method='bfill').fillna(method='ffill'))
    pdb.set_trace()
#
#
#
# get absolute timestamps of file extents (by INS session)
# INS session name -> absolute timestamp + time domain data last time for the end
#
# if using cross correlation approach, calculate binarized vector of 1's and zeros representing times of stim
# optionally, scale the 1's by the amplitude of the pulses
#
#
# nix Block rec_datetime -> absolute timestamp + td data last time for the end
#
# assign trial segments to the NSP data
# iterate through **trial segments**
# # # find coarse time offset based on absolute times
# # # apply coarse offset to INS data stream
# # # 
# # # if NSP data series has a handwritten time range, zero it outside that range
# # # detect threshold crossings from NSP data series (can be stim artifact, or TENS artifact or whatever)
# # #
# # #
