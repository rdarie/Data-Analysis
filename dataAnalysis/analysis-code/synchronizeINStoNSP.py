"""
Usage:
    synchronizeINStoNSP.py [--trialIdx=trialIdx]

Arguments:
    trialIdx            which trial to analyze
"""

from docopt import docopt
import matplotlib, pdb, pickle, traceback
matplotlib.use('TkAgg')   # generate interactive output by default
#  matplotlib.rcParams['agg.path.chunksize'] = 10000
#  matplotlib.use('PS')   # noninteract output
from matplotlib import pyplot as plt

from scipy import stats
from importlib import reload
from datetime import datetime as dt
import peakutils
import numpy as np
import pandas as pd
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.motor_encoder as mea
import dataAnalysis.helperFunctions.helper_functions as hf
import dataAnalysis.helperFunctions.estimateElectrodeImpedances as eti
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import dataAnalysis.preproc.mdt_constants as mdt_constants

import h5py
import os
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

from currentExperiment import *
#  load INS Data
############################################################

arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    insDataPath = os.path.join(
        insFolder,
        experimentName,
        ns5FileName + '_ins.nix')
    trialFilesStim = {
        'ins': {
            'origin': 'ins',
            'experimentName': experimentName,
            'folderPath': insFolder,
            'ns5FileName': ns5FileName,
            'jsonSessionNames': jsonSessionNames[trialIdx],
            'elecIDs': range(17),
            'excludeClus': [],
            'forceRecalc': True,
            'detectStim': True,
            'getINSkwargs': {
                'stimDetectOpts': stimDetectOpts,
                'stimIti': 0, 'fixedDelay': 10e-3,
                'minDist': 0.2, 'minDur': 0.2, 'thres': 3,
                'gaussWid': 200e-3,
                'gaussKerWid': 75e-3,
                'maxSpikesPerGroup': 1, 'plotting': []  # range(1, 1000, 5)
                }
            }
        }

try:
    reader = neo.io.NixIO(filename=insDataPath, mode='ro')
except:
    insBlock = preprocINS.preprocINS(
        trialFilesStim['ins'], plottingFigures=False)
    reader = neo.io.NixIO(filename=insDataPath, mode='ro')

insBlock = reader.read_block()
insBlock.create_relationship()  # need this!
reader.close()
for st in insBlock.filter(objects=SpikeTrain):
    print('unit is {}'.format(st.unit.name))
    print('spiketrain is {}'.format(st.name))
    if 'arrayAnnNames' in st.annotations.keys():
        #  print(st.annotations['arrayAnnNames'])
        for key in st.annotations['arrayAnnNames']:
            st.array_annotations.update({key: st.annotations[key]})

#  pdb.set_trace()
tdDF, accelDF, stimStatus = preprocINS.unpackINSBlock(insBlock)
td = {'data': tdDF, 't': tdDF['t']}
accel = {'data': accelDF, 't': accelDF['t']}

#  Load NSP Data
############################################################
startTime_s = None
dataLength_s = None
#  !!!!!!!! Might be broken with new naming conventions
try:
    channelData, nspBlock = preproc.getNIXData(
        fileName=ns5FileName,
        folderPath=nspFolder,
        elecIds=['ainp7'], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)
except Exception:
    traceback.print_exc()
    reader = preproc.preproc(
        fileName=ns5FileName,
        folderPath=nspFolder,
        fillOverflow=False, removeJumps=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSource='tdc',
        chunkSize=2500
        )
    channelData, nspBlock = preproc.getNIXData(
        fileName=ns5FileName,
        folderPath=nspFolder,
        elecIds=['ainp7'], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)

#  pdb.set_trace()
#  Detect NSP taps
############################################################
getTapsFromNev = False
if getTapsFromNev:
    nevFilePath = os.path.join(
        nspFolder,
        ns5FileName + '.mat')
    tapSpikes = ksa.getNevMatSpikes(
        nevFilePath, nevIDs=[135],
        excludeClus=[], plotting=False)
    tapSpikes = hf.correctSpikeAlignment(
        tapSpikes, 9
        )
    print('{}'.format(tapSpikes['TimeStamps'][0]))

if plottingFigures and False:
    try:
        hf.peekAtTaps(
            td, accel,
            channelData, trialIdx,
            tapDetectOpts, sessionTapRangesNSP,
            insX='t', plotBlocking=plotBlocking,
            allTapTimestampsINS=None,
            allTapTimestampsNSP=None,
            segmentsToPlot=[0])
    except Exception:
        traceback.print_exc()

allTapTimestampsNSP = []
#  TODO: detect all in one, should be easy enough
for trialSegment in pd.unique(td['data']['trialSegment']):
    #  Where in NSP to look
    tStart = sessionTapRangesNSP[trialIdx][trialSegment]['timeRanges'][0]
    tStop = sessionTapRangesNSP[trialIdx][trialSegment]['timeRanges'][1]
    if getTapsFromNev:
        tapTimeStampsNSP = pd.Series(tapSpikes['TimeStamps'][0])
        tapTimeStampsNSP = tapTimeStampsNSP.loc[
            (tapTimeStampsNSP > tStart) & (tapTimeStampsNSP < tStop)
            ]
    else:
        nspMask = (channelData['t'] > tStart) & (channelData['t'] < tStop)
        tapIdxNSP = hf.getTriggers(
            channelData['data'].loc[nspMask, 'ainp7'],
            thres=2, iti=0.2, minAmp=1)
        tapTimestampsNSP = channelData['t'].loc[tapIdxNSP]
    keepIdx = sessionTapRangesNSP[trialIdx][trialSegment]['keepIndex']
    tapTimestampsNSP = tapTimestampsNSP[keepIdx]
    print('tSeg {}:\n nspTaps: {}'.format(
        trialSegment, tapTimestampsNSP))
    allTapTimestampsNSP.append(tapTimestampsNSP)

#  pdb.set_trace()
#  Detect INS taps
############################################################
allTapTimestampsINS = []
for trialSegment in pd.unique(td['data']['trialSegment']):
    print('Trial Segment {}\n'.format(trialSegment))
    #  for trialSegment in [0, 1, 2]:
    accelGroupMask = accel['data']['trialSegment'] == trialSegment
    accelGroup = accel['data'].loc[accelGroupMask, :]
    tdGroupMask = td['data']['trialSegment'] == trialSegment
    tdGroup = td['data'].loc[tdGroupMask, :]

    tapTimestampsINS, peakIdx = hf.getINSTapTimestamp(
        tdGroup, accelGroup,
        tapDetectOpts[trialIdx][trialSegment]
        )
    print('tSeg {}:\n nspTaps: {}'.format(
        trialSegment, tapTimestampsINS))
    allTapTimestampsINS.append(tapTimestampsINS)

if plottingFigures:
    try:
        hf.peekAtTaps(
            td, accel,
            channelData, trialIdx,
            tapDetectOpts, sessionTapRangesNSP,
            insX='t', plotBlocking=True,
            allTapTimestampsINS=allTapTimestampsINS,
            allTapTimestampsNSP=allTapTimestampsNSP,
            segmentsToPlot=[0])
    except Exception:
        traceback.print_exc()

# perform the sync
############################################################
td['data']['NSPTime'] = np.nan
accel['data']['NSPTime'] = np.nan
allTapTimestampsINSAligned = []
for trialSegment in pd.unique(td['data']['trialSegment']):
    trialSegment = int(trialSegment)
    accelGroupMask = accel['data']['trialSegment'] == trialSegment
    accelGroup = accel['data'].loc[accelGroupMask, :]
    tdGroupMask = td['data']['trialSegment'] == trialSegment
    tdGroup = td['data'].loc[tdGroupMask, :]

    theseTapTimestampsINS = allTapTimestampsINS[trialSegment]
    theseTapTimestampsNSP = allTapTimestampsNSP[trialSegment]

    tdGroup, accelGroup, insBlock, thisINStoNSP = hf.synchronizeINStoNSP(
        theseTapTimestampsNSP, theseTapTimestampsINS,
        NSPTimeRanges=(channelData['t'].iloc[0], channelData['t'].iloc[-1]),
        td=tdGroup, accel=accelGroup, insBlock=insBlock,
        trialSegment=trialSegment, degree=0)
    td['data'].loc[tdGroupMask, 'NSPTime'] = tdGroup['NSPTime']
    accel['data'].loc[accelGroupMask, 'NSPTime'] = accelGroup['NSPTime']

    interpFunINStoNSP[trialIdx][trialSegment] = thisINStoNSP
    allTapTimestampsINSAligned.append(thisINStoNSP(
        theseTapTimestampsINS))

print('Interpolation between INS and NSP: ')
print(interpFunINStoNSP[trialIdx])

td['NSPTime'] = td['data']['NSPTime']
accel['NSPTime'] = accel['data']['NSPTime']

if plottingFigures and False:
    try:
        hf.peekAtTaps(
            td, accel,
            channelData, trialIdx,
            tapDetectOpts, sessionTapRangesNSP,
            insX='NSPTime',
            allTapTimestampsINS=allTapTimestampsINSAligned,
            allTapTimestampsNSP=allTapTimestampsNSP)
    except Exception:
        traceback.print_exc()
    

############################################################

addingToNix = True

if addingToNix:
    insBlockJustSpikes = hf.extractSignalsFromBlock(insBlock)
    insSpikeTrains = insBlockJustSpikes.filter(objects=SpikeTrain)
    #  reader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
    #  nspBlock = reader.read_block(lazy=True)
    #  nspStP = nspBlock.filter(objects=SpikeTrainProxy)
    #  nspSt = [i.load(load_waveforms=True) for i in nspStP]
    
    #  spikeReader = neo.io.nixio_fr.NixIO(filename=os.path.join(trialFilesFrom['utah']['folderPath'], 'tdc_' + trialFilesFrom['utah']['ns5FileName'], 'tdc_' + trialFilesFrom['utah']['ns5FileName'] + '.nix'))
    #  tdcBlock = spikeReader.read_block(lazy=True)
    #  tdcStP = tdcBlock.filter(objects=SpikeTrainProxy)
    #  tdcSt = [i.load(load_waveforms=True) for i in tdcStP]

    for st in insSpikeTrains:
        if st.waveforms is None:
            st.sampling_rate = 3e4*pq.Hz
            st.waveforms = np.array([]).reshape((0, 0, 0))*pq.mV
    preproc.addBlockToNIX(
        insBlockJustSpikes, segIdx=0,
        writeAsigs=False, writeSpikes=True,
        fileName=ns5FileName,
        folderPath=nspFolder,
        nixBlockIdx=0, nixSegIdx=0,
        )
    tdColumns = [
        i for i in td['data'].columns
        if 'ins_' in i]
    tdInterp = hf.interpolateDF(
        td['data'], channelData['t'],
        kind='linear', fill_value=(0, 0),
        x='NSPTime', columns=tdColumns)
    tdBlock = preproc.dataFrameToAnalogSignals(
        tdInterp,
        idxT='NSPTime',
        probeName='insTD', samplingRate=3e4*pq.Hz,
        dataCol=tdColumns,
        forceColNames=tdColumns)
    preproc.addBlockToNIX(
        tdBlock, segIdx=0,
        fileName=ns5FileName,
        folderPath=nspFolder,
        nixBlockIdx=0, nixSegIdx=0,
        )
    accelColumns = [
        'ins_accx', 'ins_accy',
        'ins_accz', 'ins_accinertia']
    accelInterp = hf.interpolateDF(
        accel['data'], channelData['t'],
        kind='linear', fill_value=(0, 0),
        x='NSPTime', columns=accelColumns)
    accelBlock = preproc.dataFrameToAnalogSignals(
        accelInterp,
        idxT='NSPTime',
        probeName='insAccel', samplingRate=3e4*pq.Hz,
        dataCol=accelColumns,
        forceColNames=accelColumns)
    preproc.addBlockToNIX(
        accelBlock, segIdx=0,
        fileName=ns5FileName,
        folderPath=nspFolder,
        nixBlockIdx=0, nixSegIdx=0,
        )
