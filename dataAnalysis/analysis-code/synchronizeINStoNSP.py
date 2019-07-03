"""07: Combine INS and NSP Data
Usage:
    synchronizeINStoNSP [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --curateManually                whether to manually confirm synch [default: False]
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

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#  load INS Data
############################################################
synchFunPath = os.path.join(
    scratchFolder,
    '{}_{}_synchFun.pickle'.format(experimentName, ns5FileName))
print('Loading INS Block...')
'''
reader = neo.io.NixIO(filename=insDataPath, mode='ro')
insBlock = reader.read_block()
insBlock.create_relationship()  # need this!
reader.close()
for st in insBlock.filter(objects=SpikeTrain):
    #  print('unit is {}'.format(st.unit.name))
    #  print('spiketrain is {}'.format(st.name))
    if 'arrayAnnNames' in st.annotations.keys():
        #  print(st.annotations['arrayAnnNames'])
        for key in st.annotations['arrayAnnNames']:
            st.array_annotations.update({key: st.annotations[key]})
'''
insBlock = preproc.loadWithArrayAnn(insDataPath)
tdDF, accelDF, stimStatus = preprocINS.unpackINSBlock(insBlock)
td = {'data': tdDF, 't': tdDF['t']}
accel = {'data': accelDF, 't': accelDF['t']}
#  Load NSP Data
############################################################
startTime_s = None
dataLength_s = None
print('Loading NSP Block...')
try:
    channelData, nspBlock = preproc.getNIXData(
        fileName=ns5FileName,
        folderPath=scratchFolder,
        elecIds=['ainp7'], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)
except Exception:
    traceback.print_exc()

allTapTimestampsNSP = []
print('Detecting NSP Timestamps...')
#  TODO: detect all in one, should be easy enough
#  pdb.set_trace()
for trialSegment in pd.unique(td['data']['trialSegment']):
    #  Where in NSP to look
    #  pdb.set_trace()
    tStart = sessionTapRangesNSP[trialIdx][trialSegment]['timeRanges'][0]
    tStop = sessionTapRangesNSP[trialIdx][trialSegment]['timeRanges'][1]
    nspMask = (channelData['t'] > tStart) & (channelData['t'] < tStop)
    #
    tapIdxNSP = hf.getTriggers(
        channelData['data'].loc[nspMask, 'ainp7'],
        thres=2, iti=0.2, minAmp=1)
    tapTimestampsNSP = channelData['t'].loc[tapIdxNSP]
    keepIdx = sessionTapRangesNSP[trialIdx][trialSegment]['keepIndex']
    tapTimestampsNSP = tapTimestampsNSP.iloc[keepIdx]
    print('tSeg {}: NSP Taps:\n{}'.format(
        trialSegment, tapTimestampsNSP))
    print('diff:\n{}'.format(
        tapTimestampsNSP.diff() * 1e3))
    allTapTimestampsNSP.append(tapTimestampsNSP)

#  Detect INS taps
############################################################
if not os.path.exists(synchFunPath):
    allTapTimestampsINS = []
    print('Detecting INS Timestamps...')
    overrideSegments = overrideSegmentsForTapSync[trialIdx]
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
        print('tSeg {}, INS Taps:\n{}'.format(
            trialSegment, tapTimestampsINS))
        print('diff:\n{}'.format(
            tapTimestampsINS.diff() * 1e3))
        allTapTimestampsINS.append(tapTimestampsINS)

    if arguments['curateManually']:
        try:
            clickDict = hf.peekAtTaps(
                td, accel,
                channelData, trialIdx,
                tapDetectOpts, sessionTapRangesNSP,
                insX='t', plotBlocking=plotBlocking,
                allTapTimestampsINS=allTapTimestampsINS,
                allTapTimestampsNSP=allTapTimestampsNSP,
                )
        except Exception:
            traceback.print_exc()
            #  pdb.set_trace()
    else:
        clickDict = {
            i: {
                'ins': [],
                'nsp': []
                }
            for i in pd.unique(td['data']['trialSegment'])}

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

        if len(clickDict[trialSegment]['ins']):
            allTapTimestampsINS[trialSegment] = clickDict[trialSegment]['ins']
        #
        theseTapTimestampsINS = allTapTimestampsINS[trialSegment]
        if len(clickDict[trialSegment]['nsp']):
            allTapTimestampsNSP[trialSegment] = clickDict[trialSegment]['nsp']
        #
        theseTapTimestampsNSP = allTapTimestampsNSP[trialSegment]

        if trialSegment in overrideSegments.keys():
            print('\t Overriding trialSegment {}'.format(trialSegment))
            theseTapTimestampsINS = allTapTimestampsINS[overrideSegments[trialSegment]]
            theseTapTimestampsNSP = allTapTimestampsNSP[overrideSegments[trialSegment]]

        tdGroup, accelGroup, insBlock, thisINStoNSP = preproc.synchronizeINStoNSP(
            theseTapTimestampsNSP, theseTapTimestampsINS,
            NSPTimeRanges=(channelData['t'].iloc[0], channelData['t'].iloc[-1]),
            td=tdGroup, accel=accelGroup, insBlock=insBlock,
            trialSegment=trialSegment, degree=0)
        td['data'].loc[tdGroupMask, 'NSPTime'] = tdGroup['NSPTime']
        accel['data'].loc[accelGroupMask, 'NSPTime'] = accelGroup['NSPTime']

        interpFunINStoNSP[trialIdx][trialSegment] = thisINStoNSP
        allTapTimestampsINSAligned.append(thisINStoNSP(
            theseTapTimestampsINS))
    with open(synchFunPath, 'wb') as f:
        pickle.dump(interpFunINStoNSP, f)
else:
    with open(synchFunPath, 'rb') as f:
        interpFunINStoNSP = pickle.load(f)

print('Interpolation between INS and NSP: ')
print(interpFunINStoNSP[trialIdx])

td['NSPTime'] = td['data']['NSPTime']
accel['NSPTime'] = accel['data']['NSPTime']

if plottingFigures:
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
        insBlockJustSpikes, neoSegIdx=[0],
        writeAsigs=False, writeSpikes=True,
        fileName=ns5FileName,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
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
        tdBlock, neoSegIdx=[0],
        fileName=ns5FileName,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
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
        accelBlock, neoSegIdx=[0],
        fileName=ns5FileName,
        folderPath=scratchFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
