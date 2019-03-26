# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 09:07:44 2019

@author: Radu
"""

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
try:
    channelData, nspBlock = preproc.getNIXData(
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
        elecIds=['ainp7'], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)
except Exception:
    traceback.print_exc()
    reader = preproc.preproc(
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
        fillOverflow=False, removeJumps=False,
        eventInfo=trialFilesFrom['utah']['eventInfo'],
        spikeSource='tdc',
        chunkSize=2500
        )
    channelData, nspBlock = preproc.getNIXData(
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
        elecIds=['ainp7'], startTime_s=startTime_s,
        dataLength_s=dataLength_s,
        signal_group_mode='split-all', closeReader=True)

#  pdb.set_trace()
#  Detect NSP taps
############################################################
getTapsFromNev = False
if getTapsFromNev:
    nevFilePath = os.path.join(
        trialFilesFrom['utah']['folderPath'],
        trialFilesFrom['utah']['ns5FileName'] + '.mat')
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
insBlockJustSpikes = hf.extractSignalsFromBlock(insBlock)
if addingToNix:
    preproc.addBlockToNIX(
        insBlockJustSpikes, segIdx=0,
        writeAsigs=False, writeSpikes=True,
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
        nixBlockIdx=0, nixSegIdx=0,
        )
    tdColumns = ['ins_td0', 'ins_td2', 'amplitude', 'program']
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
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
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
        fileName=trialFilesFrom['utah']['ns5FileName'],
        folderPath=trialFilesFrom['utah']['folderPath'],
        nixBlockIdx=0, nixSegIdx=0,
        )
############################################################
makeAnalysisNix = True

nspReader = neo.io.nixio_fr.NixIO(filename=trialBasePath)
nspBlock = nspReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
if addingToNix:
    dataBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSignals=['position', 'velocityCat'] + tdColumns)
    hf.loadBlockProxyObjects(dataBlock)
    for asig in dataBlock.filter(objects=AnalogSignal):
        chanIdx = asig.channel_index
        oldName = chanIdx.name
        newName = asig.name
        print('name change from {} to {}'.format(
            oldName, newName
            ))
        chanIdx.name = asig.name
    dataBlock.segments[0].name = 'analysis seg'
else:
    dataBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSignals=['position', 'velocityCat'])
    hf.loadBlockProxyObjects(dataBlock)
    #  tests...
    #  [i.unit.channel_index.name for i in insBlockJustSpikes.filter(objects=SpikeTrain)]
    #  [i.channel_index.name for i in nspBlock.filter(objects=AnalogSignalProxy)]
    #  [i.channel_index.name for i in dataBlock.filter(objects=AnalogSignal)]
    
    #  dataBlock already has the stim times if we wrote them to that file
    #  if not, add them here
    dataBlock.segments[0].name = 'analysis seg'
    insBlockJustSpikes.segments[0].name = 'analysis seg'
    dataBlock.merge(insBlockJustSpikes)
#  merge events
evList = []
for key in ['property', 'value']:
    #  key = 'property'
    insProp = insBlock.filter(
        objects=Event,
        name='ins_' + key
        )[0]
    rigProp = dataBlock.filter(
        objects=Event,
        name='rig_' + key
        )
    if len(rigProp):
        rigProp = rigProp[0]
        allProp = insProp.merge(rigProp)
        allProp.name = key

        evSortIdx = np.argsort(allProp.times, kind='mergesort')
        allProp = allProp[evSortIdx]
        evList.append(allProp)
    else:
        #  mini RC's don't have rig_ events
        allProp = insProp
        allProp.name = key
        evList.append(insProp)

#  make concatenated event, for viewing
concatLabels = np.array([
    (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
        elphpdb._convert_value_safe(evList[1].labels[i])) for
    i in range(len(evList[0]))
    ])
concatEvent = Event(
    name='concatenated_updates',
    times=allProp.times,
    labels=concatLabels
    )
concatEvent.merge_annotations(allProp)
evList.append(concatEvent)
dataBlock.segments[0].events = evList

testEventMerge = False
if testEventMerge:
    insProp = dataBlock.filter(
        objects=Event,
        name='ins_property'
        )[0]
    allDF = preproc.eventsToDataFrame(
        [insProp], idxT='t'
        )
    allDF[allDF['ins_property'] == 'movement']
    rigDF = preproc.eventsToDataFrame(
        [rigProp], idxT='t'
        )
testSaveability = True
#  pdb.set_trace()
#  for st in dataBlock.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
#  for st in insBlockJustSpikes.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
#  for st in insBlock.filter(objects=SpikeTrain): print('{}: t_start={}'.format(st.name, st.t_start))
dataBlock = preproc.purgeNixAnn(dataBlock)
writer = neo.io.NixIO(filename=analysisDataPath)
writer.write_block(dataBlock)
writer.close()
############################################################
confirmNixAddition = False
if confirmNixAddition:
    for idx, oUnit in enumerate(insBlock.list_units):
        if len(oUnit.spiketrains[0]):
            st = oUnit.spiketrains[0]
            break

    trialBasePath = os.path.join(
        trialFilesFrom['utah']['folderPath'],
        trialFilesFrom['utah']['ns5FileName'])
    loadedReader = neo.io.nixio_fr.NixIO(filename=trialBasePath + '.nix')
    loadedBlock = loadedReader.read_block(
        block_index=0,
        lazy=True)
    from neo.io.proxyobjects import SpikeTrainProxy
    lStPrx = loadedBlock.filter(objects=SpikeTrainProxy, name=st.name)[0]
    lSt = lStPrx.load()
    plt.eventplot(st.times, label='original', lw=5)
    plt.eventplot(lSt.times, label='loaded', colors='r')
    plt.legend()
    plt.show()
    
############################################################