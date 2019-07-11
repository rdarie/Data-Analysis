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
import shutil
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

trialInfo = {}

insTimingPath = os.path.join(sipFolder, 'RecruitmentCurveStimTimings_RD.mat')
# pulse times in msec
with h5py.File(insTimingPath, "r") as insTimingRecord:
    #  print(insTimingRecord['pulseTimings/AllTENSPulses'].shape)
    nTrials = len(np.array(insTimingRecord['pulseTimings/AllTENSPulses']))
    trialInfo['insTensTimes'] = {i: None for i in range(nTrials)}
    trialInfo['insPulseTimes'] = {i: None for i in range(nTrials)}
    trialInfo['insStimAmps'] = {i: None for i in range(nTrials)}
    for i in range(nTrials):
        trialInfo['insTensTimes'][i] = np.array(
            insTimingRecord[insTimingRecord['pulseTimings/AllTENSPulses'][i][0]]).flatten() / 500
        # allTENSPulses record is in samples, convert to seconds
        trialInfo['insPulseTimes'][i] = np.array(
            insTimingRecord[insTimingRecord['pulseTimings/INS'][i][0]]) / 1000
        # allTENSPulses record is in msec, convert to seconds
        trialInfo['insStimAmps'][i] = np.array(
            insTimingRecord[insTimingRecord['pulseTimings/stimAmps'][i][0]])

trialInfo['sipBaseNames'] = {
    i: 'RCTrial{:0>2}'.format(i + 1) for i in range(nTrials)
    }

trialInfo['emgBaseNames'] = {
    0: 'Trial001-1-EMG',
    1: 'Trial002-2-EMG',
    2: 'Trial003-2-EMG',
    3: 'Trial004-2-EMG',
    4: 'Trial005-2-EMG',
    5: None,
    6: 'Trial007-2-EMG',
    7: 'Trial008-1-EMG',
    8: 'Trial009-1-EMG',
    }

trialInfo['electrode'] = {
    0: '+SpRB4-SpRB1',
    1: '+SpRB2-SpRB1',
    2: '+C-SpRB1',
    3: '+SpCB3-SpCB4',
    4: '+C-SpCB2',
    5: '+C-SpCB3',
    6: '+C-SpCB4',
    7: '+C-SpCB1',
    8: '+C-SpRB2',
    }

#  discard first n seconds
trialInfo['discardTime'] = {
    0: 150,
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
    8: None,
    }

#  discard first n seconds
trialInfo['timestampOffset'] = {
    0: 18e-3,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    }

trialInfo['artifactChan'] = {
    0: 'SenseChannel1',
    1: 'SenseChannel1',
    2: 'SenseChannel1',
    3: 'SenseChannel3',
    4: 'SenseChannel3',
    5: None,
    6: 'SenseChannel3',
    7: 'SenseChannel3',
    8: 'SenseChannel3',
}

tensRangesINS = {
    #  each key is a trial
    0: {
        #  each key is a trialSegment
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    1: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    2: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(18.6, 19), (74.85, 74.95)], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    3: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    4: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    5: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    6: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    7: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        },
    8: {
        #  each key is a trialSegment(18.6, 19), (74.85, 74.95)
        0: {
            'timeRanges': [(), ()], 'chan': 'SenseChannel1',
            'thres': 2, 'iti': 0.1, 'keepIndex': slice(None)},
        }
    }
trialInfoDF = pd.DataFrame(trialInfo)
trialsToAnalyze = [0, 1, 3, 4, 6, 7, 8]
# trialsToAnalyze = [0]

for trialIdx in trialsToAnalyze:
    insDataPath = os.path.join(
        sipFolder, trialInfoDF.loc[trialIdx, 'sipBaseNames'] + '.nix')
    print('Loading {}...'.format(insDataPath))
    insBlock = ns5.loadWithArrayAnn(insDataPath)
    insDF = ns5.analogSignalsToDataFrame(
        insBlock.filter(objects=AnalogSignal), idxT='insT',
        useChanNames=True)
    #
    emgFullName = (
        experimentName + '_' +
        trialInfoDF.loc[trialIdx, 'emgBaseNames'])
    print('Loading {}...'.format(emgFullName))
    emgDataPath = os.path.join(oeFolder, emgFullName, emgFullName + '_filtered.nix')
    emgBlock = ns5.loadWithArrayAnn(emgDataPath)
    emgChans = [
        i
        for i in emgBlock.filter(objects=ChannelIndex)
        if 'label' in i.annotations.keys()]
    tensSyncChanIdx = [
        i
        for i in emgChans
        if i.annotations['label'] == 'TensSync'][0]
    tensSyncAsig = tensSyncChanIdx.analogsignals[0]
    #
    if not np.isnan(trialInfoDF.loc[trialIdx, 'discardTime']):
        newTStart = tensSyncAsig.t_start + trialInfoDF.loc[trialIdx, 'discardTime'] * pq.s
        firstIdx = np.flatnonzero(tensSyncAsig.times > newTStart)[0]
        tensSyncAsig = tensSyncAsig[firstIdx:]
    if np.any(trialInfoDF.loc[trialIdx, 'insTensTimes']):
        insTensTimes = pd.Series(trialInfoDF.loc[trialIdx, 'insTensTimes']) - trialInfoDF.loc[trialIdx, 'timestampOffset']
    else:
        insTensChanName = tensRangesINS[trialIdx][0]['chan']
        insTensChan = insBlock.filter(objects=ChannelIndex, name=insTensChanName)[0]
        insTensAsig = insTensChan.analogsignals[0]
        timeMask = np.zeros_like(insTensAsig.magnitude, dtype=np.bool)
        tUnits = insTensAsig.times.units
        for tStart, tEnd in tensRangesINS[trialIdx][0]['timeRanges']:
            thisMask = (
                (insTensAsig.times >= tStart * tUnits) &
                (insTensAsig.times <= tEnd * tUnits)
                )
            timeMask = timeMask | thisMask[:, np.newaxis]
        insTensAsig[~timeMask] = 0 * insTensAsig.units
        (
            peakIdx, insTensTimes, peakMask, _) = hf.getTensTrigs(
                diffThresh=tensRangesINS[trialIdx][0]['thres'],
                tensAsig=insTensAsig, plotting=True)
    (
        peakIdx, openEphysTensTimes,
        peakMask, insTensTimes) = hf.getTensTrigs(
            diffThresh=10, magThresh=15,
            tensAsig=tensSyncAsig, referenceTimes=insTensTimes,
            plotting=False)
    synchPolyCoeffs = np.polyfit(
        x=insTensTimes.values,
        y=openEphysTensTimes.values,
        deg=1)
    timeInterpFun = np.poly1d(synchPolyCoeffs)
    insDF['oeT'] = timeInterpFun(insDF['insT'])
    # get a new dummy asig, in case we truncated the tenssync one
    dummyAsig = emgBlock.filter(objects=AnalogSignal)[0]
    newT = pd.Series(dummyAsig.times.magnitude)
    interpCols = [c for c in insDF.columns if 'Sense' in c]
    insInterp = hf.interpolateDF(
        insDF, newT,
        kind='linear', fill_value=(0, 0),
        x='oeT', columns=interpCols)
    insInterpBlock = ns5.dataFrameToAnalogSignals(
        insInterp,
        idxT='oeT',
        probeName='insTD', samplingRate=dummyAsig.sampling_rate,
        dataCol=interpCols,
        forceColNames=interpCols)
    #
    alignedPulseTimes = timeInterpFun(
        trialInfoDF.loc[trialIdx, 'insPulseTimes'] - trialInfoDF.loc[trialIdx, 'timestampOffset']).flatten()
    alignedTensTimes = timeInterpFun(
        trialInfoDF.loc[trialIdx, 'insTensTimes'] - trialInfoDF.loc[trialIdx, 'timestampOffset']).flatten()
    # make events objects
    alignEventsDF = pd.DataFrame({
        't': alignedPulseTimes,
        'amplitude': np.around(trialInfoDF.loc[trialIdx, 'insStimAmps'].flatten(), decimals=3)})
    # pdb.set_trace()
    alignEventsDF.loc[:, 'stimCat'] = 'stimOn'
    alignEventsDF.loc[:, 'RateInHz'] = 2
    alignEventsDF.loc[:, 'program'] = 0
    alignEventsDF.loc[:, 'electrode'] = trialInfoDF.loc[trialIdx, 'electrode']
    alignEvents = ns5.eventDataFrameToEvents(
        alignEventsDF,
        idxT='t', annCol=None,
        eventName='seg0_stimAlignTimes', tUnits=pq.s,
        makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    insInterpBlock.segments[0].events.append(alignEvents)
    alignEvents.segment = insInterpBlock.segments[0]
    #  pdb.set_trace()
    concatLabelsDF = alignEventsDF
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatLabelsDF.iterrows()])
    concatEvents = Event(
        name='seg0_stimAlignTimesConcatenated',
        times=alignEvents.times,
        labels=concatLabels
        )
    concatEvents.annotate(nix_name=concatEvents.name)
    insInterpBlock.segments[0].events.append(concatEvents)
    concatEvents.segment = insInterpBlock.segments[0]
    tensEvents = Event(
        name='seg0_TENS',
        times=alignedTensTimes * pq.s,
        labels=['tens' for i in alignedTensTimes]
        )
    tensEvents.annotate(nix_name=tensEvents.name)
    insInterpBlock.segments[0].events.append(tensEvents)
    tensEvents.segment = insInterpBlock.segments[0]
    ns5.addBlockToNIX(
        insInterpBlock, neoSegIdx=[0],
        writeAsigs=True, writeSpikes=False, writeEvents=True,
        fileName=emgFullName + '_filtered',
        folderPath=os.path.join(oeFolder, emgFullName),
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
    outputPath = os.path.join(
        scratchFolder, 'Trial{:0>3}_analyze.nix'.format(trialIdx + 1))
    shutil.copyfile(emgDataPath, outputPath)
