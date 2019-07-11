"""

Usage:
    preprocNS5.py [options]

Options:
    --exp=exp                       which experimental day to analyze
    --trialIdx=trialIdx             which trial to analyze [default: 1]
"""

import matplotlib, pdb, pickle, traceback
matplotlib.rcParams['agg.path.chunksize'] = 10000
#matplotlib.use('PS')   # generate interactive output by default
#matplotlib.use('TkAgg')   # generate interactive output by default
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from statsmodels import robust
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
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")

from importlib import reload
from scipy import stats
import scipy.interpolate as intrp
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    experimentShorthand=arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
synchFunPath = os.path.join(
    scratchFolder,
    '{}_{}_synchFun_INStoOE.pickle'.format(experimentName, ns5FileName))
print('Loading {}...'.format(oeDataPath))
emgBlock = ns5.loadWithArrayAnn(oeRawDataPath)
print('Loading {}...'.format(insDataPath))
insBlock = ns5.loadWithArrayAnn(insDataPath)
insDF = ns5.analogSignalsToDataFrame(
    insBlock.filter(objects=AnalogSignal), idxT='insT', useChanNames=True)
emgChanIdxs = [
    i
    for i in emgBlock.filter(objects=ChannelIndex)
    if 'CH' in i.name]
segIdx = 0
insSeg = insBlock.segments[segIdx]
emgSeg = emgBlock.segments[segIdx]
emgAsigsZ = np.vstack([
    stats.zscore(np.abs(i.analogsignals[segIdx].magnitude.flatten()))
    for i in emgChanIdxs])
emgSyncAsig = emgChanIdxs[0].analogsignals[0]
emgSyncAsig[:] = np.mean(emgAsigsZ, axis=0)[:, np.newaxis] * emgSyncAsig.units
timeMask = hf.getTimeMaskFromRanges(
    emgSyncAsig.times, synchInfo['oe'][trialIdx][0]['timeRanges'])
timeMask = timeMask[:, np.newaxis]
emgSyncAsig[~timeMask] = 0 * emgSyncAsig.units
(
    peakIdx, emgTensTimes, peakMask, _) = hf.getTensTrigs(
        magThresh=synchInfo['oe'][trialIdx][0]['thresh'],
        tensAsig=emgSyncAsig, plotting=False, peakFinder='cross')
#
evProperties = insSeg.filter(
    objects=Event,
    name='seg{}_ins_property'.format(segIdx))[0]
evProperties.labels = np.array([(i.decode()) for i in evProperties.labels])
evValues = insSeg.filter(
    objects=Event,
    name='seg{}_ins_value'.format(segIdx))[0]
timeMask = hf.getTimeMaskFromRanges(
    evValues.times, synchInfo['ins'][trialIdx][0]['timeRanges'])
propMask = evProperties.labels == 'amplitude'
vMask = np.array([float(i) for i in evValues.labels]) > 0
evMask = propMask & vMask & timeMask
insTensTimes = pd.Series(evValues[evMask].magnitude)
emgTensTimes = pd.Series(emgTensTimes.values)
plotting = True
if plotting:
    fig, ax = plt.subplots()
    ax.plot(emgSyncAsig.times, emgSyncAsig)
    emgPeakMask = peakMask
    plotCenter = emgSyncAsig.times.magnitude[emgPeakMask][0]
    ax.plot(
        emgSyncAsig.times.magnitude[emgPeakMask],
        emgSyncAsig.magnitude[emgPeakMask] ** 0 - 1, 'r*')
    ax.set_title('opE to INS Synch')
    ax.set_xlabel('Time (sec)')
    #  ax.set_xlim([plotCenter - 50e-3, plotCenter + 150e-3])
    ax.set_ylabel('A.U.')
    plt.show()
insTensTimes, emgTensTimes = hf.chooseTriggers(
    insTensTimes, emgTensTimes,
    iti=None,
    plotting=plotting, verbose=True)
#
synchPolyCoeffs = np.polyfit(
    x=insTensTimes,
    y=emgTensTimes,
    deg=1)
#  synchPolyCoeffs = np.array([1, np.mean(emgTensTimes - insTensTimes)])
timeInterpFun = np.poly1d(synchPolyCoeffs)
insDF['oeT'] = timeInterpFun(insDF['insT'])
# get a new dummy asig, in case we truncated the tenssync one
dummyAsig = emgSeg.filter(objects=AnalogSignal)[0]
newT = pd.Series(dummyAsig.times.magnitude)
interpCols = [c for c in insDF.columns if 'ins_' in c]
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
#pdb.set_trace()
for iAsig in insInterpBlock.filter(objects=AnalogSignal):
    origAsigList = insBlock.filter(objects=AnalogSignal, name=iAsig.name)
    if len(origAsigList):
        origAsig = origAsigList[0]
        iAsig.annotations.update(origAsig.annotations)
        if 'td' in origAsig.name:
            elecConfig = '+{}-{}'.format(
                origAsig.annotations['plusInput'],
                origAsig.annotations['minusInput']
            )
            iAsig.annotate(label=elecConfig)
        else:
            iAsig.annotate(label=iAsig.name)
for insEv in insSeg.filter(objects=Event):
    interpEv = Event(
        name=insEv.name,
        times=timeInterpFun(insEv.times.magnitude).flatten() * insEv.times.units,
        labels=insEv.labels,
        **insEv.annotations)
    insInterpBlock.segments[segIdx].events.append(interpEv)
    interpEv.segment = insInterpBlock.segments[segIdx]
alignedTensTimes = timeInterpFun(insTensTimes).flatten()
if plotting:
    plt.plot(insTensTimes, emgTensTimes, 'bo')
    ax.set_xlabel('Difference between Open Ephys and unaligned INS times (sec)')
    plt.show()
    plt.plot(alignedTensTimes - emgTensTimes, 'ro')
    ax.set_xlabel('Difference between Open Ephys and aligned INS times (sec)')
    plt.show()
tensEvents = Event(
    name='seg{}_TENS'.format(segIdx),
    times=alignedTensTimes * pq.s,
    labels=['tens' for i in alignedTensTimes]
    )
insInterpBlock.segments[segIdx].events.append(tensEvents)
tensEvents.segment = insInterpBlock.segments[segIdx]
'''
stimOnMask = propMask & vMask
alignedPulseTimes = timeInterpFun(evValues[stimOnMask].times.magnitude).flatten()

# make events objects
alignEventsDF = pd.DataFrame({
    't': alignedPulseTimes,
    'amplitude': np.around(trialInfoDF.loc[trialIdx, 'insStimAmps'].flatten(), decimals=3)})

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
'''
ns5.addBlockToNIX(
    insInterpBlock, neoSegIdx=[segIdx],
    writeAsigs=True, writeSpikes=False, writeEvents=True,
    fileName=ns5FileName + '_oe',
    folderPath=scratchFolder,
    purgeNixNames=True,
    nixBlockIdx=0, nixSegIdx=[segIdx],
    )
