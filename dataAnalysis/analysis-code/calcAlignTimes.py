import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import quantities as pq
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Iterable
#  load options
from currentExperiment import *

#  all experimental days
dataReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
"""
    some categories need to be calculated,
    others are available; "fuzzy" ones need their
    alignment fixed
"""
fuzzyCateg = [
    'amplitude', 'program', 'RateInHz']
availableCateg = [
    'pedalVelocityCat', 'pedalMovementCat', 'pedalSizeCat']
calcFromTD = [
    'stimOffset']
signalsInAsig = [
    'position']

progAmpNames = rcsa_helpers.progAmpNames
expandCols = [
    'RateInHz', 'movement', 'amplitude', 'program']
deriveCols = []
columnsToBeAdded = (
    expandCols + deriveCols)

movementSizeBins = [-0.9, -0.45, -0.2, 0.2, 0.55, 0.9]

#  allocate block to contain events
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=dataBlock.annotations['neo_name'])

for segIdx, dataSeg in enumerate(dataBlock.segments):
    signalsInSegment = [
        'seg{}_'.format(segIdx) + i
        for i in signalsInAsig]
    asigProxysList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if asigP.annotations['neo_name'] in signalsInSegment]
    asigP = asigProxysList[0]
    asigsList = [
        asigP.load()
        for asigP in asigProxysList]
    samplingRate = asigsList[0].sampling_rate
    asigsDF = preproc.analogSignalsToDataFrame(asigsList)
    dataSeg.events = [ev.load() for ev in dataSeg.events]
    eventDF = preproc.eventsToDataFrame(
        dataSeg.events, idxT='t',
        names=['property', 'value']
        )
    stimStatus = hf.stimStatusSerialtoLong(
        eventDF, idxT='t', namePrefix='', expandCols=expandCols,
        deriveCols=deriveCols, progAmpNames=progAmpNames)
    infoFromStimStatus = hf.interpolateDF(
            stimStatus, asigsDF['t'],
            x='t', columns=columnsToBeAdded, kind='previous')

    tdDF = pd.concat((
        asigsDF,
        infoFromStimStatus.drop(columns='t')),
        axis=1)
    tdDF.rename(
        columns={
            'movement': 'pedalVelocityCat',
            'position': 'pedalPosition'},
        inplace=True)
    atRest = (
        (tdDF['pedalVelocityCat'] == 1) &
        (tdDF['pedalVelocityCat'].shift(-1) == 0))
    restPosition = tdDF.loc[atRest, 'pedalPosition'].mean()
    atPeak = (
        ((tdDF['pedalPosition'] - restPosition).abs() > 0.01) &
        (tdDF['pedalVelocityCat'] == 0))
    tdDF['movementPeak'] = np.nan
    tdDF.loc[atPeak, 'movePeak'] = tdDF.loc[atPeak, 'pedalPosition']
    tdDF['movementPeak'].interpolate(method='nearest', inplace=True)
    tdDF['movementSizeCat'] = pd.cut(
        tdDF['movementPeak'], movementSizeBins,
        labels=['XL', 'L', 'M', 'S', 'XS'])
    #  get alignment times
    movingAtAll = tdDF['pedalVelocityCat'].fillna(0).abs()
    movementOnOff = movingAtAll.diff()
    moveMask = movementOnOff == 1
    moveTimes = tdDF.loc[
        moveMask, 't']

    stopMask = movementOnOff == -1
    stopTimes = tdDF.loc[
        stopMask, 't']
    # if stopped before the first start, drop it
    dropIndices = stopTimes.index[stopTimes < moveTimes.iloc[0]]
    stopTimes.drop(index=dropIndices, inplace=True)
    
    tdDF['pedalMovementCat'] = np.nan
    tdDF.loc[
        moveMask & (tdDF['pedalVelocityCat'] == 1),
        'pedalMovementCat'] = 'return'
    tdDF.loc[
        moveMask & (tdDF['pedalVelocityCat'] == -1),
        'pedalMovementCat'] = 'outbound'
    tdDF.loc[
        stopMask & (tdDF['pedalVelocityCat'].shift(1) == -1),
        'pedalMovementCat'] = 'reachedPeak'
    tdDF.loc[
        stopMask & (tdDF['pedalVelocityCat'].shift(1) == 1),
        'pedalMovementCat'] = 'reachedBase'
    #  get intervals halfway between move stop and move start

    pauseLens = moveTimes.shift(-1).values - stopTimes
    maskForLen = pauseLens > 1.5

    halfOffsets = (
        samplingRate.magnitude * (pauseLens / 2)).fillna(0).astype(int)

    otherTimesIdx = (stopTimes.index + halfOffsets.values)[maskForLen]
    otherTimes = tdDF.loc[otherTimesIdx, 't']
    
    moveCategories = tdDF.loc[
        tdDF['t'].isin(moveTimes), fuzzyCateg + availableCateg
        ].reset_index(drop=True)
    moveCategories['pedalMetaCat'] = 'onset'
    stopCategories = tdDF.loc[
        tdDF['t'].isin(stopTimes), fuzzyCateg + availableCateg
        ].reset_index(drop=True)
    stopCategories['pedalMetaCat'] = 'offset'
    otherCategories = tdDF.loc[
        tdDF['t'].isin(otherTimes), fuzzyCateg + availableCateg
        ].reset_index(drop=True)
    otherCategories['pedalMetaCat'] = 'rest'

    otherCategories['program'] = 999
    otherCategories['RateInHz'] = 999
    otherCategories['amplitude'] = 999
    otherCategories['pedalSizeCat'] = 'Control'
    otherCategories['pedalMovementCat'] = 'Control'
    
    alignTimes = pd.concat((
        moveTimes, stopTimes, otherTimes),
        axis=0, ignore_index=True)
    #  sort align times
    alignTimes.sort_values(inplace=True, kind='mergesort')
    categories = pd.concat((
        moveCategories, stopCategories, otherCategories),
        axis=0, ignore_index=True)
    #  sort categories by align times
    #  (needed to propagate values forward)
    categories = categories.loc[alignTimes.index, :]
    alignTimes.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
        
    for colName in calcFromTD:
        categories[colName] = np.nan
    for colName in fuzzyCateg:
        categories[colName + 'Fuzzy'] = np.nan
    fudgeFactor = 300e-3  # seconds
    for idx, tOnset in alignTimes.iteritems():
        moveCat = categories.loc[idx, 'pedalMovementCat']
        metaCat = categories.loc[idx, 'pedalMetaCat']
        if metaCat == 'onset':
            tStart = max(0, tOnset - fudgeFactor)
            tStop = min(tdDF['t'].iloc[-1], tOnset + fudgeFactor)

            tdMaskPre = (tdDF['t'] > tStart) & (tdDF['t'] < tOnset)
            tdMaskPost = (tdDF['t'] > tOnset) & (tdDF['t'] < tStop)
            tdMask = (tdDF['t'] > tStart) & (tdDF['t'] < tStop)

            theseAmps = tdDF.loc[tdMask, ['t', 'amplitude']]
            ampDiff = theseAmps.diff()

            ampOnset = theseAmps.loc[
                ampDiff[ampDiff['amplitude'] > 0].index, 't']
            if len(ampOnset):
                # greater if movement after stim
                categories.loc[idx, 'stimOffset'] = tOnset - ampOnset.iloc[0]

            ampOffset = theseAmps.loc[
                ampDiff[ampDiff['amplitude'] < 0].index, 't']
            #  if there's an amp offset, use the last value where amp was on
            if len(ampOffset):
                fuzzyIdx = ampDiff[ampDiff['amplitude'] < 0].index[0] - 1
            else:
                fuzzyIdx = tdDF.loc[tdMask, :].index[-1]
            #  pdb.set_trace()
            for colName in fuzzyCateg:
                nominalValue = categories.loc[idx, colName]
                fuzzyValue = tdDF.loc[fuzzyIdx, colName]

                if (nominalValue != fuzzyValue):
                    categories.loc[idx, colName + 'Fuzzy'] = fuzzyValue
                    print('nominally, {} is {}'.format(colName, nominalValue))
                    print('changed it to {}'.format(fuzzyValue))
                else:
                    categories.loc[idx, colName + 'Fuzzy'] = nominalValue

        elif metaCat == 'offset':
            #  offsets inherit the amplitude of the onset
            for colName in fuzzyCateg:
                categories.loc[idx, colName + 'Fuzzy'] = np.nan

    categories.fillna(method='ffill', inplace=True)
    alignEventsDF = pd.concat([alignTimes, categories], axis=1)
    alignEvents = preproc.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_alignTimes'.format(segIdx), tUnits=pq.s,
        makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    newSeg = Segment(name=dataSeg.annotations['neo_name'])
    newSeg.annotate(nix_name=dataSeg.annotations['neo_name'])
    newSeg.events.append(alignEvents)
    alignEvents.segment = newSeg
    masterBlock.segments.append(newSeg)
dataReader.file.close()

masterBlock.create_relationship()
allSegs = list(range(len(masterBlock.segments)))
preproc.addBlockToNIX(
    masterBlock, neoSegIdx=allSegs,
    writeSpikes=False, writeEvents=False,
    fileName=trialFilesStim['ins']['experimentName'] + '_analyze',
    folderPath=os.path.join(
        trialFilesStim['ins']['folderPath'],
        trialFilesStim['ins']['experimentName']),
    purgeNixNames=False,
    nixBlockIdx=0, nixSegIdx=allSegs,
    )
