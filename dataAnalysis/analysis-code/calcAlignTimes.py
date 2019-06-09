"""10a: Calculate align Times
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
"""
import os, pdb, traceback
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
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']),
    arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

#  all experimental days
if arguments['--processAll']:
    dataReader = neo.io.nixio_fr.NixIO(
        filename=experimentDataPath)
else:
    dataReader = neo.io.nixio_fr.NixIO(
        filename=analysisDataPath)

dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')
for ev in dataBlock.filter(objects=EventProxy):
    ev.name = '_'.join(ev.name.split('_')[1:])

#  some categories need to be calculated,
#  others are available; "fuzzy" ones need their
#  alignment fixed
fuzzyCateg = [
    'amplitude', 'amplitudeCat', 'program', 'RateInHz']
availableCateg = [
    'pedalVelocityCat', 'pedalMovementCat',
    'pedalSizeCat', 'pedalSize', 'pedalMovementDuration']
calcFromTD = [
    'stimOffset']
signalsInAsig = [
    'position']

progAmpNames = rcsa_helpers.progAmpNames
expandCols = [
    'RateInHz', 'movement', 'program', 'trialSegment']
deriveCols = ['amplitude', 'amplitudeCat']
columnsToBeAdded = (
    expandCols + deriveCols + progAmpNames)

#  allocate block to contain events
masterBlock = Block()
masterBlock.name = dataBlock.annotations['neo_name']
masterBlock.annotate(
    nix_name=dataBlock.annotations['neo_name'])

blockIdx = 0
checkReferences = False
for segIdx, dataSeg in enumerate(dataBlock.segments):
    print('Calculating align times for trial {}'.format(segIdx))
    signalsInSegment = [
        'seg{}_'.format(segIdx) + i
        for i in signalsInAsig]
    asigProxysList = [
        asigP
        for asigP in dataSeg.filter(objects=AnalogSignalProxy)
        if asigP.annotations['neo_name'] in signalsInSegment]
    eventProxysList = dataSeg.events
    if checkReferences:
        for asigP in asigProxysList:
            da = asigP._rawio.da_list['blocks'][blockIdx]['segments'][segIdx]['data']
            print('segIdx {}, asigP.name {}'.format(
                segIdx, asigP.name))
            print('asigP._global_channel_indexes = {}'.format(
                asigP._global_channel_indexes))
            print('asigP references {}'.format(
                da[asigP._global_channel_indexes[0]]))
            try:
                assert asigP.name in da[asigP._global_channel_indexes[0]].name
            except Exception:
                traceback.print_exc()
        
        for evP in eventProxysList:
            print('segIdx {}, evP.name {}'.format(
                segIdx, evP.name))
            print('evP._event_channel_index = {}'.format(
                 evP._event_channel_index))
            evP_ch = evP._event_channel_index
            mts = evP._rawio.file.blocks[blockIdx].groups[segIdx].multi_tags
            try:
                assert evP.name in mts[evP_ch].name
            except Exception:
                traceback.print_exc()
    asigsList = [
        asigP.load()
        for asigP in asigProxysList]
    samplingRate = asigsList[0].sampling_rate
    asigsDF = preproc.analogSignalsToDataFrame(asigsList)

    dataSegEvents = [evP.load() for evP in eventProxysList]
    eventDF = preproc.eventsToDataFrame(
        dataSegEvents, idxT='t',
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
    
    #  get alignment times
    moveMask = pd.Series(False, index=tdDF.index)
    stopMask = pd.Series(False, index=tdDF.index)
    for idx, group in tdDF.groupby('trialSegment'):
        idx = int(idx)
        movingAtAll = group['pedalVelocityCat'].fillna(0).abs()
        movementOnOff = movingAtAll.diff()
        taskMask = (
            (group['t'] > alignTimeBounds[segIdx][idx][0]) &
            (group['t'] < alignTimeBounds[segIdx][idx][1])
            )
        moveMaskForSeg = (movementOnOff == 1) & taskMask
        stopMaskForSeg = (movementOnOff == -1) & taskMask
        try:
            #  group.loc[moveMaskForSeg, 't'] - tdDF['t'].iloc[0]
            assert stopMaskForSeg.sum() == moveMaskForSeg.sum(), 'unequal start and stop lengths' 
            assert stopMaskForSeg.sum() % 2 == 0, 'number of movements not divisible by 2'
        except Exception:
            traceback.print_exc()
            #  pdb.set_trace()
        moveMask.loc[moveMaskForSeg.index[moveMaskForSeg]] = True
        stopMask.loc[stopMaskForSeg.index[stopMaskForSeg]] = True
    
    #  plt.plot(movementOnOff)
    #  plt.plot(tdDF['trialSegment'].values)
    #  plt.plot(tdDF['pedalVelocityCat'].values); plt.show()
    
    moveTimes = tdDF.loc[
        moveMask, 't']
    stopTimes = tdDF.loc[
        stopMask, 't']
    
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
        
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedBase').sum())
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedPeak').sum())
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'return').sum())

    outboundMask = tdDF['pedalMovementCat'] == 'outbound'
    reachedPeakMask = tdDF['pedalMovementCat'] == 'reachedPeak'
    returnMask = tdDF['pedalMovementCat'] == 'return'
    reachedBaseMask = tdDF['pedalMovementCat'] == 'reachedBase'
    
    #  calculate movement sizes
    midPeakIdx = ((
        returnMask[returnMask].index +
        reachedPeakMask[reachedPeakMask].index) / 2).astype('int64')
    tdDF['pedalSize'] = np.nan
    tdDF.loc[midPeakIdx, 'pedalSize'] = tdDF.loc[midPeakIdx, 'pedalPosition']
    tdDF['pedalSize'].interpolate(method='nearest', inplace=True)
    tdDF['pedalSize'].fillna(method='ffill', inplace=True)
    tdDF['pedalSize'].fillna(method='bfill', inplace=True)

    #  plt.plot(tdDF['t'], tdDF['pedalSize'])
    #  plt.plot(tdDF['t'], tdDF['pedalPosition']); plt.show()
    '''
    import seaborn as sns
    ax = sns.distplot(tdDF.loc[midPeakIdx, 'pedalPosition'])
    plt.savefig(
        os.path.join(
            figureFolder, 'debugPedalSize.pdf'))
    plt.close()
    '''
    tdDF['pedalSizeCat'] = pd.cut(
        tdDF['pedalSize'], movementSizeBins,
        labels=['XL', 'L', 'M', 'S', 'XS'])
    
    #  calculate movement durations
    tdDF['pedalMovementDuration'] = np.nan
    outboundTimes = tdDF.loc[
        outboundMask,
        't']
    reachedBaseTimes = tdDF.loc[
        reachedBaseMask,
        't']
    tdDF.loc[midPeakIdx, 'pedalMovementDuration'] = (
        reachedBaseTimes.values -
        outboundTimes.values
        )
    # import seaborn as sns
    # sns.distplot(tdDF['pedalMovementDuration'].dropna())
    tdDF['pedalMovementDuration'].interpolate(method='nearest', inplace=True)
    tdDF['pedalMovementDuration'].fillna(method='ffill', inplace=True)
    tdDF['pedalMovementDuration'].fillna(method='bfill', inplace=True)

    peakTimes = tdDF.loc[midPeakIdx, 't']
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
    otherCategories['pedalSize'] = 0
    otherCategories['amplitudeCat'] = 999
    otherCategories['pedalSizeCat'] = 'Control'
    otherCategories['pedalMovementCat'] = 'Control'
    otherCategories['pedalMovementDuration'] = 999
    
    peakCategories = tdDF.loc[
        tdDF['t'].isin(peakTimes), fuzzyCateg + availableCateg
        ].reset_index(drop=True)
    peakCategories['pedalMetaCat'] = 'midPeak'
    peakCategories['pedalMovementCat'] = 'midPeak'
    
    alignTimes = pd.concat((
        moveTimes, stopTimes, otherTimes, peakTimes),
        axis=0, ignore_index=True)
    #  sort align times
    alignTimes.sort_values(inplace=True, kind='mergesort')
    categories = pd.concat((
        moveCategories, stopCategories, otherCategories, peakCategories),
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
    #  scan through and calculate fuzzy values
    fudgeFactor = 500e-3  # seconds
    '''
    import seaborn as sns
    for colName in progAmpNames:
        sns.distplot(tdDF[colName], label=colName)
    for colName in progAmpNames + ['amplitude']:
        plt.plot(tdDF[colName], label=colName)
    plt.plot(tdDF['amplitudeCat'], label=colName)
    plt.show()
    '''
    for idx, tOnset in alignTimes.iteritems():
        moveCat = categories.loc[idx, 'pedalMovementCat']
        metaCat = categories.loc[idx, 'pedalMetaCat']
        if moveCat == 'outbound':
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
            else:
                categories.loc[idx, 'stimOffset'] = 999
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
        elif metaCat == 'rest':
            for colName in fuzzyCateg:
                categories.loc[idx, colName + 'Fuzzy'] = 999
        else:
            #  everyone else inherits the categories of the ountbound leg
            for colName in fuzzyCateg:
                categories.loc[idx, colName + 'Fuzzy'] = np.nan
    categories.fillna(method='ffill', inplace=True)
    categories.fillna(method='bfill', inplace=True)
    #  fix program labels
    #  INS amplitudes are in 100s of uA
    categories['amplitude'] = categories['amplitude'] * 100
    categories['amplitudeFuzzy'] = categories['amplitudeFuzzy'] * 100
    '''
    #  pull actual electrode names
    categories['electrode'] = np.nan
    for pName in pd.unique(categories['program']):
        pMask = categories['program'] == pName
        if pName == 999:
            categories.loc[pMask, 'electrode'] = 'Control'
        else:
            unitName = 'g0p{}'.format(int(pName))
            thisUnit = dataBlock.filter(objects=Unit, name=unitName)[0]
            cathodes = thisUnit.annotations['cathodes']
            anodes = thisUnit.annotations['anodes']
            elecName = ''
            if isinstance(anodes, Iterable):
                elecName += '+ ' + ', '.join(['E{}'.format(i) for i in anodes])
            else:
                elecName += '+ E{}'.format(anodes)
            elecName += ' '
            if isinstance(cathodes, Iterable):
                elecName += '- ' + ', '.join(['E{}'.format(i) for i in cathodes])
            else:
                elecName += '- E{}'.format(cathodes)
            categories.loc[pMask, 'electrode'] = elecName
    '''

    alignEventsDF = pd.concat([alignTimes, categories], axis=1)
    #  pdb.set_trace()
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
if arguments['--processAll']:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeAsigs=False, writeSpikes=False, writeEvents=True,
        fileName=experimentName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )
else:
    preproc.addBlockToNIX(
        masterBlock, neoSegIdx=allSegs,
        writeAsigs=False, writeSpikes=False, writeEvents=True,
        fileName=ns5FileName + '_analyze',
        folderPath=scratchFolder,
        purgeNixNames=False,
        nixBlockIdx=0, nixSegIdx=allSegs,
        )