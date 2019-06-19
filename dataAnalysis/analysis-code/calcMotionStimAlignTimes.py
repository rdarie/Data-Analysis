"""10a: Calculate align Times
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                  which trial to analyze [default: 1]
    --exp=exp                            which experimental day to analyze
    --processAll                         process entire experimental day? [default: False]
    --plotParamHistograms                plot pedal size, amplitude, duration distributions? [default: False]
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
import seaborn as sns
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

#  fetch stim details
insReader = neo.NixIO(
    filename=insDataPath)
insBlock = insReader.read_block(0)

#  all experimental days?
if arguments['--processAll']:
    dataReader = neo.io.nixio_fr.NixIO(
        filename=experimentDataPath)
else:
    alignTimeBounds = [
        alignTimeBoundsLookup[int(arguments['--trialIdx'])]
    ]
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
    'pedalVelocityCat', 'pedalMovementCat', 'pedalDirection',
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

        assert stopMaskForSeg.sum() == moveMaskForSeg.sum(), 'unequal start and stop lengths' 
        assert stopMaskForSeg.sum() % 2 == 0, 'number of movements not divisible by 2'

        moveMask.loc[moveMaskForSeg.index[moveMaskForSeg]] = True
        stopMask.loc[stopMaskForSeg.index[stopMaskForSeg]] = True
    
    #  plt.plot(movementOnOff)
    #  plt.plot(tdDF['trialSegment'].values)
    #  plt.plot(tdDF['pedalVelocityCat'].values); plt.show()
    #  plt.plot(tdDF['pedalPosition'].values); plt.show()
    
    moveTimes = tdDF.loc[
        moveMask, 't']
    stopTimes = tdDF.loc[
        stopMask, 't']
    
    tdDF['pedalMovementCat'] = np.nan
    for idx, dfMoveIdx in enumerate(moveMask.index[moveMask]):
        dfStopIndex = stopMask.index[stopMask][idx]
        assert dfStopIndex > dfMoveIdx
        if (idx % 2) == 0:
            tdDF.loc[dfMoveIdx, 'pedalMovementCat'] = 'outbound'
            tdDF.loc[dfStopIndex, 'pedalMovementCat'] = 'reachedPeak'
        else:
            tdDF.loc[dfMoveIdx, 'pedalMovementCat'] = 'return'
            tdDF.loc[dfStopIndex, 'pedalMovementCat'] = 'reachedBase'
    outboundMask = tdDF['pedalMovementCat'] == 'outbound'
    reachedPeakMask = tdDF['pedalMovementCat'] == 'reachedPeak'
    returnMask = tdDF['pedalMovementCat'] == 'return'
    reachedBaseMask = tdDF['pedalMovementCat'] == 'reachedBase'
    #  check that the trials are intact
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedBase').sum())
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'reachedPeak').sum())
    assert (
        (tdDF['pedalMovementCat'] == 'outbound').sum() ==
        (tdDF['pedalMovementCat'] == 'return').sum())
    
    #  calculate movement sizes (relative to starting point)
    midPeakIdx = ((
        returnMask[returnMask].index +
        reachedPeakMask[reachedPeakMask].index) / 2).astype('int64')
    tdDF['pedalSize'] = np.nan
    tdDF.loc[midPeakIdx, 'pedalSize'] = tdDF.loc[midPeakIdx, 'pedalPosition']
    tdDF['pedalSize'].interpolate(method='nearest', inplace=True)
    tdDF['pedalSize'].fillna(method='ffill', inplace=True)
    tdDF['pedalSize'].fillna(method='bfill', inplace=True)
    for idx, group in tdDF.groupby('trialSegment'):
        idx = int(idx)
        taskMask = (
            (group['t'] > alignTimeBounds[segIdx][idx][0]) &
            (group['t'] < alignTimeBounds[segIdx][idx][1])
            )
        # get pedal start point
        pedalNeutralPoint = group.loc[taskMask, 'pedalPosition'].iloc[0]
        tdDF.loc[group.index, 'pedalSize'] = group['pedalSize'] - pedalNeutralPoint
    #  plt.plot(tdDF['t'], tdDF['pedalSize'])
    #  plt.plot(tdDF['t'], tdDF['pedalPosition']); plt.show()
    if (segIdx == 0) and arguments['--plotParamHistograms']:
        ax = sns.distplot(
            tdDF.loc[midPeakIdx, 'pedalSize'],
            bins=200, kde=False)
        plt.savefig(
            os.path.join(
                figureFolder, 'pedalSizeDistribution.pdf'))
        plt.close()
    #  determine size category
    tdDF['pedalSizeCat'] = pd.cut(
        tdDF['pedalSize'].abs(), movementSizeBins,
        labels=['XS', 'S', 'M', 'L', 'XL'])
    #  determine CW or CCW
    tdDF['pedalDirection'] = np.nan
    tdDF.loc[tdDF['pedalSize'] > 0, 'pedalDirection'] = 'CW'
    tdDF.loc[tdDF['pedalSize'] <= 0, 'pedalDirection'] = 'CCW'
    #  calculate movement durations
    tdDF['pedalMovementDuration'] = np.nan
    outboundTimes = tdDF.loc[
        outboundMask,
        't']
    reachedBaseTimes = tdDF.loc[
        reachedBaseMask,
        't']
    pedalMovementDurations = (
        reachedBaseTimes.values -
        outboundTimes.values
        )
    tdDF.loc[outboundTimes.index, 'pedalMovementDuration'] = (
        pedalMovementDurations
        )
    tdDF.loc[reachedBaseTimes.index, 'pedalMovementDuration'] = (
        pedalMovementDurations
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
    otherCategories['pedalMetaCat'] = 'control'

    otherCategories['program'] = 999
    otherCategories['RateInHz'] = 0
    otherCategories['amplitude'] = 0
    otherCategories['pedalSize'] = 0
    otherCategories['pedalDirection'] = 'NA'
    otherCategories['amplitudeCat'] = 0
    otherCategories['pedalSizeCat'] = 'NA'
    otherCategories['pedalMovementCat'] = 'NA'
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
            if len(ampOffset):
                #  if the stim ever turns off
                #  use the last value where amp was on
                fuzzyIdx = ampDiff[ampDiff['amplitude'] < 0].index[0] - 1
            else:
                #  use the last value (amp stayed constant,
                #  including the case where it was zero throughout)
                fuzzyIdx = tdDF.loc[tdMask, :].index[-1]
            for colName in fuzzyCateg:
                nominalValue = categories.loc[idx, colName]
                fuzzyValue = tdDF.loc[fuzzyIdx, colName]
                if (nominalValue != fuzzyValue):
                    categories.loc[idx, colName + 'Fuzzy'] = fuzzyValue
                    print('nominally, {} is {}'.format(colName, nominalValue))
                    print('changed it to {}'.format(fuzzyValue))
                else:
                    categories.loc[idx, colName + 'Fuzzy'] = nominalValue
            if categories.loc[idx, 'amplitudeFuzzy'] == 0:
                categories.loc[idx, 'programFuzzy'] = 999
        elif metaCat == 'control':
            for colName in fuzzyCateg:
                if colName == 'program':
                    categories.loc[idx, colName + 'Fuzzy'] = 999
                else:
                    categories.loc[idx, colName + 'Fuzzy'] = 0
        else:
            #  everyone else inherits the categories of the outbound leg
            for colName in fuzzyCateg:
                categories.loc[idx, colName + 'Fuzzy'] = np.nan
    #  fill in the nans for the offset and midpeak times
    categories.fillna(method='ffill', inplace=True)
    categories.fillna(method='bfill', inplace=True)
        
    for colName in ['RateInHz', 'RateInHzFuzzy']:
        categories.loc[categories['amplitudeCatFuzzy'] == 0, colName] = 0

    uniqProgs = pd.unique(categories['programFuzzy'])
    #  plot these if we need to reset the category
    if (segIdx == 0) and arguments['--plotParamHistograms']:
        fig, ax = plt.subplots(len(uniqProgs), 1, sharex=True)
        for idx, pName in enumerate(uniqProgs):
            ax[idx] = sns.distplot(
                categories.loc[
                    categories['programFuzzy'] == pName,
                    'amplitudeFuzzy'],
                bins=100, kde=False, ax=ax[idx]
                )
            ax[idx].set_title('prog {}'.format(pName))
        plt.savefig(
            os.path.join(
                figureFolder, 'amplitudeDistribution.pdf'))
        plt.close()
        uniqSizes = pd.unique(categories['pedalSizeCat'])
        fig, ax = plt.subplots(len(uniqSizes), 1, sharex=True)
        durationBins = np.linspace(
            categories.query('pedalMovementDuration<999')['pedalMovementDuration'].min(),
            categories.query('pedalMovementDuration<999')['pedalMovementDuration'].max(),
            100
        )
        for idx, pName in enumerate(uniqSizes):
            sizeCatQ = '&'.join([
                '(pedalSizeCat==\'{}\')'.format(pName),
                '(pedalMovementCat==\'outbound\')'
            ])
            ax[idx] = sns.distplot(
                categories.query(sizeCatQ)['pedalMovementDuration'],
                bins=durationBins, kde=False, ax=ax[idx]
                )
            ax[idx].set_title('size {}'.format(pName))
        plt.savefig(
            os.path.join(
                figureFolder, 'movementDurationDistribution.pdf'))
        plt.close()
    #  pull actual electrode names
    categories['electrodeFuzzy'] = np.nan
    for pName in uniqProgs:
        pMask = categories['programFuzzy'] == pName
        if pName == 999:
            categories.loc[pMask, 'electrodeFuzzy'] = 'control'
        else:
            unitName = 'g0p{}'.format(int(pName))
            thisUnit = insBlock.filter(objects=Unit, name=unitName)[0]
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
            categories.loc[pMask, 'electrodeFuzzy'] = elecName

    alignEventsDF = pd.concat([alignTimes, categories], axis=1)
    alignEvents = preproc.eventDataFrameToEvents(
        alignEventsDF, idxT='t',
        annCol=None,
        eventName='seg{}_motionStimAlignTimes'.format(segIdx),
        tUnits=pq.s, makeList=False)
    alignEvents.annotate(nix_name=alignEvents.name)
    #  pdb.set_trace()
    concatLabelColumns = [
        i + 'Fuzzy'
        for i in fuzzyCateg + ['electrode']
        ] + [
            'pedalMovementCat', 'pedalMetaCat',
            'pedalSizeCat', 'pedalDirection',
            'pedalMovementDuration']
    concatLabelsDF = alignEventsDF[concatLabelColumns]
    concatLabels = np.array([
        '{}'.format(row)
        for rowIdx, row in concatLabelsDF.iterrows()])
    concatEvents = Event(
        name='seg{}_motionStimAlignTimesConcatenated'.format(segIdx),
        times=alignEvents.times,
        labels=concatLabels
        )
    newSeg = Segment(name=dataSeg.annotations['neo_name'])
    newSeg.annotate(nix_name=dataSeg.annotations['neo_name'])
    newSeg.events.append(alignEvents)
    newSeg.events.append(concatEvents)
    alignEvents.segment = newSeg
    concatEvents.segment = newSeg
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
