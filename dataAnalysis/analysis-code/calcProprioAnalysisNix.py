"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze
    --exp=exp                              which experimental day to analyze
    --verbose                              print out messages? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --inputBlockSuffix=inputBlockSuffix    append a name to the resulting blocks?
    --chanQuery=chanQuery                  how to restrict channels if not providing a list? [default: fr]
    --samplingRate=samplingRate            resample the result??
    --rigOnly                              is there no INS block? [default: False]
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
from namedQueries import namedQueries
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.ns5 as ns5
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb
import traceback
from importlib import reload
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
binOpts = rasterOpts['binOpts'][arguments['analysisName']]

if arguments['inputBlockSuffix'] is None:
    arguments['inputBlockSuffix'] = ''

def calcBlockAnalysisNix():
    arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
        namedQueries, scratchFolder, **arguments)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
        )
    if not os.path.exists(analysisSubFolder):
        os.makedirs(analysisSubFolder, exist_ok=True)
    if arguments['samplingRate'] is not None:
        samplingRate = float(arguments['samplingRate']) * pq.Hz
    else:
        samplingRate = float(1 / binOpts['binInterval']) * pq.Hz
    #
    nspPath = os.path.join(
        scratchFolder,
        ns5FileName + arguments['inputBlockSuffix'] + '.nix')
    # pdb.set_trace()
    if not os.path.exists(nspPath):
        fallBackPath = os.path.join(
            scratchFolder,
            (
                ns5FileName.replace('Block', 'utah') +
                arguments['inputBlockSuffix'] +
                '.nix'))
        print('{} not found;\nFalling back to {}'.format(
            nspPath, fallBackPath
        ))
        if os.path.exists(fallBackPath):
            nspPath = fallBackPath
    nspReader = neo.io.nixio_fr.NixIO(
        filename=nspPath)
    nspBlock = ns5.readBlockFixNames(
        nspReader, reduceChannelIndexes=True, block_index=0)
    print('Loading {}'.format(nspPath))
    # print([cI.name for cI in nspBlock.channel_indexes])
    # print([asig.name for asig in spikesBlock.filter(objects=AnalogSignal)])
    # print([st.name for st in nspBlock.filter(objects=SpikeTrain)])
    # print([st.name for st in nspBlock.filter(objects=ChannelIndex)]) len(nspBlock.filter(objects=ChannelIndex))
    # print([ev.name for ev in nspBlock.filter(objects=Event)])
    spikesBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSpikes=True)
    spikesBlock = hf.loadBlockProxyObjects(spikesBlock)
    # print([cI.name for cI in spikesBlock.channel_indexes])
    #  save ins time series
    tdChanNames = ns5.listChanNames(
        nspBlock, arguments['chanQuery'], objType=AnalogSignalProxy)
    allSpikeTrains = [
        i
        for i in spikesBlock.filter(objects=SpikeTrain)
        if '#' in i.name]
    if len(allSpikeTrains):
        for segIdx, dataSeg in enumerate(spikesBlock.segments):
            spikeList = dataSeg.filter(objects=SpikeTrain)
            spikeList = ns5.loadContainerArrayAnn(trainList=spikeList)
    #  parse any serial events
    forceData = hf.parseFSE103Events(
        spikesBlock, delay=9e-3, clipLimit=1e9, formatForce='f')
    #  merge events
    # pdb.set_trace()
    evList = []
    for key in ['property', 'value']:
        #  key = 'property'
        insPropList = spikesBlock.filter(
            objects=Event,
            name='seg0_ins_' + key
            )
        rigPropList = spikesBlock.filter(
            objects=Event,
            name='seg0_rig_' + key
            )
        if len(insPropList):
            insProp = insPropList[0]
        if len(rigPropList):
            rigProp = rigPropList[0]
            if arguments['rigOnly']:
                allProp = rigProp
                allProp.name = 'seg0_' + key
                evList.append(allProp)
            else:
                allProp = insProp.merge(rigProp)
                allProp.name = 'seg0_' + key
                evSortIdx = np.argsort(allProp.times, kind='mergesort')
                allProp = allProp[evSortIdx]
                evList.append(allProp)
        else:
            #  RC's don't have rig_events
            allProp = insProp
            allProp.name = 'seg0_' + key
            evList.append(allProp)
    #  make concatenated event, for viewing
    concatLabels = np.array([
        (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
            elphpdb._convert_value_safe(evList[1].labels[i])) for
        i in range(len(evList[0]))
        ])
    concatEvent = Event(
        name='seg0_' + 'concatenated_updates',
        times=allProp.times,
        labels=concatLabels
        )
    concatEvent.merge_annotations(allProp)
    evList.append(concatEvent)
    spikesBlock.segments[0].events = evList
    for ev in evList:
        ev.segment = spikesBlock.segments[0]
    # print([asig.name for asig in spikesBlock.filter(objects=AnalogSignal)])
    # print([st.name for st in spikesBlock.filter(objects=SpikeTrain)])
    # print([st.name for st in spikesBlock.filter(objects=ChannelIndex)]) len(spikesBlock.filter(objects=ChannelIndex))
    # print([ev.name for ev in spikesBlock.filter(objects=Event)])
    spikesBlock = ns5.purgeNixAnn(spikesBlock)
    # pdb.set_trace()
    #
    tdBlock = hf.extractSignalsFromBlock(
        nspBlock, keepSpikes=False, keepSignals=tdChanNames)
    tdBlock = hf.loadBlockProxyObjects(tdBlock)
    tdDF = ns5.analogSignalsToDataFrame(
        tdBlock.filter(objects=AnalogSignal))
    #
    if not arguments['rigOnly']:
        ins_events = [
            i for i in tdBlock.filter(objects=Event)
            if 'ins_' in i.name]
        expandCols = [
                'RateInHz', 'therapyStatus',
                'activeGroup', 'program', 'trialSegment']
        deriveCols = ['amplitudeRound', 'amplitude']
        progAmpNames = rcsa_helpers.progAmpNames
        #
        stimStSer = ns5.eventsToDataFrame(
            ins_events, idxT='t')
        stimStatus = mdt.stimStatusSerialtoLong(
            stimStSer, idxT='t',  namePrefix='seg0_ins_',
            expandCols=expandCols,
            deriveCols=deriveCols, progAmpNames=progAmpNames)
        columnsToBeAdded = ['amplitude', 'program', 'RateInHz'] + progAmpNames
    # calc binarized and get new time axis
    if len(allSpikeTrains):
        spikeMatBlock = ns5.calcBinarizedArray(
            spikesBlock, samplingRate,
            binnedSpikePath.format(arguments['analysisName']),
            saveToFile=True)
        newT = pd.Series(
            spikeMatBlock.filter(objects=AnalogSignal)[0].times.magnitude)
    else:
        dummyT = nspBlock.filter(objects=AnalogSignalProxy)[0]
        newT = pd.Series(
            np.arange(
                dummyT.t_start, dummyT.t_stop + 1/samplingRate, 1/samplingRate))
    #
    if 'seg0_position' in tdDF.columns:
        tdDF.loc[:, 'seg0_position_x'] = ((
            np.cos(
                tdDF.loc[:, 'seg0_position'] *
                100 * 2 * np.pi / 360))
            .to_numpy())
        tdDF.sort_index(axis='columns', inplace=True)
        tdDF.loc[:, 'seg0_position_y'] = ((
            np.sin(
                tdDF.loc[:, 'seg0_position'] *
                100 * 2 * np.pi / 360))
            .to_numpy())
        tdDF.loc[:, 'seg0_velocity_x'] = ((
            tdDF.loc[:, 'seg0_position_y'] *
            (-1) *
            (tdDF.loc[:, 'seg0_velocity'] * 3e2))
            .to_numpy())
        tdDF.loc[:, 'seg0_velocity_y'] = ((
            tdDF.loc[:, 'seg0_position_x'] *
            (tdDF.loc[:, 'seg0_velocity'] * 3e2))
            .to_numpy())
        tdChanNames += [
            'seg0_position_x', 'seg0_position_y',
            'seg0_velocity_x', 'seg0_velocity_y']
    origTimeStep = tdDF['t'].iloc[1] - tdDF['t'].iloc[0]
    # smooth by simi fps
    simiFps = 100
    smoothWindowStd = int(1 / (origTimeStep * simiFps * 2))
    #
    debugVelCalc = False
    if debugVelCalc:
        import matplotlib.pyplot as plt
    #
    for cName in tdDF.columns:
        if '_angle' in cName:
            if debugVelCalc:
                pdb.set_trace()
            thisVelocity = (
                tdDF.loc[:, cName].diff()
                .rolling(6 * smoothWindowStd, center=True, win_type='gaussian')
                .mean(std=smoothWindowStd).fillna(0) / origTimeStep)
            thisAcceleration = (
                thisVelocity.diff()
                .rolling(6 * smoothWindowStd, center=True, win_type='gaussian')
                .mean(std=smoothWindowStd).fillna(0) / origTimeStep)
            if debugVelCalc:
                plt.plot(thisVelocity.iloc[1000000:1030000])
                plt.plot(tdDF.loc[:, cName].iloc[1000000:1030000].diff() / origTimeStep)
                plt.show()
            tdDF.loc[:, cName.replace('_angle', '_angular_velocity')] = (
                thisVelocity)
            tdDF.loc[:, cName.replace('_angle', '_angular_acceleration')] = (
                thisAcceleration)
            tdChanNames += [
                cName.replace('_angle', '_angular_velocity'),
                cName.replace('_angle', '_angular_acceleration')
                ]
    #
    if samplingRate != tdBlock.filter(objects=AnalogSignal)[0].sampling_rate:
        print('Interpolating input!')
        # pdb.set_trace()
        tdInterp = hf.interpolateDF(
            tdDF, newT,
            kind='linear', fill_value=(0, 0),
            x='t', columns=tdChanNames, verbose=arguments['verbose'])
    else:
        print('Using input as is!')
        tdInterp = tdDF
    #
    concatList = [tdInterp]
    if not arguments['rigOnly']:
        infoFromStimStatus = hf.interpolateDF(
            stimStatus, tdInterp['t'],
            x='t', columns=columnsToBeAdded, kind='previous')
        concatList.append(infoFromStimStatus.drop(columns='t'))
    if forceData is not None:
        forceDataInterp = hf.interpolateDF(
            forceData, newT,
            kind='linear', fill_value=(0, 0),
            x='NSP Timestamp', columns=['forceX', 'forceY', 'forceZ'])
        concatList.append(forceDataInterp.drop(columns='NSP Timestamp'))
    if len(concatList) > 1:
        tdInterp = pd.concat(
            concatList,
            axis=1)
    #
    tdInterp.columns = [i.replace('seg0_', '') for i in tdInterp.columns]
    if not arguments['rigOnly']:
        tdInterp.loc[:, 'RateInHz'] = (
            tdInterp.loc[:, 'RateInHz'] *
            (tdInterp.loc[:, 'amplitude'].abs() > 0))
        for pName in progAmpNames:
            if pName in tdInterp.columns:
                tdInterp.loc[:, pName.replace('amplitude', 'ACR')] = (
                    tdInterp.loc[:, pName] *
                    tdInterp.loc[:, 'RateInHz'])
                tdInterp.loc[:, pName.replace('amplitude', 'dAmpDt')] = (
                    tdInterp.loc[:, pName].diff()
                    .rolling(6 * smoothWindowStd, center=True, win_type='gaussian')
                    .mean(std=smoothWindowStd).fillna(0) / origTimeStep)
    tdInterp.sort_index(axis='columns', inplace=True)
    # tdInterp.columns = ['seg0_{}'.format(i) for i in tdInterp.columns]
    tdBlockInterp = ns5.dataFrameToAnalogSignals(
        tdInterp,
        idxT='t', useColNames=True,
        dataCol=tdInterp.drop(columns='t').columns,
        samplingRate=samplingRate)
    tdBlockInterp.name = spikesBlock.name
    tdBlockInterp.segments[0].name = spikesBlock.segments[0].name
    spikesBlock.merge(tdBlockInterp)
    # len(spikesBlock.segments)
    # del spikesBlock.segments[-1]
    #
    # for cI in spikesBlock.channel_indexes:
    #     # e.g. cI = spikesBlock.channel_indexes[0]
    #     for asig in cI.analogsignals:
    #         # e.g. asig = cI.analogsignals[0]
    #         # print('Assigning {}'.format(asig.name))
    #         asig.channel_index = cI
    #         asig.segment = spikesBlock.segments[0]
    #         spikesBlock.segments[0].analogsignals.append(asig)
    spikesBlock.create_relationship(force=True)
    # spikesBlock.channel_indexes[0].analogsignals[0]
    # pd.unique([id(asi) for asi in spikesBlock.filter(objects=AnalogSignal)])
    analysisBlockPath = analysisDataPath.format(arguments['analysisName'])
    if os.path.exists(analysisBlockPath):
        os.remove(analysisBlockPath)
    # pdb.set_trace()
    writer = neo.io.NixIO(filename=analysisBlockPath)
    writer.write_block(spikesBlock, use_obj_names=True)
    writer.close()
    # ns5.addBlockToNIX(
    #     tdBlockInterp, neoSegIdx=[0],
    #     writeSpikes=False, writeEvents=False,
    #     purgeNixNames=False,
    #     fileName=ns5FileName + '_analyze',
    #     folderPath=analysisSubFolder,
    #     nixBlockIdx=0, nixSegIdx=[0],
    #     )
    return


if __name__ == "__main__":
    runProfiler = False
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        if arguments['lazy']:
            nameSuffix = 'lazy'
        else:
            nameSuffix = 'not_lazy'
        prf.profileFunction(
            topFun=calcBlockAnalysisNix,
            modulesToProfile=[ash, ns5, hf],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        calcBlockAnalysisNix()