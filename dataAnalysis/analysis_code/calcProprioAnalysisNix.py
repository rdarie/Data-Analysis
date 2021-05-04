"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                        which trial to analyze
    --exp=exp                                  which experimental day to analyze
    --verbose                                  print out messages? [default: False]
    --lazy                                     load as neo proxy objects or no? [default: False]
    --analysisName=analysisName                append a name to the resulting blocks? [default: default]
    --sourceFilePrefix=sourceFilePrefix        Does the block have an unusual prefix
    --sourceFileSuffix=sourceFileSuffix        append a name to the resulting blocks?
    --rigFileSuffix=rigFileSuffix              append a name to the resulting blocks?
    --spikeFileSuffix=spikeFileSuffix          append a name to the resulting blocks?
    --spikeSource=spikeSource                  append a name to the resulting blocks?
    --insFilePrefix=insFilePrefix              Does the INS block have an unusual prefix
    --insFileSuffix=insFileSuffix              append a name to the resulting blocks? [default: ins]
    --emgFilePrefix=emgFilePrefix              Does the EMG block have an unusual prefix
    --emgFileSuffix=emgFileSuffix              append a name to the resulting blocks? [default: delsys]
    --hasEMG                                   is there EMG data? [default: False]
    --kinemFilePrefix=kinemFilePrefix          Does the EMG block have an unusual prefix
    --kinemFileSuffix=kinemFileSuffix          append a name to the resulting blocks? [default: simiTrigs]
    --hasKinematics                            is there motion capture data? [default: False]
    --chanQuery=chanQuery                      how to restrict channels if not providing a list? [default: fr]
    --samplingRate=samplingRate                resample the result??
    --rigOnly                                  is there no INS block? [default: False]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from neo.io import NixIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from copy import copy
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.profiling as prf
from namedQueries import namedQueries
import numpy as np
import pandas as pd
import elephant.pandas_bridge as elphpdb
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.ns5 as ns5
import quantities as pq
import rcsanalysis.packet_func as rcsa_helpers
import os, pdb, gc
import traceback
from scipy import signal
from importlib import reload
import json
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import seaborn as sns
#
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)


arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

print('\n' + '#' * 50 + '\n{}\n'.format(__file__) + '#' * 50 + '\n')
binOpts = rasterOpts['binOpts'][arguments['analysisName']]
trackMemory = True


def calcBlockAnalysisWrapper():
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
        samplingRate = float(binOpts['binInterval'] ** -1) * pq.Hz
    #
    if arguments['sourceFileSuffix'] is not None:
        sourceFileSuffix = '_' + arguments['sourceFileSuffix']
    else:
        sourceFileSuffix = ''
    if arguments['spikeFileSuffix'] is not None:
        spikeFileSuffix = '_' + arguments['spikeFileSuffix']
    else:
        spikeFileSuffix = ''
    if arguments['rigFileSuffix'] is not None:
        rigFileSuffix = '_' + arguments['rigFileSuffix']
    else:
        rigFileSuffix = ''
    if arguments['insFileSuffix'] is not None:
        insFileSuffix = '_' + arguments['insFileSuffix']
    else:
        insFileSuffix = ''
    if arguments['hasEMG']:
        if arguments['emgFileSuffix'] is not None:
            emgFileSuffix = '_' + arguments['emgFileSuffix']
        else:
            emgFileSuffix = ''
        if arguments['emgFilePrefix'] is not None:
            emgBlockBaseName = ns5FileName.replace(
                'Block', arguments['emgFilePrefix'])
        else:
            emgBlockBaseName = copy(ns5FileName)
    #
    if arguments['hasKinematics']:
        if arguments['kinemFileSuffix'] is not None:
            kinemFileSuffix = '_' + arguments['kinemFileSuffix']
        else:
            kinemFileSuffix = ''
        if arguments['kinemFilePrefix'] is not None:
            kinemBlockBaseName = ns5FileName.replace(
                'Block', arguments['kinemFilePrefix'])
        else:
            kinemBlockBaseName = copy(ns5FileName)
    #
    #  electrode array name (changes the prefix of the file)
    arrayName = arguments['sourceFilePrefix']
    if arguments['sourceFilePrefix'] is not None:
        blockBaseName = ns5FileName.replace(
            'Block', arguments['sourceFilePrefix'])
    else:
        blockBaseName = copy(ns5FileName)
    if arguments['insFilePrefix'] is not None:
        insBlockBaseName = ns5FileName.replace(
            'Block', arguments['insFilePrefix'])
    else:
        insBlockBaseName = copy(ns5FileName)
    #
    chunkingInfoPath = os.path.join(
        scratchFolder,
        blockBaseName + sourceFileSuffix + '_chunkingInfo.json'
        )
    #
    if os.path.exists(chunkingInfoPath):
        with open(chunkingInfoPath, 'r') as f:
            chunkingMetadata = json.load(f)
    else:
        chunkingMetadata = {
            '0': {
                'filename': os.path.join(
                    scratchFolder, blockBaseName + sourceFileSuffix + '.nix'
                    ),
                'partNameSuffix': '',
                'chunkTStart': 0,
                'chunkTStop': 'NaN'
            }}
    asigBlocks = {}
    spikeBlocks = {}
    eventBlocks = {}
    #
    asigReaders = {}
    spikeReaders = {}
    eventReaders = {}
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        chunkIdx = int(chunkIdxStr)
        nameSuffix = sourceFileSuffix + chunkMeta['partNameSuffix']
        nspPath = os.path.join(
            scratchFolder,
            blockBaseName + nameSuffix + '.nix')
        #####
        if os.path.exists(nspPath):
            print('Loading {}'.format(nspPath))
            asigReader, asigBlock = ns5.blockFromPath(
                nspPath, lazy=arguments['lazy'],
                reduceChannelIndexes=True)
        else:
            raise(Exception('\n{}\nDoes not exist!\n'.format(nspPath)))
        #################
        #  chunked asigBlocks have built-in timestamps
        asigBlock.annotations['chunkTStart'] = 0
        asigBlocks[chunkIdx] = asigBlock
        asigReaders[chunkIdx] = asigReader
        #########################################
        if arguments['spikeSource'] == 'tdc':
            tdcPath = os.path.join(
                scratchFolder,
                'tdc_' + blockBaseName + spikeFileSuffix + chunkMeta['partNameSuffix'],
                'tdc_' + blockBaseName + spikeFileSuffix + chunkMeta['partNameSuffix'] + '.nix'
                )
            print('Loading {}'.format(tdcPath))
            spikeReader, spikeBlock = ns5.blockFromPath(
                tdcPath, lazy=arguments['lazy'],
                reduceChannelIndexes=True)
            # tdc blocks lose their chunking information
            spikeBlock.annotations['chunkTStart'] = chunkMeta['chunkTStart']
            spikeReaders[chunkIdx] = spikeReader
        else:
            spikeBlock = asigBlock
        spikeBlocks[chunkIdx] = spikeBlock
        #####
        eventBlocks[chunkIdx] = asigBlock
        if trackMemory:
            print('Pre-loading chunk {} memory usage: {:.1f} MB'.format(
                idx, prf.memory_usage_psutil()))
    chanQuery = arguments['chanQuery']
    ##############################################################################
    # clip magnitude of signal (y axis)?
    ####
    try:
        clippingOpts = analysisClippingOpts
    except Exception:
        print('Using default clipping opts')
        clippingOpts = {}
    outputBlock = ns5.concatenateBlocks(
        asigBlocks, spikeBlocks, eventBlocks,
        chunkingMetadata, samplingRate, chanQuery,
        arguments['lazy'], trackMemory, arguments['verbose'],
        clipSignals=True, clippingOpts=clippingOpts)
    ##############################################################################
    # close open readers, etc
    for idx, (chunkIdxStr, chunkMeta) in enumerate(chunkingMetadata.items()):
        chunkIdx = int(chunkIdxStr)
        if arguments['lazy']:
            asigReaders[chunkIdx].file.close()
            if arguments['spikeSource'] == 'tdc':
                spikeReaders[chunkIdx].file.close()
        del asigBlocks[chunkIdx]
        if arguments['spikeSource'] == 'tdc':
            spikeBlocks[chunkIdx]
        gc.collect()
        if trackMemory:
            print('Deleting blocks from chunk {} memory usage: {:.1f} MB'.format(
                idx, prf.memory_usage_psutil()))
    del spikeBlocks, asigBlocks
    outputFilePath = os.path.join(
        analysisSubFolder,
        ns5FileName + '_analyze.nix'
        )
    if os.path.exists(outputFilePath):
        os.remove(outputFilePath)
    writer = NixIO(
        filename=outputFilePath, mode='ow')
    writer.write_block(outputBlock, use_obj_names=True)
    writer.close()
    ###############################################
    for asig in outputBlock.filter(objects=AnalogSignal):
        if asig.size > 0:
            dummyOutputAsig = asig
            break
    outputBlockT = pd.Series(dummyOutputAsig.times)
    if len(outputBlock.filter(objects=SpikeTrain)):
        binnedSpikePath = os.path.join(
            analysisSubFolder,
            ns5FileName + '_binarized.nix'
            )
        _ = ns5.calcBinarizedArray(
            outputBlock, samplingRate,
            binnedSpikePath,
            saveToFile=True, matchT=outputBlockT)
    #  ###### load analog inputs
    print('Loading {}'.format(nspPath))
    nspPath = os.path.join(
        scratchFolder,
        blockBaseName + rigFileSuffix + '.nix')
    nspLoadList = {'events': ['seg0_rig_property', 'seg0_rig_value']}
    nspReader, nspBlock = ns5.blockFromPath(
        nspPath, lazy=arguments['lazy'],
        reduceChannelIndexes=True, loadList=nspLoadList)
    nspSeg = nspBlock.segments[0]
    #
    insPath = os.path.join(
        scratchFolder,
        insBlockBaseName + insFileSuffix + '.nix')
    if not os.path.exists(insPath):
        print('No INS data found. Analyzing motion only')
        arguments['rigOnly'] = True
    if not arguments['rigOnly']:
        print('Loading {}'.format(insPath))
        insSignalsToLoad = ([
            'seg0_ins_td{}'.format(tdIdx)
            for tdIdx in range(4)] +
            [
            'seg0_ins_acc{}'.format(accIdx)
            for accIdx in ['x', 'y', 'z', 'inertia']
            ])
        insEventsToLoad = [
            'seg0_ins_property',
            'seg0_ins_value'
            ]
        insSpikesToLoad = ['seg0_g0p0#0']
        insLoadList = {
            'asigs': insSignalsToLoad,
            'events': insEventsToLoad,
            'spiketrains': insSpikesToLoad
            }
        insReader, insBlock = ns5.blockFromPath(
            insPath, lazy=arguments['lazy'],
            reduceChannelIndexes=True,
            loadList=insLoadList)
        insSeg = insBlock.segments[0]
        # convert stim updates to time series
        ins_events = [
            ev for ev in insBlock.filter(objects=Event)
            if ev.name in ['seg0_ins_property', 'seg0_ins_value']]
        if len(ins_events):
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
            infoFromStimStatus = hf.interpolateDF(
                stimStatus, outputBlockT,
                x='t', columns=columnsToBeAdded, kind='previous')
            infoFromStimStatus.set_index('t', inplace=True)
        else:
            infoFromStimStatus = None
    else:
        infoFromStimStatus = None
        insBlock = None
    # insBlock = nspBlock
    ######
    #  synchronize INS
    ######
    evList = []
    for key in ['property', 'value']:
        if not arguments['rigOnly']:
            insPropList = insBlock.filter(
                objects=Event,
                name='seg0_ins_' + key
                )
            if len(insPropList):
                insProp = insPropList[0]
            else:
                print(
                    'INS properties not found! analyzing rig events only.')
                arguments['rigOnly'] = True
        else:
            insPropList = []
            insProp = None
        rigPropList = nspBlock.filter(
            objects=Event,
            name='seg0_rig_' + key
            )
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
        elif not arguments['rigOnly']:
            #  RC's don't have rig_events
            if insProp is not None:
                allProp = insProp
                allProp.name = 'seg0_' + key
                evList.append(allProp)
    if len(evList):
        #  make concatenated event, for viewing
        concatLabels = np.array([
            (elphpdb._convert_value_safe(evList[0].labels[i]) + ': ' +
                elphpdb._convert_value_safe(evList[1].labels[i])) for
            i in range(len(evList[0]))
            ])
        concatEvent = Event(
            name='seg0_concatenated_updates',
            times=allProp.times,
            labels=concatLabels
            )
        concatEvent.merge_annotations(allProp)
        evList.append(concatEvent)
    rigChanQuery = '(chanName.notna())'
    alreadyThereNames = [asi.name for asi in outputBlock.filter(objects=AnalogSignal)]
    if arguments['lazy']:
        rigChanNames = ns5.listChanNames(
            nspBlock, rigChanQuery, objType=AnalogSignalProxy)
        rigChanNames = [rcn for rcn in rigChanNames if rcn not in alreadyThereNames]
        asigList = []
        for asigP in nspSeg.analogsignals:
            if asigP.name in rigChanNames:
                asig = asigP.load()
                asig.channel_index = asigP.channel_index
                asigList.append(asig)
                if trackMemory:
                    print('loading {} from proxy object. memory usage: {:.1f} MB'.format(
                        asigP.name, prf.memory_usage_psutil()))
    else:
        rigChanNames = ns5.listChanNames(
            asigBlock, rigChanQuery, objType=AnalogSignal)
        rigChanNames = [rcn for rcn in rigChanNames if rcn not in alreadyThereNames]
        asigList = [
            asig
            for asig in nspSeg.analogsignals
            if asig.name in rigChanNames
            ]
    for asig in asigList:
        if asig.size > 0:
            dummyRigAsig = asig
            break
    tdDF = ns5.analogSignalsToDataFrame(asigList)
    del asigList
    tdDF.loc[:, 't'] += asigBlock.annotations['chunkTStart']
    # origTimeStep = tdDF['t'].iloc[1] - tdDF['t'].iloc[0]
    tdDF.set_index('t', inplace=True)
    descriptiveNames = pd.Series(eventInfo['inputIDs']).reset_index()
    descriptiveNames.columns = ['newName', 'ainpName']
    renamingDict = {
        'seg0_' + row['ainpName']: 'seg0_' + row['newName']
        for rowIdx, row in descriptiveNames.iterrows()}
    tdDF.rename(columns=renamingDict, inplace=True)
    signalsForAbsoluteValue = []
    if 'seg0_forceX' in tdDF.columns:
        signalsForAbsoluteValue.append('seg0_forceX')
        signalsForAbsoluteValue.append('seg0_forceY')
        tdDF.loc[:, 'seg0_forceMagnitude'] = np.sqrt(
            tdDF['seg0_forceX'].astype(float) ** 2 +
            tdDF['seg0_forceY'].astype(float) ** 2)
        signalsForDerivative = [
            'seg0_forceX', 'seg0_forceY', 'seg0_forceMagnitude']
        for cName in signalsForDerivative:
            if cName in tdDF.columns:
                tdDF.loc[:, cName + '_prime'] = hf.applySavGol(
                    tdDF[cName],
                    window_length_sec=30e-3,
                    fs=int(dummyRigAsig.sampling_rate),
                    polyorder=3, deriv=1)
    # interpolate rig analog signals
    filterOptsPerCategory = {
        'forceSensor': {
            'names': [
                'seg0_forceX', 'seg0_forceY', 'seg0_forceMagnitude',
                'seg0_forceX_prime', 'seg0_forceY_prime', 'seg0_forceMagnitude_prime'],
            'filterOpts': {
                'low': {
                    'Wn': float(samplingRate) / 3,
                    'N': 4,
                    'btype': 'low',
                    'ftype': 'bessel'
                },
                'bandstop60Hz': {
                    'Wn': 60,
                    'nHarmonics': 2,
                    'Q': 10,
                    'N': 4,
                    'rp': 1,
                    'btype': 'bandstop',
                    'ftype': 'cheby1'
                },
                'bandstop85Hz': {
                    'Wn': 85,
                    'nHarmonics': 2,
                    'Q': 10,
                    'N': 4,
                    'rp': 1,
                    'btype': 'bandstop',
                    'ftype': 'cheby1'
                }
            }
        },
        'other': {
            'names': [],
            'filterOpts': {
                'low': {
                    'Wn': float(samplingRate) / 3,
                    'N': 4,
                    'btype': 'low',
                    'ftype': 'bessel'
                }
            }
        }
    }
    lowPassOpts = filterOptsPerCategory['other']['filterOpts']
    columnInfo = tdDF.columns.to_frame(name='feature')
    columnInfo.loc[:, 'featureGroup'] = np.nan
    for featureName, _ in columnInfo.iterrows():
        for categName, filterData in filterOptsPerCategory.items():
            if featureName in filterData['names']:
                columnInfo.loc[featureName, 'featureGroup'] = categName
    columnInfo.loc[:, 'featureGroup'] = columnInfo['featureGroup'].fillna('other')
    for groupName, group in columnInfo.groupby('featureGroup'):
        fOpts = filterOptsPerCategory[groupName]['filterOpts'].copy()
        filterCoeffs = hf.makeFilterCoeffsSOS(
            fOpts, float(dummyRigAsig.sampling_rate))
        if trackMemory:
            print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                prf.memory_usage_psutil()))
        filteredAsigs = signal.sosfiltfilt(
            filterCoeffs, tdDF.loc[:, group.index].to_numpy(),
            axis=0)
        pdb.set_trace()
        if True:
            plotCName = 'seg0_forceY'
            plotCNameIdx = group.index.get_loc(plotCName)
            fig, ax = plt.subplots()
            idx1, idx2 = int(6e4), int(9e4)
            ax.plot(
                tdDF.index[idx1:idx2], filteredAsigs[idx1:idx2, plotCNameIdx],
                label='filtered')
            ax.plot(
                tdDF.index[idx1:idx2], tdDF[plotCName].iloc[idx1:idx2],
                label='original')
            ax.set_title(plotCName)
            ax.legend()
            plt.show()
        tdDF.loc[:, group.index] = filteredAsigs
        if trackMemory:
            print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                prf.memory_usage_psutil()))
    if samplingRate != dummyRigAsig.sampling_rate:
        tdInterp = hf.interpolateDF(
            tdDF, outputBlockT,
            kind='linear', fill_value=(0, 0),
            verbose=arguments['verbose'])
        # free up memory used by full resolution asigs
        del tdDF
    else:
        tdInterp = tdDF
    # add analog traces derived from position
    if 'seg0_position' in tdInterp.columns:
        signalsForAbsoluteValue.append('seg0_velocity')
        tdInterp.loc[:, 'seg0_position_x'] = (
            np.cos(np.radians(tdInterp['seg0_position'].astype(float) * 100)))
        tdInterp.loc[:, 'seg0_position_y'] = (
            np.sin(np.radians(tdInterp['seg0_position'].astype(float) * 100)))
        tdInterp.sort_index(axis='columns', inplace=True)
        tdInterp.loc[:, 'seg0_velocity_x'] = (
            tdInterp[ 'seg0_position_y'] * (-1) * (tdInterp['seg0_velocity'].astype(float) * 3e2))
        tdInterp.loc[:, 'seg0_velocity_y'] = (
            tdInterp[ 'seg0_position_x'] * (tdInterp['seg0_velocity'].astype(float) * 3e2))
    concatList = [tdInterp]
    #
    if not arguments['rigOnly']:
        insAsigList = [
            asig
            for asig in insSeg.analogsignals
            if asig.name in insLoadList['asigs']
            ]
        for asig in insAsigList:
            if asig.size > 0:
                dummyInsAsig = asig
                break
        insDF = ns5.analogSignalsToDataFrame(insAsigList)
        # origInsTimeStep = insDF['t'].iloc[1] - insDF['t'].iloc[0]
        insDF.set_index('t', inplace=True)
        # interpolate INS analog signals
        if samplingRate != dummyInsAsig.sampling_rate:
            if samplingRate < dummyInsAsig.sampling_rate:
                filterCoeffs = hf.makeFilterCoeffsSOS(
                    lowPassOpts.copy(), float(dummyInsAsig.sampling_rate))
                if trackMemory:
                    print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                        prf.memory_usage_psutil()))
                filteredAsigs = signal.sosfiltfilt(
                    filterCoeffs, insDF.to_numpy(),
                    axis=0)
                insDF = pd.DataFrame(
                    filteredAsigs,
                    index=insDF.index,
                    columns=insDF.columns)
                if trackMemory:
                    print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                        prf.memory_usage_psutil()))
            insInterp = hf.interpolateDF(
                insDF, outputBlockT,
                kind='linear', fill_value=(0, 0),
                verbose=arguments['verbose'])
            # free up memory used by full resolution asigs
            del insDF
        else:
            insInterp = insDF
        concatList.append(insInterp)
        concatList.append(infoFromStimStatus)
    #
    #
    if arguments['hasKinematics']:
        kinemPath = os.path.join(
            scratchFolder,
            kinemBlockBaseName + kinemFileSuffix + '.nix')
        kinemReader, kinemBlock = ns5.blockFromPath(
            kinemPath, lazy=False,
            reduceChannelIndexes=True)
        kinemSeg = kinemBlock.segments[0]
        kinemAsigList = [
            asig
            for asig in kinemSeg.analogsignals
            if ('_angle' in asig.name)
            ]
        for asig in kinemAsigList:
            if asig.size > 0:
                dummyKinemAsig = asig
                break
        kinemDF = ns5.analogSignalsToDataFrame(kinemAsigList)
        kinemDF.set_index('t', inplace=True)
        # interpolate kinematic analog signals
        kinemCols = [cn for cn in kinemDF.columns if '_angle' in cn]
        for cName in kinemCols:
            signalsForAbsoluteValue.append(cName.replace('_angle', '_omega'))
            kinemDF.loc[:, cName.replace('_angle', '_omega')] = hf.applySavGol(
                kinemDF[cName],
                window_length_sec=30e-3,
                fs=int(dummyKinemAsig.sampling_rate),
                polyorder=3, deriv=1)
        filterOptsKinem = {
            'low': {
                'Wn': 33,
                'N': 4,
                'btype': 'low',
                'ftype': 'bessel'
            }
        }
        filterCoeffsKinem = hf.makeFilterCoeffsSOS(
            filterOptsKinem, float(dummyKinemAsig.sampling_rate))
        filteredKinem = signal.sosfiltfilt(
            filterCoeffsKinem, kinemDF.loc[:, kinemCols].to_numpy(),
            axis=0)
        #
        if samplingRate != dummyKinemAsig.sampling_rate:
            if samplingRate < dummyKinemAsig.sampling_rate:
                filterCoeffs = hf.makeFilterCoeffsSOS(
                    lowPassOpts.copy(), float(dummyKinemAsig.sampling_rate))
                if trackMemory:
                    print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                        prf.memory_usage_psutil()))
                filteredAsigs = signal.sosfiltfilt(
                    filterCoeffs, kinemDF.to_numpy(),
                    axis=0)
                kinemDF = pd.DataFrame(
                    filteredAsigs,
                    index=kinemDF.index,
                    columns=kinemDF.columns)
                if trackMemory:
                    print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                        prf.memory_usage_psutil()))
            kinemInterp = hf.interpolateDF(
                kinemDF, outputBlockT,
                kind='linear', fill_value=(0, 0),
                verbose=arguments['verbose'])
            # free up memory used by full resolution asigs
            del kinemDF
        else:
            kinemInterp = kinemDF
        concatList.append(kinemInterp)
    #
    if arguments['hasEMG']:
        emgPath = os.path.join(
            scratchFolder,
            emgBlockBaseName + emgFileSuffix + '.nix')
        emgReader, emgBlock = ns5.blockFromPath(
            emgPath, lazy=False,
            reduceChannelIndexes=True)
        emgSeg = emgBlock.segments[0]
        emgAsigList = [
            asig
            for asig in emgSeg.analogsignals
            if ('Acc' in asig.name) or ('Emg' in asig.name)
            ]
        for asig in emgAsigList:
            if asig.size > 0:
                dummyEmgAsig = asig
                break
        emgDF = ns5.analogSignalsToDataFrame(emgAsigList)
        emgDF.set_index('t', inplace=True)
        # interpolate emg analog signals
        emgCols = [cn for cn in emgDF.columns if 'Emg' in cn]
        # accCols = [cn for cn in emgDF.columns if 'Acc' in cn]
        highPassOpts = {
            'high': {
                'Wn': .1,
                'N': 4,
                'btype': 'high',
                'ftype': 'bessel'
            }
        }
        lowPassOptsEMG = {
            'low': {
                'Wn': 100,
                'N': 4,
                'btype': 'low',
                'ftype': 'bessel'
            }
        }
        filterCoeffsHP = hf.makeFilterCoeffsSOS(
            highPassOpts, float(dummyEmgAsig.sampling_rate))
        filterCoeffsLPEMG = hf.makeFilterCoeffsSOS(
            lowPassOptsEMG, float(dummyEmgAsig.sampling_rate))
        filteredEMG = signal.sosfiltfilt(
            filterCoeffsHP, emgDF.loc[:, emgCols].to_numpy(),
            axis=0)
        filteredEMG = np.abs(filteredEMG)
        filteredEMG = signal.sosfiltfilt(
            filterCoeffsLPEMG, filteredEMG,
            axis=0)
        emgEnvColumns = [eN.replace('Emg', 'EmgEnv') for eN in emgCols]
        emgDF.loc[:, emgEnvColumns] = filteredEMG
        #
        if samplingRate != dummyEmgAsig.sampling_rate:
            if samplingRate < dummyEmgAsig.sampling_rate:
                filterCoeffs = hf.makeFilterCoeffsSOS(
                    lowPassOpts.copy(), float(dummyEmgAsig.sampling_rate))
                if trackMemory:
                    print('Filtering analog data before downsampling. memory usage: {:.1f} MB'.format(
                        prf.memory_usage_psutil()))
                filteredAsigs = signal.sosfiltfilt(
                    filterCoeffs, emgDF.to_numpy(),
                    axis=0)
                emgDF = pd.DataFrame(
                    filteredAsigs,
                    index=emgDF.index,
                    columns=emgDF.columns)
                if trackMemory:
                    print('Just finished analog data filtering before downsampling. memory usage: {:.1f} MB'.format(
                        prf.memory_usage_psutil()))
            emgInterp = hf.interpolateDF(
                emgDF, outputBlockT,
                kind='linear', fill_value=(0, 0),
                verbose=arguments['verbose'])
            # free up memory used by full resolution asigs
            del emgDF
        else:
            emgInterp = emgDF
        concatList.append(emgInterp)
    if len(concatList) > 1:
        tdInterp = pd.concat(
            concatList, axis=1)
    for cName in signalsForAbsoluteValue:
        if cName in tdInterp.columns:
            tdInterp.loc[:, cName + '_abs'] = tdInterp[cName].abs()
    tdInterp.columns = [cN.replace('seg0_', '') for cN in tdInterp.columns]
    tdInterp.sort_index(axis='columns', inplace=True)
    tdBlockInterp = ns5.dataFrameToAnalogSignals(
        tdInterp,
        idxT=None, useColNames=True,
        dataCol=tdInterp.columns,
        samplingRate=samplingRate)
    #
    tdBlockInterp.segments[0].events = evList
    for ev in evList:
        ev.segment = tdBlockInterp.segments[0]
    # [ev.name for ev in evList]
    ns5.addBlockToNIX(
        tdBlockInterp, neoSegIdx=[0],
        writeSpikes=False, writeEvents=True,
        fileName=ns5FileName + '_analyze',
        folderPath=analysisSubFolder,
        purgeNixNames=True,
        nixBlockIdx=0, nixSegIdx=[0],
        )
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
            topFun=calcBlockAnalysisWrapper,
            modulesToProfile=[ash, ns5, hf],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        calcBlockAnalysisWrapper()