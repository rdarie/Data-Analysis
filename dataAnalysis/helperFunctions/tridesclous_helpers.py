import tridesclous as tdc
import pyqtgraph as pg
import pandas as pd
import numpy as np
#  from matplotlib import pyplot
import time
import os
import glob
import re
import pdb
import traceback
from collections import Iterable, OrderedDict
import quantities as pq
import neo
from neo.core import (
    Block, Segment, ChannelIndex,
    Unit, SpikeTrain)
import shutil
import json
import gc
import dataAnalysis.helperFunctions.helper_functions as hf
from scipy.spatial.distance import minkowski
import scipy.signal


def cmpToDF(arrayFilePath):
    arrayMap = pd.read_csv(
        arrayFilePath, sep='\t',
        skiprows=13)
    cmpDF = pd.DataFrame(
        np.nan, index=range(146),
        columns=[
            'xcoords', 'ycoords', 'elecName',
            'elecID', 'label', 'bank', 'bankID', 'nevID']
        )
    bankLookup = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    
    for rowIdx, row in arrayMap.iterrows():
        #  label matches non-digits index matches digit
        elecSplit = re.match(r'(\D*)(\d*)', row['label']).groups()
        elecName = elecSplit[0]
        elecIdx = int(elecSplit[-1])
        nevIdx = int(row['elec']) + bankLookup[row['bank']] * 32
        cmpDF.loc[nevIdx, 'elecID'] = elecIdx
        cmpDF.loc[nevIdx, 'nevID'] = nevIdx
        cmpDF.loc[nevIdx, 'elecName'] = elecName
        cmpDF.loc[nevIdx, 'xcoords'] = row['row']
        cmpDF.loc[nevIdx, 'ycoords'] = row['//col']
        cmpDF.loc[nevIdx, 'label'] = row['label']
        cmpDF.loc[nevIdx, 'bank'] = row['bank']
        cmpDF.loc[nevIdx, 'bankID'] = int(row['elec'])
    cmpDF.dropna(inplace=True)
    return cmpDF


def cmpDFToPrb(
        cmpDF, filePath=None,
        names=None, banks=None,
        contactSpacing=400,  # units of um
        groupIn=None):

    if names is not None:
        keepMask = cmpDF['elecName'].isin(names)
        cmpDF = cmpDF.loc[keepMask, :]
    if banks is not None:
        keepMask = cmpDF['bank'].isin(banks)
        cmpDF = cmpDF.loc[keepMask, :]
    #  
    cmpDF.reset_index(inplace=True, drop=True)
    prbDict = {}
    
    if groupIn is not None:
        groupingCols = []
        for key, spacing in groupIn.items():
            uniqueValues = np.unique(cmpDF[key])
            bins = int(round(
                (uniqueValues.max() - uniqueValues.min() + 1) / spacing))
            cmpDF[key + '_group'] = np.nan
            cmpDF.loc[:, key + '_group'] = pd.cut(
                cmpDF[key], bins, include_lowest=True, labels=False)
            groupingCols.append(key + '_group')
    else:
        groupingCols = ['elecName']
    #  pdb.set_trace()
    for idx, (name, group) in enumerate(cmpDF.groupby(groupingCols)):
        theseChannels = []
        theseGeoms = {}
        for rName, row in group.iterrows():
            theseChannels.append(int(rName))
            theseGeoms.update({
                int(rName): (
                    contactSpacing * row['xcoords'],
                    contactSpacing * row['ycoords'])})
        prbDict.update({idx: {
            'channels': theseChannels,
            'geometry': theseGeoms
            }})
    """
    tallyChans = []
    tallyGeoms = {}
    for k,v in prbDict.items():
        tallyChans += v['channels']
        tallyGeoms.update(v['geometry'])
    import pdb; pdb.set_trace()
    """
    if filePath is not None:
        with open(filePath, 'w') as f:
            f.write('channel_groups = ' + str(prbDict))
    return prbDict


def cmpDFToPrbAddDummies(
        cmpDF, filePath=None,
        names=None, banks=None,
        contactSpacing=400,  # units of um
        groupIn=None,
        prependDummy=0, appendDummy=0):

    if names is not None:
        keepMask = cmpDF['elecName'].isin(names)
        cmpDF = cmpDF.loc[keepMask, :]
    if banks is not None:
        keepMask = cmpDF['bank'].isin(banks)
        cmpDF = cmpDF.loc[keepMask, :]
    #  import pdb; pdb.set_trace()
    cmpDF.reset_index(inplace=True, drop=True)
    prbDict = {}

    if prependDummy > 0:
        prbDict.update({0: {
            'channels': list(range(prependDummy)),
            'geometry': {
                dummyIdx: (
                    contactSpacing * dummyIdx,
                    contactSpacing * dummyIdx) for dummyIdx in range(prependDummy)}
        }})
        idxOffset = 1
    else:
        idxOffset = 0

    if groupIn is not None:
        groupingCols = []
        for key, spacing in groupIn.items():
            uniqueValues = np.unique(cmpDF[key])
            bins = int(round(len(uniqueValues) / spacing))
            cmpDF[key + '_group'] = np.nan
            cmpDF.loc[:, key + '_group'] = pd.cut(
                cmpDF[key], bins, include_lowest=True, labels=False)
            groupingCols.append(key + '_group')
    else:
        groupingCols = ['elecName']
    for idx, (name, group) in enumerate(cmpDF.groupby(groupingCols)):
        #  import pdb; pdb.set_trace()
        #  group['nevID'].astype(int).values
        prbDict.update({idx + idxOffset: {
            'channels': list(
                group.index +
                int(prependDummy)),
            'geometry': {
                int(rName) + int(prependDummy): (
                    contactSpacing * row['xcoords'],
                    contactSpacing * row['ycoords']) for rName, row in group.iterrows()}
        }})
        lastChan = group.index[-1] + int(prependDummy)

    if appendDummy > 0:
        #  pdb.set_trace()
        appendChanList = list(range(lastChan + 1, lastChan + appendDummy + 1))
        prbDict.update({idx + idxOffset + 1: {
            'channels': appendChanList,
            'geometry': {
                dummyIdx: (
                    contactSpacing * dummyIdx,
                    contactSpacing * dummyIdx) for dummyIdx in appendChanList}
        }})

    if filePath is not None:
        with open(filePath, 'w') as f:
            f.write('channel_groups = ' + str(prbDict))
    return prbDict


def initialize_catalogueconstructor(
        folderPath, fileName,
        triFolder, prbPath,
        name='catalogue_constructor',
        removeExisting=False, fileFormat='NIX'):
    #  set up file source
    if os.path.exists(triFolder) and removeExisting:
        #  remove is already exists
        import shutil
        shutil.rmtree(triFolder)
    if fileFormat == 'NIX':
        filePath = os.path.join(folderPath, fileName + '.nix')
    if fileFormat == 'Blackrock':
        filePath = os.path.join(folderPath, fileName + '.ns5')
    print(filePath)  # check
    dataio = tdc.DataIO(dirname=triFolder)
    dataio.set_data_source(
        type=fileFormat, filenames=[filePath])
    print(dataio)  # check

    #  set up probe file
    dataio.set_probe_file(prbPath)
    for chan_grp in dataio.channel_groups.keys():
        cc = tdc.CatalogueConstructor(
            dataio=dataio, name=name, chan_grp=chan_grp)
        print(cc)
    return


def preprocess_signals_and_peaks(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        chunksize=4096,
        highpass_freq=250.,
        lowpass_freq=5000.,
        filter_order=3,
        smooth_size=5,
        relative_threshold=5.,
        peak_span=1e-3,
        common_ref_removal=True,
        noise_estimate_duration=60.,
        sample_snippet_duration=240.,
        signalpreprocessor_engine='numpy',
        peakdetector_engine='numpy'):

    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    print(dataio)

    cc.set_preprocessor_params(
        chunksize=chunksize,
        signalpreprocessor_engine=signalpreprocessor_engine,
        memory_mode='memmap',
        highpass_freq=highpass_freq,
        lowpass_freq=lowpass_freq,
        filter_order=filter_order,
        smooth_size=smooth_size,
        common_ref_removal=common_ref_removal,
        #  peak detector
        peakdetector_engine=peakdetector_engine,
        #  peakdetector_engine='opencl',
        peak_sign='-',
        relative_threshold=relative_threshold,
        peak_span=peak_span  # units of seconds
        )

    t1 = time.perf_counter()
    cc.estimate_signals_noise(
        seg_num=0, duration=noise_estimate_duration)
    t2 = time.perf_counter()
    print('estimate_signals_noise took {} seconds'.format(t2-t1))
    
    t1 = time.perf_counter()
    cc.run_signalprocessor(
        duration=sample_snippet_duration)
    t2 = time.perf_counter()
    print('run_signalprocessor took {} seconds'.format(t2-t1))
    print(cc)
    return


def extract_waveforms_pca(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        wave_extract_mode='rand',
        n_left=-34, n_right=66, nb_max=90000,
        nb_noise_snippet=2000,
        align_waveform=False,
        subsample_ratio=20,
        feature_method='neighborhood_pca',
        n_components_by_channel=10,
        n_components=10,
        n_components_by_neighborhood=10,
        radius_um=600):

    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)

    if wave_extract_mode == 'all':
        nb_max = None
    t1 = time.perf_counter()
    cc.extract_some_waveforms(
        mode=wave_extract_mode, n_left=n_left, n_right=n_right,
        align_waveform=align_waveform, subsample_ratio=subsample_ratio,
        nb_max=nb_max)
    #  cc.extract_some_waveforms(mode='all', n_left=-45, n_right=60)
    t2 = time.perf_counter()
    print('extract_some_waveforms took {} seconds'.format(t2-t1))
    
    t1 = time.perf_counter()
    cc.clean_waveforms(
        alien_value_threshold=100.)
    t2 = time.perf_counter()
    print('clean_waveforms took {} seconds'.format(t2-t1))

    #  extract_some_noise
    t1 = time.perf_counter()
    cc.extract_some_noise(
        nb_snippet=nb_noise_snippet)
    t2 = time.perf_counter()
    print('extract_some_noise took {} seconds'.format(t2-t1))

    t1 = time.perf_counter()
    featureArgs = {}
    if feature_method == 'pca_by_channel':
        featureArgs.update({
            'n_components_by_channel': n_components_by_channel})
    
    elif feature_method == 'neighborhood_pca':
        featureArgs.update({
            'n_components_by_neighborhood': n_components_by_neighborhood,
            'radius_um': radius_um})
    
    elif feature_method == 'global_pca':
        featureArgs.update({
            'n_components': n_components})
    
    cc.extract_some_features(
        method=feature_method, **featureArgs)
    t2 = time.perf_counter()
    print('project took {} seconds'.format(t2-t1))
    print(cc)
    return


def cluster(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        cluster_method='kmeans',
        n_clusters=100,
        dbscan_eps=0.5,
        autoMerge=False,
        auto_merge_threshold=0.9,
        auto_make_catalog=True):

    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    
    t1 = time.perf_counter()
    clusterArgs = {}
    if cluster_method in ['kmeans', 'gmm', 'agglomerative']:
        clusterArgs.update({'n_clusters': n_clusters})
    if cluster_method == 'dbscan':
        clusterArgs.update({'eps': dbscan_eps})
    cc.find_clusters(
        method=cluster_method, **clusterArgs)
    t2 = time.perf_counter()
    print('find_clusters took {} seconds'.format(t2-t1))

    if autoMerge:
        try:
            print(cc)
            t1 = time.perf_counter()
            #  cc.compute_spike_waveforms_similarity()
            cc.auto_merge_high_similarity(
                threshold=auto_merge_threshold)
            t2 = time.perf_counter()
            print('auto_merge took {} seconds'.format(t2-t1))
        except Exception:
            traceback.print_exc()

    cc.order_clusters(by='waveforms_rms')

    cc.trash_small_cluster(n=25)
    
    print(cc)
    if auto_make_catalog:
        cc.make_catalogue_for_peeler()
    return


def open_cataloguewindow(
        triFolder, chan_grp=0,
        name='catalogue_constructor'):
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    
    app = pg.mkQApp()
    win = tdc.CatalogueWindow(cc)
    win.show()
    
    app.exec_()   
    return 


def clean_catalogue(
        triFolder,
        name='catalogue_constructor', min_nb_peak=10, chan_grp=0):
    #  the catalogue need strong attention with teh catalogue windows.
    #  here a dirty way a cleaning is to take on the first 20 bigger cells
    #  the peeler will only detect them
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    
    #  re order by rms
    cc.order_clusters(by='waveforms_rms')

    #  re label >20 to trash (-1)
    mask = cc.all_peaks['nb_peak'] < min_nb_peak
    cc.all_peaks['cluster_label'][mask] = -1
    cc.on_new_cluster()
    
    #  save catalogue before peeler
    cc.make_catalogue_for_peeler()
    return


def run_peeler(
        triFolder, chan_grp=0,
        shape_distance_threshold=2,
        shape_boundary_threshold=3,
        debugging=False, progressbar=False,
        duration=None, useOpenCL=False, trackTiming=False):

    dataio = tdc.DataIO(dirname=triFolder)
    initial_catalogue = dataio.load_catalogue(chan_grp=chan_grp)
    peeler = tdc.Peeler(dataio)

    if useOpenCL:
        peeler.change_params(
            catalogue=initial_catalogue,
            shape_distance_threshold=shape_distance_threshold,
            debugging=debugging)
    else:
        peeler.change_params(
            catalogue=initial_catalogue,
            shape_distance_threshold=shape_distance_threshold,
            shape_boundary_threshold=shape_boundary_threshold,
            debugging=debugging)

    if trackTiming:
        t1 = time.perf_counter()

    peeler.run(duration=duration, progressbar=progressbar)

    if trackTiming:
        t2 = time.perf_counter()
        print('peeler.run_loop', t2-t1)
    return


def open_PeelerWindow(triFolder, chan_grp=0):
    dataio = tdc.DataIO(dirname=triFolder)
    initial_catalogue = dataio.load_catalogue(chan_grp=chan_grp)

    app = pg.mkQApp()
    win = tdc.PeelerWindow(dataio=dataio, catalogue=initial_catalogue)
    win.show()
    app.exec_()
    return


def export_spikes_after_peeler(triFolder):
    dataio = tdc.DataIO(dirname=triFolder)
    dataio.export_spikes(formats='csv')
    return


def neo_block_after_peeler(
        triFolder, chan_grps=None,
        shape_distance_threshold=None,
        shape_boundary_threshold=None,
        energy_reduction_threshold=None,
        refractory_period=None, ignoreTags=['so_bad'],
        FRThresh=1):
    dataio = tdc.DataIO(dirname=triFolder)

    chanNames = np.array(
        dataio.datasource.get_channel_names())
    #  
    blockName = 'tdc sorted spikes'
    block = Block(name=blockName)

    for segIdx in [0]:
        seg = Segment()
        block.segments.append(seg)

        maxTime = (
            dataio.get_segment_length(segIdx) /
            dataio.sample_rate)

        catalogue = dataio.load_catalogue(chan_grp=chan_grps[0])
        window1 = scipy.signal.triang(2 * int(-catalogue['n_left']) + 1)
        window2 = scipy.signal.triang(2 * int(catalogue['n_right']) + 1)
        window = np.concatenate(
            (
                window1[:int(-catalogue['n_left'])],
                window2[int(catalogue['n_right']) + 1:]),
            axis=-1)
        #  discount edges a lot
        window[window < 0.5] = 0.1
        #  normalize to sum 1, so that the distance is an average
        #  deviation
        distance_window = (window) / np.sum(window)
        boundary_window = window

        for chan_grp in chan_grps:
            #  chan_grp=0
            channelIds = np.array(
                dataio.channel_groups[chan_grp]['channels'])

            catalogue = dataio.load_catalogue(chan_grp=chan_grp)
            
            clustersDF = pd.DataFrame(catalogue['clusters'])
            if not len(clustersDF) > 0:
                # no events from this entire channel
                continue

            clustersDF['max_on_channel_id'] = (
                channelIds[clustersDF['max_on_channel']])
            
            #  choose clusters that aren't tagged as so_bad
            #  note that the trash spikes are excluded automatically
            if len(ignoreTags):
                exportMask = ~clustersDF['tag'].isin(ignoreTags)
                #  iterate over channels
                channelGrouper = (
                    clustersDF.loc[exportMask, :].groupby('max_on_channel_id'))
                #  iterate over units
                unitGrouper = (
                    clustersDF.loc[exportMask, :].groupby('cell_label'))
            else:
                #  iterate over channels
                channelGrouper = (
                    clustersDF.groupby('max_on_channel_id'))
                #  iterate over units
                unitGrouper = (
                    clustersDF.groupby('cell_label'))

            #  keep track of how many units are on each channel
            unitNumberLookup = {}
            for idx, group in channelGrouper:
                unitNumberLookup.update({
                    cl: i for i, cl in (
                        enumerate(pd.unique(group.cluster_label)))
                    }
                )
                chanLabel = chanNames[idx]
                #  create channel indexes
                chanIdx = ChannelIndex(
                    name='{}'.format(chanLabel),
                    index=[idx])
                block.channel_indexes.append(chanIdx)
                #  pdb.set_trace()

            for unitName, group in unitGrouper:
                #  assert group['max_on_channel_id'] only has one element
                chanId = group['max_on_channel_id'].values[0]
                chanLabel = chanNames[chanId]
                chanName = '{}'.format(chanLabel)  # sanitize to string
                chanIdx = block.filter(objects=ChannelIndex, name=chanName)[0]
                #  create unit indexes
                thisUnit = Unit(
                    name='{}#{}'.format(
                        chanName, unitNumberLookup[unitName]))
                print(thisUnit.name)
                #  get spike times
                spike = dataio.get_spikes(
                    seg_num=segIdx, chan_grp=chan_grp)
                unitMask = np.isin(
                    spike['cluster_label'],
                    group['cluster_label'])
                #  discard edges
                edgeMask = (
                    (spike['index'] + catalogue['n_left'] > 0) &
                    (
                        spike['index'] + catalogue['n_right'] <
                        dataio.get_segment_length(segIdx))
                )
                unitMask = unitMask & edgeMask
                try:
                    if not unitMask.any():
                        # no events for this unit
                        raise Exception(
                            '{} has no spikes'.format(thisUnit.name))

                    thisUnitIndices = spike[unitMask]['index']
                    spikeTimes = (
                        (
                            thisUnitIndices +
                            spike[unitMask]['jitter']) /
                        dataio.sample_rate)

                    try:
                        spikeWaveforms = dataio.get_some_waveforms(
                            seg_num=segIdx, chan_grp=chan_grp,
                            spike_indexes=thisUnitIndices,
                            n_left=catalogue['n_left'],
                            n_right=catalogue['n_right']
                            )
                    except Exception:
                        traceback.print_exc()

                    spikeWaveforms = np.swapaxes(
                        spikeWaveforms,
                        1, 2)

                    timesDF = pd.DataFrame(spikeTimes, columns=['times'])
                    timesDF['templateDist'] = np.nan
                    timesDF['maxDeviation'] = np.nan
                    timesDF['energyReduction'] = np.nan
                    meanWaveform = np.mean(spikeWaveforms, axis=0)
                    #  mirror naming convention from tdc
                    pred_wf = meanWaveform
                    norm_factor = 1
                    for idx in timesDF.index:
                        wf = spikeWaveforms[idx, :, :]
                        wf_resid = (wf-pred_wf)
                        normalized_deviation = (
                            np.abs(wf_resid) *
                            boundary_window)
                        normalized_max_deviation = np.max(normalized_deviation)
                        timesDF.loc[idx, 'maxDeviation'] = normalized_max_deviation
                        pred_distance = minkowski(
                            wf / norm_factor,
                            pred_wf / norm_factor,
                            p=1, w=distance_window)
                        timesDF.loc[idx, 'templateDist'] = pred_distance
                        energy_reduction = (
                            (np.sum(wf**2) - np.sum(wf_resid**2)) /
                            wf.shape[0])
                        timesDF.loc[idx, 'energyReduction'] = energy_reduction

                    if shape_boundary_threshold is not None:
                        tooFar = timesDF.index[
                            timesDF['maxDeviation'] > shape_boundary_threshold]
                        #  pdb.set_trace()
                        timesDF.drop(index=tooFar, inplace=True)
                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]
                        timesDF.reset_index(drop=True, inplace=True)

                    if shape_distance_threshold is not None:
                        tooFar = timesDF.index[
                            timesDF['templateDist'] > shape_distance_threshold]
                        #  pdb.set_trace()
                        timesDF.drop(index=tooFar, inplace=True)
                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]
                        timesDF.reset_index(drop=True, inplace=True)

                    timesDF['isi'] = timesDF['times'].diff().fillna(method='bfill')
                    aveSpS = np.nanmedian(timesDF['isi']) ** (-1)
                    #
                    if refractory_period is not None:
                        breaksRefractory = timesDF['isi'] < refractory_period
                        while breaksRefractory.any() and timesDF['times'].any():
                            dropIndices = []
                            for idx in timesDF.loc[breaksRefractory, :].index:
                                thisSpikeDist = timesDF.loc[idx, 'templateDist']
                                #  if the previous spike looks worse,
                                #  we delete that one
                                DFidx = timesDF.index.get_loc(idx)
                                prevIdx = timesDF.index[DFidx - 1]
                                prevSpikeDist = (
                                    timesDF.loc[prevIdx, 'templateDist'])
                                if (thisSpikeDist > prevSpikeDist):
                                    dropIndices.append(idx)
                                else:
                                    dropIndices.append(prevIdx)
                            #  pdb.set_trace()
                            timesDF.drop(
                                index=pd.unique(dropIndices), inplace=True)
                            timesDF['isi'] = timesDF['times'].diff().fillna(method='bfill')
                            breaksRefractory = timesDF['isi'] < refractory_period

                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]

                    if not spikeTimes.any():
                        raise Exception(
                            '{} has no spikes'.format(thisUnit.name))

                    arrayAnn = {
                        'templateDist': timesDF['templateDist'].values,
                        'maxDeviation': timesDF['maxDeviation'].values,
                        'energyReduction': timesDF['energyReduction'].values,
                        'isi': timesDF['isi'].values}
                    arrayAnnNames = {'arrayAnnNames': list(arrayAnn.keys())}
                    #  pdb.set_trace()
                    if group['tag'].iloc[0] == '':
                        if group['max_peak_amplitude'].iloc[0] < -7:
                            group.loc[:, 'tag'] = 'so_good'
                        else:
                            group.loc[:, 'tag'] = 'good'
                    st = SpikeTrain(
                        name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                        times=spikeTimes, units='sec',
                        waveforms=spikeWaveforms * pq.uV,
                        t_stop=maxTime, t_start=0,
                        left_sweep=catalogue['n_left'],
                        sampling_rate=dataio.sample_rate * pq.Hz,
                        array_annotations=arrayAnn,
                        **arrayAnn, **arrayAnnNames)
                    for colName in group.columns:
                        v = group[colName].iloc[0]
                        if v:
                            st.annotations.update(
                                {colName: v}
                            )
                    #  pdb.set_trace()
                    st.annotations.update({'chan_grp': chan_grp})
                except Exception:
                    traceback.print_exc()
                    arrayAnn = {
                        'templateDist': np.array([]),
                        'maxDeviation': np.array([]),
                        'energyReduction': np.array([]),
                        'isi': np.array([])}
                    arrayAnnNames = {'arrayAnnNames': list(arrayAnn.keys())}
                    st = SpikeTrain(
                        name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                        times=np.array([]), units='sec',
                        waveforms=np.array([]).reshape((0, 0, 0))*pq.uV,
                        t_stop=maxTime, t_start=0,
                        left_sweep=catalogue['n_left'],
                        sampling_rate=dataio.sample_rate * pq.Hz,
                        array_annotations=arrayAnn,
                        **arrayAnn, **arrayAnnNames)
                    for colName in group.columns:
                        v = group[colName].iloc[0]
                        if v:
                            st.annotations.update(
                                {colName: v}
                            )
                    st.annotations.update({'chan_grp': chan_grp})

                #  create relationships
                chanIdx.units.append(thisUnit)
                thisUnit.channel_index = chanIdx
                thisUnit.spiketrains.append(st)
                seg.spiketrains.append(st)
                st.unit = thisUnit

            #  end iterating unitGrouper
            seg.create_relationship()
    for chanIdx in block.channel_indexes:
        chanIdx.create_relationship()
    block.create_relationship()

    triName = os.path.basename(os.path.normpath(triFolder))
    writer = neo.io.NixIO(
        filename=os.path.join(triFolder, triName + '.nix'),
        mode='ow')
    writer.write_block(block)
    writer.close()
    return block


def purgeNeoBlock(triFolder):
    trialName = 'tdc_' + triFolder.split('tdc_')[-1]
    for fl in glob.glob(os.path.join(triFolder, trialName + '.nix')):
        os.remove(fl)
    return


def purgePeelerResults(
        triFolder, chan_grps=None, purgeAll=False):
    if not purgeAll:
        assert chan_grps is not None, 'Need to specify chan_grps!'

        for chan_grp in chan_grps:
            #  chan_grp = 0
            grpFolder = 'channel_group_{}'.format(chan_grp)
            segFolder = os.path.join(
                triFolder, grpFolder, 'segment_0')
            shutil.rmtree(segFolder)
            for fl in glob.glob(os.path.join(triFolder, grpFolder, 'nearMiss*.png')):
                os.remove(fl)
            for fl in glob.glob(os.path.join(triFolder, 'templateHist_{}.png'.format(chan_grp))):
                os.remove(fl)
            #  TODO implement selective removal of spikes or processed signs
            """
            arrayInfoPath = os.path.join(segFolder, "arrays.json")
            with open(arrayInfoPath, "r") as f:
                arraysInfo = json.load(f)
            try:
                arraysInfo.pop('spikes')
                os.remove(os.path.join(segFolder, "spikes.raw"))
            except Exception:
                pass
            os.remove(arrayInfoPath)
            with open(arrayInfoPath, "w") as f:
                json.dump(arraysInfo, f)
            """
    else:
        #  purging all
        grpFolders = [
            f
            for f in os.listdir(triFolder)
            if os.path.isdir(os.path.join(triFolder, f))]
        for fl in glob.glob(os.path.join(triFolder, 'templateHist*.png')):
            os.remove(fl)
        for grpFolder in grpFolders:
            try:
                segFolder = os.path.join(
                    triFolder, grpFolder, 'segment_0')
                shutil.rmtree(segFolder)
                for fl in glob.glob(os.path.join(triFolder, grpFolder, 'nearMiss*.png')):
                    os.remove(fl)
            except Exception:
                traceback.print_exc()
    return


def transferTemplates(
        triFolderSource, triFolderDest, chan_grps, removeExisting=True):
    #  triFolderSource = triFolder
    #  triFolderDest = triFolder.replace('3', '1')
    #  chan_grps = chansToAnalyze[:-1]
    #  removeExisting = True
    for chan_grp in chan_grps:
        #  chan_grp = 0
        grpFolderSource = os.path.join(
            triFolderSource, 'channel_group_{}'.format(chan_grp))
        catFolderSource = os.path.join(
            grpFolderSource, 'catalogues', 'initial')
        grpFolderDest = os.path.join(
            triFolderDest, 'channel_group_{}'.format(chan_grp))
        catFolderDest = os.path.join(
            grpFolderDest, 'catalogues', 'initial')
        assert os.path.exists(catFolderSource), 'source catalogue does not exist!'
        assert os.path.exists(grpFolderDest), 'destination folder does not exist!'

        catDoesntExist = not os.path.exists(catFolderDest)
        catExistsButOverride = os.path.exists(catFolderDest) and removeExisting
        if catDoesntExist or catExistsButOverride:
            os.makedirs(catFolderDest, exist_ok=True)
            #  list files in source...
            allSrcFiles = [
                os.path.join(catFolderSource, i)
                for i in os.listdir(catFolderSource)]
            #  and whittle down to files, not subfolders
            srcFiles = [i for i in allSrcFiles if os.path.isfile(i)]
            for fileName in srcFiles:
                shutil.copy(fileName, catFolderDest)
                """
                tdch.run_peeler(
                    triFolderDest, shape_distance_threshold=2e-3,
                    debugging=False,
                    useOpenCL=False, chan_grp=chan_grp)
                """
    return


def batchPreprocess(
        triFolder, chansToAnalyze,
        relative_threshold=5.5,
        peak_span=1e-3,
        cluster_method='kmeans',
        n_clusters=4,
        common_ref_removal=False,
        noise_estimate_duration=120.,
        sample_snippet_duration=240.,
        chunksize=4096, n_left=-34, n_right=66,
        align_waveform=False, subsample_ratio=20,
        autoMerge=False, auto_merge_threshold=0.8,
        n_components_by_channel=15, attemptMPI=False
        ):
    print('Batch preprocessing...')
    try:
        if attemptMPI:
            from mpi4py import MPI
            COMM = MPI.COMM_WORLD
            SIZE = COMM.Get_size()
            RANK = COMM.Get_rank()
        else:
            raise(Exception('MPI aborted by cmd line argument'))
    except Exception:
        RANK = 0
        SIZE = 1

    print('RANK={}, SIZE={}'.format(RANK, SIZE))
    for idx, chan_grp in enumerate(chansToAnalyze):
        if idx % SIZE == RANK:
            print('memory usage: {}'.format(
                hf.memory_usage_psutil()))
            preprocess_signals_and_peaks(
                triFolder, chan_grp=chan_grp,
                chunksize=chunksize,
                signalpreprocessor_engine='numpy',
                peakdetector_engine='numpy',
                highpass_freq=300.,
                lowpass_freq=6000.,
                filter_order=4,
                smooth_size=5,
                relative_threshold=relative_threshold,
                peak_span=peak_span,
                common_ref_removal=common_ref_removal,
                noise_estimate_duration=noise_estimate_duration,
                sample_snippet_duration=sample_snippet_duration)

            successfulDimRed = False
            attemptsCounter = 0
            while (not successfulDimRed) & (attemptsCounter < 3):
                try:
                    extract_waveforms_pca(
                        triFolder, feature_method='pca_by_channel',
                        align_waveform=align_waveform,
                        subsample_ratio=subsample_ratio,
                        n_left=n_left, n_right=n_right,
                        n_components_by_channel=n_components_by_channel,
                        chan_grp=chan_grp)
                    successfulDimRed = True
                except Exception:
                    traceback.print_exc()
                    n_components_by_channel = (
                        int(n_components_by_channel * 0.75))
                    attemptsCounter += 1

            cluster(
                triFolder, cluster_method=cluster_method,
                n_clusters=n_clusters, chan_grp=chan_grp,
                auto_make_catalog=False,
                autoMerge=autoMerge, auto_merge_threshold=auto_merge_threshold)
    return


def batchPeel(
        triFolder, chansToAnalyze,
        shape_distance_threshold=2,
        shape_boundary_threshold=3,
        attemptMPI=False
        ):
    print('Batch peeling...')
    try:
        if attemptMPI:
            from mpi4py import MPI
            COMM = MPI.COMM_WORLD
            SIZE = COMM.Get_size()
            RANK = COMM.Get_rank()
        else:
            raise(Exception('MPI aborted by cmd line argument'))
    except Exception:
        RANK = 0
        SIZE = 1
        
    print('RANK={}, SIZE={}'.format(RANK, SIZE))
    for idx, chan_grp in enumerate(chansToAnalyze):
        if idx % SIZE == RANK:
            print('memory usage: {}'.format(
                hf.memory_usage_psutil()))
            run_peeler(
                triFolder, shape_distance_threshold=shape_distance_threshold,
                shape_boundary_threshold=shape_boundary_threshold,
                debugging=True,
                chan_grp=chan_grp, progressbar=False)
            gc.collect()
    return


if __name__ == '__main__':
    #  initialize_catalogueconstructor(dataPath, chan_grp=0)
    #  preprocess_signals_and_peaks(chan_grp=0)
    #  extract_waveforms_pca_cluster(chan_grp=0)
    #  open_cataloguewindow(chan_grp=0)
    #  run_peeler(chan_grp=0)
    #  open_PeelerWindow(chan_grp=0)
    pass
