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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import gc
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
from scipy.spatial.distance import minkowski
import scipy.signal
import dill as pickle


def initialize_catalogueconstructor(
        folderPath, fileName,
        triFolder, prbPath=None,
        spikeSortingOpts=None,
        name='catalogue_constructor',
        removeExisting=False, fileFormat='NIX'):
    #  set up file source
    if os.path.exists(triFolder) and removeExisting:
        #  remove if already exists
        import shutil
        shutil.rmtree(triFolder)
    if fileFormat == 'NIX':
        filePath = os.path.join(folderPath, fileName + '.nix')
    if fileFormat == 'Blackrock':
        filePath = os.path.join(folderPath, fileName + '.ns5')
    print('dataIO: ')
    dataio = tdc.DataIO(dirname=triFolder)
    print('using file: {}'.format(filePath))  # check
    dataio.set_data_source(
        type=fileFormat, filenames=[filePath])
    #
    print('initialize_catalogueconstructor DataIO is: ')
    print(dataio)  # check
    #  set up probe file
    if prbPath is None:
        electrodeMapPath = spikeSortingOpts['electrodeMapPath']
        mapExt = electrodeMapPath.split('.')[-1]
        if mapExt == 'cmp':
            cmpDF = prb_meta.cmpToDF(electrodeMapPath)
        elif mapExt == 'map':
            cmpDF = prb_meta.mapToDF(electrodeMapPath)
        csvPath = electrodeMapPath.replace(mapExt, 'csv')
        prbPath = electrodeMapPath.replace(mapExt, 'prb')
        cmpDF.to_csv(csvPath)
        #
        labelsInFile = dataio.datasource.sig_channels['name']
        labelsInMap = cmpDF['label'].unique().tolist()
        #
        cmpDF = cmpDF.set_index('label').reindex(labelsInFile).reset_index()
        cmpDF.loc[:, ['xcoords', 'ycoords', 'zcoords']] = (
            cmpDF.loc[:, ['xcoords', 'ycoords', 'zcoords']]
            .fillna(-1))
        if spikeSortingOpts['remakePrb']:
            excludeChans = spikeSortingOpts['excludeChans']
            for label in labelsInFile:
                if (label not in labelsInMap) and (label not in excludeChans):
                    excludeChans.append(label)
            #
            prb_meta.cmpDFToPrb(
                cmpDF, filePath=prbPath,
                labels=cmpDF.loc[~cmpDF['label'].isin(excludeChans), 'label'].to_list(),
                **spikeSortingOpts['prbOpts']
                )
    #
    dataio.set_probe_file(prbPath)
    #
    for chan_grp in dataio.channel_groups.keys():
        cc = tdc.CatalogueConstructor(
            dataio=dataio, name=name, chan_grp=chan_grp)
        print('initialize_catalogueconstructor cat constructor is: ')
        print(cc)
    return


def preprocess_signals_and_peaks(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        chunksize=4096,
        fill_overflow=False,
        highpass_freq=250.,
        lowpass_freq=5000.,
        notch_freq=60.,
        common_ref_freq=1000.,
        filter_order=3,
        smooth_size=0,
        relative_threshold=5.,
        peak_span=1e-3,
        common_ref_removal=False,
        output_dtype='float32',
        noise_estimate_duration=60.,
        sample_snippet_duration=240.,
        signalpreprocessor_engine='numpy',
        peakdetector_engine='numpy',
        verboseTiming=False):
    #
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    print(dataio)
    #
    cc.set_preprocessor_params(
        chunksize=chunksize,
        signalpreprocessor_engine=signalpreprocessor_engine,
        memory_mode='memmap',
        fill_overflow=fill_overflow,
        highpass_freq=highpass_freq,
        lowpass_freq=lowpass_freq,
        notch_freq=notch_freq,
        common_ref_freq=common_ref_freq,
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
    #
    total_duration = np.floor(
        cc.dataio.get_segment_shape(0)[0] / cc.dataio.sample_rate)
    if verboseTiming:
        t1 = time.perf_counter()
    if noise_estimate_duration == 'all':
        noise_estimate_duration = total_duration
    cc.estimate_signals_noise(
        seg_num=0, duration=noise_estimate_duration)
    if verboseTiming:
        t2 = time.perf_counter()
        print('estimate_signals_noise took {} seconds'.format(t2-t1))
    if sample_snippet_duration == 'all':
        sample_snippet_duration = total_duration
    if verboseTiming:
        t1 = time.perf_counter()
    cc.run_signalprocessor(
        duration=sample_snippet_duration)
    if verboseTiming:
        t2 = time.perf_counter()
        print('run_signalprocessor took {} seconds'.format(t2-t1))
    print(cc)
    return


def extract_waveforms_pca(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        nb_noise_snippet=2000,
        minWaveforms=10,
        alien_value_threshold=150.,
        extractOpts=dict(
            mode='rand',
            n_left=-34, n_right=66, nb_max=500000,
            align_waveform=False,
            subsample_ratio=20),
        featureOpts={
            'method': 'neighborhood_pca',
            'n_components_by_neighborhood': 10,
            'radius_um': 600},
        verboseTiming=False
        ):
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    if extractOpts['mode'] == 'all':
        extractOpts['nb_max'] = None
    if verboseTiming:
        t1 = time.perf_counter()
    cc.extract_some_waveforms(**extractOpts)
    if verboseTiming:
        t2 = time.perf_counter()
        print('extract_some_waveforms took {} seconds'.format(t2-t1))
        t1 = time.perf_counter()
    cc.clean_waveforms(
        alien_value_threshold=alien_value_threshold)
    if verboseTiming:
        t2 = time.perf_counter()
        print('clean_waveforms took {} seconds'.format(t2-t1))
    if cc.some_waveforms is not None:
        if (cc.some_waveforms.shape[0] < minWaveforms):
            print('No waveforms found!')
            return
    else:
        print('No waveforms found!')
        return
    #  extract_some_noise
    if verboseTiming:
        t1 = time.perf_counter()
    cc.extract_some_noise(
        nb_snippet=nb_noise_snippet)
    if verboseTiming:
        t2 = time.perf_counter()
        print('extract_some_noise took {} seconds'.format(t2-t1))
        t1 = time.perf_counter()
    cc.extract_some_features(**featureOpts)
    if verboseTiming:
        t2 = time.perf_counter()
        print('project took {} seconds'.format(t2-t1))
    if cc.projector is not None:
        ccFolderName = os.path.dirname(cc.info_filename)
        projectorPath = os.path.join(ccFolderName, 'projector.pickle')
        if 'GlobalPUMAP' in cc.projector.__repr__():
            saveSubfolderPath = os.path.join(ccFolderName, 'umap')
            if not os.path.exists(saveSubfolderPath):
                os.makedirs(saveSubfolderPath)
                cc.projector.umap.save(saveSubfolderPath, useH5=False)
                # cc.projector.umap = None
        with open(projectorPath, 'wb') as f:
            pickle.dump({'projector': cc.projector}, f)
    print(cc)
    return


def extract_pca(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        featureOpts={
            'method': 'neighborhood_pca',
            'n_components_by_neighborhood': 10,
            'radius_um': 600}
        ):
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    t1 = time.perf_counter()
    cc.extract_some_features(**featureOpts)
    t2 = time.perf_counter()
    print('project took {} seconds'.format(t2-t1))
    if cc.projector is not None:
        ccFolderName = os.path.dirname(cc.info_filename)
        projectorPath = os.path.join(ccFolderName, 'projector.pickle')
        if 'GlobalPUMAP' in cc.projector.__repr__():
            saveSubfolderPath = os.path.join(ccFolderName, 'umap')
            if not os.path.exists(saveSubfolderPath):
                os.makedirs(saveSubfolderPath)
                cc.projector.umap.save(saveSubfolderPath, useH5=False)
                # cc.projector.umap = None
        with open(projectorPath, 'wb') as f:
            pickle.dump({'projector': cc.projector}, f)
    print(cc)
    return


def extract_waveforms(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        nb_noise_snippet=2000,
        minWaveforms=10,
        extractOpts=dict(
            mode='rand',
            n_left=-34, n_right=66, nb_max=500000,
            align_waveform=False,
            subsample_ratio=20)
        ):
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    if extractOpts['mode'] == 'all':
        extractOpts['nb_max'] = None
    t1 = time.perf_counter()
    cc.extract_some_waveforms(**extractOpts)
    t2 = time.perf_counter()
    print('extract_some_waveforms took {} seconds'.format(t2-t1))
    t1 = time.perf_counter()
    cc.clean_waveforms(
        alien_value_threshold=100.)
    t2 = time.perf_counter()
    print('clean_waveforms took {} seconds'.format(t2-t1))
    if cc.some_waveforms is not None:
        if (cc.some_waveforms.shape[0] < minWaveforms):
            print('No waveforms found!')
            return
    else:
        print('No waveforms found!')
        return
    #  extract_some_noise
    t1 = time.perf_counter()
    cc.extract_some_noise(
        nb_snippet=nb_noise_snippet)
    t2 = time.perf_counter()
    print('extract_some_noise took {} seconds'.format(t2-t1))
    print(cc)
    return


def cluster(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        minFeatures=10,
        clusterOpts={
            'method': 'kmeans',
            'n_clusters': 10
        },
        autoMerge=False,
        auto_merge_threshold=0.9,
        minClusterSize=5,
        auto_make_catalog=False,
        verboseTiming=False):

    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    if cc.some_features is not None:
        if cc.some_features.shape[0] < minFeatures:
            print('Not enough threshold crossings to cluster!')
            return
    else:
        print('Not enough threshold crossings to cluster!')
        return
    if verboseTiming:
        t1 = time.perf_counter()
    cc.find_clusters(**clusterOpts)
    if verboseTiming:
        t2 = time.perf_counter()
        print('find_clusters took {} seconds'.format(t2-t1))
    if autoMerge:
        try:
            print(cc)
            if verboseTiming:
                t1 = time.perf_counter()
            #  cc.compute_spike_waveforms_similarity()
            cc.auto_merge_high_similarity(
                threshold=auto_merge_threshold)
            if verboseTiming:
                t2 = time.perf_counter()
                print('auto_merge took {} seconds'.format(t2-t1))
        except Exception:
            traceback.print_exc()
    #
    # TODO Fix moving small clusters to trash
    # cc.move_small_cluster_to_trash(n=minClusterSize)
    #
    cc.order_clusters(by='waveforms_rms')
    print(cc)
    if auto_make_catalog:
        cc.make_catalogue_for_peeler()
    return


def open_cataloguewindow(
        triFolder, chan_grp=0,
        name='catalogue_constructor',
        minTotalWaveforms=None):
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(
        dataio=dataio, name=name, chan_grp=chan_grp)
    #
    openWindow = True
    if minTotalWaveforms is not None:
        # negative labels are reserved for "special" use, such as trash or alien
        mask = cc.all_peaks['cluster_label'] >= 0
        if mask.sum() < minTotalWaveforms:
            print('Skipping chan grp {} because it has {} spikes'.format(
                chan_grp, mask.sum()
            ))
            openWindow = False
            cc.change_spike_label(mask, -1)
            # cc.all_peaks['cluster_label'][mask] = -1
            # cc.on_new_cluster()
    print(cc)
    if openWindow:
        app = pg.mkQApp()
        win = tdc.CatalogueWindow(cc)
        win.show()
        #
        app.exec_()
        #  re order by rms
        cc.order_clusters(by='waveforms_rms')
    #  save catalogue before peeler
    cc.make_catalogue_for_peeler()
    return


def clean_catalogue(
        triFolder,
        name='catalogue_constructor',
        min_nb_peak=10, chan_grp=0):
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
        shape_distance_threshold=3,
        shape_boundary_threshold=4,
        confidence_threshold=0.6,
        n_max_passes=3,
        energy_reduction_threshold=0,
        debugging=False, progressbar=False,
        chunksize=2**12,
        duration=None, useOpenCL=False, verboseTiming=False):

    dataio = tdc.DataIO(dirname=triFolder)
    initial_catalogue = dataio.load_catalogue(chan_grp=chan_grp)
    if initial_catalogue is None:
        return
    peeler = tdc.Peeler(dataio)

    if useOpenCL:
        peeler.change_params(
            catalogue=initial_catalogue,
            shape_distance_threshold=shape_distance_threshold,
            energy_reduction_threshold=energy_reduction_threshold,
            n_max_passes=n_max_passes,
            debugging=debugging, chunksize=chunksize)
    else:
        peeler.change_params(
            catalogue=initial_catalogue,
            shape_distance_threshold=shape_distance_threshold,
            shape_boundary_threshold=shape_boundary_threshold,
            confidence_threshold=confidence_threshold,
            energy_reduction_threshold=energy_reduction_threshold,
            n_max_passes=n_max_passes,
            debugging=debugging, chunksize=chunksize)

    if verboseTiming:
        t1 = time.perf_counter()

    peeler.run(duration=duration, progressbar=progressbar)

    if verboseTiming:
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
        shape_distance_threshold=3,
        shape_boundary_threshold=4,
        energy_reduction_threshold=1,
        refractory_period=None, ignoreTags=['so_bad'],
        plotting=False, altDataIOInfo=None,
        waveformSignalType='processed',
        FRThresh=1):
    dataio = tdc.DataIO(
        dirname=triFolder,
        altInfo=altDataIOInfo)
    #
    chanNames = np.array(
        dataio.datasource.get_channel_names())
    #
    blockName = 'tdc sorted spikes'
    block = Block(name=blockName)
    #
    for segIdx in [0]:
        seg = Segment()
        block.segments.append(seg)
        maxTime = (
            dataio.get_segment_length(segIdx) /
            dataio.sample_rate)
        catalogue = None
        i = 0
        while catalogue is None:
            catalogue = dataio.load_catalogue(chan_grp=chan_grps[i])
        n_left = int(1.5 * catalogue['n_left'])
        n_right = int(1.5 * catalogue['n_right'])
        window1 = scipy.signal.triang(2 * int(-n_left) + 1)
        window2 = scipy.signal.triang(2 * int(n_right) + 1)
        window = np.concatenate(
            (
                window1[:int(-n_left)],
                window2[int(n_right) + 1:]),
            axis=-1)
        discountEdges = False
        if discountEdges:
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
            #
            # cc = tdc.CatalogueConstructor(
            #     dataio=dataio, name=catConstructorName, chan_grp=chan_grp)
            catalogue = dataio.load_catalogue(chan_grp=chan_grp)
            if catalogue is None:
                continue
            #
            clustersDF = pd.DataFrame(catalogue['clusters'])
            #
            if not len(clustersDF) > 0:
                # no events from this entire channel
                continue
            #
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
                    clustersDF.loc[exportMask, :].groupby('cluster_label'))
            else:
                #  iterate over channels
                channelGrouper = (
                    clustersDF.groupby('max_on_channel_id'))
                #  iterate over units
                unitGrouper = (
                    clustersDF.groupby('cluster_label'))
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
            if plotting:
                breakDownPath = os.path.join(
                    triFolder,
                    'feature_distances_{}.png'.format(chan_grp))
                nCats = 7
                nUnits = unitGrouper.size().size
                fig, ax = plt.subplots(nUnits, nCats, sharey=True)
                fig.set_size_inches(4 * nCats, 4 * nUnits)
            for uIdx, (unitName, group) in enumerate(unitGrouper):
                #  assert group['max_on_channel_id'] only has one element
                chanId = group['max_on_channel_id'].values[0]
                idxOfChanLabel = np.argwhere(channelIds == chanId).flatten()[0]
                chanLabel = chanNames[chanId]
                chanName = '{}'.format(chanLabel)  # sanitize to string
                chanIdx = block.filter(objects=ChannelIndex, name=chanName)[0]
                #  create unit indexes
                thisUnit = Unit(
                    name='{}#{}'.format(
                        chanName, unitNumberLookup[unitName]))
                print(thisUnit.name)
                #  get spike times
                try:
                    spike = dataio.get_spikes(
                        seg_num=segIdx, chan_grp=chan_grp)
                    # except Exception:
                    #     traceback.print_exc()
                    #     pdb.set_trace()
                    unitMask = np.isin(
                        spike['cluster_label'],
                        group['cluster_label'])
                    #  discard edges
                    edgeMask = (
                        (spike['index'] + n_left > 0) &
                        (
                            spike['index'] + n_right <
                            dataio.get_segment_length(segIdx))
                        )
                    unitMask = unitMask & edgeMask
                    # try:
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
                            n_left=n_left,
                            n_right=n_right,
                            signal_type=waveformSignalType
                            )
                        wvfBaseline = spikeWaveforms.mean(axis=1)
                        spikeWaveforms = spikeWaveforms - wvfBaseline[:, np.newaxis, :]
                    except Exception:
                        traceback.print_exc()
                        pdb.set_trace()
                    spikeWaveforms = np.swapaxes(
                        spikeWaveforms,
                        1, 2)
                    timesDF = pd.DataFrame(spikeTimes, columns=['times'])
                    timesDF['templateDist'] = spike[unitMask]['feature_distance']
                    timesDF['maxDeviation'] = np.nan
                    timesDF['energyReduction'] = np.nan
                    meanWaveform = np.mean(
                        spikeWaveforms[:, idxOfChanLabel, :], axis=0)
                    #  TODO: compare mean waveforms; if sufficiently, similar, collapse the two units
                    #  mirror naming convention from tdc
                    pred_wf = meanWaveform
                    if waveformSignalType == 'initial':
                        pred_wf = (pred_wf) / catalogue['signals_mads'][idxOfChanLabel]
                    norm_factor = 1
                    for idx in timesDF.index:
                        if waveformSignalType == 'processed':
                            wf = spikeWaveforms[idx, idxOfChanLabel, :]
                        else:
                            wf = (spikeWaveforms[idx, idxOfChanLabel, :]) / catalogue['signals_mads'][idxOfChanLabel]
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
                        resid_energy = np.sqrt(np.sum(wf_resid**2) / wf_resid.shape[0])
                        wf_energy = np.sqrt(np.sum(wf**2) / wf.shape[0])
                        energy_reduction = (wf_energy - resid_energy) / resid_energy
                        timesDF.loc[idx, 'energyReduction'] = energy_reduction
                    if shape_boundary_threshold is not None:
                        tooFar = timesDF.index[
                            timesDF['maxDeviation'] > shape_boundary_threshold]
                        timesDF.drop(index=tooFar, inplace=True)
                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]
                        timesDF.reset_index(drop=True, inplace=True)
                    if shape_distance_threshold is not None:
                        tooFar = timesDF.index[
                            timesDF['templateDist'] > shape_distance_threshold]
                        timesDF.drop(index=tooFar, inplace=True)
                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]
                        timesDF.reset_index(drop=True, inplace=True)
                    if energy_reduction_threshold is not None:
                        tooFar = timesDF.index[
                            timesDF['energyReduction'] < energy_reduction_threshold]
                        timesDF.drop(index=tooFar, inplace=True)
                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]
                        timesDF.reset_index(drop=True, inplace=True)
                    timesDF['isi'] = (
                        timesDF['times']
                        .diff()
                        .fillna(method='bfill'))
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
                            timesDF.drop(
                                index=pd.unique(dropIndices), inplace=True)
                            timesDF['isi'] = timesDF['times'].diff().fillna(method='bfill')
                            breaksRefractory = timesDF['isi'] < refractory_period
                        #
                        spikeTimes = spikeTimes[timesDF.index]
                        spikeWaveforms = spikeWaveforms[timesDF.index, :, :]
                    #
                    if not spikeTimes.any():
                        raise Exception(
                            '{} has no spikes'.format(thisUnit.name))
                    timesDF.reset_index(inplace=True, drop=True)
                    arrayAnn = {
                        'templateDist': timesDF['templateDist'].values,
                        #  'maxDeviation': timesDF['maxDeviation'].values,
                        #  'energyReduction': timesDF['energyReduction'].values,
                        'isi': timesDF['isi'].values}
                    arrayAnnNames = {'arrayAnnNames': list(arrayAnn.keys())}
                    #  
                    if group['tag'].iloc[0] == '':
                        if group['max_peak_amplitude'].iloc[0] < -7:
                            group.loc[:, 'tag'] = 'so_good'
                        else:
                            group.loc[:, 'tag'] = 'good'
                    if plotting:
                        fitBreakDown = pd.cut(timesDF['templateDist'], nCats)
                        # 
                        uniqueCats = fitBreakDown.unique().sort_values()
                        for pltIdx, cat in enumerate(uniqueCats):
                            catIdx = fitBreakDown.index[fitBreakDown == cat]
                            pltWvf = pd.DataFrame(spikeWaveforms[catIdx, idxOfChanLabel, :])
                            pltWvf = pltWvf.unstack().reset_index()
                            pltWvf.columns = ['bin', 'index', 'signal']
                            sns.lineplot(
                                x='bin', y='signal', units='index',
                                data=pltWvf,
                                ci='sd',
                                sort=False, ax=ax[uIdx][pltIdx],
                                estimator=None, alpha=0.1)
                            ax[uIdx][pltIdx].set_title('{}'.format(cat))
                        # 
                    # pdb.set_trace()
                    if waveformSignalType == 'processed':
                        spikeWaveforms = spikeWaveforms * catalogue['signals_mads'][idxOfChanLabel]
                    st = SpikeTrain(
                        name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                        times=spikeTimes, units='sec',
                        waveforms=spikeWaveforms[:, idxOfChanLabel, :][:, None, :] * pq.uV,
                        t_stop=maxTime, t_start=0,
                        left_sweep=n_left,
                        sampling_rate=dataio.sample_rate * pq.Hz,
                        array_annotations=arrayAnn,
                        **arrayAnn, **arrayAnnNames)
                    for colName in group.columns:
                        v = group[colName].iloc[0]
                        if v:
                            st.annotations.update(
                                {colName: v}
                            )
                    #  
                    st.annotations.update({'chan_grp': chan_grp})
                except Exception:
                    traceback.print_exc()
                    arrayAnn = {
                        'templateDist': np.array([]),
                        #  'maxDeviation': np.array([]),
                        #  'energyReduction': np.array([]),
                        'isi': np.array([])}
                    arrayAnnNames = {'arrayAnnNames': list(arrayAnn.keys())}
                    st = SpikeTrain(
                        name='seg{}_{}'.format(int(segIdx), thisUnit.name),
                        times=np.array([]), units='sec',
                        waveforms=np.array([]).reshape((0, 0, 0))*pq.uV,
                        t_stop=maxTime, t_start=0,
                        left_sweep=n_left,
                        sampling_rate=dataio.sample_rate * pq.Hz,
                        # array_annotations=arrayAnn,
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
            if plotting:
                plt.savefig(breakDownPath)
                plt.close()
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
        triFolder, chan_grps=None, purgeAll=False, diagnosticsOnly=False, altDataIOInfo=None):
    if not purgeAll:
        assert chan_grps is not None, 'Need to specify chan_grps!'

        for chan_grp in chan_grps:
            #  chan_grp = 0
            grpFolder = 'channel_group_{}'.format(chan_grp)
            if not diagnosticsOnly:
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
        # for fl in glob.glob(os.path.join(triFolder, 'templateHist*.png')):
        #     os.remove(fl)
        for fl in glob.glob(os.path.join(triFolder, 'feature_distances*.png')):
            os.remove(fl)
        for grpFolder in grpFolders:
            try:
                if not diagnosticsOnly:
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
        try:
            assert os.path.exists(catFolderSource), 'source catalogue does not exist!'
        except Exception:
            traceback.print_exc()
            print('{}'.format(catFolderSource))
            pdb.set_trace()
        assert os.path.exists(grpFolderDest), 'destination folder does not exist!'
        ccFolderSource = os.path.join(
            grpFolderSource, 'catalogue_constructor')
        ccFolderDest = os.path.join(
            grpFolderDest, 'catalogue_constructor')
        sklearnObjs = ['projector', 'supervised_projector', 'classifier']
        for skObjName in sklearnObjs:
            if os.path.exists(os.path.join(ccFolderSource, skObjName + '.pickle')):
                shutil.copy(
                    os.path.join(ccFolderSource, skObjName + '.pickle'),
                    os.path.join(ccFolderDest, skObjName + '.pickle'))
        umapNNFolders = ['umap', 'supervised-umap']
        for umapFolder in umapNNFolders:
            if os.path.exists(os.path.join(ccFolderSource, umapFolder)):
                shutil.copytree(
                    os.path.join(ccFolderSource, umapFolder),
                    os.path.join(ccFolderDest, umapFolder)
                )
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
        fill_overflow=False,
        highpass_freq=300.,
        lowpass_freq=6000.,
        notch_freq=60.,
        common_ref_freq=1000.,
        filter_order=4,
        smooth_size=0,
        output_dtype='float32',
        peak_span=1e-3,
        featureOpts={
            'method': 'pca_by_channel',
            'n_components_by_channel': 15},
        clusterOpts={
            'method': 'kmeans',
            'n_clusters': 4,
            },
        extractOpts=dict(
            mode='rand',
            n_left=-34, n_right=66, nb_max=500000,
            align_waveform=False,
            subsample_ratio=20),
        autoMerge=False, auto_merge_threshold=0.8,
        auto_make_catalog=False,
        common_ref_removal=False,
        noise_estimate_duration=120.,
        sample_snippet_duration=240.,
        chunksize=int(2**12),
        nb_noise_snippet=2000,
        minWaveforms=10, minWaveformRate=None,
        alien_value_threshold=150.,
        attemptMPI=False
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
    if minWaveformRate is not None:
        minWaveforms = int(sample_snippet_duration / minWaveformRate)
    for idx, chan_grp in enumerate(chansToAnalyze):
        if idx % SIZE == RANK:
            print('memory usage: {}'.format(
                prf.memory_usage_psutil()))
            print('About to run preprocess_signals_and_peaks()...')
            preprocess_signals_and_peaks(
                triFolder, chan_grp=chan_grp,
                chunksize=chunksize,
                signalpreprocessor_engine='numpy',
                peakdetector_engine='numpy',
                fill_overflow=fill_overflow,
                highpass_freq=highpass_freq,
                lowpass_freq=lowpass_freq,
                notch_freq=notch_freq,
                common_ref_freq=common_ref_freq,
                filter_order=filter_order,
                smooth_size=smooth_size,
                output_dtype=output_dtype,
                relative_threshold=relative_threshold,
                peak_span=peak_span,
                common_ref_removal=common_ref_removal,
                noise_estimate_duration=noise_estimate_duration,
                sample_snippet_duration=sample_snippet_duration)
            print('About to run extract_waveforms_pca()...')
            extract_waveforms_pca(
                triFolder,
                chan_grp=chan_grp,
                nb_noise_snippet=nb_noise_snippet,
                minWaveforms=minWaveforms,
                alien_value_threshold=alien_value_threshold,
                extractOpts=extractOpts,
                featureOpts=featureOpts
                )
            print('About to run cluster()...')
            cluster(
                triFolder,
                clusterOpts=clusterOpts,
                chan_grp=chan_grp, auto_make_catalog=auto_make_catalog,
                autoMerge=autoMerge, auto_merge_threshold=auto_merge_threshold)
    return


def batchRunClustering(
        triFolder, chansToAnalyze,
        featureOpts={
            'method': 'pca_by_channel',
            'n_components_by_channel': 15},
        clusterOpts={
            'method': 'kmeans',
            'n_clusters': 4,
            },
        autoMerge=False, auto_merge_threshold=0.8,
        attemptMPI=False
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
                prf.memory_usage_psutil()))
            extract_pca(
                triFolder,
                chan_grp=chan_grp,
                featureOpts=featureOpts
                )
            cluster(
                triFolder,
                clusterOpts=clusterOpts,
                chan_grp=chan_grp, auto_make_catalog=False,
                autoMerge=autoMerge, auto_merge_threshold=auto_merge_threshold)
    return


def batchPrepWaveforms(
        triFolder, chansToAnalyze,
        relative_threshold=5.5,
        fill_overflow=False,
        highpass_freq=300.,
        lowpass_freq=6000.,
        notch_freq=60.,
        common_ref_freq=1000.,
        filter_order=4,
        smooth_size=0,
        output_dtype='float32',
        peak_span=1e-3,
        common_ref_removal=False,
        noise_estimate_duration=120.,
        sample_snippet_duration=240.,
        chunksize=4096,
        extractOpts=dict(
            mode='rand',
            n_left=-34, n_right=66, nb_max=500000,
            align_waveform=False,
            subsample_ratio=20),
        attemptMPI=False
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
                prf.memory_usage_psutil()))
            preprocess_signals_and_peaks(
                triFolder, chan_grp=chan_grp,
                chunksize=chunksize,
                signalpreprocessor_engine='numpy',
                peakdetector_engine='numpy',
                fill_overflow=fill_overflow,
                highpass_freq=highpass_freq,
                lowpass_freq=lowpass_freq,
                notch_freq=notch_freq,
                common_ref_freq=common_ref_freq,
                filter_order=filter_order,
                smooth_size=smooth_size,
                output_dtype=output_dtype,
                relative_threshold=relative_threshold,
                peak_span=peak_span,
                common_ref_removal=common_ref_removal,
                noise_estimate_duration=noise_estimate_duration,
                sample_snippet_duration=sample_snippet_duration)
            extract_waveforms(
                triFolder,
                chan_grp=chan_grp,
                extractOpts=extractOpts
                )
    return


def batchPeel(
        triFolder, chansToAnalyze,
        shape_distance_threshold=3,
        shape_boundary_threshold=4,
        confidence_threshold=0.6,
        energy_reduction_threshold=0,
        n_max_passes=3,
        chunksize=2**12,
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
            print('chan_grp {}, memory usage: {}'.format(
                chan_grp,
                prf.memory_usage_psutil()))
            try:
                run_peeler(
                    triFolder,
                    shape_distance_threshold=shape_distance_threshold,
                    shape_boundary_threshold=shape_boundary_threshold,
                    confidence_threshold=confidence_threshold,
                    energy_reduction_threshold=energy_reduction_threshold,
                    n_max_passes=n_max_passes,
                    debugging=True, chunksize=chunksize,
                    chan_grp=chan_grp, progressbar=False)
                gc.collect()
            except Exception:
                traceback.print_exc()
                pass
    return
