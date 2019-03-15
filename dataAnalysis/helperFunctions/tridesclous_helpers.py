import tridesclous as tdc
import pyqtgraph as pg
import pandas as pd
import numpy as np
from matplotlib import pyplot
import time
import os
import re


def cmpToDF(arrayFilePath):
    arrayMap = pd.read_table(
        arrayFilePath,
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
        elecName = re.split(r'\d*', row['label'])[0]
        elecIdx = int(re.split(r'\D*', row['label'])[-1])
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
        prependDummy=0, appendDummy=0):
    if names is not None:
        keepMask = cmpDF['elecName'].isin(names)
        cmpDF = cmpDF.loc[keepMask, :]
    if banks is not None:
        keepMask = cmpDF['bank'].isin(banks)
        cmpDF = cmpDF.loc[keepMask, :]

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

    for idx, (name, group) in enumerate(cmpDF.groupby('elecName')):
        prbDict.update({idx + idxOffset: {
            'channels': list(group.index + int(prependDummy)),
            'geometry': {
                rName + int(prependDummy): (
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
        removeExisting=False, fileFormat='NIX'):
    #  setup file source
    
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

    #  setup probe file
    dataio.set_probe_file(prbPath)
    for chan_grp in dataio.channel_groups.keys():
        catalogueconstructor = tdc.CatalogueConstructor(
            dataio=dataio, chan_grp=chan_grp)
        print(catalogueconstructor)

    #import pdb; pdb.set_trace()
    return


def preprocess_signals_and_peaks(
        triFolder, chan_grp=0,
        chunksize=1024,
        highpass_freq=250.,
        lowpass_freq=5000.,
        relative_threshold=4.,
        peak_span=0.5e-3,
        common_ref_removal=True,
        noise_estimate_duration=60.,
        sample_snippet_duration=240.,
        signalpreprocessor_engine='numpy',
        peakdetector_engine='numpy'):

    dataio = tdc.DataIO(dirname=triFolder)
    catalogueconstructor = tdc.CatalogueConstructor(
        dataio=dataio, chan_grp=chan_grp)
    print(dataio)

    catalogueconstructor.set_preprocessor_params(
        chunksize=chunksize,
        signalpreprocessor_engine=signalpreprocessor_engine,
        memory_mode='memmap',
        highpass_freq=highpass_freq,
        lowpass_freq=lowpass_freq,
        common_ref_removal=common_ref_removal,
        #  peak detector
        peakdetector_engine=peakdetector_engine,
        #  peakdetector_engine='opencl',
        peak_sign='-',
        relative_threshold=relative_threshold,
        peak_span=peak_span  # units of seconds
        )

    t1 = time.perf_counter()
    catalogueconstructor.estimate_signals_noise(
        seg_num=0, duration=noise_estimate_duration)
    t2 = time.perf_counter()
    print('estimate_signals_noise took {} seconds'.format(t2-t1))

    t1 = time.perf_counter()
    catalogueconstructor.run_signalprocessor(
        duration=sample_snippet_duration)
    t2 = time.perf_counter()
    print('run_signalprocessor took {} seconds'.format(t2-t1))

    print(catalogueconstructor)
    return


def extract_waveforms_pca_cluster(
        triFolder, chan_grp=0,
        wave_extract_mode='rand',
        n_left=-16, n_right=48, nb_max=1000,
        feature_method='neighborhood_pca',
        n_components_by_channel=10,
        n_components=10,
        n_components_by_neighborhood=10,
        radius_um=600,
        cluster_method='kmeans',
        n_clusters=100,
        auto_merge_threshold=0.9):

    dataio = tdc.DataIO(dirname=triFolder)
    catalogueconstructor = tdc.CatalogueConstructor(
        dataio=dataio, chan_grp=chan_grp)

    if wave_extract_mode == 'all':
        nb_max = None
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_waveforms(
        mode=wave_extract_mode, n_left=n_left, n_right=n_right,
        nb_max=nb_max)
    #  catalogueconstructor.extract_some_waveforms(mode='all', n_left=-45, n_right=60)
    t2 = time.perf_counter()
    print('extract_some_waveforms took {} seconds'.format(t2-t1))

    t1 = time.perf_counter()
    catalogueconstructor.clean_waveforms(
        alien_value_threshold=100.)
    t2 = time.perf_counter()
    print('clean_waveforms took {} seconds'.format(t2-t1))

    #  extract_some_noise
    t1 = time.perf_counter()
    catalogueconstructor.extract_some_noise(
        nb_snippet=400)
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
    catalogueconstructor.extract_some_features(
        method=feature_method, **featureArgs)
    t2 = time.perf_counter()
    print('project took {} seconds'.format(t2-t1))
    
    t1 = time.perf_counter()
    clusterArgs = {}
    if cluster_method in ['kmeans', 'gmm', 'agglomerative']:
        clusterArgs.update({'n_clusters': n_clusters})
    catalogueconstructor.find_clusters(
        method=cluster_method, **clusterArgs)
    t2 = time.perf_counter()
    print('find_clusters took {} seconds'.format(t2-t1))
    
    t1 = time.perf_counter()
    catalogueconstructor.compute_spike_waveforms_similarity()
    catalogueconstructor.auto_merge_high_similarity(
        threshold=auto_merge_threshold)
    t2 = time.perf_counter()
    print('auto_merge took {} seconds'.format(t2-t1))
    
    print(catalogueconstructor)
    
    catalogueconstructor.order_clusters(by='waveforms_rms')
    return


def open_cataloguewindow(triFolder, chan_grp=0):
    dataio = tdc.DataIO(dirname=triFolder)
    catalogueconstructor = tdc.CatalogueConstructor(
        dataio=dataio, chan_grp=chan_grp)
    
    app = pg.mkQApp()
    win = tdc.CatalogueWindow(catalogueconstructor)
    win.show()
    
    app.exec_()   
    return 


def clean_catalogue(triFolder, chan_grp=0):
    #  the catalogue need strong attention with teh catalogue windows.
    #  here a dirty way a cleaning is to take on the first 20 bigger cells
    #  the peeler will only detect them
    dataio = tdc.DataIO(dirname=triFolder)
    cc = tdc.CatalogueConstructor(dataio=dataio, chan_grp=chan_grp)
    
    #  re order by rms
    cc.order_clusters(by='waveforms_rms')

    #  re label >20 to trash (-1)
    mask = cc.all_peaks['cluster_label'] > 20
    cc.all_peaks['cluster_label'][mask] = -1
    cc.on_new_cluster()
    
    #  save catalogue before peeler
    cc.make_catalogue_for_peeler()
    return


def run_peeler(triFolder, chan_grp=0, duration=None):
    dataio = tdc.DataIO(dirname=triFolder)
    initial_catalogue = dataio.load_catalogue(chan_grp=chan_grp)

    print(dataio)
    peeler = tdc.Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue)

    t1 = time.perf_counter()
    peeler.run(duration=duration)
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


if __name__ == '__main__':
    #  initialize_catalogueconstructor(dataPath, chan_grp=0)
    #  preprocess_signals_and_peaks(chan_grp=0)
    #  extract_waveforms_pca_cluster(chan_grp=0)
    #  open_cataloguewindow(chan_grp=0)
    #  run_peeler(chan_grp=0)
    #  open_PeelerWindow(chan_grp=0)
    pass
