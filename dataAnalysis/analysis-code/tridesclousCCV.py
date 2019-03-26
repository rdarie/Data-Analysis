import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
from currentExperiment import *

"""
tdch.initialize_catalogueconstructor(
    trialFilesFrom['utah']['folderPath'],
    trialFilesFrom['utah']['ns5FileName'],
    triFolder,
    nspPrbPath,
    removeExisting=False, fileFormat='Blackrock')
"""
dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))
chansToAnalyze = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
    21, 22, 23, 24]
"""
for chan_grp in chansToAnalyze:
        tdch.preprocess_signals_and_peaks(
            triFolder, chan_grp=chan_grp,
            chunksize=2048,
            signalpreprocessor_engine='numpy',
            peakdetector_engine='numpy',
            highpass_freq=300.,
            lowpass_freq=5000.,
            relative_threshold=6,
            peak_span=.5e-3,
            common_ref_removal=False,
            noise_estimate_duration=60.,
            sample_snippet_duration=150.)

        tdch.extract_waveforms_pca(
            triFolder, feature_method='pca_by_channel',
            n_components_by_channel=30, chan_grp=chan_grp)

        tdch.cluster(
            triFolder, cluster_method='kmeans',
            n_clusters=10, chan_grp=chan_grp)

for chan_grp in chansToAnalyze:
        tdch.open_cataloguewindow(triFolder, chan_grp=chan_grp)
"""
for chan_grp in chansToAnalyze:
        tdch.run_peeler(
            triFolder, strict_multiplier=2,
            useOpenCL=False, chan_grp=chan_grp)
"""
for chan_grp in dataio.channel_groups.keys():
    tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)

tdch.neo_block_after_peeler(triFolder, chan_grps=chansToAnalyze)
"""