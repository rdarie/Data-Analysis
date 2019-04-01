import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
from currentExperiment import *
import traceback
try:
    tdch.initialize_catalogueconstructor(
        trialFilesFrom['utah']['folderPath'],
        trialFilesFrom['utah']['ns5FileName'],
        triFolder,
        nspPrbPath,
        removeExisting=False, fileFormat='Blackrock')
except Exception:
    traceback.print_exc()

dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))

chansToAnalyze = [
    68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95]
for chan_grp in chansToAnalyze:
        tdch.preprocess_signals_and_peaks(
            triFolder, chan_grp=chan_grp,
            chunksize=2048,
            signalpreprocessor_engine='opencl',
            peakdetector_engine='opencl',
            highpass_freq=300.,
            lowpass_freq=5000.,
            relative_threshold=4,
            peak_span=.5e-3,
            common_ref_removal=False,
            noise_estimate_duration=60.,
            sample_snippet_duration=150.)

        tdch.extract_waveforms_pca(
            triFolder, feature_method='pca_by_channel',
            n_components_by_channel=30, chan_grp=chan_grp)

        tdch.cluster(
            triFolder, cluster_method='kmeans',
            n_clusters=5, chan_grp=chan_grp)

for chan_grp in chansToAnalyze:
        tdch.open_cataloguewindow(triFolder, chan_grp=chan_grp)

for chan_grp in chansToAnalyze:
        tdch.run_peeler(
            triFolder, strict_multiplier=0.02,
            debugging=True,
            useOpenCL=False, chan_grp=chan_grp)
#  
for chan_grp in dataio.channel_groups.keys():
    tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)

chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:-1]
tdch.neo_block_after_peeler(triFolder, chan_grps=chansToAnalyze)

triFolderSource = triFolder
destinationList = [triFolderSource.replace('3', i) for i in ['1']]

for triFolderDest in destinationList:
    tdch.transferTemplates(
        triFolderSource, triFolderDest,
        chansToAnalyze)
