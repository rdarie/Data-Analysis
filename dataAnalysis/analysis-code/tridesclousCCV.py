"""
Usage:
    tridesclousCCV.py [--trialIdx=trialIdx] [--visuals]

Arguments:
    trialIdx            which trial to analyze

Options:
    --visuals           include visualization steps
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.helper_functions as hf
from currentExperiment import *
import os, gc

arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    triFolder = os.path.join(
        nspFolder, 'tdc_' + ns5FileName)

#  catName = 'F:\\Murdoc Neural Recordings\\201901271000-Proprio\\tdc_Trial001\\channel_group_4\\catalogues\\initial\\catalogue.pickle'
#  with open(catName, 'rb') as f:
#      data = pickle.load(f)

try:
    tdch.initialize_catalogueconstructor(
        nspFolder,
        ns5FileName,
        triFolder,
        nspPrbPath,
        removeExisting=False, fileFormat='Blackrock')
except Exception:
    pass

purgePeeler = False
if purgePeeler:
    tdch.purgePeelerResults(
        triFolder, purgeAll=True)

dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))

'''
chansToAnalyze = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95]
'''

chansToAnalyze = []
for chan_grp in chansToAnalyze:
        tdch.preprocess_signals_and_peaks(
            triFolder, chan_grp=chan_grp,
            chunksize=2048,
            signalpreprocessor_engine='numpy',
            peakdetector_engine='numpy',
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
    #   chan_grp = 15
    print('memory usage: {}'.format(
        hf.memory_usage_psutil()))
    tdch.run_peeler(
        triFolder, strict_multiplier=2e-3,
        debugging=False,
        useOpenCL=False, chan_grp=chan_grp, progressbar=False)
    gc.collect()

for chan_grp in chansToAnalyze:
    tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)

chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[:-1]
if chansToAnalyze:
    tdch.neo_block_after_peeler(triFolder, chan_grps=chansToAnalyze)
