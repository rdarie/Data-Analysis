"""
Usage:
    tridesclousLocal.py [--blockIdx=blockIdx] [--visuals] [--showProgressbar]

Arguments:

Options:
    --blockIdx=blockIdx            which trial to analyze
    --visuals           include visualization steps
    --showProgressbar   show progress bar when running peeler
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.helper_functions as hf
from currentExperiment import *
import os, gc

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
'''
arguments = {
  "--showProgressbar": False,
  "--blockIdx": "3",
  "--visuals": False}
'''

#  if overriding currentExperiment
if arguments['blockIdx']:
    print(arguments)
    blockIdx = int(arguments['blockIdx'])
    ns5FileName = 'Block00{}'.format(blockIdx)
    triFolder = os.path.join(
        nspFolder, 'tdc_' + ns5FileName)

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

purgeDiagnostics = False
if purgeDiagnostics:
    tdch.purgePeelerResults(
        triFolder, purgeAll=True, diagnosticsOnly=True)

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

chansToAnalyze = chansToAnalyze[:-1]
"""
chan_grp = 35
tdch.purgePeelerResults(
    triFolder, chan_grps=[chan_grp])
"""

chansToAnalyze = [1, 11, 21, 27, 41, 51, 61, 71, 81, 91]
for chan_grp in chansToAnalyze:
    #  print('memory usage: {}'.format(
    #      hf.memory_usage_psutil()))

    tdch.run_peeler(
        triFolder, shape_distance_threshold=3e-2,
        debugging=True,
        chan_grp=chan_grp,
        progressbar=arguments['showProgressbar'])
    gc.collect()

for chan_grp in chansToAnalyze:
    tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)

if chansToAnalyze:
    tdch.neo_block_after_peeler(triFolder, chan_grps=chansToAnalyze)
