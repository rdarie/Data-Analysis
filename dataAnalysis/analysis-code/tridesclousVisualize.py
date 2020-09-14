"""02: Curate constructor clusters
Usage:
    tridesclousVisualize [options]

Options:
    --blockIdx=blockIdx            which trial to analyze [default: 1]
    --exp=exp                      which experimental day to analyze
    --arrayName=arrayName          which electrode array to analyze [default: utah]
    --chan_start=chan_start        which chan_grp to start on [default: 0]
    --chan_stop=chan_stop          which chan_grp to stop on [default: 80]
    --peeler                       visualize Peeler results
    --constructor                  visualize Catalogue Constructor Results
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.helper_functions as hf
from currentExperiment import parseAnalysisOptions
#from exp201901211000 import *
import os, gc, traceback
from numba.errors import NumbaPerformanceWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
#
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
#


arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
mapExt = electrodeMapPath.split('.')[-1]
nspCsvPath = electrodeMapPath.replace(mapExt, 'csv')
nspPrbPath = electrodeMapPath.replace(mapExt, 'prb')
ns5FileName = ns5FileName.replace('Block', arguments['arrayName'])
triFolder = os.path.join(
    scratchFolder, 'tdc_{}{:0>3}'.format(arguments['arrayName'], blockIdx))

viewPeeler = False
if arguments['peeler']:
    viewPeeler = arguments['peeler']
viewConstructor = False
if arguments['constructor']:
    viewConstructor = arguments['constructor']

chan_start = int(arguments['chan_start'])
chan_stop = int(arguments['chan_stop'])
dataio = tdc.DataIO(dirname=triFolder)
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[chan_start:chan_stop]

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

# chansToAnalyze = [66, 71]
#  chansToAnalyze = chansToAnalyze[80:]
#  chansToAnalyze = [11, 14, 16, 18, 20, 22, 29, 36, 37, 39, 41, 48, 51, 52, 59, 60, 64, 66, 73, 75, 77, 80, 81, 88, 95]
#  chansToAnalyze = [2, 3, 7, 10, 19, 20, 26, 27, 52, 53, 54, 59, 66, 80, 81, 91, 92, 93]
if viewConstructor:
    for chan_grp in chansToAnalyze:
        print('\n\n\n\nOn channel group {}\n\n\n\n'.format(chan_grp))
        tdch.open_cataloguewindow(triFolder, chan_grp=chan_grp)
        gc.collect()
        #  try:
        #      tdch.clean_catalogue(
        #          triFolder,
        #          min_nb_peak=50, chan_grp=chan_grp)
        #  except Exception:
        #      traceback.print_exc()

if viewPeeler:
    for chan_grp in chansToAnalyze:
        print('\n\n\n\nOn channel group {}\n\n\n\n'.format(chan_grp))
        try:
            tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)
        except Exception:
            traceback.print_exc()
        gc.collect()
