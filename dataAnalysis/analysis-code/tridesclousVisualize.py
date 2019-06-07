"""02: Curate constructor clusters
Usage:
    tridesclousVisualize [options]

Options:
    --trialIdx=trialIdx        which trial to analyze [default: 1]
    --chan_start=chan_start    which chan_grp to start on [default: 0]
    --chan_stop=chan_stop      which chan_grp to stop on [default: 96]
    --peeler                   visualize Peeler results
    --constructor              visualize Catalogue Constructor Results
"""

from docopt import docopt
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import dataAnalysis.helperFunctions.helper_functions as hf
from currentExperiment import *
#from exp201901211000 import *
import os, gc, traceback

arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    triFolder = os.path.join(
        nspFolder, 'tdc_' + ns5FileName)

viewPeeler = False
if arguments['--peeler']:
    viewPeeler = arguments['--peeler']
viewConstructor = False
if arguments['--constructor']:
    viewConstructor = arguments['--constructor']

chan_start = int(arguments['--chan_start'])
chan_stop = int(arguments['--chan_stop'])
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

#  chansToAnalyze = chansToAnalyze[80:]
#  chansToAnalyze = [1, 8, 43, 46, 50, 53, 57, 59, 62, 67, 73, 74, 82, 84, 90, 91]
chansToAnalyze = [34, 81, 84, 92]
print(chansToAnalyze)
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
        tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)
        gc.collect()
