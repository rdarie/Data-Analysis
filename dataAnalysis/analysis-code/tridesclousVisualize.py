"""
Usage:
    tridesclousCCV.py [--trialIdx=trialIdx] [--peeler] [--constructor]

Arguments:
    trialIdx           which trial to analyze

Options:
    --peeler           visualize Peeler results
    --constructor      visualize Catalogue Constructor Results
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

viewPeeler = False
if arguments['--peeler']:
    viewPeeler = arguments['--peeler']
viewConstructor = False
if arguments['--constructor']:
    viewPeeler = arguments['--constructor']

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

chansToAnalyze = [0, 10, 20, 30, 40, 50, 60, 70, 90]

if viewConstructor:
    for chan_grp in chansToAnalyze:
        tdch.open_cataloguewindow(triFolder, chan_grp=chan_grp)

if viewPeeler:
    for chan_grp in chansToAnalyze:
        tdch.open_PeelerWindow(triFolder, chan_grp=chan_grp)
