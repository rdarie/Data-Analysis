"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --blockIdx=blockIdx              which trial to analyze
    --exp=exp                        which experimental day to analyze
    --verbose                        print statements? [default: False]
    --makePlots                      make diagnostic plots? [default: False]
    --showPlots                      show diagnostic plots? [default: False]
    --disableStimDetection           disable stimulation time detection? [default: False]
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate interactive qt output
# matplotlib.use('PS')   # generate offline postscript
import seaborn as sns
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=0.75, color_codes=True)
import dataAnalysis.preproc.mdt as preprocINS
import os
from importlib import reload
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

import line_profiler
import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

figureOutputFolder = os.path.join(
    figureFolder, 'insDiagnostics')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
#
if not arguments['makePlots']:
    trialFilesStim['ins']['getINSkwargs']['plotting'] = []

def preprocINSWrapper(
        trialFilesStim=None,
        insDataPath=None,
        figureOutputFolder=None,
        arguments=None
        ):
    if arguments['disableStimDetection']:
        trialFilesStim['ins']['detectStim'] = False
    insBlock = preprocINS.preprocINS(
        trialFilesStim['ins'],
        insDataPath, blockIdx=int(arguments['blockIdx']),
        figureOutputFolder=figureOutputFolder,
        verbose=arguments['verbose'],
        showPlots=arguments['showPlots'],
        makePlots=arguments['makePlots'])
    return


if __name__ == "__main__":
    preprocINSWrapper(
        trialFilesStim=trialFilesStim,
        insDataPath=insDataPath,
        figureOutputFolder=figureOutputFolder,
        arguments=arguments)
