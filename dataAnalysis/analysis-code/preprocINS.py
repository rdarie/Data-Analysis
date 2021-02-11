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
    --outputSuffix=outputSuffix      append a string to the resulting filename?
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
import dataAnalysis.preproc.mdt as mdt
import os, pdb
from importlib import reload
import warnings
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
# warnings.filterwarnings("error")
figureOutputFolder = os.path.join(
    figureFolder, 'insDiagnostics')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
#
if not arguments['makePlots']:
    trialFilesStim['ins']['getINSkwargs']['plotting'] = []

if arguments['outputSuffix'] is not None:
    insDataPath = insDataPath.replace(
        '.nix',
        '_{}.nix'.format(arguments['outputSuffix'])
        )
    outputSuffix = '_{}'.format(arguments['outputSuffix'])
else:
    outputSuffix = ''

def preprocINSWrapper(
        trialFilesStim=None,
        insDataPath=None,
        figureOutputFolder=None,
        arguments=None
        ):
    # pdb.set_trace()
    jsonSessionNames = trialFilesStim['ins'].pop('jsonSessionNames')
    trialFilesStim = trialFilesStim['ins']
    if arguments['disableStimDetection']:
        trialFilesStim['detectStim'] = False
    for jsn in jsonSessionNames:
        trialFilesStim['jsonSessionNames'] = [jsn]
        insDataPath = os.path.join(
            scratchFolder, '{}{}.nix'.format(jsn, outputSuffix))
        insBlock = mdt.preprocINS(
            trialFilesStim,
            insDataPath,
            # blockIdx=int(arguments['blockIdx']),
            blockIdx='{}{}'.format(jsn, outputSuffix),
            deviceName=deviceName,
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
