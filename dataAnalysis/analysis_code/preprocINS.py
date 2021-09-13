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
import os, pdb, traceback, sys
from importlib import reload
import warnings
#  load options
from currentExperiment import parseAnalysisOptions

from datetime import datetime
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
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
    jsonSessionNames = trialFilesStim['ins'].pop('jsonSessionNames')
    trialFilesStim = trialFilesStim['ins']
    if arguments['disableStimDetection']:
        trialFilesStim['detectStim'] = False
    for jsnIdx, jsn in enumerate(jsonSessionNames):
        try:
            overrideStartTimes = stimDetectOverrideStartTimes[blockIdx][jsnIdx]
            trialFilesStim['getINSkwargs']['overrideStartTimes'] = overrideStartTimes
        except Exception:
            traceback.print_exc()
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
    print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
