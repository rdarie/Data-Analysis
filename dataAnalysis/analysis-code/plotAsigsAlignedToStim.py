"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --rowName=rowName               break down by row
    --rowControl=rowControl         rows to exclude from comparison
    --hueName=hueName               break down by hue  [default: amplitude]
    --hueControl=hueControl         hues to exclude from comparison
    --colName=colName               break down by col  [default: electrode]
    --colControl=colControl         cols to exclude from comparison [default: control]
    --alignQuery=alignQuery         what will the plot be aligned to? [default: (stimCat==\'stimOn\')]
    --window=window                 process with short window? [default: shortWindow]
"""

import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as preproc
import seaborn as sns
import os
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("white")

rowName = arguments['--rowName']
try:
    rowControl = int(arguments['--rowControl'])
except Exception:
    rowControl = arguments['--rowControl']
colName = arguments['--colName']
try:
    colControl = int(arguments['--colControl'])
except Exception:
    colControl = arguments['--colControl']
hueName = arguments['--hueName']
try:
    hueControl = int(arguments['--hueControl'])
except Exception:
    hueControl = arguments['--hueControl']

if arguments['--processAll']:
    dataBlock = preproc.loadWithArrayAnn(
        os.path.join(
            scratchFolder,
            experimentName + '_triggered_{}.nix'.format(
                arguments['--window'])))
    pdfName = '{}_{}_asigs_by_{}_aligned_to_{}'.format(
        experimentName, arguments['--window'],
        hueName, arguments['--alignQuery'])
else:
    dataBlock = preproc.loadWithArrayAnn(
        os.path.join(
            scratchFolder,
            ns5FileName + '_triggered_{}.nix'.format(
                arguments['--window'])))
    pdfName = '{}_{}_{}_asigs_by_{}_aligned_to_{}'.format(
        experimentName, arguments['--trialIdx'], arguments['--window'],
        hueName, arguments['--alignQuery'])

dataQuery = '&'.join([
    '((RateInHz==100)|(RateInHz==0))',
    arguments['--alignQuery']
    ])
testStride = 20e-3
testWidth = 100e-3
testTStart = 0
testTStop = 500e-3
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates
unitNames = [
    'amplitude#0', 'ins_td3#0', 'ins_td2#0']
#  unitNames = ['ins_td3#0', 'position#0']
asp.plotAsigsAligned(
    dataBlock,
    dataQuery=dataQuery,
    figureFolder=figureFolder,
    rowName=rowName,
    rowControl=rowControl,
    colName=colName,
    colControl=colControl,
    hueName=hueName,
    testStride=testStride,
    testWidth=testWidth,
    testTStart=testTStart,
    testTStop=testTStop,
    removeFuzzyName=False,
    pThresh=pThresh,
    #  linePlotEstimator=None,
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    amplitudeColumn='amplitude',
    programColumn='program',
    electrodeColumn='electrode',
    enablePlots=True,
    colorPal=colorPal,
    printBreakDown=True,
    pdfName=pdfName,
    unitNames=unitNames)
