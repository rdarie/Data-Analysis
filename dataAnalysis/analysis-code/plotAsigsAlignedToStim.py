"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: short]
    --blockName=blockName           name for new block [default: fr]
    --chanQuery=chanQuery           how to restrict channels? [default: (chanName.str.endswith(\'fr\'))]
    --selector=selector             filename if using a unit selector
    --alignQuery=alignQuery         what will the plot be aligned to? [default: (stimCat==\'stimOn\')]
    --nameSuffix=nameSuffix         add an identifier to the pdf name? [default: alignedToStimOn]
    --selector=selector             filename if using a unit selector
    --rowName=rowName               break down by row
    --rowControl=rowControl         rows to exclude from comparison
    --hueName=hueName               break down by hue  [default: amplitude]
    --hueControl=hueControl         hues to exclude from comparison
    --colName=colName               break down by col  [default: electrode]
    --colControl=colControl         cols to exclude from comparison [default: control]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default

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
    prefix = experimentName
else:
    prefix = ns5FileName

dataBlock = preproc.loadWithArrayAnn(
    os.path.join(
        scratchFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['--blockName'], arguments['--window'])))
pdfName = '{}_{}_{}_{}'.format(
    prefix,
    arguments['--blockName'], arguments['--window'],
    arguments['--nameSuffix'])

dataQuery = '&'.join([
    '((RateInHz==100)|(RateInHz==0))',
    arguments['--alignQuery']
    ])
testStride = 20e-3
testWidth = 100e-3
testTStart = 0
testTStop = 500e-3
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates

if arguments['--selector'] is not None:
    with open(
        os.path.join(
            scratchFolder,
            arguments['--selector'] + '.pickle'),
            'rb') as f:
        selectorMetadata = pickle.load(f)
    chanNames = [
        i.replace(selectorMetadata['inputBlockName'], '')
        for i in selectorMetadata['outputFeatures']]
else:
    chanNames = None

asp.plotAsigsAligned(
    dataBlock,
    dataQuery=dataQuery,
    chanNames=chanNames, chanQuery=arguments['--chanQuery'],
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
    pThresh=pThresh,
    #  linePlotEstimator=None,
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    amplitudeColumn='amplitude',
    programColumn='program',
    electrodeColumn='electrode',
    removeFuzzyName=False,
    enablePlots=True,
    colorPal=colorPal,
    printBreakDown=True,
    pdfName=pdfName)
