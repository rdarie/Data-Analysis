"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --verbose                       print diagnostics? [default: True]
    --rowName=rowName               break down by row  [default: pedalDirection]
    --rowControl=rowControl         rows to exclude from comparison
    --hueName=hueName               break down by hue  [default: amplitudeCat]
    --hueControl=hueControl         hues to exclude from comparison
    --colName=colName               break down by col  [default: electrode]
    --colControl=colControl         cols to exclude from comparison [default: control]
    --alignQuery=alignQuery         what will the plot be aligned to? [default: (pedalMovementCat==\'outbound\')]
    --window=window                 process with short window? [default: short]
    --chanQuery=chanQuery           how to restrict channels? [default: (chanName.str.contains(\'pca\'))]
    --blockName=blockName           which trig_ block to pull [default: pca]
"""

import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as ns5
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
    dataBlock = ns5.loadWithArrayAnn(
        os.path.join(
            scratchFolder,
            experimentName + '_trig_{}_{}.nix'.format(
                arguments['--blockName'], arguments['--window'])))
    pdfName = '{}_{}_{}_by_{}_aligned_to_{}'.format(
        experimentName, arguments['--blockName'],
        arguments['--window'], hueName, arguments['--alignQuery'])
else:
    dataBlock = ns5.loadWithArrayAnn(
        os.path.join(
            scratchFolder,
            ns5FileName + '_trig_{}_{}.nix'.format(
                arguments['--blockName'], arguments['--window'])))
    pdfName = '{}_{}_{}_{}_by_{}_aligned_to_{}'.format(
        experimentName, arguments['--trialIdx'],
        arguments['--blockName'], arguments['--window'],
        hueName, arguments['--alignQuery'])

# during movement and stim
pedalSizeQuery = '(' + '|'.join([
    '(pedalSizeCat == \'{}\')'.format(i)
    for i in ['M', 'L', 'XL']
    ]) + ')'

dataQuery = '&'.join([
    '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))',
    #  '((amplitudeCatFuzzy>=2)|(amplitudeCatFuzzy==0))',
    arguments['--alignQuery'],
    pedalSizeQuery,
    #  '(pedalDirection == \'CW\')'
    ])
testStride = 20e-3
testWidth = 100e-3
testTStart = 0
testTStop = 500e-3
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates

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
    pThresh=pThresh,
    #  linePlotEstimator=None,
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    enablePlots=True,
    colorPal=colorPal,
    printBreakDown=True,
    pdfName=pdfName,
    chanNames=None, chanQuery=arguments['--chanQuery'], verbose=arguments['--verbose'])
