"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --verbose                       print diagnostics? [default: True]
    --window=window                 process with short window? [default: short]
    --blockName=blockName           which trig_ block to pull [default: pca]
    --chanQuery=chanQuery           how to restrict channels? [default: (chanName.str.contains(\'pca\'))]
    --selector=selector             filename if using a unit selector
    --alignQuery=alignQuery         what will the plot be aligned to? [default: (pedalMovementCat==\'outbound\')]
    --nameSuffix=nameSuffix         add an identifier to the pdf name? [default: alignedToOutbound]
    --selector=selector             filename if using a unit selector
    --rowName=rowName               break down by row  [default: pedalDirection]
    --rowControl=rowControl         rows to exclude from comparison
    --hueName=hueName               break down by hue  [default: amplitudeCat]
    --hueControl=hueControl         hues to exclude from comparison
    --colName=colName               break down by col  [default: electrode]
    --colControl=colControl         cols to exclude from comparison [default: control]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as ns5
import seaborn as sns
import os
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
import dill as pickle
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("white")

rowName = arguments['--rowName'] if len(arguments['--rowName']) else None
if rowName is not None:
    try:
        rowControl = int(arguments['--rowControl'])
    except Exception:
        rowControl = arguments['--rowControl']
else:
    rowControl = None
colName = arguments['--colName'] if len(arguments['--colName']) else None
if colName is not None:
    try:
        colControl = int(arguments['--colControl'])
    except Exception:
        colControl = arguments['--colControl']
else:
    colControl = None
hueName = arguments['--hueName'] if len(arguments['--hueName']) else None
if hueName is not None:
    try:
        hueControl = int(arguments['--hueControl'])
    except Exception:
        hueControl = arguments['--hueControl']
else:
    hueControl = None

if arguments['--processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['--blockName'], arguments['--window']))
print('loading {}'.format(triggeredPath))
dataBlock = ns5.loadWithArrayAnn(triggeredPath)
pdfName = '{}_{}_{}_{}'.format(
    prefix, arguments['--blockName'], arguments['--window'],
    arguments['--nameSuffix'])

# during movement and stim
pedalSizeQuery = '(' + '|'.join([
    '(pedalSizeCat == \'{}\')'.format(i)
    for i in ['M', 'L', 'XL']
    ]) + ')'

dataQuery = '&'.join([
    '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))',
    #  '((amplitudeCatFuzzy>=2)|(amplitudeCatFuzzy==0))',
    #  pedalSizeQuery,
    #  '(pedalDirection == \'CW\')'
    ])

# dataQuery = '(amplitudeCatFuzzy==0)'
if arguments['--alignQuery'] is not None:
    if len(arguments['--alignQuery']):
        dataQuery = '&'.join([
            dataQuery,
            arguments['--alignQuery'],
            ])
if not len(arguments['--chanQuery']):
    arguments['--chanQuery'] = None
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
    enablePlots=True,
    colorPal=colorPal,
    printBreakDown=True,
    pdfName=pdfName,
    verbose=arguments['--verbose'])
