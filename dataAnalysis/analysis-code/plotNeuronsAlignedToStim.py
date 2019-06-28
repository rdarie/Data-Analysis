"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --window=window                 process with short window? [default: short]
    --chanQuery=chanQuery           how to restrict channels?
    --selector=selector             filename if using a unit selector
    --alignQuery=alignQuery         what will the plot be aligned to? [default: (stimCat==\'stimOn\')]
    --nameSuffix=nameSuffix         add an identifier to the pdf name? [default: alignedToStimOn]
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
import dill as pickle
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
rasterBlock = preproc.loadWithArrayAnn(
    os.path.join(
        scratchFolder,
        prefix + '_raster_{}.nix'.format(
            arguments['--window'])))
frBlock = preproc.loadWithArrayAnn(
    os.path.join(
        scratchFolder,
        prefix + '_fr_{}.nix'.format(
            arguments['--window'])))
pdfName = '{}_{}_neurons_{}'.format(
    prefix,
    arguments['--window'],
    arguments['--nameSuffix'])

dataQuery = '&'.join([
    '((RateInHz==100)|(RateInHz==0))',
    #  '((amplitudeCatFuzzy==3)|(amplitudeCatFuzzy==0))',
    arguments['--alignQuery']
    ])
testStride = 20e-3
testWidth = 100e-3
testTStart = 0
testTStop = 5
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates

if arguments['--selector'] is not None:
    with open(
        os.path.join(
            scratchFolder,
            arguments['--selector'] + '.pickle'),
            'rb') as f:
        selectorMetadata = pickle.load(f)
    unitNames = [
        i.replace(selectorMetadata['inputBlockName'] + '#0', '')
        for i in selectorMetadata['outputFeatures']]
else:
    unitNames = None

asp.plotNeuronsAligned(
    rasterBlock,
    frBlock,
    dataQuery=dataQuery,
    chanNames=unitNames, chanQuery=arguments['--chanQuery'],
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
    amplitudeColumn='amplitude',
    programColumn='program',
    electrodeColumn='electrode',
    pThresh=pThresh,
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    enablePlots=True,
    colorPal=colorPal,
    printBreakDown=True,
    pdfName=pdfName)
