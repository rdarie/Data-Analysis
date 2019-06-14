"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --rowName=rowName               break down by row  [default: pedalDirection]
    --rowControl=rowControl         rows to exclude from comparison
    --hueName=hueName               break down by hue  [default: amplitudeCatFuzzy]
    --hueControl=hueControl         hues to exclude from comparison
    --colName=colName               break down by col  [default: electrodeFuzzy]
    --colControl=colControl         cols to exclude from comparison [default: control]
    --processShort                  process with short window? [default: False]
"""
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as preproc
import seaborn as sns

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
    if arguments['--processShort']:
        dataBlock = preproc.loadWithArrayAnn(
            experimentTriggeredShortPath)
        pdfName = '{}_short_asigs_by_{}'.format(
            experimentName, hueName)
    else:
        dataBlock = preproc.loadWithArrayAnn(
            experimentTriggeredLongPath)
        pdfName = '{}_long_asigs_by_{}'.format(
            experimentName, hueName)
else:
    if arguments['--processShort']:
        dataBlock = preproc.loadWithArrayAnn(
            trialTriggeredShortPath)
        pdfName = '{}_{}_short_asigs_by_{}'.format(
            experimentName, arguments['--trialIdx'], hueName)
    else:
        dataBlock = preproc.loadWithArrayAnn(
            trialTriggeredLongPath)
        pdfName = '{}_{}_long_asigs_by_{}'.format(
            experimentName, arguments['--trialIdx'], hueName)

# during movement and stim
pedalSizeQuery = '(' + '|'.join([
    '(pedalSizeCat == \'{}\')'.format(i)
    for i in ['M', 'L', 'XL']
    ]) + ')'

dataQuery = '&'.join([
    '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))',
    #  '((amplitudeCatFuzzy>=2)|(amplitudeCatFuzzy==0))',
    '(pedalMovementCat==\'outbound\')',
    pedalSizeQuery,
    #  '(pedalDirection == \'CW\')'
    ])
testStride = 20e-3
testWidth = 100e-3
testTStart = 0
testTStop = 500e-3
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates
unitNames = [
    'amplitude#0', 'ins_td2#0', 'ins_td3#0',
    'position#0', 'program#0', 'velocityCat#0']
#  unitNames = None  # ['ins_td3#0', 'position#0']
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
    unitNames=unitNames)
