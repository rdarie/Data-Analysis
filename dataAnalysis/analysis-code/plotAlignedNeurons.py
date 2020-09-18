"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --window=window                        process with short window? [default: short]
    --unitQuery=unitQuery                  how to restrict channels?
    --selector=selector                    filename if using a unit selector
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from comparison
    --hueName=hueName                      break down by hue  [default: amplitude]
    --enableOverrides                      delete outlier trials? [default: False]
    --hueControl=hueControl                hues to exclude from comparison
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=hueControl              styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from comparison [default: control]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --overlayStats                         overlay ANOVA significance stars? [default: False]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Agg')   # generate postscript output
matplotlib.use('QT5Agg')   # generate postscript output


import os
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import pandas as pd
import dill as pickle
import pdb
from copy import deepcopy
#
from currentExperiment import parseAnalysisOptions
from docopt import docopt
from namedQueries import namedQueries
import seaborn as sns
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
#
figureStatsFolder = os.path.join(
    alignSubFolder, 'figureStats'
    )
if not os.path.exists(figureStatsFolder):
    os.makedirs(figureStatsFolder, exist_ok=True)
alignedRastersFolder = os.path.join(
    figureFolder, arguments['analysisName'],
    'alignedRasters')
if not os.path.exists(alignedRastersFolder):
    os.makedirs(alignedRastersFolder, exist_ok=True)

rowColOpts = asp.processRowColArguments(arguments)
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    calcSubFolder, prefix, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=True,
    makeControlProgram=True,
    metaDataToCategories=False))
#
rasterBlockPath = os.path.join(
    alignSubFolder,
    prefix + '_raster_{}.nix'.format(
        arguments['window']))
print('loading {}'.format(rasterBlockPath))
rasterReader, rasterBlock = ns5.blockFromPath(
    rasterBlockPath, lazy=arguments['lazy'])
frBlockPath = os.path.join(
    alignSubFolder,
    prefix + '_fr_{}.nix'.format(
        arguments['window']))

print('loading {}'.format(frBlockPath))
frReader, frBlock = ns5.blockFromPath(
    frBlockPath, lazy=arguments['lazy'])
pdfName = '{}_neurons_{}_{}'.format(
    prefix,
    arguments['window'],
    arguments['alignQuery'])
statsTestPath = os.path.join(figureStatsFolder, pdfName + '_stats.h5')
statsTestOpts.update({
    'tStop': rasterOpts['windowSizes'][arguments['window']][1]})
#  Overrides
################################################################
limitPages = None
showNow = False
if arguments['enableOverrides']:
    nrnRelplotKWArgs.update({
        'legend': 'brief',
        'height': 4,
        'aspect': 2,
        'facet1_kws': {  # raster axes
            'sharey': False,
            # 'legend_out': False,
            'gridspec_kws': {
                'wspace': 0.01,
                'hspace': 0.01
            }},
        'facet2_kws': {  # firing rate axes
            'sharey': False,
            # 'legend_out': False,
            'gridspec_kws': {
                'wspace': 0.01,
                'hspace': 0.01
            }}
        })
    ##########################################################################
    alignedAsigsKWargs.update({'windowSize': (-.2, .5)})
    ##########################################################################
#     # currWindow = rasterOpts['windowSizes'][arguments['window']]
#     # fullWinSize = currWindow[1] - currWindow[0]
#     # redWinSize = (
#     #     alignedAsigsKWargs['windowSize'][1] -
#     #     alignedAsigsKWargs['windowSize'][0])
#     # nrnRelplotKWArgs['aspect'] = (
#     #     nrnRelplotKWArgs['aspect'] * redWinSize / fullWinSize)
#     # alignedAsigsKWargs.update({'decimate': 10})
#     # statsTestOpts.update({
#     #     'plotting': True,
#     #     'testStride': 500e-3,
#     #     'testWidth': 500e-3,
#     #     'tStart': -2000e-3,
#     #     'tStop': 2250e-3})
# #  End Overrides
# ################################################################

if arguments['overlayStats']:
    if os.path.exists(statsTestPath):
        sigValsWide = pd.read_hdf(statsTestPath, 'sig')
        sigValsWide.columns.name = 'bin'
    else:
        print('Recalculating t test results')
        alignedAsigsKWargsStats = deepcopy(alignedAsigsKWargs)
        if alignedAsigsKWargsStats['unitNames'] is not None:
            alignedAsigsKWargsStats['unitNames'] = [
                i.replace('_#0', '_raster#0')
                for i in alignedAsigsKWargsStats['unitNames']
            ]
        (
            pValsWide, statValsWide,
            sigValsWide) = ash.facetGridCompareMeans(
            rasterBlock, statsTestPath,
            limitPages=limitPages,
            compareISIs=True,
            loadArgs=alignedAsigsKWargsStats,
            rowColOpts=rowColOpts,
            statsTestOpts=statsTestOpts)
else:
    sigValsWide = None

asp.plotNeuronsAligned(
    rasterBlock,
    frBlock,
    limitPages=limitPages,
    showNow=showNow,
    verbose=arguments['verbose'],
    loadArgs=alignedAsigsKWargs,
    sigTestResults=sigValsWide,
    figureFolder=alignedRastersFolder,
    enablePlots=True,
    plotProcFuns=[
        # asp.genYLimSetterTwin((0, 150)),
        asp.genTicksToScaleTwin(
            lineOpts={'lw': 2}, shared=False,
            # for evoked lfp report
            # xUnitFactor=1e3, yUnitFactor=1,
            # xUnits='msec', yUnits='uV',
            # for evoked emg report
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='spk/s',
            ),
        asp.xLabelsTime, asp.genLegendRounder(decimals=2),
        asp.genDespiner(right=True, left=True, trim=True),
        asp.genVLineAdder([0], nrnVLineOpts),
        asp.genBlockShader(nrnBlockShadingOpts)
        ],
    pdfName=pdfName,
    **rowColOpts,
    twinRelplotKWArgs=nrnRelplotKWArgs, sigStarOpts=nrnSigStarOpts)
if arguments['overlayStats']:
    asp.plotSignificance(
        sigValsWide,
        pdfName=pdfName + '_pCount',
        figureFolder=alignedRastersFolder,
        **rowColOpts,
        **statsTestOpts)

if arguments['lazy']:
    frReader.close()
    rasterReader.close()
