"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierBlocks                  delete everything *except* outlier trials? [default: False]
    --enableOverrides                      modify default plot opts? [default: False]
    --individualTraces                     mean+sem or individual traces? [default: False]
    --overlayStats                         overlay ANOVA significance stars? [default: False]
    --recalcStats                          overlay ANOVA significance stars? [default: False]
    --rowName=rowName                      break down by row  [default: pedalDirection]
    --rowControl=rowControl                rows to exclude from stats test
    --hueName=hueName                      break down by hue  [default: amplitude]
    --hueControl=hueControl                hues to exclude from stats test
    --sizeName=sizeName                    break down by hue  [default: RateInHz]
    --sizeControl=sizeControl              hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=styleControl            styles to exclude from stats test
    --colName=colName                      break down by col  [default: electrode]
    --colControl=colControl                cols to exclude from stats test [default: control]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --winStart=winStart                    start of window [default: 200]
    --winStop=winStop                      end of window [default: 400]
    --limitPages=limitPages                how many pages to print, max?
    --noStim                               disable references to "amplitude"
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Agg')   # generate postscript output
matplotlib.use('QT5Agg')   # generate postscript output


from namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if 'rowColOverrides' in locals():
    arguments['rowColOverrides'] = rowColOverrides

#############################################
#  room for custom code
#############################################
#
#
minNObservations = 3
plotProcFuns = [
    asp.genTicksToScale(
        lineOpts={'lw': 2}, shared=True,
        # for evoked lfp report
        # xUnitFactor=1e3, yUnitFactor=1,
        # xUnits='msec', yUnits='uV',
        # for evoked emg report
        xUnitFactor=1e3, xUnits='msec',
        # yUnitFactor=1, yUnits='uV',
        yUnitFactor=1e3, yUnits='uA/mm^3',
        ),
    asp.genYLabelChanger(
        lookupDict={}, removeMatch='#0'),
    # asp.genYLimSetter(newLims=[-75, 100], forceLims=True),
    asp.genYLimSetter(quantileLims=0.95, forceLims=True),
    asp.xLabelsTime,
    # asp.genStimVLineAdder(
    #     'RateInHz', vLineOpts, tOnset=0, tOffset=.3, includeRight=False),
    # asp.genVLineAdder([0], nrnVLineOpts),
    asp.genLegendRounder(decimals=2),
    ]
statsTestOpts = dict(
    testStride=25e-3,
    testWidth=25e-3,
    tStart=-100e-3,
    tStop=None,
    pThresh=5e-2,
    correctMultiple=False
    )
# alignedAsigsKWargs['procFun'] = ash.genDetrender(timeWindow=(-200e-3, -100e-3))
#
#############################################

blockBaseName, analysisSubFolder, alignSubFolder, figureStatsFolder, alignedFeaturesFolder, calcSubFolder, triggeredPath, pdfName, statsTestPath = asp.processFigureFolderTree(
    arguments, scratchFolder, figureFolder)
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
rowColOpts, alignedAsigsKWargs, statsTestOpts = asp.processFigureLoadArgs(
    arguments, namedQueries, analysisSubFolder, calcSubFolder, blockBaseName,
    rasterOpts, alignedAsigsKWargs, statsTestOpts)

#############################################
#  room for custom code
#############################################
#
#
plotProcFuns.append(asp.genXLimSetter(alignedAsigsKWargs['windowSize']))
plotProcFuns.append(
    asp.genBlockVertShader([
            max(0e-3, alignedAsigsKWargs['windowSize'][0]),
            min(1000e-3, alignedAsigsKWargs['windowSize'][1])],
        asigPlotShadingOpts),)
#
#
#############################################

relplotKWArgs, minNObservations, limitPages = asp.processRelplotKWArgs(
    relplotKWArgs, minNObservations, arguments, alignedAsigsKWargs,
    _rasterOpts=rasterOpts, changeRelPlotAspectRatio=False)

#############################################
#  room for custom code
#############################################
#
#
relplotUpdates = {
    'legend': 'brief',
    'height': 4,
    'aspect': 2,
    'facet_kws': {
        'sharey': True,
        # 'legend_out': False,
        'gridspec_kws': {
            'wspace': 0.01,
            'hspace': 0.01
        }}
    }
if 'kcsd' in arguments['inputBlockSuffix']:
    relplotUpdates.update({
        'palette': "ch:0.6,.3,dark=.1,light=0.7,reverse=1"
        })
relplotKWArgs.update(relplotUpdates)
#
#
#############################################

asp.plotAsigsAlignedWrapper(
    arguments, statsTestPath, dataReader, dataBlock,
    alignedAsigsKWargs, rowColOpts, limitPages, statsTestOpts,
    alignedFeaturesFolder, minNObservations, plotProcFuns,
    pdfName, relplotKWArgs, asigSigStarOpts,
    )
