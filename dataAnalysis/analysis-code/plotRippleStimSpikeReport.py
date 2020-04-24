#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    generateSpikeReport [options]

Options:
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --nameSuffix=nameSuffix                add anything to the output name?
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --inputBlockName=inputBlockName        which trig_ block to pull
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: isichoremg]
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierBlocks                  delete outlier trials? [default: False]
    --individualTraces                     mean+sem or individual traces? [default: False]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: all]
    --hueName=hueName                      break down by hue  [default: nominalCurrent]
    --hueControl=hueControl                hues to exclude from stats test
    --styleName=styleName                  break down by style [default: RateInHz]
    --styleControl=styleControl            styles to exclude from stats test
    --groupPagesBy=groupPagesBy            break down each page
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
#
from namedQueries import namedQueries
import neo
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as preproc
import pandas as pd
import numpy as np
from importlib import reload
import os, pdb
from tqdm import tqdm
#
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import seaborn as sns
#
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("dark")
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
if arguments['nameSuffix']:
    nameSuffix = arguments['nameSuffix']
else:
    nameSuffix = ''
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
if arguments['inputBlockName'] is not None:
    # plotting aligned features
    alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
    if not os.path.exists(alignSubFolder):
        os.makedirs(alignSubFolder, exist_ok=True)
    if arguments['processAll']:
        prefix = assembledName
    else:
        prefix = ns5FileName
    dataPath = os.path.join(
        alignSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['inputBlockName'], arguments['window']))
    reportName = prefix + '_{}_{}_{}_topo'.format(
            arguments['inputBlockName'], arguments['window'],
            arguments['alignQuery']) + nameSuffix
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'], 'alignedFeatures')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
else:
    # plotting aligned spike waveforms
    if arguments['processAll']:
        dataPath = experimentDataPath
        reportName = '_isi_spike_report' + nameSuffix
    else:
        dataPath = analysisDataPath
        reportName = ns5FileName + '_isi_spike_report' + nameSuffix
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
dataPath = dataPath.format(arguments['analysisName'])
pdfName = os.path.join(figureOutputFolder, reportName + '.pdf')
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
    namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
    ash.processUnitQueryArgs(
        namedQueries, analysisSubFolder, **arguments))
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    alignSubFolder, prefix, **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    metaDataToCategories=False,
    removeFuzzyName=False, decimate=1,
    transposeToColumns='feature', concatOn='columns',))
rippleMapDF = prb_meta.mapToDF(rippleMapFile)
rippleMapDF.loc[
    rippleMapDF['label'].str.contains('caudal'),
    'ycoords'] += 800

if 'delsysMapDict' in locals():
    delsysMapDF = pd.DataFrame(delsysMapDict)
    mapsDict = {
        'ripple': rippleMapDF,
        'delsys': delsysMapDF}
else:
    mapsDict = {
        'ripple': rippleMapDF}

#############################
# for stim spike report
# alignedAsigsKWargs.update(dict(
#     windowSize=(6e-5, 1.2e-3)))
# for evoked lfp report
# alignedAsigsKWargs.update(dict(
#     windowSize=(-350e-3, 350e-3)))
alignedAsigsKWargs.update(dict(
    windowSize=(-20e-3, 40e-3)))
alignedAsigsKWargs.update(dict(decimate=1))
flipInfo = {
    'ripple': {'lr': True, 'ud': True},
    'delsys': {'lr': False, 'ud': False}
    }
mapSpecificRelplotKWArgs = {
    'ripple': {
        'facet_kws': {
            'gridspec_kws': {
                'width_ratios': [
                    1, 0.2, 1, 1,
                    1, 0.2, 1, 1,
                    1, 0.2, 1, 1,
                    1, 0.2, 1],
                'height_ratios': [
                    1, 1, 0.2, 1,
                    1, 0.2, 1, 1]}}},
    'delsys': {
        'facet_kws': {
            'gridspec_kws': {
                'width_ratios': [
                    1, 1, 1, 1, 1],
                'height_ratios': [
                    1, 0.5, 1, 1, 0.5, 1]}}}
    }
relplotKWArgs.update({
    'legend': 'brief',
    # 'legend': False,
    'palette': "ch:0.6,.3,dark=.1,light=0.7,reverse=1",
    'height': 4,
    'aspect': 2,
    'facet_kws': {
        'sharey': False,
        'legend_out': False,
        'gridspec_kws': {
            'wspace': 0.01,
            'hspace': 0.01
        }}})
plotProcFuns = [
    # asp.genYLimSetter(quantileLims=0.99, forceLims=True),
    # asp.genYLabelChanger(
    #     lookupDict={}, removeMatch='#0'),
    # asp.genYLimSetter(newLims=[-75, 100], forceLims=True),
    # asp.xLabelsTime,
    asp.genLegendRounder(decimals=2),
    asp.genDespiner(
        top=True, right=True,
        left=True, bottom=True,
        offset=None, trim=False),
    asp.genGridAnnotator(
        xpos=1, ypos=1, template='{}', colNames=['feature'],
        textOpts={
            'verticalalignment': 'top',
            'horizontalalignment': 'right'
        }, shared=False),
    # for evoked lfp report, add stim times
    # asp.genBlockVertShader([
    #         max(0e-3, alignedAsigsKWargs['windowSize'][0]),
    #         min(300e-3, alignedAsigsKWargs['windowSize'][1])],
    #     asigPlotShadingOpts),
    asp.genVLineAdder(
        # np.arange(0, min(300e-3, alignedAsigsKWargs['windowSize'][1]), 10e-3),
        [0],
        vLineOpts)
        ]
mapSpecificPlotProcFuns = {
    'ripple': [
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=False,
            # for evoked lfp report
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            # for stim spike
            # xUnitFactor=1e3, yUnitFactor=1e-6,
            # xUnits='msec', yUnits='V',
            )
        ],
    'delsys': [
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=False,
            xUnitFactor=1e3, yUnitFactor=1e6,
            xUnits='msec', yUnits='mV',
            )]}
addSpacesFromMap = True
extraSpaces = {
    'ripple': [
        (34, 52), (85, 260),
        (34, 364), (34, 852),
        (85, 1060)],
    'delsys': [
        (1, 2), (4, 2)
        ]}
# pdb.set_trace()
limitPages = None
if arguments['individualTraces']:
    relplotKWArgs['alpha'] = 0.3
    relplotKWArgs['estimator'] = None
    relplotKWArgs['units'] = 't'
    pdfName = pdfName.replace('.pdf', '_traces.pdf')
if arguments['invertOutlierBlocks']:
    pdfName = pdfName.replace('.pdf', '_outliers.pdf')
################
# from here on we can start defining a function
print('loading {}'.format(dataPath))
dataReader, dataBlock = preproc.blockFromPath(
    dataPath, lazy=arguments['lazy'])
asigWide = preproc.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
#
asigStack = (
    asigWide
    .stack(level=['feature', 'lag'])
    .reset_index(name='signal')
    .dropna())
asigStack['parentChanName'] = (
    asigStack['feature']
    .apply(lambda x: x.replace('_stim#0', '').replace('#0', '')))
dummyDict = {}
for probeName, mapDF in mapsDict.items():
    probeMask = asigStack['parentChanName'].isin(mapDF['label'])
    for cName in ['xcoords', 'ycoords']:
        mapSer = pd.Series(
            mapDF[cName].to_numpy(),
            index=mapDF['label'])
        asigStack.loc[probeMask, cName] = (
            asigStack
            .loc[probeMask, 'parentChanName']
            .map(mapSer))
    asigStack.loc[probeMask, 'mapGroup'] = probeName
    dummyList = []
    if addSpacesFromMap:
        for name, group in mapDF.groupby(['xcoords', 'ycoords']):
            dummySer = pd.Series(np.nan, index=asigStack.columns)
            dummySer['xcoords'] = name[0]
            dummySer['ycoords'] = name[1]
            dummySer['mapGroup'] = probeName
            dummyList.append(dummySer)
    if extraSpaces[probeName] is not None:
        for name in extraSpaces[probeName]:
            dummySer = pd.Series(np.nan, index=asigStack.columns)
            dummySer['xcoords'] = name[0]
            dummySer['ycoords'] = name[1]
            dummySer['mapGroup'] = probeName
            dummyList.append(dummySer)
    if len(dummyList):
        dummyDict[probeName] = (
            pd.concat(dummyList, axis='columns', ignore_index=True)
            .transpose())

if arguments['groupPagesBy'] is None:
    pageGrouper = [('all', asigStack)]
else:
    pageGrouper = asigStack.groupby(arguments['groupPagesBy'])
#
with PdfPages(pdfName) as pdf:
    for pageIdx, (pageName, pageGroup) in enumerate(tqdm(pageGrouper)):
        if limitPages is not None:
            if pageIdx >= limitPages:
                break
        for probeName, probeGroup in pageGroup.groupby('mapGroup'):
            if pd.isnull(probeName):
                continue
            if probeName in dummyDict:
                thisAsigStack = pd.concat(
                    [probeGroup, dummyDict[probeName]],
                    ignore_index=True)
            else:
                thisAsigStack = probeGroup
            if flipInfo[probeName]['ud']:
                thisAsigStack['xcoords'] *= -1
            if flipInfo[probeName]['lr']:
                thisAsigStack['ycoords'] *= -1
            if 'facet_kws' in relplotKWArgs:
                if 'gridspec_kws' in relplotKWArgs['facet_kws']:
                    relplotKWArgs['facet_kws']['gridspec_kws'].update(
                        mapSpecificRelplotKWArgs[probeName]['facet_kws']['gridspec_kws'])
                    # workaround to get margins between axes to be spaced
                    # as a fraction of the *absolute* axis size
                    absWids = [
                        ratio * relplotKWArgs['aspect'] * relplotKWArgs['height']
                        for ratio in relplotKWArgs['facet_kws']['gridspec_kws']['width_ratios']]
                    updateWid = np.sum(absWids)
                    relplotKWArgs['facet_kws']['gridspec_kws']['wspace'] = (
                        relplotKWArgs['height'] * 0.1 / np.mean(absWids))
                    absHeights = [
                        ratio * relplotKWArgs['height']
                        for ratio in relplotKWArgs['facet_kws']['gridspec_kws']['height_ratios']]
                    updateHeight = np.sum(absHeights)
                    relplotKWArgs['facet_kws']['gridspec_kws']['hspace'] = (
                        relplotKWArgs['height'] * 0.1 / np.mean(absHeights))
            g = sns.relplot(
                data=thisAsigStack,
                x='bin', y='signal', hue=arguments['hueName'],
                row='xcoords', col='ycoords', **relplotKWArgs)
            for (ro, co, hu), dataSubset in g.facet_data():
                # if sigTestResults is not None:
                #     addSignificanceStars(
                #         g, sigTestResults.query(
                #             "unit == '{}'".format(unitName)),
                #         ro, co, hu, dataSubset, sigStarOpts=sigStarOpts)
                allProbePlotFuns = (plotProcFuns + mapSpecificPlotProcFuns[probeName])
                if len(allProbePlotFuns):
                    for procFun in allProbePlotFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.set_titles("")
            g.set_axis_labels("", "")
            if 'facet_kws' in relplotKWArgs:
                if 'gridspec_kws' in relplotKWArgs['facet_kws']:
                    g.fig.set_size_inches(updateWid, updateHeight)
            pageTitle = g.fig.suptitle(pageName)
            saveLegendOpts = {
                    'bbox_extra_artists': [pageTitle]}
            # contrived way of pushing legend outside without
            # resizing the figure
            allLegends = [
                a.get_legend()
                for a in g.axes.flat
                if a.get_legend() is not None]
            if len(allLegends):
                bb = matplotlib.transforms.Bbox([[-1.01, 0.01], [-0.01, 1.01]])
                allLegends[0].set_bbox_to_anchor(bb)
                saveLegendOpts.update({
                    'bbox_extra_artists': [pageTitle] + allLegends})
            # if not plt.rcParams['figure.constrained_layout.use']:
            #     g.fig.tight_layout(pad=0)
            pdf.savefig(bbox_inches='tight', pad_inches=0, **saveLegendOpts)
            # plt.show()
            plt.close()
# pdb.set_trace()
