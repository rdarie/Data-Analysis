#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    generateSpikeReport [options]

Options:
    --blockIdx=blockIdx                                  which trial to analyze [default: 1]
    --exp=exp                                            which experimental day to analyze
    --processAll                                         process entire experimental day? [default: False]
    --nameSuffix=nameSuffix                              add anything to the output name?
    --lazy                                               load from raw, or regular? [default: False]
    --window=window                                      process with short window? [default: short]
    --winStart=winStart                                  start of window [default: None]
    --winStop=winStop                                    end of window [default: None]
    --alignFolderName=alignFolderName                    append a name to the resulting blocks? [default: motion]
    --analysisName=analysisName                          append a name to the resulting blocks? [default: default]
    --inputBlockSuffix=inputBlockSuffix                  which trig_ block to pull
    --inputBlockPrefix=inputBlockPrefix                  which trig_ block to pull
    --unitQuery=unitQuery                                how to restrict channels if not supplying a list? [default: isispinaloremg]
    --maskOutlierBlocks                                  delete outlier trials? [default: False]
    --invertOutlierBlocks                                delete outlier trials? [default: False]
    --individualTraces                                   mean+sem or individual traces? [default: False]
    --alignQuery=alignQuery                              what will the plot be aligned to? [default: all]
    --hueName=hueName                                    break down by hue  [default: nominalCurrent]
    --hueControl=hueControl                              hues to exclude from stats test
    --sizeName=sizeName                                  break down by size
    --sizeControl=sizeControl                            sizes to exclude from stats test
    --styleName=styleName                                break down by style
    --styleControl=styleControl                          styles to exclude from stats test
    --groupPagesBy=groupPagesBy                          break down each page
    --amplitudeFieldName=amplitudeFieldName              what is the amplitude named? [default: nominalCurrent]
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
import dataAnalysis.helperFunctions.profiling as prf
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
import json
from importlib import reload
import os, pdb
from tqdm import tqdm
import dill as pickle
#
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import seaborn as sns
#
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)
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
#
if arguments['groupPagesBy'] is not None:
    groupPagesBy = arguments['groupPagesBy'].split(', ')
else:
    groupPagesBy = None
if arguments['inputBlockSuffix'] is not None:
    # plotting aligned features
    alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
    if not os.path.exists(alignSubFolder):
        os.makedirs(alignSubFolder, exist_ok=True)
    calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
    if not os.path.exists(calcSubFolder):
        os.makedirs(calcSubFolder, exist_ok=True)
    if arguments['processAll']:
        prefix = 'Block'
    else:
        prefix = ns5FileName
    dataPath = os.path.join(
        alignSubFolder,
        prefix + '_{}_{}.nix'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    cachedH5Path = os.path.join(
        calcSubFolder,
        prefix + '_{}_{}.h5'.format(
            arguments['inputBlockSuffix'], arguments['window']))
    reportName = prefix + '_{}_{}_{}_topo'.format(
            arguments['inputBlockSuffix'], arguments['window'],
            arguments['alignQuery']) + nameSuffix
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'],
        arguments['alignFolderName'], 'alignedFeatures')
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)
    alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
        scratchFolder, prefix, **arguments)
else:
    # plotting aligned spike waveforms
    if arguments['processAll']:
        dataPath = experimentDataPath
        reportName = '_isi_spike_report' + nameSuffix
    else:
        dataPath = analysisDataPath
        reportName = ns5FileName + '_isi_spike_report' + nameSuffix
    figureOutputFolder = figureFolder
dataPath = dataPath.format(arguments['analysisName'])
pdfName = os.path.join(figureOutputFolder, reportName + '.pdf')
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
    namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
    ash.processUnitQueryArgs(
        namedQueries, analysisSubFolder, **arguments))
#
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    metaDataToCategories=False,
    removeFuzzyName=False,
    getMetaData=[
        'RateInHz', 'feature', 'electrode',
        arguments['amplitudeFieldName'],
        'stimCat', 'originalIndex', 'segment', 't'],
    transposeToColumns='bin', concatOn='index'))
#
#############################
# for stim spike report
# alignedAsigsKWargs.update(dict(
#     windowSize=(6e-5, 1.2e-3)))
# for evoked lfp report
# alignedAsigsKWargs.update(dict(
#     windowSize=(-2e-3, 10e-3)))
# for evoked emg report
alignedAsigsKWargs.update(dict(
    windowSize=(-20e-3, 100e-3)))
# alignedAsigsKWargs.update(dict(
#     windowSize=(-25e-3, 125e-3)))
if 'windowSize' not in alignedAsigsKWargs:
    alignedAsigsKWargs['windowSize'] = list(rasterOpts['windowSizes'][arguments['window']])
if arguments['winStart'] == 'None':
    alignedAsigsKWargs['windowSize'][0] = float(arguments['winStart']) * (1e-3)
if arguments['winStop'] == 'None':
    alignedAsigsKWargs['windowSize'][1] = float(arguments['winStop']) * (1e-3)

alignedAsigsKWargs.update({'amplitudeColumn': arguments['amplitudeFieldName']})
#
if 'proprio' in blockExperimentType:
    electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
    mapExt = electrodeMapPath.split('.')[-1]
    if mapExt == 'cmp':
        mapDF = prb_meta.cmpToDF(electrodeMapPath)
    elif mapExt == 'map':
        mapDF = prb_meta.mapToDF(electrodeMapPath)
else:
    rippleMapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
    rippleMapDF.loc[
        rippleMapDF['label'].str.contains('caudal'),
        'ycoords'] += 800
    #
    if 'delsysMapDict' in locals():
        delsysMapDF = pd.DataFrame(delsysMapDict)
        mapsDict = {
            'ripple': rippleMapDF,
            'delsys': delsysMapDF}
    else:
        mapsDict = {
            'ripple': rippleMapDF}
    
    flipInfo = {
        'ripple': {'lr': True, 'ud': False},
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
                    1, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 1, 0.5, 1]}}}
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
sharedYAxes = relplotKWArgs['facet_kws']['sharey']
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
    asp.genBlockVertShader([
            max(0e-3, alignedAsigsKWargs['windowSize'][0]),
            min(.733e-3, alignedAsigsKWargs['windowSize'][1])],
        asigPlotShadingOpts),
    asp.genStimVLineAdder(
        'RateInHz', vLineOpts, tOnset=0, tOffset=.3, includeRight=False)
        ]
mapSpecificPlotProcFuns = {
    'ripple': [
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=sharedYAxes,
            # for evoked lfp/emg report
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            # for stim spike
            # xUnitFactor=1e3, yUnitFactor=1e-6,
            # xUnits='msec', yUnits='V',
            )
        ],
    'delsys': [
        asp.genTicksToScale(
            lineOpts={'lw': 2}, shared=sharedYAxes,
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            )]}
addSpacesFromMap = True
extraSpaces = {
    'ripple': [
        (34, 52), (85, 260),
        (34, 364), (34, 852),
        (85, 1060)],
    'delsys': [
        (2, 2), (4, 2), (6, 2), (8, 2), (11, 2)
        ]}
#
limitPages = None

minNObservations = 5
if arguments['individualTraces']:
    relplotKWArgs['alpha'] = 0.3
    relplotKWArgs['estimator'] = None
    relplotKWArgs['units'] = 't'
    pdfName = pdfName.replace('.pdf', '_traces.pdf')
if arguments['invertOutlierBlocks']:
    pdfName = pdfName.replace('.pdf', '_outliers.pdf')
#
saveFigMetaToPath = pdfName.replace('.pdf', '_metadata.pickle')
####################
'''optsSourceFolder = os.path.join(
    figureFolder, 'default', 'alignedFeatures')
loadFigMetaPath = os.path.join(
    optsSourceFolder,
    '_emg_XS_stimOn_topo_metadata.pickle'
    )
with open(loadFigMetaPath, 'rb') as _f:
    loadedFigMeta = pickle.load(_f)'''
# plotProcFuns.append(

#     asp.genAxLimSaver(
#         filePath=saveAxLimsToPath, keyColName='feature')
#     )
################
# from here on we can start defining a function
# TODO delete this and rework, right now it is very hacky
useCached = False
if useCached:
    print('loading {}'.format(cachedH5Path))
    asigWide = pd.read_hdf(cachedH5Path, 'clean')
    tMask = (asigWide.columns > alignedAsigsKWargs['windowSize'][0]) & (asigWide.columns < alignedAsigsKWargs['windowSize'][-1])
    asigWide = asigWide.loc[:, tMask]
    # pdb.set_trace()
else:
    print('loading {}'.format(dataPath))
    dataReader, dataBlock = preproc.blockFromPath(
        dataPath, lazy=arguments['lazy'])
    asigWide = preproc.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
#
prf.print_memory_usage('loaded asig wide')
# TODO check that these actually should be here
if 'nominalCurrentCat' in asigWide.index.names:
    asigWide.index = asigWide.index.droplevel('nominalCurrentCat')
# asigStack = (
#     asigWide
#     .stack(level=['feature', 'lag'])
#     .reset_index(name='signal')
#     .dropna())
trialInfo = asigWide.index.to_frame().reset_index(drop=True)
#
if minNObservations is not None:
    nObsCountedFeatures = ['feature']
    for extraName in groupPagesBy + [arguments['hueName'], arguments['sizeName'], arguments['styleName']]:
        if extraName is not None:
            if extraName not in nObsCountedFeatures:
                nObsCountedFeatures.append(extraName)
    nObs = trialInfo.groupby(nObsCountedFeatures).count().iloc[:, 0].to_frame(name='obsCount')
    nObs['keepMask'] = nObs['obsCount'] > minNObservations
    #
    def lookupKeep(x):
        keepVal = nObs.loc[tuple(x.loc[nObsCountedFeatures]), 'keepMask']
        # print(keepVal)
        return(keepVal)
    #
    keepMask = trialInfo.apply(lookupKeep, axis=1).to_numpy()
    asigWide = asigWide.loc[keepMask, :]
    trialInfo = asigWide.index.to_frame().reset_index(drop=True)

trialInfo['parentChanName'] = (
    trialInfo['feature']
    .apply(lambda x: x.replace('_stim#0', '').replace('#0', '')))
dummyDict = {}

for probeName, mapDF in mapsDict.items():
    probeMask = trialInfo['parentChanName'].isin(mapDF['label'])
    for cName in ['xcoords', 'ycoords']:
        mapSer = pd.Series(
            mapDF[cName].to_numpy(),
            index=mapDF['label'])
        trialInfo.loc[probeMask, cName] = (
            trialInfo
            .loc[probeMask, 'parentChanName']
            .map(mapSer))
    trialInfo.loc[probeMask, 'mapGroup'] = probeName
    dummyList = []
    if addSpacesFromMap:
        for name, group in mapDF.groupby(['xcoords', 'ycoords']):
            dummySer = pd.Series(np.nan, index=trialInfo.columns)
            dummySer['xcoords'] = name[0]
            dummySer['ycoords'] = name[1]
            dummySer['mapGroup'] = probeName
            dummyList.append(dummySer)
    if extraSpaces[probeName] is not None:
        for name in extraSpaces[probeName]:
            dummySer = pd.Series(np.nan, index=trialInfo.columns)
            dummySer['xcoords'] = name[0]
            dummySer['ycoords'] = name[1]
            dummySer['mapGroup'] = probeName
            dummyList.append(dummySer)
    if len(dummyList):
        dummyDF = (
            pd.concat(dummyList, axis='columns', ignore_index=True)
            .transpose())
        dummyDF['bin'] = np.nan
        dummyDF['signal'] = np.nan
        # dummySeries = pd.Series(np.nan, index=pd.MultiIndex.from_frame(dummyDF))
        dummyDict[probeName] = dummyDF

asigWide.index = pd.MultiIndex.from_frame(trialInfo)
if groupPagesBy is None:
    pageGrouper = [('all', asigWide)]
else:
    pageGrouper = asigWide.groupby(groupPagesBy)
#
pageCount = 0
if saveFigMetaToPath is not None:
    figMetaData = []
with PdfPages(pdfName) as pdf:
    for pageIdx, (pageName, pageGroup) in enumerate(tqdm(pageGrouper)):
        #
        if limitPages is not None:
            if pageCount > limitPages:
                break
        for probeName, probeGroup in pageGroup.groupby('mapGroup'):
            if pd.isnull(probeName):
                continue
            thisAsigStack = (
                probeGroup.stack()
                .reset_index(name='signal')
                .dropna())
            if probeName in dummyDict:
                thisAsigStack = pd.concat(
                    [thisAsigStack, dummyDict[probeName]],
                    ignore_index=True
                    )
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
            # pdb.set_trace()
            g = sns.relplot(
                data=thisAsigStack,
                x='bin', y='signal', hue=arguments['hueName'],
                size=arguments['sizeName'], style=arguments['styleName'],
                row='xcoords', col='ycoords', **relplotKWArgs)
            for (ro, co, hu), dataSubset in g.facet_data():
                emptySubset = (
                    (dataSubset.empty) or
                    (dataSubset['segment'].isna().all()))
                # if sigTestResults is not None:
                #     addSignificanceStars(
                #         g, sigTestResults.query(
                #             "unit == '{}'".format(unitName)),
                #         ro, co, hu, dataSubset, sigStarOpts=sigStarOpts)
                allProbePlotFuns = (plotProcFuns + mapSpecificPlotProcFuns[probeName])
                if len(allProbePlotFuns):
                    for procFun in allProbePlotFuns:
                        procFun(g, ro, co, hu, dataSubset)
                if saveFigMetaToPath is not None:
                    if 'facetMetaList' not in locals():
                        facetMetaList = []
                    if not emptySubset:
                        facetMeta = {}
                        for gVarName in [g._row_var, g._col_var, g._hue_var]:
                            if gVarName is not None:
                                gVarValue = dataSubset[gVarName].dropna().unique()
                                assert isinstance(gVarValue, np.ndarray)
                                assert gVarValue.shape == (1,)
                                facetMeta.update({gVarName: gVarValue[0]})
                        gFeatName = dataSubset['feature'].dropna().unique()
                        if isinstance(gFeatName, np.ndarray):
                            if gFeatName.shape == (1,):
                                facetMeta.update({'feature': gFeatName[0]})
                        facetMeta.update({
                            'xlim_left': g.axes[ro, co].get_xlim()[0],
                            'xlim_right': g.axes[ro, co].get_xlim()[1],
                            'ylim_bottom': g.axes[ro, co].get_ylim()[0],
                            'ylim_top': g.axes[ro, co].get_ylim()[1]
                            })
                        facetMetaList.append(pd.Series(facetMeta))
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
            if saveFigMetaToPath is not None:
                pageFullIdx = {
                    key: value
                    for key, value in zip(
                        groupPagesBy + ['probe'], list(pageName) + [probeName])
                    }
                facetMetaDF = pd.concat(facetMetaList, axis='columns')
                facetMetaDF.pageFullIdx = pageFullIdx
                figMetaData.append(facetMetaDF.T)
                del facetMetaList
            #     g.fig.tight_layout(pad=0)
            pdf.savefig(bbox_inches='tight', pad_inches=0, **saveLegendOpts)
            # plt.show()
            plt.close()
            pageCount += 1
    if saveFigMetaToPath is not None:
        with open(saveFigMetaToPath, 'wb') as _f:
            pickle.dump(figMetaData, _f)
