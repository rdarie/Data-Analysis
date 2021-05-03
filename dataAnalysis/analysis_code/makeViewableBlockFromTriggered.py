"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                    which experimental day to analyze
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --processAll                                 process entire experimental day? [default: False]
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --window=window                              process with short window? [default: short]
    --arrayName=arrayName                        name of electrode array? (for map file) [default: utah]
    --inputBlockSuffix=inputBlockSuffix          which block to pull
    --inputBlockPrefix=inputBlockPrefix          which block to pull [default: Block]
    --useKCSD                                    which algorithm to use [default: False]
    --lazy                                       load from raw, or regular? [default: False]
    --unitQuery=unitQuery                        how to restrict channels?
    --alignQuery=alignQuery                      choose a subset of the data? [default: all]
    --verbose                                    print diagnostics? [default: False]
    --plotting                                   plot results?
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import seaborn as sns
from docopt import docopt
from neo.io import NixIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
import os, gc
import pdb, traceback
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.preproc.ns5 as ns5
from dataAnalysis.lfpAnalysis import csd
from scipy.stats import zscore, chi2
from scipy import interpolate, ndimage, signal
# import pingouin as pg
import pandas as pd
import numpy as np
from elephant import current_source_density as elph_csd
from kcsd import KCSD2D
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from copy import deepcopy
from tqdm import tqdm
from random import sample
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
import quantities as pq

sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

#
blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)

if arguments['plotting']:
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)

calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
#
triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))
outputPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}_viewable.nix'.format(
        inputBlockSuffix, arguments['window']))

reqUnitNames, unitQuery = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
outlierTrials = ash.processOutlierTrials(
    scratchPath, blockBaseName, **arguments)

#####
# DEBUG_ARRAY_RESHAPING = False
#####


if __name__ == "__main__":
    print('loading {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    if reqUnitNames is None:
        reqUnitNames = ns5.listChanNames(
            dataBlock, unitQuery,
            objType=Unit)
    unitList = [
        un
        for un in dataBlock.filter(objects=Unit)
        if un.name in reqUnitNames]
    unitNames = [ci.name for ci in unitList]
    #
    dummyStList = []
    for segIdx, seg in enumerate(dataBlock.segments):
        stLikeList = [
            un.spiketrains[segIdx]
            for un in unitList
            ]
        if arguments['lazy']:
            stLikeList = [
                un.spiketrains[segIdx]
                for un in unitList]
            dummySt = ns5.loadObjArrayAnn(ns5.loadStProxy(stLikeList[0]))
        else:
            stLikeList = [
                un.spiketrains[segIdx]
                for un in unitList]
            dummySt = stLikeList[0]
        dummyStList.append(dummySt)
        originalDummyUnitAnns = dummySt.annotations.pop('unitAnnotations', None)
        if 'chanIndex' not in locals():
            '''
            xcoords, ycoords = [], []
            for stIdx, st in enumerate(stLikeList):
                xcoords.append(st.annotations['xCoords'] * 400)
                ycoords.append(st.annotations['yCoords'] * 400)
                assert ns5.childBaseName(st.name, 'seg') == unitNames[stIdx]
            coordinates = np.concatenate(
                [np.asarray(xcoords)[:, np.newaxis], np.asarray(ycoords)[:, np.newaxis]],
                axis=1) * pq.um
            #
            xIdx, yIdx = ssplt.coordsToIndices(
                coordinates[:, 0], coordinates[:, 1])
            coordinateIndices = np.concatenate(
                [xIdx[:, np.newaxis], yIdx[:, np.newaxis]],
                axis=1)
            '''
            chanIndex = ChannelIndex(
                name=arguments['arrayName'],
                index=np.arange(len(unitNames)),
                channel_ids=np.arange(len(unitNames)),
                channel_names=unitNames,
                # coordinates=coordinates,
                # coordinateIndices=coordinateIndices
                )
            '''
            annotationsLong = pd.DataFrame(
                chanIndex.coordinates, columns=['x', 'y'])
            annotationsLong.loc[:, ['xIdx', 'yIdx']] = coordinateIndices
            annotationsLong.loc[:, 'chanName'] = unitNames
            annotations2D = {
                key: annotationsLong.pivot(index='y', columns='x', values=key)
                for key in ['chanName']}
            '''
        #
        nTrials = dummySt.waveforms.shape[0]
        nBins = dummySt.waveforms.shape[2]
        allWaveforms = np.zeros((nTrials * nBins, len(unitNames)))
        for stIdx, st in enumerate(stLikeList):
            if arguments['lazy']:
                thisSpikeTrain = ns5.loadStProxy(st)
            else:
                thisSpikeTrain = st
            allWaveforms[:, stIdx] = thisSpikeTrain.waveforms.magnitude.flatten()
        asigs = AnalogSignal(
            allWaveforms,
            units=dummySt.waveforms.units, t_start=0*pq.s,
            sampling_rate=dummySt.sampling_rate,
            name='seg{}_{}'.format(segIdx, arguments['arrayName']),
            )
    outputBlock = Block(name='csd')
    for segIdx, seg in enumerate(dataBlock.segments):
        dummySt = dummyStList[segIdx]
        newSeg = Segment(name='seg{}'.format(segIdx))
        newSeg.block = outputBlock
        outputBlock.segments.append(newSeg)
        #
        evLabelsDF = pd.DataFrame(dummySt.array_annotations)
        evLabelsDF.loc[:, 't'] = dummySt.times
        evLabels = ['{}'.format(row) for rowIdx, row in evLabelsDF.iterrows()]
        evTimes = dummySt.t_start + np.arange(nTrials) * nBins * dummySt.sampling_rate ** (-1)
        concatEvents = Event(
            name='seg{}_trialInfo'.format(segIdx),
            times=evTimes,
            labels=evLabels
            )
        concatEvents.segment = newSeg
        newSeg.events.append(concatEvents)
        for cidx, cName in enumerate(chanIndex.channel_names):
            print('Saving channel {}'.format(cName))
            if segIdx == 0:
                newChIdx = ChannelIndex(
                    name='{}'.format(cName),
                    index=np.asarray([0]).flatten(),
                    channel_ids=np.asarray([cidx]).flatten(),
                    channel_names=np.asarray([cName]).flatten(),
                    # coordinates=coordinates[cidx, :][np.newaxis, :],  # coordinates must be 2d
                    # coordinateIndices=coordinateIndices[cidx, :],  # any other annotation *cannot* be 2d...
                    )
                newChIdx.block = outputBlock
                outputBlock.channel_indexes.append(newChIdx)
            else:
                newChIdx = outputBlock.channel_indexes[cidx]
            thisAsig = AnalogSignal(
                asigs[:, cidx],
                name='seg{}_{}'.format(segIdx, cName),
                units=dummySt.waveforms.units, sampling_rate=dummySt.sampling_rate,
                t_start=dummySt.t_start,
                )
            '''
            thisAsig.annotations['xCoords'] = float(newChIdx.coordinates[:, 0])
            thisAsig.annotations['yCoords'] = float(newChIdx.coordinates[:, 1])
            thisAsig.annotations['coordUnits'] = '{}'.format(
                newChIdx.coordinates[:, 0].units)
            '''
            #
            newChIdx.analogsignals.append(thisAsig)
            newSeg.analogsignals.append(thisAsig)
            thisAsig.channel_index = newChIdx
    #
    if arguments['lazy']:
        dataReader.file.close()
    outputBlock.create_relationship()
    outputBlock = ns5.purgeNixAnn(outputBlock)
    if os.path.exists(outputPath):
        os.remove(outputPath)
    writer = NixIO(
        filename=outputPath, mode='ow')
    writer.write_block(outputBlock, use_obj_names=True)
    writer.close()
    print('Done writing viewable matrix')
