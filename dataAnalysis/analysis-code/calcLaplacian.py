"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                    which experimental day to analyze
    --blockIdx=blockIdx                          which trial to analyze [default: 1]
    --processAll                                 process entire experimental day? [default: False]
    --analysisName=analysisName                  append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName            append a name to the resulting blocks? [default: motion]
    --arrayName=arrayName                        name of electrode array? (for map file) [default: utah]
    --inputBlockSuffix=inputBlockSuffix          which block to pull
    --inputBlockPrefix=inputBlockPrefix          which block to pull [default: Block]
    --lazy                                       load from raw, or regular? [default: False]
    --chanQuery=chanQuery                        how to restrict channels? [default: raster]
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
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
import pdb, traceback
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
# import pingouin as pg
import pandas as pd
import numpy as np
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from copy import deepcopy
from tqdm import tqdm
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
dataPath = os.path.join(
    analysisSubFolder,
    blockBaseName + '{}.nix'.format(inputBlockSuffix))

arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
    namedQueries, scratchFolder, **arguments)


def plotLFP2D(asig, chanIndex=None, fig=None, ax=None):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    if chanIndex is None:
        assert isinstance(asig, AnalogSignal)
        chanIndex = asig.channel_index
    coordinateIndices = chanIndex.annotations['coordinateIndices']
    coordinates = chanIndex.coordinates
    # coordsToIndices is guaranteed to return indices between 0 and max
    xIdxMax = coordinateIndices[:, 0].max()
    yIdxMax = coordinateIndices[:, 1].max()
    pdb.set_trace()
    # lfp = np.zeros((, ))
    return


if __name__ == "__main__":
    print('loading {}'.format(dataPath))
    #
    dataReader, dataBlock = ns5.blockFromPath(
        dataPath, lazy=arguments['lazy'])
    if arguments['chanNames'] is None:
        arguments['chanNames'] = ns5.listChanNames(
            dataBlock, arguments['chanQuery'], objType=ChannelIndex, condition='hasAsigs')
    chanNames = arguments['chanNames']
    channelIndexes = [
        chIdx
        for chIdx in dataBlock.filter(objects=ChannelIndex)
        if chIdx.name in chanNames]
    #
    for segIdx, seg in enumerate(dataBlock.segments):
        if arguments['lazy']:
            asigNameList = [
                chIdx.analogsignals[segIdx].name
                for chIdx in channelIndexes]
            asigList = ns5.loadAsigList(
                seg,
                listOfAsigProxyNames=asigNameList,
                replaceInParents=True)
        else:
            asigList = [
                chIdx.analogsignals[segIdx]
                for chIdx in channelIndexes]
    dummyAsig = asigList[0]
    if segIdx == 0:
        if not (('xcoords' in dummyAsig.annotations) and ('ycoords' in dummyAsig.annotations)):
            electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
            mapExt = electrodeMapPath.split('.')[-1]
            if mapExt == 'cmp':
                mapDF = prb_meta.cmpToDF(electrodeMapPath)
            elif mapExt == 'map':
                mapDF = prb_meta.mapToDF(electrodeMapPath)
            #
            mapCoords = (
                mapDF
                .loc[:, ['label', 'xcoords', 'ycoords', 'zcoords']]
                .set_index('label'))
            coordinatesDF = mapCoords.loc[chanNames, :]
            coordinates = coordinatesDF.to_numpy() * 400 * pq.um
            xcoords, ycoords = ssplt.coordsToIndices(
                coordinatesDF['xcoords'],
                coordinatesDF['ycoords'])
            coordinateIndices = np.concatenate(
                [xcoords[:, np.newaxis], ycoords[:, np.newaxis]],
                axis=1)
            xMin = coordinates[:, 0].min()
            xMax = coordinates[:, 0].max()
            yMin = coordinates[:, 1].min()
            yMax = coordinates[:, 1].max()
        #
        chanIndex = ChannelIndex(
            name=arguments['arrayName'],
            index=np.arange(len(chanNames)),
            channel_ids=np.arange(len(chanNames)),
            channel_names=chanNames,
            coordinates=coordinates,
            coordinateIndices=coordinateIndices,
            xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax
            )
    asigs = AnalogSignal(
        np.concatenate(
            [asig.magnitude for asig in asigList],
            axis=1),
        units=dummyAsig.units,
        t_start=dummyAsig.t_start,
        t_stop=dummyAsig.t_stop,
        sampling_rate=dummyAsig.sampling_rate,
        name='seg{}_{}'.format(segIdx, arguments['arrayName']),
        )
    asigs.channel_index = chanIndex
    chanIndex.analogsignals.append(asigs)
    plotLFP2D(asigs[10, :], chanIndex=chanIndex)
    pdb.set_trace()
