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
from scipy import interpolate, ndimage
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


def compose2D_single(
        quant, chanIndex, fullLongIndex,
        procFun=None, fillerFun=None):
    longData = pd.DataFrame(
        {
            'x': chanIndex.coordinates[:, 0],
            'y': chanIndex.coordinates[:, 1],
            'signal': quant}
        ).set_index(['x', 'y'])
    longData.dropna(inplace=True)
    missingIndices = fullLongIndex[~fullLongIndex.isin(longData.index)]
    if fillerFun is not None:
        fillerDF = fillerFun(longData, missingIndices)
    else:
        fillerDF = pd.DataFrame(np.nan, index=missingIndices, columns=['signal'])
    lfpLong = pd.concat([longData, fillerDF]).reset_index().sort_values(by=['x', 'y'])
    lfpDF = lfpLong.pivot(index='y', columns='x', values='signal')
    if procFun is not None:
        lfpDF = procFun(lfpDF)
    return lfpDF


def compose2D(asig, chanIndex, procFun=None, fillerFun=None):
    coordinateIndices = chanIndex.annotations['coordinateIndices']
    # coordsToIndices is guaranteed to return indices between 0 and max
    xMin = chanIndex.coordinates[:, 0].min()
    xMax = chanIndex.coordinates[:, 0].max()
    yMin = chanIndex.coordinates[:, 1].min()
    yMax = chanIndex.coordinates[:, 1].max()
    #
    yIdxMax = coordinateIndices[:, 1].max()
    yStepSize = (yMax - yMin) / yIdxMax
    yIndex = (np.arange(yIdxMax + 1) * yStepSize)
    xIdxMax = coordinateIndices[:, 0].max()
    xStepSize = (xMax - xMin) / xIdxMax
    xIndex = (np.arange(xIdxMax + 1) * xStepSize)
    fullLongIndex = pd.MultiIndex.from_product([xIndex, yIndex], names=['x', 'y'])
    # pdb.set_trace()
    if asig.ndim == 1:
        # asig is a 1D Quantity
        lfpDF = compose2D_single(
            asig, chanIndex, fullLongIndex,
            procFun=procFun, fillerFun=fillerFun)
        return lfpDF
    else:
        # asig is a 2D AnalogSignal
        lfpList = []
        for tIdx, t in asig.times:
            lfpDF = compose2D_single(
                asig, chanIndex, fullLongIndex,
                procFun=procFun, fillerFun=fillerFun)
            lfpList.append(lfpDF)
        return asig.times, lfpList


def plotLfp2D(
        asig=None, chanIndex=None,
        lfpDF=None, procFun=None, fillerFun=None,
        fig=None, ax=None,
        heatmapKWs={}):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    returnList = [fig, ax]
    if lfpDF is None:
        lfpDF = compose2D(
            asig, chanIndex,
            procFun=procFun, fillerFun=fillerFun)
        returnList.append(lfpDF)
    sns.heatmap(lfpDF, ax=ax, **heatmapKWs)
    return returnList


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
        #
        chanIndex = ChannelIndex(
            name=arguments['arrayName'],
            index=np.arange(len(chanNames)),
            channel_ids=np.arange(len(chanNames)),
            channel_names=chanNames,
            coordinates=coordinates,
            coordinateIndices=coordinateIndices
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

    def interpLfp(longData, missingIndices):
        fillerVals = interpolate.griddata(
            longData.index.to_frame(), longData['signal'],
            missingIndices.to_frame())
        fillerDF = pd.DataFrame(
            fillerVals,
            index=missingIndices, columns=['signal'])
        if fillerDF['signal'].isna().any():
            stillMissing = fillerDF.index[fillerDF['signal'].isna()]
            fillerForFiller = interpolate.griddata(
                longData.index.to_frame(), longData['signal'],
                stillMissing.to_frame(), method='nearest')
            fillerDF.loc[stillMissing, 'signal'] = fillerForFiller
        # pdb.set_trace()
        return fillerDF

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)
    _, _, lfpDF = plotLfp2D(
        asig=asigs[10, :], chanIndex=chanIndex,
        fillerFun=interpLfp, fig=fig, ax=ax[0])
    ax[0].set_title('Original')
    sigma = 1
    smoothedDF = pd.DataFrame(
        ndimage.gaussian_filter(lfpDF, sigma), index=lfpDF.index,
        columns=lfpDF.columns)
    plotLfp2D(
        lfpDF=smoothedDF, fig=fig, ax=ax[1])
    ax[1].set_title('Smoothed (sigma = {})'.format(sigma))
    laplDF = pd.DataFrame(
        ndimage.laplace(smoothedDF), index=lfpDF.index,
        columns=lfpDF.columns)
    plotLfp2D(
        lfpDF=laplDF, fig=fig, ax=ax[2], heatmapKWs={'cmap': 'mako'})
    ax[2].set_title('Laplacian of smoothed')
    plt.show()
    pdb.set_trace()
