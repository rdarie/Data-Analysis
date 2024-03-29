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
    --outputBlockSuffix=outputBlockSuffix        what to name the output [default: csd]
    --useKCSD                                    which algorithm to use [default: False]
    --lazy                                       load from raw, or regular? [default: False]
    --chanQuery=chanQuery                        how to restrict channels? [default: raster]
    --verbose                                    print diagnostics? [default: False]
    --plotting                                   plot results?
    --eventSubfolder=eventSubfolder              name of folder where the event block is
    --eventBlockSuffix=eventBlockSuffix          name of events object to align to
    --recalcKCSDCV                               recalculate optimal kCSD hyperparameters [default: False]
"""

import matplotlib, os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('PS')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
#
import matplotlib.pyplot as plt
import seaborn as sns
from docopt import docopt
from neo.io import NixIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
import pdb, traceback 
import os, gc
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
from copy import copy, deepcopy
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

if arguments['outputBlockSuffix'] is not None:
    outputBlockSuffix = '_{}'.format(arguments['outputBlockSuffix'])
else:
    outputBlockSuffix = ''
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
dataPath = os.path.join(
    analysisSubFolder,
    blockBaseName + '{}.nix'.format(inputBlockSuffix))
outputPath = os.path.join(
    analysisSubFolder,
    blockBaseName + '{}.nix'.format(outputBlockSuffix))
annotationsPath = os.path.join(
    analysisSubFolder, '{}_kcsd_meta.h5'.format(blockBaseName))

arguments['chanNames'], arguments['chanQuery'] = ash.processChannelQueryArgs(
    namedQueries, scratchFolder, **arguments)

if __name__ == "__main__":
    if arguments['eventBlockSuffix'] is not None:
        eventPath = os.path.join(
            scratchFolder, ns5FileName + '_{}.nix'.format(arguments['eventBlockSuffix'])
            )
        if (blockExperimentType == 'proprio-miniRC') or (blockExperimentType == 'proprio-RC'):
            eventName = 'stimAlignTimes'
        elif blockExperimentType == 'proprio-motionOnly':
            eventName = 'motionAlignTimes'
        elif blockExperimentType == 'proprio':
            eventName = 'stimPerimotionAlignTimes'
        print('loading {}'.format(eventPath))
        eventReader, eventBlock = ns5.blockFromPath(
            eventPath, lazy=arguments['lazy'])
    else:
        eventBlock = None
    # pdb.set_trace()
    print('loading {}'.format(dataPath))
    dataReader, dataBlock = ns5.blockFromPath(
        dataPath, lazy=arguments['lazy'])
    if arguments['chanNames'] is None:
        arguments['chanNames'] = ns5.listChanNames(
            dataBlock, arguments['chanQuery'], objType=ChannelIndex, condition='hasAsigs')
    reqChanNames = arguments['chanNames']
    if 'csdOpts' not in locals():
        csdOpts = {}
    skipChannels = csdOpts.pop('skipChannels', None)
    csdTimeFilterOpts = csdOpts.pop('filterOpts', None)
    csdOptimalHyperparameters = csdOpts.pop('optimalHyperparameters', None)
    NSamplesForCV = csdOpts.pop('NSamplesForCV', 1000)
    chunkSize = csdOpts.pop('chunkSize', 20000)
    if skipChannels is not None:
        reqChanNames = [cn for cn in reqChanNames if cn not in skipChannels]
    #
    channelIndexes = [
        chIdx
        for chIdx in dataBlock.filter(objects=ChannelIndex)
        if chIdx.name in reqChanNames]
    chanNames = [ci.name for ci in channelIndexes]
    #
    for segIdx, seg in enumerate(dataBlock.segments):
        if arguments['lazy']:
            asigNameList = [
                chIdx.analogsignals[segIdx].name
                for chIdx in channelIndexes]
            asigList = ns5.loadAsigList(
                seg,
                listOfAsigProxyNames=asigNameList,
                replaceInParents=False)
        else:
            asigList = [
                chIdx.analogsignals[segIdx]
                for chIdx in channelIndexes]
        dummyAsig = asigList[0].copy()
        if 'chanIndex' not in locals():
            if not (('xCoords' in dummyAsig.annotations) and ('yCoords' in dummyAsig.annotations)):
                electrodeMapPath = spikeSortingOpts[arguments['arrayName']]['electrodeMapPath']
                mapExt = electrodeMapPath.split('.')[-1]
                if mapExt == 'cmp':
                    mapDF = prb_meta.cmpToDF(electrodeMapPath)
                elif mapExt == 'map':
                    mapDF = prb_meta.mapToDF(electrodeMapPath)
                #
                mapCoords = (
                    mapDF
                    .loc[:, ['label', 'xcoords', 'ycoords']]
                    .set_index('label'))
                coordinatesDF = mapCoords.loc[chanNames, :]
                coordinates = coordinatesDF.to_numpy() * 400 * pq.um
            else:
                xcoords, ycoords = [], []
                for asigIdx, asig in enumerate(asigList):
                    xcoords.append(copy(asig.annotations['xCoords']) * 400)
                    ycoords.append(copy(asig.annotations['yCoords']) * 400)
                    assert asig.annotations['parentChanName'] == chanNames[asigIdx]
                coordinates = np.concatenate(
                    [np.asarray(xcoords)[:, np.newaxis], np.asarray(ycoords)[:, np.newaxis]],
                    axis=1) * pq.um
            #
            xIdx, yIdx = ssplt.coordsToIndices(
                coordinates[:, 0], coordinates[:, 1])
            coordinateIndices = np.concatenate(
                [xIdx[:, np.newaxis], yIdx[:, np.newaxis]],
                axis=1)
            chanIndex = ChannelIndex(
                name=arguments['arrayName'],
                index=np.arange(len(chanNames)),
                channel_ids=np.arange(len(chanNames)),
                channel_names=chanNames,
                coordinates=coordinates,
                coordinateIndices=coordinateIndices
                )
            annotationsLong = pd.DataFrame(
                chanIndex.coordinates, columns=['x', 'y'])
            annotationsLong.loc[:, ['xIdx', 'yIdx']] = coordinateIndices
            annotationsLong.loc[:, 'chanName'] = chanNames
            annotations2D = {
                key: annotationsLong.pivot(index='y', columns='x', values=key)
                for key in ['chanName']}
        #
        prf.print_memory_usage(prefix='Concatenating input asigs')
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
        del asigList
        gc.collect()
        prf.print_memory_usage(prefix='    done concatenating input asigs')
        # asigs.channel_index = chanIndex
        # chanIndex.analogsignals.append(asigs)
        ##
        decimateFactor = None
        sigma = 1
        #
        # TODO: bring in events block, use perievent times to do cross validation
        estimateAsigs = asigs
        # estimateAsigs = asigs[:100, :]
        # NSamplesForCV = 10
        # chunkSize = 10
        #
        #
        if arguments['plotting'] and (segIdx == 0):
            sns.set(font_scale=.8)
            if arguments['useKCSD']:
                fig, ax = plt.subplots(1, 2)
                origAx = ax[0]
                csdAx = ax[1]
                methodName = 'kcsd'
            else:
                fig, ax = plt.subplots(1, 3)
                origAx = ax[0]
                smoothedAx = ax[1]
                csdAx = ax[2]
                methodName = 'laplacian'
            #
            fig.set_size_inches(5 * len(ax), 5)
            _, _, lfpDF = csd.plotLfp2D(
                asig=estimateAsigs[0, :], chanIndex=chanIndex,
                fillerFun=csd.interpLfp, fig=fig, ax=origAx)
            origAx.set_title('Original')
        if arguments['useKCSD']:
            print('estimating csd with kcsd...')
            nElecX = chanIndex.annotations['coordinateIndices'][:, 0].max()
            ordMagX = np.ceil(np.log10(nElecX))
            nElecY = chanIndex.annotations['coordinateIndices'][:, 1].max()
            ordMagY = np.ceil(np.log10(nElecY))
            #
            kcsdKWArgs = {
                'cv_iterator': True,
                'verbose': True,
                'Rs': np.asarray([0.2, 0.25, 0.3]),
                # 'lambdas': np.logspace(-2, -10, 10, base=10.),
                'n_lambda_suggestions': 10,
                'gdx': 0.4, 'ext_x': 0.2, 'gdy': 0.4, 'ext_y': 0.2,
                'n_src_init': (3 * 10 ** ordMagX) * (3 * 10 ** ordMagY)
                }
            print('calcKCSD using {} sources'.format(kcsdKWArgs['n_src_init']))
            needKCSDCV = (not os.path.exists(annotationsPath)) and (csdOptimalHyperparameters is None)
            if arguments['recalcKCSDCV'] or needKCSDCV:
                # testParams = pd.MultiIndex.from_product(
                #     [[0.2, 0.5, 1], ], names=['h']
                #     ).to_frame()
                # testParams = pd.DataFrame([0.2, 0.5, 1], columns=['h'])
                testParams = pd.DataFrame([0.2, 1], columns=['h'])
                allErrsList = []
                sampleIdx = sample(
                    list(range(estimateAsigs.shape[0])),
                    min(estimateAsigs.shape[0], NSamplesForCV))
                for tpIdx, tp in testParams.iterrows():
                    # run cross validator on a subset of the data
                    sampleLFP = estimateAsigs.magnitude[sampleIdx, :] * estimateAsigs.units
                    theseArgs = kcsdKWArgs.copy()
                    theseArgs.update(tp.to_dict())
                    kcsd, csdAsigs, cv_R, cv_lambda = csd.runKcsd(
                        sampleLFP,
                        chanIndex.coordinates[:, :2],
                        kwargs=theseArgs.copy(),
                        process_estimate=True)
                    if 'lambdas' not in theseArgs:
                        lambdas = np.concatenate([kcsd.suggest_lambda(), [0]])
                    else:
                        lambdas = theseArgs['lambdas']
                    if 'Rs' not in theseArgs:
                        Rs = [kcsd.R_init]
                    else:
                        Rs = theseArgs['Rs']
                    theseErrs = {}
                    for ridx, R in enumerate(Rs):
                        theseErrs[R] = pd.DataFrame(
                            kcsd.errs_per_ele[ridx],
                            index=lambdas, columns=chanNames)
                    errThisTP = pd.concat(theseErrs, names=['R_init', 'lambd'])
                    errThisTP.columns.name = 'electrode'
                    errThisTP = errThisTP.stack().reset_index()
                    errThisTP.rename(columns={0: 'error'}, inplace=True)
                    for prmNm in tp.index:
                        errThisTP.loc[:, prmNm] = tp[prmNm]
                    allErrsList.append(errThisTP)
                kcsdErrsDF = pd.concat(allErrsList)
                #
                kcsdErrsDF.to_hdf(annotationsPath, 'kcsd_error')
                testParams.to_hdf(annotationsPath, 'testParams')
                paramNames = testParams.columns.to_list() + ['R_init', 'lambd']
                meanErrs = kcsdErrsDF.groupby(paramNames).mean()['error']
                #
                optiParams = {
                    paramNames[pidx]: p
                    for pidx, p in enumerate(meanErrs.idxmin())}
                if arguments['plotting']:
                    #
                    plotGroups = ['R_init', 'h']
                    # meanGroups = ['lambd']
                    nPlotHeadings = kcsdErrsDF.groupby(plotGroups).ngroups
                    nRows = int(np.ceil(np.sqrt(nPlotHeadings)))
                    nCols = int(np.ceil(nPlotHeadings / nRows))
                    fig2, ax2 = plt.subplots(nRows, nCols)
                    fig2.set_size_inches(8 * nRows, 8 * nCols)
                    flatAxes = ax2.flatten()
                    globalMeanErr = kcsdErrsDF.groupby('electrode').mean()['error']
                    for axIdx, (params, errGroup) in enumerate(kcsdErrsDF.groupby(plotGroups)):
                        thisMeanErr = errGroup.groupby('electrode').mean()['error']
                        errAx = flatAxes[axIdx]
                        csdErrPerElec = AnalogSignal(
                            thisMeanErr.to_numpy()[np.newaxis, :],
                            units=pq.uV, t_start=0 * pq.s,
                            sampling_rate=dummyAsig.sampling_rate
                            )
                        _, _, errDF = csd.plotLfp2D(
                            asig=csdErrPerElec[0, :], chanIndex=chanIndex,
                            fig=fig2, ax=errAx,
                            heatmapKWs={
                                'cmap': 'viridis',
                                'annot': annotations2D['chanName'],
                                'vmin': globalMeanErr.min(),
                                'vmax': globalMeanErr.max(),
                                'fmt': '^5'})
                        errAx.set_title('mean err = {:3f}; {} = {}'.format(
                            thisMeanErr.mean(), plotGroups, params))
                    fig2.suptitle('kCSD CV Error (optimal params = {})'.format(optiParams))
                    fig2.savefig(
                        os.path.join(
                            figureOutputFolder,
                            '{}_{}_error_by_electrode.pdf'.format(
                                blockBaseName, methodName)),
                        bbox_inches='tight', pad_inches=0
                        )
            elif csdOptimalHyperparameters is not None:
                optiParams = csdOptimalHyperparameters
            else:
                kcsdErrsDF = pd.read_hdf(annotationsPath, 'kcsd_error')
                testParams = pd.read_hdf(annotationsPath, 'testParams')
                paramNames = testParams.columns.to_list() + ['R_init', 'lambd']
                meanErrs = kcsdErrsDF.groupby(paramNames).mean()['error']
                #
                optiParams = {
                    paramNames[pidx]: p
                    for pidx, p in enumerate(meanErrs.idxmin())}
            print('After CV, updating params to: {}'.format(optiParams))
            kcsdKWArgs.update(optiParams)
            #
            nChunks = max(len(estimateAsigs) // chunkSize, 1)
            adjChunkSize = int(np.ceil(len(estimateAsigs) / nChunks))
            csdLongList = []
            for chunkIdx in range(nChunks):
                seeker = slice(adjChunkSize * chunkIdx, adjChunkSize * (chunkIdx + 1))
                kcsd, csdAsigs = csd.runKcsd(
                    estimateAsigs[seeker],
                    chanIndex.coordinates[:, :2],
                    kwargs=kcsdKWArgs.copy(),
                    process_estimate=False)
                '''
                csdAsigs = elph_csd.estimate_csd(
                    estimateAsigs, chanIndex.coordinates[:, :2],
                    method='KCSD2D'
                )
                '''
                print('Smoothing and downsampling')
                for tIdx in tqdm(range(csdAsigs.shape[0])):
                    csdDFFull = pd.DataFrame(
                        csdAsigs[tIdx, :],
                        index=csdAsigs.annotations['y_coords'][0, :],
                        columns=csdAsigs.annotations['x_coords'][:, 0],
                        )
                    if decimateFactor is not None:
                        win = decimateFactor
                        halfWin = int(np.ceil(win/2))
                        csdDF = (
                            csdDFFull
                            .rolling(window=win, center=True, axis=0).mean()
                            .iloc[halfWin:-halfWin:win, :]
                            .rolling(window=win, center=True, axis=1).mean()
                            .iloc[:, halfWin:-halfWin:win])
                    else:
                        csdDF = csdDFFull
                    csdDF.index.name = 'y'
                    csdDF.columns.name = 'x'
                    csdLong = csdDF.unstack()
                    csdLongList.append(csdLong.to_numpy()[np.newaxis, :])
                    if 'csdChanIndex' not in locals():
                        csdCoordinates = (
                            csdLong.index.to_frame().to_numpy() *
                            1000 * pq.um)
                        csdX = csdCoordinates[:, 0]
                        csdY = csdCoordinates[:, 1]
                        csdXIdx, csdYIdx = ssplt.coordsToIndices(
                            np.floor(csdX), np.floor(csdY))
                        csdCoordinateIndices = np.concatenate(
                            [csdXIdx[:, np.newaxis], csdYIdx[:, np.newaxis]],
                            axis=1)
                        csdChanNames = [
                            arguments['arrayName'] + '_csd_{}'.format(idx)
                            for idx, xy in enumerate(csdCoordinateIndices)
                            ]
                        csdChanIndex = ChannelIndex(
                            name=arguments['arrayName'] + '_csd',
                            index=np.arange(len(csdChanNames)),
                            channel_ids=np.arange(len(csdChanNames)),
                            channel_names=csdChanNames,
                            coordinates=csdCoordinates,
                            coordinateIndices=csdCoordinateIndices
                            )
                        csdUnits = csdAsigs.units
            prf.print_memory_usage(prefix='   Done running csd.estimate_csd')
            del asigs, estimateAsigs
            gc.collect()
            prf.print_memory_usage(prefix='   Deleted inputAsigs')
            csdAsigsLong = AnalogSignal(
                np.concatenate(csdLongList, axis=0),
                units=csdAsigs.units,
                sampling_rate=dummyAsig.sampling_rate,
                t_start=dummyAsig.t_start
                )
            csdAsigsLong.channel_index = csdChanIndex
            csdChanIndex.analogsignals.append(csdAsigsLong)
        else:
            # 2d laplacian
            print('reshaping data for laplacian')
            lfpTimes, lfpList = csd.compose2D(
                estimateAsigs, chanIndex,
                fillerFun=csd.interpLfp,
                fillerFunKWArgs=dict(
                    coordCols=['x', 'y'], groupCols=['t'],
                    method='bypass', tqdmProgBar=True),
                tqdmProgBar=True)
            prf.print_memory_usage(prefix='   Done running csd.estimate_csd')
            del asigs, estimateAsigs
            gc.collect()
            prf.print_memory_usage(prefix='   Deleted inputAsigs')
            laplList = []
            print('done reshaping, estimating csd with laplacian')
            for tIdx, lfpDF in enumerate(tqdm(lfpList)):
                laplDF = pd.DataFrame(
                    ndimage.laplace(ndimage.gaussian_filter(lfpDF, sigma)),
                    index=lfpDF.index,
                    columns=lfpDF.columns).unstack()
                laplList.append(laplDF.to_numpy()[np.newaxis, :])
                if 'csdChanIndex' not in locals():
                    csdCoordinates = laplDF.index.to_frame().to_numpy() * 1000 * pq.um
                    csdX = csdCoordinates[:, 0]
                    csdY = csdCoordinates[:, 1]
                    csdXIdx, csdYIdx = ssplt.coordsToIndices(
                        csdX, csdY)
                    csdCoordinateIndices = np.concatenate(
                        [csdXIdx[:, np.newaxis], csdYIdx[:, np.newaxis]],
                        axis=1)
                    csdChanNames = [
                        arguments['arrayName'] + '_csd_{}'.format(idx)
                        for idx, xy in enumerate(csdCoordinateIndices)
                        ]
                    csdChanIndex = ChannelIndex(
                        name=arguments['arrayName'] + '_csd',
                        index=np.arange(len(csdChanNames)),
                        channel_ids=np.arange(len(csdChanNames)),
                        channel_names=csdChanNames,
                        coordinates=csdCoordinates,
                        coordinateIndices=csdCoordinateIndices
                        )
            print('    done estimating csd with laplacian')
            csdUnits = dummyAsig.units / (csdX.units * csdY.units)
            csdAsigsLong = AnalogSignal(
                np.concatenate(laplList, axis=0),
                units=csdUnits,
                sampling_rate=dummyAsig.sampling_rate,
                t_start=dummyAsig.t_start
                )
            csdAsigsLong.channel_index = csdChanIndex
            csdChanIndex.analogsignals.append(csdAsigsLong)
            # end laplacian option
    if csdTimeFilterOpts is not None:
        if 'low' in csdTimeFilterOpts:
            if 'Wn' not in csdTimeFilterOpts['low']:
                csdTimeFilterOpts['low']['Wn'] = float(dummyAsig.sampling_rate) / 2 - 1
        filterCoeffs = hf.makeFilterCoeffsSOS(
            csdTimeFilterOpts.copy(), float(dummyAsig.sampling_rate))
        print('time domain filtering csd estimate...')
        filteredAsigs = signal.sosfiltfilt(
            filterCoeffs, csdAsigsLong.magnitude,
            axis=0)
        csdAsigsLong.magnitude[:] = filteredAsigs
    if arguments['plotting']:
        _, _, csdDF = csd.plotLfp2D(
            asig=csdAsigsLong[0, :], chanIndex=csdChanIndex,
            fillerFun=csd.interpLfp, fig=fig, ax=csdAx,
            heatmapKWs={'cmap': 'mako'})
        csdAx.set_title('CSD estimate ({})'.format(methodName))
        #
        if not arguments['useKCSD']:
            smoothedDF = pd.DataFrame(
                ndimage.gaussian_filter(lfpDF, sigma), index=lfpDF.index,
                columns=lfpDF.columns)
            csd.plotLfp2D(
                lfpDF=smoothedDF, fig=fig, ax=smoothedAx)
            smoothedAx.set_title('Smoothed (sigma = {})'.format(sigma))
        fig.savefig(
            os.path.join(
                figureOutputFolder,
                '{}_{}_example.pdf'.format(blockBaseName, methodName)),
            bbox_inches='tight', pad_inches=0
        )
        # plt.show()
        plt.close()
    outputBlock = Block(name='csd')
    for segIdx, seg in enumerate(dataBlock.segments):
        newSeg = Segment(name='seg{}_csd'.format(segIdx))
        newSeg.block = outputBlock
        outputBlock.segments.append(newSeg)
        for cidx, csdName in enumerate(csdChanIndex.channel_names):
            if segIdx == 0:
                newChIdx = ChannelIndex(
                    name='{}'.format(csdName),
                    index=np.asarray([0]).flatten(),
                    channel_ids=np.asarray([cidx]).flatten(),
                    channel_names=np.asarray([csdName]).flatten(),
                    coordinates=csdCoordinates[cidx, :][np.newaxis, :],  # coordinates must be 2d
                    coordinateIndices=csdCoordinateIndices[cidx, :],  #  any other annotation *cannot* be 2d...
                    )
                newChIdx.block = outputBlock
                outputBlock.channel_indexes.append(newChIdx)
            else:
                newChIdx = outputBlock.channel_indexes[cidx]
            thisAsig = AnalogSignal(
                csdAsigsLong[:, cidx],
                name='seg{}_{}'.format(segIdx, csdName),
                units=csdUnits, sampling_rate=dummyAsig.sampling_rate,
                t_start=dummyAsig.t_start,
                )
            thisAsig.annotations['xCoords'] = float(newChIdx.coordinates[:, 0])
            thisAsig.annotations['yCoords'] = float(newChIdx.coordinates[:, 1])
            thisAsig.annotations['coordUnits'] = '{}'.format(
                newChIdx.coordinates[:, 0].units)
            #
            print('asig {}, shape {}'.format(thisAsig.name, thisAsig.shape))
            newChIdx.analogsignals.append(thisAsig)
            newSeg.analogsignals.append(thisAsig)
            thisAsig.channel_index = newChIdx
    #
    outputBlock.create_relationship()
    writer = NixIO(
        filename=outputPath, mode='w')
    writer.write_block(outputBlock, use_obj_names=True)
    writer.close()
    print('Done writing CSD matrix')
