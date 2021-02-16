import numpy as np
import pandas as pd
import quantities as pq
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from kcsd import KCSD2D
from scipy import interpolate, ndimage


def compose2D(
        asig, chanIndex, procFun=None,
        fillerFun=None, fillerFunKWArgs={}, unstacked=False,
        tqdmProgBar=False):
    coordinateIndices = chanIndex.annotations['coordinateIndices']
    # coordsToIndices is guaranteed to return indices between 0 and max
    xMin = chanIndex.coordinates[:, 0].min()
    xMax = chanIndex.coordinates[:, 0].max()
    yMin = chanIndex.coordinates[:, 1].min()
    yMax = chanIndex.coordinates[:, 1].max()
    #
    yIdxMax = coordinateIndices[:, 1].max()
    yStepSize = (yMax - yMin) / yIdxMax
    yIndex = yMin + (np.arange(yIdxMax + 1) * yStepSize)
    xIdxMax = coordinateIndices[:, 0].max()
    xStepSize = (xMax - xMin) / xIdxMax
    xIndex = xMin + (np.arange(xIdxMax + 1) * xStepSize)
    fullLongIndex = pd.MultiIndex.from_product(
        [xIndex, yIndex], names=['x', 'y'])
    fullLongCoords = fullLongIndex.to_frame().reset_index(drop=True)
    fullLongCoords.loc[:, 'isPresent'] = False
    presentCoords = pd.DataFrame(
        chanIndex.coordinates[:, :2], columns=['x', 'y'])
    for coordIdx, coord in fullLongCoords.iterrows():
        deltaX = (presentCoords['x'] - coord['x']).abs()
        deltaY = (presentCoords['y'] - coord['y']).abs()
        fullLongCoords.loc[coordIdx, 'isPresent'] = (
            (deltaX < 1e-3) &
            (deltaY < 1e-3)
            ).any()
    if asig.ndim == 1:
        allAsig = asig.magnitude[np.newaxis, :].copy()
        asigTimes = np.asarray([0]) * pq.s
    else:
        allAsig = asig.magnitude.copy()
        asigTimes = asig.times
    if (~fullLongCoords['isPresent']).any():
        missingCoords = fullLongCoords[~fullLongCoords['isPresent']]
        allCoords = pd.concat(
            [presentCoords, missingCoords.loc[:, ['x', 'y']]],
            ignore_index=True, axis=0)
        filler = np.empty((allAsig.shape[0], missingCoords.shape[0]))
        filler[:] = np.nan
        allAsig = np.concatenate([allAsig, filler], axis=1)
    else:
        allCoords = presentCoords
    allAsigDF = pd.DataFrame(
        allAsig, index=asigTimes,
        columns=allCoords.set_index(['x', 'y']).index)
    allAsigDF.index.name = 't'
    if fillerFun is not None:
        fillerFun(allAsigDF, **fillerFunKWArgs)
    # asig is a 2D AnalogSignal
    lfpList = []
    if tqdmProgBar:
        tIterator = tqdm(total=allAsig.shape[0], miniters=100)
    for tIdx, asigSrs in allAsigDF.iterrows():
        asigSrs.name = 'signal'
        lfpDF = asigSrs.reset_index().pivot(index='y', columns='x', values='signal')
        if procFun is not None:
            lfpDF = procFun(lfpDF)
        if unstacked:
            lfpList.append(lfpDF.unstack())
        else:
            lfpList.append(lfpDF)
        if tqdmProgBar:
            tIterator.update(1)
    return asigTimes, lfpList


def plotLfp2D(
        asig=None, chanIndex=None,
        lfpDF=None, procFun=None, fillerFun=None, fillerFunKWArgs={},
        fig=None, ax=None,
        heatmapKWs={}):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    returnList = [fig, ax]
    if lfpDF is None:
        _, lfpList = compose2D(
            asig, chanIndex,
            procFun=procFun, fillerFun=fillerFun,
            fillerFunKWArgs=fillerFunKWArgs)
        lfpDF = lfpList[0]
        returnList.append(lfpDF)
    sns.heatmap(lfpDF, ax=ax, **heatmapKWs)
    return returnList


def runKcsd(
        lfp, coordinates,
        kwargs={}, process_estimate=True):
    scaled_coords = []
    for coord in coordinates:
        try:
            scaled_coords.append(coord.rescale(pq.mm))
        except AttributeError:
            raise AttributeError('No units given for electrode spatial \
            coordinates')
    '''
    input_array = np.zeros(
        (len(lfp), lfp[0].magnitude.shape[0]))
    print('runKcsd(): rescaling inputs')
    for ii, jj in enumerate(tqdm(lfp)):
        input_array[ii, :] = jj.rescale(pq.mV).magnitude
    '''
    input_array = lfp.rescale(pq.mV).magnitude
    lambdas = kwargs.pop('lambdas', None)
    Rs = kwargs.pop('Rs', None)
    k = KCSD2D(
        np.array(scaled_coords),
        input_array.T, **kwargs)
    if process_estimate:
        cv_R, cv_lambda = k.cross_validate(lambdas, Rs)
    estm_csd = k.values()
    estm_csd = np.rollaxis(estm_csd, -1, 0)
    #
    returnValues = []
    if isinstance(lfp, AnalogSignal):
        output = AnalogSignal(
            estm_csd * pq.uA / pq.mm**3,
            t_start=lfp.t_start,
            sampling_rate=lfp.sampling_rate)
        dim = len(scaled_coords[0])
        if dim == 1:
            output.annotate(
                x_coords=np.round(k.estm_x, decimals=7))
        elif dim == 2:
            output.annotate(
                x_coords=np.round(k.estm_x, decimals=7),
                y_coords=np.round(k.estm_y, decimals=7))
        elif dim == 3:
            output.annotate(
                x_coords=np.round(k.estm_x, decimals=7),
                y_coords=np.round(k.estm_y, decimals=7),
                z_coords=np.round(k.estm_z, decimals=7))
        returnValues += [k, output]
    else:
        returnValues += [k, estm_csd * pq.uA / pq.mm**3]
    if process_estimate:
        returnValues += [cv_R, cv_lambda]
    return returnValues


def interpLfp(
        wideDF,
        coordCols=None, groupCols=None,
        method='linear', tqdmProgBar=False):
    longSrs = wideDF.unstack()
    longSrs.name = 'signal'
    if coordCols is None:
        coordCols = longSrs.index.names
    longDF = longSrs.reset_index()
    del longSrs
    longDF.loc[:, 'validMask'] = longDF.notna().all(axis='columns')
    validAxes = (
        (
            longDF.loc[longDF['validMask'], coordCols].max() -
            longDF.loc[longDF['validMask'], coordCols].min())
        != 0)
    coordCols = validAxes.loc[validAxes].index.to_list()
    tIterator = None
    if groupCols is None:
        grouper = [('all', longDF)]
    else:
        # if isinstance(groupCols, int):
        #     longDF.loc[:, 'groupIdx'] = 0
        grouper = longDF.groupby(groupCols)
        if tqdmProgBar:
            tIterator = tqdm(total=grouper.ngroups, miniters=100)
    fillerDF = longDF.loc[~longDF['validMask'], :].copy()
    print('interpolating...')
    for name, group in grouper:
        if not method == 'bypass':
            fillerVals = interpolate.griddata(
                group.loc[group['validMask'], coordCols],
                group.loc[group['validMask'], 'signal'],
                group.loc[~group['validMask'], coordCols], method=method)
        else:
            fillerVals = group.loc[~group['validMask'], 'signal'].to_numpy() * 0
        nanFillers = np.isnan(fillerVals)
        if nanFillers.any():
            # try again with nearest neighbor interpolation
            dummyGroup = group.copy()
            dummyGroup.loc[~group['validMask'], 'signal'] = fillerVals
            dummyValidMask = dummyGroup.notna().all(axis='columns')
            backupFillerVals = interpolate.griddata(
                dummyGroup.loc[dummyValidMask, coordCols],
                dummyGroup.loc[dummyValidMask, 'signal'],
                dummyGroup.loc[~dummyValidMask, coordCols], method='nearest')
            fillerVals[nanFillers] = backupFillerVals
        theseFillerIndices = group.loc[~group['validMask'], :].index
        fillerDF.loc[theseFillerIndices, 'signal'] = fillerVals
        if tIterator is not None:
            tIterator.update(1)
    fillerDF.drop(columns='validMask', inplace=True)
    fillerDF.set_index(list(fillerDF.columns.drop('signal')), inplace=True)
    fillerDF = fillerDF.unstack(wideDF.columns.names)
    fillerDF = fillerDF.droplevel(0, axis='columns')
    wideDF.loc[fillerDF.index, fillerDF.columns] = fillerDF
    return

