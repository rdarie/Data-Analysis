import os, sys, warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# sys.stderr = open(os.devnull, "w")  # silence stderr
import pingouin as pg
import dill as pickle
# sys.stderr = sys.__stderr__  # unsilence stderr

import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.helper_functions_new as hf
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
import multiprocessing
from scipy import stats
from statsmodels.stats.multitest import multipletests as mt
from copy import copy
import pdb, traceback
from tqdm import tqdm


def processAlignQueryArgs(
        namedQueries, alignQuery=None, **kwargs):
    if (alignQuery is None) or (not len(alignQuery)):
        dataQuery = None
    else:
        if alignQuery in namedQueries['align']:
            dataQuery = namedQueries['align'][alignQuery]
        else:
            dataQuery = alignQuery
    return dataQuery


def processChannelQueryArgs(
        namedQueries, scratchFolder, selector=None, chanQuery=None,
        inputBlockName='', **kwargs):
    #
    if selector is not None:
        with open(
            os.path.join(
                scratchFolder,
                selector + '.pickle'),
                'rb') as f:
            selectorMetadata = pickle.load(f)
        chanNames = selectorMetadata['outputFeatures']
        outputQuery = None
    else:
        chanNames = None
        if chanQuery in namedQueries['chan']:
            outputQuery = namedQueries['chan'][chanQuery]
        else:
            outputQuery = chanQuery
    return chanNames, outputQuery


def processOutlierTrials(
        scratchPath, prefix,
        maskOutlierBlocks=False,
        invertOutlierBlocks=False,
        window=None, alignFolderName=None,
        **kwargs
        ):
    if maskOutlierBlocks:
        resultPath = os.path.join(
            scratchPath, 'outlierTrials', alignFolderName,
            prefix + '_{}_outliers.h5'.format(window))
        oBlocks = pd.read_hdf(resultPath, 'rejectBlock')
        oBlocksCount = oBlocks.sum()
        oBlocksSize = oBlocks.shape[0]
        print('Loading outlier trials. Rejecting a proportion of {:.2f} ({} out of {})'.format(
            oBlocksCount / oBlocksSize, oBlocksCount, oBlocksSize))
        if invertOutlierBlocks:
            oBlocks = ~oBlocks.astype(bool)
        return oBlocks
    else:
        return None


def processUnitQueryArgs(
        namedQueries, scratchFolder,
        selector=None, unitQuery=None,
        inputBlockName='', **kwargs):
    if selector is not None:
        with open(
            os.path.join(
                scratchFolder,
                selector + '.pickle'),
                'rb') as f:
            selectorMetadata = pickle.load(f)
        unitNames = [
            '{}_{}#0'.format(i, inputBlockName)
            for i in selectorMetadata['outputFeatures']]
        outputQuery = None
    else:
        unitNames = None
        if unitQuery in namedQueries['unit']:
            outputQuery = namedQueries['unit'][unitQuery]
        else:
            outputQuery = unitQuery
    return unitNames, outputQuery


def applyFun(
        triggeredPath=None, resultPath=None, resultNames=None,
        fun=None, funArgs=[], funKWargs={},
        lazy=None, loadArgs={},
        secondaryPath=None, secondaryUnitQuery=None,
        loadType='all', applyType='self',
        verbose=False):
    if verbose:
        prf.print_memory_usage('about to load dataBlock {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(triggeredPath, lazy=lazy)
    if secondaryPath is not None:
        secondaryReader, secondaryBlock = ns5.blockFromPath(
            secondaryPath, lazy=lazy)
    if verbose:
        prf.print_memory_usage('done loading dataBlock')
    if loadType == 'all':
        alignedAsigsDF = ns5.alignedAsigsToDF(
            dataBlock, **loadArgs)
        if verbose:
            prf.print_memory_usage('just loaded alignedAsigs')
        if applyType == 'self':
            result = getattr(alignedAsigsDF, fun)(*funArgs, **funKWargs)
            if (not isinstance(result, list)) or (not isinstance(result, tuple)):
                result = [result]
        if applyType == 'func':
            result = fun(alignedAsigsDF, *funArgs, **funKWargs)
            if (not isinstance(result, list)) or (not isinstance(result, tuple)):
                result = [result]
    elif loadType == 'elementwise':
        if loadArgs['unitNames'] is None:
            unitNames = ns5.listChanNames(
                dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
        loadArgs.pop('unitNames')
        loadArgs.pop('unitQuery')
        for idxOuter, firstUnit in enumerate(unitNames):
            if verbose:
                prf.print_memory_usage(' firstUnit: {}'.format(firstUnit))
            firstDF = ns5.alignedAsigsToDF(
                dataBlock, [firstUnit],
                **loadArgs)
            tempRes = fun(firstDF, *funArgs, **funKWargs)
            if idxOuter == 0:
                result = [
                    pd.Series(
                        0, index=unitNames, dtype='float32')
                    for i in range(len(tempRes))]
            for i in range(len(tempRes)):
                result[i].loc[firstUnit] = tempRes[i]
    elif loadType == 'pairwise':
        if loadArgs['unitNames'] is None:
            unitNames = ns5.listChanNames(
                dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
        loadArgs.pop('unitNames')
        loadArgs.pop('unitQuery')
        if secondaryUnitQuery is None:
            remainingUnits = copy(unitNames)
        else:
            if secondaryPath is None:
                remainingUnits = ns5.listChanNames(
                    dataBlock, secondaryUnitQuery, objType=ns5.Unit)
            else:
                remainingUnits = ns5.listChanNames(
                    secondaryBlock, secondaryUnitQuery, objType=ns5.Unit)
        for idxOuter, firstUnit in enumerate(unitNames):
            if secondaryUnitQuery is None:
                remainingUnits.remove(firstUnit)
            if verbose:
                prf.print_memory_usage(' firstUnit: {}'.format(firstUnit))
                print('{} secondary units to analyze'.format(len(remainingUnits)))
            firstDF = ns5.alignedAsigsToDF(
                dataBlock, [firstUnit],
                **loadArgs)
            for idxInner, secondUnit in enumerate(remainingUnits):
                if verbose:
                    prf.print_memory_usage('secondUnit: {}'.format(secondUnit))
                if secondaryPath is None:
                    secondDF = ns5.alignedAsigsToDF(
                        dataBlock, [secondUnit],
                        **loadArgs)
                else:
                    secondDF = ns5.alignedAsigsToDF(
                        secondaryBlock, [secondUnit],
                        **loadArgs)
                tempRes = fun(
                    firstDF, secondDF, *funArgs, **funKWargs)
                if (idxInner == 0) and (idxOuter == 0):
                    result = [
                        pd.DataFrame(
                            0, index=unitNames,
                            columns=remainingUnits, dtype='float32')
                        for i in range(len(tempRes))]
                for i in range(len(tempRes)):
                    result[i].loc[firstUnit, secondUnit] = tempRes[i]
    elif loadType == 'oneToMany':
        if loadArgs['unitNames'] is None:
            unitNames = ns5.listChanNames(
                dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
        loadArgs.pop('unitNames')
        loadArgs.pop('unitQuery')
        if secondaryUnitQuery is None:
            targetUnits = copy(unitNames)
        else:
            if secondaryPath is None:
                targetUnits = ns5.listChanNames(
                    dataBlock, secondaryUnitQuery, objType=ns5.Unit)
            else:
                targetUnits = ns5.listChanNames(
                    secondaryBlock, secondaryUnitQuery, objType=ns5.Unit)
        for idxOuter, firstUnit in enumerate(unitNames):
            if verbose:
                prf.print_memory_usage(' firstUnit: {}'.format(firstUnit))
            firstDF = ns5.alignedAsigsToDF(
                dataBlock, [firstUnit],
                **loadArgs)
            if secondaryPath is None:
                secondDF = ns5.alignedAsigsToDF(
                    dataBlock, targetUnits,
                    **loadArgs)
            else:
                secondDF = ns5.alignedAsigsToDF(
                    secondaryBlock, targetUnits,
                    **loadArgs)
            tempRes = fun(
                firstDF, secondDF, *funArgs, **funKWargs)
            if (idxOuter == 0):
                result = [
                    pd.Series(
                        0, index=unitNames, dtype='float32')
                    for i in range(len(tempRes))]
            for i in range(len(tempRes)):
                result[i].loc[firstUnit] = tempRes[i]
    for i in range(len(resultNames)):
        result[i].to_hdf(resultPath, resultNames[i], format='fixed')
    if lazy:
        dataReader.file.close()
    return result


def splitApplyCombine(
        asigWide, fun=None, resultPath=None,
        funArgs=[], funKWArgs={},
        rowKeys=None, colKeys=None, useDask=False, nPartitionMultiplier=1,
        daskPersist=True, daskProgBar=True, daskResultMeta=None, daskViaDisk=False,
        daskComputeOpts={}, newMetadataNames=[], sortOutputIndex=True, sortIndexBy=None,
        sortOutputColumns=True, sortColumnsBy=None, okToDeleteInput=False):
    if isinstance(rowKeys, str):
        rowKeys = [rowKeys, ]
    if isinstance(colKeys, str):
        colKeys = [colKeys, ]
    # TODO: test column iteration
    # TODO: handle transformation / aggregation / filtration
    # look to https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
    wideColumns = asigWide.columns.copy()
    wideIndex = asigWide.index.copy()
    #
    if colKeys is not None:
        asigStack = asigWide.stack(level=colKeys)
        stackedColNames = colKeys
    else:
        stackedColNames = []
        asigStack = asigWide.copy()
    if okToDeleteInput:
        del asigWide
    # TODO: edge case for series?
    stackColumns = asigStack.columns.copy()
    stackIndex = asigStack.index.copy()
    #
    dataColNames = stackColumns.to_list()
    #
    asigStack.reset_index(inplace=True)
    if useDask:
        if daskViaDisk:
            tempPath = '/users/rdarie/Desktop/Scratch Partition Neural Recordings/dask-temp.par'
            asigToSave = asigStack
            asigToSave.columns = ['{}'.format(cN) for cN in asigToSave.columns]
            dataColNames = ['{}'.format(cN) for cN in dataColNames]
            asigToSave.to_parquet(tempPath)  # checkDF = pd.read_parquet(tempPath)
            del asigStack, asigToSave
            tempDaskDF = dd.read_parquet(tempPath)
        else:
            tempDaskDF = dd.from_pandas(
                asigStack,
                npartitions=nPartitionMultiplier*multiprocessing.cpu_count())
        funKWArgs['dataColNames'] = dataColNames
        #
        if daskResultMeta is not None:
            daskResultMeta = pd.concat([stackIndex.to_frame().dtypes, daskResultMeta])
            resultCollection = (
                tempDaskDF
                .groupby(by=rowKeys, group_keys=False)
                .apply(fun, *funArgs, meta=daskResultMeta, **funKWArgs))
        else:
            resultCollection = (
                tempDaskDF
                .groupby(by=rowKeys, group_keys=False)
                .apply(fun, *funArgs, **funKWArgs))
        #
        if 'scheduler' in daskComputeOpts:
            schedulerType = daskComputeOpts.pop('scheduler')
            if schedulerType == 'single-threaded':
                # put it back, no client
                daskComputeOpts['scheduler'] = 'single-threaded'
                daskClient = None
                daskClient = Client()
            elif schedulerType == 'processes':
                daskClient = Client(LocalCluster(processes=True))
            elif schedulerType == 'threads':
                daskClient = Client(LocalCluster(processes=False))
            else:
                daskClient = Client()
                print('Scheduler name is not correct!')
        else:
            daskClient = Client()
        if daskClient is not None:
            print(
                '\n\nDiagnostics available at {}\n(from a browser on the same node as the computation)\n\n'.format(
                    daskClient.dashboard_link))
        #
        if daskPersist:
            resultCollection.persist()
        if daskProgBar:
            with ProgressBar(minimum=10, dt=5):
                resultDF = resultCollection.compute(**daskComputeOpts)
        else:
            resultDF = resultCollection.compute(**daskComputeOpts)
    else:
        funKWArgs['dataColNames'] = dataColNames
        resultDF = (
            asigStack
            .groupby(by=rowKeys, group_keys=False)
            .apply(fun, *funArgs, **funKWArgs))
    # sort the index (comes back from dask out of order)
    resultDF.sort_index(kind='mergesort', axis='index', inplace=True, sort_remaining=False)
    # find out which metadata fields are still present after the function
    presentMetaNames = [cN for cN in stackIndex.names if cN in resultDF.columns]
    # if I stacked a metadata level and it is still there, unstack it.
    if len(stackedColNames):
        stackedColNamesOut = [cN for cN in stackedColNames if cN in presentMetaNames]
    else:
        stackedColNamesOut = []
    # if columns of the result are metadata, indicate in fuction call and add here
    allMetaNames = presentMetaNames + newMetadataNames
    resultDF.set_index(allMetaNames, inplace=True)
    if len(stackedColNamesOut):
        resultDF = resultDF.unstack(stackedColNamesOut)
    if sortOutputIndex:
        if sortIndexBy is None:
            sortIndexBy = ['segment', 'originalIndex', 't']
            for cN in ['bin']:
                if cN in resultDF.index.names:
                    sortIndexBy = sortIndexBy + [cN]
            for cN in ['lag', 'feature']:
                if cN in resultDF.index.names:
                    sortIndexBy = [cN] + sortIndexBy
        resultDF.sort_index(
            level=sortIndexBy,
            axis='index', inplace=True,
            kind='mergesort', sort_remaining=False)
    if sortOutputColumns:
        resultDF.sort_index(
            level=sortColumnsBy,
            axis='columns',
            inplace=True, kind='mergesort', sort_remaining=False)
    # resultDF.index.get_level_values('t')
    # asigStack.index.get_level_values('t')
    return resultDF


def applyFunGrouped(
        asigWide, groupBy, testVar,
        fun=None, funArgs=[], funKWargs={},
        resultNames=None,
        plotting=False):
    # groupBy refers to index groups
    # testVar refers to column groups
    # really should rename these to something more intuitive
    if (isinstance(groupBy, list)) and (len(groupBy) == 1):
        groupBy = groupBy[0]
    if isinstance(groupBy, str):
        resultIndex = pd.Index(
            sorted(asigWide.groupby(by=groupBy).groups.keys()))
        resultIndex.name = groupBy
    elif groupBy is None:
        resultIndex = pd.Index(['all'])
        resultIndex.name = 'all'
    else:
        resultIndex = pd.MultiIndex.from_tuples(
            sorted(asigWide.groupby(by=groupBy).groups.keys()),
            names=groupBy)
    #
    if (isinstance(testVar, list)) and (len(testVar) == 1):
        testVar = testVar[0]
    if isinstance(testVar, str):
        resultColumns = pd.Index(
            sorted(asigWide.groupby(by=testVar).groups.keys()))
        resultColumns.name = testVar
    elif testVar is None:
        resultColumns = pd.Index(['all'])
        resultColumns.name = 'all'
    else:
        resultColumns = pd.MultiIndex.from_tuples(
            sorted(asigWide.groupby(by=testVar).groups.keys()),
            names=testVar)
    #
    result = {
        rName: pd.DataFrame(
            np.nan, index=resultIndex, columns=resultColumns)
        for rName in resultNames}
    if groupBy is not None:
        groupIter = asigWide.groupby(groupBy)
    else:
        groupIter = {'all': asigWide}.items()
    for name, group in groupIter:
        if testVar is not None:
            subGroupIter = group.groupby(testVar)
        else:
            subGroupIter = {'all': group}.items()
        for subName, subGroup in subGroupIter:
            tempRes = fun(subGroup, *funArgs, **funKWargs)
            # 
            for resIdx, res in enumerate(np.atleast_1d(tempRes)):
                rName = resultNames[resIdx]
                # print('{}, {}, {}'.format(rName, name, subName))
                result[rName].loc[name, subName] = res
    return result


def compareMeansGrouped(
        asigWide, groupBy, testVar, referenceTimeWindow=None,
        tStart=None, tStop=None,
        testWidth=100e-3, testStride=20e-3, pThresh=1e-3,
        correctMultiple=True, plotting=False):
    if tStart is None:
        tStart = asigWide.columns[0]
    if tStop is None:
        tStop = asigWide.columns[-1]
    testBins = np.arange(
        tStart + testWidth / 2, tStop - testWidth / 2, testStride)
    if isinstance(groupBy, str):
        if (groupBy not in asigWide.index.names) and (asigWide.size > 0):
            asigWide.loc[:, groupBy] = 'NA'
            asigWide.set_index(groupBy, append=True, inplace=True)
    elif isinstance(groupBy, list):
        for gb in groupBy:
            if (gb not in asigWide.index.names) and (asigWide.size > 0):
                asigWide.loc[:, gb] = 'NA'
                asigWide.set_index(gb, append=True, inplace=True)
    if isinstance(testVar, str):
        if (testVar not in asigWide.index.names) and (asigWide.size > 0):
            asigWide.loc[:, testVar] = 'NA'
            asigWide.set_index(testVar, append=True, inplace=True)
    elif isinstance(testVar, list):
        for gb in testVar:
            if (gb not in asigWide.index.names) and (asigWide.size > 0):
                asigWide.loc[:, gb] = 'NA'
                asigWide.set_index(gb, append=True, inplace=True)
    if (isinstance(groupBy, list)) and (len(groupBy) == 1):
        groupBy = groupBy[0]

    if (isinstance(testVar, list)) and (len(testVar) == 1):
        testVar = testVar[0]
    # 
    if isinstance(groupBy, str):
        testIndex = pd.Index(
            asigWide.groupby(by=groupBy).groups.keys())
        testIndex.name = groupBy
    elif groupBy is None:
        testIndex = pd.Index(['all'])
        testIndex.name = 'all'
    else:
        testIndex = pd.MultiIndex.from_tuples(
            asigWide.groupby(by=groupBy).groups.keys(),
            names=groupBy)
    pVals = pd.DataFrame(
        np.nan,
        index=testIndex,
        columns=testBins)
    pVals.columns.name = 'bin'
    statVals = pd.DataFrame(
        np.nan,
        index=testIndex,
        columns=testBins)
    statVals.columns.name = 'bin'
    if referenceTimeWindow is not None:
        refTMask = (
            (asigWide.columns >= referenceTimeWindow[0]) &
            (asigWide.columns < referenceTimeWindow[1])
            )
    for testBin in testBins:
        tMask = (
            (asigWide.columns >= testBin - testWidth / 2) &
            (asigWide.columns < testBin + testWidth / 2)
            )
        if groupBy is not None:
            groupIter = asigWide.groupby(groupBy)
        else:
            groupIter = {'all': asigWide}.items()
        for name, group in groupIter:
            testAsig = group.loc[:, tMask]
            testGroups = [
                i.mean(axis=1).to_numpy()
                for nm, i in testAsig.groupby(testVar)]
            #
            if referenceTimeWindow is not None:
                refAsig = group.loc[:, refTMask]
                refGroups = [
                    # np.ravel(i)
                    i.mean(axis=1).to_numpy()
                    for nm, i in refAsig.groupby(testVar)]
                testGroups = [testGroups[0], refGroups[0]]
            # groupSizes = [i.shape[0] for i in testGroups]
            # maxSize = int(np.mean(groupSizes))
            # testGroups = [t[:maxSize] for t in testGroups]
            if len(testGroups) > 1:
                try:
                    # isNormal = [pg.normality(tg)['normal'].iloc[0] for tg in testGroups]
                    # equalVariance = pg.homoscedasticity(testGroups)
                    stat, p = stats.kruskal(*testGroups, nan_policy='raise')
                    if plotting:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        distBins = np.linspace(
                            group.min().min(), group.max().max(), 100)
                        fig, ax = plt.subplots(len(testGroups), 1, sharex=True)
                        for tIdx, tGrp in enumerate(testGroups):
                            sns.distplot(
                                tGrp, bins=distBins, kde=False, ax=ax[tIdx])
                        featName = pd.unique(
                            asigWide.index.get_level_values('feature'))
                        if len(featName):
                            ax[tIdx].set_xlabel('{}, {}'.format(featName[0], name))
                        if referenceTimeWindow is not None:
                            ax[0].set_title('test group')
                            ax[1].set_title('reference group')
                        plt.suptitle(
                            'epoch {:.3f}: t_stat={:.3f}; p_val={:.6f}'
                            .format(testBin, stat, p))
                        # plt.suptitle('sample size {}'.format(tMask.sum()))
                        plt.show()
                    pVals.loc[name, testBin] = p
                    statVals.loc[name, testBin] = stat
                except Exception:
                    traceback.print_exc()
                    pVals.loc[name, testBin] = 1
                    statVals.loc[name, testBin] = np.nan
    if correctMultiple:
        flatPvals = pVals.stack()
        _, fixedPvals, _, _ = mt(flatPvals.values, method='holm')
        flatPvals.loc[:] = fixedPvals
        flatPvals = flatPvals.unstack('bin')
        pVals.loc[flatPvals.index, flatPvals.columns] = flatPvals
    significanceVals = pVals < pThresh
    return pVals, statVals, significanceVals


def compareISIsGrouped(
        asigWide, groupBy, testVar,
        tStart=None, tStop=None, referenceTimeWindow=None,
        testWidth=100e-3, testStride=20e-3,
        pThresh=1e-3, correctMultiple=True, plotting=False):
    if tStart is None:
        tStart = asigWide.columns[0]
    if tStop is None:
        tStop = asigWide.columns[-1]
    testBins = np.arange(
        tStart + testWidth / 2, tStop - testWidth / 2, testStride)
    #
    if (isinstance(groupBy, list)) and (len(groupBy) == 1):
        groupBy = groupBy[0]
    if (isinstance(testVar, list)) and (len(testVar) == 1):
        testVar = testVar[0]
    #
    if isinstance(groupBy, str):
        testIndex = pd.Index(
            asigWide.groupby(by=groupBy).groups.keys())
        testIndex.name = groupBy
    elif groupBy is None:
        testIndex = pd.Index(['all'])
        testIndex.name = 'all'
    else:
        testIndex = pd.MultiIndex.from_tuples(
            asigWide.groupby(by=groupBy).groups.keys(),
            names=groupBy)
    pVals = pd.DataFrame(
        np.nan,
        index=testIndex,
        columns=testBins)
    pVals.columns.name = 'bin'
    statVals = pd.DataFrame(
        np.nan,
        index=testIndex,
        columns=testBins)
    statVals.columns.name = 'bin'
    for testBin in testBins:
        #  try:
        tMask = (
            (asigWide.columns > testBin - testWidth / 2) &
            (asigWide.columns < testBin + testWidth / 2)
            )
        #  except Exception:
        #      
        testAsig = asigWide.loc[:, tMask]
        if groupBy is not None:
            groupIter = testAsig.groupby(groupBy)
        else:
            groupIter = {'all': testAsig}.items()
        for name, group in groupIter:
            def getSC(rasterDF):
                # listISI = []
                listSpikeCount = []
                for rIdx, row in rasterDF.iterrows():
                    # spikeTimes = row.index[row > 0]
                    listSpikeCount.append(np.sqrt((row > 0).sum()))
                    # if any(spikeTimes):
                    #     listISI += (np.diff(spikeTimes).tolist())
                return listSpikeCount
            testGroups = [
                getSC(i)
                for nm, i in group.groupby(testVar)]
            # groupSizes = [i.shape[0] for i in testGroups]
            # maxSize = int(np.mean(groupSizes))
            # testGroups = [t[:maxSize] for t in testGroups]
            if len(testGroups) > 1:
                try:
                    # isNormal = [pg.normality(tg)['normal'].iloc[0] for tg in testGroups]
                    # equalVariance = pg.homoscedasticity(testGroups)
                    stat, p = stats.kruskal(*testGroups, nan_policy='raise')
                    # stat, p = stats.f_oneway(*testGroups)
                    if plotting:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(len(testGroups), 1, sharex=True)
                        for tIdx, tGrp in enumerate(testGroups):
                            sns.distplot(
                                tGrp, kde=False, ax=ax[tIdx])
                        featName = pd.unique(
                            asigWide.index.get_level_values('feature'))
                        if len(featName):
                            ax[tIdx].set_xlabel('{}'.format(featName[0]))
                        plt.suptitle(
                            't = {:.3f}; p = {:.6f}'
                            .format(stat, p))
                        # plt.suptitle('sample size {}'.format(tMask.sum()))
                        plt.show()
                    pVals.loc[name, testBin] = p
                    statVals.loc[name, testBin] = stat
                except Exception:
                    #  traceback.print_exc()
                    pVals.loc[name, testBin] = 1
                    statVals.loc[name, testBin] = np.nan
    if correctMultiple:
        flatPvals = pVals.stack()
        _, fixedPvals, _, _ = mt(flatPvals.values, method='holm')
        flatPvals.loc[:] = fixedPvals
        flatPvals = flatPvals.unstack('bin')
        pVals.loc[flatPvals.index, flatPvals.columns] = flatPvals
    significanceVals = pVals < pThresh
    return pVals, statVals, significanceVals


def facetGridApplyFunGrouped(
        dataBlock, resultPath,
        fun=None, funArgs=[], funKWargs={},
        resultNames=None,
        loadArgs={},
        rowColOpts={},
        limitPages=None, plotting=False, verbose=False):
    #  get list of units
    if loadArgs['unitNames'] is None:
        loadArgs['unitNames'] = ns5.listChanNames(
            dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
    originalUnitNames = loadArgs.pop('unitNames')
    unitNames = originalUnitNames
    originalUnitQuery = loadArgs.pop('unitQuery')
    #  set up significance testing
    if (rowColOpts['rowControl'] is None) and (rowColOpts['colControl'] is None):
        # test all rows and columns
        subQuery = None
    else:
        subQueryList = []
        if rowColOpts['rowControl'] is not None:
            if isinstance(rowColOpts['rowControl'], str):
                rowQ = '\'' + rowColOpts['rowControl'] + '\''
            else:
                rowQ = rowColOpts['rowControl']
            subQueryList.append(
                '({} != {})'.format(rowColOpts['rowName'], rowQ))
        if rowColOpts['colControl'] is not None:
            if isinstance(rowColOpts['colControl'], str):
                colQ = '\'' + rowColOpts['colControl'] + '\''
            else:
                colQ = rowColOpts['colControl']
            subQueryList.append(
                '({} != {})'.format(rowColOpts['colName'], colQ))
        subQuery = '&'.join(subQueryList)
    #  test each row and column separately
    rowColGroupBy = [
        i
        for i in [rowColOpts['rowName'], rowColOpts['colName']]
        if i is not None
        ]
    if not len(rowColGroupBy):
        rowColGroupBy = None
    # hue and/or style variable will index the result
    indVar = [
        i
        for i in [rowColOpts['hueName'], rowColOpts['styleName']]
        if i is not None
        ]
    assert indVar is not None
    allRes = {}
    for idx, unitName in enumerate(unitNames):
        if verbose:
            print('on unit {}'.format(unitName))
        asigWide = ns5.alignedAsigsToDF(
            dataBlock, [unitName],
            **loadArgs)
        if subQuery is not None:
            asigWide = asigWide.query(subQuery)
        #  get results
        tempRes = applyFunGrouped(
            asigWide, rowColGroupBy, indVar,
            fun=fun, funArgs=funArgs, funKWargs=funKWargs,
            resultNames=resultNames,
            plotting=plotting)
        allRes.update({unitName: tempRes})
        if limitPages is not None:
            if idx >= limitPages:
                break
    # give these back in case needed (dictionaries are passed by reference)
    loadArgs['unitNames'] = originalUnitNames
    loadArgs['unitQuery'] = originalUnitQuery
    return allRes


def facetGridCompareMeans(
        dataBlock=None, statsTestPath=None,
        limitPages=None, verbose=False, compareISIs=False,
        loadArgs={}, rowColOpts={}, statsTestOpts={}):
    #  get list of units
    if loadArgs['unitNames'] is None:
        loadArgs['unitNames'] = ns5.listChanNames(
            dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
    originalUnitNames = loadArgs.pop('unitNames')
    unitNames = originalUnitNames
    originalUnitQuery = loadArgs.pop('unitQuery')
    #  set up significance testing
    if (rowColOpts['rowControl'] is None) and (rowColOpts['colControl'] is None):
        # test all rows and columns
        sigTestQuery = None
    else:
        sigTestQueryList = []
        if rowColOpts['rowControl'] is not None:
            if isinstance(rowColOpts['rowControl'], str):
                rowQ = '\'' + rowColOpts['rowControl'] + '\''
            else:
                rowQ = rowColOpts['rowControl']
            sigTestQueryList.append(
                '({} != {})'.format(rowColOpts['rowName'], rowQ))
        if rowColOpts['colControl'] is not None:
            if isinstance(rowColOpts['colControl'], str):
                colQ = '\'' + rowColOpts['colControl'] + '\''
            else:
                colQ = rowColOpts['colControl']
            sigTestQueryList.append(
                '({} != {})'.format(rowColOpts['colName'], colQ))
        sigTestQuery = '&'.join(sigTestQueryList)
    #  test each row and column separately
    sigTestGroupBy = [
        i
        for i in [rowColOpts['rowName'], rowColOpts['colName']]
        if i is not None
        ]
    if not len(sigTestGroupBy):
        sigTestGroupBy = None
    # test jointly for significance of hue and/or style
    sigTestVar = [
        i
        for i in [rowColOpts['hueName'], rowColOpts['styleName']]
        if i is not None
        ]
    if statsTestOpts['referenceTimeWindow'] is None:
        # if comparing across a level, must have something there
        assert len(sigTestVar) > 0
    #
    correctMultiple = statsTestOpts.pop('correctMultiple')
    #
    allPVals = {}
    allStatVals = {}
    allSigVals = {}
    for idx, unitName in enumerate(unitNames):
        if verbose:
            print('    facetGridCompareMeans on unit {}'.format(unitName))
        ##
        # debugging
        # statsTestOpts['plotting'] = (unitName == 'position#0')
        ##
        asigWide = ns5.alignedAsigsToDF(
            dataBlock, [unitName],
            **loadArgs)
        if sigTestQuery is not None:
            sigTestAsig = asigWide.query(sigTestQuery)
        else:
            sigTestAsig = asigWide
        # cannot have multiple traces per facet (one p-value per time bin)
        if len(sigTestVar) > 0:
            if statsTestOpts['referenceTimeWindow'] is not None:
                assert sigTestAsig.groupby(sigTestVar).ngroups == 1
        #  get significance test results (correct for multiple comparionsons at the end, not here)
        if not compareISIs:
            pVals, statVals, sigVals = compareMeansGrouped(
                sigTestAsig, testVar=sigTestVar,
                groupBy=sigTestGroupBy, correctMultiple=False,
                **statsTestOpts)
        else:
            pVals, statVals, sigVals = compareISIsGrouped(
                sigTestAsig, testVar=sigTestVar,
                groupBy=sigTestGroupBy, correctMultiple=False,
                **statsTestOpts)
        allPVals.update({unitName: pVals})
        allStatVals.update({unitName: statVals})
        allSigVals.update({unitName: sigVals})
        if limitPages is not None:
            if idx >= limitPages:
                break
    allPValsWide = pd.concat(allPVals, names=['unit'] + pVals.index.names)
    if correctMultiple:
        origShape = allPValsWide.shape
        flatPvals = allPValsWide.to_numpy().reshape(-1)
        try:
            _, fixedPvals, _, _ = mt(flatPvals, method='holm')
        except Exception:
            fixedPvals = flatPvals * flatPvals.size
        allPValsWide.iloc[:, :] = fixedPvals.reshape(origShape)
        allSigValsWide = allPValsWide < statsTestOpts['pThresh']
    allPValsWide.to_hdf(statsTestPath, 'p', format='fixed')
    allStatValsWide = pd.concat(allStatVals, names=['unit'] + statVals.index.names)
    allStatValsWide.to_hdf(statsTestPath, 'stat', format='fixed')
    allSigValsWide = pd.concat(allSigVals, names=['unit'] + sigVals.index.names)
    allSigValsWide.to_hdf(statsTestPath, 'sig', format='fixed')
    # give these back in case needed (dictionaries are passed by reference)
    loadArgs['unitNames'] = originalUnitNames
    loadArgs['unitQuery'] = originalUnitQuery
    return allPValsWide, allStatValsWide, allSigValsWide

def facetGridCompareMeansDataFrame(
        dataDF=None, statsTestPath=None,
        limitPages=None, verbose=False, compareISIs=False,
        rowColOpts={}, statsTestOpts={}):
    #  set up significance testing
    if (rowColOpts['rowControl'] is None) and (rowColOpts['colControl'] is None):
        # test all rows and columns
        sigTestQuery = None
    else:
        sigTestQueryList = []
        if rowColOpts['rowControl'] is not None:
            if isinstance(rowColOpts['rowControl'], str):
                rowQ = '\'' + rowColOpts['rowControl'] + '\''
            else:
                rowQ = rowColOpts['rowControl']
            sigTestQueryList.append(
                '({} != {})'.format(rowColOpts['rowName'], rowQ))
        if rowColOpts['colControl'] is not None:
            if isinstance(rowColOpts['colControl'], str):
                colQ = '\'' + rowColOpts['colControl'] + '\''
            else:
                colQ = rowColOpts['colControl']
            sigTestQueryList.append(
                '({} != {})'.format(rowColOpts['colName'], colQ))
        sigTestQuery = '&'.join(sigTestQueryList)
    #  test each row and column separately
    sigTestGroupBy = [
        i
        for i in [rowColOpts['rowName'], rowColOpts['colName']]
        if i is not None
        ]
    if not len(sigTestGroupBy):
        sigTestGroupBy = None
    # test jointly for significance of hue and/or style
    sigTestVar = [
        i
        for i in [rowColOpts['hueName'], rowColOpts['styleName']]
        if i is not None
        ]
    if statsTestOpts['referenceTimeWindow'] is None:
        # if comparing across a level, must have something there
        assert len(sigTestVar) > 0
    #
    correctMultiple = statsTestOpts.pop('correctMultiple')
    #
    allPVals = {}
    allStatVals = {}
    allSigVals = {}
    if verbose:
        print('running facetGridCompareMeansDataFrame...\n')
        columnsIter = tqdm(dataDF.columns)
    else:
        columnsIter = (dataDF.columns)
    for idx, unitName in enumerate(columnsIter):
        ##
        # debugging
        # statsTestOpts['plotting'] = (unitName == 'position#0')
        ##
        asigWide = dataDF.loc[:, [unitName]]
        if 'bin' in asigWide.index.names:
            asigWide = asigWide.stack(asigWide.columns.names).unstack('bin')
        if sigTestQuery is not None:
            sigTestAsig = asigWide.query(sigTestQuery)
        else:
            sigTestAsig = asigWide
        # cannot have multiple traces per facet (one p-value per time bin)
        if len(sigTestVar) > 0:
            if statsTestOpts['referenceTimeWindow'] is not None:
                assert sigTestAsig.groupby(sigTestVar).ngroups == 1
        #  get significance test results (correct for multiple comparionsons at the end, not here)
        if not compareISIs:
            pVals, statVals, sigVals = compareMeansGrouped(
                sigTestAsig, testVar=sigTestVar,
                groupBy=sigTestGroupBy, correctMultiple=False,
                **statsTestOpts)
        else:
            pVals, statVals, sigVals = compareISIsGrouped(
                sigTestAsig, testVar=sigTestVar,
                groupBy=sigTestGroupBy, correctMultiple=False,
                **statsTestOpts)
        allPVals.update({unitName: pVals})
        allStatVals.update({unitName: statVals})
        allSigVals.update({unitName: sigVals})
        if limitPages is not None:
            if idx >= limitPages:
                break
    allPValsWide = pd.concat(allPVals, names=dataDF.columns.names)
    if correctMultiple:
        origShape = allPValsWide.shape
        flatPvals = allPValsWide.to_numpy().reshape(-1)
        try:
            _, fixedPvals, _, _ = mt(flatPvals, method='holm')
        except Exception:
            fixedPvals = flatPvals * flatPvals.size
        allPValsWide.iloc[:, :] = fixedPvals.reshape(origShape)
        allSigValsWide = allPValsWide < statsTestOpts['pThresh']
    allPValsWide.to_hdf(statsTestPath, 'p', format='fixed')
    allStatValsWide = pd.concat(allStatVals, names=dataDF.columns.names)
    allStatValsWide.to_hdf(statsTestPath, 'stat', format='fixed')
    allSigValsWide = pd.concat(allSigVals, names=dataDF.columns.names)
    allSigValsWide.to_hdf(statsTestPath, 'sig', format='fixed')
    return allPValsWide, allStatValsWide, allSigValsWide


def meanRAUC(
        asigWide, baseline=None,
        tStart=None, tStop=None):
    rAUCDF = rAUC(
        asigWide, baseline=baseline,
        tStart=tStart, tStop=tStop)
    return rAUCDF.mean()


def rAUC(
        asigWide, baseline=None,
        tStart=None, tStop=None):
    #
    if tStart is None:
        tStart = asigWide.columns[0]
    if tStop is None:
        tStop = asigWide.columns[-1]
    tMask = hf.getTimeMaskFromRanges(
        asigWide.columns, [(tStart, tStop)])
    if baseline is not None:
        if baseline == 'mean':
            bLine = asigWide.mean(axis=1)
        elif baseline == 'median':
            bLine = asigWide.median(axis=1)
        else:
            bLine = baseline
    else:
        bLine = 0
    '''dt = asigWide.columns[1] - asigWide.columns[0]
    rAUCDF = (
        asigWide.loc[:, tMask].subtract(bLine, axis='index')
        .abs().sum(axis='columns') * dt)'''
    rAUCDF = (asigWide.loc[:, tMask] - bLine).abs().mean(axis='columns')
    # rAUCDF = (asigWide.loc[:, tMask] - bLine).std(axis='columns')
    return rAUCDF


def genDetrender(
        timeWindow=None, useMean=False):
    def detrend(waveDF, spkTrain):
        if timeWindow is None:
            trendMask = slice(None)
        else:
            trendMask = (
                (waveDF.columns >= timeWindow[0]) &
                (waveDF.columns < timeWindow[1]))
            if not trendMask.any():
                print(Warning('detrender failed to find matching time range! Returning data as is'))
                return waveDF
        if useMean:
            trend = waveDF.loc[:, trendMask].mean(axis='columns')
        else:
            trend = waveDF.loc[:, trendMask].median(axis='columns')
        return waveDF.sub(trend, axis='rows')
    return detrend
