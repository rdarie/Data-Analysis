import dill as pickle
import os
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.profiling as prf
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests as mt
from copy import copy
import pdb, traceback


def processAlignQueryArgs(namedQueries, alignQuery=None, **kwargs):
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


def processUnitQueryArgs(
        namedQueries, scratchFolder, selector=None, unitQuery=None,
        inputBlockName='', **kwargs):
    #
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
        triggeredPath=None, resultPath=None, resultName=None,
        fun=None, funArgs=[], funKWargs={},
        lazy=None, loadArgs={},
        loadType='all', applyType='self',
        verbose=False):
    if verbose:
        prf.print_memory_usage('about to load dataBlock')
    dataReader, dataBlock = ns5.blockFromPath(triggeredPath, lazy=lazy)
    if verbose:
        prf.print_memory_usage('done loading dataBlock')
    if loadType == 'all':
        alignedAsigsDF = ns5.alignedAsigsToDF(
            dataBlock, **loadArgs)
        if verbose:
            prf.print_memory_usage('just loaded alignedAsigs')
        if applyType == 'self':
            result = getattr(alignedAsigsDF, fun)(*funArgs, **funKWargs)
        if applyType == 'func':
            result = fun(alignedAsigsDF, *funArgs, **funKWargs)
    elif loadType == 'elementwise':
        if loadArgs['unitNames'] is None:
            unitNames = ns5.listChanNames(
                dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
        loadArgs.pop('unitNames')
        loadArgs.pop('unitQuery')
        result = pd.Series(
            0, index=unitNames, dtype='float32')
        for idxOuter, firstUnit in enumerate(unitNames):
            if verbose:
                prf.print_memory_usage(' firstUnit: {}'.format(firstUnit))
            firstDF = ns5.alignedAsigsToDF(
                dataBlock, [firstUnit],
                **loadArgs)
            result.loc[firstUnit] = fun(firstDF)
    elif loadType == 'pairwise':
        if loadArgs['unitNames'] is None:
            unitNames = ns5.listChanNames(
                dataBlock, loadArgs['unitQuery'], objType=ns5.Unit)
        loadArgs.pop('unitNames')
        loadArgs.pop('unitQuery')
        remainingUnits = copy(unitNames)
        result = pd.DataFrame(
            0, index=unitNames, columns=unitNames, dtype='float32')
        for idxOuter, firstUnit in enumerate(unitNames):
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
                secondDF = ns5.alignedAsigsToDF(
                    dataBlock, [secondUnit],
                    **loadArgs)
                result.loc[firstUnit, secondUnit], _ = fun(
                    firstDF.to_numpy().flatten(),
                    secondDF.to_numpy().flatten())
    result.to_hdf(resultPath, resultName, format='table')
    if lazy:
        dataReader.file.close()
    return result


def compareMeansGrouped(
        asigWide, groupBy, testVar,
        tStart=None, tStop=None,
        testWidth=100e-3, testStride=20e-3,
        pThresh=1e-3,
        correctMultiple=True):
    
    if tStart is None:
        tStart = asigWide.columns[0]
    if tStop is None:
        tStop = asigWide.columns[-1]
    testBins = np.arange(
        tStart, tStop, testStride)

    if (isinstance(groupBy, list)) and (len(groupBy) == 1):
        groupBy = groupBy[0]

    if (isinstance(testVar, list)) and (len(testVar) == 1):
        testVar = testVar[0]
    
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
        tMask = (
            (asigWide.columns > testBin - testWidth / 2) &
            (asigWide.columns < testBin + testWidth / 2)
            )
        testAsig = asigWide.loc[:, tMask]
        if groupBy is not None:
            groupIter = testAsig.groupby(groupBy)
        else:
            groupIter = {'all': testAsig}.items()
        for name, group in groupIter:
            testGroups = [
                np.ravel(i)
                for _, i in group.groupby(testVar)]
            groupSizes = [i.shape[0] for i in testGroups]
            maxSize = int(np.mean(groupSizes))
            testGroups = [t[:maxSize] for t in testGroups]
            if len(testGroups) > 1:
                try:
                    # stat, p = stats.kruskal(*testGroups)
                    stat, p = stats.f_oneway(*testGroups)
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


def facetGridCompareMeans(
        dataBlock, statsTestPath,
        loadArgs={},
        rowColOpts={},
        statsTestOpts={}):
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
    assert sigTestVar is not None
    #
    correctMultiple = statsTestOpts.pop('correctMultiple')
    #
    allPVals = {}
    allStatVals = {}
    allSigVals = {}
    for idx, unitName in enumerate(unitNames):
        asigWide = ns5.alignedAsigsToDF(
            dataBlock, [unitName],
            **loadArgs)
        if sigTestQuery is not None:
            sigTestAsig = asigWide.query(sigTestQuery)
        else:
            sigTestAsig = asigWide
        #  get significance test results
        pVals, statVals, sigVals = compareMeansGrouped(
            sigTestAsig, testVar=sigTestVar,
            groupBy=sigTestGroupBy, correctMultiple=False,
            **statsTestOpts)
        allPVals.update({unitName: pVals})
        allStatVals.update({unitName: statVals})
        allSigVals.update({unitName: sigVals})
    allPValsWide = pd.concat(allPVals, names=['unit'] + pVals.index.names)
    if correctMultiple:
        flatPvals = allPValsWide.stack()
        _, fixedPvals, _, _ = mt(flatPvals.values, method='holm')
        flatPvals.loc[:] = fixedPvals
        flatPvals = flatPvals.unstack('bin')
        allPValsWide.loc[flatPvals.index, flatPvals.columns] = flatPvals
        allSigValsWide = allPValsWide < statsTestOpts['pThresh']
    allPValsWide.to_hdf(statsTestPath, 'p', format='table')
    allStatValsWide = pd.concat(allStatVals, names=['unit'] + statVals.index.names)
    allStatValsWide.to_hdf(statsTestPath, 'stat', format='table')
    allSigValsWide = pd.concat(allSigVals, names=['unit'] + sigVals.index.names)
    allSigValsWide.to_hdf(statsTestPath, 'sig', format='table')
    # give these back in case needed (dictionaries are passed by reference)
    loadArgs['unitNames'] = originalUnitNames
    loadArgs['unitQuery'] = originalUnitQuery
    return allPValsWide, allStatValsWide, allSigValsWide
