import dill as pickle
import os
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.helper_functions_new as hf
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests as mt
from copy import copy
import pdb, traceback


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
        alignSubFolder, prefix,
        maskOutlierTrials=False, window=None,
        **kwargs
        ):
    if maskOutlierTrials:
        resultPath = os.path.join(
            alignSubFolder,
            prefix + '_{}_{}_calc.h5'.format('fr', window))
        return pd.read_hdf(resultPath, 'rejectTrial')
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
        prf.print_memory_usage('about to load dataBlock')
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
        asigWide, groupBy, testVar,
        tStart=None, tStop=None,
        testWidth=100e-3, testStride=20e-3,
        pThresh=1e-3,
        correctMultiple=True,
        plotting=False):

    if tStart is None:
        tStart = asigWide.columns[0]
    if tStop is None:
        tStop = asigWide.columns[-1]
    testBins = np.arange(
        tStart + testWidth / 2, tStop - testWidth / 2, testStride)

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
            testGroups = [
                np.ravel(i)
                for _, i in group.groupby(testVar)]
            groupSizes = [i.shape[0] for i in testGroups]
            maxSize = int(np.mean(groupSizes))
            testGroups = [t[:maxSize] for t in testGroups]
            if len(testGroups) > 1:
                try:
                    stat, p = stats.kruskal(*testGroups, nan_policy='raise')
                    # stat, p = stats.f_oneway(*testGroups)
                    if plotting:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        for tGrp in testGroups:
                            sns.distplot(tGrp, kde=False)
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
        dataBlock, statsTestPath,
        limitPages=None, verbose=False,
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
        if verbose:
            print('on unit {}'.format(unitName))
        asigWide = ns5.alignedAsigsToDF(
            dataBlock, [unitName],
            **loadArgs)
        if sigTestQuery is not None:
            sigTestAsig = asigWide.query(sigTestQuery)
        else:
            sigTestAsig = asigWide
        #  get significance test results (correct for multiple comparionsons at the end, not here)
        pVals, statVals, sigVals = compareMeansGrouped(
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
    dt = asigWide.columns[1] - asigWide.columns[0]
    # 
    rAUCDF = (
        asigWide.loc[:, tMask]
        .subtract(bLine, axis='index')
        .abs().sum(axis=1) * dt)
    return rAUCDF
