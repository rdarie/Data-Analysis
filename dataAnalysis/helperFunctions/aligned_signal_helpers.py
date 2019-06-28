import dill as pickle
import os
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.profiling as prf
import pandas as pd
from copy import copy
import pdb


def processAlignQueryArgs(namedQueries, alignQuery=None, **kwargs):
    if (alignQuery is None) or (not len(alignQuery)):
        dataQuery = None
    else:
        if alignQuery in namedQueries['align']:
            dataQuery = namedQueries['align'][alignQuery]
        else:
            dataQuery = alignQuery
    return dataQuery


def processUnitQueryArgs(
        namedQueries, scratchFolder, selector=None, unitQuery=None, **kwargs):
    #
    if selector is not None:
        with open(
            os.path.join(
                scratchFolder,
                selector + '.pickle'),
                'rb') as f:
            selectorMetadata = pickle.load(f)
        unitNames = [
            i.replace(selectorMetadata['inputBlockName'], '')
            for i in selectorMetadata['outputFeatures']]
        unitQuery = None
    else:
        unitNames = None
        if unitQuery in namedQueries['unit']:
            outputQuery = namedQueries['unit'][unitQuery]
        else:
            outputQuery = unitQuery
    return unitNames, outputQuery


def applyFun(
        triggeredPath=None, resultPath=None, resultName=None,
        fun=None, lazy=None, loadArgs=None,
        loadType='all', applyType='self',
        verbose=False):
    if verbose:
        prf.print_memory_usage('about to load dataBlock')
    if lazy:
        dataReader = ns5.nixio_fr.NixIO(
            filename=triggeredPath)
        dataBlock = dataReader.read_block(
            block_index=0, lazy=True,
            signal_group_mode='split-all')
    else:
        dataBlock = ns5.loadWithArrayAnn(triggeredPath)
    if verbose:
        prf.print_memory_usage('done loading dataBlock')
    if loadType == 'all':
        alignedAsigsDF = ns5.alignedAsigsToDF(
            dataBlock, **loadArgs)
        if verbose:
            prf.print_memory_usage('just loaded alignedAsigs')
        if applyType == 'self':
            result = getattr(alignedAsigsDF, fun)()
        if applyType == 'func':
            result = fun(alignedAsigsDF)
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
                if idxInner > 5: break
            if idxOuter > 5: break
    result.to_hdf(resultPath, resultName, format='table')
    if lazy:
        dataReader.file.close()
    return result
