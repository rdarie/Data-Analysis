"""08: Calculate binarized array and relevant analogsignals
Usage:
    calcBlockAnalysisNix.py [options]

Options:
    --blockIdx=blockIdx                          which trial to analyze
    --exp=exp                                    which experimental day to analyze
    --lazy                                       load from raw, or regular? [default: False]
    --trigRate=trigRate                          inter trigger interval [default: 60]
    --plotting                                   whether to show diagnostic plots (must have display) [default: False]
    --chanQuery=chanQuery                        how to restrict channels if not providing a list?
    --nspBlockSuffix=nspBlockSuffix              which block to pull
    --nspBlockPrefix=nspBlockPrefix              which block to pull
"""

from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo import (
    Block, Segment, ChannelIndex, Unit,
    Event, Epoch, AnalogSignal, SpikeTrain)
# import neo
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
# from namedQueries import namedQueries
import numpy as np
import pandas as pd
# import elephant.pandas_bridge as elphpdb
# import dataAnalysis.preproc.mdt as preprocINS
import quantities as pq
# import rcsanalysis.packet_func as rcsa_helpers
import os, pdb, traceback
# from importlib import reload

#  load options
from namedQueries import namedQueries
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


def synchronizeDelsysToNSP():
    if arguments['nspBlockPrefix'] is not None:
        nspBlockBasename = '{}{:03d}'.format(arguments['nspBlockPrefix'], arguments['blockIdx'])
    else:
        nspBlockBasename = ns5FileName
    if arguments['nspBlockSuffix'] is not None:
        nspBlockSuffix = '_{}'.format(arguments['nspBlockSuffix'])
    else:
        nspBlockSuffix = ''
    nspPath = os.path.join(
        scratchFolder,
        nspBlockBasename + nspBlockSuffix + '.nix')
    # 
    nspReader, nspBlock = ns5.blockFromPath(
        nspPath, lazy=arguments['lazy'])
    #
    oePath = os.path.join(
        scratchFolder,
        ns5FileName + '_delsys.nix')
    oeReader, oeBlock = ns5.blockFromPath(
        oePath, lazy=arguments['lazy'])

    interTriggerInterval = float(arguments['trigRate']) ** (-1)
    segIdx = 0
    nspSeg = nspBlock.segments[segIdx]
    oeSeg = oeBlock.segments[segIdx]
    #
    oeSyncAsig = oeSeg.filter(name='seg0_{}'.format(synchInfo['delsysToNsp']['synchChanName']))[0]
    try:
        tStart, tStop = synchInfo['delsysToNsp'][blockIdx]['timeRanges']
    except Exception:
        traceback.print_exc()
        tStart = float(oeSyncAsig.times[0] + 2 * pq.s)
        tStop = float(oeSyncAsig.times[-1] - 2 * pq.s)

    oeTimeMask = hf.getTimeMaskFromRanges(
        oeSyncAsig.times, [(tStart, tStop)])
    oeSrs = pd.Series(oeSyncAsig.magnitude[oeTimeMask].flatten())
    oeSrs.loc[oeSrs == 0] = np.nan
    oeSrs.interpolate(method='linear', inplace=True)
    oeSrs.fillna(method='bfill', inplace=True)
    print(
        'On block {}, detecting Delsys threshold crossings.'
        .format(blockIdx))

    oeLims = oeSrs.quantile([1e-6, 1-1e-6]).to_list()
    oeDiffUncertainty = oeSrs.diff().abs().quantile(1-1e-6) / 4
    oeThresh = (oeLims[-1] - oeLims[0]) / 2

    nspSynchChanName = synchInfo['delsysToNsp'][blockIdx]['synchChanName']
    nspSyncAsig = nspSeg.filter(name='seg0_{}'.format(nspSynchChanName))[0]
    try:
        tStart, tStop = synchInfo['nspForDelsys'][blockIdx]['timeRanges']
    except Exception:
        traceback.print_exc()
        tStart = float(nspSyncAsig.times[0] + 2 * pq.s)
        tStop = float(nspSyncAsig.times[-1] - 2 * pq.s)

    nspTimeMask = hf.getTimeMaskFromRanges(
        nspSyncAsig.times, [(tStart, tStop)])
    nspSrs = pd.Series(nspSyncAsig.magnitude[nspTimeMask].flatten())

    nspLims = nspSrs.quantile([1e-3, 1-1e-3]).to_list()
    nspDiffUncertainty = nspSrs.diff().abs().quantile(1-1e-3) / 4
    nspThresh = (nspLims[-1] - nspLims[0]) / 2
    
    oePeakIdx, oeCrossMask = hf.getThresholdCrossings(
        oeSrs, thresh=oeThresh,
        iti=interTriggerInterval, fs=float(oeSyncAsig.sampling_rate),
        edgeType='both', itiWiggle=.2,
        absVal=False, plotting=arguments['plotting'], keep_max=False)
    # oePeakIdx = hf.getTriggers(
    #     oeSrs, iti=interTriggerInterval, fs=float(oeSyncAsig.sampling_rate),
    #     thres=1.5, edgeType='falling', plotting=arguments['plotting'])
    # oeCrossMask = oeSrs.index.isin(oePeakIdx)
    print('Found {} triggers'.format(oePeakIdx.size))
    #
    print(
        'On trial {}, detecting NSP threshold crossings.'
        .format(blockIdx))
    nspPeakIdx, nspCrossMask = hf.getThresholdCrossings(
        nspSrs, thresh=nspThresh,
        iti=interTriggerInterval, fs=float(nspSyncAsig.sampling_rate),
        edgeType='both', itiWiggle=.2,
        absVal=False, plotting=arguments['plotting'], keep_max=False)
    # nspPeakIdx = hf.getTriggers(
    #     nspSrs, iti=interTriggerInterval, itiWiggle=1,
    #     fs=float(nspSyncAsig.sampling_rate), plotting=arguments['plotting'],
    #     thres=2.58, edgeType='both')
    # nspCrossMask = nspSrs.index.isin(nspPeakIdx)
    print('Found {} triggers'.format(nspPeakIdx.size))
    try:
        chooseCrossings = synchInfo['delsysToNsp'][blockIdx]['chooseCrossings']
        oeTimes = (
            oeSyncAsig.times[oeTimeMask][oeCrossMask][chooseCrossings])
    except Exception:
        traceback.print_exc()
        oeTimes = (
            oeSyncAsig.times[oeTimeMask][oeCrossMask])
    try:
        chooseCrossings = synchInfo['nspForDelsys'][blockIdx]['chooseCrossings']
        nspTimes = (
            nspSyncAsig.times[nspTimeMask][nspCrossMask][chooseCrossings])
    except Exception:
        traceback.print_exc()
        nspTimes = (
            nspSyncAsig.times[nspTimeMask][nspCrossMask])
    ###########
    nMissingTriggers = nspTimes.size - oeTimes.size
    sampleWiggle = 5 * oeSyncAsig.sampling_rate.magnitude ** (-1)
    prelimOEMismatch = np.abs(np.diff(oeTimes) - interTriggerInterval * pq.s)
    prelimNSPMismatch = np.abs(np.diff(nspTimes) - interTriggerInterval * pq.s)
    listDiscontinuitiesNSP = np.flatnonzero(prelimNSPMismatch > sampleWiggle)
    listDiscontinuitiesOE = np.flatnonzero(prelimOEMismatch > sampleWiggle)
    # 
    if nMissingTriggers > 0:
        # np.diff(oeTimes)[listDiscontinuitiesOE]
        # np.diff(nspTimes)[listDiscontinuitiesNSP]
        # nspTimes[listDiscontinuitiesNSP]
        # oeTimes[listDiscontinuitiesOE]
        # nspTimes[listDiscontinuitiesNSP] - nspTimes[0]
        # oeTimes[listDiscontinuitiesOE] - oeTimes[0]
        listDiscontinuities = listDiscontinuitiesOE
        nMissingTriggers = nspTimes.size - oeTimes.size
        print('Found {} discontinuities!'.format(len(listDiscontinuities)))
    else:
        # 
        listDiscontinuities = np.flatnonzero(np.abs(prelimNSPMismatch - prelimOEMismatch) > sampleWiggle)
    if len(listDiscontinuities):
        print(' On Delsys clock, discontinuities at:')
        for dIdx in listDiscontinuities:
            print(oeTimes[dIdx])
        oeDiscRound = np.zeros_like(oeTimes.magnitude)
        nspDiscRound = np.zeros_like(nspTimes.magnitude)
        for j, discIdx in enumerate(listDiscontinuities):
            oeDiscRound[discIdx+1:] += 1
            if nMissingTriggers > 0:
                nspDiscRound[discIdx+1+j] = 999  # use 999 as a discard marker
                nMissingTriggers -= 1
                print('Skipping NSP pulse at t={:.3f}'.format(nspTimes[discIdx+1+j]))
                nspDiscRound[discIdx+2+j:] += 1
            else:
                nspDiscRound[discIdx+1:] += 1
        if np.sum(nspDiscRound < 999) > np.sum(oeDiscRound < 999):
            # if there are more nsp triggers at the end, discard
            nspDiscRound[np.sum(oeDiscRound < 999):] = 999
        if np.sum(oeDiscRound < 999) > np.sum(nspDiscRound < 999):
            # if there are more nsp triggers at the end, discard
            oeDiscRound[np.sum(nspDiscRound < 999):] = 999
        pwSyncDict = {}  # piecewise sync parameters
        uniqueOeRounds = np.unique(oeDiscRound[oeDiscRound < 999])
        for roundIdx in uniqueOeRounds:
            try:
                thesePolyCoeffs = np.polyfit(
                    x=oeTimes[oeDiscRound == roundIdx],
                    y=nspTimes[nspDiscRound == roundIdx], deg=1)
            except Exception:
                traceback.print_exc()
                pdb.set_trace()
            thisInterpFun = np.poly1d(thesePolyCoeffs)
            if roundIdx == 0:
                pwSyncDict[roundIdx] = {
                    'inStart': 0,
                    'inStop': np.max(oeTimes[oeDiscRound == roundIdx].magnitude),
                    'tInterpFun': thisInterpFun}
            elif roundIdx == uniqueOeRounds[-1]:
                pwSyncDict[roundIdx] = {
                    'inStart': np.max(oeTimes[oeDiscRound == roundIdx-1].magnitude),
                    'inStop': 1e6,
                    'tInterpFun': thisInterpFun}
            else:
                pwSyncDict[roundIdx] = {
                    'inStart': np.max(oeTimes[oeDiscRound == roundIdx-1].magnitude),
                    'inStop': np.max(oeTimes[oeDiscRound == roundIdx].magnitude),
                    'tInterpFun': thisInterpFun}
        #
        def timeInterpFun(inputT):
            outputT = np.zeros_like(inputT)
            for k in sorted(pwSyncDict.keys()):
                inTimeMask = (
                    (inputT >= pwSyncDict[k]['inStart']) &
                    (inputT < pwSyncDict[k]['inStop']))
                outputT[inTimeMask] = pwSyncDict[k]['tInterpFun'](
                    inputT[inTimeMask])
            plotting = False
            if plotting:
                import matplotlib.pyplot as plt
                for k in sorted(pwSyncDict.keys()):
                    inTimeMask = (
                        (inputT >= pwSyncDict[k]['inStart']) &
                        (inputT < pwSyncDict[k]['inStop']))
                    plt.plot(inputT[inTimeMask], outputT[inTimeMask])
                plt.show()
            return outputT
    else:
        # assert np.max(np.abs((np.diff(oeTimes) - np.diff(nspTimes)))) < 1e-4
        # np.flatnonzero(np.abs(np.diff(oeTimes)) > 0.018)
        # np.sum(np.abs(np.diff(nspTimes)) < 0.017)
        # nspSynchDur = nspTimes[-1] - nspTimes[0]
        # oeSynchDur = oeTimes[-1] - oeTimes[0]
        ###########
        if oeTimes.size > nspTimes.size:
            # if there are more nsp triggers at the end, discard
            oeTimes = oeTimes[:nspTimes.size]
        if nspTimes.size > oeTimes.size:
            # if there are more nsp triggers at the end, discard
            nspTimes = nspTimes[:oeTimes.size]
        synchPolyCoeffs = np.polyfit(x=oeTimes, y=nspTimes, deg=1)
        # synchPolyCoeffs = np.array([1, np.mean(nspTimes - oeTimes)])
        # synchPolyCoeffs = np.array([1, np.mean(nspTimes[0] - oeTimes[0])])
        timeInterpFun = np.poly1d(synchPolyCoeffs)

    # account for delay because of
    # analog filters on one or both recording devices (units of pq.s)
    filterDelay = 2.3 * 1e-3  # Trigno analog input, filtered DC-100Hz
    # filterDelay = 2.3 * 1e-4  # Trigno analog input, filtered DC-1000Hz
    # TODO: figure out why the block below doesn't work
    # for event in oeBlock.filter(objects=Event):
    #     event.magnitude[:] = (
    #         timeInterpFun(event.times.magnitude) + filterDelay)
    # for asig in oeBlock.filter(objects=AnalogSignal):
    #     asig.times.magnitude[:] = (
    #         timeInterpFun(asig.times.magnitude) + filterDelay)
    # for st in oeBlock.filter(objects=SpikeTrain):
    #     st.times.magnitude[:] = (
    #         timeInterpFun(st.times.magnitude) + filterDelay)

    oeDF = ns5.analogSignalsToDataFrame(
        oeBlock.filter(objects=AnalogSignal),
        idxT='oeT', useChanNames=True)
    oeDF['nspT'] = timeInterpFun(oeDF['oeT']) + filterDelay
    dummyAsig = nspSeg.filter(objects=AnalogSignal)[0]
    newT = pd.Series(dummyAsig.times.magnitude)
    interpCols = oeDF.columns.drop(['oeT', 'nspT'])
    #
    if arguments['chanQuery'] is not None:
        if arguments['chanQuery'] in namedQueries['chan']:
            chanQuery = namedQueries['chan'][arguments['chanQuery']]
        else:
            chanQuery = arguments['chanQuery']
        chanQuery = chanQuery.replace('chanName', 'interpCols')
        interpCols = interpCols[eval(chanQuery)]
    dummyOeAsig = oeBlock.filter(objects=AnalogSignal)[0]
    # pdb.set_trace()
    listOfCols = list(interpCols)
    oeBlock = ns5.dataFrameToAnalogSignals(
        oeDF.loc[:, ['nspT'] + listOfCols],
        idxT='nspT',
        probeName='openEphys', samplingRate=dummyOeAsig.sampling_rate,
        dataCol=listOfCols, forceColNames=listOfCols, verbose=True)
    outputFilePath = os.path.join(
        scratchFolder, ns5FileName + '_delsys_synchronized.nix'
        )
    writer = ns5.NixIO(
        filename=outputFilePath, mode='ow')
    writer.write_block(oeBlock, use_obj_names=True)
    writer.close()
    '''
    oeInterp = hf.interpolateDF(
        oeDF, newT,
        kind='pchip', fill_value=(0, 0),
        x='nspT', columns=interpCols, verbose=True)
    oeInterpBlock = ns5.dataFrameToAnalogSignals(
        oeInterp,
        idxT='nspT',
        probeName='openEphys', samplingRate=dummyAsig.sampling_rate,
        dataCol=interpCols, forceColNames=interpCols, verbose=True)
    #
    ns5.addBlockToNIX(
        oeInterpBlock, neoSegIdx=[0],
        writeAsigs=True, writeSpikes=False, writeEvents=False,
        purgeNixNames=True,
        fileName=ns5FileName,
        folderPath=scratchFolder,
        nixBlockIdx=0, nixSegIdx=[0],
        )
    '''
    return


if __name__ == "__main__":
    runProfiler = True
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=synchronizeDelsysToNSP,
            modulesToProfile=[hf, ash, ns5],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        synchronizeDelsysToNSP()