"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --saveResults                          load from raw, or regular? [default: False]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --outputBlockName=outputBlockName      which trig_ block to pull [default: pca_clean]
    --verbose                              print diagnostics? [default: False]
    --plotting                             plot results?
    --window=window                        process with short window? [default: long]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: all]
    --alignQuery=alignQuery                query what the units will be aligned to? [default: all]
    --selector=selector                    filename if using a unit selector
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
"""

import pdb, traceback
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.preproc.ns5 as ns5
from scipy.stats import zscore, chi2
from scipy import signal
import pingouin as pg
from tqdm import tqdm
import pandas as pd
import numpy as np
import quantities as pq
from copy import deepcopy
from dask.distributed import Client
from docopt import docopt
from tqdm import tqdm
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.utils.random import sample_without_replacement as swr
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['plotting']:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    figureOutputFolder = os.path.join(
        figureFolder, arguments['analysisName'])
    if not os.path.exists(figureOutputFolder):
        os.makedirs(figureOutputFolder, exist_ok=True)

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)

calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName

triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
triggeredH5Path = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}.h5'.format(
        arguments['inputBlockName'], arguments['window']))
outputPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['outputBlockName'], arguments['window']))
outputH5Path = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}.h5'.format(
        arguments['outputBlockName'], arguments['window']))

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False, removeFuzzyName=False,
    decimate=1, windowSize=(-150e-3, 400e-3),
    procFun=None,
    metaDataToCategories=False,
    transposeToColumns='bin', concatOn='index',
    verbose=False))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, alignSubFolder, **arguments)


def cleanISIData(
        group, dataColNames=None,
        fillInFastSettle=True,
        refWinMask=None, tau=None,
        plottingLineNoiseMatch=False,
        artifactDiagnosticPlots=False,
        removeReferenceBaseline=True,
        nExtraSteps=1, interpMethod='pchip',
        remove60Hz=True, verbose=False):
    try:
        featureName = group['feature'].unique()[0]
        electrodeName = group['electrode'].unique()[0]
        firstPW = group['firstPW'].unique()[0]
        secondPW = group['secondPW'].unique()[0]
        totalPW = group['totalPW'].unique()[0]
        stimCat = group['stimCat'].unique()[0]
        #
        dataColMask = group.columns.isin(dataColNames)
        procDF = group.copy()
        groupData = procDF.loc[:, dataColMask]
        if plottingLineNoiseMatch or artifactDiagnosticPlots:
            plotTrialIdx = 1
            groupMean = groupData.iloc[plotTrialIdx, :].mean()
            saveOriginalForPlotting = groupData.iloc[plotTrialIdx, :].copy()
        if remove60Hz and ('EmgEnv' not in featureName):
            if plottingLineNoiseMatch:
                # group.iloc[plotTrialIdx, :] = 13.64 * referenceSines[0][17: 17 + group.shape[1]]
                fig, (axBA, axM) = plt.subplots(2, 1, sharex=True)
                fig2, axXCorr = plt.subplots(2, 1)
                cPalette = sns.color_palette(n_colors=numKernels)
                axBA.plot(
                    groupData.columns,
                    saveOriginalForPlotting - groupMean,
                    label='Original')
            for hOrder, kernel in enumerate(noiseKernels):
                hPeriod = ((hOrder + 1) * lineNoiseFilterOpts['Wn']) ** -1 # seconds
                hPeriodSamples = int(samplingRate * hPeriod)
                #
                def corrWithKern(x, ker):
                    y = signal.correlate(
                        x, kernel, mode='valid')
                    return y
                #
                # cross corelate the ref window with the kernel
                signalXCorr = np.apply_along_axis(
                    corrWithKern, 1,
                    groupData.loc[:, refWinMask], kernel)
                # find the max lag in the xcorr, corresponding to the phase of the noise
                maxLagSamples = np.apply_along_axis(np.argmax, 1, (signalXCorr[:, :hPeriodSamples]))
                noiseModel = pd.DataFrame(np.nan, index=groupData.index, columns=groupData.columns)
                if plottingLineNoiseMatch:
                    axXCorr[0].plot(
                        signalXCorr[plotTrialIdx, :],
                        label='cross correlogram')
                    # axXCorr[1].plot(
                    #     (saveOriginalForPlotting.iloc[refWinMask]).to_numpy(),
                    #     label='Original')
                    # axXCorr[1].plot(
                    #     referenceSines[hOrder],
                    #     label='lookup waveform')
                    # axXCorr[1].plot(
                    #     kernel,
                    #     label='kernel')
                    axXCorr[0].set_title('maxLag == {}'.format(maxLagSamples[plotTrialIdx]))
                    # axXCorr[1].legend()
                for rowIdx, maxLag in enumerate(maxLagSamples):
                    tweakLag = maxLag
                    noiseAmp = signalXCorr[rowIdx, maxLag]
                    noiseModel.iloc[rowIdx, :] = (noiseAmp * referenceSines[hOrder][hPeriodSamples - tweakLag: hPeriodSamples - tweakLag + groupData.shape[1]])
                # print('Noise Amp Max {}'.format(noiseModel.iloc[plotTrialIdx, :].max()))
                # print('orig max / noise max {}'.format(groupData.iloc[plotTrialIdx, :].max() / noiseModel.iloc[plotTrialIdx, :].max()))
                if plottingLineNoiseMatch:
                    axM.plot(
                        noiseModel.columns, noiseModel.iloc[plotTrialIdx, :], '--',
                        label='noise model order {}'.format(hOrder + 1),
                        c=cPalette[hOrder])
                    axM.plot(
                        groupData.columns, saveOriginalForPlotting - groupMean, '-',
                        label='before match order {}'.format(hOrder + 1), lw=2,
                        c=cPalette[hOrder])
                procDF.loc[:, dataColMask] = procDF.loc[:, dataColMask] - noiseModel
                if plottingLineNoiseMatch:
                    axBA.plot(
                        procDF.iloc[:, dataColMask].columns,
                        procDF.iloc[plotTrialIdx, dataColMask] - groupMean,
                        label='after match order {}'.format(hOrder + 1))
                    axBA.legend()
                    axBA.set_title(
                        '{} during stim on {}'.format(featureName, electrodeName))
                    axBA.set_xlabel('Time (sec)')
                    axM.legend()
                    axM.set_xlabel('Time (sec)')
            if plottingLineNoiseMatch and not (artifactDiagnosticPlots):
                plt.show()
        if removeReferenceBaseline:
            procDF.loc[:, dataColMask] = (
                procDF.loc[:, dataColMask].apply(
                    lambda x: x - np.median(x[refWinMask]),
                    axis=1, raw=True,
                    ))
        if (
                fillInFastSettle and
                (
                    ('caudal' in featureName) or
                    ('rostral' in featureName))):
            fastSettleModeList = group['fastSettleTriggers'].unique()
            # if not (fastSettleModeList.size == 1):
            #     pdb.set_trace()
            assert fastSettleModeList.size == 1
            fastSettleMode = fastSettleModeList[0]
            correctSwitch = True
            timeStep = (3e4 ** (-1))
            blankingDur = totalPW
            if fastSettleMode == 'same':
                if (
                        (
                            ('caudal' in featureName) &
                            ('caudal' in electrodeName)) |
                        (
                            ('rostral' in featureName) &
                            ('rostral' in electrodeName))):
                    pass
                    # correctSwitch = False
                    blankingDur += nExtraSteps * timeStep
            if fastSettleMode == 'any':
                pass
                # correctSwitch = False
                blankingDur += nExtraSteps * timeStep
            #
            # print('{} during stim on {}; Cleaning {:.2f} usec'.format(
            #     featureName, electrodeName, blankingDur * 1e6))
            t = groupData.columns
            if stimCat == 'stimOn':
                blankMask = (t >= 0) & (t <= 0 + blankingDur)
            elif stimCat == 'stimOff':
                blankMask = (t >= - blankingDur) & (t <= 0)
            # remove saturated readings
            # satLimit = 6e3
            #
            lastBlankIdx = np.flatnonzero(blankMask)[-1]
            # nextIdx = min(lastBlankIdx+1, blankMask.size)
            # while (groupData.iloc[:, nextIdx].abs() > satLimit).any():
            #     lastBlankIdx += 1
            #     nextIdx = min(lastBlankIdx+1, blankMask.size)
            #     if nextIdx == blankMask.size:
            #         break
            # blankMask[np.flatnonzero(blankMask)[-1]:lastBlankIdx+1] = True
            # # firstBlankIdx = np.flatnonzero(blankMask)[0]
            # lastBlankIdx = np.flatnonzero(blankMask)[-1]
            if correctSwitch:
                deltaV = (groupData.iloc[:, lastBlankIdx+1])
                artifactDF = calcStimArtifact(
                    groupData, t, t[lastBlankIdx+1],
                    deltaV, tau)
                blankMask[lastBlankIdx+1] = True
            if artifactDiagnosticPlots:
                fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
                ax[0].plot(
                    t, saveOriginalForPlotting, '.-',
                    label='original')
                if correctSwitch:
                    ax[0].plot(
                        t, artifactDF.iloc[plotTrialIdx, :],
                        label='discharge')
                ax[0].plot(
                    t[blankMask], saveOriginalForPlotting[blankMask],
                    label='blank period')
            #
            if correctSwitch:
                procDF.loc[:, dataColMask] = (
                    procDF.loc[:, dataColMask] - artifactDF)
            maskToNan = dataColMask.copy()
            maskToNan[dataColMask] = blankMask
            procDF.loc[:, maskToNan] = np.nan
            if interpMethod in ['ffill', 'bfill']:
                procDF.loc[:, dataColMask] = (
                    procDF.loc[:, dataColMask]
                    .fillna(method=interpMethod, axis=1))
            else:
                procDF.loc[:, dataColMask] = (
                    procDF.loc[:, dataColMask]
                    .interpolate(method=interpMethod, axis=1))
            if artifactDiagnosticPlots:
                plotFixed = procDF.iloc[plotTrialIdx, dataColMask]
                ax[1].plot(t, plotFixed)
                ax[0].legend()
                ax[0].set_title('{} during stim on {} (pid {})'.format(
                    featureName, electrodeName, os.getpid()))
                plt.show()
    except Exception:
        traceback.print_exc()
        pdb.set_trace()
    if verbose:
        print('\nHello! This is cleanISI at process {}'.format(os.getpid()))
    return procDF


if __name__ == "__main__":
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    useCachedInput = False
    if useCachedInput:
        dataDF = pd.read_hdf(triggeredH5Path, arguments['inputBlockName'])
        samplingRate = float((dataDF.columns[1] - dataDF.columns[0]) ** -1)
    else:
        print('loading {}'.format(triggeredPath))
        dataDF = ns5.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
        dataDF.columns = dataDF.columns.astype(np.float)
        dataDF.to_hdf(triggeredH5Path, arguments['inputBlockName'])
        samplingRate = float(
            dataBlock
            .filter(objects=ns5.SpikeTrain)[0]
            .sampling_rate
            .magnitude)  # Hz
    trialInfo = dataDF.index.to_frame()
    trialInfo['blockID'] = trialInfo['segment'].map(assembledSegmentToBlockLookup)

    assert 'rippleFastSettleTriggers' in locals()
    stimFastSettle = {k: v['stim'] for k, v in rippleFastSettleTriggers.items()}
    trialInfo['fastSettleTriggers'] = trialInfo['blockID'].map(stimFastSettle)
    for extraMetaDataName in ['blockID', 'fastSettleTriggers']:
        dataDF.set_index(
                pd.Index(trialInfo[extraMetaDataName], name=extraMetaDataName),
                append=True, inplace=True)
    # procDF = dataDF.copy()

    groupBy = [
        'electrode', 'nominalCurrent',
        'stimCat', 'feature',
        'totalPW',
        'firstPW', 'secondPW'
        ]

    def calcStimArtifact(DF, t, tOffset, vMag, tau):
        artDF = pd.DataFrame(0., index=DF.index, columns=DF.columns)
        tMask = (t - tOffset) >= 0
        for rowIdx in artDF.index:
            artDF.loc[rowIdx, tMask] = (
                vMag[rowIdx] * np.exp(-(t[tMask] - tOffset) / tau))
        return artDF

    # line noise match filter
    lineNoiseFilterOpts = {
        'type': 'sin',
        'Wn': 60,
        'nHarmonics': 2,
        'timeWindow': [-150e-3, -1e-3]  # split the epoch into a reference window (pre-stimulus) and the rest
        }
    refWinMask = (dataDF.columns >= lineNoiseFilterOpts['timeWindow'][0]) & (dataDF.columns < lineNoiseFilterOpts['timeWindow'][-1])
    matchType = lineNoiseFilterOpts['type']
    matchFreq = lineNoiseFilterOpts['Wn']
    # how many 60Hz cycles fill the reference window
    nCycles = np.floor(
        matchFreq * (
            lineNoiseFilterOpts['timeWindow'][-1] -
            lineNoiseFilterOpts['timeWindow'][0])) - 1
    nMatchSamples = np.ceil(nCycles * samplingRate / matchFreq).astype(np.int)
    noiseKernels = []
    referenceSines = []
    plottingLineNoiseMatch = False
    artifactDiagnosticPlots = False
    if plottingLineNoiseMatch:
        fig, axK = plt.subplots()

    if matchType == 'sin':
        nMatchHarmonics = lineNoiseFilterOpts['nHarmonics']
        for harmonicOrder in range(1, nMatchHarmonics + 1):
            hPeriod = (harmonicOrder * matchFreq) ** -1
            kernX = 2 * np.pi * np.arange(dataDF.columns[0], dataDF.columns[-1] + 3 * hPeriod, samplingRate ** -1) / hPeriod  # seconds
            # create a list of waveforms, slightly larger than the full window, to lookup, scale and subtract, later
            referenceWaveform = np.sin(kernX)
            referenceWaveform = referenceWaveform / referenceWaveform.max()
            referenceSines.append(referenceWaveform)
            kernel = referenceWaveform[:nMatchSamples]
            # 
            kernel = kernel / np.sum(kernel ** 2)
            noiseKernels.append(kernel)
            if plottingLineNoiseMatch:
                tK = dataDF.columns[:nMatchSamples]
                axK.plot(
                    tK, kernel,
                    label='model order {}'.format(harmonicOrder))
        numKernels = len(noiseKernels)
    else:
        numKernels = 1

    if plottingLineNoiseMatch:
        axK.set_title('matched filter kernels')
        axK.set_xlabel('Time (sec)')
        plt.show()

    cleanOpts = dict(
        refWinMask=refWinMask,
        fillInFastSettle=True,
        tau=30e-6, nExtraSteps=4,
        remove60Hz=True, removeReferenceBaseline=False,
        interpMethod='ffill',
        plottingLineNoiseMatch=plottingLineNoiseMatch,
        artifactDiagnosticPlots=artifactDiagnosticPlots,)
    # daskClient = Client()
    # daskComputeOpts = {}
    daskComputeOpts = dict(
        scheduler='processes'
        # scheduler='single-threaded'
        )
    procDF = ash.splitApplyCombine(
        dataDF, fun=cleanISIData, resultPath=outputPath,
        funArgs=[], funKWArgs=cleanOpts,
        rowKeys=groupBy, colKeys=None, useDask=True,
        daskPersist=False, daskProgBar=True,
        daskComputeOpts=daskComputeOpts,
        reindexFromInput=True)

    # procDF.columns.name = 'bin'
    '''
        for name, group in tqdm(dataDF.groupby(groupBy)):
        try:
            if remove60Hz and ('EmgEnv' not in name[2]):
                if plottingLineNoiseMatch:
                    plotTrialIdx = 1
                    # group.iloc[plotTrialIdx, :] = 13.64 * referenceSines[0][17: 17 + group.shape[1]]
                    fig, (axBA, axM) = plt.subplots(2, 1, sharex=True)
                    fig2, axXCorr = plt.subplots(2, 1)
                    cPalette = sns.color_palette(n_colors=numKernels)
                    groupMean = group.iloc[plotTrialIdx, :].mean()
                    axBA.plot(
                        group.columns, group.iloc[plotTrialIdx, :] - groupMean,
                        label='Original')
                for hOrder, kernel in enumerate(noiseKernels):
                    hPeriod = ((hOrder + 1) * lineNoiseFilterOpts['Wn']) ** -1 # seconds
                    hPeriodSamples = int(samplingRate * hPeriod)
                    #
                    def corrWithKern(x, ker):
                        y = signal.correlate(
                            x, kernel, mode='valid')
                        return y
                    #
                    # cross corelate the ref window with the kernel
                    signalXCorr = np.apply_along_axis(
                        corrWithKern, 1,
                        group.loc[:, refWinMask], kernel)
                    # find the max lag in the xcorr, corresponding to the phase of the noise
                    maxLagSamples = np.apply_along_axis(np.argmax, 1, (signalXCorr[:, :hPeriodSamples]))
                    noiseModel = pd.DataFrame(np.nan, index=group.index, columns=group.columns)
                    if plottingLineNoiseMatch:
                        axXCorr[0].plot(
                            signalXCorr[plotTrialIdx, :],
                            label='cross correlogram')
                        axXCorr[1].plot(
                            (group.iloc[plotTrialIdx, refWinMask] - groupMean).to_numpy(),
                            label='Original')
                        axXCorr[1].plot(
                            referenceSines[hOrder],
                            label='lookup waveform')
                        axXCorr[1].plot(
                            kernel,
                            label='kernel')
                        axXCorr[0].set_title('maxLag == {}'.format(maxLagSamples[plotTrialIdx]))
                        axXCorr[1].legend()
                    for rowIdx, maxLag in enumerate(maxLagSamples):
                        tweakLag = maxLag
                        noiseAmp = signalXCorr[rowIdx, maxLag]
                        noiseModel.iloc[rowIdx, :] = (noiseAmp * referenceSines[hOrder][hPeriodSamples - tweakLag: hPeriodSamples - tweakLag + group.shape[1]])
                    # print('Noise Amp Max {}'.format(noiseModel.iloc[plotTrialIdx, :].max()))
                    # print('orig max / noise max {}'.format(group.iloc[plotTrialIdx, :].max() / noiseModel.iloc[plotTrialIdx, :].max()))
                    if plottingLineNoiseMatch:
                        axM.plot(
                            noiseModel.columns, noiseModel.iloc[plotTrialIdx, :], '--',
                            label='noise model order {}'.format(hOrder + 1),
                            c=cPalette[hOrder])
                        axM.plot(
                            group.columns, group.iloc[plotTrialIdx, :] - groupMean, '-',
                            label='before match order {}'.format(hOrder + 1), lw=2,
                            c=cPalette[hOrder])
                    procDF.loc[group.index, :] = group - noiseModel
                    if plottingLineNoiseMatch:
                        axBA.plot(
                            procDF.loc[group.index, :].columns,
                            procDF.loc[group.index, :].iloc[plotTrialIdx, :] - groupMean,
                            label='after match order {}'.format(hOrder + 1))
                        axBA.legend()
                        axBA.set_title('{} during stim on {}'.format(name[2], name[0]))
                        axBA.set_xlabel('Time (sec)')
                        axM.legend()
                        axM.set_xlabel('Time (sec)')
                if plottingLineNoiseMatch:
                    plt.show()
                    # electrodeName_trace()
            if fillInFastSettle and (('caudal' in name[2]) or ('rostral' in name[2])):
                blankingDur = (name[3] + name[4]) * 1e-6 + 6 * 33.3 * 1e-6
                t = group.columns
                if name[5] == 'stimOn':
                    blankMask = (t >= 0) & (t <= 0 + blankingDur)
                # remove saturated readings
                satLimit = 6e3
                #
                lastBlankIdx = np.flatnonzero(blankMask)[-1]
                nextIdx = min(lastBlankIdx+1, blankMask.size)
                while (group.iloc[:, nextIdx].abs() > satLimit).any():
                    lastBlankIdx += 1
                    nextIdx = min(lastBlankIdx+1, blankMask.size)
                    if nextIdx == blankMask.size:
                        break
                blankMask[np.flatnonzero(blankMask)[-1]:lastBlankIdx+1] = True
                firstBlankIdx = np.flatnonzero(blankMask)[0]
                lastBlankIdx = np.flatnonzero(blankMask)[-1]
                fastSettleModeList = (
                    trialInfo
                    .loc[group.index, 'fastSettleTriggers']
                    .unique())
                assert fastSettleModeList.size == 1
                fastSettleMode = fastSettleModeList[0]
                correctSwitch = False
                if fastSettleMode == 'same':
                    if (
                            (
                                ('caudal' in name[2]) &
                                ('caudal' in name[0])) |
                            (
                                ('rostral' in name[2]) &
                                ('rostral' in name[0]))):
                        correctSwitch = True
                if fastSettleMode == 'any':
                    correctSwitch = True
                if correctSwitch:
                    deltaV = (group.iloc[:, lastBlankIdx+1])
                    artifactDF = calcStimArtifact(group, t, t[lastBlankIdx+1], deltaV, tau)
                    blankMask[lastBlankIdx+1] = True
                artifactDiagnosticPlots = False
                if artifactDiagnosticPlots:
                    plotTrialIdx = 1
                    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
                    ax[0].plot(t, group.iloc[plotTrialIdx, :], '.-')
                    ax[0].plot(t[blankMask], group.iloc[plotTrialIdx, :][blankMask])
                    if correctSwitch:
                        ax[0].plot(t, artifactDF.iloc[plotTrialIdx, :])
                #
                if correctSwitch:
                    procDF.loc[group.index, :] = procDF.loc[group.index, :] - artifactDF
                procDF.loc[group.index, blankMask] = np.nan
                #
                if artifactDiagnosticPlots:
                    plotFixed = procDF.loc[group.index, :].iloc[plotTrialIdx, :]
                    ax[1].plot(t, plotFixed)
                    ax[0].set_title('{} during stim on {}'.format(name[2], name[0]))
                    plt.show()
        except Exception:
            traceback.print_exc()
        electrodeName_trace()
    '''
    del dataDF
    #
    masterBlock = ns5.alignedAsigDFtoSpikeTrain(procDF, dataBlock)
    if arguments['lazy']:
        dataReader.file.close()
    masterBlock = ns5.purgeNixAnn(masterBlock)

    if os.path.exists(outputPath):
        os.remove(outputPath)

    writer = ns5.NixIO(filename=outputPath)
    writer.write_block(masterBlock, use_obj_names=True)
    writer.close()

    if os.path.exists(outputH5Path):
        os.remove(outputH5Path)
    procDF.to_hdf(outputH5Path, arguments['outputBlockName'])