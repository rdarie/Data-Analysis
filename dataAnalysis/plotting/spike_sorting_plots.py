import dataAnalysis.helperFunctions.helper_functions_new as hf
import os
import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import numpy as np
import pandas as pd
import pdb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import seaborn as sns
import random
import traceback
from math import gcd
from tqdm import tqdm
sns.set(rc={
    'figure.figsize': (14, 18),
    'legend.fontsize': 10,
    'legend.handlelength': 2
    })


def plotSpikePanel(
        spikeStruct, spikes, labelFontSize=5,
        padOverride=1e-3, figSize=(18, 18),
        hideUnused=True, useLGACoords=False,
        coordsToIndicesOpts=None,
        colorPal="ch:2,-.1,dark=.2,light=0.8,reverse=1"):
    sns.set_style("dark", {"axes.facecolor": ".9"})
    if coordsToIndicesOpts is None:
        coordsToIndicesOpts = {
            'reverseY': True
        }
    if useLGACoords:
        spikeStruct.loc[:, 'xcoords'] = spikeStruct.loc[:, 'lgaXCoords']
        spikeStruct.loc[:, 'ycoords'] = spikeStruct.loc[:, 'lgaYCoords']
    if hideUnused:
        spikeStruct = spikeStruct.loc[
            spikeStruct['label'].isin(spikes['ChannelID']), :]
    spikeStruct.loc[:, 'panelX'], spikeStruct.loc[:, 'panelY'] = coordsToIndices(
        spikeStruct['xcoords'], spikeStruct['ycoords'],
        **coordsToIndicesOpts
        )
    #
    fig = plt.figure()
    nCols = int(max(spikeStruct['panelX'].unique()) + 1)
    nRows = int(max(spikeStruct['panelY'].unique()) + 1)
    gs = fig.add_gridspec(
        ncols=nCols, nrows=nRows)
    axDF = pd.DataFrame(index=range(nRows), columns=range(nCols))
    for axY in range(nRows):  # rows are y
        for axX in range(nCols):  # cols are x
            axDF.loc[axY, axX] = fig.add_subplot(gs[axY, axX])
            axDF.loc[axY, axX].channelContainsSpikes = False
            containsThisPosition = (
                (spikeStruct['panelX'] == axX) &
                (spikeStruct['panelY'] == axY))
            axDF.loc[axY, axX].channelIsMapped = containsThisPosition.any()
            if axDF.loc[axY, axX].channelIsMapped:
                thisRow = spikeStruct.loc[containsThisPosition, :]
                assert thisRow.shape[0] == 1
                axDF.loc[axY, axX].bankInfo = '{}{:>02d}'.format(
                    thisRow['bank'].iloc[0],
                    int(thisRow['bankID'].iloc[0]),
                    )
            else:
                axDF.loc[axY, axX].bankInfo = ''

    ax = axDF.to_numpy()
    fig.set_size_inches(figSize)
    axHighLims = np.empty(ax.shape)
    axHighLims[:] = np.nan
    axLowLims = np.empty(ax.shape)
    axLowLims[:] = np.nan
    with sns.color_palette(colorPal, 8):
        for idx, channel in enumerate(spikes['ChannelID']):
            thisRow = spikeStruct.loc[spikeStruct['label'] == channel, :]
            assert thisRow.shape[0] == 1
            curAx = ax[thisRow['panelY'].iloc[0], thisRow['panelX'].iloc[0]]
            # curAx.set_aspect('equal', adjustable='datalim')
            plotSpike(spikes, channel, ax=curAx)
            curAxLim = curAx.get_ylim()
            axHighLims[thisRow['panelY'].iloc[0], thisRow['panelX'].iloc[0]] = curAxLim[1]
            axLowLims[thisRow['panelY'].iloc[0], thisRow['panelX'].iloc[0]] = curAxLim[0]
            curAx.channelContainsSpikes = True
        # get xLim from last axis that has spikes, in order to make the label
        xLim = ax[thisRow['panelY'].iloc[0], thisRow['panelX'].iloc[0]].get_xlim()
        sns.despine()
        #
        newAxMin = np.nanmean(axLowLims) - .5 * np.nanstd(axLowLims)
        newAxMax = np.nanmean(axHighLims) + .5 * np.nanstd(axHighLims)
        #
        for idx, channel in enumerate(spikes['ChannelID']):
            thisRow = spikeStruct.loc[spikeStruct['label'] == channel, :]
            assert thisRow.shape[0] == 1
            curAx = ax[thisRow['panelY'].iloc[0], thisRow['panelX'].iloc[0]]
            curAx.set_ylim(newAxMin, newAxMax)
            trnsf = curAx.transAxes
            textOpts = {
                'verticalalignment': 'top',
                'horizontalalignment': 'right',
                'fontsize': labelFontSize
                }
            curAx.text(
                1, 1, channel,
                transform=trnsf, **textOpts)
            ax[thisRow['panelY'].iloc[0], thisRow['panelX'].iloc[0]].channelName = channel
        #
        for idx, curAx in enumerate(ax.flatten()):
            if idx != 0:
                curAx.tick_params(
                    left=False, top=False, right=False, bottom=False,
                    labelleft=False, labeltop=False, labelright=False,
                    pad=-3 * labelFontSize,
                    labelbottom=False, labelsize=labelFontSize,
                    direction='in',
                    length=labelFontSize)
            else:
                curAx.tick_params(
                    left=True, top=False, right=False, bottom=True,
                    labelleft=True, labeltop=False, labelright=False,
                    pad=-3 * labelFontSize,
                    labelbottom=True, direction='in',
                    labelsize=labelFontSize, length=labelFontSize)
                curAx.set_ylim(newAxMin, newAxMax)
                curAx.set_xlim(*xLim)
                curAx.set_xlabel(
                    'msec', fontsize=labelFontSize,
                    labelpad=-3 * labelFontSize)
                curAx.set_ylabel(
                    spikes['Units'], fontsize=labelFontSize,
                    labelpad=-3 * labelFontSize)
        plt.tight_layout(
            pad=padOverride, w_pad=padOverride,
            h_pad=padOverride)
    return fig, ax


def coordsToIndices(
        xcoords, ycoords,
        swapXY=False,
        reverseX=False, reverseY=False):
    # ordMag = np.floor(np.log10(xcoords.abs().mean().mean()))
    xcoordsInt = (xcoords - xcoords.min()).astype(np.int)
    ycoordsInt = (ycoords - ycoords.min()).astype(np.int)
    #
    # xcoordsInt = xcoords.astype(np.int) - xcoords.astype(np.int).min()
    # ycoordsInt = ycoords.astype(np.int) - ycoords.astype(np.int).min()
    #
    xSpacing = np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), xcoordsInt)
    ySpacing = np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), ycoordsInt)
    #
    xIdx = np.array(
        np.divide(xcoords.astype(np.int), xSpacing),
        dtype=np.int)
    yIdx = np.array(
        np.divide(ycoords.astype(np.int), ySpacing),
        dtype=np.int)
    #
    if reverseX:
        xIdx = np.max(xIdx) - xIdx
    if reverseY:
        yIdx = np.max(yIdx) - yIdx
    if swapXY:
        xIdx, yIdx = yIdx, xIdx
    return xIdx, yIdx


def plotSpike(
        spikes, channel, showNow=False, ax=None,
        acrossArray=False, xcoords=None, ycoords=None,
        axesLabel=False, errorMultiplier=1, ignoreUnits=[],
        channelPlottingName=None, chanNameInLegend=False, legendTags=[]):

    if channelPlottingName is None:
        channelPlottingName = str(channel)

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = pd.unique(spikes['Classification'][ChanIdx])
    if 'ClassificationLabel' in spikes.keys():
        unitsLabelsOnThisChan = pd.unique(spikes['ClassificationLabel'][ChanIdx])
    else:
        unitsLabelsOnThisChan = None

    if acrossArray:
        # Check that we didn't ask to plot the spikes across channels into a single axis
        assert ax is None
        # Check that we have waveform data everywhere
        assert len(spikes['Waveforms'][ChanIdx].shape) == 3

    if ax is None and not acrossArray:
        fig, ax = plt.subplots()
    if ax is not None and not acrossArray:
        fig = ax.figure

    if unitsOnThisChan is not None:
        if acrossArray:
            xIdx, yIdx = coordsToIndices(xcoords, ycoords)
            fig, ax = plt.subplots(
                nrows=max(np.unique(xIdx)) + 1,
                ncols=max(np.unique(yIdx)) + 1)

        colorPalette = sns.color_palette(n_colors=40)
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            #print('ignoreUnits are {}'.format([-1] + ignoreUnits))
            if unitName in [-1] + ignoreUnits:
                continue

            unitMask = spikes['Classification'][ChanIdx] == unitName

            if 'ClassificationLabel' in spikes.keys():
                unitPlottingName = unitsLabelsOnThisChan[unitIdx]
            else:
                unitPlottingName = unitName

            if chanNameInLegend:
                #  labelName = 'chan %s: unit %s (%d SDs)' % (channelPlottingName, unitPlottingName, errorMultiplier)
                labelName = '{}#{}'.format(channelPlottingName, unitPlottingName)
            else:
                labelName = '#{}'.format(unitPlottingName)

            for legendTag in legendTags:
                if legendTag in spikes['basic_headers']:
                    if unitName in spikes['basic_headers'][legendTag]:
                        labelName += ' {}: {}'.format(
                            legendTag,
                            spikes['basic_headers'][legendTag][unitName]
                        )
                    else:
                        print('{} not found in header!'.format(unitName))
                else:
                    print('{} not found! in legendTags'.format(legendTag))

            if acrossArray:
                for idx, channel in enumerate(spikes['ChannelID']):
                    curAx = ax[xIdx[idx], yIdx[idx]]
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, idx]
                    thisSpike = np.nanmean(waveForms, axis = 0)
                    thisError = np.nanstd(waveForms, axis = 0)
                    #  thisError = stats.sem(waveForms, nan_policy='omit')
                    timeRange = np.arange(len(thisSpike)) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                    curAx.fill_between(timeRange, thisSpike - errorMultiplier*thisError,
                        thisSpike + errorMultiplier*thisError, alpha=0.4,
                        facecolor=colorPalette[unitIdx],
                        label=labelName)
                    curAx.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])

                sns.despine()
                for curAx in ax.flatten():
                    curAx.tick_params(
                        left='off', top='off', right='off',
                        bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
                plt.tight_layout(pad=0.01)

            else:
                if len(spikes['Waveforms'][ChanIdx].shape) == 3:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, ChanIdx]
                else:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :]
                thisSpike = np.nanmean(waveForms, axis=0)
                #  
                thisError = np.nanstd(waveForms, axis=0)
                #  thisError = stats.sem(waveForms, nan_policy='omit')
                timeRange = np.arange(len(thisSpike)) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                colorPalette = sns.color_palette(n_colors=40)

                ax.fill_between(
                    timeRange, thisSpike - errorMultiplier * thisError,
                    thisSpike + errorMultiplier*thisError, alpha=0.4,
                    facecolor=colorPalette[unitIdx],
                    label=labelName)
                ax.plot(timeRange, thisSpike, linewidth=1, color=colorPalette[unitIdx])
                if axesLabel:
                    ax.set_ylabel(spikes['Units'])
                    ax.set_xlabel('Time (msec)')
                    ax.set_title('Units on channel {}'.format(channelPlottingName))
                    ax.legend(loc='lower right')
        if showNow:
            plt.show()
    return fig, ax


def plot_spikes(
        spikes, channel, ignoreUnits=[], showNow=False, ax=None,
        acrossArray=False, xcoords=None, ycoords=None,
        axesLabel=False, channelPlottingName=None, chanNameInLegend=True,
        legendTags=[], lineWidth=1, lineRasterized=False,
        maskSpikes=None, maxSpikes=200, lineAlpha=0.025):
        
    if channelPlottingName is None:
        channelPlottingName = str(channel)

    ChanIdx = spikes['ChannelID'].index(channel)
    unitsOnThisChan = pd.unique(spikes['Classification'][ChanIdx])
    if 'ClassificationLabel' in spikes.keys():
        unitsLabelsOnThisChan = pd.unique(spikes['ClassificationLabel'][ChanIdx])
    else:
        unitsLabelsOnThisChan = None

    if acrossArray:
        # Check that we didn't ask to plot the spikes across channels into a single axis
        assert ax is None
        # Check that we have waveform data everywhere
        assert len(spikes['Waveforms'][ChanIdx].shape) == 3

    if ax is None and not acrossArray:
        fig, ax = plt.subplots()
        fig.set_tight_layout({'pad': 0.01})
    if ax is not None and not acrossArray:
        fig = ax.figure

    if unitsOnThisChan is not None:
        if acrossArray:
            xIdx, yIdx = coordsToIndices(xcoords, ycoords)
            fig, ax = plt.subplots(nrows=max(np.unique(xIdx)) + 1, ncols=max(np.unique(yIdx)) + 1)
            fig.set_tight_layout({'pad': 0.01})
        colorPalette = sns.color_palette(n_colors=40)
        for unitIdx, unitName in enumerate(unitsOnThisChan):
            #  print('ignoreUnits are {}'.format([-1] + ignoreUnits))
            if unitName in [-1] + ignoreUnits:
                #  always ignore units marked -1
                continue
            unitMask = spikes['Classification'][ChanIdx] == unitName

            if 'ClassificationLabel' in spikes.keys():
                unitPlottingName = unitsLabelsOnThisChan[unitIdx]
            else:
                unitPlottingName = unitName
            if chanNameInLegend:
                labelName = 'chan %s, unit %s' % (channelPlottingName, unitPlottingName)
            else:
                labelName = 'unit %s' % (unitPlottingName)
            for legendTag in legendTags:
                if legendTag in spikes['basic_headers']:
                    labelName = '{} {}: {}'.format(
                        labelName,  legendTag,
                        spikes['basic_headers'][legendTag][unitName]
                    )
            #  plot all channels aligned to this spike?
            if acrossArray:
                waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, 0]
                if maxSpikes < waveForms.shape[0]:
                        selectIdx = random.sample(range(waveForms.shape[0]), maxSpikes)
                for idx, channel in enumerate(spikes['ChannelID']):
                    curAx = ax[xIdx[idx], yIdx[idx]]
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, idx]
                    if maskSpikes is not None:
                        waveForms = waveForms[maskSpikes, :]
                    if maxSpikes < waveForms.shape[0]:
                        waveForms = waveForms[selectIdx]
                    timeRange = np.arange(len(waveForms[0])) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                    for spIdx, thisSpike in enumerate(waveForms):
                        thisLabel = labelName if idx == 0 else None
                        curAx.plot(
                            timeRange, thisSpike, label=thisLabel,
                            linewidth=lineWidth, color=colorPalette[unitIdx],
                            alpha=lineAlpha, rasterized=lineRasterized)
                sns.despine()
                for curAx in ax.flatten():
                    curAx.tick_params(
                        left='off', top='off', right='off',
                        bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
            else:
                #  plot only on main chan
                if len(spikes['Waveforms'][ChanIdx].shape) == 3:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :, ChanIdx]
                else:
                    waveForms = spikes['Waveforms'][ChanIdx][unitMask, :]
                
                timeRange = np.arange(len(waveForms[0])) / spikes['basic_headers']['TimeStampResolution'] * 1e3
                
                if maskSpikes is not None:
                    waveForms = waveForms[maskSpikes, :]
                if maxSpikes < waveForms.shape[0]:
                    waveForms = waveForms[random.sample(range(waveForms.shape[0]), maxSpikes), :]
                for spIdx, thisSpike in enumerate(waveForms):
                    thisLabel = labelName if spIdx == 0 else None
                    ax.plot(
                        timeRange, thisSpike,
                        linewidth=lineWidth, color=colorPalette[unitIdx],
                        alpha=lineAlpha, label=thisLabel, rasterized=lineRasterized)
                ax.set_xlim(timeRange[0], timeRange[-1])
                if axesLabel:
                    ax.set_ylabel(spikes['Units'])
                    ax.set_xlabel('Time (msec)')
                    ax.set_title('Units on channel {}'.format(channelPlottingName))
                    ax.legend(loc='lower right')
        if showNow:
            plt.show()
    return fig, ax


def spikePDFReport(
        folderPath, spikes, spikeStruct,
        arrayName=None, arrayInfo=None,
        correctAlignmentSpikes=0,
        plotOpts={'type': 'ticks', 'errorBar': 'sem'},
        trialStats=None, enableFR=False, rasterOpts={'alignTo': None},
        plotSpikePanelOpts=None,
        newReportName=None, colorByAmpBank=False, limitPages=None
        ):
    if plotSpikePanelOpts is None:
        plotSpikePanelOpts = {
            'hideUnused': True,
            'useLGACoords': False,
            'coordsToIndicesOpts': None,
        }
    if correctAlignmentSpikes:  # correctAlignmentSpikes units in samples
        spikes = hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

    if newReportName is None:
        pdfName = os.path.join(folderPath, 'spikePDFReport.pdf')
    else:
        pdfName = os.path.join(folderPath, newReportName + '.pdf')

    #  if any((arrayName is None, arrayInfo is None)):
    #      arrayName, arrayInfo, partialRasterOpts = (
    #          ksa.trialBinnedSpikesNameRetrieve(newReportName))
    #      arrayInfo['nevIDs'] = spikes['ChannelID']

    with PdfPages(pdfName) as pdf:
        spPanelFig, spPanelAx = plotSpikePanel(
            spikeStruct, spikes, colorPal=None,
            labelFontSize=14, padOverride=10e-3,
            **plotSpikePanelOpts
            )
        ######################################################
        #  color axes by amplifier bank
        #######################################################
        if colorByAmpBank:
            bankColorLookup = {
                'A': list(mplcolors.to_rgba('tab:brown')),
                'B': list(mplcolors.to_rgba('tab:pink')),
                'C': list(mplcolors.to_rgba('tab:olive')),
                'D': list(mplcolors.to_rgba('tab:cyan')),
                'noSpikes': list(mplcolors.to_rgba('tab:gray')),
                }
            for curAx in spPanelAx.flatten():
                if hasattr(curAx, 'channelName'):
                    chMapInfo = spikeStruct.loc[spikeStruct['label'] == curAx.channelName, :]
                    chBank = chMapInfo['bank'].iloc[0]
                    chColor = bankColorLookup[chBank]
                    chColor[-1] = 0.3
                    curAx.set_facecolor(chColor)
                if hasattr(curAx, 'channelContainsSpikes'):
                    if not curAx.channelContainsSpikes:
                        chColor = bankColorLookup['noSpikes']
                        chColor[-1] = 0.3
                        curAx.set_facecolor(chColor)
                    else:
                        trnsf = curAx.transAxes
                        textOpts = {
                            'verticalalignment': 'bottom',
                            'horizontalalignment': 'right',
                            'fontsize': 12
                            }
                        curAx.text(
                            1, 0,
                            '{}'.format(curAx.bankInfo),
                            transform=trnsf, **textOpts)
                if hasattr(curAx, 'channelIsMapped'):
                    if not curAx.channelIsMapped:
                        chColor = bankColorLookup['noSpikes']
                        chColor[-1] = 1
                        curAx.set_facecolor(chColor)

        ###########################################################
        #  make spikepanel square
        # figWidth, figHeight = plt.gcf().get_size_inches()
        # aspectRatio = (spikeStruct['xcoords'].max() - spikeStruct['xcoords'].min()) / (spikeStruct['ycoords'].max() - spikeStruct['ycoords'].min())
        # plt.gcf().set_size_inches(figWidth, figWidth)
        pdf.savefig(bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        numChanPlotted = 0
        for channel in tqdm(sorted(spikes['ChannelID'])):
            if limitPages is not None:
                if numChanPlotted > limitPages:
                    continue
            idx = spikes['ChannelID'].index(channel)
            unitsOnThisChan = np.unique(spikes['Classification'][idx])
            if unitsOnThisChan is not None:
                if len(unitsOnThisChan) > 0:
                    #  allocate axes
                    fig = plt.figure(tight_layout={'pad': 0.01})
                    numChanPlotted += 1
                    gs = gridspec.GridSpec(2, 3)
                    try:
                        spikeAx = fig.add_subplot(gs[1, 0])
                        spikesBadAx = fig.add_subplot(gs[1, 1])
                        plotSpike(
                            spikes, channel=channel, ax=spikeAx,
                            axesLabel=True,
                            legendTags=[
                                'tag'])
                        try:
                            dummyUnit = unitsOnThisChan[0]
                            chan_grp = (
                                spikes
                                ['basic_headers']['chan_grp'][dummyUnit]
                            )
                            spikeAx.set_title(
                                'chan_grp {}: {}'.format(
                                    chan_grp,
                                    spikeAx.get_title()
                                ))
                        except Exception:
                            traceback.print_exc()
                        plot_spikes(
                            spikes, channel=channel, ax=spikesBadAx,
                            axesLabel=True, maxSpikes=1000,
                            lineWidth=2, lineAlpha=0.025, lineRasterized=False)
                        spikesBadAx.set_ylim(spikeAx.get_ylim())
                    except Exception:
                        traceback.print_exc()
                    try:
                        isiAx = fig.add_subplot(gs[0, :2])
                        templateAx = fig.add_subplot(gs[0, 2])
                        lastBin = 150
                        isiBins = np.arange(0, lastBin, 1)
                        distBins = np.arange(0, 5, 0.2)
                        kde_kws = {
                            'clip': (isiBins[0] * 0.9, isiBins[-1] * 1.1),
                            'bw': 'silverman', 'gridsize': 500}
                        ksa.plotISIHistogram(
                            spikes, channel=channel, bins=isiBins,
                            ax=isiAx, kde_kws=kde_kws)
                        isiAx.set_title(
                            '{}: {}'.format(
                                channel,
                                isiAx.get_title()
                            ))
                        noiseHarmonics = [
                            i * 1e3 * (60) ** (-1)
                            for i in range(1, 10)]
                        for noiseHarmonic in noiseHarmonics:
                            isiAx.axvline(
                                noiseHarmonic, color='b',
                                linestyle='--', zorder=0)
                        stimHarmonics = [100]
                        for stimHarmonic in stimHarmonics:
                            isiAx.axvline(
                                stimHarmonic, color='r',
                                linestyle='--', zorder=0)
                        ksa.plotSpikePropertyHistogram(
                            spikes, channel=channel,
                            whichProp='templateDist',
                            bins=distBins,
                            ax=templateAx, kde_kws=kde_kws)
                        templateAx.set_title(
                            '{}: {}'.format(
                                'shape distance',
                                templateAx.get_title()
                            ))
                        pdf.savefig()
                        plt.close()
                    except Exception:
                        traceback.print_exc()
                    if len(spikes['Waveforms'][idx].shape) == 3:
                        plotSpike(
                            spikes, channel=channel, acrossArray=True,
                            xcoords=spikeStruct['xcoords'],
                            ycoords=spikeStruct['ycoords'])
                        pdf.savefig()
                        plt.close()
            #  if idx > 2:
            #      break
            #  for loop over channels
