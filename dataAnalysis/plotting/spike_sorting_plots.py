import dataAnalysis.helperFunctions.helper_functions as hf
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import numpy as np

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def spikePDFReport(
        folderPath, spikes, spikeStruct,
        arrayName=None, arrayInfo=None,
        correctAlignmentSpikes=0,
        plotOpts={'type': 'ticks', 'errorBar': 'sem'},
        trialStats=None, enableFR=False, rasterOpts={'alignTo': None}, newName=None):

    if correctAlignmentSpikes:  # correctAlignmentSpikes units in samples
        spikes = hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

    if newName is None:
        pdfName = os.path.join(folderPath, 'spikePDFReport.pdf')
    else:
        pdfName = os.path.join(folderPath, newName + '.pdf')

    if any((arrayName is None, arrayInfo is None)):
        arrayName, arrayInfo, partialRasterOpts = (
            ksa.trialBinnedSpikesNameRetrieve(newName))
        arrayInfo['nevIDs'] = spikes['ChannelID']

    with PdfPages(pdfName) as pdf:
        ksa.plotSpikePanel(spikeStruct, spikes)
        pdf.savefig()
        plt.close()

        for idx, channel in enumerate(spikes['ChannelID']):
            if os.fstat(0) == os.fstat(1):
                endChar = '\r'
            else:
                endChar = ''
            print(
                "Running spikePDFReport: %d%%" % int(
                    (idx + 1) * 100 / len(spikes['ChannelID'])),
                end=endChar)
            unitsOnThisChan = np.unique(spikes['Classification'][idx])
            if unitsOnThisChan is not None:
                if len(unitsOnThisChan) > 0:
                    fig, ax = plt.subplots(nrows=1, ncols=2)
                    ksa.plotSpike(
                        spikes, channel=channel, ax=ax[0],
                        axesLabel=True)
                    isiBins = np.linspace(0, 80, 40)
                    kde_kws = {
                        'clip': (isiBins[0] * 0.8, isiBins[-1] * 1.2),
                        'bw': 'silverman', 'gridsize': 500}
                    ksa.plotISIHistogram(
                        spikes, channel=channel, bins=isiBins,
                        ax=ax[1], kde_kws=kde_kws)
                    pdf.savefig()
                    plt.close()

                    if len(spikes['Waveforms'][idx].shape) == 3:
                        ksa.plotSpike(
                            spikes, channel=channel, acrossArray=True,
                            xcoords=spikeStruct['xcoords'],
                            ycoords=spikeStruct['ycoords'])
                        pdf.savefig()
                        plt.close()

                    if rasterOpts['alignTo'] is not None and trialStats is not None:
                        spikeMats, categories, selectedIndices = ksa.loadTrialBinnedSpike(folderPath,
                            arrayName, arrayInfo,
                            channel,
                            rasterOpts,
                            trialStats=trialStats, spikes=spikes,
                            correctAlignmentSpikes=0,
                            forceRecalc=False)

                        spikeMats, categories, plotFig, plotAx, uniqueCategories, curLine = hf.plotTrialRaster(
                            trialStats=trialStats, channel=channel,
                            spikeMats=spikeMats, categories=categories,
                            plotOpts=plotOpts)

                        #  plotAx = plotRaster(spikes, trialStats, alignTo = plotRastersAlignedTo, windowSize = (-0.5, 2), channel = channel, separateBy = plotRastersSeparatedBy)
                        if enableFR:
                            plotTrialFR(
                                spikeMats=spikeMats, categories=categories,
                                fig=plotFig, ax=plotAx,
                                uniqueCategories=uniqueCategories, twin=True,
                                plotOpts=plotOpts)
                            #  plotFR(spikes, trialStats, alignTo = plotRastersAlignedTo, windowSize = (-0.5, 2), channel = channel, separateBy = plotRastersSeparatedBy, ax = plotAx, twin = True)
                        pdf.savefig()
                        plt.close()