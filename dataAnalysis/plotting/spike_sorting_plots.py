import dataAnalysis.helperFunctions.helper_functions as hf
import os
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import numpy as np
import pdb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
sns.set(rc={
    'figure.figsize': (12, 8),
    'legend.fontsize': 10,
    'legend.handlelength': 2
    })


def spikePDFReport(
        folderPath, spikes, spikeStruct,
        arrayName=None, arrayInfo=None,
        correctAlignmentSpikes=0,
        plotOpts={'type': 'ticks', 'errorBar': 'sem'},
        trialStats=None, enableFR=False, rasterOpts={'alignTo': None},
        newReportName=None):

    if correctAlignmentSpikes:  # correctAlignmentSpikes units in samples
        spikes = hf.correctSpikeAlignment(spikes, correctAlignmentSpikes)

    if newReportName is None:
        pdfName = os.path.join(folderPath, 'spikePDFReport.pdf')
    else:
        pdfName = os.path.join(folderPath, newReportName + '.pdf')

    if any((arrayName is None, arrayInfo is None)):
        arrayName, arrayInfo, partialRasterOpts = (
            ksa.trialBinnedSpikesNameRetrieve(newReportName))
        arrayInfo['nevIDs'] = spikes['ChannelID']

    with PdfPages(pdfName) as pdf:
        ksa.plotSpikePanel(
            spikeStruct, spikes,
            colorPal=None,
            labelFontSize=1, padOverride=5e-2)
        #  make spikepanel square
        figWidth, _ = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(figWidth, figWidth)
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
                    #  allocate axes
                    fig = plt.figure(tight_layout={'pad': 0.01})
                    gs = gridspec.GridSpec(2, 3)
                    try:
                        spikeAx = fig.add_subplot(gs[1, 0])
                        spikesBadAx = fig.add_subplot(gs[1, 1])
                        #  spikesBadAx.get_shared_y_axes().join(
                        #      spikesBadAx, spikeAx)
                        ksa.plotSpike(
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
                        hf.plot_spikes(
                            spikes, channel=channel, ax=spikesBadAx,
                            axesLabel=True, maxSpikes=500, lineAlpha=0.025)
                        spikesBadAx.set_ylim(spikeAx.get_ylim())
                        #  spikeAx.set_ylim([-12, 5])
                    except Exception:
                        traceback.print_exc()

                    try:
                        isiAx = fig.add_subplot(gs[0, :2])
                        templateAx = fig.add_subplot(gs[0, 2])
                        lastBin = 150
                        isiBins = np.arange(0, lastBin, 3)
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
                        #  pdb.set_trace()
                        ksa.plotSpikePropertyHistogram(
                            spikes, channel=channel, whichProp='templateDist',
                            bins=distBins,
                            ax=templateAx, kde_kws=kde_kws)
                        templateAx.set_title(
                            '{}: {}'.format(
                                'shape distance',
                                templateAx.get_title()
                            ))
                        #  import pdb; pdb.set_trace()
                        pdf.savefig()
                        plt.close()
                    except Exception:
                        traceback.print_exc()

                    if len(spikes['Waveforms'][idx].shape) == 3:
                        ksa.plotSpike(
                            spikes, channel=channel, acrossArray=True,
                            xcoords=spikeStruct['xcoords'],
                            ycoords=spikeStruct['ycoords'])
                        pdf.savefig()
                        plt.close()
            #  if idx > 2:
            #      break
            #  for loop over channels
